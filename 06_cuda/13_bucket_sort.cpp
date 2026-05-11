#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int* bucket, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range){
    bucket[i] = 0;
  }
  // grid.sync();
}

//Imagine each thread holding a number waiting to add to the bucket like queuing (max speed up 5x, min speed up 0x)
__global__ void add_to_bucket(int* bucket, int* keys, int N){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N){
    atomicAdd(&bucket[keys[i]], 1);
  }
  // grid.sync();
}

__global__ void write_back(int* bucket, int* keys, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < range){
    //Calculate offset to write on it's own area
    //printf("%d %d\n",i, bucket[i]);
    int start = 0;
    for(int j = 0; j < i; j++){
      start += bucket[j];
    }
    int end = start + bucket[i];
    //Write once found it's own space
    for(int j = start; j < end; j++){
      keys[j] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int* gpu_keys;
  int* gpu_bucket;
  cudaMalloc(&gpu_keys, sizeof(int) * n);
  cudaMalloc(&gpu_bucket, sizeof(int) * range);
  cudaMemcpy(gpu_keys, key.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
  init<<<(range + 1024 - 1)/1024, 1024>>>(gpu_bucket, range);
  cudaDeviceSynchronize();
  add_to_bucket<<<(n + 1024 - 1)/ 1024, 1024>>>(gpu_bucket, gpu_keys, n);
  cudaDeviceSynchronize();
  write_back<<<(range + 1024 -1 ) / 1024, 1024>>>(gpu_bucket, gpu_keys, range);
  cudaMemcpy(&key[0], gpu_keys, sizeof(int) * n, cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(gpu_bucket);
  cudaFree(gpu_keys);
}
