#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cuda/barrier>
using namespace std;
using namespace nvcuda;

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  int offset_a_m = 64 * blockIdx.x;
  int offset_b_n = 64 * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;
  //Set up work
  //Two Stage Pipelined Fetching
  __shared__ half block_a[2][16][80];
  __shared__ half block_b[2][16][80];


  //One stage is for read and one is for write so that the memory fetching penalty from RAM to GPU will be ignorable (0 -> read, 1 -> write || 1 -> write, 0 -> write)
  int write_stage = 0;
  int read_stage = 0;

  //Now we manually fetch the first round (k = 0 - k = 15)
  #pragma unroll
  for(int j = 0; j < 16; j++){
    //We let a to fetch in row majority (It is in the same cache line, 64 threads fetch 1 cache lines at a time)
    block_a[write_stage][j][i] = __float2half(d_a[(0 + j) *  dim_m + offset_a_m + i]);
    //block_b[j][i] = __float2half(d_b[(offset_b_n + i) * dim_k + 0 + j]);
  }
  //Each time 64 threads 64 items out. We read in row order to mitigate fetch across cache line
  //Thread read horizontally and write vertically
  #pragma unroll
  for(int item = 0; item < 1024; item += 64){
    int current_item = item + i;
    int read_row = current_item / 16;
    int read_col = current_item % 16;
    block_b[write_stage][read_col][read_row] = __float2half(d_b[(offset_b_n + read_row) * dim_k + 0 + read_col]);
  }
  __syncthreads();

  // //Now stage 0 is ready for read and stage 1 is fetching
  // write_stage ^= 1;

  //Now we declare once for all
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];

  #pragma unroll
  for (int r = 0; r < 2; r++)
    #pragma unroll
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);
  //END OF INITIALIZATION//



  //Main computation
  //k=0-15 is done manually
  for (int k = 16; k < dim_k; k += 16) {
    write_stage = read_stage ^ 1;
    //Now we prefetch next stage (DO NOT SYNC THIS TIME!!)
    #pragma unroll
    for(int j = 0; j < 16; j++){
      //We let a to fetch in row majority (It is in the same cache line, 64 threads fetch 1 cache lines at a time)
      block_a[write_stage][j][i] = __float2half(d_a[(k + j) *  dim_m + offset_a_m + i]);
      //block_b[j][i] = __float2half(d_b[(offset_b_n + i) * dim_k + 0 + j]);
    }
    //Each time 64 threads 64 items out. We read in row order to mitigate fetch across cache line
    //Thread read horizontally and write vertically
    #pragma unroll
    for(int item = 0; item < 1024; item += 64){
      int current_item = item + i;
      int read_row = current_item / 16;
      int read_col = current_item % 16;
      block_b[write_stage][read_col][read_row] = __float2half(d_b[(offset_b_n + read_row) * dim_k + k + read_col]);
    }


    //Now the fun part! (We tricked GPU to keep fetching the next stage while letting the tensor core to compute the current stage)
    //Power of eliminating data dependency
    #pragma unroll
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_id * 2 + r;
      //wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[read_stage][0][row_tile * 16], 80);
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        //wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[read_stage][0][c * 16], 80);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }
    //Safe to overwrite if everyone completed this stage ->
    __syncthreads();

    // //Now this round is over, we flip the read and write stage
    read_stage ^= 1;
    // write_stage ^= 1;
    
  }
  //Last found is fetched but never computed. Now let's compute it

#pragma unroll
  for(int r = 0; r < 2; r++){
    int row_tile = warp_id * 2 + r;
    wmma::load_matrix_sync(a_frag, &block_a[read_stage][0][row_tile * 16], 80);
    #pragma unroll
    for(int c = 0; c < 4; c++){
	wmma::load_matrix_sync(b_frag, &block_b[read_stage][0][c*16], 80);
	wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
    }
  }

  //Write Back
#pragma unroll
  for (int r = 0; r < 2; r++) {
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_id * 2 + r) * 16;
      int c_n = offset_b_n + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int tile = 64;
  dim3 block = dim3(tile);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
