#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  //j index
  float idx[N];
  for(int i = 0; i < N; i++){
    idx[i] = i;
  }

  for(int i=0; i<N; i++) {
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 xjvec = _mm512_load_ps(x);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 yjvec = _mm512_load_ps(y);
    __m512 xansvec = _mm512_sub_ps(xivec, xjvec);
    __m512 yansvec = _mm512_sub_ps(yivec, yjvec);

    __m512 xans2vec = _mm512_mul_ps(xansvec, xansvec);
    __m512 yans2vec = _mm512_mul_ps(yansvec, yansvec);

    __m512 rsumvec = _mm512_add_ps(xans2vec, yans2vec);
    __m512 rsqvec = _mm512_rsqrt14_ps(rsumvec);
    //Now we handle the condition that i == j, which means r == 0
    __m512 onevec = _mm512_set1_ps(1);
    __m512 idxvec = _mm512_load_ps(idx);
    __m512 ivec = _mm512_set1_ps(i);
    __mmask16 mask = _mm512_cmp_ps_mask(idxvec, ivec, _CMP_EQ_OQ);
    __m512 rvec1 = _mm512_mask_blend_ps(mask, rsqvec, onevec);
    __m512 rvec = _mm512_mul_ps(rvec1, rvec1);
    rvec = _mm512_mul_ps(rvec, rvec1);
    //float temp[N];
    //_mm512_store_ps(temp, rvec);
    //for(int l = 0; l < N; l++){
    //    printf("%f ", temp[l]);
    //}
    //printf("\n");
    //Final answer
    __m512 mvec = _mm512_load_ps(m);
    __m512 ansxvec = _mm512_mul_ps(xansvec, mvec);
    __m512 ansyvec = _mm512_mul_ps(yansvec, mvec);
    ansxvec = _mm512_mul_ps(ansxvec, rvec);
    ansyvec = _mm512_mul_ps(ansyvec, rvec);
    fx[i] = _mm512_reduce_add_ps(ansxvec);
    fy[i] = _mm512_reduce_add_ps(ansyvec);
    fx[i] *= -1;
    fy[i] *= -1;
    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
