#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "softmax.cuh"

// Implement this

// B = block size
// N = seq length
// NH = number of heads
// d = head dimension
// NH*d = embedding dimension
// each thread handles three corresponding values in Q,K,V triplet

// input is in the form B*N*3*NH*d
// thread is indexed by B*NH*N*d
// output {q,k,v} should be in the form B*NH*N*d
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < B*N*NH*d) {
        int _d = (i % d);
        int _n = (i / d) % N;
        int _nh = (i / (N*d)) % NH;
        int _b = (i / (NH*N*d));

        int inp_idx = _b*(N*3*NH*d) + _n*(3*NH*d) + _nh*d + _d;
        int out_idx = _b*(NH*N*d) + _nh*(N*d) + _n*(d) + _d;

        q[out_idx] = inp[inp_idx];
        k[out_idx] = inp[inp_idx + NH * d];
        v[out_idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

// Input dimensions: (B,NH,N,d)
// Output dimensions: (B,N,NH,d)
__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < B*N*NH*d) {

        int _d = i % d;
        int _nh = (i / d) % NH;
        int _n = (i / (NH*d)) % N;
        int _b = (i / (N*NH*d));

        int inp_idx = _b*(NH*N*d) + _nh*(N*d) + _n*d + _d;
        int out_idx = _b*(N*NH*d) + _n*(NH*d) + _nh*d + _d;

        out[out_idx] = inp[inp_idx];
    }  
}

// Implement this (to be used in attention_forward in between permute and softmax)
__global__ void bmm_kernel(float *A, float *B, float *C,
        int batch_size, int n, int m, int p) {
  
}

// Implement this (to be used in attention_forward between softmax and unpermute)
__global__ void bmm2_kernel(float *A, float *B, float *C,
        int batch_size, int n, int m, int p) {
  
}

// Launch bmm_kernel here
void bmm(float *A, float *B, float *C, int batch_size, int n, int m, int p) {

}

// Launch bmm2_kernel here
void bmm2(float *A, float *B, float *C, int batch_size, int n, int m, int p) {
  
}

// Launch permute_kernel, softmax_forward_kernel, unpermute_kernel along with 
// the use of bmm and bmm2 to complete the attention forward pass here
void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    
}

#endif // __ATTENTION_CUH__