#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "softmax.cuh"

// Implement this
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    
}

// Implement this
__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
   
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