#ifndef __ATTENTION_CUH__
#define __ATTENTION_CUH__

#include <cuda_runtime.h>
#include "softmax.cuh"

// Implement this

#define BMM_TILE_WIDTH 32

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
__global__ void bmm_kernel(float *A, float *B, float *C, int N, int T, int HS) {
    __shared__ float tile_m[BMM_TILE_WIDTH][BMM_TILE_WIDTH];
    __shared__ float tile_n[BMM_TILE_WIDTH][BMM_TILE_WIDTH+1];
  
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; // Which (T,T)@(T,T) are we looking at
  
    int tx = threadIdx.x;
    int ty = threadIdx.y;
  
    float pvalue = 0.0;
  
    int row = by*BMM_TILE_WIDTH+ty;
    int col = bx*BMM_TILE_WIDTH+tx;
  
    for (int q = 0; q < (HS-1)/BMM_TILE_WIDTH + 1; q++) {
  
      // Collaboratively Load Tiles
      if (row < T && q*BMM_TILE_WIDTH+tx < HS) {
        tile_m[ty][tx] = A[(bz*T*HS) + row*HS + q*BMM_TILE_WIDTH + tx];
      } else {
        tile_m[ty][tx] = 0;
      }
  
      if (col < T && q*BMM_TILE_WIDTH+ty < HS) {
        tile_n[ty][tx] = B[(bz*T*HS) + col*HS + q*BMM_TILE_WIDTH + ty];
      } else {
        tile_n[ty][tx] = 0;
      }
  
      __syncthreads();
  
      if (row < T && col < T) {
        for (int k = 0; k < BMM_TILE_WIDTH; k++) {
          pvalue += tile_m[ty][k] * tile_n[k][tx];
        }
      }
  
      __syncthreads();
  
    }
  
    if (row < T && col < T)
      C[(bz*T*T) + row*T + col] = pvalue;
}
  
// Implement this (to be used in attention_forward between softmax and unpermute)
// __global__ void bmm2_kernel(float *A, float *B, float *C,
        // int batch_size, int n, int m, int p) {
__global__ void bmm2_kernel(float *A, float *B, float *C, int N, int T, int HS) {
    __shared__ float tile_m[BMM_TILE_WIDTH][BMM_TILE_WIDTH];
    __shared__ float tile_n[BMM_TILE_WIDTH][BMM_TILE_WIDTH+1];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; // Which (T,T)@(T,TH) are we looking at
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float pvalue = 0.0;

    int row = by*BMM_TILE_WIDTH+ty;
    int col = bx*BMM_TILE_WIDTH+tx;

    for (int q = 0; q < (T-1)/BMM_TILE_WIDTH + 1; q++) {

        // Collaboratively Load Tiles

        if (row < T && q*BMM_TILE_WIDTH+tx < T) {
            tile_m[ty][tx] = A[(bz*T*T) + row*T + q*BMM_TILE_WIDTH + tx];
        } else {
            tile_m[ty][tx] = 0;
        }

        if (col < HS && q*BMM_TILE_WIDTH+ty < T) {
            tile_n[ty][tx] = B[(bz*T*HS) + (q*BMM_TILE_WIDTH + ty)*HS + col];
        } else {
            tile_n[ty][tx] = 0;
        }

        __syncthreads();

        if (row < T && col < HS) {
            for (int k = 0; k < BMM_TILE_WIDTH; k++) {
                pvalue += tile_m[ty][k] * tile_n[k][tx];
            }
        }

        __syncthreads();

    }

    if (row < T && col < HS)
        C[(bz*T*HS) + row*HS + col] = pvalue;
}
  

// Launch bmm_kernel here
// void bmm(float *A, float *B, float *C, int batch_size, int n, int m, int p) {
void bmm(float *A, float *B, float *C, int N, int T, int HS) {
    dim3 gridDim(ceil((1.0*T)/BMM_TILE_WIDTH),ceil((1.0*T)/BMM_TILE_WIDTH),N);
    dim3 blockDim(BMM_TILE_WIDTH,BMM_TILE_WIDTH,1);

    bmm_kernel<<<gridDim, blockDim>>>(A,B,C,N,T,HS);
}

// Launch bmm2_kernel here
// void bmm2(float *A, float *B, float *C, int batch_size, int n, int m, int p) {
void bmm2(float *A, float *B, float *C, int N, int T, int HS) {
    dim3 gridDim(ceil((1.0*HS)/BMM_TILE_WIDTH),ceil((1.0*T)/BMM_TILE_WIDTH),N);
    dim3 blockDim(BMM_TILE_WIDTH,BMM_TILE_WIDTH,1);

    bmm2_kernel<<<gridDim,blockDim>>>(A,B,C,N,T,HS);
}

// Launch permute_kernel, softmax_forward_kernel, unpermute_kernel along with 
// the use of bmm and bmm2 to complete the attention forward pass here
void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    
    int HS = C / NH; // head size
    
    // Setup Q, K, V pointers in qkvr buffer
    float *q = qkvr + 0 * B * T * C;
    float *k = qkvr + 1 * B * T * C;
    float *v = qkvr + 2 * B * T * C;
    
    // Launch permute kernel to separate Q, K, V
    // inp: (B, T, 3, NH, HS) -> q,k,v: (B, NH, T, HS)
    int total_threads = B * T * NH * HS;
    int block_size = 256;
    int num_blocks = (total_threads + block_size - 1) / block_size;
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    
    // Compute Q @ K^T -> preatt (stored in inp as scratch buffer)
    // q: (B, NH, T, HS), k: (B, NH, T, HS) -> preatt: (B, NH, T, T)
    float* preatt = inp;
    bmm(q, k, preatt, B * NH, T, HS);
    
    // Apply softmax with scaling
    float scale = 1.0f / sqrtf(HS);
    softmax_forward(att, scale, preatt, B * NH, T);
    
    // Compute att @ V -> vaccum (stored in inp as scratch buffer)
    // att: (B, NH, T, T), v: (B, NH, T, HS) -> vaccum: (B, NH, T, HS)
    float* vaccum = inp;
    bmm2(att, v, vaccum, B * NH, T, HS);
    
    // Unpermute from (B, NH, T, HS) to (B, T, NH, HS)
    num_blocks = (total_threads + block_size - 1) / block_size;
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

#endif // __ATTENTION_CUH__