#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>

#define TILE_SIZE 32

// Simple MatMul
// Transposed weight matrix
// x * w^t + b = w * x^t + b
__global__ void matmul_forward_kernel(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int B, int T, int C, int OC) {

    __shared__ float tile_x[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_w[TILE_SIZE][TILE_SIZE+1]; // added 1 to avoid bank conflicts

    int my_batch = blockIdx.z;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float accum = 0.0f;
    for (int k = 0; k < (C - 1)/TILE_SIZE + 1; k++) {
        // Collaboratively load tiles
        // Load from input tile
        if (out_row < T && k*TILE_SIZE + tx < C) {
            tile_x[ty][tx] = inp[my_batch*(T*C) + out_row*C + k*TILE_SIZE + tx];
        } else {
            tile_x[ty][tx] = 0.0f;
        }

        // Load from weight tile
        if (k*TILE_SIZE + tx < C && blockIdx.x*TILE_SIZE + ty < OC) {
            tile_w[ty][tx] = weight[C*(blockIdx.x*TILE_SIZE + ty) + k*TILE_SIZE + tx];
        } else {
            tile_w[ty][tx] = 0.0f;
        }

        __syncthreads();
        
        // Perform tile matmul
        for (int q = 0; q < TILE_SIZE; q++) {
            accum += tile_x[ty][q] * tile_w[tx][q];
        }

        __syncthreads();
    }

    if (out_row < T && out_col < OC) {
        out[my_batch*(T*OC) + out_row*OC + out_col] = accum + (bias == NULL ? 0.0f : bias[out_col]);
    }
}

// Launch matmul_forward_kernel here
void matmul_forward(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    
    dim3 gridDim(ceil((1.0*OC)/TILE_SIZE),ceil((1.0*T)/TILE_SIZE),B);
    dim3 blockDim(TILE_SIZE,TILE_SIZE,1);
    matmul_forward_kernel<<<gridDim,blockDim>>>(out,inp,weight,bias,B,T,C,OC);
}

#endif // __MATMUL_KERNEL_CUH__