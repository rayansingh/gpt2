#ifndef RESIDUAL_KERNEL_CUH_
#define RESIDUAL_KERNEL_CUH_

#include <cuda_runtime.h>

#define RESIDUAL_BLOCK_SIZE 1024

// Implement this
__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    float2* inp1_vals = reinterpret_cast<float2*>(&inp1[2*i]);
    float2* inp2_vals = reinterpret_cast<float2*>(&inp2[2*i]);
    float2* out_vals = reinterpret_cast<float2*>(&out[2*i]);
    *out_vals = {inp1_vals->x + inp2_vals->x, inp1_vals->y + inp2_vals->y};
}

// Launch residual_forward_kernel here
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    dim3 gridDim(ceil((1.0*N)/(RESIDUAL_BLOCK_SIZE*2)));
    dim3 blockDim(RESIDUAL_BLOCK_SIZE);
    residual_forward_kernel<<<gridDim,blockDim>>>(out,inp1,inp2,N);
}

#endif // RESIDUAL_KERNEL_CUH_