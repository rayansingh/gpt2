#ifndef GELU_KERNEL_CUH_
#define GELU_KERNEL_CUH_

#include <cuda_runtime.h>

#define GELU_BLOCK_SIZE 1024
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// Implement this
__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    const float2* inp_ptr = reinterpret_cast<const float2*>(&inp[i*2]);
    float2* out_ptr = reinterpret_cast<float2*>(&out[i*2]);

    const float2 xi = *inp_ptr;
    float2 cube = {0.044715f * xi.x * xi.x * xi.x, 0.044715f * xi.y * xi.y * xi.y};
    *out_ptr = {0.5f * xi.x * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi.x + cube.x))),
                0.5f * xi.y * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi.y + cube.y)))};
}

// Launch gelu_forward_kernel here
void gelu_forward(float* out, const float* inp, int N) {
    dim3 gridDim(ceil((1.0*N)/(GELU_BLOCK_SIZE*2)));
    dim3 blockDim(GELU_BLOCK_SIZE);
    gelu_forward_kernel<<<gridDim,blockDim>>>(out,inp,N);
    
}
#endif // GELU_KERNEL_CUH_