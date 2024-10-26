#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>

// Implement this
__global__ void matmul_forward_kernel(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
  
}

// Launch matmul_forward_kernel here
void matmul_forward(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
  
}

#endif // __MATMUL_KERNEL_CUH__