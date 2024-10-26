#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <math.h>
#include <assert.h>
#include <float.h>

// Implement this
__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd,
                                    const float*  inp, const float*  weight,
                                    const float* bias, int N, int C) {
    
}

// Launch layernorm_forward_kernel here
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    
}

#endif // __LAYERNORM_KERNEL_CUH__