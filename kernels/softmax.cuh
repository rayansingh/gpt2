#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <assert.h>
#include <math.h>
#include <float.h>

// Implement this (to be used in attention_forward)
__global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int N, int T) {

    

}


#endif // __SOFTMAX_KERNEL_CUH__