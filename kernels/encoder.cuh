#ifndef ENCODER_FORWARD_KERNEL_CUH
#define ENCODER_FORWARD_KERNEL_CUH

#include <cuda_runtime.h>


// Implement this
__global__ void encoder_forward_kernel(float* out, const int* inp,
        const float* wte, const float* wpe, int B, int T, int C) {
    
}

// Launch encoder_forward_kernel here
void encoder_forward(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    
}


#endif // ENCODER_FORWARD_KERNEL_CUH