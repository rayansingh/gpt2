#ifndef ENCODER_FORWARD_KERNEL_CUH
#define ENCODER_FORWARD_KERNEL_CUH

#include <cuda_runtime.h>

#define ENCODER_BLOCK_SIZE 768

// Implement this
__global__ void encoder_forward_kernel(float* out, const int* inp,
        const float* wte, const float* wpe, int B, int T, int C) {
    
    __shared__ int token_embedding_pos;

    int seq_pos = blockIdx.x % T;
    int i = threadIdx.x;

    if (i == 0) token_embedding_pos = inp[blockIdx.x];

    __syncthreads();

    // C == ENCODER_BLOCK_SIZE

    out[blockIdx.x*C + i] = wte[token_embedding_pos*C + i] + wpe[seq_pos*C + i];
}

// Launch encoder_forward_kernel here
void encoder_forward(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    // each block handles one embedding's positional encoding

    dim3 gridDim(B*T);
    dim3 blockDim(ENCODER_BLOCK_SIZE);
    encoder_forward_kernel<<<gridDim,blockDim>>>(out,inp,wte,wpe,B,T,C);
}


#endif // ENCODER_FORWARD_KERNEL_CUH