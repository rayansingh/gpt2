#ifndef __LAYERNORM_KERNEL_CUH__
#define __LAYERNORM_KERNEL_CUH__

#include <math.h>
#include <assert.h>
#include <float.h>

#define WARPS_PER_EMB (128 / 32)
#define BLOCK_SIZE (128)

// Implement this
__global__ void layernorm_forward_kernel(float* out, float* mean, float* rstd,
                                    const float* inp, const float*  weight,
                                    const float* bias, int N, int C) {
    
    int i = threadIdx.x;
    int inp_idx = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float warpSums[WARPS_PER_EMB];
    __shared__ float _mean;
    __shared__ float _std;

    // Warp-wise reduction
    float accum = inp[inp_idx];
    __syncwarp();

    accum += __shfl_down_sync(0xFFFFFFFF,accum,16);
    accum += __shfl_down_sync(0x0000FFFF,accum,8);
    accum += __shfl_down_sync(0x000000FF,accum,4);
    accum += __shfl_down_sync(0x0000000F,accum,2);
    accum += __shfl_down_sync(0x00000003,accum,1);

    if (i % 32 == 0) {
        warpSums[i / 32] = accum;
    }
    __syncthreads();

    if (i == 0) {
        accum = (warpSums[0] + warpSums[1] + warpSums[2] + warpSums[3]) / C;
        _mean = accum;
        mean[blockIdx.x] = accum;
    }

    __syncthreads();

    // Calculate rstd
    accum = inp[inp_idx] - _mean;
    accum *= accum;
    __syncwarp();

    accum += __shfl_down_sync(0xFFFFFFFF,accum,16);
    accum += __shfl_down_sync(0x0000FFFF,accum,8);
    accum += __shfl_down_sync(0x000000FF,accum,4);
    accum += __shfl_down_sync(0x0000000F,accum,2);
    accum += __shfl_down_sync(0x00000003,accum,1);

    if (i % 32 == 0) {
        warpSums[i / 32] = accum;
    }
    __syncthreads();

    if (i == 0) {
        accum = 1.0f / sqrtf((warpSums[0] + warpSums[1] + warpSums[2] + warpSums[3]) / BLOCK_SIZE + 1e-5f);
        _std = accum;
        rstd[blockIdx.x] = accum;
    }

    __syncthreads();

    float n = _std * (inp[inp_idx] - _mean);
    out[inp_idx] = n * weight[i] + bias[i];

}

// Launch layernorm_forward_kernel here
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    

    dim3 gridDim(B*T);
    dim3 blockDim(BLOCK_SIZE);

    layernorm_forward_kernel<<<gridDim,blockDim>>>(out,mean,rstd,inp,weight,bias,B*T,C);
}

#endif // __LAYERNORM_KERNEL_CUH__