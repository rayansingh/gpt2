#ifndef __SOFTMAX_KERNEL_CUH__
#define __SOFTMAX_KERNEL_CUH__

#include <assert.h>
#include <math.h>
#include <float.h>

#define SOFTMAX_BLOCK_SIZE 256
#define WARPS_PER_BLOCK (SOFTMAX_BLOCK_SIZE / 32)

// Implement this (to be used in attention_forward)
__global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int N, int T) {

    __shared__ float warptemp[WARPS_PER_BLOCK];
    __shared__ float maxval;
    __shared__ float norm;

    int row_offset = blockIdx.x * T;
    int row_idx = blockIdx.x % T;
    int i = threadIdx.x;

    if (i % 32 == 0) warptemp[i / 32] = -FLT_MAX;

    __syncthreads(); 

    // Scan to find masked row max
    for (int start = 0; start <= row_idx; start += SOFTMAX_BLOCK_SIZE) {
        float mymax = start + i <= row_idx ? inp[row_offset + start + i] : -FLT_MAX;

        __syncwarp();

        mymax = max(mymax,__shfl_down_sync(0xFFFFFFFF,mymax,1));
        mymax = max(mymax,__shfl_down_sync(0x55555555,mymax,2));
        mymax = max(mymax,__shfl_down_sync(0x11111111,mymax,4));
        mymax = max(mymax,__shfl_down_sync(0x01010101,mymax,8));
        mymax = max(mymax,__shfl_down_sync(0x00010001,mymax,16));

        if (i % 32 == 0) {
            warptemp[i / 32] = max(mymax,warptemp[i / 32]);
        }

        __syncwarp();
    }

    __syncthreads();

    // Maybe make this into a warp-reduction? for now it's ok, only 8 iterations
    if (i == 0) {
        float _max = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            _max = max(_max,warptemp[w]);
            warptemp[w] = 0.0f;
        }
        maxval = _max;
    }

    __syncthreads();

    // Reduction to find sum of row
    for (int start = 0; start <= row_idx; start += SOFTMAX_BLOCK_SIZE) {
        float mysum = start + i <= row_idx ? expf(inv_temperature * (inp[row_offset + start + i] - maxval)) : 0.0f;

        if (start + i <= row_idx) out[row_offset + start + i] = mysum;

        __syncwarp();

        mysum += __shfl_down_sync(0xFFFFFFFF,mysum,1);
        mysum += __shfl_down_sync(0x55555555,mysum,2);
        mysum += __shfl_down_sync(0x11111111,mysum,4);
        mysum += __shfl_down_sync(0x01010101,mysum,8);
        mysum += __shfl_down_sync(0x00010001,mysum,16);

        if (i % 32 == 0) {
            warptemp[i / 32] += mysum;
        }

        __syncwarp();
    }

    __syncthreads();

    // Maybe make this into a warp-reduction? for now it's ok, only 8 iterations
    if (i == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            sum += warptemp[w];
        }
        norm = 1.0f / sum;
    }

    __syncthreads();

    for (int start = 0; start < T; start += SOFTMAX_BLOCK_SIZE) {
        if (start + i < T) {
            if (start + i <= row_idx) {
                out[row_offset + start + i] *= norm;
            } else {
                out[row_offset + start + i] = 0.0f;
            }
        }
    }
}

// N = num rows to softmax (B*NH*Seqlen)
void softmax_forward(float* out, float inv_temperature, const float* inp, int N, int T) {

    dim3 dimGrid(N*T);
    dim3 dimBlock(SOFTMAX_BLOCK_SIZE);
    softmax_forward_kernel<<<dimGrid,dimBlock>>>(out,inv_temperature,inp,N*T,T);

}

#endif // __SOFTMAX_KERNEL_CUH__