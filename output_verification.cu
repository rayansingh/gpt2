#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

// Include CPU kernels
#include "cpu_kernels/attention.cuh"
#include "cpu_kernels/encoder.cuh"
#include "cpu_kernels/gelu.cuh"
#include "cpu_kernels/layernorm.cuh"
#include "cpu_kernels/matmul.cuh"
#include "cpu_kernels/residual.cuh"

// Include GPU kernels
#include "kernels/attention.cuh"
#include "kernels/encoder.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/matmul.cuh"
#include "kernels/residual.cuh"
#include "kernels/softmax.cuh"

// Include CUDA utilities
#include "utils/cuda_utils.cuh"

int main() {

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    
    const int B = 4; // Batch size
    const int T = 128; // Sequence length - needs to be divisible by 4
    const int C = 768; // Input channels
    const int OC = 4 * C; // Output channels

    const int NH = 12; // Number of heads
    const int sqrt_block_size = 16; // Square root of the block size for the matmul kernel

    const float error_threshold = 0.001; // Error threshold for GPU vs CPU output comparisons: 0.1 %
    const float ep = 0.0001; // Epsilon value for percentage calculation to avoid getting infinity

    // used for permute, unpermute, and softmax unit testing
    const int N = B * T * C;
    const int HS = C / NH; // head size
    
    // Generate random input data
    srand(time(NULL));
    
    /*------------------------ encoder_forward test ------------------------*/

    // Allocate memory for input data
    int* enc_inp_cpu = (int*)malloc(B * T * sizeof(int));

    // Allocate memory for wte and wpe
    float* wte_enc = (float*)malloc(C * C * sizeof(float)); 
    float* wpe_enc = (float*)malloc(T * C * sizeof(float)); 

    for (int i = 0; i < B * T; ++i) {
        enc_inp_cpu[i] = (int)rand() / RAND_MAX;
    }
    for (int i = 0; i < C * C; ++i) {
        wte_enc[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < T * C; ++i) {
        wpe_enc[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    int* enc_inp_gpu_d;
    float* wte_enc_d;
    float* wpe_enc_d;
    cudaMalloc(&enc_inp_gpu_d, B * T * sizeof(int));
    cudaMalloc(&wte_enc_d, C * C * sizeof(float));
    cudaMalloc(&wpe_enc_d, T * C * sizeof(float));
    cudaMemcpy(enc_inp_gpu_d, enc_inp_cpu, B * T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(wte_enc_d, wte_enc, C * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wpe_enc_d, wpe_enc, T * C * sizeof(float), cudaMemcpyHostToDevice);


    // Allocate memory for output
    float* enc_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* enc_out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* enc_out_gpu_d;
    cudaMalloc(&enc_out_gpu_d, B * T * C * sizeof(float));

    // Call CPU function
    encoder_forward_cpu(enc_out_cpu, enc_inp_cpu, wte_enc, wpe_enc, B, T, C);

    // Call GPU function
    encoder_forward(enc_out_gpu_d, enc_inp_gpu_d, wte_enc_d, wpe_enc_d, B, T, C);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(enc_out_gpu, enc_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int enc_totElements = B * T * C;
    int enc_error = 0;
    for (int i = 0; i < enc_totElements; i++) {
        if (fabs(enc_out_cpu[i] - enc_out_gpu[i]) / (enc_out_cpu[i] + ep) > error_threshold) {
            enc_error++;
        }
    }

    if (enc_error > 0) {
        printf("+++++++++++++++ encoder_forward test failed with %d differing values +++++++++++++++\n", enc_error);
    } else {
        printf("--------------------encoder_forward test passed!--------------------\n");
    }

    /*------------------------ layernorm_forward test ------------------------*/

    // Allocate memory for input data
    float* lay_inp_cpu = (float*)malloc(B * T * C * sizeof(float));

    // Allocate memory for other params
    float* weight_ln = (float*)malloc(C * sizeof(float)); 
    float* bias_ln = (float*)malloc(C * sizeof(float)); 
    float* rstd_ln = (float*)malloc(B * T * sizeof(float));
    float* mean_ln = (float*)malloc(B * T * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        lay_inp_cpu[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < B * T; ++i) {
        rstd_ln[i] = (float)rand() / RAND_MAX;
        mean_ln[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < C; ++i) {
        weight_ln[i] = (float)rand() / RAND_MAX;
        bias_ln[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* lay_inp_gpu_d; 
    float* weight_ln_d;
    float* bias_ln_d;
    float* rstd_ln_d;
    float* mean_ln_d;
    cudaMalloc(&lay_inp_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&weight_ln_d, C * sizeof(float));
    cudaMalloc(&bias_ln_d, C * sizeof(float));
    cudaMalloc(&rstd_ln_d, B * T * sizeof(float));
    cudaMalloc(&mean_ln_d, B * T * sizeof(float));
    cudaMemcpy(lay_inp_gpu_d, lay_inp_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_ln_d, weight_ln, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_ln_d, bias_ln, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(rstd_ln_d, rstd_ln, B * T * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mean_ln_d, mean_ln, B * T * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for output
    float* lay_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* lay_out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* lay_out_gpu_d;
    cudaMalloc(&lay_out_gpu_d, B * T * C * sizeof(float));

    // Call CPU function
    layernorm_forward_cpu(lay_out_cpu, mean_ln, rstd_ln, lay_inp_cpu, weight_ln, bias_ln, B, T, C);

    // Call GPU function
    layernorm_forward(lay_out_gpu_d, mean_ln_d, rstd_ln_d, lay_inp_gpu_d, weight_ln_d, bias_ln_d, B, T, C);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(lay_out_gpu, lay_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int lay_totElements = B * T * C;
    int lay_error = 0;
    for (int i = 0; i < lay_totElements; i++) {
        if (fabs(lay_out_cpu[i] - lay_out_gpu[i])/(lay_out_cpu[i] + ep) > error_threshold*100) {
            lay_error++;
        }
    }

    if (lay_error > 0) {
        printf("+++++++++++++++ layernorm_forward test failed with %d differing values +++++++++++++++\n", lay_error);
    } else {
        printf("--------------------layernorm_forward test passed!--------------------\n");
    }

    /*------------------------ permute_kernel_test ------------------------*/

    // Allocate memory for input data
    float* perm_inp_cpu = (float*)malloc(B * T * 3 * NH * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * 3 * NH * C; ++i) {
        perm_inp_cpu[i] = (float)rand() / RAND_MAX;
    }

    float *perm_q_cpu = (float*)malloc(B * T * NH * HS * sizeof(float));
    float *perm_k_cpu = (float*)malloc(B * T * NH * HS * sizeof(float));
    float *perm_v_cpu = (float*)malloc(B * T * NH * HS * sizeof(float));

    // Call CPU function
    permute_kernel_cpu(perm_q_cpu, perm_k_cpu, perm_v_cpu, perm_inp_cpu, B, T, NH, HS);

    // Allocate GPU memory
    float* perm_inp_gpu_d;
    float *perm_q_d, *perm_k_d, *perm_v_d;
    cudaMalloc(&perm_inp_gpu_d, B * T * 3 * NH * C * sizeof(float));
    cudaMalloc(&perm_q_d, B * T * NH * HS * sizeof(float));
    cudaMalloc(&perm_k_d, B * T * NH * HS * sizeof(float));
    cudaMalloc(&perm_v_d, B * T * NH * HS * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(perm_inp_gpu_d, perm_inp_cpu, B * T * 3 * NH * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call GPU kernel
    int total_threads = B * NH * T * HS;
    int perm_num_blocks = CEIL_DIV(total_threads, 256);  // Assuming a block size 256
    permute_kernel<<<perm_num_blocks, 256>>>(perm_q_d, perm_k_d, perm_v_d, perm_inp_gpu_d, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    float* perm_q_gpu = (float*)malloc(B * T * NH * HS * sizeof(float));
    float* perm_k_gpu = (float*)malloc(B * T * NH * HS * sizeof(float));
    float* perm_v_gpu = (float*)malloc(B * T * NH * HS * sizeof(float));
    cudaMemcpy(perm_q_gpu, perm_q_d, B * T * NH * HS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(perm_k_gpu, perm_k_d, B * T * NH * HS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(perm_v_gpu, perm_v_d, B * T * NH * HS * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results

    int perm_totElements = B * T * NH * HS;
    int perm_error = 0;
    // Loop through the results to count differences  
    for (int i = 0; i < perm_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(perm_q_cpu[i] - perm_q_gpu[i])/(perm_q_cpu[i] + ep) > error_threshold) {
            ++perm_error;
        }
        if (fabs(perm_k_cpu[i] - perm_k_gpu[i])/(perm_k_cpu[i] + ep) > error_threshold) {
            ++perm_error;
        }
        if (fabs(perm_v_cpu[i] - perm_v_gpu[i])/(perm_v_cpu[i] + ep) > error_threshold) {
            ++perm_error;
        }

    }

    if (perm_error > 0) {
        printf("+++++++++++++++ permute_kernel test failed with %d differing values +++++++++++++++\n", perm_error);
    } else {
        printf("--------------------permute_kernel test passed!--------------------\n");
    }

    /*------------------------ unpermute_kernel_test ------------------------*/

    // Allocate memory for input data
    float* unperm_inp_cpu = (float*)malloc(B * NH * T * HS * sizeof(float));
    float* unperm_out_cpu = (float*)malloc(B * NH * T * HS * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * NH * T * HS; ++i) {
        unperm_inp_cpu[i] = (float)rand() / RAND_MAX;
    }

    // Call CPU function
    unpermute_kernel_cpu(unperm_inp_cpu, unperm_out_cpu, B, T, NH, HS);

    // Allocate GPU memory
    float* unperm_inp_gpu_d;
    float* unperm_out_gpu_d;
    cudaMalloc(&unperm_inp_gpu_d, B * NH * T * HS * sizeof(float));
    cudaMalloc(&unperm_out_gpu_d, B * NH * T * HS * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(unperm_inp_gpu_d, unperm_inp_cpu, B * NH * T * HS * sizeof(float), cudaMemcpyHostToDevice);

    // Call GPU kernel
    int unperm_num_blocks = CEIL_DIV(B * NH * T * HS, 256); // Assuming a block size 256
    unpermute_kernel<<<unperm_num_blocks, 256>>>(unperm_inp_gpu_d, unperm_out_gpu_d, B, T, NH, HS); 
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    float* unperm_out_gpu = (float*)malloc(B * NH * T * HS * sizeof(float));
    cudaMemcpy(unperm_out_gpu, unperm_out_gpu_d, B * NH * T * HS * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    int unperm_totElements = B * NH * T * HS;
    int unperm_error = 0;
    // Loop through the results to count differences  
    for (int i = 0; i < unperm_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(unperm_out_cpu[i] - unperm_out_gpu[i])/(unperm_out_cpu[i] + ep) > error_threshold) {
            ++unperm_error;
        }
    }

    if (unperm_error > 0) {
        printf("+++++++++++++++ unpermute_kernel test failed with %d differing values +++++++++++++++\n", unperm_error);
    } else {
        printf("--------------------unpermute_kernel test passed!--------------------\n");
    }

    /*------------------------ softmax_forward_test ------------------------*/

    // Allocate memory for input data
    float* softmax_inp_cpu = (float*)malloc(B * NH * T * T * sizeof(float));
    float* softmax_out_cpu = (float*)malloc(B * NH * T * T * sizeof(float));
    float* softmax_inp_gpu = (float*)malloc(B * NH * T * T * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * NH * T * T; ++i) {
        softmax_inp_cpu[i] = (float)rand() / RAND_MAX;
        softmax_inp_gpu[i] = softmax_inp_cpu[i];
    }

    // Call CPU function
    float scale = 1.0 / sqrtf(HS);
    softmax_forward_cpu(softmax_out_cpu, scale, softmax_inp_cpu, B * NH, T);

    // Allocate GPU memory
    float* softmax_inp_gpu_d;
    float* softmax_out_gpu_d;
    cudaMalloc(&softmax_inp_gpu_d, B * NH * T * T * sizeof(float));
    cudaMalloc(&softmax_out_gpu_d, B * NH * T * T * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(softmax_inp_gpu_d, softmax_inp_gpu, B * NH * T * T * sizeof(float), cudaMemcpyHostToDevice);

    // Call GPU kernel
    // int softmax_block_size = 256;
    // int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    // softmax_forward_kernel<<<grid_size, softmax_block_size>>>(softmax_out_gpu_d, scale, softmax_inp_gpu_d, B * NH, T);
    softmax_forward(softmax_out_gpu_d, scale, softmax_inp_gpu_d, B*NH, T);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    float* softmax_out_gpu = (float*)malloc(B * NH * T * T * sizeof(float));
    cudaMemcpy(softmax_out_gpu, softmax_out_gpu_d, B * NH * T * T * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU results
    int softmax_totElements = B * NH * T * T;
    int softmax_error = 0;
    // Loop through the results to count differences  
    for (int i = 0; i < softmax_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(softmax_out_cpu[i] - softmax_out_gpu[i])/(softmax_out_cpu[i] + ep) > error_threshold) {
            ++softmax_error;
        }
    }

    if (softmax_error > 0) {
        printf("+++++++++++++++ softmax_forward test failed with %d differing values +++++++++++++++\n", softmax_error);
    } else {
        printf("--------------------softmax_forward test passed!--------------------\n");
    }

    /*------------------------ attention_forward test ------------------------*/

    int inp_size = B * T * max(3 * C, NH * T);
    // Allocate memory for input data
    float* att_inp_cpu = (float*)malloc(inp_size * sizeof(float));
    // create separate copy of input for GPU to avoid input memory being overwritten 
    float* att_inp_gpu = (float*)malloc(inp_size * sizeof(float)); 

    // Allocate memory for other params
    float* qkvr = (float*)malloc(3 * B * T * C * sizeof(float));
    float* att = (float*)malloc(B * NH * NH * T * T * sizeof(float));

    // Generate random input data
    for (int i = 0; i < inp_size; ++i) {
        att_inp_cpu[i] = (float)rand() / RAND_MAX;
        att_inp_gpu[i] = att_inp_cpu[i];
    }

    // copy data to GPU
    float* att_inp_gpu_d;
    float* qkvr_d;
    float* att_d;
    cudaMalloc(&att_inp_gpu_d, inp_size * sizeof(float));
    cudaMalloc(&qkvr_d, 3 * B * T * C * sizeof(float));
    cudaMalloc(&att_d, B * NH * NH * T * T * sizeof(float));
    cudaMemcpy(att_inp_gpu_d, att_inp_gpu, inp_size * sizeof(float), cudaMemcpyHostToDevice);
    
    

    // Allocate memory for output
    float* att_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* att_out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* att_out_gpu_d;
    cudaMalloc(&att_out_gpu_d, B * T * C * sizeof(float));

    // Call CPU function
    attention_forward_cpu(att_out_cpu, qkvr, att, att_inp_cpu, B, T, C, NH);
    
    // Call GPU function
    attention_forward(att_out_gpu_d, qkvr_d, att_d, att_inp_gpu_d, B, T, C, NH);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(att_out_gpu, att_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int att_totElements = B * T * C;
    int att_diff = 0;
    // Loop through the results to count differences  
    for (int i = 0; i < att_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(att_out_cpu[i] - att_out_gpu[i])/(att_out_cpu[i] + ep) > error_threshold) {
            ++att_diff;
        }
    }

    if (att_diff > 0) {
        printf("+++++++++++++++ attention_forward test failed with %d differing values +++++++++++++++\n", att_diff);
    } else {
        printf("--------------------attention_forward test passed!--------------------\n");
    }

    /*------------------------ residual_forward test ------------------------*/

    // Allocate memory for input data
    float* res_inp1_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* res_inp2_cpu = (float*)malloc(B * T * C * sizeof(float));
    
    // Allocate memory for other params
    float* res_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* res_out_gpu = (float*)malloc(B * T * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        res_inp1_cpu[i] = (float)rand() / RAND_MAX;
        res_inp2_cpu[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* res_inp1_gpu_d;
    float* res_inp2_gpu_d;
    float* res_out_gpu_d;
    cudaMalloc(&res_inp1_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&res_inp2_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&res_out_gpu_d, B * T * C * sizeof(float));
    cudaMemcpy(res_inp1_gpu_d, res_inp1_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(res_inp2_gpu_d, res_inp2_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function
    residual_forward_cpu(res_out_cpu, res_inp1_cpu, res_inp2_cpu, B * T * C);

    // Call GPU function
    residual_forward(res_out_gpu_d, res_inp1_gpu_d, res_inp2_gpu_d, B * T * C);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(res_out_gpu, res_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int res_totElements = B * T * C;
    int res_diff = 0;
    // Loop through the results to count differences  
    for (int i = 0; i < res_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(res_out_cpu[i] - res_out_gpu[i])/ (res_out_cpu[i] + ep) > error_threshold) {
            ++res_diff;
        }
    }

    if (res_diff > 0) {
        printf("+++++++++++++++ residual_forward test failed with %d differing values +++++++++++++++\n", res_diff);
    } else {
        printf("--------------------residual_forward test passed!--------------------\n");
    }

    /*------------------------ gelu_forward test ------------------------*/

    // Allocate memory for input data
    float* gelu_inp_cpu = (float*)malloc(B * T * C * sizeof(float));

    // Allocate memory for other params
    float* gelu_out_cpu = (float*)malloc(B * T * C * sizeof(float));
    float* gelu_out_gpu = (float*)malloc(B * T * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        gelu_inp_cpu[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* gelu_inp_gpu_d;
    float* gelu_out_gpu_d;
    cudaMalloc(&gelu_inp_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&gelu_out_gpu_d, B * T * C * sizeof(float));
    cudaMemcpy(gelu_inp_gpu_d, gelu_inp_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function
    gelu_forward_cpu(gelu_out_cpu, gelu_inp_cpu, B * T * C);

    // Call GPU function
    gelu_forward(gelu_out_gpu_d, gelu_inp_gpu_d, B * T * C);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    cudaMemcpy(gelu_out_gpu, gelu_out_gpu_d, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    int gelu_totElements = B * T * C;
    int gelu_diff = 0;
    // Loop through the results to count differences  
    for (int i = 0; i < gelu_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(gelu_out_cpu[i] - gelu_out_gpu[i])/(gelu_out_cpu[i] + ep) > error_threshold) {
            ++gelu_diff;
        }
    }

    if (gelu_diff > 0) {
        printf("+++++++++++++++ gelu_forward test failed with %d differing values +++++++++++++++\n", gelu_diff);
    } else {
        printf("--------------------gelu_forward test passed!--------------------\n");
    }

    /*------------------------ matmul_forward test ------------------------*/

    // Allocate memory for input data
    float* mat_inp_cpu = (float*)malloc(B * T * C * sizeof(float));

    // Allocate memory for output
    float* mat_out_cpu = (float*)malloc(B * T * 4 * C * sizeof(float));

    // Allocate memory for other params
    float* matmul_weight = (float*)malloc(4 * C * C * sizeof(float));
    float* matmul_bias = (float*)malloc(4 * C * sizeof(float));

    // Generate random input data
    for (int i = 0; i < B * T * C; ++i) {
        mat_inp_cpu[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < 4 * C * C; ++i) {
        matmul_weight[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < 4 * C; ++i) {
        matmul_bias[i] = (float)rand() / RAND_MAX;
    }

    // copy data to GPU
    float* mat_inp_gpu_d;
    float* matmul_weight_d;
    float* matmul_bias_d;
    float* mat_out_gpu_d;
    cudaMalloc(&mat_inp_gpu_d, B * T * C * sizeof(float));
    cudaMalloc(&matmul_weight_d, 4 * C * C * sizeof(float));
    cudaMalloc(&matmul_bias_d, 4 * C * sizeof(float));
    cudaMalloc(&mat_out_gpu_d, B * T * 4 * C * sizeof(float));
    cudaMemcpy(mat_inp_gpu_d, mat_inp_cpu, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matmul_weight_d, matmul_weight, 4 * C * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matmul_bias_d, matmul_bias, 4 * C * sizeof(float), cudaMemcpyHostToDevice);

    // Call CPU function
    matmul_forward_cpu(mat_out_cpu, mat_inp_cpu, matmul_weight, matmul_bias, B, T, C, OC);

    // Call GPU function
    matmul_forward(mat_out_gpu_d, mat_inp_gpu_d, matmul_weight_d, matmul_bias_d, B, T, C, OC);
    cudaCheck(cudaGetLastError());
    cudaDeviceSynchronize();

    // Copy output from GPU
    float* mat_out_gpu = (float*)malloc(B * T * 4 * C * sizeof(float));
    cudaMemcpy(mat_out_gpu, mat_out_gpu_d, B * T * 4 * C * sizeof(float), cudaMemcpyDeviceToHost);

    int mat_totElements = B * T * 4 * C;
    int mat_diff = 0;
    // Loop through the results to count differences
    for (int i = 0; i < mat_totElements; ++i) {
        // Check if the result differs by more than the tolerance
        if (fabs(mat_out_cpu[i] - mat_out_gpu[i])/(mat_out_cpu[i] + ep) > error_threshold) {
            ++mat_diff;
        }
    }

    if (mat_diff > 0) {
        printf("+++++++++++++++ matmul_forward test failed with %d differing values +++++++++++++++\n", mat_diff);
    } else {
        printf("--------------------matmul_forward test passed!--------------------\n");
    }

    printf("********************* All tests finished! *********************\n");

    // Free allocated memory

    free(enc_inp_cpu);
    free(enc_out_cpu);
    free(wte_enc);
    free(wpe_enc);
    free(enc_out_gpu);
    cudaFree(enc_inp_gpu_d);
    cudaFree(wte_enc_d);
    cudaFree(wpe_enc_d);
    cudaFree(enc_out_gpu_d);
    
    free(lay_inp_cpu);
    free(lay_out_cpu);
    free(weight_ln);
    free(bias_ln);
    free(rstd_ln);
    free(mean_ln);
    free(lay_out_gpu);
    cudaFree(lay_inp_gpu_d);
    cudaFree(weight_ln_d);
    cudaFree(bias_ln_d);
    cudaFree(rstd_ln_d);
    cudaFree(mean_ln_d);
    cudaFree(lay_out_gpu_d);

    free(perm_inp_cpu);
    free(perm_q_cpu);
    free(perm_k_cpu);
    free(perm_v_cpu);
    free(perm_q_gpu);
    free(perm_k_gpu);
    free(perm_v_gpu);
    cudaFree(perm_inp_gpu_d);
    cudaFree(perm_q_d);
    cudaFree(perm_k_d);
    cudaFree(perm_v_d);

    free(unperm_inp_cpu);
    free(unperm_out_cpu);
    free(unperm_out_gpu);
    cudaFree(unperm_inp_gpu_d);
    cudaFree(unperm_out_gpu_d);

    free(softmax_inp_cpu);
    free(softmax_inp_gpu);
    free(softmax_out_cpu);
    free(softmax_out_gpu);
    cudaFree(softmax_inp_gpu_d);
    cudaFree(softmax_out_gpu_d);

    free(att_inp_cpu);
    free(att_out_cpu);
    free(att_inp_gpu);
    free(att_out_gpu);
    free(qkvr);
    free(att);
    cudaFree(att_inp_gpu_d);
    cudaFree(qkvr_d);
    cudaFree(att_d);
    cudaFree(att_out_gpu_d);

    free(res_inp1_cpu);
    free(res_inp2_cpu);
    free(res_out_cpu);
    free(res_out_gpu);
    cudaFree(res_inp1_gpu_d);
    cudaFree(res_inp2_gpu_d);
    cudaFree(res_out_gpu_d);

    free(gelu_inp_cpu);
    free(gelu_out_cpu);
    free(gelu_out_gpu);
    cudaFree(gelu_inp_gpu_d);
    cudaFree(gelu_out_gpu_d);

    free(mat_inp_cpu);
    free(mat_out_gpu);
    free(matmul_weight);
    free(matmul_bias);
    cudaFree(mat_inp_gpu_d);
    cudaFree(matmul_weight_d);
    cudaFree(matmul_bias_d);
    cudaFree(mat_out_gpu_d);

    return 0;
}

