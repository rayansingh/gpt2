/*
  Helpful CUDA utilities for error checking and other common operations.
*/
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

#endif // CUDA_UTILS_CUH