#ifndef RESIDUAL_CUH
#define RESIDUAL_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = inp1[i] + inp2[i];
    }
}

#endif // RESIDUAL_CUH