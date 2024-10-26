#ifndef ENCODER_CUH
#define ENCODER_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void encoder_forward_cpu(float* out,
                         const int* inp, const float* wte, const float* wpe,
                         int B, int T, int C) {
    assert(C % 4 == 0);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                int ix = inp[b * T + t];
                out[b * T * C + t * C + c] = wte[ix * C + c] + wpe[t * C + c];
            }
        }
    }
}

#endif // ENCODER_CUH