#ifndef LAYERNORM_CUH
#define LAYERNORM_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum += inp[b * T * C + t * C + c];
            }
            float m = sum / C;
            mean[b * T + t] = m;

            sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                float diff = inp[b * T * C + t * C + c] - m;
                sum += diff * diff;
            }
            float s = 1.0f / sqrtf(sum / C + 1e-5f);
            rstd[b * T + t] = s;

            for (int c = 0; c < C; ++c) {
                float n = s * (inp[b * T * C + t * C + c] - m);
                out[b * T * C + t * C + c] = n * weight[c] + bias[c];
            }
        }
    }
}

#endif // LAYERNORM_CUH