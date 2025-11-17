#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cuda_runtime.h>
#include <cublas_v2.h>
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "utils/tokenizer.h"
// defines: cudaCheck, cublasCheck, cublas_handle, cublas_compute_type, CEIL_DIV
#include "utils/cuda_utils.cuh"

// kernels 
#include "kernels/attention.cuh"
#include "kernels/encoder.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/matmul.cuh"
#include "kernels/residual.cuh"
#include "kernels/softmax.cuh"

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    int Vp = config.padded_vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, int on_device) {
    // on_device: 0 = CPU, 1 = GPU
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once on the device
    float* params_memory;
    if (on_device) {
        cudaCheck(cudaMalloc((void**)&params_memory, num_parameters * sizeof(float)));
    } else {
        params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    }
    // assign all the tensors their place in the array
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* atty; // (L, B, T, C)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)

    float* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    float* output;
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * C; // atty
    act_sizes[5] = L * B * NH * T * T; // att
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = L * B * T; // ln2_mean
    act_sizes[10] = L * B * T; // ln2_rstd
    act_sizes[11] = L * B * T * 4*C; // fch
    act_sizes[12] = L * B * T * 4*C; // fch_gelu
    act_sizes[13] = L * B * T * C; // fcproj
    act_sizes[14] = L * B * T * C; // residual3
    act_sizes[15] = B * T * C; // lnf
    act_sizes[16] = B * T; // lnf_mean
    act_sizes[17] = B * T; // lnf_rstd
    act_sizes[18] = B * T; // losses
    act_sizes[19] = L * B * T * 3*C; // qkvr
    act_sizes[20] = B * T * max(3*C, max(NH*T, Vp)); // output / scratch
}

// Backward pass is conceptually quite different from forward, because we can discard
// the activations of a layer as soon as we're done with it. This lets us aggressively
// reuse memory, so that we need far fewer tensors for backward state.
#define NUM_BACKWARD_TENSORS 3
typedef struct {
    float* bt4c; // (B, T, 4*C)
    float* preatt; // (B, NH, T, T)
    float* residual3; // (B, T, C)
} GradActTensors;

void fill_in_grad_act_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C; // bt4c
    act_sizes[1] = B * NH * T * T; // preatt
    act_sizes[2] = B * T * C; // residual3
}


float* malloc_and_point(float** targets[], const size_t* act_sizes, int n) {
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
    };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

float* malloc_and_point_backward(GradActTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->bt4c, &acts->preatt, &acts->residual3
    };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    GradActTensors grads_acts;
    size_t num_grad_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) {
        // was bumped from 1 -> 3 to incorporate the padded vocab size
        fprintf(stderr, "Bad version in model file\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_memory_cpu);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}


void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T) {
    // targets are optional and could be NULL
    
    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float* scratch = acts.output;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    model->mean_loss = -1.0f;
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->grads_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    cudaFreeHost(model->cpu_losses);
}

#define GPT2_EOT 50256
#define MAX_INPUT_LENGTH 1024

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float* logits, int n, float coin) {
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    // Fallback: return token with highest logit instead of last token (which is EOT!)
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

int sample_argmax(const float* logits, int n) {
    // Greedy sampling - just pick the most likely token
    int max_i = 0;
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_i = i;
        }
    }
    return max_i;
}

int sample_top_k(const float* logits, int n, int k, float coin) {
    // Top-k sampling: only sample from the k most likely tokens
    // First, find the k-th largest logit value
    float* logits_copy = (float*)malloc(n * sizeof(float));
    memcpy(logits_copy, logits, n * sizeof(float));
    
    // Partial sort to find k-th element
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < n; j++) {
            if (logits_copy[j] > logits_copy[i]) {
                float tmp = logits_copy[i];
                logits_copy[i] = logits_copy[j];
                logits_copy[j] = tmp;
            }
        }
    }
    float threshold = logits_copy[k-1];
    free(logits_copy);
    
    // Now sample only from tokens with logit >= threshold
    double norm = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] >= threshold) {
            norm += expf(logits[i]);
        }
    }
    coin *= norm;
    float cdf = 0.0f;
    int last_valid_token = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] >= threshold) {
            cdf += expf(logits[i]);
            last_valid_token = i;
            if (coin < cdf) {
                return i;
            }
        }
    }
    // Return the last valid token if we somehow didn't sample one
    return last_valid_token;
}

void softmax_forward_cpu_single(float* out, float inv_temperature, const float* inp, int T) {
    float sumval = 0.0f;
    for (int i = 0; i < T; i++) {
        float ev = expf(inv_temperature * inp[i]);
        sumval += ev;
        out[i] = ev;
    }
    float norm = 1.0f / sumval;
    for (int i = 0; i < T; ++i) {
        out[i] *= norm;
    }
}

// ----------------------------------------------------------------------------
// main inference loop
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s \"<prompt text>\"\n", argv[0]);
        fprintf(stderr, "Example: %s \"Once upon a time\"\n", argv[0]);
        return 1;
    }

    int B = 1;
    int T = 8192;

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // build the GPT-2 model
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // setup for generation
    unsigned long long rng_state = (long long)time(NULL);
    float* cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));
    int* gen_tokens = (int*)malloc(B * T * sizeof(int));

    // Read input from command line
    char input_buffer[MAX_INPUT_LENGTH];
    strncpy(input_buffer, argv[1], MAX_INPUT_LENGTH - 1);
    input_buffer[MAX_INPUT_LENGTH - 1] = '\0';

    // Tokenize input using greedy longest-match encoding
    int input_length = 0;
    size_t pos = 0;
    size_t text_len = strlen(input_buffer);
    
    // Remove trailing newline
    if (text_len > 0 && input_buffer[text_len-1] == '\n') {
        input_buffer[text_len-1] = '\0';
        text_len--;
    }
    
    while (pos < text_len && input_length < MAX_INPUT_LENGTH) {
        // Greedy: try to match the longest token starting at pos
        int best_token = -1;
        size_t best_len = 0;
        
        // Try all possible lengths from longest to shortest
        for (size_t len = (text_len - pos) < 20 ? (text_len - pos) : 20; len > 0; len--) {
            char temp[21];
            strncpy(temp, input_buffer + pos, len);
            temp[len] = '\0';
            
            // Try to find this substring in the vocabulary
            for (uint32_t i = 0; i < tokenizer.vocab_size; i++) {
                if (strcmp(temp, tokenizer.token_table[i]) == 0) {
                    best_token = i;
                    best_len = len;
                    break;
                }
            }
            if (best_token != -1) break;
        }
        
        if (best_token != -1) {
            gen_tokens[input_length++] = best_token;
            pos += best_len;
        } else {
            // Skip this character if we can't encode it
            fprintf(stderr, "Warning: Could not encode character at position %zu: '%c'\n", pos, input_buffer[pos]);
            pos++;
        }
    }

    if (input_length == 0) {
        fprintf(stderr, "Error: No tokens were successfully encoded. Check your input.\n");
        exit(1);
    }

    // fill rest with EOT
    for (int i = input_length; i < B * T; i++) {
        gen_tokens[i] = GPT2_EOT;
    }

    float temperature = 0.85f;
    int top_k = 50;
    int use_greedy = 0;
    float repetition_penalty = 1.2f;
    float eot_confidence_threshold = 15.0f;
    
    // Print the input tokens first
    for (int i = 0; i < input_length; i++) {
        const char* token_str = tokenizer_decode(&tokenizer, gen_tokens[i]);
        if (token_str != NULL) {
            printf("%s", token_str);
        }
    }
    fflush(stdout);

    // autoregressive generation
    int t;
    int max_tokens = input_length + 500;
    if (max_tokens > T) max_tokens = T;
    for (t = input_length; t < max_tokens; t++) {
        gpt2_forward(&model, gen_tokens, NULL, B, T);
        
        float* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
        cudaCheck(cudaMemcpy(cpu_logits, logits, model.config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        if (repetition_penalty != 1.0f) {
            int lookback_window = 20;
            for (int j = max(input_length, t - lookback_window); j < t; j++) {
                int token = gen_tokens[j];
                if (token < model.config.vocab_size) {
                    if (cpu_logits[token] > 0) {
                        cpu_logits[token] /= repetition_penalty;
                    } else {
                        cpu_logits[token] *= repetition_penalty;
                    }
                }
            }
        }
        
        if (temperature != 1.0f) {
            float inv_temp = 1.0f / temperature;
            for (int i = 0; i < model.config.vocab_size; i++) {
                cpu_logits[i] *= inv_temp;
            }
        }
        
        int next_token;
        int min_tokens_before_eot = 750;
        int max_resample_attempts = 10;
        
        bool has_sentence_end = false;
        if (t > input_length + 10) {
            for (int j = input_length; j < t; j++) {
                const char* tok_str = tokenizer_decode(&tokenizer, gen_tokens[j]);
                if (tok_str != NULL && (strcmp(tok_str, ".") == 0 || strcmp(tok_str, "!") == 0 || strcmp(tok_str, "?") == 0)) {
                    has_sentence_end = true;
                    break;
                }
            }
        }
        
        for (int attempt = 0; attempt < max_resample_attempts; attempt++) {
            if (use_greedy) {
                next_token = sample_argmax(cpu_logits, model.config.vocab_size);
            } else if (top_k > 0 && top_k < model.config.vocab_size) {
                float coin = random_f32(&rng_state);
                next_token = sample_top_k(cpu_logits, model.config.vocab_size, top_k, coin);
            } else {
                float coin = random_f32(&rng_state);
                next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
            }
            
            if (next_token == GPT2_EOT) {
                if (t - input_length < min_tokens_before_eot) {
                    continue;
                }
                
                float max_other_logit = -FLT_MAX;
                for (int i = 0; i < model.config.vocab_size; i++) {
                    if (i != GPT2_EOT && cpu_logits[i] > max_other_logit) {
                        max_other_logit = cpu_logits[i];
                    }
                }
                
                float eot_advantage = cpu_logits[GPT2_EOT] - max_other_logit;
                if (eot_advantage < eot_confidence_threshold) {
                    continue;
                }
                
                if (!has_sentence_end) {
                    continue;
                }
            }
            break;
        }
        
        if (next_token == GPT2_EOT && t - input_length < min_tokens_before_eot) {
            float best_logit = -FLT_MAX;
            int best_token = 0;
            for (int i = 0; i < model.config.vocab_size; i++) {
                if (i != GPT2_EOT && cpu_logits[i] > best_logit) {
                    best_logit = cpu_logits[i];
                    best_token = i;
                }
            }
            next_token = best_token;
        }
        
        gen_tokens[t] = next_token;
        
        if (next_token == GPT2_EOT) {
            printf("\n");
            break; 
        }
        
        const char* token_str = tokenizer_decode(&tokenizer, next_token);
        if (token_str != NULL) {
            printf("%s", token_str);
            fflush(stdout);
        }
    }
    
    printf("\n");

    // cleanup
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(cpu_logits);
    free(gen_tokens);
    
    return 0;
}