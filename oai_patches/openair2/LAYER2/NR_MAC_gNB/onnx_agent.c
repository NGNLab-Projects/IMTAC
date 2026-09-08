#include "onnx_agent.h"
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const OrtApi* g_ort = NULL;
static OrtEnv* g_env = NULL;
static OrtSession* g_session = NULL;
static OrtSessionOptions* g_session_options = NULL;
static OrtMemoryInfo* g_memory_info = NULL;
static OrtAllocator* g_allocator = NULL;

static char* g_input_name = NULL;
static char* g_out_names[4] = {NULL, NULL, NULL, NULL};

int init_onnx_agent(const char *model_path) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        printf("Failed to get OrtApi\n");
        return -1;
    }

    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_agent", &g_env);
    g_ort->CreateSessionOptions(&g_session_options);
    
    // Set graph optimization level
    g_ort->SetIntraOpNumThreads(g_session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(g_session_options, ORT_ENABLE_BASIC);

    // Create session
    OrtStatus* status = g_ort->CreateSession(g_env, model_path, g_session_options, &g_session);
    if (status != NULL) {
        printf("Failed to create session: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        return -1;
    }

    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_memory_info);
    g_ort->GetAllocatorWithDefaultOptions(&g_allocator);

    g_ort->SessionGetInputName(g_session, 0, g_allocator, &g_input_name);
    for (int i = 0; i < 4; i++) {
        g_ort->SessionGetOutputName(g_session, i, g_allocator, &g_out_names[i]);
    }

    printf("ONNX Agent initialized successfully. Input: %s, Outputs: %s, %s, %s, %s\n", g_input_name, g_out_names[0], g_out_names[1], g_out_names[2], g_out_names[3]);
    return 0;
}

void run_onnx_agent(float srs_real[32][4], float srs_imag[32][4], float snr, int previous_mcs, int previous_harq, int previous_alpha, int *out_mcs, int *out_alpha) {
    if (!g_session) return;
    
    // Create state vector
    // 1(snr) + 1(prev harq) + 1(prev mcs) + 1(prev alpha) + 256(srs) = 260
    float input_tensor_values[260];
    int idx = 0;

    // 1. Normalized SNR
    float snr_min = 10.0f;
    float snr_max = 20.0f;
    float snr_clamped = snr;
    if (snr_clamped < snr_min) snr_clamped = snr_min;
    if (snr_clamped > snr_max) snr_clamped = snr_max;
    float normalized_snr = 2.0f * (snr_clamped - snr_min) / (snr_max - snr_min) - 1.0f;
    input_tensor_values[idx++] = normalized_snr;

    // 2. HARQ of previous slot
    input_tensor_values[idx++] = (float)previous_harq;

    // 3. Normalized MCS selected in previous slot
    input_tensor_values[idx++] = (float)previous_mcs / 24.0f;

    // 4. Normalized Alpha
    input_tensor_values[idx++] = (float)previous_alpha / 4.0f;

    // 5. SRS Vector
    // Calculate Frobenius norm of complex channel
    float frob_norm = 0.0f;
    for (int col = 0; col < 32; col++) {
        for (int row = 0; row < 4; row++) {
            frob_norm += srs_real[col][row] * srs_real[col][row] + srs_imag[col][row] * srs_imag[col][row];
        }
    }
    frob_norm = sqrtf(frob_norm);

    // Flatten to match MATLAB's interleaved layout: reshape(H_wb, 1, [])
    for (int col = 0; col < 32; col++) {
        for (int row = 0; row < 4; row++) {
            float r = srs_real[col][row];
            float i = srs_imag[col][row];
            if (frob_norm > 1e-6f) {
                r /= frob_norm;
                i /= frob_norm;
            }
            input_tensor_values[idx++] = r;
            input_tensor_values[idx++] = i;
        }
    }

    // Create input tensor
    int64_t input_node_dims[2] = {1, 260};
    OrtValue* input_tensor = NULL;
    g_ort->CreateTensorWithDataAsOrtValue(g_memory_info, input_tensor_values, 260 * sizeof(float), input_node_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    const char* input_node_names[] = {g_input_name};
    const char* output_node_names[] = {g_out_names[0], g_out_names[1], g_out_names[2], g_out_names[3]};

    OrtValue* output_tensors[4] = {NULL, NULL, NULL, NULL};
    OrtStatus* status = g_ort->Run(g_session, NULL, input_node_names, (const OrtValue* const*)&input_tensor, 1, output_node_names, 4, output_tensors);
    if (status != NULL) {
        printf("Failed to run session: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(input_tensor);
        return -1;
    }

    float* a1_logits;
    float* a2_logits;
    float* a3_logits;
    g_ort->GetTensorMutableData(output_tensors[1], (void**)&a1_logits);
    g_ort->GetTensorMutableData(output_tensors[2], (void**)&a2_logits);
    g_ort->GetTensorMutableData(output_tensors[3], (void**)&a3_logits);

    int argmax_a1 = 0;
    float max_a1 = a1_logits[0];
    for (int i=1; i<5; i++) {
        if (a1_logits[i] > max_a1) {
            max_a1 = a1_logits[i];
            argmax_a1 = i;
        }
    }

    int argmax_a2 = 0;
    float max_a2 = a2_logits[0];
    for (int i=1; i<5; i++) {
        if (a2_logits[i] > max_a2) {
            max_a2 = a2_logits[i];
            argmax_a2 = i;
        }
    }

    int argmax_a3 = 0;
    float max_a3 = a3_logits[0];
    for (int i=1; i<5; i++) {
        if (a3_logits[i] > max_a3) {
            max_a3 = a3_logits[i];
            argmax_a3 = i;
        }
    }

    int mcs = 5 * argmax_a2 + argmax_a3;

    // ensure valid range
    if (mcs < 0) mcs = 0;
    if (mcs > 28) mcs = 28;

    if (out_mcs) *out_mcs = mcs;
    if (out_alpha) *out_alpha = argmax_a1;

    // printf("[ONNX_AGENT] Inferred MCS: %d (A2=%d, A3=%d), Alpha: %d | SNR: %.1f\n", mcs, argmax_a2, argmax_a3, argmax_a1, snr);

    g_ort->ReleaseValue(input_tensor);
    for (int i=0; i<4; i++) {
        if (output_tensors[i]) g_ort->ReleaseValue(output_tensors[i]);
    }
}
