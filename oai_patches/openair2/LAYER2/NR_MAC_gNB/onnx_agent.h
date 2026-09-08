#ifndef ONNX_AGENT_H
#define ONNX_AGENT_H

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the ONNX runtime session. Returns 0 on success.
int init_onnx_agent(const char *model_path);

// Runs the ONNX agent and computes the new MCS and alpha.
void run_onnx_agent(float srs_real[32][4], float srs_imag[32][4], float snr, int previous_mcs, int previous_harq, int previous_alpha, int *out_mcs, int *out_alpha);

#ifdef __cplusplus
}
#endif

#endif // ONNX_AGENT_H
