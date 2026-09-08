#ifndef TORCH_INFER_H
#define TORCH_INFER_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Torch model from the given path
int init_torch_model(const char* model_path);

// Run inference
// input: array of floats representing the state
// input_size: number of elements in input array
// output: array of floats to store the action logits
// output_size: expected number of elements in output array
int run_torch_inference(const float* input, int input_size, float* output, int output_size);

// Process inference end-to-end: state extraction, RL forward pass, distortion math
int process_torch_inference(int* ue_cqis, int num_ues, float P_pa,
                            float* prb_powers, float* ue_cqi_penalties, int* k_active,
                            int inference_step);

#ifdef __cplusplus
}
#endif

#endif // TORCH_INFER_H
