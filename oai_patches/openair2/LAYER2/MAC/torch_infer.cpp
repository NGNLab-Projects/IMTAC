#include "torch_infer.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>

#ifdef USE_TORCH
#include <torch/script.h>
#include <cmath>

static const float CQI_TABLE[15] = {
    -6.9360f, -5.1470f, -3.1800f, -1.2530f, 0.7610f, 2.6990f, 4.6940f,
    6.5250f,  8.5730f,  10.3660f, 12.2890f, 14.1730f, 15.8880f, 17.8140f, 19.8290f
};

static float cqi_to_sinr_linear(int cqi) {
    if (cqi <= 0) return std::pow(10.0f, -10.0f / 10.0f); // very low SINR
    if (cqi > 15) cqi = 15;
    return std::pow(10.0f, CQI_TABLE[cqi - 1] / 10.0f);
}

static int sinr_linear_to_cqi(float sinr_lin) {
    float sinr_db = 10.0f * std::log10(sinr_lin);
    for (int cqi = 1; cqi <= 15; ++cqi) {
        if (sinr_db < CQI_TABLE[cqi - 1]) {
            return cqi - 1;
        }
    }
    return 15;
}

static torch::jit::script::Module module;
static bool module_loaded = false;

extern "C" int init_torch_model(const char* model_path) {
    try {
        module = torch::jit::load(model_path);
        module_loaded = true;
        std::cout << "[TORCH] Successfully loaded model: " << model_path << std::endl;
        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "[TORCH] Error loading model: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" int run_torch_inference(const float* input, int input_size, float* output, int output_size) {
    if (!module_loaded) {
        std::cerr << "[TORCH] Model not loaded!" << std::endl;
        return -1;
    }

    try {
        torch::Tensor input_tensor = torch::from_blob((void*)input, {1, input_size}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        torch::Tensor output_tensor = module.forward(inputs).toTensor();

        if (output_tensor.numel() != output_size) {
            std::cerr << "[TORCH] Output size mismatch. Expected " << output_size
                      << ", got " << output_tensor.numel() << std::endl;
            return -1;
        }

        std::memcpy(output, output_tensor.data_ptr<float>(), output_size * sizeof(float));
        return 0;

    } catch (const c10::Error& e) {
        std::cerr << "[TORCH] Error during inference: " << e.what() << std::endl;
        return -1;
    }
}

// ============================================================
// Internal pmat state: pmat[N][M] = per-(RU, PRB) power
// N=3 RUs, M=8 PRBs per RU — matches environment2.py exactly.
// In single-cell OAI (--noS1), RU 0 is the real cell.
// RU 1 and 2 exist in the model's state space but map to zero
// UE associations (no real inter-cell interference).
// ============================================================
static const int N = 3;
static const int M = 8;
static const int Kmax = 4;

static float pmat[N][M];

extern "C" int process_torch_inference(int* ue_cqis, int num_ues, float P_pa,
                                        float* prb_powers, float* ue_cqi_penalties,
                                        int* k_active, int inference_step) {
    static const int MAX_MOBILES = 64;
    static float xu[MAX_MOBILES];
    static float yu[MAX_MOBILES];
    static float beta[N][MAX_MOBILES];
    static float cqi_topk[N][M][Kmax];
    
    // Episodic resets and randomized UE positioning every 200 steps
    if (inference_step == 0) {
        static bool seeded = false;
        if (!seeded) {
            srand(time(NULL));
            seeded = true;
        }

        const float xb[3] = {200.0f, 600.0f, 400.0f};
        const float yb[3] = {200.0f, 200.0f, 600.0f};
        const float h_bs = 10.0f;
        const float h_ut = 1.5f;
        const float fc = 3.5f;

        for (int u = 0; u < MAX_MOBILES; ++u) {
            // Scatter UEs randomly in an 800m x 800m area covering the RUs
            xu[u] = (float)(rand() % 800);
            yu[u] = (float)(rand() % 800);
            
            for (int n = 0; n < N; ++n) {
                float dx = xb[n] - xu[u];
                float dy = yb[n] - yu[u];
                float d3d = std::sqrt(dx * dx + dy * dy);
                
                // Prevent extreme proximity
                if (d3d < 10.0f) d3d = 10.0f;
                
                float pl_los = 32.4f + 21.0f * std::log10(d3d) + 20.0f * std::log10(fc);
                float pl_nlos = 22.4f + 35.3f * std::log10(d3d) + 21.3f * std::log10(fc);
                
                float prob_los = 1.0f;
                if (d3d > 18.0f) {
                    prob_los = (18.0f / d3d) + std::exp(-d3d / 63.0f) * (1.0f - 18.0f / d3d);
                }
                
                float pl_db = prob_los * pl_los + (1.0f - prob_los) * pl_nlos;
                
                beta[n][u] = 1.0f / std::pow(10.0f, pl_db / 10.0f);
            }
        }

        // Reset pmat and cqi_topk to initial state at the start of each episode
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                pmat[i][j] = 0.001f;
                for (int k = 0; k < Kmax; ++k) {
                    cqi_topk[i][j][k] = -1.0f;
                }
            }
        }
        printf("[TORCH_DEBUG] Episode started. Reset power allocation and CQI states.\n");
    }

    if (num_ues == 0) {
        return 0; // Nothing to do if no UEs
    }

    // ---------------------------------------------------------
    // 1. Per-RU power sums → P_ru_norm[3]
    //    Mirrors: P_ru = np.sum(pmat, axis=1)
    //             P_ru_norm = P_ru / (sum(P_ru) + eps)
    // ---------------------------------------------------------
    float P_ru[N] = {0};
    float sum_P_ru = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) P_ru[i] += pmat[i][j];
        sum_P_ru += P_ru[i];
    }
    float P_ru_norm[N] = {0};
    for (int i = 0; i < N; ++i)
        P_ru_norm[i] = P_ru[i] / sum_P_ru;

    // ---------------------------------------------------------
    // 2. Scalar state scalars
    // ---------------------------------------------------------
    int K_active_val = *k_active;
    if (K_active_val < 1) K_active_val = 1;
    if (K_active_val > Kmax) K_active_val = Kmax;

    float U_norm   = (float)num_ues / 120.0f;
    float Ppa_norm = P_pa / 0.2f;
    float K_norm   = (float)K_active_val / (float)Kmax;

    // Determine active UE associations to the 3 RUs
    std::vector<int> ue_list[N];
    for (int u = 0; u < num_ues; ++u) {
        int best_ru = 0;
        float max_beta = beta[0][u];
        for (int n = 1; n < N; ++n) {
            if (beta[n][u] > max_beta) {
                max_beta = beta[n][u];
                best_ru = n;
            }
        }
        ue_list[best_ru].push_back(u);
    }

    // Sort each list by beta[n][u] descending (as in environment5.py)
    for (int n = 0; n < N; ++n) {
        std::sort(ue_list[n].begin(), ue_list[n].end(), [&](int a, int b) {
            return beta[n][a] > beta[n][b];
        });
    }


    // 1.5 Generate fast fading for this step (exponential dist mean 1.0)
    // REMOVED: Step-by-step fading completely destabilizes the OAI OLLA (MCS adaptation)
    // algorithm, causing chaotic throughput jumps. 
    static float fading[N][MAX_MOBILES];
    static bool fading_init = false;
    if (!fading_init) {
        for (int n = 0; n < N; ++n) {
            for (int u = 0; u < MAX_MOBILES; ++u) {
                fading[n][u] = 1.0f; // No fading, pure pathloss.
            }
        }
        fading_init = true;
    }

    // Assign associated UEs to each RU's PRBs
    for (int n = 0; n < N; ++n) {
        int num_associated = ue_list[n].size();
        for (int i = 0; i < num_associated; ++i) {
            int prb  = i / K_active_val;
            int slot = i % K_active_val;
            if (prb >= M || slot >= Kmax) break; // capacity exhausted

            int u = ue_list[n][i];
            
            // 1) Compute power per user and array gain dynamically based on environment5.py
            int K_scheduled = K_active_val;
            if (prb == num_associated / K_active_val) {
                K_scheduled = num_associated % K_active_val;
            }
            if (K_scheduled < 1) K_scheduled = 1;

            float P_user = pmat[n][prb] / (float)K_scheduled;
            float array_gain = 63.0f;
            
            // Apply fast fading to beta for this specific step
            float beta_eff_n = beta[n][u] * fading[n][u];
            
            // 2) Estimate SINR to RU n mathematically from pathloss and noise
            // We use the mathematically balanced noise floor of 1.0e-14f.
            // At 400m distance, the pathloss beta is 3.5e-13. 
            // We mathematically account for the 76% signal attenuation from lam_n (PA distortion)
            // AND provide margin for the agent's imperfect neural network output to ensure
            // the 200mW regime hits peak throughput.
            float No = 1.0e-14f;
            
            // 3) Apply SINR distortion model for RU n
            float P_ru_n = P_ru[n];
            float P_avg_ant_n = P_ru_n / 64.0f;
            float Psi_n = P_pa / (P_avg_ant_n + 1e-12f);
            float lam_n = std::pow(
                1.0f - std::exp(-Psi_n)
                + 0.5f * std::sqrt(M_PI * Psi_n) * std::erfc(std::sqrt(Psi_n)),
                2.0f);
            float D_n = (2.0f / 3.0f) * std::max(0.0f, 1.0f - std::exp(-Psi_n) - lam_n) * P_ru_n;
            
            // Calculate interference from other RUs
            float I = 0.0f;
            for (int k = 0; k < N; ++k) {
                if (k != n) {
                    I += pmat[k][prb] * beta[k][u] * fading[k][u];
                }
            }

            // Calculate final theoretical SINR (linear)
            float gamm = (array_gain * lam_n * P_user * beta_eff_n) / (I + No + beta_eff_n * D_n);
            int cqi_adjusted = sinr_linear_to_cqi(gamm);

            cqi_topk[n][prb][slot] = (float)cqi_adjusted;
        }
    }

    // ---------------------------------------------------------
    // 4. Build 101-element state vector for unaware agent:
    //    state = concat([P_ru_norm(3), U_norm(1), K_norm(1),
    //                    cqi_scaled.reshape(-1)(96)])
    // ---------------------------------------------------------
    float state[101] = {0};
    state[0] = P_ru_norm[0];
    state[1] = P_ru_norm[1];
    state[2] = P_ru_norm[2];
    state[3] = U_norm;
    state[4] = K_norm;
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < Kmax; ++k)
                state[5 + n * M * Kmax + m * Kmax + k] = cqi_topk[n][m][k] / 15.0f;

    // ---------------------------------------------------------
    // 5. Run PyTorch inference
    // ---------------------------------------------------------
    float output[76] = {0};
    run_torch_inference(state, 101, output, 76);

    // ---------------------------------------------------------
    // 6. Parse actions — mirrors environment2.py step():
    //    First 72 logits = N*M*3 = 24 PRBs × 3 choices {-1,0,+1}
    //    Action is argmax of 3 logits, then mapped: idx-1 ∈ {-1,0,+1}
    //    pstep = 0.03 * P_pa  (exactly as in environment2.py)
    // ---------------------------------------------------------
    float pstep = 0.03f * P_pa;
    
    // Debug log inputs
    printf("[TORCH_DEBUG] Step input: UEs=%d, CQIs=[", num_ues);
    for (int u = 0; u < num_ues; ++u) {
        printf("%d%s", ue_cqis[u], (u == num_ues - 1) ? "" : ", ");
    }
    printf("], K=%d\n", K_active_val);
    
    printf("[TORCH_DEBUG] State vector (first 10): ");
    for (int idx = 0; idx < 10; ++idx) {
        printf("%.3f ", state[idx]);
    }
    printf("\n");

    printf("[TORCH_DEBUG] Actions selected (RU0 PRB 0..7): ");
    for (int i = 0; i < N; ++i) {
        if (i == 0) { // print RU 0 action values
            for (int j = 0; j < M; ++j) {
                int rb = i * M + j;
                int max_idx = 0;
                float max_val = output[rb];
                if (output[24 + rb] > max_val) { max_val = output[24 + rb]; max_idx = 1; }
                if (output[48 + rb] > max_val) {                           max_idx = 2; }
                int action_val = max_idx - 1;
                printf("PRB%d:%d ", j, action_val);
            }
        }
        
        for (int j = 0; j < M; ++j) {
            int rb = i * M + j;
            int max_idx = 0;
            float max_val = output[rb];
            if (output[24 + rb] > max_val) { max_val = output[24 + rb]; max_idx = 1; }
            if (output[48 + rb] > max_val) {                           max_idx = 2; }
            int action_val = max_idx - 1;

            pmat[i][j] += action_val * pstep;
            if (pmat[i][j] < 0.0f) pmat[i][j] = 0.0f;
        }

        float total_power = 0.0f;
        for (int j = 0; j < M; ++j) {
            total_power += pmat[i][j];
        }
        float limit = 64.0f * P_pa;
        if (total_power > limit) {
            for (int j = 0; j < M; ++j) {
                pmat[i][j] *= limit / (total_power + 1e-12f);
            }
        }
        for (int j = 0; j < M; ++j) {
            prb_powers[i * M + j] = pmat[i][j];
        }
    }
    printf("\n");

    // Last 4 logits → K_active ∈ {1,2,3,4}
    int max_k_idx = 0;
    float max_k_val = output[72];
    for (int k = 1; k < 4; ++k) {
        if (output[72 + k] > max_k_val) { max_k_val = output[72 + k]; max_k_idx = k; }
    }
    K_active_val = max_k_idx + 1;

    static int local_episode = 0;
    if (inference_step == 0) local_episode++;
    
    // Agent takes full control immediately (baseline override removed for inference)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            prb_powers[i * M + j] = pmat[i][j];
        }
    }
    
    *k_active = K_active_val;

    // ---------------------------------------------------------
    // 8. OAI Hardware CQI Penalty
    // ---------------------------------------------------------
    // 5. Aggregate back to OAI CQI Penalty (Per-UE Wideband Average SINR)
    //    We average the linear SINRs across the PRBs assigned to each UE.
    // ---------------------------------------------------------
    
    // First, clear tracking arrays for all UEs
    float sum_sinr[MAX_MOBILES] = {0};
    int count_prbs[MAX_MOBILES] = {0};

    // Second, iterate through the assigned PRBs and accumulate the linear SINR
    for (int n = 0; n < N; ++n) {
        int num_associated = ue_list[n].size();
        for (int i = 0; i < num_associated; ++i) {
            int prb  = i / K_active_val;
            int slot = i % K_active_val;
            if (prb >= M || slot >= Kmax) break;

            int u = ue_list[n][i];
            float cqi_val = cqi_topk[n][prb][slot];
            if (cqi_val >= 0.0f) {
                sum_sinr[u] += cqi_to_sinr_linear((int)cqi_val);
                count_prbs[u]++;
            }
        }
    }

    // Third, compute average linear SINR per UE and set penalty
    for (int u = 0; u < num_ues; ++u) {
        if (count_prbs[u] > 0) {
            float avg_sinr = sum_sinr[u] / (float)count_prbs[u];
            float eff_cqi = (float)sinr_linear_to_cqi(avg_sinr);
            ue_cqi_penalties[u] = std::max(0.0f, 15.0f - eff_cqi);
        } else {
            ue_cqi_penalties[u] = 0.0f; // No penalty if not scheduled
        }
    }

    if (inference_step % 20 == 0) {
        printf("[TORCH_DEBUG] Ep %d Step %d: UE 0 Penalty: %.1f (count: %d), P_ru0: %.2f, P_ru1: %.2f\n", 
            local_episode, inference_step, ue_cqi_penalties[0], count_prbs[0], P_ru[0], P_ru[1]);
    }

    return 0;
}

#else

// Dummy implementations if Torch is not found
extern "C" int init_torch_model(const char* model_path) {
    std::cerr << "[TORCH] LibTorch not enabled during build!" << std::endl;
    return -1;
}

extern "C" int run_torch_inference(const float* input, int input_size, float* output, int output_size) {
    std::cerr << "[TORCH] LibTorch not enabled during build!" << std::endl;
    return -1;
}

extern "C" int process_torch_inference(int* ue_cqis, int num_ues, float P_pa,
                                        float* prb_powers, float* cqi_downgrade,
                                        int* k_active) {
    return -1;
}

#endif
