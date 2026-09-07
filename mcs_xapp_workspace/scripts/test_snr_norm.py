import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("/home/ngn-6g/OAI/openairinterface5g1/GSM_D3QN_theo_online.onnx")

print("SNR (dB) | Avg Selected MCS (Averaged over 50 random channel realizations)")
print("-" * 75)

for snr in range(-20, 36, 1):
    snr_min = -20.0
    snr_max = 35.0
    snr_clamped = max(snr_min, min(snr_max, float(snr)))
    norm_snr = 2.0 * (snr_clamped - snr_min) / (snr_max - snr_min) - 1.0
    
    mcs_sum = 0
    # Average over 50 different normalized channel realizations
    for _ in range(50):
        state = np.zeros((1, 260), dtype=np.float32)
        state[0, 0] = norm_snr
        state[0, 1] = 1.0 # harq
        state[0, 2] = 12.0/24.0 # mcs
        state[0, 3] = 2.0/4.0 # alpha
        
        # Create a random CSI vector with Frobenius norm = 1 (just like MATLAB)
        rand_csi = np.random.randn(256)
        rand_csi = rand_csi / np.linalg.norm(rand_csi)
        state[0, 4:260] = rand_csi
        
        outs = sess.run(None, {sess.get_inputs()[0].name: state})
        a2 = outs[2][0]
        a3 = outs[3][0]
        mcs_sum += (5 * np.argmax(a2) + np.argmax(a3))
        
    avg_mcs = mcs_sum / 50.0
    print(f"{snr:8.1f} | {avg_mcs:12.1f}")
