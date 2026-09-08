import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("/home/ngn-6g/OAI/openairinterface5g1/GSM_D3QN_theo_online.onnx")

def test_snr(snr_val):
    snr_min = 10.0
    snr_max = 20.0
    snr_clamped = max(snr_min, min(snr_max, snr_val))
    norm_snr = 2.0 * (snr_clamped - snr_min) / (snr_max - snr_min) - 1.0
    
    state = np.zeros((1, 260), dtype=np.float32)
    state[0, 0] = norm_snr
    state[0, 1] = 1.0 # harq
    state[0, 2] = 12.0/24.0 # mcs
    state[0, 3] = 0.0 # alpha
    
    outs = sess.run(None, {sess.get_inputs()[0].name: state})
    a2 = outs[2][0]
    a3 = outs[3][0]
    mcs = 5 * np.argmax(a2) + np.argmax(a3)
    return mcs, norm_snr

print("SNR (dB) | Normalized | Selected MCS")
print("-" * 40)
for snr in range(-20, 36, 1):
    mcs, norm_snr = test_snr(float(snr))
    print(f"{snr:8.1f} | {norm_snr:10.2f} | {mcs:12d}")
