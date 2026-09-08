import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("/home/ngn-6g/OAI/openairinterface5g1/AgentNetwork.onnx")

def test_snr(snr_val):
    snr_min = -20.0
    snr_max = 35.0
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

print(f"SNR 35 dB (norm {test_snr(35.0)[1]:.2f}) -> MCS {test_snr(35.0)[0]}")
print(f"SNR 20 dB (norm {test_snr(20.0)[1]:.2f}) -> MCS {test_snr(20.0)[0]}")
print(f"SNR 0 dB (norm {test_snr(0.0)[1]:.2f}) -> MCS {test_snr(0.0)[0]}")
print(f"SNR -20 dB (norm {test_snr(-20.0)[1]:.2f}) -> MCS {test_snr(-20.0)[0]}")
