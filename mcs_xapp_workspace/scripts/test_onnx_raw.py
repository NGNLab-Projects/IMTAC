import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession("/home/ngn-6g/OAI/openairinterface5g1/AgentNetwork.onnx")

def test_raw(snr_val):
    state = np.zeros((1, 260), dtype=np.float32)
    state[0, 0] = snr_val
    outs = sess.run(None, {sess.get_inputs()[0].name: state})
    mcs = 5 * np.argmax(outs[2][0]) + np.argmax(outs[3][0])
    return mcs

print(f"RAW SNR 35 dB -> MCS {test_raw(35.0)}")
print(f"RAW SNR 20 dB -> MCS {test_raw(20.0)}")
print(f"RAW SNR 0 dB -> MCS {test_raw(0.0)}")
print(f"RAW SNR -20 dB -> MCS {test_raw(-20.0)}")
