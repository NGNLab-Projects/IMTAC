import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('/home/ngn-6g/OAI/openairinterface5g1/AgentNetwork.onnx')
state = np.zeros((1, 260), dtype=np.float32)

print("Testing recurrence:")
for prev_mcs in range(25):
    state[0, 2] = prev_mcs / 24.0
    
    # -10 dB
    state[0, 0] = 2.0 * (-10 - (-20)) / (35 - (-20)) - 1.0
    outs = sess.run(None, {sess.get_inputs()[0].name: state})
    mcs_neg10 = 5 * np.argmax(outs[2]) + np.argmax(outs[3])
    
    # 20 dB
    state[0, 0] = 2.0 * (20 - (-20)) / (35 - (-20)) - 1.0
    outs = sess.run(None, {sess.get_inputs()[0].name: state})
    mcs_20 = 5 * np.argmax(outs[2]) + np.argmax(outs[3])
    
    # 40 dB
    state[0, 0] = 2.0 * (40 - (-20)) / (35 - (-20)) - 1.0
    outs = sess.run(None, {sess.get_inputs()[0].name: state})
    mcs_40 = 5 * np.argmax(outs[2]) + np.argmax(outs[3])
    
    print(f"Prev MCS {prev_mcs:2d} -> Outputs (-10dB: {mcs_neg10:2d}, 20dB: {mcs_20:2d}, 40dB: {mcs_40:2d})")
