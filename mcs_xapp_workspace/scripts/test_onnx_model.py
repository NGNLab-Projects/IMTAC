import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("/home/ngn-6g/OAI/openairinterface5g1/AgentNetwork.onnx")

# Try with zeros (no CDL-C)
state_zeros = np.zeros((1, 260), dtype=np.float32)
# set SNR to 25 dB normalized
state_zeros[0, 0] = 0.5

out_zeros = sess.run(None, {sess.get_inputs()[0].name: state_zeros})[0]
print(f"Output with zeros: {out_zeros}")
print(f"Calculated MCS: {int(out_zeros[0][0])}")

# Try with CDL-C fading
# the CDL-C fading matrices are around 0.1 to 1.0
state_fading = np.random.randn(1, 260).astype(np.float32) * 0.1
state_fading[0, 0] = 0.5 # SNR
state_fading[0, 1] = 1.0 # HARQ
state_fading[0, 2] = 12.0/24.0 # MCS
state_fading[0, 3] = 0.0 # dummy

out_fading = sess.run(None, {sess.get_inputs()[0].name: state_fading})[0]
print(f"Output with fading: {out_fading}")
print(f"Calculated MCS: {int(out_fading[0][0])}")

# Try with SNR = -10 dB normalized
state_fading[0, 0] = -1.25
out_fading2 = sess.run(None, {sess.get_inputs()[0].name: state_fading})[0]
print(f"Output with fading (SNR=-10): {out_fading2}")
print(f"Calculated MCS: {int(out_fading2[0][0])}")
