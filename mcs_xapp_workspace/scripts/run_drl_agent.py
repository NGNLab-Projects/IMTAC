#!/usr/bin/env python3
import time

print("Switching gNB MAC Scheduler to DRL Agent Mode (Virtual GSM TAS + ONNX MCS)...")
try:
    with open('/tmp/drl_mode.txt', 'w') as f:
        f.write('1\n')
    print("Mode updated successfully. OAI will now use the DRL agent.")
except Exception as e:
    print(f"Failed to switch mode: {e}")
