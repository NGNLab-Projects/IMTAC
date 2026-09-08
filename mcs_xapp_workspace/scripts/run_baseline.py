#!/usr/bin/env python3
import time

print("Switching gNB MAC Scheduler to Standard OAI Mode (CQI-MCS Table, No TAS)...")
try:
    with open('/tmp/drl_mode.txt', 'w') as f:
        f.write('0\n')
    print("Mode updated successfully. OAI will now use standard CQI-based scheduling.")
except Exception as e:
    print(f"Failed to switch mode: {e}")
