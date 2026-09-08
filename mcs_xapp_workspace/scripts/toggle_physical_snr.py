import socket
import time

def send_telnet_command(cmd, port=9090):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(('127.0.0.1', port))
        # Ensure command ends with newline
        s.sendall((cmd + "\n").encode('utf-8'))
        s.close()
        return True
    except Exception as e:
        print(f"Failed to send telnet command to {port}: {e}")
        return False

def read_current_snr():
    try:
        with open('/tmp/oai_snr.txt', 'r') as f:
            return float(f.read().strip())
    except:
        return None

print("Starting Closed-Loop Physical SNR Toggler...")
print("Connecting to OAI RF Simulator Telnet on ports 9090 and 9091...")

toggle = False
last_toggle_time = time.time()
target_snr = 20.0
current_noise = -15.0 # Start with some baseline noise

# P Controller constants (Tuned VERY slow to prevent overshoot and link failure)
Kp = 0.1  # Proportional gain
Ki = 0.0  # Integral gain
integral = 0.0

last_ctrl_time = time.time()

while True:
    now = time.time()
    
    # Toggle every 60 seconds
    if now - last_toggle_time > 60.0:
        toggle = not toggle
        last_toggle_time = now
        
        if toggle:
            target_snr = 10.0
            print(f"[{time.strftime('%H:%M:%S')}] Switched to 10 dB (Low SNR) Downlink")
            current_noise = -4.0 # Calibrated noise for ~10 dB DL SNR
        else:
            target_snr = 20.0
            print(f"[{time.strftime('%H:%M:%S')}] Switched to 20 dB (High SNR) Downlink")
            current_noise = -14.0 # Calibrated noise for ~20 dB DL SNR
            
        # ONLY apply noise to the Downlink (UE port 9091) to keep the Uplink ideal!
        send_telnet_command(f"channelmod modify 0 noise_power_dB {current_noise:.2f}", 9091)
        send_telnet_command(f"channelmod modify 1 noise_power_dB {current_noise:.2f}", 9091)
        
        # Write the target Downlink SNR to a file so the C++ agent can read it
        try:
            with open('/tmp/dl_snr.txt', 'w') as f:
                f.write(str(target_snr))
        except:
            pass
        
    time.sleep(0.5)
