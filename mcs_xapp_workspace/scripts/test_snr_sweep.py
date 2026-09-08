import time
import telnetlib

def send_telnet_command(command, port):
    try:
        tn = telnetlib.Telnet("127.0.0.1", port, timeout=1)
        tn.read_until(b"softmodem> ")
        tn.write((command + "\n").encode('ascii'))
        tn.read_until(b"softmodem> ")
        tn.close()
        return True
    except Exception as e:
        return False

def read_current_snr():
    try:
        with open('/tmp/oai_snr.txt', 'r') as f:
            return float(f.read().strip())
    except:
        return None

print("Starting SNR Sweep...")
for noise in range(-15, 20):
    send_telnet_command(f"channelmod modify 0 noise_power_dB {noise}", 9090)
    send_telnet_command(f"channelmod modify 1 noise_power_dB {noise}", 9090)
    time.sleep(2) # wait for SNR to settle
    snr = read_current_snr()
    print(f"Noise: {noise:3d} dB -> SNR: {snr} dB")
