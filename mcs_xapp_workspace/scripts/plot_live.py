#!/usr/bin/env python3
import socket
import struct
import matplotlib.pyplot as plt
from collections import deque
import time
import csv
import datetime

UDP_IP = "127.0.0.1"
UDP_PORT = 9998

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((UDP_IP, UDP_PORT))

# Data structures for plotting
max_points = 100
times = deque(maxlen=max_points)
tputs = deque(maxlen=max_points)
mcs_vals = deque(maxlen=max_points)
snr_vals = deque(maxlen=max_points)

# Full history for the final overall plot
full_times = []
full_tputs = []
full_mcs = []
full_snr = []

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

line1, = ax1.plot([], [], 'g-', linewidth=2)
ax1.set_title("Live Dynamic SNR (dB)")
ax1.set_ylabel("SNR (dB)")
    # Removed static y-limits to enable auto-scaling
ax1.grid(True)

line2, = ax2.plot([], [], 'r-', linewidth=2)
ax2.set_title("Live Average MCS")
ax2.set_ylabel("MCS Index")
ax2.set_ylim(-1, 29)
ax2.grid(True)

line3, = ax3.plot([], [], 'b-', linewidth=2)
ax3.set_title("Live OAI gNB Downlink Throughput")
ax3.set_ylabel("Throughput (Mbps)")
ax3.set_xlabel("Time (s)")
ax3.grid(True)

fig.tight_layout()

start_time = time.time()
last_print_time = -5.0 # Force immediate print on first packet

print(f"Listening for telemetry on {UDP_IP}:{UDP_PORT}...")
print("Close the plot window or press Ctrl+C to stop.")

# Determine mode for filename
mode_tag = "TABLE"
try:
    with open('/tmp/drl_mode.txt', 'r') as f:
        if "1" in f.read():
            mode_tag = "IMTAS"
except:
    pass

log_filename = f"session_log_{mode_tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Time(s)", "SNR(dB)", "MCS", "Throughput(Mbps)"])
print(f"Logging data to {log_filename}")

try:
    while True:
        data, _ = sock.recvfrom(1024)
        avg_mcs = 0
        snr = 0.0
        
        if len(data) >= 28:
            tput_mbps, num_ues, ru0, ru1, ru2, avg_mcs, snr = struct.unpack('fIfffIf', data[:28])
        elif len(data) >= 24:
            tput_mbps, num_ues, ru0, ru1, ru2, avg_mcs = struct.unpack('fIfffI', data[:24])
        elif len(data) >= 20:
            tput_mbps, num_ues, ru0, ru1, ru2 = struct.unpack('fIfff', data[:20])
        elif len(data) >= 8:
            tput_mbps, num_ues = struct.unpack('fI', data[:8])
            ru0, ru1, ru2 = 0.0, 0.0, 0.0
        elif len(data) >= 4:
            tput_mbps = struct.unpack('f', data[:4])[0]
            num_ues = 1
            ru0, ru1, ru2 = 0.0, 0.0, 0.0
            
        current_time = time.time() - start_time
        
        # Append to live rolling window
        times.append(current_time)
        tputs.append(tput_mbps)
        mcs_vals.append(avg_mcs)
        snr_vals.append(snr)
        
        # Append to full history
        full_times.append(current_time)
        full_tputs.append(tput_mbps)
        full_mcs.append(avg_mcs)
        full_snr.append(snr)
        
        try:
            with open('/tmp/oai_snr.txt', 'w') as f:
                f.write(str(snr))
        except Exception:
            pass
            
        csv_writer.writerow([f"{current_time:.2f}", f"{snr:.2f}", f"{avg_mcs}", f"{tput_mbps:.2f}"])
        log_file.flush()

        
        if current_time - last_print_time >= 5.0:
            print(f"[t={current_time:6.1f}s] SNR: {snr:5.1f} | MCS: {avg_mcs:2d} | Tput: {tput_mbps:6.2f} Mbps", flush=True)
            last_print_time = current_time

        # Update plots
        line1.set_data(times, snr_vals)
        ax1.relim()
        ax1.autoscale_view()
        
        line2.set_data(times, mcs_vals)
        ax2.relim()
        ax2.autoscale_view()

        line3.set_data(times, tputs)
        ax3.relim()
        ax3.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
        
except KeyboardInterrupt:
    print("\nLive plotting stopped. Generating overall session plot...")
    log_file.close()
    plt.ioff()
    plt.close(fig)
    
    fig_final, (ax1_f, ax2_f, ax3_f) = plt.subplots(3, 1, figsize=(10, 10))
    
    ax1_f.plot(full_times, full_snr, 'g-', linewidth=2)
    ax1_f.set_title("Overall Session Dynamic SNR (dB)")
    ax1_f.set_ylabel("SNR (dB)")
    # Auto-scale y-axis
    ax1_f.grid(True)
    
    ax2_f.plot(full_times, full_mcs, 'r-', linewidth=2)
    ax2_f.set_title("Overall Session Average MCS")
    ax2_f.set_ylabel("MCS Index")
    ax2_f.set_ylim(-1, 29)
    ax2_f.grid(True)
    
    ax3_f.plot(full_times, full_tputs, 'b-', linewidth=2)
    ax3_f.set_title("Overall Session Downlink Throughput")
    ax3_f.set_ylabel("Throughput (Mbps)")
    ax3_f.set_xlabel("Time (s)")
    ax3_f.grid(True)
    
    fig_final.tight_layout()
    plt.show()
