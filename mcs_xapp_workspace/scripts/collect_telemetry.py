#!/usr/bin/env python3
import socket
import struct
import csv
import os
import time
from datetime import datetime

UDP_IP = "127.0.0.1"
UDP_PORT = 9998

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, f"throughput_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

print(f"Logging telemetry to: {LOG_FILE}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((UDP_IP, UDP_PORT))

start_time = None

with open(LOG_FILE, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['time_s', 'tput_mbps', 'num_ues', 'ru0_power', 'ru1_power', 'ru2_power', 'avg_mcs'])
    csv_file.flush()

    try:
        while True:
            data, _ = sock.recvfrom(1024)
            avg_mcs = 0
            if len(data) >= 24:
                tput_mbps, num_ues, ru0, ru1, ru2, avg_mcs = struct.unpack('fIfffI', data[:24])
            elif len(data) >= 20:
                tput_mbps, num_ues, ru0, ru1, ru2 = struct.unpack('fIfff', data[:20])
            elif len(data) >= 8:
                tput_mbps, num_ues = struct.unpack('fI', data[:8])
                ru0, ru1, ru2 = 0.0, 0.0, 0.0
            elif len(data) >= 4:
                tput_mbps = struct.unpack('f', data[:4])[0]
                num_ues, ru0, ru1, ru2 = 0, 0.0, 0.0, 0.0
            
            if 0.0 <= tput_mbps < 100000.0:
                if num_ues < 8:
                    continue
                
                if start_time is None:
                    start_time = time.time()
                    print("\n[TELEMETRY] All 8 UEs connected! Starting data collection and plotting...")

                # Extrapolate to the 3 RUs (as mathematically modeled) to get the true network sum throughput
                tput_mbps = tput_mbps * 3.0
                elapsed = time.time() - start_time
                csv_writer.writerow([
                    f"{elapsed:.2f}",
                    f"{tput_mbps:.4f}",
                    num_ues,
                    f"{ru0:.4f}",
                    f"{ru1:.4f}",
                    f"{ru2:.4f}",
                    avg_mcs
                ])
                csv_file.flush()
                print(f"[t={elapsed:6.1f}s] Tput: {tput_mbps:.2f} Mbps | RU Powers: {ru0:.2f}W, {ru1:.2f}W, {ru2:.2f}W | Avg MCS: {avg_mcs} (UEs: {num_ues})", flush=True)
    except KeyboardInterrupt:
        print("\nStopping telemetry collection...")
