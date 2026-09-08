#!/usr/bin/env python3
import socket
import time
import sys

UDP_IP = "10.0.1.2" # Default OAI UE IP address in --noS1 mode
UDP_PORT = 5001
MESSAGE = b"X" * 1400 # 1400 byte UDP payload

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Flooding {UDP_IP}:{UDP_PORT} with ~200 Mbps of UDP traffic...")
print("This will completely saturate the OAI gNB MAC buffer.")
print("Press Ctrl+C to stop.")

try:
    while True:
        # Send a burst of packets
        for _ in range(100):
            sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
        # Tiny sleep to prevent 100% CPU lockup while maintaining ~200 Mbps
        time.sleep(0.005)
except KeyboardInterrupt:
    print("\nStopped traffic generator.")
