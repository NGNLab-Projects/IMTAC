import socket
import time
import os

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# bind to interface

target = ("10.0.1.2", 9999)
payload = os.urandom(1400)

print("Starting 5 Mbps UDP traffic generator...")
while True:
    for _ in range(50):
        try:
            sock.sendto(payload, target)
        except Exception:
            pass
    time.sleep(0.01)
