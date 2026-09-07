import socket
import struct
import time
import numpy as np
import scipy.linalg as la

UDP_IP = "127.0.0.1"
UDP_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 3GPP TR 38.901 CDL-C simplified Power Delay Profile
# Normalized delays and powers (approximate for CDL-C)
delays = np.array([0.0, 0.204, 0.224, 0.395, 0.588, 0.73, 1.026, 1.044, 1.766, 2.001, 2.332, 3.208, 4.095, 4.417])
powers_dB = np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -15.1, -20.2])
powers = 10**(powers_dB / 10.0)
powers = powers / np.sum(powers) # Normalize

# Simulation parameters
num_prbs = 32
num_rx_antennas = 4
subcarrier_spacing = 30e3 # 30 kHz
prb_bandwidth = 12 * subcarrier_spacing
freqs = np.arange(num_prbs) * prb_bandwidth
delay_spread = 300e-9 # 300 ns

# Spatial correlation matrix (simplified Kronecker model for Rx)
# Assuming a uniform linear array (ULA) with lambda/2 spacing
theta = np.pi / 6 # mean angle of arrival
angular_spread = np.pi / 18 # 10 degrees
d_spacing = 0.5

R_rx = np.zeros((num_rx_antennas, num_rx_antennas), dtype=complex)
for i in range(num_rx_antennas):
    for j in range(num_rx_antennas):
        distance = i - j
        # Simplistic spatial correlation function
        R_rx[i, j] = np.exp(1j * 2 * np.pi * d_spacing * distance * np.sin(theta)) * np.exp(-0.5 * (2 * np.pi * d_spacing * distance * np.cos(theta) * angular_spread)**2)

# Cholesky decomposition for coloring the noise
L_rx = la.cholesky(R_rx, lower=True)

print(f"Starting CDL-C Channel Generator...")
print(f"Streaming 32x4 spatial matrices to {UDP_IP}:{UDP_PORT} at 100 Hz.")

# Doppler parameters
doppler_freq = 50.0 # 50 Hz
dt = 0.01 # 10ms
time_sec = 0.0

# Initialize path phases
path_phases = np.random.uniform(0, 2*np.pi, len(delays))

try:
    while True:
        # Update path phases based on Doppler
        path_phases += 2 * np.pi * doppler_freq * dt * np.random.randn(len(delays))
        
        # Generate complex Gaussian taps for each path and each antenna (spatially uncorrelated first)
        h_uncorr = np.random.randn(num_rx_antennas, len(delays)) + 1j * np.random.randn(num_rx_antennas, len(delays))
        h_uncorr *= np.sqrt(powers / 2) * np.exp(1j * path_phases)
        
        # Apply spatial correlation
        h_corr = L_rx @ h_uncorr
        
        # Generate Frequency Domain Response (OFDM channel)
        # H(f) = sum_l h_l * exp(-j * 2pi * f * tau_l)
        H_freq = np.zeros((num_prbs, num_rx_antennas), dtype=complex)
        for p in range(num_prbs):
            f = freqs[p]
            steering_vector = np.exp(-1j * 2 * np.pi * f * delays * delay_spread)
            H_freq[p, :] = np.sum(h_corr * steering_vector, axis=1)
            
        # OAI requires srs_real[32][4] and srs_imag[32][4]
        srs_real = np.real(H_freq).astype(np.float32)
        srs_imag = np.imag(H_freq).astype(np.float32)
        
        # Flatten and pack into struct
        # We need 256 floats: 128 real, 128 imag
        real_flat = srs_real.flatten() # 128
        imag_flat = srs_imag.flatten() # 128
        
        payload = struct.pack(f'{128}f', *real_flat) + struct.pack(f'{128}f', *imag_flat)
        sock.sendto(payload, (UDP_IP, UDP_PORT))
        
        time_sec += dt
        time.sleep(dt)

except KeyboardInterrupt:
    print("Stopped CDL-C Generator.")
    sock.close()
