###########################################################################################
# Fabrizzio Arguello
#
# 9 x 64 - Simulated Transmitter and Receiver Signal Transmission
#
# 03.04.2025
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt

# Constants
BIT_AMOUNT = 1000  # Increased bit count for better accuracy
SIG_POWER = 10 ** -7  # Signal power
NOISE_POWER = 10 ** -7  # Fixed noise power
SNR_RANGE = np.arange(0, 41, 2)  # SNR values in dB

# New: Channel Matrix Size (9 x 64)
NUM_RX = 9    # 9 receivers
NUM_TX = 64   # 64 transmitters

# BER results storage
ber_results = np.zeros(len(SNR_RANGE))

# BER Calculation Loop
for i, snr_db in enumerate(SNR_RANGE):
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_var = NOISE_POWER / snr_linear  # Adjust noise variance

    # Generate random bit sequence for all transmitters
    bits = np.random.randint(0, 2, (BIT_AMOUNT, NUM_TX))

    # Generate Rayleigh fading channel matrix [9 x 64]
    h_real = (1 / np.sqrt(2)) * np.random.normal(0, 1, (NUM_RX, NUM_TX))
    h_img = (1 / np.sqrt(2)) * np.random.normal(0, 1, (NUM_RX, NUM_TX))
    h = h_real + 1j * h_img  # Complex H (9 x 64)

    voltage = np.sqrt(SIG_POWER)  # Signal voltage

    # Generate noise for all receivers (Gaussian noise) [9 x 64]
    noise_real = (1 / np.sqrt(2)) * np.random.normal(0, np.sqrt(noise_var), (BIT_AMOUNT, NUM_RX, NUM_TX))
    noise_img = (1 / np.sqrt(2)) * np.random.normal(0, np.sqrt(noise_var), (BIT_AMOUNT, NUM_RX, NUM_TX))
    noise = noise_real + 1j * noise_img  # Complex AWGN (BIT_AMOUNT x 9 x 64)

    # Transmitted signal [BIT_AMOUNT x NUM_TX]
    t_sig = np.where(bits == 0, voltage, -voltage)

    # Apply channel attenuation (element-wise multiplication)
    att_sig = h[np.newaxis, :, :] * t_sig[:, np.newaxis, :]  # (BIT_AMOUNT x 9 x 64)

    # Received signal
    r_sig = att_sig + noise  # (BIT_AMOUNT x 9 x 64)
    real_r_sig = np.real(r_sig / h[np.newaxis, :, :])  # Element-wise division

    # Decoding bits using threshold
    decoded_bits = (real_r_sig < 0).astype(int)
    error_cnt = np.sum(decoded_bits != bits[:, np.newaxis, :])  # Compare all bits
    ber_results[i] = error_cnt / (BIT_AMOUNT * NUM_RX * NUM_TX)  # Compute BER

# Plot BER vs. SNR
plt.figure(figsize=(8, 5))
plt.semilogy(SNR_RANGE, ber_results, 'o-', linewidth=2, label='Rayleigh Fading')

plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title(f'BER vs SNR for {BIT_AMOUNT} bits under Rayleigh Fading (9x64 Channel)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.yscale('log')  # Log scale for BER

plt.show()
