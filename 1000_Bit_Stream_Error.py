#########################################################################################
# Version 0.3
#
# Fabrizzio Arguello
# Date:02/12/2025
#
# Description: Checks N bits error from signal transmission
#
#########################################################################################

#Libraries Used
import numpy as np
import matplotlib.pyplot as plt

# Constants
BIT_AMOUNT = 100000  # Increased bit count for better accuracy
POWER = 10 ** -7  # Signal power (can be changed independently)
NOISE_POWER = 10 ** -8  # Fixed noise power (does not change with SNR)
SNR_RANGE = np.arange(0, 51, 2)  # Increased SNR range
TRIALS = 1000  # Number of trials per SNR

# Lists to store BER results
error_rate = []

# Simulate BER for each SNR
for snr_db in SNR_RANGE:
    total_errors = 0

    for _ in range(TRIALS):
        bits = np.random.randint(0, 2, BIT_AMOUNT)  # Generate random bit sequence

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)

        # Set fixed noise variance
        noise_var = NOISE_POWER  # Noise power stays constant

        # Adjust only signal power to control SNR
        signal_power = POWER * snr_linear
        voltage = np.sqrt(signal_power)  # Signal voltage
        threshold = voltage / 2  # Decision threshold

        # Channel gain (still modeled as √SNR)
        channel = np.sqrt(snr_linear)

        # Generate noise (Gaussian with fixed variance)
        noise = np.random.normal(0, np.sqrt(noise_var), BIT_AMOUNT)

        # Received signal
        r_sig = channel * (bits * voltage) + noise

        # Decode received bits
        decoded_bits = (r_sig >= threshold).astype(int)

        # Count bit errors
        total_errors += np.sum(decoded_bits != bits)

    # Compute BER
    ber = total_errors / (BIT_AMOUNT * TRIALS)
    error_rate.append(ber)

# Plot BER vs. SNR
plt.figure(figsize=(8, 6))
plt.semilogy(SNR_RANGE, error_rate, marker='x', linestyle='-', color='b', label="BER")
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs. SNR with Fixed Noise Power')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
