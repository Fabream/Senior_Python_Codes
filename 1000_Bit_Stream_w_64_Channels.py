###########################################################################################
# Fabrizzio Arguello
#
# 64 - Channel Coefficient Signal Transmission
#
# 03.04.2025
###########################################################################################
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as xl
import pandas as pd
import sys

# Constants
# BIT_AMOUNT = 15998976  # For a Large Matrix that is Multiple of 64
BIT_AMOUNT = 4096 # For a 64 by 64 Matrix output
SIG_POWER = 10 ** -7  # Signal power
NOISE_POWER = 10 ** -7  # Fixed noise power
SNR_RANGE = np.arange(0, 41, 2)  # SNR values in dB

#################### Make Sure that Bit amount is divisible by 64 ########################
if BIT_AMOUNT % 64 != 0:
    print("Warning: BIT AMOUNT is not divisible by 64", file=sys.stderr)
    sys.exit(1)
##########################################################################################

# BER results storage
ber_results = np.zeros(len(SNR_RANGE))

# Create H as a ready list or matrix
h = []

# BER Calculation Loop
for i, snr_db in enumerate(SNR_RANGE):
    print(f"Processing SNR: {snr_db} dB")
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_var = NOISE_POWER / snr_linear  # Adjust noise variance

    # Generate random bit sequence and reshape into (num_symbols, 64)
    bits = np.random.randint(0, 2, BIT_AMOUNT)
    bit_matrix = bits.reshape(-1, 64)  # Each row has 64 bits

    voltage = np.sqrt(SIG_POWER)  # Signal voltage

    # Map bits to BPSK symbols
    t_sig = np.where(bit_matrix == 0, voltage, -voltage)

    # Generate complex Rayleigh fading coefficients for each channel
    h_real = (1 / np.sqrt(2)) * np.random.normal(0, 1, t_sig.shape)
    h_imag = (1 / np.sqrt(2)) * np.random.normal(0, 1, t_sig.shape)
    h = h_real + 1j * h_imag  # Complex Rayleigh fading coefficients



    # Generate complex noise (AWGN)
    noise_real = (1 / np.sqrt(2)) * np.random.normal(0, np.sqrt(noise_var), t_sig.shape)
    noise_imag = (1 / np.sqrt(2)) * np.random.normal(0, np.sqrt(noise_var), t_sig.shape)
    noise = noise_real + 1j * noise_imag  # Complex noise

    # Apply channel effects and noise
    r_sig = h * t_sig + noise

    # Equalization and detection
    real_r_sig = np.real(r_sig / h)
    decoded_bits = (real_r_sig < 0).astype(int)

    # Compute BER
    error_cnt = np.sum(decoded_bits.ravel() != bits)
    ber_results[i] = error_cnt / BIT_AMOUNT  # Compute BER

################# Write to Fading Channel H-Coefficients to Excel####################
#***********************ONLY USE IF MATRIX IS TO VISUALIZED*************************#

# # Convert to Pandas DataFrame
# df = pd.DataFrame(h)
#
# # Save DataFrame to Excel
# df.to_excel("H_64_Channel_Coefficients.xlsx", index=False, header=False)
# print(f"Data saved successfully")

#####################################################################################

# Plot BER vs. SNR
plt.figure(figsize=(8, 5))
plt.semilogy(SNR_RANGE, ber_results, 'o-', linewidth=2, label='64-Channel Rayleigh Fading')

plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title(f'BER vs SNR for {BIT_AMOUNT} bits under 64-Channel Rayleigh Fading')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.yscale('log')  # Explicitly set Y-axis to log scale

plt.show()
