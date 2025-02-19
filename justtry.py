#########################################################################################
# Version 0.5 - FINAL SNR FIX
#
# Fabrizzio Arguello
# Date: 02/12/2025
#
# Description: Computes bit errors for a signal transmission system using:
# y = h * x + n (bits * attenuation + noise)
# Keeps noise constant while adjusting signal power to see its effect on BER.
#########################################################################################

# Libraries Used
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl as xl

# Constants
BIT_AMOUNT = 10000000  # Increased bit count for better accuracy
SIG_POWER = 10 ** -4  # Signal power (P_s) - Change this to test effects
NOISE_POWER = 10 ** -8  # Fixed noise power (P_n) - Constant noise
SNR_RANGE = np.arange(0, 51, 2)  # SNR range in dB
TRIALS = 1  # Number of trials per SNR

# Lists to store BER results
error_rate = []
signal_power_list = []
noise_power_list = []
attenuation_list = []
attenuated_signal_list = []
threshold_list = []
received_signal_list = []
SNR_linear_list = []

# Simulate BER for each SNR
for snr_db in SNR_RANGE:
    total_errors = 0

    for _ in range(TRIALS):
        # Generate random bit sequence (0 or 1)
        bits = np.random.randint(0, 2, BIT_AMOUNT)

        # Compute SNR directly from power
        snr_linear = SIG_POWER / NOISE_POWER  # Corrected SNR Calculation
        SNR_linear_list.append(snr_linear)

        # Use SNR to determine attenuation
        h = np.sqrt(snr_linear)
        attenuation_list.append(h)

        # Store power values
        signal_power_list.append(SIG_POWER)
        noise_power_list.append(NOISE_POWER)

        # Compute signal voltage
        voltage = np.sqrt(SIG_POWER)

        # Generate Gaussian noise with proper scaling
        noise = np.random.normal(0, np.sqrt(NOISE_POWER), BIT_AMOUNT)

        # Compute received signal: y = h * x + n
        t_sig = bits * voltage  # Transmitted signal
        r_sig = h * t_sig + noise  # Attenuated signal + noise

        # **Fix Thresholding**
        threshold = np.mean(r_sig)  # Adaptive threshold
        threshold_list.append(threshold)

        # Store single values for Excel
        attenuated_signal_list.append(np.mean(t_sig))
        received_signal_list.append(np.mean(r_sig))

        # Decode signal based on threshold
        decoded_bits = (r_sig >= threshold).astype(int)  # Decision rule
        error_cnt = np.sum(decoded_bits != bits)  # Count bit errors

        # Compute BER
        ber = error_cnt / BIT_AMOUNT
        error_rate.append(max(ber, 1e-12))  # Avoid zero values for log-scale plot

        # Debugging output
        print(f"\nFor SNR (dB) = {snr_db}")
        print(f"Signal Power: {SIG_POWER}")
        print(f"Attenuation (h): {h}")
        print(f"Noise Sample: {noise[0]}")
        print(f"Threshold: {threshold}")
        print(f"BER: {ber:.10e}")

# Convert to Pandas DataFrame
data_dict = {
    "SNR (dB)": SNR_RANGE,
    "SNR Linear": SNR_linear_list,
    "BER": error_rate,
    "Signal Power": signal_power_list,
    "Noise Power": noise_power_list,
    "Attenuation": attenuation_list,
    "Attenuated Signal": attenuated_signal_list,
    "Threshold": threshold_list,
    "Received Signal": received_signal_list
}

df = pd.DataFrame(data_dict)

# Save DataFrame to Excel
df.to_excel("BER_vs_SNR_results.xlsx", index=False)
print(f"Data saved successfully")

# Plot BER vs. SNR
plt.figure(figsize=(8, 6))
plt.semilogy(SNR_RANGE, error_rate, marker='x', linestyle='-', color='b', label="BER")
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs. SNR with Fixed Noise Power')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
