###########################################################################################
# Fabrizzio Arguello
#
# One Channel Coefficient Signal Transmission with Pilot-Based Channel Estimation
#
# 03.04.2025
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################### CONSTANTS ########################################
BIT_AMOUNT = 10000
SIG_POWER = 10 ** -7
VOLTAGE = np.sqrt(SIG_POWER)
SNR_RANGE = np.arange(0, 41, 2)

ber_results = []

################################### BIT LOOP ########################################
for snr_db in SNR_RANGE:
    # Create SNR values
    snr_linear = 10 ** (snr_db / 10)
    noise_var = SIG_POWER / snr_linear

    # Create random bit stream
    rand_bits = np.random.randint(0, 2, BIT_AMOUNT)
    bits = np.where(rand_bits == 0, VOLTAGE, -VOLTAGE)
    pilot_bit = VOLTAGE

    # Create Random Channel
    h_real = (1 / np.sqrt(2)) * np.random.randn(BIT_AMOUNT)
    h_imag = (1 / np.sqrt(2)) * np.random.randn(BIT_AMOUNT)
    h = h_real + 1j * h_imag

    # Create Noise in the Pilot Stream
    noise_real_pilot = (1 / np.sqrt(2)) * np.sqrt(noise_var) * np.random.randn(BIT_AMOUNT)
    noise_imag_pilot = (1 / np.sqrt(2)) * np.sqrt(noise_var) * np.random.randn(BIT_AMOUNT)
    noise_pilot_bit = noise_real_pilot + 1j * noise_imag_pilot

    # Create Noise in the Bit Stream
    noise_real_bits = (1 / np.sqrt(2)) * np.sqrt(noise_var) * np.random.randn(BIT_AMOUNT)
    noise_imag_bits = (1 / np.sqrt(2)) * np.sqrt(noise_var) * np.random.randn(BIT_AMOUNT)
    noise_bits = noise_real_bits + 1j * noise_imag_bits

    # Receive the Output Pair
    y1 = h * pilot_bit + noise_pilot_bit
    y2 = h * bits + noise_bits

    # Receive the Estimated Channel Bit
    h_est = y1 / pilot_bit

    # Extract the Transmitted Bit from the Estimated Channel and Next Received Bit
    x_hat = y2 / h_est
    detected_bits = (np.real(x_hat) < 0).astype(int)

    # Calculate Bit Error Rate
    bit_errors = np.sum(detected_bits != rand_bits)
    ber = bit_errors / BIT_AMOUNT
    ber_results.append(ber)

##################################### WRITE BER RESULTS TO EXCEL #######################################
# df = pd.DataFrame({'SNR (dB)': SNR_RANGE, 'BER': ber_results})
# df.to_excel("BER_Results.xlsx", index=False)
# print("BER results saved to BER_Results.xlsx")

##################################### PLOT #######################################
plt.figure(figsize=(8, 6))
plt.semilogy(SNR_RANGE, ber_results, 'o-', linewidth=2, label='Pilot-based channel est.')
plt.ylim([1e-7, 1])
plt.xlim([0, 40])
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title(f'BER vs SNR ({BIT_AMOUNT} pilot+data pairs)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()
