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
import pandas as pd
import openpyxl as xl

# Constants
BIT_AMOUNT = 100000000  # Increased bit count for better accuracy
SIG_POWER = 10 ** -7# Signal power (can be changed independently)
NOISE_POWER = 10 ** -7  # Fixed noise power (does not change with SNR)
SNR_RANGE = np.arange(0, 21, 2)  # Increased SNR range

# Lists to store BER results
error_rate = []
signal_power_list = []
noise_power_list = []
attenuation_list = []
attenuated_signal_list = []
threshold_list = []
received_signal_list = []
SNR_linear_list = []

data_dict = {
    "SNR (dB)": SNR_RANGE,
    "SNR Linear": SNR_linear_list,
    "BER": error_rate,
    #"Signal Power": signal_power_list,
    "Noise Power": noise_power_list,
    #"Attenuation": attenuation_list,
    #"Attenuated Signal": attenuated_signal_list,
    "Threshold": threshold_list,
    "Received Signal": received_signal_list
}


for snr_db in SNR_RANGE:

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Set fixed noise variance
    noise_var = NOISE_POWER/snr_linear

    bits = np.random.randint(0, 2, BIT_AMOUNT)  # Generate random bit sequence

    # Adjust only signal power to control SNR
    # signal_power = SIG_POWER * snr_linear
    # print(f"\nFor SNRdb of {snr_db}\nThis is the signal_power: {signal_power}")

    voltage = np.sqrt(SIG_POWER)  # Signal voltage
    threshold = voltage  # Decision threshold
    amplitude = voltage  #amplitude of signal

    # Channel Power
    #channel = np.sqrt(snr_linear)
    #channel = np.sqrt(1)
    #print(f"This is the attenuation {channel}")

    # Generate noise (Gaussian with fixed variance)
    #noise = np.random.normal(0, np.sqrt(noise_var), BIT_AMOUNT)
    #noise = (snr_linear/snr_db)*NOISE_POWER
    noise_real = (1/np.sqrt(2)) * (np.random.normal(0, np.sqrt(noise_var), BIT_AMOUNT))      # Generates (real) AWGN with zero mean and standard deviation sqrt(noise_variance)
    noise_img = (1/np.sqrt(2)) * (np.random.normal(0, np.sqrt(noise_var), BIT_AMOUNT))       # Generates (imaginary) AWGN with zero mean and standard deviation sqrt(noise_variance)

    noise = (noise_real + 1j * noise_img)                               # Combines both real and imaginary noise into one variable
    #print(f"This is the noise {noise[snr_db]}")

    t_sig = np.where(bits==0, amplitude, -amplitude) #original transmitted signal

    #att_sig = channel * t_sig #attenuated signal
    #print(f"This is the attenuated signal {att_sig[snr_db]}")

    # Received signal
    #r_sig = att_sig + noise #attentuated signal plus the noise
    r_sig = t_sig + noise
    real_r_sig = np.real(r_sig)
    print(f"This is the threshold {threshold}")
    print(f"This is the received signal {r_sig[snr_db]}")

    decoded_bits = (real_r_sig < 0).astype(int)                  # Since the symbols are centered around zero, I use 0 as the threshold. If received signal is < 0, it is decoded as '1' (because -amplitude corresponds to bit 1)

    error_cnt = np.sum(decoded_bits != bits)  # Uses np.sum to add up all the error bits by checking if the decoded_bits are not equal to the bits from the start.

    # Compute BER
    ber = error_cnt / BIT_AMOUNT
    print(f"This is the BER {ber:.30e}")
    print(f"The actual SNR linear is {snr_linear}")

    error_rate.append(ber)
    #signal_power_list.append(signal_power)
    noise_power_list.append(noise[snr_db])
    #attenuation_list.append(channel)
    #attenuated_signal_list.append(att_sig[snr_db])
    threshold_list.append(threshold)
    received_signal_list.append(r_sig[snr_db])
    SNR_linear_list.append(snr_linear)


# Convert to Pandas DataFrame
df = pd.DataFrame(data_dict)

# Save DataFrame to Excel
df.to_excel("BER_vs_SNR_results.xlsx", index=False)
print(f"Data saved successfully")

# Plot BER vs. SNR
plt.figure(figsize=(8, 5))
plt.semilogy(SNR_RANGE, error_rate, marker='x', linestyle='-', color='b', label="BER")
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs. SNR with Fixed Noise Power')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
