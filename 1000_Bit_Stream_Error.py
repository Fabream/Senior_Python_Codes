#########################################################################################

import numpy as np


num_bits = 10000
bits = np.random.randint(0, 2, num_bits)                # Takes num_bits and build
print(type(bits))
print(bits)

# Step 2: Define transmission voltage based on power
power = 10**-7                                         # Given power threshold
voltage = np.sqrt(power)                               # Voltage recived at the reciever

# print(voltage)

threshold = voltage / 2                                # Detection threshold at half the voltage

#print(threshold)

# Step 3: Multiply bits by voltage
transmitted_signal = bits * voltage  # 0 remains 0, 1 becomes sqrt(10^-7)

# Step 4: Generate Gaussian noise with proper variance
noise_variance = power / 10  # Adjusted variance based on professor's notes
noise = np.random.normal(0, np.sqrt(noise_variance), num_bits)  # Proper scaling of noise

# Display first 20 noise values
# print("Generated Noise (First 20 samples):", noise[:20])

# Step 5: Add Gaussian noise to transmitted signal
received_signal = transmitted_signal + noise

# Step 6: Detection threshold
decoded_bits = (received_signal >= threshold).astype(int)  # Logical operation for decision

# Step 7: Compute Bit Error Rate (BER)
bit_errors = np.sum(decoded_bits != bits)
ber = bit_errors / num_bits

# # Display results
print("Transmitted Bits:", bits[:20])  # Show first 20 transmitted bits
print("Received Signal:", received_signal[:20])  # Show first 20 received signals
print("Decoded Bits:", decoded_bits[:20])  # Show first 20 decoded bits
print(f"Total Bit Errors: {bit_errors}")
print(f"Bit Error Rate (BER): {ber:.5f}")