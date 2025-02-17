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
import pandas as pd
import openpyxl as xl

#Constants
BIT_AMOUNT = 100                                         #size of bit stream
POWER = 10**-7                                             #excpeted POWER output from signal
TRIALS = 1000                                              #total number of trials

error_rate = [] #Create empty list
t_sig_list = []
r_sig_list = []


SNR = 70

for val in range(0,SNR,1):

    bits = np.random.randint(0, 2, BIT_AMOUNT)   # A random bit array is made based on bit amount
    print(bits)

    voltage = np.sqrt(POWER)                               # Voltage received from POWER output of signal

    threshold = voltage / 2                                # For simplistic reason threshold will be half the voltage
                                                           # for detection

    t_sig=(bits * voltage)    # transmitted signal with correct amplitude
    print(t_sig)


    channel = np.sqrt(val)
    print(channel)


    #Generate an array of noise values
    noise_var = POWER                       # Variance of noise signal
    noise = np.random.normal(0,noise_var, BIT_AMOUNT)  #array made with mean = 0, variance, array size

    #Received Signal function with signal amplitude and added noise
    r_sig = channel*t_sig + noise
    print(r_sig)

    decoded_bits = []                                       #Create an empty list
    error_cnt = 0                                           #Initialize error count

    #Iterate through the bits and decode than given the threshold set
    for i in range(BIT_AMOUNT):
        if r_sig[i] >= threshold:
            decoded_bits.append(int(1))                     #add 1 to the list if bigger than threshold
        else:
            decoded_bits.append(int(0))                     #add  to the list if lesser than threshold

        if decoded_bits[i] != bits[i]:
            error_cnt += 1                                  #count the error of decoded bits that don't match bit stream

    t_sig_list.append(t_sig)
    r_sig_list.append(r_sig)

    #Error count list
    error_rate.append((error_cnt / BIT_AMOUNT))

#Results:
dt = pd.DataFrame(t_sig_list).T
dt.to_excel('signal.xlsx',index=False, sheet_name='Signal')
dr = pd.DataFrame(r_sig_list).T
dr.to_excel('real.xlsx',index=False, sheet_name='Real')
print(f'\n{error_rate}')
print(f'\nAverage Percentage Error of Signal Transmission: {np.mean(error_rate):.3f}%')
