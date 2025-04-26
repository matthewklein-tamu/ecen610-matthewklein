#MATTHEW KLEIN
#ECEN 610
#LAB 6
# LMS Calibration of Pipeline ADC

import numpy as np
import matplotlib.pyplot as plt

# MDAC function
def mdac(Vin, voffset, finite_ota, ota_offset, cap_mismatch, nonlinear_open, finite_bandwidth):
    gain = 4
    thresholds = np.array([-5/8, -3/8, -1/8, 1/8, 3/8, 5/8]) + voffset
    bits = np.zeros_like(Vin, dtype=int)
    Vout = np.zeros_like(Vin)
    for i, v in enumerate(Vin):
        if v < thresholds[0]:
            bits[i] = 0
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v + 0.75))+ota_offset) * finite_bandwidth
        elif v < thresholds[1]:
            bits[i] = 1
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v + 0.50))+ota_offset) * finite_bandwidth
        elif v < thresholds[2]:
            bits[i] = 2
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v + 0.25))+ota_offset) * finite_bandwidth
        elif v < thresholds[3]:
            bits[i] = 3
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v + 0))+ota_offset) * finite_bandwidth
        elif v < thresholds[4]:
            bits[i] = 4
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v - 0.25))+ota_offset) * finite_bandwidth
        elif v < thresholds[5]:
            bits[i] = 5
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v - 0.50))+ota_offset) * finite_bandwidth
        else:
            bits[i] = 6
            Vout[i] = ((gain * finite_ota * cap_mismatch * (nonlinear_open**2) * (v - 0.75))+ota_offset) * finite_bandwidth
    return Vout, bits

# Parameters
Vref = 1.0  # Reference voltage is 1 V
f_input = 200e6  # 200 MHz tone
f_sample = 503e6  # ~500 MHz sampling
A = 1  # Input amplitude
duration = (1/f_sample)*20  # sim duration
voffset = 0/8 # comparator offset
finite_ota = 1
ota_offset = 0
cap_mismatch = 1
nonlinear_open = 1
finite_bandwidth = 1

# Signals
Ts = 1 / f_sample
time = np.arange(0, duration, Ts)

tones = 128
bw = 200e6
f_start = 0  # Starting frequency
f_step = bw / tones
Vin_25bit = np.zeros_like(time)

for n in range(tones):
    f_n = f_start + n * f_step
    Vin_25bit += (1/tones) * np.cos(2*np.pi*f_n*time)

#Vin_25bit = A * np.sin(2 * np.pi * f_input * time)
bits_25bit = []
num_stages = 13  # stages pipeline
for stage in range(num_stages):
    Vout, bits = mdac(Vin_25bit, voffset, finite_ota, ota_offset, cap_mismatch, nonlinear_open, finite_bandwidth)
    bits_25bit.append(bits)
    Vin_25bit = Vout

Dout_25bit = np.zeros_like(Vin_25bit, dtype=int)
for i in range(len(Vin_25bit)):
    raw_bits = [bits_25bit[stage][i] for stage in range(num_stages)]
    corrected = 0
    for stage in range(num_stages):
        b = raw_bits[stage]
        if b == 0:
            b_val = -3
        elif b == 1:
            b_val = -2
        elif b == 2:
            b_val = -1
        elif b == 3:
            b_val = 0
        elif b == 4:
            b_val = 1
        elif b == 5:
            b_val = 2
        else:
            b_val = 3
        corrected += b_val * (4**(num_stages - 1 - stage))
    Dout_25bit[i] = corrected
Dout_25bit = (Dout_25bit / (4**num_stages)) * Vref #- Vref  # Scale to -1 to 1 V

original_bits = bits_25bit

# Calculate SNR
#Vin = A * np.sin(2 * np.pi * f_input * time)
Vin = np.zeros_like(time)

for n in range(tones):
    f_n = f_start + n * f_step
    Vin += (1/tones) * np.cos(2*np.pi*f_n*time)
LSB = 2 * Vref / (2**13) # Delta
Vin_ideal = np.round((Vin + Vref) / LSB) * LSB - Vref

signal_power = np.mean(Vin_ideal**2)
noise_power = np.mean((Vin_ideal - Dout_25bit)**2)
snr = 10 * np.log10(signal_power / noise_power)

# Plotting
plt.plot(time, Dout_25bit, label=f'SNR={snr:.1f} dB')
plt.xlabel('time')
plt.ylabel('Dout')
plt.title('13-bit 6-stage pipeline ADC')
plt.legend(loc='upper right')
plt.show()

# Print SNR values
print(f"Pipeline SNR: {snr:.1f} dB")

########################################################################################################################
# Add LMS calibration

# Parameters
Vref = 1.0  # Reference voltage is 1 V
f_input = 200e6  # 200 MHz tone
f_sample = 503e6  # ~500 MHz sampling
A = 1  # Input amplitude
voffset = 4/8 # comparator offset
finite_ota = 1
ota_offset = 0
cap_mismatch = 1
nonlinear_open = 1
finite_bandwidth = 1

# Signals
Ts = 1 / f_sample
time = np.arange(0, duration, Ts)

#Vin_25bit = A * np.sin(2 * np.pi * f_input * time)

Vin_25bit = np.zeros_like(time)

for n in range(tones):
    f_n = f_start + n * f_step
    Vin_25bit += (1/tones) * np.cos(2*np.pi*f_n*time)

Vin_25bit /= np.max(np.abs(Vin_25bit))  # Normalize to 1V

bits_25bit = []
num_stages = 13  # stages pipeline
for stage in range(num_stages):
    Vout, bits = mdac(Vin_25bit, voffset, finite_ota, ota_offset, cap_mismatch, nonlinear_open, finite_bandwidth)
    bits_25bit.append(bits)
    Vin_25bit = Vout

badDout_25bit = np.zeros_like(Vin_25bit, dtype=int)
for i in range(len(Vin_25bit)):
    raw_bits = [bits_25bit[stage][i] for stage in range(num_stages)]
    corrected = 0
    for stage in range(num_stages):
        b = raw_bits[stage]
        if b == 0:
            b_val = -3
        elif b == 1:
            b_val = -2
        elif b == 2:
            b_val = -1
        elif b == 3:
            b_val = 0
        elif b == 4:
            b_val = 1
        elif b == 5:
            b_val = 2
        else:
            b_val = 3
        corrected += b_val * (4**(num_stages - 1 - stage))
    badDout_25bit[i] = corrected
badDout_25bit = (badDout_25bit / (4**num_stages)) * Vref #- Vref  # Scale to -1 to 1 V

# Calculate SNR
#Vin = A * np.sin(2 * np.pi * f_input * time)
Vin = np.zeros_like(time)

for n in range(tones):
    f_n = f_start + n * f_step
    Vin += (1/tones) * np.cos(2*np.pi*f_n*time)

LSB = 2 * Vref / (2**13) # Delta
Vin_ideal = np.round((Vin + Vref) / LSB) * LSB - Vref

signal_power = np.mean(Vin_ideal**2)
noise_power = np.mean((Vin_ideal - badDout_25bit)**2)
bad_snr = 10 * np.log10(signal_power / noise_power)

# Plotting
plt.plot(time, badDout_25bit, label=f'SNR={bad_snr:.1f} dB')
plt.xlabel('time')
plt.ylabel('Dout')
plt.title('13-bit 6-stage pipeline ADC')
plt.legend(loc='upper right')
plt.show()

# Print SNR values
print(f"Erroneous Pipeline SNR: {bad_snr:.1f} dB")

######## LMS

# LMS parameters
mu = 0.2              # learning rate
w = np.array([1.0, 0.0])  # initial weights [gain, offset]

# Prepare inputs for LMS: x[n] = [adc_output, 1] to estimate gain and offset
x = np.vstack((badDout_25bit, np.ones(len(time)))).T
d = Dout_25bit      # desired output

# Storage for corrected output
corrected_output = np.zeros(len(time))

# Run LMS
for n in range(len(time)):
    y = np.dot(w, x[n])           # predicted (corrected) output
    e = d[n] - y                  # error
    w = w + mu * e * x[n]         # weight update
    corrected_output[n] = y      # save corrected value

#
# NLMS loop
epsilon = 1e-6     # small constant to avoid division by zero
for n in range(len(time)):
    x_n = x[n]
    y = np.dot(w, x_n)
    e = d[n] - y
    norm_factor = np.dot(x_n, x_n) + epsilon
    w += (mu / norm_factor) * e * x_n
    corrected_output[n] = y

signal_power = np.mean(Vin_ideal**2)
noise_power = np.mean((Vin_ideal - corrected_output)**2)
corrected_snr = 10 * np.log10(signal_power / noise_power)

print(f"Calibrated Pipeline SNR: {corrected_snr:.1f} dB")

# Plotting
plt.plot(time, Dout_25bit, label=f'SNR={snr:.1f} dB')
plt.plot(time, badDout_25bit, label=f'SNR={bad_snr:.1f} dB')
plt.plot(time, corrected_output, label=f'SNR={corrected_snr:.1f} dB')
plt.xlabel('time')
plt.ylabel('Dout')
plt.title('13-bit 6-stage pipeline ADC')
plt.legend(loc='upper right')
plt.show()

MSE = ((np.sum(Dout_25bit) - np.sum(corrected_output))**2)/len(time)
print(f'MSE = {MSE:.1f}')
diffcount = 0
i = 0
while i < len(bits_25bit):
    if bits_25bit[i] != original_bits[i]:
        diffcount += 1
        i += 1
    else:
        i += 1
BER = diffcount / len(bits_25bit)
print(f'BER = {BER:.1f}')
