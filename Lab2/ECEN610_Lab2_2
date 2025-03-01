#MATTHEW KLEIN
#ECEN 610
#LAB 2
# Quantization

# Create a perfect quantizer with 6 bits of resolution and with flexible sampling rate.
# For a 200 MHz full scale input tone, sample and quantize the sinewave at 400 MHz and plot the PSD of 30 periods.
# What is the SNR? Repeat the SNR calculation for 100 periods of the same signal.
# Make your own conclusions about this test regarding periodicity of quantization noise and the impact of this in the SNR.
# How can you solve this problem?

import numpy as np
import matplotlib.pyplot as plt

N = 12 #6 bits of resolution
Fin = 200e6 # input frequency
Fs = 400e6 # sample frequency (flexible sampling rate?)
periods = 100 * (1/Fin) # Plot the PSD of 30 periods

# Tone
tone_time = np.arange(0, periods, 1/Fs) # nTs
tone = np.sin(2 * np.pi * Fin * tone_time)

# Quantize
# 6 bit quantization

# HINT: Write a python function using the numpy.round to perform Quantization on your waveform.

def quantization(signal, N): # Everything is zero? ???
    minmax_signal = ((2*(signal - np.min(signal))) / (np.max(signal) - np.min(signal))) - 1 # Min-max to get everything from 1 to -1?
    quantized_signal = np.round(minmax_signal * ((2**N) - 1)) / ((2**N) - 1)
    return quantized_signal

quantized_tone = quantization(tone, N)

DFT_result = np.fft.fft(quantized_tone)
DFT_frequency = np.fft.fftfreq(len(quantized_tone), 1/Fs)

plt.plot(DFT_frequency, 10*np.log10(np.abs((DFT_result))))
plt.title('DFT of quantized tone')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.show()

frequency_index = np.argmax(np.abs(DFT_result))
signal_power = (np.abs(DFT_result[frequency_index]) ** 2)
noise_power = np.sum(np.abs(DFT_result)**2) - signal_power
SNR = 10 * np.log10(signal_power/noise_power)
print(f"SNR: {SNR:.2f} dB")

psd = np.abs(DFT_result) ** 2 / (len(quantized_tone) * Fs)
plt.plot(DFT_frequency, 10*np.log10(psd))
plt.title('Plot of Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

# Hanning #############################################################################################################

hanning_tone = np.hanning(len(tone_time)) * quantized_tone
DFT_result = np.fft.fft(hanning_tone)
DFT_frequency = np.fft.fftfreq(len(hanning_tone), 1/Fs)

plt.plot(DFT_frequency, 10*np.log10(np.abs((DFT_result))))
plt.title('DFT of quantized tone (Hanning)')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.show()

frequency_index = np.argmax(np.abs(DFT_result))
signal_power = (np.abs(DFT_result[frequency_index]) ** 2) * 1.63
noise_power = np.sum(np.abs(DFT_result)**2) - signal_power
SNR = 10 * np.log10(signal_power/noise_power)
print(f"SNR Hanning: {SNR:.2f} dB")
