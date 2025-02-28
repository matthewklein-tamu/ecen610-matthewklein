#MATTHEW KLEIN
#ECEN 610
#LAB 2
# Signal to Noise Ratio

# Generate a tone with frequency 2 MHz and amplitude 1 V.
# Sample the tone at frequency Fs = 5 MHz.

import numpy as np
import matplotlib.pyplot as plt

ft = 2e6 # tone frequency
A = 1 # tone amplitude
Fs = 5e6 # sampling frequency

# vsig(t) = cos(2 * pi * fin * t)
# t -> n * Ts = n / Fs
# vsig[n] = cos(2 * pi * fin/Fs * n)

nTs = np.arange(0, 50e-6, 1 / Fs) # n * Ts
tone = A * np.cos(2 * np.pi * ft * nTs) # cos(2 * pi * fin/Fs * n)

### Plotting the tone
#plt.plot(nTs, tone)
#plt.title('2 MHz Tone Sampled at 5 MHz')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude (V)')
#plt.show()

# Add Gaussian noise to the sampled sine wave such that the signal SNR is 50 dB.
# First find the variance of the Gaussian noise needed to produce the target SNR.

# SNR = 50 dB
# variance = delta^2 / 12
# delta = VFS / 2^N
# SNR = 50 dB, N = 8

# Finding the variance of the Gaussian noise needed to produce the target SNR
snrVariance = ((1 / (2**8))**2) / 12
snrSTD = snrVariance**0.5

# Add Gaussian noise to the sampled sine wave such that the signal SNR is 50 dB
GaussianNoise = np.random.normal(0, snrSTD, len(nTs))
noisy_sample = tone + GaussianNoise

# Calculate and plot the PSD from the DFT of the noisy samples.

DFTNoisy = np.fft.fft(noisy_sample, 200) # 200 points

DFTNoisy_freq = np.fft.fftfreq(200, 1/Fs)
psd = np.abs(DFTNoisy) ** 2 / (len(noisy_sample) * Fs) # Calculate the PSD and plot it

###### NORMALIZING ####################################################################################################

#DFTNoisy_normalized = DFTNoisy / np.max(DFTNoisy)

#plt.plot(DFTNoisy_freq, abs(DFTNoisy_normalized))
#plt.show()

psd_normalized = psd / np.max(psd)

plt.plot(DFTNoisy_freq, 10*np.log10(psd_normalized))
plt.title('Plot of Power Spectral Density (Gaussian) (Normalized)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

#signal_power = (np.abs(DFTNoisy_normalized[80]**2))+(np.abs(DFTNoisy_normalized[120]**2))
#noise_power = np.sum((np.abs(DFTNoisy_normalized)**2)) - signal_power
#print("Gaussian noise:")
#print(10*np.log10(signal_power/noise_power))

#######################################################################################################################

plt.plot(DFTNoisy_freq, 10*np.log10(psd))
plt.title('Plot of Power Spectral Density (Gaussian)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

# Corroborate that the SNR calculation from the DFT plot gives the theoretical result.

#print(DFTNoisy_freq[80])
#print(DFTNoisy_freq[120])

# SNR result shows that N must be greater than or equal to 8 to have an SNR of at least 50.
# N of 7 could be below 50 dB SNR

signal_power = (np.abs(DFTNoisy[80]**2))+(np.abs(DFTNoisy[120]**2))
noise_power = np.sum((np.abs(DFTNoisy)**2)) - signal_power
print("Gaussian noise:")
print(10*np.log10(signal_power/noise_power))

# What would be the variance of a uniformly distributed noise to obtain the same SNR.

UniformNoise = np.random.uniform(0, 2*snrSTD, len(nTs)) # Approximately double the variance
Uniform_noisy_sample = tone + UniformNoise
Uniform_DFTNoisy = np.fft.fft(Uniform_noisy_sample, 200)
Uniform_DFTNoisy_freq = np.fft.fftfreq(200, 1/Fs)
Uniform_psd = np.abs(Uniform_DFTNoisy) ** 2 / (len(Uniform_noisy_sample) * Fs)
plt.plot(Uniform_DFTNoisy_freq, 10*np.log10(Uniform_psd))
plt.title('Plot of Power Spectral Density (Uniform Noise)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()
Uniform_psd_normalized = Uniform_psd / np.max(Uniform_psd)
plt.plot(Uniform_DFTNoisy_freq, 10*np.log10(Uniform_psd_normalized))
plt.title('Plot of Power Spectral Density (Uniform Noise) (Normalized)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()
Uniform_signal_power = (np.abs(Uniform_DFTNoisy[80]**2))+(np.abs(Uniform_DFTNoisy[120]**2))
Uniform_noise_power = np.sum((np.abs(Uniform_DFTNoisy)**2)) - Uniform_signal_power
print("Uniform noise:")
print(10*np.log10(Uniform_signal_power/Uniform_noise_power))

# b

# Apply a Hanning window

# Take noisy sample and apply hanning window

hanning = np.hanning(len(nTs))
hanning_signal = hanning * noisy_sample
DFT_hanning = np.fft.fft(hanning_signal, 200)
Hanning_freq = np.fft.fftfreq(200, 1/Fs)
psd_hanning = np.abs(DFT_hanning) ** 2 / (len(hanning_signal) * Fs)

plt.plot(Hanning_freq, 10*np.log10(psd_hanning))
plt.title('Plot of Power Spectral Density (Hanning)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

signal_power = (np.abs(DFT_hanning[80]**2))+(np.abs(DFT_hanning[120]**2))
noise_power = np.sum((np.abs(DFT_hanning)**2)) - signal_power

signal_power = signal_power * (1.63) # Account for spread of signal power, hanning correction factor

print("Hanning:")
print(10*np.log10(signal_power/noise_power))

# noramlized hanning?

plt.plot(Hanning_freq, 10*np.log10(psd_hanning/max(psd_hanning)))
plt.title('Plot of Power Spectral Density (Hanning) (Normalized)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

# Apply a Hamming window

hamming = np.hamming(len(nTs))
hamming_signal = hamming * noisy_sample
DFT_hamming = np.fft.fft(hamming_signal, 200)
Hamming_freq = np.fft.fftfreq(200, 1/Fs)
psd_hamming = np.abs(DFT_hamming) ** 2 / (len(hamming_signal) * Fs)

plt.plot(Hamming_freq, 10*np.log10(psd_hamming))
plt.title('Plot of Power Spectral Density (Hamming)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

signal_power = (np.abs(DFT_hamming[80]**2))+(np.abs(DFT_hamming[120]**2))
noise_power = np.sum((np.abs(DFT_hamming)**2)) - signal_power

signal_power = signal_power * (1.59) # Account for spread of signal power, hamming correction factor

print("Hamming:")
print(10*np.log10(signal_power/noise_power))

# normalize hamming

plt.plot(Hamming_freq, 10*np.log10(psd_hamming/max(psd_hamming)))
plt.title('Plot of Power Spectral Density (Hamming) (Normalized)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

# Blackman

blackman = np.blackman(len(nTs))
blackman_signal = blackman * noisy_sample
DFT_blackman = np.fft.fft(blackman_signal, 200)
blackman_freq = np.fft.fftfreq(200, 1/Fs)
psd_blackman = np.abs(DFT_blackman) ** 2 / (len(blackman_signal) * Fs)

plt.plot(blackman_freq, 10*np.log10(psd_blackman))
plt.title('Plot of Power Spectral Density (Blackman)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()

signal_power = (np.abs(DFT_blackman[80]**2))+(np.abs(DFT_blackman[120]**2))
noise_power = np.sum((np.abs(DFT_blackman)**2)) - signal_power

signal_power = signal_power * (1.97) # Account for spread of signal power, hamming correction factor

print("Blackman:")
print(10*np.log10(signal_power/noise_power))

# normalize blackman

plt.plot(blackman_freq, 10*np.log10(psd_blackman/max(psd_blackman)))
plt.title('Plot of Power Spectral Density (Hamming) (Normalized)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.grid()
plt.show()
