#MATTHEW KLEIN
#ECEN 610
#LAB 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import tf2zpk

#######################################################################################################################

#f1 = 300*10^6  # x1
#f2 = 800*10^6  # x2
#fs = 500*10^6  # sampling frequency
#ts = np.arange(0, 1 / fs, (1 / (10 * fs))) # sampling time array (to create signal)

# true signals
#signal1 = np.cos(2 * np.pi * f1 * ts)
#signal2 = np.cos(2 * np.pi * f2 * ts)

# sampled signals
#sample1 = np.cos(2 * np.pi * f1 * np.arange(0, len(ts)) / fs)
#sample2 = np.cos(2 * np.pi * f2 * np.arange(0, len(ts)) / fs)

# plotting the signals
#plt.figure(figsize=(6, 4.8))

# 300 mhz signal
#plt.subplot(2, 2, 1)
#plt.plot(ts, signal1, color='blue', label='300 MHz signal')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('300 MHz signal')
#plt.legend()

#the 300mhz sample
#plt.subplot(2, 2, 2)
#plt.plot(ts, sample1, color='red', label='sampled signal')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('300 MHz sampled signal')
#plt.legend()

# 800 mhz signal
#plt.subplot(2, 2, 3)
#plt.plot(ts, signal2, color='blue', label='800 MHz signal')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('800 MHz singal')
#plt.legend()

# the 800 mhz sample
#plt.subplot(2, 2, 4)
#plt.plot(ts, sample2, color='red', label='sampled signal')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('800 MHz sampled signal')
#plt.legend()

#plt.show()

#######################################################################################################################

f1 = 300*10^6  # x1
fs = 500*10^6  # sampling frequency
T = 10 / f1
Ts = 1 / fs

sample = np.arange(0, T-Ts, Ts)

original_signal = np.cos(2 * np.pi * f1)

sampled_signal = np.cos(2 * np.pi * f1 * sample)

plt.figure(figsize=(10, 6))
plt.plot(sample, sampled_signal, label='sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('sampled signal')
plt.legend()
plt.grid(True)
plt.show()

