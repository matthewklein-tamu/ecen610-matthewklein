#MATTHEW KLEIN
#ECEN 610
#LAB 3 GMC Filters
# 2

import numpy as np
import matplotlib.pyplot as plt

# parameters
ch = 15.425e-12
cr = 0.5e-12
a = ch / (ch + cr)
fs = 2.4e9
frequencies = np.linspace(0, 2*fs, 10000)  # Frequency range (0 to Nyquist frequency)

omega = 2 * np.pi * frequencies / fs

# Compute the transfer function
H = 1 / (1 - a * np.exp(-1j * omega)) # z = e^-jw
H = 20 * np.log10(np.abs(H))

# Plot
plt.plot(frequencies, H)
plt.title('Transfer Function Plot 2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()
