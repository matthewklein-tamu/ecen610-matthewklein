#MATTHEW KLEIN
#ECEN 610
#LAB 3 GMC Filters
# 1a
import numpy as np
import matplotlib.pyplot as plt

# Paramters
N = 8
freq = 2.4e9 # (Hz)
T = 1 / freq
Cs = 15.925e-12 # Capacitance (F)

f = np.linspace(0, 2.5e9, 10000) # Frequency array from 0 to 1000 MHz

H = ((N*T)/Cs) * np.sinc(f * (N*T)) # Transfer function

# Plot
plt.plot(f, 20 * np.log10(np.abs(H)))
plt.title('Transfer Function Plot 1. (a)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

# b

# Parameters
f = 2400e6
frequencies = np.linspace(0, 2*f, 10000)  # Frequency range in Hz

# z = e^jw
omega = 2 * np.pi * frequencies / f
z = np.exp(1j * omega)

Hf = (1 / ((15.925e-12) * (2.4e9)))*(sum([z**(-n) for n in range(1, 8)]))  # z^-1 + z^-2 + ... + z^-7

# compute magnitude in dB
magnitude_dB = 20*np.log10(np.abs(Hf))

# Plot
plt.plot(frequencies, magnitude_dB)
plt.title('Transfer Function Plot 1. (b)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()
