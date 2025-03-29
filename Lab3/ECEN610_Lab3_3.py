#MATTHEW KLEIN
#ECEN 610
#LAB 3 GMC Filters
# 3a
import numpy as np
import matplotlib.pyplot as plt

# similar to 2

# parameters
ch = 15.425e-12
cr = 0.5e-12
a = ch / (ch + cr)
fs = 2.4e9
frequencies = np.linspace(0, 2*fs, 10000)  # Frequency range (0 to Nyquist frequency)

omega = 2 * np.pi * frequencies / fs

# Compute the transfer function
H = (1 / (1 - a * np.exp(-1j * omega))) + (1 / (1 - a * np.exp(-1j * omega)**2)) + (1 / (1 - a * np.exp(-1j * omega)**3)) + (1 / (1 - a * np.exp(-1j * omega)**4))
H = 20 * np.log10(np.abs(H))

# Plot
plt.plot(frequencies, H)
plt.title('Transfer Function Plot 3. (a)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

########################################################################################################################
# 3 (b)

ch = 15.425e-12
cr = 0.5e-12
Cs = 0.5e-12 # Capacitance (F)
a = ch / (ch + cr)
fs = 2.4e9
N = 8
T = 1 / fs
frequencies = np.linspace(0, 5e9, 10000)  # Frequency range (0 to Nyquist frequency)

omega = 2 * np.pi * frequencies / fs

# Compute the transfer function
H = ((((N*T)/Cs) * np.sinc(frequencies * (N*T))) / (1 - a * np.exp(-1j * omega))) + ((((N*T)/Cs) * np.sinc(frequencies * (N*T))) / (1 - a * np.exp(-1j * omega)**2))
H = 20 * np.log10(np.abs(H))

# Plot
plt.plot(frequencies, H)
plt.title('Transfer Function Plot 3. (b)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()

########################################################################################################################
# 3 (c)
# similar to 3a but with different CS

# parameters
ch = 15.425e-12
cr1 = 0.5e-12
a1 = ch / (ch + cr1)
cr2 = 0.4e-12
cr3 = 0.3e-12
cr4 = 0.6e-12
a2 = ch / (ch + cr2)
a3 = ch / (ch + cr3)
a4 = ch / (ch + cr4)
fs = 2.4e9
frequencies = np.linspace(0, 2*fs, 10000)  # Frequency range (0 to Nyquist frequency)

omega = 2 * np.pi * frequencies / fs

# Compute the transfer function
H = (1 / (1 - a1 * np.exp(-1j * omega))) + (1 / (1 - a2 * np.exp(-1j * omega)**2)) + (1 / (1 - a3 * np.exp(-1j * omega)**3)) + (1 / (1 - a4 * np.exp(-1j * omega)**4))
H = 20 * np.log10(np.abs(H))

# Plot
plt.plot(frequencies, H)
plt.title('Transfer Function Plot 3. (c)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()
