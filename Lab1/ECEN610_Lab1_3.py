#MATTHEW KLEIN
#ECEN 610
#LAB 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import tf2zpk

#######################################################################################################################

# x(t) = cos(2 * pi * F * t) where F = 2 MHZ
# sample the signal at Fs = 5 MHz
# compute a 64 point DFT  in python and plot the output

F = 2e6
Fs = 5e6

# scipy says get the spacing of the sample? Ts?
# need the signal?

xt = np.cos(2 * np.pi* F * (np.arange(64)/Fs))

xt_blackman = xt * np.blackman(64)

xf = np.fft.fft(xt_blackman, 64) # perform fft

x_axis = np.fft.fftfreq(64, 1/Fs) #x-axis from scipy
plt.plot(x_axis, np.abs(xf)) # complex values? scipy says to take absolute
plt.title('DFT of the signal')
plt.xlabel(' Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()

#######################################################################################################################

# y(t) = cos(2 * pi * F1 * t) + cos(2 * pi * F2 * t)
# where F1 = 200 MHz and F2 = 400 MHz.
# sample the signal at Fs = 1 GHz.
# plot a 64 point DFT

# similar to above

F1 = 200e6
F2 = 400e6
Fs = 500e6

# make the signal like above?

yt = np.cos(2 * np.pi * F1 * (np.arange(64)/Fs)) + np.cos(2 * np.pi * F2 * (np.arange(64)/Fs))
yt_blackman = yt * np.blackman(64)
yf = np.fft.fft(yt_blackman, 64)

x_axis = np.fft.fftfreq(64, 1/Fs) # plot

# Plot the DFT
plt.plot(x_axis, np.abs(yf))
plt.title('dft of the signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()

