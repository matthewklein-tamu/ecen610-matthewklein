#MATTHEW KLEIN
#ECEN 610
#LAB 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import tf2zpk

#######################################################################################################################
# 1 a

# The transfer function

numerator = [1, 1]
denominator = [1, -1]

# use the freqz

w, h = freqz(numerator, denominator, worN=512)

# try to plot this? Scipy say "Using Matplotlib’s matplotlib.pyplot.plot function as the callable for plot produces
# unexpected results, as this plots the real part of the complex transfer function, not the magnitude.
# Try lambda w, h: plot(w, np.abs(h)). ???

# magnitude

plt.figure()
plt.plot(w, 20*np.log10(np.abs(h))) #args? this works yes?
plt.title('Magnitude Response')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.show()

# phase

plt.figure()
plt.plot(w, np.angle(h))
plt.title('Phase Response')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.show()

####################################################################################################################
# Calculate poles and zeros
# 1 b

# coefficients
numerator_1b = [1, 1, 1, 1, 1]
denominator_1b = [1]

# get zeros and poles

zeros, poles, gain = tf2zpk(numerator_1b, denominator_1b)

print("Zeros:", zeros)
print("Poles:", poles)

# plot the poles and zeros

plt.figure()

plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros')
plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles')
plt.title('Poles and Zeros')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')

plt.legend()
plt.grid()
plt.show()

