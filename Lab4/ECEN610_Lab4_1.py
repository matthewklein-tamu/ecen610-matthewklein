#MATTHEW KLEIN
#ECEN 610
#LAB 4 Sampler Error Modeling and Correction
# 1.

import numpy as np
import matplotlib.pyplot as plt

tau = 10e-12 # 10 ps
input_frequency = 1e9  # 1 GHz
sampling_frequency = 10e9  # 10 GHz

sampling_period = (1 / sampling_frequency)
sampling_no = 20 * sampling_period # increase the number of samples
t = np.linspace(0, sampling_no, int(sampling_no * 1e12)) # time array (sin2pift)

Vin = np.sin(2 * np.pi * input_frequency * t) # VIN
Vout = np.zeros_like(t) # fill arary

loop = 0 #loop
for i in range(1, len(t)):
    if t[i] // sampling_period > t[loop] // sampling_period:
        loop = i
        Vout[i] = Vin[i]
    else:
        Vout[i] = Vout[i-1]

plt.plot(t * 1e9, Vout)
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.title('Output Voltage')
plt.grid()
plt.show()
