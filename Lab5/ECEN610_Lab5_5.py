#MATTHEW KLEIN
#ECEN 610
#LAB 5 Data Conversion Basics
# 5.

import numpy as np
import matplotlib.pyplot as plt

# Given DNL values in LSB
dnl = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])
codes = np.arange(8)
inl_uncorrected = np.cumsum(dnl)
m, b = np.polyfit(codes, inl_uncorrected, 1)
inl_corrected = inl_uncorrected - (m * codes + b)

offset_error = 0.5           # in LSB
full_scale_error = 0.5       # in LSB
gain_correction = full_scale_error / (len(codes) - 1)

# Transfer curve
actual = codes + inl_corrected + offset_error + gain_correction * codes

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(codes, actual)
plt.title("3-bit ADC Transfer Curve")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()
