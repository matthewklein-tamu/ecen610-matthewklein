#MATTHEW KLEIN
#ECEN 610
#LAB 4 Sampler Error Modeling and Correction
# 3.

import numpy as np
import matplotlib.pyplot as plt

N = 7  # Number of ADC bits
full_scale_range = 1.0  # Full-scale range of the ADC (in volts)
delta = full_scale_range / (2 ** N)  # Quantization step size
variance_quantization_noise = delta**2 / 12  # Variance of quantization noise

fs = 10e9  # Sampling frequency (10 GHz)
f_tones = [0.2e9, 0.58e9, 1e9, 1.7e9, 2.4e9]
t = np.linspace(0, 1e-6, int(fs * 1e-6))
signal = sum(np.sin(2 * np.pi * f * t) for f in f_tones)

sampling_error = 1.4945*(10**-5) 
quantization_noise = (delta**2)/12
adc_output = signal + sampling_error + quantization_noise

# least squares
def fir_filter(adc_output, M):
    # Create the filter matrix
    X = np.zeros((len(adc_output) - M + 1, M))
    for i in range(M):
        X[:, i] = adc_output[i:len(adc_output) - M + 1 + i]
    # Estimate sampling error using least squares
    y = adc_output[M-1:]  # Target values
    h = np.linalg.lstsq(X, y, rcond=None)[0]  # FIR filter coefficients
    estimated_error = np.convolve(adc_output, h, mode='same')  # Filtered output
    return estimated_error

M_values = range(2, 11)
variance_ratios = []

for M in M_values:
    estimated_error = fir_filter(adc_output, M)
    corrected_output = adc_output + estimated_error  # Correct the ADC output
    E = corrected_output - signal  # Compute error signal
    variance_E = np.var(E)  # Variance of the error signal
    variance_ratios.append(variance_E / variance_quantization_noise)

plt.plot(M_values, variance_ratios, marker='o')
plt.title("Variance Ratio vs Taps")
plt.xlabel("M (Number of Taps in FIR Filter)")
plt.ylabel("Variance Ratio")
plt.grid()
plt.show()
