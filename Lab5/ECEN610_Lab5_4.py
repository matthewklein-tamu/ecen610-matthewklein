#MATTHEW KLEIN
#ECEN 610
#LAB 5 Data Conversion Basics
# 4.

import numpy as np

#vector
h = np.array([43, 115, 85, 101, 122, 170, 75, 146, 125, 60, 95, 95, 115, 40, 120, 242])

# a.) Calculate DNL and INL
num_codes = len(h)
total_counts = np.sum(h)
ideal_count = total_counts / num_codes

# Differential Non-Linearity (DNL)
dnl = (h - ideal_count) / ideal_count

# Integral Non-Linearity (INL)
inl = np.cumsum(dnl)

print("DNL:", np.round(dnl, 3))
print("INL:", np.round(inl, 3))

# b.) Peak DNL and INL
peak_dnl = np.max(np.abs(dnl))
peak_inl = np.max(np.abs(inl))

print("Peak DNL:", round(peak_dnl, 3))
print("Peak INL:", round(peak_inl, 3))

# c.) Monotonicity: ADC is monotonic if all DNL > -1 (no missing codes)
is_monotonic = np.all(dnl > -1)

if is_monotonic:
    print("The ADC is monotonic")
else:
    print("the ADC is not monotonic")
