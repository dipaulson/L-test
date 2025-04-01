# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math

N = 1000

# Open dict
with open('test_powers.pkl', 'rb') as f:
    powers = pickle.load(f)

# Helper function for error bars
def errors(A):
    errs = np.zeros(len(A))
    for i in range(len(A)):
        errs[i] = 1.96 * math.sqrt((A[i] * (1-A[i])) / N)
    return errs

for key, value in powers.items():
    # Split the key by commas to get the variable names
    variable_names = key.split(', ')
    
    # Iterate over the variable names and assign the corresponding value from the array
    for name, val in zip(variable_names, value):
        globals()[name] = val

# PLOT 1:
# Fix sparsity_1 = 10, sparsity_2 = 5, rho = 0
# Vary signal

x = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
F = [F_0_5_1_0, F_1_5_1_0, F_2_5_1_0, F_3_5_1_0, F_4_5_1_0, F_5_5_1_0]
F_errs = errors(F)
O = [O_0_5_1_0, O_1_5_1_0, O_2_5_1_0, O_3_5_1_0, O_4_5_1_0, O_5_5_1_0]
O_errs = errors(O)

plt.figure()
# Plot lines
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Signal", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_sig_low_corr.png')


# PLOT 2:
# Fix sparsity_1 = 10, sparsity_2 = 5, rho = 0.9
# Vary signal

x = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
F = [F_0_5_1_5, F_1_5_1_5, F_2_5_1_5, F_3_5_1_5, F_4_5_1_5, F_5_5_1_5]
F_errs = errors(F)
O = [O_0_5_1_5, O_1_5_1_5, O_2_5_1_5, O_3_5_1_5, O_4_5_1_5, O_5_5_1_5]
O_errs = errors(O)

plt.figure()
# Plot lines 
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Signal", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_sig_high_corr.png')


# PLOT 3:
# Fix sparsity_2 = 5, rho = 0, sig = 0.2
# Vary sparsity_1
x = [1, 2, 4, 6, 8, 10]
F = [F_1_0_1_0, F_1_1_1_0, F_1_2_1_0, F_1_3_1_0, F_1_4_1_0, F_1_5_1_0]
F_errs = errors(F)
O = [O_1_0_1_0, O_1_1_1_0, O_1_2_1_0, O_1_3_1_0, O_1_4_1_0, O_1_5_1_0]
O_errs = errors(O)

plt.figure()
# Plot lines 
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Sparsity in null coefficients", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_spars1_low_sig.png')


# PLOT 4:
# Fix sparsity_2 = 5, rho = 0, sig = 0.6
# Vary sparsity_1
x = [1, 2, 4, 6, 8, 10]
F = [F_3_0_1_0, F_3_1_1_0, F_3_2_1_0, F_3_3_1_0, F_3_4_1_0, F_3_5_1_0]
F_errs = errors(F)
O = [O_3_0_1_0, O_3_1_1_0, O_3_2_1_0, O_3_3_1_0, O_3_4_1_0, O_3_5_1_0]
O_errs = errors(O)

plt.figure()
# Plot lines
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Sparsity in null coefficients", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_spars1_high_sig.png')

# PLOT 5:
# Fix sparsity_1 = 4, signal = 0.2, rho = 0
# Vary sparsity_2
x = [0, 5, 10, 20, 30, 40]
F = [F_1_2_0_0, F_1_2_1_0, F_1_2_2_0, F_1_2_3_0, F_1_2_4_0, F_1_2_5_0]
F_errs = errors(F)
O = [O_1_2_0_0, O_1_2_1_0, O_1_2_2_0, O_1_2_3_0, O_1_2_4_0, O_1_2_5_0]
O_errs = errors(O)

plt.figure()
# Plot lines 
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Sparsity in non-null coefficients", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_spars2_low_sig.png')

# PLOT 6:
# Fix sparsity_1 = 4, signal = 0.4, rho = 0
# Vary sparsity_2
x = [0, 5, 10, 20, 30, 40]
F = [F_2_2_0_0, F_2_2_1_0, F_2_2_2_0, F_2_2_3_0, F_2_2_4_0, F_2_2_5_0]
F_errs = errors(F)
O = [O_2_2_0_0, O_2_2_1_0, O_2_2_2_0, O_2_2_3_0, O_2_2_4_0, O_2_2_5_0]
O_errs = errors(O)

plt.figure()
# Plot lines 
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Sparsity in non-null coefficients", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_spars2_med_sig.png')


# PLOT 7:
# Fix sparsity_1 = 4, signal = 0.6, rho = 0
# Vary sparsity_2
x = [0, 5, 10, 20, 30, 40]
F = [F_3_2_0_0, F_3_2_1_0, F_3_2_2_0, F_3_2_3_0, F_3_2_4_0, F_3_2_5_0]
F_errs = errors(F)
O = [O_3_2_0_0, O_3_2_1_0, O_3_2_2_0, O_3_2_3_0, O_3_2_4_0, O_3_2_5_0]
O_errs = errors(O)

plt.figure()
# Plot lines 
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Sparsity in non-null coefficients", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_spars2_high_sig.png')


# PLOT 8:
# Fix sparsity_1 = 4, sparsity_2 = 5, signal = 0.4
# Vary rho
x = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
F = [F_2_2_1_0, F_2_2_1_1, F_2_2_1_2, F_2_2_1_3, F_2_2_1_4, F_2_2_1_5]
F_errs = errors(F)
O = [O_2_2_1_0, O_2_2_1_1, O_2_2_1_2, O_2_2_1_3, O_2_2_1_4, O_2_2_1_5]
O_errs = errors(O)

plt.figure()
# Plot lines 
plt.errorbar(x, F, F_errs, fmt='o-', color="#2a9d8f", markersize=2, label='Min rule')
plt.errorbar(x, O, O_errs, fmt='o-', color='#b78876', markersize=2, label='1se rule')

plt.xlabel("Correlation", fontsize=14)
plt.ylabel("Power", fontsize=14)
#plt.legend()
plt.yticks(np.arange(0, 1.1, step=0.2))
plt.savefig('setting_1/vary_corr.png')