# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math

N = 1000

# Open dicts
with open('p_vals_set_0.pkl', 'rb') as f:
    p_vals_0 = pickle.load(f)

with open('p_vals_set_1.pkl', 'rb') as f:
    p_vals_1 = pickle.load(f)

with open('p_vals_set_2.pkl', 'rb') as f:
    p_vals_2 = pickle.load(f)

with open('p_vals_set_3.pkl', 'rb') as f:
    p_vals_3 = pickle.load(f)

with open('p_vals_set_4.pkl', 'rb') as f:
    p_vals_4 = pickle.load(f)

with open('p_vals_set_5.pkl', 'rb') as f:
    p_vals_5 = pickle.load(f)

# PLOT 1: Sparsity_1 = k, Sparsity_2 = s, rho = 0
std_0_ov = np.log(np.std(p_vals_0, ddof=1)).item()
std_0_wi = np.log(np.sqrt(np.mean(np.var(p_vals_0, axis=1, ddof=1)))).item()

std_1_ov = np.log(np.std(p_vals_1, ddof=1)).item()
std_1_wi = np.log(np.sqrt(np.mean(np.var(p_vals_1, axis=1, ddof=1)))).item()

std_2_ov = np.log(np.std(p_vals_2, ddof=1)).item()
std_2_wi = np.log(np.sqrt(np.mean(np.var(p_vals_2, axis=1, ddof=1)))).item()

std_3_ov = np.log(np.std(p_vals_3, ddof=1)).item()
std_3_wi = np.log(np.sqrt(np.mean(np.var(p_vals_3, axis=1, ddof=1)))).item()

std_4_ov = np.log(np.std(p_vals_4, ddof=1)).item()
std_4_wi = np.log(np.sqrt(np.mean(np.var(p_vals_4, axis=1, ddof=1)))).item()

std_5_ov = np.log(np.std(p_vals_5, ddof=1)).item()
std_5_wi = np.log(np.sqrt(np.mean(np.var(p_vals_5, axis=1, ddof=1)))).item()

within = [std_0_wi, std_1_wi, std_2_wi, std_3_wi, std_4_wi, std_5_wi]
overall = [std_0_ov, std_1_ov, std_2_ov, std_3_ov, std_4_ov, std_5_ov]

x = [-1, -2, -3, -4, -5, -6, -7, -8, -9]
y = [-1, -2, -3, -4, -5, -6, -7, -8, -9]

plt.figure()
# Plot lines 
plt.plot(overall, within, marker='o', linestyle='-', color='#0E8784', label='Variation')
plt.plot(x, y, linestyle='--', color='black', label='Identity line')

plt.xlabel("log(Overall s.d.)", fontsize=14)
plt.ylabel("log(Within-replicate s.d.)", fontsize=14)
plt.legend()
plt.savefig('randomization.pdf')

print(within)
print(overall)