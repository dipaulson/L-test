# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math 

N = 1000

x = [10, 16, 20, 26, 30, 36, 40]

# Helper function for error bars
def errors(A):
    errs = np.zeros(len(A))
    for i in range(len(A)):
        errs[i] = 1.96 * math.sqrt((A[i] * (1-A[i])) / N)
    return errs

# Setting #1 -- Heavy-tailed errors

dfs = [2, 5, 10, 15, 20, 30]

with open('F_tail.pkl', 'rb') as f:
    powers_F_tail = pickle.load(f)

with open('L_tail.pkl', 'rb') as f:
    powers_L_tail = pickle.load(f)

for i in range(6):
    plt.figure()
    plt.errorbar(x, powers_F_tail[i], errors(powers_F_tail[i]), fmt='o-', color='#264653', markersize=2, label='F-test')
    plt.errorbar(x, powers_L_tail[i], errors(powers_L_tail[i]), fmt='o-', color='#2a9d8f', markersize=2, label='L-test')
    plt.xlabel("Sample size", fontsize=14)
    plt.ylabel("Type I error", fontsize=14)
    #plt.title(f"Heavy-tailed error (df = {dfs[i]})")
    #plt.legend()
    plt.yticks(np.arange(0, 0.2, step=0.05))
    plt.grid(True)
    plt.savefig(f"heavy_tailed_error_df_{dfs[i]}.png")
    plt.close()

# Setting #2 -- Skewed errors

alphas = [1, 2, 4, 6, 8, 10]

with open('F_skew.pkl', 'rb') as f:
    powers_F_skew = pickle.load(f)

with open('L_skew.pkl', 'rb') as f:
    powers_L_skew = pickle.load(f)

for i in range(6):
    plt.figure()
    plt.errorbar(x, powers_F_skew[i], errors(powers_F_skew[i]), fmt='o-', color='#264653', markersize=2, label='F-test')
    plt.errorbar(x, powers_L_skew[i], errors(powers_L_skew[i]), fmt='o-', color='#2a9d8f', markersize=2, label='L-test')
    plt.xlabel("Sample size", fontsize=14)
    plt.ylabel("Type I error", fontsize=14)
    #plt.title(rf"Skewed error ($\alpha = {alphas[i]}$)")
    #plt.legend()
    plt.yticks(np.arange(0, 0.2, step=0.05))
    plt.grid(True)
    plt.savefig(f"skewed_error_alpha_{alphas[i]}.png")
    plt.close()

# Setting #3 -- Heteroskedastic errors

etas = [0.01, 0.25, 0.5, 1, 4, 8]

with open('F_sked.pkl', 'rb') as f:
    powers_F_sked = pickle.load(f)

with open('L_sked.pkl', 'rb') as f:
    powers_L_sked = pickle.load(f)


for i in range(6):
    plt.figure()
    plt.errorbar(x, powers_F_sked[i], errors(powers_F_sked[i]), fmt='o-', color='#264653', markersize=2, label='F-test')
    plt.errorbar(x, powers_L_sked[i], errors(powers_L_sked[i]), fmt='o-', color='#2a9d8f', markersize=2, label='L-test')
    plt.xlabel("Sample size", fontsize=14)
    plt.ylabel("Type I error", fontsize=14)
    #plt.title(rf'Heteroskedastic error ($\eta = {etas[i]}$)')
    #plt.legend()
    plt.yticks(np.arange(0, 0.2, step=0.05))
    plt.grid(True)
    plt.savefig(f"heteroskedastic_error_eta_{etas[i]}.png")
    plt.close()

# Setting #4 -- Model non-linearity

deltas = [0.3, 0.5, 1, 2, 3, 4]

with open('F_lin.pkl', 'rb') as f:
    powers_F_lin = pickle.load(f)

with open('L_lin.pkl', 'rb') as f:
    powers_L_lin = pickle.load(f)

for i in range(6):
    plt.figure()
    plt.errorbar(x, powers_F_lin[i], errors(powers_F_lin[i]), fmt='o-', color='#264653', markersize=2, label='F-test')
    plt.errorbar(x, powers_L_lin[i], errors(powers_L_lin[i]), fmt='o-', color='#2a9d8f', markersize=2, label='L-test')
    plt.xlabel("Sample size", fontsize=14)
    plt.ylabel("Type I error", fontsize=14)
    #plt.title(rf'Non-linearity ($\delta = {deltas[i]}$)')
    #plt.legend()
    plt.yticks(np.arange(0, 0.2, step=0.05))
    plt.grid(True)
    plt.savefig(f"non_linearity_delta_{deltas[i]}.png")
    plt.close()

F_max_nonlin = (0, 0)
L_max_nonlin = (0, 0)
for i in range(6):
    if np.max(powers_F_lin[i]) >= F_max_nonlin[0]:
        F_max_nonlin = (np.max(powers_F_lin[i]), i)
    if np.max(powers_L_lin[i]) >= L_max_nonlin[0]:
        L_max_nonlin = (np.max(powers_L_lin[i]), i)
print(F_max_nonlin, L_max_nonlin)


F_max_sked = (0, 0)
L_max_sked = (0, 0)
for i in range(6):
    if np.max(powers_F_sked[i]) >= F_max_sked[0]:
        F_max_sked = (np.max(powers_F_sked[i]), i)
    if np.max(powers_L_sked[i]) >= L_max_sked[0]:
        L_max_sked = (np.max(powers_L_sked[i]), i)
print(F_max_sked, L_max_sked)


F_max_skew = (0, 0)
L_max_skew = (0, 0)
for i in range(6):
    if np.max(powers_F_skew[i]) >= F_max_skew[0]:
        F_max_skew = (np.max(powers_F_skew[i]), i)
    if np.max(powers_L_skew[i]) >= L_max_skew[0]:
        L_max_skew = (np.max(powers_L_skew[i]), i)
print(F_max_skew, L_max_skew)


F_max_tail = (0, 0)
L_max_tail = (0, 0)
for i in range(6):
    if np.max(powers_F_tail[i]) >= F_max_tail[0]:
        F_max_tail = (np.max(powers_F_tail[i]), i)
    if np.max(powers_L_tail[i]) >= L_max_tail[0]:
        L_max_tail = (np.max(powers_L_tail[i]), i)
print(F_max_tail, L_max_tail)

    