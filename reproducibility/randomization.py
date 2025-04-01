# Import packages
import os 
from tests import *
import numpy as np
import random
import math
import pickle
from celer import GroupLasso, GroupLassoCV

# Set params
n = 100 # num rows
d = 50 # num cols
var = 1 # variance
k = 10 # number of coefs null sets to 0
alpha = 0.05 # significance level

def generate_design(rho):
    # Construct covariance matrix
    cov = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    # Construct mean matrix
    mean = np.zeros(d)
    # Generate n samples from d-dim multivariate normal distribution
    X = np.random.multivariate_normal(mean, cov, size=n)
    # Standardize the columns of X
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def generate_beta(sig, spars_1, spars_2):
    # Select non-zero entries among first k
    sample_1 = random.sample(list(range(0, k)), spars_1)
    # Select non-zero entries among remaining d-k
    sample_2 = random.sample(list(range(k, d)), spars_2)
    # Determine amplitude of non-zero entries among first k
    a = sig / (math.sqrt(spars_1))
    beta = np.zeros(d)
    for i in sample_1:
        sign = random.choice([-1, 1])
        beta[i] = sign * a
    for i in sample_2:
        sign = random.choice([-1, 1])
        beta[i] = sign * 5
    return beta

def power(sig, spars_1, spars_2, rho):
    p_vals = np.zeros(100)
    for i in range(1):
        # Set seed
        task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        random.seed(task_id)
        np.random.seed(task_id)
        X = generate_design(rho)
        beta = generate_beta(sig, spars_1, spars_2)
        y = np.random.multivariate_normal(X@beta, (var ** 2) * np.identity(n))
        for j in range(100):
            # Set seed
            random.seed(j)
            np.random.seed(j)
            p_vals[j] = L_pval(y, X, k)
    return p_vals

signal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1]) # signal of first k coeffs

# Fix sparsity_1 = k, sparsity_2 = s, rho = 0
# Vary signal
task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

p_vals_0_0 = power(signal[0], k, 5, 0)
p_vals_1_0 = power(signal[1], k, 5, 0)
p_vals_2_0 = power(signal[2], k, 5, 0)
p_vals_3_0 = power(signal[3], k, 5, 0)
p_vals_4_0 = power(signal[4], k, 5, 0)
p_vals_5_0 = power(signal[5], k, 5, 0)

p_vals = np.zeros((6, 100))
p_vals[0] = p_vals_0_0
p_vals[1] = p_vals_1_0
p_vals[2] = p_vals_2_0
p_vals[3] = p_vals_3_0
p_vals[4] = p_vals_4_0
p_vals[5] = p_vals_5_0

with open(f'p_vals_{task_id}.pkl', 'wb') as f:
    pickle.dump(p_vals, f)
