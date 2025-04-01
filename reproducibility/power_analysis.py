# Import packages
import tests
import os 
import numpy as np
import random
import math
from scipy.stats import f

# F-test implementation
def F_stat(y, X, V, params):
    [n, d, k] = params
    V_k = V[:,0:k]
    P_k = tests.proj(X[:,k:])
    sigma = np.linalg.norm(y - P_k@y)
    beta_OLS = tests.OLS(y, X)
    norm_sq = np.linalg.norm(V_k.T@X[:,0:k]@beta_OLS[0:k])**2
    return (norm_sq/k)/((sigma**2 - norm_sq)/(n-d))

def F_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = tests.construct_V(X, params)
    obs = F_stat(y, X, V, params)
    dfn, dfd = k, (n - d)
    return (1 - f.cdf(obs, dfn, dfd))

# Generate design matrix
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

# Generate coefficient vector
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

alpha = 0.05 # significance level
def power(sig, spars_1, spars_2, rho):
    rejects_F = 0 # F-test
    rejects_O = 0 # Oracle test
    rejects_L = 0 # L-test
    rejects_R1 = 0 # Recentered F-test with gamma_1
    rejects_R2 = 0 # Recentered F-test with gamma_2
    for i in range(1):
        X = generate_design(rho)
        beta = generate_beta(sig, spars_1, spars_2)
        y = np.random.multivariate_normal(X@beta, (var ** 2) * np.identity(n))
        # Calc p-values
        p_F = F_pval(y, X, k)
        p_O = tests.oracle_pval(y, X, beta, k)
        p_L = tests.L_pval(y, X, k)
        gam_1 = tests.gamma_1(y, X, k)
        gam_2 = tests.gamma_2(y, X, k)
        p_R1 = tests.R_pval(y, X, gam_1, k)
        if np.all(gam_2 == 0):
            p_R2 = p_F
        else:
            p_R2 = R_pval(y, X, gam_2, k)
        # Reject/retain
        if (p_F <= alpha):
            rejects_F += 1
        if (p_O <= alpha):
            rejects_O += 1
        if (p_L <= alpha):
            rejects_L += 1
        if (p_R1 <= alpha):
            rejects_R1 += 1
        if (p_R2 <= alpha):
            rejects_R2 += 1
    return [rejects_F, rejects_O, rejects_L, rejects_R1, rejects_R2]

"""
# Low-dimensional model setting:

n = 100 # num rows
d = 50 # num cols
var = 1 # variance
k = 10 # number of null coefs
alpha = 0.05 # significance level

signal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
sparsity_1 = np.array([1, 2, 4, 6, 8, 10])
sparsity_2 = np.array([0, 5, 10, 20, 30, 40])
corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
"""

"""
# High-dimensional model setting:

n = 100 # num rows
d = 90 # num cols
var = 1 # variance
k = 10 # number of null coefs
alpha = 0.05 # significance level

signal = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
sparsity_1 = np.array([1, 2, 4, 6, 8, 10])
sparsity_2 = np.array([0, 10, 20, 40, 60, 80])
corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
"""

"""
# Large model setting:

n = 1000 # num rows
d = 500 # num cols
var = 1 # variance
k = 10 # number of null coefs
alpha = 0.05 # significance level

signal = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25])
sparsity_1 = np.array([1, 2, 4, 6, 8, 10])
sparsity_2 = np.array([0, 50, 100, 200, 300, 490])
corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
"""

# Set seeds
task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
random.seed(task_id)
np.random.seed(task_id)

# Vary signal, low corr
print(power(signal[0], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[2], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[4], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[5], sparsity_1[5], sparsity_2[1], corr[0]))

# Vary signal, high corr
print(power(signal[0], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[1], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[2], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[3], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[4], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[5], sparsity_1[5], sparsity_2[1], corr[5]))

# Vary sparsity_1, low signal
print(power(signal[1], sparsity_1[0], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[1], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[3], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[4], sparsity_2[1], corr[0]))

# Vary sparsity_1, high signal
print(power(signal[3], sparsity_1[0], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[1], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[3], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[4], sparsity_2[1], corr[0]))

# Vary sparsity_2, low signal
print(power(signal[1], sparsity_1[2], sparsity_2[0], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[2], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[3], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[4], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[5], corr[0]))

# Vary sparsity_2, medium signal
print(power(signal[2], sparsity_1[2], sparsity_2[0], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[2], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[3], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[4], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[5], corr[0]))

# Vary sparsity_2, high signal
print(power(signal[3], sparsity_1[2], sparsity_2[0], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[2], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[3], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[4], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[5], corr[0]))

# Vary corr
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[1]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[2]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[3]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[4]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[5]))