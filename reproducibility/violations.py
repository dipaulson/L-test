# Import packages
import os
from tests import *
import numpy as np
import random
import math
from scipy.stats import f, t
from celer import GroupLassoCV, GroupLasso

# Model params
num_rows = [10, 16, 20, 26, 30, 36, 40]

# Set seeds
task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
random.seed(task_id)
np.random.seed(task_id)

def generate_design(params):
    [n, d, k] = params
    # Construct covariance matrix
    cov = np.eye(d)
    # Construct mean matrix
    mean = np.zeros(d)
    # Generate n samples from d-dim mvn distribution
    X = np.random.multivariate_normal(mean, cov, size=n)
    # Standardize the columns of X
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def generate_beta(params):
    [n, d, k] = params
    # Select non-zero entries among remaining d-k
    sample_2 = random.sample(list(range(k, d)), 1)
    # Set value of non-zero entries
    A = 5
    # Initialize beta
    beta = np.zeros(d)
    for i in sample_2:
        sign = random.choice([-1, 1])
        beta[i] = sign * A
    return beta

# F-test implementation
def F_stat(y, X, V, params):
    [n, d, k] = params
    V_k = V[:,0:k]
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm(y - P_k@y)
    beta_OLS = OLS(y, X)
    norm_sq = np.linalg.norm(V_k.T@X[:,0:k]@beta_OLS[0:k])**2
    return (norm_sq/k)/((sigma**2 - norm_sq)/(n-d))

def F_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    obs = F_stat(y, X, V, params)
    dfn, dfd = k, (n - d)
    return (1 - f.cdf(obs, dfn, dfd))

# Setting 1: Heavy-tailed errors

dfs = [2, 5, 10, 15, 20, 30]

def type_1_error_tail(n, df):
    d = int(n/2)
    k = 3
    params = [n, d, k]
    rejects_F = 0
    rejects_L = 0
    X = generate_design(params)
    beta = generate_beta(params)
    errors = t.rvs(df, size=n)
    if (df>2):
        std = math.sqrt(df / (df-2))
        errors = errors / std
    y = X@beta + errors
    p_F = F_pval(y, X, k)
    p_L = L_pval(y, X, k)
    if (p_F <= 0.05):
        rejects_F += 1
    if (p_L <= 0.05):
        rejects_L += 1
    return rejects_F, rejects_L

for i in range(6):
    df = dfs[i]
    for j in range(7):
        n = num_rows[j]
        print(type_1_error_tail(n, df))

# Setting 2: Skewed errors

alphas = [1, 2, 4, 6, 8, 10]

def type_1_error_skew(n, alpha):
    d = int(n/2)
    k = 3
    params = [n, d, k]
    rejects_F = 0
    rejects_L = 0
    X = generate_design(params)
    beta = generate_beta(params)
    errors = np.random.gamma(alpha, 1, size=n)
    errors = errors - alpha
    errors = errors / math.sqrt(alpha)
    y = X@beta + errors
    p_F = F_pval(y, X, k)
    p_L = L_pval(y, X, k)
    if (p_F <= 0.05):
        rejects_F += 1
    if (p_L <= 0.05):
        rejects_L += 1
    return rejects_F, rejects_L

for i in range(6):
    alpha = alphas[i]
    for j in range(7):
        n = num_rows[j]
        print(type_1_error_skew(n, alpha))

# Setting 3: Heteroskedastic errors

etas = [0.01, 0.25, 0.5, 1, 4, 8]

def type_1_error_sked(n, eta):
    d = int(n/2)
    k = 3
    params = [n, d, k]
    rejects_F = 0
    rejects_L = 0
    X = generate_design(params)
    row_means = np.mean(X, axis=1)
    row_med = np.median(row_means)
    beta = generate_beta(params)
    errors = np.zeros(n)
    for i in range(n):
        if (row_means[i] <= row_med):
            errors[i] = np.random.normal(0, 1, size=1).item()
        else:
            errors[i] = np.random.normal(0, eta, size=1).item()
    y = X@beta + errors
    p_F = F_pval(y, X, k)
    p_L = L_pval(y, X, k)
    if (p_F <= 0.05):
        rejects_F += 1
    if (p_L <= 0.05):
        rejects_L += 1
    return rejects_F, rejects_L

for i in range(6):
    eta = etas[i]
    for j in range(7):
        n = num_rows[j]
        print(type_1_error_sked(n, eta))

# Setting 4: Model non-linearity

deltas = [0.3, 0.5, 1, 2, 3, 4]

def type_1_error_lin(n, delta):
    d = int(n/2)
    k = 3
    params = [n, d, k]
    rejects_F = 0
    rejects_L = 0
    X = generate_design(params)
    beta = generate_beta(params)
    errors = np.random.normal(0, 1, size=n)
    y = (np.abs(X) ** delta)@beta + errors
    p_F = F_pval(y, X, k)
    p_L = L_pval(y, X, k)
    if (p_F <= 0.05):
        rejects_F += 1
    if (p_L <= 0.05):
        rejects_L += 1
    return rejects_F, rejects_L

for i in range(6):
    delta = deltas[i]
    for j in range(7):
        n = num_rows[j]
        print(type_1_error_lin(n, delta))