# Import packages
import os 
import numpy as np
import random
import math
from celer import GroupLasso, GroupLassoCV

# Set seeds
task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
random.seed(task_id)
np.random.seed(task_id)

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

def proj(X):
    return X@np.linalg.inv(X.T@X)@X.T

# Matrix V from Lemma 1
def construct_V(X):
    I = np.identity(n)
    V_k = np.zeros((n, k))
    for i in range(k):
        V_k[:,i] = (I - proj(X[:,(i+1):]))@X[:,i] / np.linalg.norm((I - proj(X[:,(i+1):]))@X[:,i])
    U, S, VT = np.linalg.svd(X)
    V_remainder = U[:,d:]
    V = np.hstack((V_k, V_remainder))
    return V

def OLS(y, X):
    # Moore-Penrose pseudoinverse
    MP = (np.linalg.inv(X.T@X))@X.T
    return MP@y

def conditional_sample(y, X, V):
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm(y - P_k@y)
    # Sample from unit sphere
    u = np.random.multivariate_normal(np.zeros(n - d + k), np.identity(n - d + k))
    u = u / np.linalg.norm(u)
    return P_k@y + sigma*(V@u)

def L_pval(y, X, V, l):
    model = GroupLasso(groups, alpha=l, fit_intercept=False)
    model.fit(X, y)
    beta_lasso = model.coef_
    V_k = V[:,0:k]
    M = V_k@V_k.T@X[:,0:k]
    obs = np.linalg.norm(M@beta_lasso[0:k])**2
    samples = np.zeros(MC)
    for i in range(MC):
        y_tilde = conditional_sample(y, X, V)
        model = GroupLasso(groups, alpha=l, fit_intercept=False)
        model.fit(X, y_tilde)
        beta_lasso = model.coef_
        samples[i] = float(np.linalg.norm(M@beta_lasso[0:k])**2>=obs)
    return (1+ sum(samples)) / (MC + 1)
 
def generate_penalties(y, X, V):
    model_1 = GroupLassoCV(groups, fit_intercept=False, cv=10)
    y_tilde = conditional_sample(y, X, V)
    model_1.fit(X, y_tilde)                                            
    lambda_1 = model_1.alpha_
    model_2 = GroupLassoCV(groups, fit_intercept=False, cv=10)
    P_k = proj(X[:,k:])
    y_hat = P_k@y
    model_2.fit(X, y_hat)                                          
    lambda_2 = model_2.alpha_
    return lambda_1, lambda_2

def new_test(y, X, V):
    model = GroupLassoCV(groups, fit_intercept=False, cv=10)
    model.fit(X, y)
    l = model.alpha_
    beta_lasso = model.coef_
    V_k = V[:,0:k]
    M = V_k@V_k.T@X[:,0:k]
    obs = np.linalg.norm(M@beta_lasso[0:k])**2
    samples = np.zeros(MC)
    for i in range(MC):
        y_tilde = conditional_sample(y, X, V)
        model = GroupLasso(groups, alpha=l, fit_intercept=False)
        model.fit(X, y_tilde)
        beta_lasso = model.coef_
        samples[i] = float(np.linalg.norm(M@beta_lasso[0:k])**2>=obs)
    return (1+ sum(samples)) / (MC + 1)

def power(sig, spars_1, spars_2, rho):
    rejects_L_1 = 0
    rejects_L_2 = 0
    rejects_L_3 = 0
    for i in range(1):
        X = generate_design(rho)
        V = construct_V(X)
        beta = generate_beta(sig, spars_1, spars_2)
        y = np.random.multivariate_normal(X@beta, (var ** 2) * np.identity(n))
        lam_1, lam_2 = generate_penalties(y, X, V)
        # Calc p-values
        p_L_1 = L_pval(y, X, V, lam_1)
        p_L_2 = L_pval(y, X, V, lam_2)
        p_L_3 = new_test(y, X, V)
        # Reject/retain
        if (p_L_1 <= alpha):
            rejects_L_1 += 1
        if (p_L_2 <= alpha):
            rejects_L_2 += 1
        if (p_L_3 <= alpha):
            rejects_L_3 += 1
    return rejects_L_1, rejects_L_2, rejects_L_3

# Low-dimensional model setting:

n = 100 # num rows
d = 50 # num cols
var = 1 # variance
k = 10 # number of null coefs
alpha = 0.05 # significance level
MC = 200 # number of sims
first_sublist = list(range(0, k))
remaining_sublists = [[i] for i in range(k, d)]
groups = [first_sublist] + remaining_sublists

signal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
sparsity_1 = np.array([1, 2, 4, 6, 8, 10])
sparsity_2 = np.array([0, 5, 10, 20, 30, 40])
corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])

"""
# High-dimensional model setting:

n = 100 # num rows
d = 90 # num cols
var = 1 # variance
k = 10 # number of null coefs
alpha = 0.05 # significance level
MC = 200 # number of sims
first_sublist = list(range(0, k))
remaining_sublists = [[i] for i in range(k, d)]
groups = [first_sublist] + remaining_sublists

signal = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
sparsity_1 = np.array([1, 2, 4, 6, 8, 10])
sparsity_2 = np.array([0, 10, 20, 40, 60, 80])
corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
"""

# Fix sparsity_1 = 10, sparsity_2 = 5, rho = 0
# Vary signal
print(power(signal[0], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[2], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[4], sparsity_1[5], sparsity_2[1], corr[0]))
print(power(signal[5], sparsity_1[5], sparsity_2[1], corr[0]))

# Fix sparsity_1 = 10, sparsity_2 = 5, rho = 0.9
# Vary signal
print(power(signal[0], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[1], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[2], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[3], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[4], sparsity_1[5], sparsity_2[1], corr[5]))
print(power(signal[5], sparsity_1[5], sparsity_2[1], corr[5]))

# Fix sparsity_2 = 5, rho = 0, sig = 0.2
# Vary sparsity_1
print(power(signal[1], sparsity_1[0], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[1], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[3], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[4], sparsity_2[1], corr[0]))
#print(power(signal[1], sparsity_1[5], sparsity_2[1], corr[0]))

# Fix sparsity_2 = 5, rho = 0, sig = 0.6
# Vary sparsity_1
print(power(signal[3], sparsity_1[0], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[1], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[3], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[4], sparsity_2[1], corr[0]))
#print(power(signal[3], sparsity_1[5], sparsity_2[1], corr[0]))

# Fix sparsity_1 = 4, signal = 0.2, rho = 0
# Vary sparsity_2
print(power(signal[1], sparsity_1[2], sparsity_2[0], corr[0]))
#print(power(signal[1], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[2], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[3], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[4], corr[0]))
print(power(signal[1], sparsity_1[2], sparsity_2[5], corr[0]))

# Fix sparsity_1 = 4, signal = 0.4, rho = 0
# Vary sparsity_2
print(power(signal[2], sparsity_1[2], sparsity_2[0], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[2], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[3], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[4], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[5], corr[0]))

# Fix sparsity_1 = 4, signal = 0.6, rho = 0
# Vary sparsity_2
print(power(signal[3], sparsity_1[2], sparsity_2[0], corr[0]))
#print(power(signal[3], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[2], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[3], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[4], corr[0]))
print(power(signal[3], sparsity_1[2], sparsity_2[5], corr[0]))

# Fix sparsity_1 = 4, sparsity_2 = 5, signal = 0.4
# Vary rho
#print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[0]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[1]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[2]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[3]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[4]))
print(power(signal[2], sparsity_1[2], sparsity_2[1], corr[5]))