# Import packages
from tests import *
import numpy as np
import random
import math
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import f, t
from scipy.integrate import dblquad
from scipy.linalg import sqrtm
from scipy.special import gammaln
from celer import GroupLasso, GroupLassoCV
import time

def generate_design(rho, params):
    [n, d, k] = params
    # Construct covariance matrix
    cov = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    # Construct mean matrix
    mean = np.zeros(d)
    # Generate n samples from d-dim multivariate normal distribution
    X = np.random.multivariate_normal(mean, cov, size=n)
    # Standardize the columns of X
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def generate_beta(sig, spars_1, spars_2, params):
    [n, d, k] = params
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

# Analytic recentered F-test p-value
def R_pval_analytic(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    l = generate_penalty(y, X, V, params)
    # Calc observed unit vector
    I = np.identity(n)
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm((I-P_k)@y)
    V_k = V[:,0:k]
    u_obs = (1/sigma) * V_k.T@y
    # Calc recentering
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y)
    recenter = -(1/sigma)*np.linalg.inv(X[:,0:k].T@V_k)@X[:,0:k].T@(P_k@y-X[:,k:]@model.coef_)
    norm = np.linalg.norm(recenter)
    # Observed statistic
    obs_u = np.linalg.norm(u_obs - recenter)**2
    # Compute normalization constant
    C = norm_const(n, d, k)
    # Compute p-value
    h = lambda y, x: (1-y)**((n-d-2)/2) * (y - ((y + norm**2 - x) / (2*norm))**2)**((k-3)/2)
    if (obs_u < (norm-1)**2):
        result_1, error_1 = dblquad(h, obs_u, (norm-1)**2, lambda x: norm**2+x-2*norm*math.sqrt(x), lambda x: norm**2+x+2*norm*math.sqrt(x))
        result_2, error_2 = dblquad(h, (norm-1)**2, (norm+1)**2, lambda x: norm**2+x-2*norm*math.sqrt(x), lambda x: 1)
        return 0.5 * (C / norm) * (result_1 + result_2)
    else:
        result, error = dblquad(h, obs_u, (norm+1)**2, lambda x: norm**2+x-2*norm*math.sqrt(x), lambda x: 1)
        return 0.5 * (C / norm) * result

# Recentered F-test p-value w/ MC
def R_pval_MC(y, X, k, MC):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    l = generate_penalty(y, X, V, params)
    # Calc observed unit vector
    I = np.identity(n)
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm((I-P_k)@y)
    V_k = V[:,0:k]
    u_obs = (1/sigma) * V_k.T@y
    # Calc recentering
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y)
    recenter = -(1/sigma)*np.linalg.inv(X[:,0:k].T@V_k)@X[:,0:k].T@(P_k@y-X[:,k:]@model.coef_)
    # Observed statistic
    obs_u = np.linalg.norm(u_obs - recenter)**2
    # MC p-value
    samples = np.zeros(MC)
    for i in range(MC):
        u_sample = np.random.multivariate_normal(np.zeros(n - d + k), np.identity(n - d + k))
        u_sample = (u_sample / np.linalg.norm(u_sample))[0:k]
        samples[i] = float(np.linalg.norm(u_sample - recenter)**2>=obs_u)
    return (1 + sum(samples)) / (MC + 1)

def time_tests(model_size, rho, sig, spars_1, spars_2, MC):
    n=model_size[0]
    k=model_size[2]
    R_pvalues_MC = np.zeros(N)
    R_pvalues = np.zeros(N)
    # L_pvalues = np.zeros(N)
    R_times_MC = np.zeros(N)
    R_times = np.zeros(N)
    # L_times = np.zeros(N)
    for i in range(100):
        # Set seeds
        random.seed(i)
        np.random.seed(i)
        # Generate model
        X = generate_design(rho, model_size)
        V = construct_V(X, model_size)
        beta = generate_beta(sig, spars_1, spars_2, model_size)
        y = np.random.multivariate_normal(X@beta, np.identity(n))
        # Time p-value computation
        # Monte Carlo recentered F-test
        start_R_MC = time.time()
        p_R_MC = R_pval_MC(y, X, k, MC)
        end_R_MC = time.time()
        R_times_MC[i] = end_R_MC - start_R_MC
        R_pvalues_MC[i] = p_R_MC
        # Analytic recentered F-test
        start_R = time.time()
        p_R = R_pval_analytic(y, X, k)
        end_R = time.time()
        R_times[i] = end_R - start_R
        R_pvalues[i] = p_R
        """
        # L-test
        start_L = time.time()
        p_L = L_pval(y, X, k)
        end_L = time.time()
        L_times[i] = end_L - start_L
        L_pvalues[i] = p_L
        """
    print(f"MC recentered F-test avg p-value: {np.mean(R_pvalues_MC)}")
    print(f"Avg time: {np.mean(R_times_MC)}")
    print(f"Analytic recentered F-test avg p-value: {np.mean(R_pvalues)}")
    print(f"Avg time: {np.mean(R_times)}")
    """
    print(f"L-test avg p-value: {np.mean(L_pvalues)}")
    print(f"Avg time: {np.mean(L_times)}")
    """

"""
# Table 4.2.1: Increase Monte Carlo sims
print(time_tests([100, 50, 10], 0, 1, 10, 5, 200))
print(time_tests([100, 50, 10], 0, 1, 10, 5, 2000))
print(time_tests([100, 50, 10], 0, 1, 10, 5, 20000))    
print(time_tests([100, 50, 10], 0, 1, 10, 5, 200000))  
"""

"""
# Table 4.2.2: Increase model size
print(time_tests([100, 50, 10], 0, 0.1, 10, 5, 200))
print(time_tests([1000, 500, 10], 0, 0.1, 10, 5, 200)) 
"""
