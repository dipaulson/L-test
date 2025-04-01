# Import packages
import os 
import numpy as np
import random
import math
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LassoCV, ElasticNetCV, RidgeCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import t
from scipy.integrate import dblquad
from scipy.linalg import sqrtm
from scipy.special import gammaln
from celer import GroupLasso, GroupLassoCV
from asgl import Regressor

"""
Parameter specification
"""

MC = 200 # number of L-test Monte Carlo sims

"""
Collection of helper functions required for the implementation of 
all of our tests.
"""

# Groupings of indices for GroupLasso
def gen_groups(params):
    [n, d, k] = params
    first_sublist = list(range(0, k))
    remaining_sublists = [[i] for i in range(k, d)]
    groups = [first_sublist] + remaining_sublists
    return groups

# Projection matrix for input columnspace
def proj(X):
    return X@np.linalg.inv(X.T@X)@X.T

# OLS estimate
def OLS(y, X):
    MP = (np.linalg.inv(X.T@X))@X.T
    return MP@y

# Matrix V from Lemma 2
def construct_V(X, params):
    [n, d, k] = params
    I = np.identity(n)
    V_k = np.zeros((n, k))
    for i in range(k):
        V_k[:,i] = (I - proj(X[:,(i+1):]))@X[:,i] / np.linalg.norm((I - proj(X[:,(i+1):]))@X[:,i])
    U, S, VT = np.linalg.svd(X)
    V_remainder = U[:,d:]
    V = np.hstack((V_k, V_remainder))
    return V

# Sample from y | S^{1:k} under H_{1:k} using Lemma 2
def conditional_sample(y, X, V, params):
    [n, d, k] = params
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm(y - P_k@y)
    # Sample from unit sphere
    u = np.random.multivariate_normal(np.zeros(n - d + k), np.identity(n - d + k))
    u = u / np.linalg.norm(u)
    return P_k@y + sigma*(V@u)

# Generate L-test penalty parameter
def generate_penalty(y, X, V, params):
    groups = gen_groups(params)
    model = GroupLassoCV(groups, fit_intercept=False, cv=10)
    y_tilde = conditional_sample(y, X, V, params)
    model.fit(X, y_tilde)                                            
    l = model.alpha_
    return l

"""
Code to run oracle test given the data (y, X), the direction of B_{1:k}, and the
number of coefficients to be tested k (wlog the first k).
"""

def oracle_stat(y, X, beta, params):
    [n, d, k] = params
    X_1 = X[:,0:k]@beta[0:k] / np.linalg.norm(beta[0:k])
    X_1 = X_1.reshape(n, 1)
    X_new = np.concatenate((X_1, X[:,k:]), axis = 1)
    P_1 = proj(X_new[:,1:])
    P = proj(X_new)
    I = np.identity(n)
    B_1 = (X_1.T@(I - P_1)@y) / (np.linalg.norm((I - P_1)@X_1)**2)
    sigma = np.linalg.norm((I - P)@y) / np.sqrt(n-(d-k+1))
    T = B_1 / (sigma * np.sqrt(np.linalg.inv(X_new.T@X_new)[0, 0]))
    return T.item()

# Oracle-test p-value
def oracle_pval(y, X, beta, k):
    n, d = X.shape
    params = [n, d, k]
    obs = oracle_stat(y, X, beta, params)
    df = n-d+k-1
    return (1 - t.cdf(obs, df))

"""
Code to run the L-test given the data (y, X) and the number of coefficients 
to be tested k (wlog the first k).
"""

# L-test p-value
def L_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    l = generate_penalty(y, X, V, params)
    groups = gen_groups(params)
    model = GroupLasso(groups, alpha=l, fit_intercept=False)
    model.fit(X, y)
    beta_lasso = model.coef_
    V_k = V[:,0:k]
    M = V_k.T@X[:,0:k]
    obs = np.linalg.norm(M@beta_lasso[0:k])
    samples = np.zeros(MC)
    for i in range(MC):
        y_tilde = conditional_sample(y, X, V, params)
        model = GroupLasso(groups, alpha=l, fit_intercept=False)
        model.fit(X, y_tilde)
        beta_lasso = model.coef_
        samples[i] = float(np.linalg.norm(M@beta_lasso[0:k])>=obs)
    return (1+ sum(samples)) / (MC + 1)

"""
Below is the code to run the recentered F-test given the data (y, X), the recentering 
vector gamma, and the number of coefficients to be tested k (wlog the first k).
"""

# Recentering stat from Equation (3.5)
def gamma_1(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    l = generate_penalty(y, X, V, params)
    I = np.identity(n)
    P_k = proj(X[:,k:])
    S = X[:,0:k].T@(I-P_k)@X[:,0:k]
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y)
    gamma = np.linalg.inv(S)@X[:,0:k].T@(P_k@y-X[:,k:]@model.coef_)
    return gamma

# Group lasso estimate for linear model in S^{1:k}
def gamma_2(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    groups = gen_groups(params)
    W = sqrtm(X[:,k:].T@X[:,k:])
    U = X[:,k:]@np.linalg.inv(W)
    z = U.T@y
    X_z = np.concatenate((U.T@X[:,0:k], W), axis=1)
    model = GroupLassoCV(groups, fit_intercept=False, cv=10)
    model.fit(X_z, z)
    beta_est = model.coef_
    gamma = beta_est[0:k]
    return gamma

# Normalization constant for recentered F-test stat density
def norm_const(params):
    [n, d, k] = params
    numerator_log = gammaln(k / 2) + gammaln((n - d + k) / 2)
    denominator_log = (math.log(math.sqrt(math.pi))
                        + gammaln((k - 1) / 2)
                        + gammaln(k / 2)
                        + gammaln((n - d) / 2))
    result_log = numerator_log - denominator_log
    result = math.exp(result_log)
    return result

# Recentered F-test p-value
def R_pval(y, X, gamma, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    # Calc observed value
    V_k = V[:,0:k]
    M = V_k.T@X[:,0:k]
    beta_OLS = OLS(y, X)
    obs = np.linalg.norm(M@(beta_OLS[0:k] + gamma))**2
    # Convert to u-space
    I = np.identity(n)
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm((I-P_k)@y)
    obs_u = obs / sigma**2
    recenter = -(1/sigma)*(V_k.T@X[:,0:k])@gamma
    norm = np.linalg.norm(recenter)
    # Calc normalization constant
    D = norm_const(params)
    # Compute p-value
    h = lambda y, x: (1-y)**((n-d-2)/2) * (y - ((y + norm**2 - x) / (2*norm))**2)**((k-3)/2)
    if (obs_u < (norm-1)**2):
        result_1, error_1 = dblquad(h, obs_u, (norm-1)**2, lambda x: norm**2+x-2*norm*math.sqrt(x), lambda x: norm**2+x+2*norm*math.sqrt(x))
        result_2, error_2 = dblquad(h, (norm-1)**2, (norm+1)**2, lambda x: norm**2+x-2*norm*math.sqrt(x), lambda x: 1)
        return 0.5 * (D / norm) * (result_1 + result_2)
    else:
        result, error = dblquad(h, obs_u, (norm+1)**2, lambda x: norm**2+x-2*norm*math.sqrt(x), lambda x: 1)
        return 0.5 * (D / norm) * result

"""
Below are L-test modifications that use alternative LASSO-based estimates to replace
the OLS estimate in the conditional F-test statistic in Eq (2.3).
"""

# Standard LASSO test p-value
def lasso_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    model = LassoCV(fit_intercept=False, cv=10)
    y_tilde = conditional_sample(y, X, V, params)
    model.fit(X, y_tilde)                                             
    l = model.alpha_
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X, y)
    beta_lasso = model.coef_
    V_k = V[:,0:k]
    M = V_k.T@X[:,0:k]
    obs = np.linalg.norm(M@beta_lasso[0:k])
    samples = np.zeros(MC)
    for i in range(MC):
        y_tilde = conditional_sample(y, X, V, params)
        model = Lasso(alpha=l, fit_intercept=False)
        model.fit(X, y_tilde)
        beta_lasso = model.coef_
        samples[i] = float(np.linalg.norm(M@beta_lasso[0:k])>=obs)
    return (1 + sum(samples)) / (MC + 1)

# Elastic net test p-value
def elastic_net_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False, cv=10)
    y_tilde = conditional_sample(y, X, V, params)
    model.fit(X, y_tilde)                                             
    a = model.alpha_
    l1 = model.l1_ratio_
    model = ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=False)
    model.fit(X, y)
    beta_lasso = model.coef_
    V_k = V[:,0:k]
    M = V_k.T@X[:,0:k]
    obs = np.linalg.norm(M@beta_lasso[0:k])
    samples = np.zeros(MC)
    for i in range(MC):
        y_tilde = conditional_sample(y, X, V, params)
        model = ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=False)
        model.fit(X, y_tilde)
        beta_lasso = model.coef_
        samples[i] = float(np.linalg.norm(M@beta_lasso[0:k])>=obs)
    return (1 + sum(samples)) / (MC + 1)


# Sparse group LASSO test p-value
def sparse_g_lasso_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    group_index = np.zeros(d) 
    group_index[k:] = np.arange(1, d-k+1)
    # Custom CV step following recommendation of asgl package: https://github.com/alvaromc317/asgl/blob/9631bcdb55d0245274fca4d80d7e7f064965c9e9/user_guide.ipynb
    y_tilde = conditional_sample(y, X, V, params)
    model = Regressor(model='lm', penalization='sgl', fit_intercept=False)
    param_grid = {'lambda1': [1e-4, 1e-3, 1e-2, 1e-1, 1], 'alpha': [.1, .5, .7, .9, .95, .99, 1]}
    gscv = GridSearchCV(model, param_grid, scoring='neg_median_absolute_error', cv=10)
    gscv.fit(X, y_tilde, **{'group_index': group_index})
    l1 = gscv.best_params_['lambda1']
    a = gscv.best_params_['alpha']
    # Fit the model
    model = Regressor(model='lm', penalization='sgl', fit_intercept=False, lambda1=l1, alpha=a)
    model.fit(X, y, group_index=group_index)
    beta_lasso = model.coef_
    V_k = V[:,0:k]
    M = V_k.T@X[:,0:k]
    obs = np.linalg.norm(M@beta_lasso[0:k])
    samples = np.zeros(MC)
    for i in range(MC):
        y_tilde = conditional_sample(y, X, V, params)
        model = Regressor(model='lm', penalization='sgl', fit_intercept=False, lambda1=l1, alpha=a)
        model.fit(X, y_tilde, group_index=group_index)
        beta_lasso = model.coef_
        samples[i] = float(np.linalg.norm(M@beta_lasso[0:k])>=obs)
    return (1 + sum(samples)) / (MC + 1)