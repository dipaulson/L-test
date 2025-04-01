# Import packages
import os
from tests import *
import numpy as np
import pandas as pd
import random
import math
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import f, beta
from scipy.integrate import dblquad
from scipy.special import gammaln
from celer import GroupLassoCV, GroupLasso
from collections import defaultdict

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

# Recentered F-test p-value for special case k = 1
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def u_cdf(x, params):
    [n, d, k] = params
    a=0.5
    b=(n-d)/2
    return 0.5+sign(x)*0.5*beta.cdf(x**2, a, b)

def R_pval_1D(y, X, gamma, k):
    n, d = X.shape
    params = [n, d, k]
    V = construct_V(X, params)
    # Calc observed value
    V_k = V[:,0:k]
    M = V_k@V_k.T@X[:,0:k]
    beta_OLS = OLS(y, X)
    obs = np.linalg.norm(M@(beta_OLS[0:k] + gamma))
    # Convert to u-space
    I = np.identity(n)
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm((I-P_k)@y)
    obs_u = obs / sigma
    recenter = -(1/sigma)*(V_k.T@X[:,0:k])@gamma
    return (1 - u_cdf(recenter+obs_u, params) + u_cdf(recenter-obs_u, params)).item()

# Function to generate p-values
def HIV_pvals(X, y):
    # Log-transform the drug resistance measurements
    y = np.log(y)

    # Remove patients with missing measurements
    missing = np.isnan(y)
    y = y[~missing]
    X = X.loc[~missing, :]

    # Remove predictors that appear less than 3 times
    X = X.loc[:, (X.sum(axis=0) >= 3)]

    # Remove duplicate predictors
    corr_matrix = X.corr()
    unique_cols = np.where((np.abs(corr_matrix - 1) < 1e-4).sum(axis=0) == 1)[0]
    X = X.iloc[:, unique_cols]

    # Get mutation names
    mutations = X.columns
    
    # Identify groups of mutations for testing
    groups = defaultdict(list)
    for index, code in enumerate(mutations):
        num = code.split('.')[1][1:]
        groups[num].append(index)

    # Create a list of tuples from the grouped indices
    null_groups = list(groups.values())
    
    # Convert dataframes to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()
    
    # Normalize design matrix
    col_norms = np.linalg.norm(X, axis=0)
    X = X / col_norms

    # Get stats
    (n, d) = X.shape

    # Initialize p-values
    p_vals = np.zeros((len(null_groups), 4))

    # Perform F- and L-tests
    for i, indices in enumerate(null_groups, 1):
        k = len(indices)
        params = [n, d, k]
        null_hyp_cols = X[:,indices]
        remaining_cols = np.delete(X, indices, axis=1)
        X_adj = np.hstack((null_hyp_cols, remaining_cols))
        p_F = F_pval(y, X_adj, k)
        p_L = L_pval(y, X_adj, k)
        if (k == 1):
            p_R = R_pval_1D(y, X_adj, gamma_1(y, X_adj, k), k)
        else:
            p_R = R_pval(y, X_adj, gamma_1(y, X_adj, k), k)
        p_vals[i-1] = [p_F, p_L, p_R, k]
    return p_vals

# Get task ID
task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

# Set seeds
random.seed(task_id)
np.random.seed(task_id)

# Get data
if (1<=task_id<=7):
    df_1 = pd.read_csv("data_part_1.csv")
    drug_names_1 = df_1.columns[:7]
    X = df_1.drop(drug_names_1, axis=1)
    y = df_1[drug_names_1[task_id-1]]
elif (8<=task_id<=13):
    df_2 = pd.read_csv("data_part_2.csv")
    drug_names_2 = df_2.columns[:6]
    X = df_2.drop(drug_names_2, axis=1)
    y = df_2[drug_names_2[task_id-8]]
else:
    df_3 = pd.read_csv("data_part_3.csv")
    drug_names_3 = df_3.columns[:3]
    X = df_3.drop(drug_names_3, axis=1)
    y = df_3[drug_names_3[task_id-14]]

# Compute p-values and save
p_vals = HIV_pvals(X, y)
np.savetxt(f'p_vals_reg_{task_id}.csv', p_vals, delimiter=',', fmt='%.4f')