# Import functions
from tests_main import *
from tests_support import *
from utils import generate_penalty

# Import packages
import os
import numpy as np
import pandas as pd

"""
To regenerate the desired figure, uncomment the appropriate block. Sub-figures are referred to 
alphabetically from left to right and top to bottom in the corresponding figure.
"""

# Figure 2A, 14A, 15A, 16A
# num_points = 6
# n = np.full(num_points, 100)
# d = np.full(num_points, 50)
# k = np.full(num_points, 10)
# signal = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1])
# sparsity_1 = np.full(num_points, 10)
# sparsity_2 = np.full(num_points, 4)
# corr = np.full(num_points, 0.0)
# indep_var_values = signal
# indep_var_name = "Signal"

# Figure 2B, 15B, 16B
# num_points = 6
# n = np.full(num_points, 100)
# d = np.full(num_points, 50)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.4)
# sparsity_1 = [1, 2, 4, 6, 8, 10]
# sparsity_2 = np.full(num_points, 4)
# corr = np.full(num_points, 0.0)
# indep_var_values = sparsity_1
# indep_var_name = "Sparsity in null coefficients"

# Figure 2C, 15C, 16C
# num_points = 6
# n = np.full(num_points, 100)
# d = np.full(num_points, 50)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.4)
# sparsity_1 = np.full(num_points, 4)
# sparsity_2 = [0, 5, 10, 20, 30, 40]
# corr = np.full(num_points, 0.0)
# indep_var_values = sparsity_2
# indep_var_name = "Sparsity in non-null coefficients"

# Figure 2D, 3A, 3B, 3C, 15D, 15E, 16D
# num_points = 6
# n = np.full(num_points, 100)
# d = np.full(num_points, 50)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.4)
# sparsity_1 = np.full(num_points, 4)
# sparsity_2 = np.full(num_points, 4)
# corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
# indep_var_values = corr
# indep_var_name = "Correlation"

# Figure 13A
# num_points = 6
# n = np.full(num_points, 100)
# d = np.array([40, 50, 60, 70, 80, 90])
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.8)
# sparsity_1 = np.full(num_points, 10)
# sparsity_2 = np.array([3, 4, 5, 6, 7, 8])
# corr = np.full(num_points, 0.0)
# indep_var_values = d
# indep_var_name = "Dimension"

# 14B
# num_points = 6
# n = np.full(num_points, 100)
# d = np.full(num_points, 50)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.4)
# sparsity_1 = [1, 2, 4, 6, 8, 10]
# sparsity_2 = np.full(num_points, 4)
# corr = np.full(num_points, 0.9)
# indep_var_values = sparsity_1
# indep_var_name = "Sparsity in null coefficients"

# 14C
# num_points = 6
# n = np.full(num_points, 100)
# d = np.full(num_points, 50)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.4)
# sparsity_1 = np.full(num_points, 4)
# sparsity_2 = [0, 5, 10, 20, 30, 40]
# corr = np.full(num_points, 0.9)
# indep_var_values = sparsity_2
# indep_var_name = "Sparsity in non-null coefficients"

# Figure 12A
# num_points = 6
# n = np.full(num_points, 1000)
# d = np.full(num_points, 500)
# k = np.full(num_points, 10)
# signal = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25])
# sparsity_1 = np.full(num_points, 10)
# sparsity_2 = np.full(num_points, 49)
# corr = np.full(num_points, 0.0)
# indep_var_values = signal
# indep_var_name = "Signal"

# Figure 12B
# num_points = 6
# n = np.full(num_points, 1000)
# d = np.full(num_points, 500)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.1)
# sparsity_1 = [1, 2, 4, 6, 8, 10]
# sparsity_2 = np.full(num_points, 49)
# corr = np.full(num_points, 0.0)
# indep_var_values = sparsity_1
# indep_var_name = "Sparsity in null coefficients"

# Figure 12C
# num_points = 6
# n = np.full(num_points, 1000)
# d = np.full(num_points, 500)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.1)
# sparsity_1 = np.full(num_points, 4)
# sparsity_2 = [0, 50, 100, 200, 300, 490]
# corr = np.full(num_points, 0.0)
# indep_var_values = sparsity_2
# indep_var_name = "Sparsity in non-null coefficients"

# Figure 12D
# num_points = 6
# n = np.full(num_points, 1000)
# d = np.full(num_points, 500)
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.1)
# sparsity_1 = np.full(num_points, 4)
# sparsity_2 = np.full(num_points, 49)
# corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
# indep_var_values = corr
# indep_var_name = "Correlation"

# Figure 13B
# num_points = 6
# n = np.full(num_points, 1000)
# d = np.array([400, 500, 600, 700, 800, 990])
# k = np.full(num_points, 10)
# signal = np.full(num_points, 0.6)
# sparsity_1 = np.full(num_points, 10)
# sparsity_2 = np.array([39, 49, 59, 69, 79, 98])
# corr = np.full(num_points, 0.0)
# indep_var_values = d
# indep_var_name = "Dimension"

num_sims = 1000
num_jobs = 250
its_per_job = int(num_sims / num_jobs)

def generate_design(n, d, k, rho, orthog=False):
    cov = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    mean = np.zeros(d)
    X = np.random.multivariate_normal(mean, cov, size=n)
    if orthog:
        X1 = X[:, :k]
        X2 = X[:, k:]
        P1 = proj(X1)
        X2_orth = (np.eye(n) - P1) @ X2
        X[:, k:] = X2_orth
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X

def generate_beta(d, k, sig, spars_1, spars_2, anti=None):
    beta = np.zeros(d)
    if anti is True or anti is False:
        a = sig / np.sqrt(k)
        if anti:
            signs = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
            beta[:k] = a * signs
        else:
            beta[:k] = a
    else:
        sample_1 = np.random.choice(k, spars_1, replace=False)
        a = sig / np.sqrt(spars_1)
        beta[sample_1] = np.random.choice([-1, 1], size=spars_1) * a
    sample_2 = np.random.choice(np.arange(k, d), spars_2, replace=False)
    beta[sample_2] = np.random.normal(0, 1, size=spars_2)
    return beta

def run_tests(alpha=0.05):
    """
    Adjust the dataframe depending on which
    tests you want to run. 
    """
    results = pd.DataFrame({
        indep_var_name: indep_var_values,
        "F-test": 0,
        "Oracle": 0,
        r"Bonf-$\ell$": 0,
        "gLASSO": 0,
        "L-test": 0,
        "MC-free": 0
    })
    for i in range(its_per_job):
        for j in range(num_points):
            # Generate data
            X = generate_design(n[j], d[j], k[j], corr[j])
            beta = generate_beta(d[j], k[j], signal[j], sparsity_1[j], sparsity_2[j])
            y = np.random.multivariate_normal(X@beta, np.eye(n[j]))
            V = construct_V(X, k[j])
            P_k = proj(X[:,k[j]:])
            y_hat = P_k @ y
            sigma = np.linalg.norm(y - P_k@y)
            u = np.random.randn(n[j] - d[j] + k[j])
            u /= np.linalg.norm(u)
            y_tilde = y_hat + sigma * V @ u
            # Run tests
            """
            Adjust the tests being run by calling their corresponding functions
            and adding their p-values to the 'pvals' array.
            """
            F_pval = F_test(y, X, k[j])
            oracle_pval = oracle(y, X, k[j], beta[0:k[j]] / np.linalg.norm(beta[0:k[j]]))
            bonf_pval = bonf_ell(y, X, k[j])
            l, estimate = generate_penalty(y_tilde, X, k[j])
            glasso_pval = gLASSO(y, X, k[j], penalty = l)
            L_pval = L_test(y, X, k[j], penalty=l, point=estimate)
            MCfree_pval = MC_free(y, X, k[j], penalty = l, point = estimate)
            pvals = np.array([F_pval, oracle_pval, bonf_pval, glasso_pval, L_pval, MCfree_pval])
            results.iloc[j, 1:] += (pvals <= alpha).astype(int)
    return results

if __name__ == "__main__":
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    np.random.seed(task_id)

    """
    Default significance level is 0.05, but users can set to alternative desired level
    with run_tests(alpha=***).
    """
    test_rejections = run_tests()
    test_rejections.to_csv(f"rejections_{task_id}.csv", index=False)