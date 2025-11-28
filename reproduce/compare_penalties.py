# Import functions
from tests_main import *
from utils import generate_penalty
from compare_tests import generate_design, generate_beta

# Import packages
import os
import numpy as np
import pandas as pd

"""
To regenerate the desired figure, uncomment the appropriate block.
"""

# Figure 7A, 9A
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

# Figure 7B, 9B
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

# Figure 7C, 9C
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

# Figure 7D, 9D
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

num_sims = 1000
num_jobs = 250
its_per_job = int(num_sims / num_jobs)

def run_tests(alpha=0.05):
    """
    Adjust the dataframe depending on which
    tests you want to run. 
    """
    results = pd.DataFrame({
        indep_var_name: indep_var_values,
        "L_min": 0,
        "L_1se": 0,
        "L_full": 0,
        "L_proj": 0
    })
    for i in range(its_per_job):
        for j in range(num_points):
            # Generate data
            X = generate_design(n[j], d[j], corr[j])
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
            l_1, estimate_1 = generate_penalty(y_tilde, X, k[j])
            l_2, estimate_2 = generate_penalty(y_tilde, X, k[j], rule="1se")
            l_3, estimate_3 = generate_penalty(y, X, k[j])
            l_4, estimate_4 = generate_penalty(y_hat, X, k[j])
            L_pval_min = L_test(y, X, k[j], penalty = l_1, point = estimate_1)
            L_pval_1se = L_test(y, X, k[j], penalty = l_2, point = estimate_2)
            L_pval_full = L_test(y, X, k[j], penalty = l_3, point = estimate_3)
            L_pval_proj = L_test(y, X, k[j], penalty = l_4, point = estimate_4)
            pvals = np.array([L_pval_min, L_pval_1se, L_pval_full, L_pval_proj])
            results.iloc[j, 1:] += (pvals <= alpha).astype(int)
    return results

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
np.random.seed(task_id)

"""
Default significance level is 0.05, but users can set to altnerative desired level
with run_tests(alpha=***).
"""
test_rejections = run_tests()
test_rejections.to_csv(f"rejections_{task_id}.csv", index=False)

