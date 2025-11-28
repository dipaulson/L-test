# Import functions
from tests_main import *
from tests_support import *
from compare_tests import generate_design

# Import packages
import os
import pickle
import numpy as np
import pandas as pd

num_sims = 1000
num_jobs = 250
its_per_job = int(num_sims / num_jobs)

num_points = 8
num_settings = 6
n = np.array([10, 16, 20, 26, 30, 36, 40, 46])
d = (n / 2).astype(int)
k = np.full(num_points, 3)
violations = ["heavy_tail", "skewed", "hetero", "non_linear"]
dfs = np.array([2, 5, 10, 15, 20, 30])
alphas = np.array([1, 2, 4, 6, 8, 10])
etas = np.array([0.01, 0.25, 0.5, 1, 4, 8])
deltas = np.array([0.3, 0.5, 1, 2, 3, 4])

def run_tests(alpha=0.05):
    outputs = {}
    for vtype in violations:
        for s in range(num_settings):
            results = pd.DataFrame({
                "Sample size": n,
                "F-test": 0,
                "L-test": 0
            })
            for i in range(its_per_job):
                for j in range(num_points):
                    # Generate data
                    X = generate_design(n[j], d[j], k[j], 0.0) # change rho = 0.5 for high correlation results
                    beta = np.zeros(d[j])
                    idx = np.random.choice(np.arange(k[j], d[j]))
                    beta[idx] = np.random.normal(0, 1)
                    if (vtype == "non_linear"):
                        delta = deltas[s]
                        X = np.sign(X) * (np.abs(X) ** delta)
                        errors = np.random.normal(0, 1, size=n[j])
                    elif (vtype == "heavy_tail"):
                        df = dfs[s]
                        errors = np.random.standard_t(df, size=n[j])
                        if (df>2):
                            errors /= np.sqrt(df / (df - 2))
                    elif (vtype == "skewed"):
                        a = alphas[s]
                        errors = (np.random.gamma(a, 1, size=n[j]) - a) / np.sqrt(a)
                    else:
                        eta = etas[s]
                        row_means = np.mean(X, axis=1)
                        mask = row_means <= 0
                        errors = np.empty(n[j])
                        errors[mask]  = np.random.normal(0, 1,  size=mask.sum())
                        errors[~mask] = np.random.normal(0, eta, size=(~mask).sum())
                    y = X@beta + errors
                    # Run tests
                    F_pval = F_test(y, X, k[j])
                    L_pval = L_test(y, X, k[j])
                    pvals = np.array([F_pval, L_pval])
                    results.iloc[j, 1:] += (pvals <= alpha).astype(int)
            key = f"{vtype}_{s}"
            outputs[key] = results
    return outputs

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
np.random.seed(task_id)

"""
Default significance level is 0.05, but users can set to alternative desired level
with run_tests(alpha=***).
"""
test_rejections = run_tests()
with open(f"rejections_{task_id}.pkl", "wb") as f:
    pickle.dump(test_rejections, f)
