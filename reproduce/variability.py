# Import functions
from tests_main import *
from utils import generate_penalty
from compare_tests import generate_design, generate_beta

# Import packages
import os
import numpy as np
import pandas as pd

n = 100
d = 50
k = 10
r = 1 # Change depending on how many replicates of penalty and point you want to average for the test
signals = np.array([0.1, 0.2, 0.4, 0.6, 0.8])
NUM_SIMS = np.array([2000, 2000, 20000, 20000, 200000])
num_jobs = 2000

m_outer = 10
m_inner = 100

def run_tests():
    all_pvals = np.zeros((len(signals) * m_outer, m_inner), dtype=float)
    for s, sig in enumerate(signals):
        tot_sims = NUM_SIMS[s]
        for i in range(m_outer):
            X = generate_design(n, d, k, 0.0)
            beta = generate_beta(d, k, sig, 10, 5)
            y = np.random.multivariate_normal(X@beta, np.eye(n))
            V = construct_V(X, k)
            P_k = proj(X[:,k:])
            y_hat = P_k @ y
            sigma = np.linalg.norm(y - P_k@y)
            row_pvals = np.empty(m_inner, dtype=float)
            for j in range(m_inner):
                lambdas = np.empty((0,), dtype=float)
                points = np.empty((0, d), dtype=float)
                for _ in range(r):
                    u = np.random.randn(n - d + k)
                    u /= np.linalg.norm(u)
                    y_tilde = y_hat + sigma * V @ u
                    l, estimate = generate_penalty(y_tilde, X, k)
                    lambdas = np.append(lambdas, l)
                    points = np.vstack([points, estimate])
                lambda_exp_mean = np.exp(np.mean(np.log(lambdas)))
                point_mean = points.mean(axis=0)
                # Change depending on whether running L-test or recentered test
                L_pval = L_test(y, X, k, penalty = lambda_exp_mean, point = point_mean, MC=tot_sims)
                row_pvals[j] = L_pval
            row = s * m_outer + i
            all_pvals[row, :] = row_pvals
    index = [f"signal_{sig}_{i}" for sig in signals for i in range(m_outer)]
    cols = [f"run_{j+1}" for j in range(m_inner)]
    results = pd.DataFrame(all_pvals, index=index, columns=cols)
    return results

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
np.random.seed(task_id)

test_pvals = run_tests()
test_pvals.to_csv(f"pvals_rep_{task_id}.csv")
