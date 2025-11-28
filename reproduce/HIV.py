# Import functions
from tests_main import *
from tests_support import F_test
from utils import *
from collections import defaultdict

# Import packages
import os
import numpy as np
import pandas as pd
import time

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

    # Initialize p-values and times; adjust sizing depending on number of tests
    p_vals = np.zeros((len(null_groups), 5))
    times = np.zeros((len(null_groups), 4))

    # Perform tests
    for i, indices in enumerate(null_groups, 1):
        k = len(indices)
        null_hyp_cols = X[:,indices]
        remaining_cols = np.delete(X, indices, axis=1)
        X_adj = np.hstack((null_hyp_cols, remaining_cols))
        V = construct_V(X_adj, k)
        P_k = proj(X_adj[:,k:])
        y_hat = P_k @ y
        sigma = np.linalg.norm(y - P_k@y)
        u = np.random.randn(n - d + k)
        u /= np.linalg.norm(u)
        y_tilde = y_hat + sigma * V @ u
        l, estimate = generate_penalty(y_tilde, X_adj, k)
        # Adjust which tests are run depending on single or multiple testing analysis
        t1 = time.time()
        F_pval = F_test(y, X_adj, k)
        t2 = time.time()
        glasso_pval = gLASSO(y, X_adj, k, penalty = l)
        t3 = time.time()
        L_pval = L_test(y, X_adj, k, penalty = l, point = estimate, MC=5000) 
        t4 = time.time()
        MCfree_pval = MC_free(y, X_adj, k, penalty = l, point = estimate)
        t5 = time.time()
        p_vals[i-1] = [F_pval, glasso_pval, L_pval, MCfree_pval, k]
        times[i-1] = [t2-t1, t3-t2, t4-t3, t5-t4]
    return p_vals, times

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
np.random.seed(task_id)

reg_num = ((task_id - 1) % 16) + 1
iteration = ((task_id - 1) // 16) + 1

# Get data
if (1<=reg_num<=7):
    df_1 = pd.read_csv("data_part_1.csv")
    drug_names_1 = df_1.columns[:7]
    X = df_1.drop(drug_names_1, axis=1)
    y = df_1[drug_names_1[reg_num-1]]
elif (8<=reg_num<=13):
    df_2 = pd.read_csv("data_part_2.csv")
    drug_names_2 = df_2.columns[:6]
    X = df_2.drop(drug_names_2, axis=1)
    y = df_2[drug_names_2[reg_num-8]]
else:
    df_3 = pd.read_csv("data_part_3.csv")
    drug_names_3 = df_3.columns[:3]
    X = df_3.drop(drug_names_3, axis=1)
    y = df_3[drug_names_3[reg_num-14]]

# Compute p-values and save
p_vals, times = HIV_pvals(X, y)
np.savetxt(f'p_vals_reg_{reg_num}_it_{iteration}.csv', p_vals, delimiter=',', fmt='%.18e')
np.savetxt(f'times_reg_{reg_num}_it_{iteration}.csv', times, delimiter=',', fmt='%.18e')