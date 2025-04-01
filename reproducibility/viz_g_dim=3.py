# Import packages
import os
import numpy as np
import random
import math
from sklearn.linear_model import Lasso, LassoCV
from celer import GroupLasso, GroupLassoCV
import matplotlib.pyplot as plt
from itertools import product

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

def generate_penalty(y, X, V):
    model = GroupLassoCV(groups, fit_intercept=False, cv=10)
    y_tilde = conditional_sample(y, X, V)
    model.fit(X, y_tilde)                                            
    l = model.alpha_
    return l

def gen_model(sig, spars_1, spars_2, rho):
    X = generate_design(rho)
    V = construct_V(X)
    beta = generate_beta(sig, spars_1, spars_2)
    y = np.random.multivariate_normal(X@beta, (var ** 2) * np.identity(n))
    l = generate_penalty(y, X, V)
    I = np.identity(n)
    P_k = proj(X[:,k:])
    S = X[:,0:k].T@(I - P_k)@X[:,0:k]
    return y, X, S, l

def f_inv(y, y_hat, X, S, l, b):
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y - X[:,0:k]@b)
    lasso = model.coef_
    unit = b / np.linalg.norm(b)
    return -np.linalg.inv(S)@(X[:,0:k].T@(y_hat-X[:,0:k]@b-X[:,k:]@lasso)-n*l*unit)

# Set params
n = 100
d = 50 # set to 90 for high-dim setting
var = 1
k = 3
first_sublist = list(range(0, k))
remaining_sublists = [[i] for i in range(k, d)]
groups = [first_sublist] + remaining_sublists
sig = 0.4 # set to 0.75 for high-dim setting
spars_1 = k
spars_2 = 5 # set to 10 for high-dim setting
rho = 0
for i in range (5):
    random.seed(i)
    np.random.seed(i)
    y, X, S, l = gen_model(sig, spars_1, spars_2, rho)
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y)
    lasso = model.coef_
    P_k = proj(X[:,k:])
    y_hat = P_k@y
    beta_tilde = np.linalg.inv(S)@X[:,0:k].T@(y_hat-X[:,k:]@lasso)

    # Create grid
    axes = [np.linspace(-20, 20, 10) for _ in range(k)] # set to (-10, 10, 10) for high-dim setting
    grid = list(product(*axes))
    grid = [point for point in grid if not np.allclose(point, 0)]
    b_lambdas = np.array(grid)
    b_OLS = np.array([f_inv(y, y_hat, X, S, l, b) for b in b_lambdas])
    b_OLS += beta_tilde

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(b_OLS[:, 0], b_OLS[:, 1], b_OLS[:, 2], color='blue', label='Input', s=10)

    for p, q in zip(b_OLS, b_lambdas):
        ax.quiver(p[0], p[1], p[2], q[0]-p[0], q[1]-p[1], q[2]-p[2],
                arrow_length_ratio=0.2, color='black', alpha=0.8, linewidth=0.8, normalize=False)

    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-30, 30)

    ax.set_xticks(range(-30, 31, 10))
    ax.set_yticks(range(-30, 31, 10))
    ax.set_zticks(range(-30, 31, 10))
    ax.tick_params(labelsize=14)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig(f"Iteration_{i}.png", dpi=300, bbox_inches='tight')
    plt.close()