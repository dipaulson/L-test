# Import packages
import numpy as np
import random
import math
from sklearn.linear_model import Lasso, LassoCV
from celer import GroupLasso, GroupLassoCV
import matplotlib.pyplot as plt

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

# Matrix V from Lemma 2
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

# Sample from y | S^{1:k} under H_{1:k} using Lemma 2
def conditional_sample(y, X, V):
    P_k = proj(X[:,k:])
    sigma = np.linalg.norm(y - P_k@y)
    # Sample from unit sphere
    u = np.random.multivariate_normal(np.zeros(n - d + k), np.identity(n - d + k))
    u = u / np.linalg.norm(u)
    return P_k@y + sigma*(V@u)

# Generate L-test penalty parameter
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
    return y, X, S, V, l

def f_inv(y, y_hat, X, S, V, l, b):
    V_k = V[:,0:k]
    b = np.linalg.inv(V_k.T@X[:,0:k])@b
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y - X[:,0:k]@b)
    lasso = model.coef_
    unit = b / np.linalg.norm(b)
    return -np.linalg.inv(S)@(X[:,0:k].T@(y_hat-X[:,0:k]@b-X[:,k:]@lasso)-n*l*unit)

# Set params
n = 100
d = 50 # set to 90 for high-dim setting
var = 1
k = 2
sig = 0.4 # set to 0.75 for high-dim setting
spars_1 = k
spars_2 = 5 # set to 10 for high-dim setting
rho = 0
first_sublist = list(range(0, k))
remaining_sublists = [[i] for i in range(k, d)]
groups = [first_sublist] + remaining_sublists

for i in range (5):
    random.seed(i)
    np.random.seed(i)
    y, X, S, V, l = gen_model(sig, spars_1, spars_2, rho)
    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X[:,k:], y)
    lasso = model.coef_
    P_k = proj(X[:,k:])
    y_hat = P_k@y
    beta_tilde = np.linalg.inv(S)@X[:,0:k].T@(y_hat-X[:,k:]@lasso)

    # Create grid
    x_range=(-20, 20) # set to (-10, 10) for high-dim setting
    y_range=(-20, 20) # set to (-10, 10) for high-dim setting
    grid_size=10
    xx = np.linspace(x_range[0], x_range[1], grid_size)
    yy = np.linspace(y_range[0], y_range[1], grid_size)
    XX, YY = np.meshgrid(xx, yy)
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)
    b_lambdas = np.array([p for p in points if not np.allclose(p, 0)])
    b_OLS = np.array([f_inv(y, y_hat, X, S, V, l, b) for b in b_lambdas])
    b_OLS += beta_tilde

    # Plot
    plt.figure(figsize=(8, 8))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.xticks(np.arange(-20, 21, 10))
    plt.yticks(np.arange(-20, 21, 10))
    plt.grid(True)
    plt.gca().set_aspect('equal')

    plt.scatter(b_OLS[:, 0], b_OLS[:, 1], color='blue', label='Input', s=10)

    X = b_OLS[:, 0]
    Y = b_OLS[:, 1]
    U = b_lambdas[:, 0] - X
    V = b_lambdas[:, 1] - Y

    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1,
            color='black', alpha=0.8, width=0.003, headwidth=4, headlength=6, headaxislength=5)

    plt.xlabel("X", fontsize=16)
    plt.ylabel("Y", fontsize=16)
    plt.tick_params(labelsize=14)
    plt.savefig(f"Iteration_{i}.png", dpi=300, bbox_inches='tight')
    plt.close()
