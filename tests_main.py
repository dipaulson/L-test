# Import helper functions
from utils import *

# Import packages
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import t
from scipy.integrate import dblquad
from celer import GroupLasso, GroupLassoCV

def oracle_test(y, X, k, beta_dir):
    n, d = X.shape
    X1 = X[:,:k] @ beta_dir
    X_new = np.column_stack([X1, X[:,k:]])
    df = n-d+k-1
    XtX_inv = np.linalg.inv(X_new.T@X_new)
    beta_hat = XtX_inv@X_new.T@y
    res = y - X_new@beta_hat
    sigma_hat_sq = (res @ res) / df
    se_beta_0 = np.sqrt(sigma_hat_sq * XtX_inv[0, 0])
    T = beta_hat[0] / se_beta_0
    return t.sf(T, df)

def glasso_test(y, X, k, penalty = None, MC = 200):
    n, d = X.shape
    m = n - d + k
    groups = construct_groups(k, d)
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)

    # Set penalty
    if penalty is None:
        u = np.random.randn(m)
        u /= np.linalg.norm(u)
        y_tilde = y_hat + sigma * V @ u
        model = GroupLassoCV(groups, fit_intercept=False, cv=10)
        model.fit(X, y_tilde)                                            
        l = model.alpha_
    else:
        l = penalty

    model = GroupLasso(groups, alpha=l, fit_intercept=False)
    model.fit(X, y)
    beta_k = model.coef_[0:k]
    M = V[:,:k].T @ X[:,:k]

    # Test statistic
    test_stat = np.linalg.norm(M @ beta_k)

    # Monte Carlo samples
    U = np.random.randn(m, MC)
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    Y_tilde = y_hat[:, np.newaxis] + sigma * (V @ U)
    count = 0
    for i in range(MC):
        model.fit(X, Y_tilde[:, i])
        beta_k = model.coef_[0:k]
        if (np.linalg.norm(M @ beta_k) >= test_stat):
            count += 1
    return (1 + count) / (MC + 1)

def L_test(y, X, k, penalty = None, point = None, MC = 200):
    n, d = X.shape
    m = n - d + k
    groups = construct_groups(k, d)
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)

    # Set penalty and approximation point
    if penalty is None and point is None:
        u = np.random.randn(m)
        u /= np.linalg.norm(u)
        y_tilde = y_hat + sigma * V @ u
        model = GroupLassoCV(groups, fit_intercept=False, cv=10)
        model.fit(X, y_tilde)                                            
        l = model.alpha_
        b = model.coef_[:k]
    else:
        l = penalty
        b = point[:k]

    V_k = V[:,:k]
    X_k = X[:,:k]
    X_rest = X[:,k:]

    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X_rest, y - X_k@b)
    lasso = model.coef_
    nz_idx = nonzero_indices(lasso, k)
    X_A = X[:,nz_idx]
    P_A = proj(X_A)
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        A = V_k.T@X_k@X_k.T@V_k
    else:
        A = V_k.T@X_k@np.linalg.inv(X_k.T@X_k - X_k.T@P_A@X_k + (n * l / norm_b) * np.eye(k))@X_k.T@V_k

    nu = (1/sigma)*np.linalg.inv(X_k.T@V_k)@X_k.T@(y_hat-X_rest@lasso - P_A@X_k@b)

    # Test statistic
    u = (1/sigma) * V_k.T@y
    test_stat = np.linalg.norm(A@(u + nu))

    # Monte Carlo samples
    U = np.random.randn(n - d + k, MC)
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    U_top = U[:k, :]
    l2_norms = np.linalg.norm(A@(U_top + nu[:, np.newaxis]), axis=0)

    return (1 + np.sum(l2_norms >= test_stat)) / (MC + 1)

def R_test(y, X, k, penalty = None, point = None):
    n, d = X.shape
    m = n - d + k
    groups = construct_groups(k, d)
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)

    # Set penalty and approximation point
    if penalty is None and point is None:
        u = np.random.randn(m)
        u /= np.linalg.norm(u)
        y_tilde = y_hat + sigma * V @ u
        model = GroupLassoCV(groups, fit_intercept=False, cv=10)
        model.fit(X, y_tilde)                                            
        l = model.alpha_
        b = model.coef_[:k]
    else:
        l = penalty
        b = point[:k]

    V_k = V[:,:k]
    X_k = X[:,:k]
    X_rest = X[:,k:]

    model = Lasso(alpha=l, fit_intercept=False)
    model.fit(X_rest, y - X_k@b)
    lasso = model.coef_
    nz_idx = nonzero_indices(lasso, k)
    X_A = X[:, nz_idx]
    P_A = proj(X_A)
    nu = -(1/sigma)*np.linalg.inv(X_k.T@V_k)@X_k.T@(y_hat-X_rest@lasso - P_A@X_k@b)

    # Test statistic
    u = (1/sigma) * V_k.T@y
    test_stat = np.linalg.norm(u - nu)

    if k == 1:
        return (1 - u_cdf(nu+test_stat, n, d) + u_cdf(nu-test_stat, n, d)).item()
    else:
        # Precompute constants
        D = norm_const(n, d, k)
        nu_norm = np.linalg.norm(nu)
        nu_sq = nu_norm**2
        alpha = (n - d - 2) / 2
        beta = (k - 3) / 2
        # Integrand
        def h(t, z):
            one_minus_t = 1 - t
            midpoint = (t + nu_sq - z**2) / (2 * nu_norm)
            quad_term = t - midpoint**2
            return z * one_minus_t**alpha * quad_term**beta
        # Set integration bounds
        result = dblquad(h, test_stat, nu_norm + 1, 
                            lambda z: nu_sq - 2 * z * nu_norm + z**2, 
                            lambda z: min(1, nu_sq + 2 * z * nu_norm + z**2))[0]
        return (D / nu_norm) * result

