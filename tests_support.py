# Import helper functions
from utils import *

# Import packages
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import f
from asgl import Regressor

def F_test(y, X, k):
    n, d = X.shape
    P = proj(X)
    res_full = y - P@y
    RSS_full = res_full@res_full
    P_k = proj(X[:,k:])
    res_red = y - P_k@y
    RSS_red = res_red@res_red
    dfn, dfd = k, (n - d)
    F = ((RSS_red - RSS_full) / dfn) / (RSS_full / dfd)
    return f.sf(F, dfn, dfd)

def PC_test(y, X, k, var_percent = 0.85):
    X_k = X[:,:k]
    pca = PCA(n_components=None)
    pca.fit(X_k)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    r = np.searchsorted(cumulative_variance, var_percent) + 1
    X_k_red = pca.transform(X_k)[:, :r]
    X_new = np.hstack([X_k_red, X[:,k:]])
    return F_test(y, X_new, r)

def phi_test(y, X, k, penalty = None, point = None, MC = 200):
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

    # Test statistic
    u = (1/sigma) * V_k.T@y
    test_stat = np.linalg.norm(A@u)

    # Monte Carlo samples
    U = np.random.randn(n - d + k, MC)
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    U_top = U[:k, :]
    l2_norms = np.linalg.norm(A@U_top, axis=0)

    return (1 + np.sum(l2_norms >= test_stat)) / (MC + 1)

def _load_r_pkg(path="l_testing.R"):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

    numpy2ri.activate()

    with open(path, "r") as r_file:
        r_code = r_file.read()
    r_pkg = SignatureTranslatedAnonymousPackage(r_code, "r_pkg")
    return r_pkg

def bonf_ell_test(y, X, k, r_script_path="l_testing.R"):
    r_pkg = _load_r_pkg(r_script_path)
    p_min = 1.0
    for i in range(k):
        p_i = r_pkg.l_test(y, X, i + 1)[0]
        if p_i <= p_min:
            p_min = p_i
    return p_min * k

def sglasso_test(y, X, k, MC = 200):
    n, d = X.shape
    m = n - d + k
    group_index = np.zeros(d) 
    group_index[k:] = np.arange(1, d-k+1)
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)

    # Set penalties
    u = np.random.randn(m)
    u /= np.linalg.norm(u)
    y_tilde = y_hat + sigma * V @ u
    # Custom CV step following recommendation of asgl package: https://github.com/alvaromc317/asgl/blob/9631bcdb55d0245274fca4d80d7e7f064965c9e9/user_guide.ipynb
    model = Regressor(model='lm', penalization='sgl', fit_intercept=False)
    param_grid = {'lambda1': [1e-4, 1e-3, 1e-2, 1e-1, 1], 'alpha': [.1, .5, .7, .9, .95, .99, 1]}
    gscv = GridSearchCV(model, param_grid, scoring='neg_median_absolute_error', cv=10)
    gscv.fit(X, y_tilde, **{'group_index': group_index})
    l1 = gscv.best_params_['lambda1']
    a = gscv.best_params_['alpha']

    model = Regressor(model='lm', penalization='sgl', fit_intercept=False, lambda1=l1, alpha=a)
    model.fit(X, y, group_index=group_index)
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
        model.fit(X, Y_tilde[:, i], group_index=group_index)
        beta_k = model.coef_[0:k]
        if (np.linalg.norm(M @ beta_k) >= test_stat):
            count += 1
    return (1 + count) / (MC + 1)


# Standard LASSO test p-value
def lasso_test(y, X, k, MC = 200):
    n, d = X.shape
    m = n - d + k
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)

    # Set penalty
    u = np.random.randn(m)
    u /= np.linalg.norm(u)
    y_tilde = y_hat + sigma * V @ u
    model = LassoCV(fit_intercept=False, cv=10)
    model.fit(X, y_tilde)                                            
    l = model.alpha_

    model = Lasso(alpha=l, fit_intercept=False)
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

def enet_test(y, X, k, MC = 200):
    n, d = X.shape
    m = n - d + k
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)

    # Set penalties
    u = np.random.randn(m)
    u /= np.linalg.norm(u)
    y_tilde = y_hat + sigma * V @ u
    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False, cv=10)
    model.fit(X, y_tilde)  
    a = model.alpha_                                          
    l1 = model.l1_ratio_

    model = ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=False)
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
