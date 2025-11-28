import numpy as np
import math
from scipy.stats import beta
from scipy.special import gammaln
from celer import GroupLasso, GroupLassoCV

# Projection matrix
def proj(X):
    return X@np.linalg.inv(X.T@X)@X.T

# Creates group LASSO groups
def construct_groups(k, d):
    first_sublist = list(range(0, k))
    remaining_sublists = [[i] for i in range(k, d)]
    groups = [first_sublist] + remaining_sublists
    return groups

# Constructs matrix V
def construct_V(X, k):
    n, d = X.shape
    I = np.eye(n)
    V_k = np.zeros((n, k))
    for i in range(k):
        V_k[:,i] = (I - proj(X[:,(i+1):]))@X[:,i] / np.linalg.norm((I - proj(X[:,(i+1):]))@X[:,i])
    U, S, VT = np.linalg.svd(X)
    V_remainder = U[:,d:]
    V = np.hstack((V_k, V_remainder))
    return V

# Identifies active set for L-test statistic
def nonzero_indices(lasso, k):
    return [i + k for i, val in enumerate(lasso) if val != 0]

# Normalization constant D in Proposition 2.3
def norm_const(n, d, k):
    numerator_log = gammaln(k / 2) + gammaln((n - d + k) / 2)
    denominator_log = (math.log(math.sqrt(math.pi))
                        + gammaln((k - 1) / 2)
                        + gammaln(k / 2)
                        + gammaln((n - d) / 2))
    result_log = numerator_log - denominator_log
    result = math.exp(result_log)
    return result

# Helper functions for distribution of |u - \nu| when k = 1
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def u_cdf(x, n, d):
    a=0.5
    b=(n-d)/2
    return 0.5*(1 + sign(x) * beta.cdf(x**2, a, b))

# Generate penalty parameter; default uses min rule as recommended
def generate_penalty(y, X, k, rule="min"):
    n, d = X.shape
    groups = construct_groups(k, d)
    glcv = GroupLassoCV(groups, fit_intercept=False, cv=10).fit(X, y)
    if rule == "min":                                 
        penalty = glcv.alpha_
        estimate = glcv.coef_
    elif rule == "1se":
        alphas = glcv.alphas_
        mean_cv_errors = np.mean(glcv.mse_path_, axis=1)
        std_cv_errors = np.std(glcv.mse_path_, axis=1)
        min_error_index = np.argmin(mean_cv_errors)
        min_error = mean_cv_errors[min_error_index]
        min_error_std = std_cv_errors[min_error_index]  
        alpha_1se_index = np.where(mean_cv_errors <= min_error + min_error_std)[0][0]
        penalty = alphas[alpha_1se_index]
        gl = GroupLasso(groups, alpha=penalty, fit_intercept=False).fit(X, y)
        estimate = gl.coef_
    else:
        raise ValueError("rule must be 'min' or '1se'")
    return penalty, estimate