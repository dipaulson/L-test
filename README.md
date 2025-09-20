# Code repository for "An alternative to the F-test with enhanced power in sparse linear models" by Paulson, Sengupta, and Janson (2025)

This repository contains the source code for implementing all of the testing procedures and reproducing all of the figures presented in the [paper](https://arxiv.org/abs/2406.18390). The L-test is a new procedure for testing the signifiance of a subset of covariates (ie. $H_0: \beta_{1:k} = 0$) in a Gaussian linear model with $n \geq d$. Under the same assumptions as the classical F-test, the L-test delivers the exact same statistical guarantees while achieving higher power when the nuisance parameters $\beta_{-1:k}$ are sparse.

## Example of methods application
The file `tests_main.py` contains the main methods introduced in the paper, including the L-test, its computationally efficient variant the R-test, and the oracle test, which is used to provide power intuition. The file `utils.py` contains helper functions to implement these main methods, and `tests_support.py` contains additional tests considered in the paper. Below, we walk through a few examples of how a user can apply our methods.

```python
from tests_main import *
from utils import proj, generate_penalty
import numpy as np

np.random.seed(1)

# Set parameters
n = 100
d = 50
k = 10
var = 1

# Generate data
X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)
X = (X - X.mean(axis=0)) / X.std(axis=0)
beta = np.zeros(d)
sample = np.random.choice(np.arange(k, d), 5, replace=False)
beta[sample] = np.random.choice([-1, 1], size=5) * 5
a = 0.6 / np.sqrt(k)
beta[0:k] = np.full(k, a)
y = np.random.multivariate_normal(X@beta, (var ** 2) * np.eye(n))
P_k = proj(X[:,k:])
y_hat = P_k @ y
sigma = np.linalg.norm(y - P_k@y)

"""
Below, we run the oracle and L-test. All tests assess the signifiance of the first k covariates.
"""
oracle_pval = oracle_test(y, X, k, beta[0:k] / np.linalg.norm(beta[0:k]))
L_pval = L_test(y, X, k)

"""
For precise p-values, users can either increase the number of Monte Carlo samples
used to compute the L-test p-value or run the efficient R-test, which does not require resampling.
"""

L_pval = L_test(y, X, k, MC=500)
R_pval = R_test(y, X, k)

"""
To run the tests with the same penalty parameter, users can generate the
penalty separately and then specify it as an input.
"""

u = np.random.randn(n - d + k)
u /= np.linalg.norm(u)
y_tilde = y_hat + sigma * V @ u
l, estimate = generate_penalty(y_tilde, X, k)
L_pval = L_test(y, X, k, penalty = l, point = estimate)
R_pval = R_test(y, X, k, penalty = l, point = estimate)
```

## Reproducing Figures


## Citation

@article{DP-SS-LJ:2025,
  title={An alternative to the $F$-test with enhanced power in sparse linear models},
  author={Paulson, Danielle and Sengupta, Souhardya and Janson, Lucas},
  journal={arXiv preprint xxx},
  year={2025}
}
