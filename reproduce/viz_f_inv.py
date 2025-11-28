# import functions
from compare_tests import generate_design, generate_beta
from tests_support import *

# import packages
import numpy as np
import matplotlib.pyplot as plt

n = 100
d = 50
k = 2

sig = 0.3
spars_1 = k
spars_2 = 5
corr = 0.0 # Change to 0.9 for high correlation plots

radii = [1, 5, 10]
colors = ["red", "blue", "green"]

for i in range(3):
    np.random.seed(i)
    X = generate_design(n, d, k, corr)
    beta = generate_beta(d, k, sig, spars_1, spars_2, anti=False)
    y = np.random.multivariate_normal(X@beta, np.eye(n))
    V = construct_V(X, k)
    P_k = proj(X[:,k:])
    y_hat = P_k @ y
    sigma = np.linalg.norm(y - P_k@y)
    u = np.random.randn(n - d + k)
    u /= np.linalg.norm(u)
    y_tilde = y_hat + sigma * V @ u
    l, estimate = generate_penalty(y_tilde, X, k)
    
    V_k = V[:,:k]
    X_k = X[:,:k]
    X_rest = X[:,k:]

    def f_inv(b):
        model = Lasso(alpha=l, fit_intercept=False)
        model.fit(X[:,k:], y - X[:,0:k]@b)
        lasso = model.coef_
        unit = b / np.linalg.norm(b)
        return -(1/sigma)*np.linalg.inv(X_k.T@V_k)@(X[:,0:k].T@(y_hat-X[:,0:k]@b-X[:,k:]@lasso)-n*l*unit)
    
    theta = np.linspace(0, 2*np.pi, 240, endpoint=False)
    arrow_step = 24

    fig, ax = plt.subplots(figsize=(6, 6))
    all_pts = []
    
    for r, c in zip(radii, colors):
        circle = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        mapped = np.vstack([f_inv(b) for b in circle])

        ax.plot(circle[:, 0], circle[:, 1], color=c, lw=2)
        ax.plot(mapped[:, 0], mapped[:, 1], color=c, lw=2, ls="--")

        for idx in range(0, len(theta), arrow_step):
            ax.annotate(
                "", xy=mapped[idx], xytext=circle[idx],
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2, shrinkA=0, shrinkB=0)
            )
        all_pts.append(circle)
        all_pts.append(mapped)
    all_pts = np.vstack(all_pts)
    pad = 0.1 * (all_pts.max() - all_pts.min())
    xmin = all_pts[:,0].min() - pad
    xmax = all_pts[:,0].max() + pad
    ymin = all_pts[:,1].min() - pad
    ymax = all_pts[:,1].max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", fontsize=30)
    ax.set_ylabel("y", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig(f"finv_plots/f_inv_iter_{i}.png", dpi=300)
    plt.close()