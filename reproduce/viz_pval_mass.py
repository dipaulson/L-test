# Import functions
from tests_main import *
from tests_support import *
from compare_tests import generate_design, generate_beta

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

def sample_from_unit_sphere(dim, num_samples):
    samples = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), size=num_samples)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples/norms
    return samples

def F_heatmap(unit_vec, fig_title, plot_name):
    dim = n-d+k
    num_samples = 10000
    samples = sample_from_unit_sphere(dim, num_samples)
    projected_points = samples[:, :k]

    # Create heatmap
    cmap = cm.get_cmap("coolwarm")
    ax = sns.kdeplot(x=projected_points[:, 0], y=projected_points[:, 1], cmap=cmap, fill=True, levels=100, alpha=0.5, bw=2)
    cbar = plt.colorbar(ax.collections[-1], ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.05, "High", ha='center', va='bottom', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Low", ha='center', va='top', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.set_label("Density")

    # Plot unit vector
    plt.arrow(0, 0, unit_vec[0], unit_vec[1], head_width=0.03, head_length=0.03, fc='black', ec='black', linewidth=2)

    # Draw circle around unit vector
    radius = np.linalg.norm(unit_vec)
    circle_around_point = plt.Circle([0, 0], radius, color='black', fill=False, linewidth=2, linestyle='dashed')
    plt.gca().add_patch(circle_around_point)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.yticks(np.arange(-1.0, 1.1, 0.5))
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.title(fig_title, fontsize=30)
    plt.savefig(f'heatmaps/low_corr/{plot_name}.png')
    plt.close()

def oracle_heatmap(unit_vec, mid, fig_title, plot_name):
    dim = n-d+k
    num_samples = 10000
    samples = sample_from_unit_sphere(dim, num_samples)
    projected_points = samples[:, :k]

    # Create heatmap
    cmap = cm.get_cmap("coolwarm")
    ax = sns.kdeplot(x=projected_points[:, 0], y=projected_points[:, 1], cmap=cmap, fill=True, levels=100, alpha=0.5, bw=2)
    cbar = plt.colorbar(ax.collections[-1], ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.05, "High", ha='center', va='bottom', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Low", ha='center', va='top', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.set_label("Density")

    # Plot unit vector and midpoint
    plt.arrow(0, 0, unit_vec[0], unit_vec[1], head_width=0.03, head_length=0.03, fc='black', ec='black', linewidth=2)
    plt.arrow(0, 0, mid[0], mid[1], head_width=0.03, head_length=0.03, fc="black", ec="black", linewidth=2)

    # Draw line through unit vector head tangent to recentering vector
    tangent_direction = np.array([-mid[1], mid[0]])
    tangent_direction = tangent_direction / np.linalg.norm(tangent_direction)
    line_length = 2
    x_vals = np.linspace(unit_vec[0] - line_length * tangent_direction[0], 
                         unit_vec[0] + line_length * tangent_direction[0], 100)
    y_vals = np.linspace(unit_vec[1] - line_length * tangent_direction[1], 
                         unit_vec[1] + line_length * tangent_direction[1], 100)
    plt.plot(x_vals, y_vals, color='black', linewidth=2, linestyle='dashed')

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.yticks(np.arange(-1.0, 1.1, 0.5))
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.title(fig_title, fontsize=30)
    plt.savefig(f'heatmaps/low_corr/{plot_name}.png')
    plt.close()

def L_heatmap(unit_vec, center, axes, angle, fig_title, plot_name):
    dim = n-d+k
    num_samples = 10000
    samples = sample_from_unit_sphere(dim, num_samples)
    projected_points = samples[:, :k]

    # Create heatmap
    cmap = cm.get_cmap("coolwarm")
    ax = sns.kdeplot(x=projected_points[:, 0], y=projected_points[:, 1], cmap=cmap, fill=True, levels=100, alpha=0.5, bw=2)
    cbar = plt.colorbar(ax.collections[-1], ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.05, "High", ha='center', va='bottom', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Low", ha='center', va='top', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.set_label("Density")

    # Plot unit vector and ellipse
    plt.arrow(0, 0, unit_vec[0], unit_vec[1], head_width=0.03, head_length=0.03, fc='black', ec='black', linewidth=2)
    center2 = np.asarray(center).ravel()[:2]
    plt.scatter(center2[0], center2[1], s=20, color='black')
    a, b = float(axes[0]), float(axes[1])
    ell = Ellipse(
        xy=center2,
        width=2*a, height=2*b,      
        angle=float(angle),     
        edgecolor='black', facecolor='none', linewidth=2, linestyle='--')
    plt.gca().add_patch(ell)
    
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.yticks(np.arange(-1.0, 1.1, 0.5))
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.title(fig_title, fontsize=30)
    plt.savefig(f'heatmaps/low_corr/{plot_name}.png')
    plt.close()

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

    # Geometry
    center = -nu
    M = A.T@A
    c = test_stat**2
    M = M / c
    eigvals, eigvecs = np.linalg.eigh(M)
    semi_axes = 1.0 / np.sqrt(eigvals)
    order = np.argsort(semi_axes)[::-1]
    semi_axes = semi_axes[order]
    eigvecs = eigvecs[:, order]
    angle_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return (1 + np.sum(l2_norms >= test_stat)) / (MC + 1), center, semi_axes, angle_deg

n = 100
d = 50
k = 2
def make_heatmaps(sig, spars_1, spars_2, corr):
    for i in range(3):
        np.random.seed(i)
        X = generate_design(n, d, k, corr)
        beta = generate_beta(d, k, sig, spars_1, spars_2, anti=False)
        y = np.random.multivariate_normal(X@beta, np.eye(n))
        V = construct_V(X, k)
        P_k = proj(X[:,k:])
        y_hat = P_k @ y
        sigma = np.linalg.norm(y - P_k@y)
        # p-values
        F_pval = F_test(y, X, k)
        oracle_pval = oracle_test(y, X, k, beta[0:k] / np.linalg.norm(beta[0:k]))
        L_pval, center, axes, angle = L_test(y, X, k)
        # oracle midpoint
        oracle_mid = -(1/sigma)*V[:,:k].T@X[:,:k]@((100/np.linalg.norm(beta[:k]))*beta[:k])
        # unit vector
        u_obs = (1/sigma) * V[:,:k].T@y
        # heatmaps
        F_heatmap(u_obs, f"$p={F_pval:.4f}$", f"F_iteration_{i}")
        oracle_heatmap(u_obs, oracle_mid, f"$p={oracle_pval:.4f}$", f"oracle_iteration_{i}")
        L_heatmap(u_obs, center, axes, angle, f"$p={L_pval:.4f}$", f"L_iteration_{i}")

"""
Uncomment to get plots for high correlation setting.
"""
make_heatmaps(0.3, 2, 5, 0.0)

"""
Uncomment to get plots for high correlation setting.
"""
# make_heatmaps(0.3, 2, 5, 0.9)