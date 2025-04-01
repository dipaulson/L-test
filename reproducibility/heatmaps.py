from tests import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.linear_model import Lasso, LassoCV
from celer import GroupLassoCV, GroupLasso
from scipy.linalg import sqrtm
from scipy.special import gammaln
from scipy.integrate import dblquad
from scipy.stats import f, t

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

def construct_unit(y, X, V, sigma):
    V_k = V[:,0:k]
    return (1/sigma)*V_k.T@y

def construct_oracle_mid(y, X, V, sigma, beta):
    C = 100
    gamma = (C/np.linalg.norm(beta[0:k]))*beta[0:k]
    V_k = V[:,0:k]
    return -(1/sigma)*V_k.T@X[:,0:k]@gamma

def construct_mid(y, X, V, sigma, gamma):
    V_k = V[:,0:k]
    return -(1/sigma)*V_k.T@X[:,0:k]@gamma

def sample_from_unit_sphere(dim, num_samples):
    samples = np.random.multivariate_normal(np.zeros(dim), np.identity(dim), size=num_samples)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    samples = samples/norms
    return samples

# F-test p-value
def F_stat(y, X, V, params):
    [n, d, k] = params
    V_k = V[:,0:k]
    P_k = tests.proj(X[:,k:])
    sigma = np.linalg.norm(y - P_k@y)
    beta_OLS = tests.OLS(y, X)
    norm_sq = np.linalg.norm(V_k.T@X[:,0:k]@beta_OLS[0:k])**2
    return (norm_sq/k)/((sigma**2 - norm_sq)/(n-d))

def F_pval(y, X, k):
    n, d = X.shape
    params = [n, d, k]
    V = tests.construct_V(X, params)
    obs = F_stat(y, X, V, params)
    dfn, dfd = k, (n - d)
    return (1 - f.cdf(obs, dfn, dfd))

# heatmaps

def generate_heatmap_with_mid(u, mid, mid_color, plot_name):
    # Set seeds
    np.random.seed(100)
    
    # Parameters
    dim = n-d+k
    num_samples = 10000

    # Sample points and project onto first two coordinates
    samples = sample_from_unit_sphere(dim, num_samples)
    projected_points = samples[:, :2]

    # Create heat map with custom colormap and transparency
    cmap = cm.get_cmap("coolwarm")
    ax = sns.kdeplot(x=projected_points[:, 0], y=projected_points[:, 1], cmap=cmap, fill=True, levels=100, alpha=0.5, bw=2)
    cbar = plt.colorbar(ax.collections[-1], ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.05, "High", ha='center', va='bottom', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Low", ha='center', va='top', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.set_label("Density")

    low_density_color = mcolors.to_hex(cmap(0))

    # Plot unit vector and recentering vector
    plt.arrow(0, 0, u[0], u[1], head_width=0.03, head_length=0.03, fc='black', ec='black', linewidth=0.5)
    plt.arrow(0, 0, mid[0], mid[1], head_width=0.03, head_length=0.03, fc=mid_color, ec=mid_color, linewidth=0.5)

    radius = np.linalg.norm(u - mid)
    circle_around_point = plt.Circle(mid, radius, color='gray', fill=False, linewidth=1, linestyle='dashed')
    plt.gca().add_patch(circle_around_point)
    plt.plot([u[0], mid[0]], [u[1], mid[1]], linewidth=1, linestyle='--', color='gray')

    # Formatting
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.yticks(np.arange(-1.0, 1.1, 0.5))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f'setting_3/{plot_name}.png')
    plt.close()

def generate_heatmap_oracle(u, mid, mid_color, plot_name):
    # Set seed
    np.random.seed(100)
    
    # Parameters
    dim = n-d+k
    num_samples = 10000

    # Sample points and project onto first two coordinates
    samples = sample_from_unit_sphere(dim, num_samples)
    projected_points = samples[:, :2]

    # Create heat map with custom colormap and transparency
    cmap = cm.get_cmap("coolwarm")
    ax = sns.kdeplot(x=projected_points[:, 0], y=projected_points[:, 1], cmap=cmap, fill=True, levels=100, alpha=0.5, bw=2)
    cbar = plt.colorbar(ax.collections[-1], ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.05, "High", ha='center', va='bottom', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Low", ha='center', va='top', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.set_label("Density")

    low_density_color = mcolors.to_hex(cmap(0))

    # Plot unit vector and recentering vector
    plt.arrow(0, 0, u[0], u[1], head_width=0.03, head_length=0.03, fc='black', ec='black', linewidth=0.5)
    plt.arrow(0, 0, mid[0], mid[1], head_width=0.03, head_length=0.03, fc=mid_color, ec=mid_color, linewidth=0.5)

    # Draw line through unit vector head tangent to recentering vector
    tangent_direction = np.array([-mid[1], mid[0]])
    tangent_direction = tangent_direction / np.linalg.norm(tangent_direction)

    line_length = 2
    x_vals = np.linspace(u[0] - line_length * tangent_direction[0], 
                         u[0] + line_length * tangent_direction[0], 100)
    y_vals = np.linspace(u[1] - line_length * tangent_direction[1], 
                         u[1] + line_length * tangent_direction[1], 100)
    plt.plot(x_vals, y_vals, color='gray', linewidth=1, linestyle='dashed')

    # Formatting
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.yticks(np.arange(-1.0, 1.1, 0.5))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f'{plot_name}.png')
    plt.close()

def generate_heatmap_F(u, plot_name):
    # Set seeds
    np.random.seed(100)
    
    # Parameters
    dim = n-d+k
    num_samples = 10000

    # Sample points and project onto first two coordinates
    samples = sample_from_unit_sphere(dim, num_samples)
    projected_points = samples[:, :2]

    # Create heat map with custom colormap and transparency
    cmap = cm.get_cmap("coolwarm")
    ax = sns.kdeplot(x=projected_points[:, 0], y=projected_points[:, 1], cmap=cmap, fill=True, levels=100, alpha=0.5, bw=2)
    cbar = plt.colorbar(ax.collections[-1], ax=ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.05, "High", ha='center', va='bottom', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.05, "Low", ha='center', va='top', fontsize=10, color='black', transform=cbar.ax.transAxes)
    cbar.set_label("Density")

    low_density_color = mcolors.to_hex(cmap(0))

    # Plot unit vector
    plt.arrow(0, 0, u[0], u[1], head_width=0.03, head_length=0.03, fc='black', ec='black', linewidth=0.5)

    radius = np.linalg.norm(u)
    circle_around_point = plt.Circle([0, 0], radius, color='gray', fill=False, linewidth=1, linestyle='dashed')
    plt.gca().add_patch(circle_around_point)

    # Formatting
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.yticks(np.arange(-1.0, 1.1, 0.5))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f'setting_3/{plot_name}.png')
    plt.close()

def generate_plots(sig, spars_1, spars_2, rho):
    params = [n, d, k]
    p_values = np.zeros((3, 4))
    for i in range(3):
        # set seeds
        random.seed(i)
        np.random.seed(i)
        # generate data for i-th iteration (3 total)
        X = generate_design(rho)
        V = construct_V(X, params)
        beta = generate_beta(sig, spars_1, spars_2)
        y = np.random.multivariate_normal(X@beta, np.identity(n))
        I = np.identity(n)
        P_k = proj(X[:,k:])
        sigma = np.linalg.norm((I - P_k)@y)
        gam_1 = gamma_1(y, X, k)
        gam_2 = gamma_2(y, X, k)
        u = construct_unit(y, X, V, sigma)
        mid_oracle = construct_oracle_mid(y, X, V, sigma, beta)
        mid_1 = construct_mid(y, X, V, sigma, gam_1)
        mid_2 = construct_mid(y, X, V, sigma, gam_2)
        # get heatmaps for iteration i
        generate_heatmap_oracle(u, mid_oracle, "#f68080", f"oracle_iteration_{i}")
        generate_heatmap_with_mid(u, mid_1, "#8ab17d", f"recent_1_iteration_{i}")
        generate_heatmap_with_mid(u, mid_2, "#FFEA00", f"recent_2_iteration_{i}")
        generate_heatmap_F(u, f"F_iteration_{i}")
        # get p-values
        p_F = F_pval(y, X, k)
        p_oracle = oracle_pval(y, X, beta, k)
        p_1 = R_pval(y, X, gam_1, k)
        p_2 = R_pval(y, X, gam_2, k)
        p_values[i] = np.array([p_F, p_oracle, p_1, p_2])
    print(p_values)

# Set params
n = 100
d = 50
k = 2

# Generate plots

#generate_plots(0.2, 2, 5, 0) # first row of panels

#generate_plots(0.2, 2, 35, 0) # second row of panels

#generate_plots(0.2, 2, 5, 0.9) # third row of panels


