import numpy as np


# Univariate Gaussian
def univariate_gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Multivariate Gaussian
def multivariate_gaussian(x, mu_vec, cov_mat):
    """
    Calculates the multivariate Gaussian probability density function (PDF).

    Args:
    - x: The input vector (numpy array) of size (d,).
    - mu_vec: The mean vector (numpy array) of size (d,).
    - sigma_mat: The covariance matrix (numpy array) of size (d, d).

    Returns:
    - The probability density value (float).
    """
    d = len(mu_vec)
    norm_factor = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(cov_mat)))
    delta_x_mu = x - mu_vec
    exp_factor = -0.5 * np.dot(np.dot(delta_x_mu.T, np.linalg.inv(cov_mat)), delta_x_mu)
    return norm_factor * np.exp(exp_factor)


# why do we take the determinant and what does that mean ?
