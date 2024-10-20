import numpy as np

# Gaussian (RBF) kernel function used for Soft Parzen
def gaussian_kernel(distances, sigma):
    return np.exp(-(distances**2) / (2 * sigma**2))