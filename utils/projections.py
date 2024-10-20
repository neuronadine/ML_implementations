import numpy as np

# random projections
def random_projections(X, A):
    return (1 / np.sqrt(2)) * np.dot(X, A)