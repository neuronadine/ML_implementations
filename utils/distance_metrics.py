import numpy as np

# Minkowski
def minkowski(x,y,p=2):
  return np.sum(np.abs(x-y)**p)**(1/p)

def minkowski_mat(X,y,p=2):
  return np.sum(np.abs(X-y)**p, axis=1)**(1/p)

# Manhattan (L1)
def manhattanL1(X, y):
    return np.sum(np.abs(X - y), axis=1)

# Euclidean

# Tanimoto