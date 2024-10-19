import numpy as np

# Minkowski
def minkowski(x,y,p=2):
  return np.sum(np.abs(x-y)**p)**(1/p)

def minkowski_mat(X,y,p=2):
  return np.sum(np.abs(X-y)**p, axis=1)**(1/p)


def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)

