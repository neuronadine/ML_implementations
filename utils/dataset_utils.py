import numpy as np

# Splits dataset into train, validation and test sets.
def split_dataset(iris):
    n = iris.shape[0]
    train = iris[(np.arange(n)%5 <= 2)]
    validation = iris[(np.arange(n)%5 == 3)]
    test = iris[(np.arange(n)%5 == 4)]
    return train, validation, test