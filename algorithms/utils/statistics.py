import numpy as np

# Calculates feature means
def get_feature_means(X):
    """
    Calculate the mean vector for the feature columns in the dataset.

    Parameters:
    - X: A numpy array of shape (N, d), 
            where N is the number of samples, and d is the number of features.

    Returns:
    - A numpy array of shape (d,) representing the mean vector.
    """
    return X.mean(axis=0)

# Empirical covariance calculation
def get_empirical_covariance(X):
    """
    Calculate the empirical covariance matrix of the feature columns.

    Parameters:
    - X: A numpy array of shape (N, d), 
            where N is the number of samples, and d is the number of features.

    Returns:
    - A numpy array of shape (d, d) representing the covariance matrix.
    """
    return np.cov(X, rowvar=False)

# def feature_means(self, iris):
#     return iris[:, :4].mean(axis=0)


# def empirical_covariance(self, iris):
#     return np.cov(iris[:, :4], rowvar=False)

# Empirical covariance calculation
# def feature_means_class_1(self, iris):
#     iris_class_1 = iris[iris[:, 4] == 1]
#     return self.feature_means(iris_class_1)

# def empirical_covariance_class_1(self, iris):
#     iris_class_1 = iris[iris[:, 4] == 1]
#     return self.empirical_covariance(iris_class_1)