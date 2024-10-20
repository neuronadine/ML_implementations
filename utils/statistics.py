import numpy as np

# Calculates feature means
def feature_means(self, iris):
    return iris[:, :4].mean(axis=0)

# Empirical covariance calculation
def empirical_covariance(self, iris):
    return np.cov(iris[:, :4], rowvar=False)

# Empirical covariance calculation
# def feature_means_class_1(self, iris):
#     iris_class_1 = iris[iris[:, 4] == 1]
#     return self.feature_means(iris_class_1)

# def empirical_covariance_class_1(self, iris):
#     iris_class_1 = iris[iris[:, 4] == 1]
#     return self.empirical_covariance(iris_class_1)