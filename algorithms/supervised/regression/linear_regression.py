# algorithms/supervised/regression/linear_regression.py
"""
True linear regression API (continuous outputs) built on the shared GD implementation.
"""

import numpy as np
# Reuse the implementation & training loop from the classification module
from algorithms.supervised.classification.linear_classifiers_gd import LinearRegression as _LinearRegressionBase

class LinearRegression(_LinearRegressionBase):
    """
    Pure regression version:
      - predict() returns continuous values (X @ w)
      - error() returns RMSE (named 'error_rate' to stay compatible with the shared trainer)
    """

    def predict(self, X):
        # Continuous prediction
        return X @ self.w

    def error_rate(self, X, y):
        # RMSE as the regression metric
        preds = self.predict(X)
        return float(np.sqrt(np.mean((preds - y) ** 2)))
