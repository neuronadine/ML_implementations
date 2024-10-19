import numpy as np
from ....utils import draw_rand_label


class KNNClassifier:
    """
    A K-Nearest Neighbors (KNN) classifier.

    This class implements the K-Nearest Neighbors algorithm for classification
    tasks, using a specified distance function to find the k-nearest neighbors
    of a given test sample.

    Args:
        dist_func (callable): A function that computes the distance for k.
        k (int): The number of nearest neighbors to consider for predicting.

    Author: Nadine Mohamed
    Date: October 19, 2024

    Additional Information:
    -----------------------
    (A) Complexity:
        - Training time complexity: O(1).
        - Prediction time complexity: O(N * D).
            - N is the number of training samples and;
            - D is the dimensionality of the data.

    (B) Non-parametric

    (C) Supports both binary and multiclass classification.

    (D) Hyperparameters:
        - k: The number of nearest neighbors to use for prediction.
        - dist_func: The distance metric (e.g., Euclidean, Manhattan).

    (E) Tasks:
        - Suitable for classification tasks with non-linear decision boundaries.
        - Used in image recognition, pattern recognition, and
            other tasks involving similarity-based decision-making.

    (F) Limitations:
        - Computationally expensive during prediction for large datasets
            due to distance calculations.
        - Sensitive to the choice of k; small k may lead to overfitting,
            while large k may smooth out decision boundaries.
        - Curse of dimensionality
    """

    def __init__(self, dist_func, k):
        self.dist_func = dist_func
        self.k = k

    def fit(self, X_train, y_train):
        """
        Fits the classifier to the training data.

        Args:
            X_train (ndarray): The training data of shape (N, D), where N is the
                number of samples and D is the number of features.
            y_train (ndarray): The class labels for training set of shape (N,).
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predicts the class labels for the given test data.

        Args:
            X_test (ndarray): The test data of shape (M, D), where M is the
                number of test samples and D is the number of features.

        Returns:
            ndarray: An array of predicted class labels for the test data.
        """

        num_test = X_test.shape[0]
        classes_pred = np.zeros(num_test)
        classes = np.unique(self.y_train)

        for i, x_test in enumerate(X_test):
            distances = self.dist_func(self.X_train, x_test)
            idx_neighbors = np.argsort(distances)[:, self.k]

            if len(idx_neighbors) > 0:
                unique_labels, counts = np.unique(
                    self.y_train[idx_neighbors], return_counts=True
                )
                classes_pred[i] = unique_labels[np.argmax(counts)]
            else:
                classes_pred[i] = draw_rand_label(x_test, classes)
        return classes_pred
