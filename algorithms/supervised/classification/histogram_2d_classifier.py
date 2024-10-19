import numpy as np


class Histogram2DClassifier:
    """
    A 2D Histogram classifier.

    This class implements a histogram-based classifier for a two-dimensional
    feature space. It uses a grid-based approach to estimate the probability
    distribution for each class.

    Args:
        m1 (int): The number of bins along feature_1 (dimention_1)
        m2 (int): The number of bins along feature_2 (dimention_2)
        x1min (float), x1max (float): The min and max values for feature_1.
        x2min (float), x1max (float): The min and max values for feature_2.
        n_classes: The number of classes (e.g. number of species of iris).

    Returns:
        None

    Author : Nadine Mohamed
    Date : October 19, 2024
    """

    def __init__(self, m1, x1min, x1max, m2, x2min, x2max, nclasses):
        # hyperparams
        self.m1 = m1
        self.x1min = x1min
        self.x1max = x1max
        self.m2 = m2
        self.x2min = x2min
        self.x2max = x2max

        # Initialize 3D tensor to count occurances of classes in each 2D bin.
        self.C = np.zeros((m1, m2, nclasses))
        self.bin_size1 = (x1max - x1min) / m1
        self.bin_size2 = (x2max - x2min) / m2

    def train(self, X_train):
        """
        Trains the 2D Histogram classifier using the provided training data.

        Args:
            X_train (list of tuples): Each tuple is of the form (x, y),
            where x is a feature vector with two elements, and y is the class
            label.
        """
        for x, y in X_train:
            i = int(np.floor((x[0] - self.x1min) / self.d1))
            j = int(np.floor((x[1] - self.x2min) / self.d2))
            self.C[i, j, y] += 1

    def class_prob(self, x_test):
        """
        Calculates the class probabilities for a given test sample.

        Args:
            x_test (array-like): A 2-element array representing the test point.

        Returns:
            np.array: A 1D array of probabilities for each class.
        """
        i = int(np.floor((x_test[0] - self.x1min) / self.d1))
        j = int(np.floor((x_test[1] - self.x2min) / self.d2))
        count_per_class = self.C[i, j, :]
        total_count = np.sum(count_per_class)
        if total_count == 0:
            # If no samples fall into the bin, return equal probability
            return np.ones(self.nclasses) / self.nclasses
        return count_per_class / total_count

    def pred(self, x_test):
        """
        Predicts the class for a given test sample.

        Args:
            x_test (array-like): A 2-element array representing the test point.

        Returns:
            int: The predicted class label.
        """
        probs = self.class_prob(x_test)
        return np.argmax(probs)
