import numpy as np
from ....utils.distance_metrics import manhattanL1


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels.astype(int)

    def predict(self, test_data):
        num_test = test_data.shape[0]
        classes_pred = np.zeros(num_test)

        for i, test_point in enumerate(test_data):
            distances = manhattanL1(self.train_inputs - test_point)
            weights = np.exp(-(distances**2) / (2 * self.sigma**2))
            label_weights = np.bincount(self.train_labels, weights=weights)
            classes_pred[i] = np.argmax(label_weights)
        return classes_pred
