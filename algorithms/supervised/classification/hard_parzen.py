import numpy as np
from ....utils import draw_rand_label


class HardParzen:
    def __init__(self, h):
        self.h = h

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels.astype(int)

    def predict(self, test_data):
        num_test = test_data.shape[0]
        classes_pred = np.zeros(num_test)
        classes = np.unique(self.train_labels)

        for i, test_point in enumerate(test_data):
            distances = np.sum(np.abs(test_point - self.train_inputs), axis=1)

            neighbors = np.where(distances < self.h)[0]

            if len(neighbors) > 0:
                unique_labels, counts = np.unique(
                    self.train_labels[neighbors], return_counts=True
                )
                classes_pred[i] = unique_labels[np.argmax(counts)]
            else:
                classes_pred[i] = draw_rand_label(test_point, classes)
        return classes_pred
