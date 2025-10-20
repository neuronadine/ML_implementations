"""
Linear Classifiers with Gradient Descent
Implements: LinearModel (base), LinearRegression, Perceptron, SVM, LogisticRegression
Adapted from 'Labo 4: Classifieurs Linéaires et Descente de Gradient' (IFT6390)
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter(dataset, marker='o'):
    d1 = dataset[dataset[:, -1] > 0]
    d2 = dataset[dataset[:, -1] < 0]
    plt.scatter(d1[:, 0], d1[:, 1], c='b', marker=marker, label='class 1', alpha=.7)
    plt.scatter(d2[:, 0], d2[:, 1], c='g', marker=marker, label='class 0', alpha=.7)
    plt.xlabel('x₀')
    plt.ylabel('x₁')

def finalize_plot(title):
    plt.title(title)
    plt.grid()
    plt.legend()

def decision_boundary(w):
    if w[1] == 0:
        raise RuntimeWarning("Decision boundary vertical or undefined.")
    xlim, ylim = plt.xlim(), plt.ylim()
    xx = np.linspace(-10, 10, 2)
    yy = -(w[2] + w[0] * xx) / w[1]
    plt.plot(xx, yy, c='r', lw=2, label='f(x)=0')
    plt.xlim(xlim)
    plt.ylim(ylim)

class LinearModel:
    """Parent class for all linear models."""

    def __init__(self, w0, reg):
        self.w = np.array(w0, dtype=float)
        self.reg = reg

    def predict(self, X):
        """Binary classification prediction."""
        return np.sign(X @ self.w)

    def error_rate(self, X, y):
        """Mean classification error."""
        preds = self.predict(X)
        return np.mean(preds != y)

    def loss(self, X, y):
        raise NotImplementedError

    def gradient(self, X, y):
        raise NotImplementedError

    def train(self, data, stepsize, n_steps, plot=False):
        """Full-batch gradient descent."""
        X, y = data[:, :-1], data[:, -1]
        losses, errors = [], []

        for _ in range(n_steps):
            grad = self.gradient(X, y)
            self.w -= stepsize * grad

            losses.append(self.loss(X, y))
            errors.append(self.error_rate(X, y))

            if plot and _ % 10 == 0:
                plt.figure()
                scatter(data, marker='o')
                decision_boundary(self.w)
                finalize_plot(f"Iteration {_}")
                plt.show()

        print(f"Training completed. Final training error: {errors[-1]*100:.2f}%")
        return np.array(losses), np.array(errors)



class LinearRegression(LinearModel):
    def loss(self, X, y):
        n = X.shape[0]
        preds = X @ self.w
        mse = 0.5 * np.mean((preds - y) ** 2)
        reg_term = 0.5 * self.reg * np.sum(self.w ** 2)
        return mse + reg_term

    def gradient(self, X, y):
        n = X.shape[0]
        preds = X @ self.w
        grad = (X.T @ (preds - y)) / n + self.reg * self.w
        return grad



class Perceptron(LinearModel):
    def loss(self, X, y):
        margins = -y * (X @ self.w)
        return np.mean(np.maximum(0, margins))

    def gradient(self, X, y):
        margins = -y * (X @ self.w)
        mask = margins > 0
        grad = -np.mean((y[mask, None] * X[mask]), axis=0) + self.reg * self.w
        return grad



class SVM(LinearModel):
    def loss(self, X, y):
        margins = 1 - y * (X @ self.w)
        return np.mean(np.maximum(0, margins)) + 0.5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        margins = 1 - y * (X @ self.w)
        mask = margins > 0
        grad = -np.mean((y[mask, None] * X[mask]), axis=0) + self.reg * self.w
        return grad



class LogisticRegression(LinearModel):
    def loss(self, X, y):
        logits = -y * (X @ self.w)
        log_term = np.log(1 + np.exp(logits))
        return np.mean(log_term) + 0.5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        logits = -y * (X @ self.w)
        sigmoid = 1 / (1 + np.exp(-logits))
        grad = -np.mean((y * X.T * (1 - sigmoid)).T, axis=0) + self.reg * self.w
        return grad


def test_model(modelclass, data, w0=None, reg=0.1, stepsize=0.1, n_steps=100):
    if w0 is None:
        w0 = np.random.randn(data.shape[1] - 1)
    model = modelclass(w0, reg)
    losses, errors = model.train(data, stepsize, n_steps)
    print(f"Final weights: {model.w}")
    return model, losses, errors
