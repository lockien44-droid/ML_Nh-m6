"""Ridge regression trained by full-batch gradient descent."""
import numpy as np


class LinearRegression:
    def __init__(self, alpha=1.0, learning_rate=0.1, n_iterations=3000):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer.")
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.shape != (len(X),) or not X.size:
            raise ValueError("X and y must be nonempty and aligned.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training data must be finite.")
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        n_samples = len(y)
        # Minimize (sum((prediction-y)^2) + alpha*sum(weights^2))/(2*n).
        # The intercept is excluded from the regularization penalty.
        for _ in range(self.n_iterations):
            predicted = X @ self.weights + self.bias
            error = predicted - y
            gradient_weights = (X.T @ error + self.alpha * self.weights) / n_samples
            gradient_bias = error.mean()
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias
            if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
                raise ValueError("Gradient descent diverged. Reduce learning_rate.")
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features must match training features.")
        return X @ self.weights + self.bias


Model = LinearRegression
