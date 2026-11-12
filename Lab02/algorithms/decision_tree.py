"""Regression tree minimizing squared error over quantile split candidates."""
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=10, max_features=None, random_state=42):
        if max_depth < 1 or min_samples_leaf < 1:
            raise ValueError("Depth and leaf size must be positive.")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.shape != (len(X),) or len(y) == 0:
            raise ValueError("X and y must be nonempty and aligned.")
        self.rng = np.random.default_rng(self.random_state)
        self.root = self._grow(X, y, 0)
        self.n_features_in_ = X.shape[1]
        return self

    def _grow(self, X, y, depth):
        value = float(y.mean())
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.var(y) < 1e-12:
            return value
        features = np.arange(X.shape[1])
        if self.max_features == "sqrt":
            features = self.rng.choice(features, max(1, int(np.sqrt(len(features)))), replace=False)
        best_loss = float(np.sum((y - value) ** 2))
        best_split = None
        for feature in features:
            thresholds = np.unique(np.quantile(X[:, feature], np.linspace(0, 1, 17)[1:-1]))
            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                count = int(mask.sum())
                if min(count, len(y) - count) < self.min_samples_leaf:
                    continue
                left, right = y[mask], y[~mask]
                loss = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
                if loss < best_loss - 1e-10:
                    best_loss = float(loss)
                    best_split = (int(feature), float(threshold), mask)
        if best_split is None:
            return value
        feature, threshold, mask = best_split
        return (feature, threshold, self._grow(X[mask], y[mask], depth + 1),
                self._grow(X[~mask], y[~mask], depth + 1))

    def predict(self, X):
        if self.root is None:
            raise ValueError("Call fit before predict.")
        result = []
        for row in np.asarray(X, dtype=float):
            node = self.root
            while isinstance(node, tuple):
                feature, threshold, left, right = node
                node = left if row[feature] <= threshold else right
            result.append(node)
        return np.asarray(result)


Model = DecisionTree
