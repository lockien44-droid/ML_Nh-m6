"""Bootstrap ensemble of independently seeded regression trees."""
import numpy as np
from .decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=60, max_depth=10, min_samples_leaf=5, random_state=42):
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            indices = rng.integers(0, len(y), len(y))
            tree = DecisionTree(self.max_depth, self.min_samples_leaf, "sqrt",
                                int(rng.integers(0, 2**31)))
            self.trees.append(tree.fit(X[indices], y[indices]))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("Call fit before predict.")
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


Model = RandomForest
