"""Squared-error gradient boosting with residual regression trees."""
import numpy as np
from .decision_tree import DecisionTree


class GradientBoosting:
    def __init__(self, n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42):
        if n_estimators < 1 or learning_rate <= 0:
            raise ValueError("Estimator count and learning rate must be positive.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.initial = float(y.mean())
        predicted = np.full(len(y), self.initial)
        self.trees = []
        for iteration in range(self.n_estimators):
            residual = y - predicted
            tree = DecisionTree(self.max_depth, 5, random_state=self.random_state + iteration)
            tree.fit(X, residual)
            predicted += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not self.trees:
            raise ValueError("Call fit before predict.")
        predicted = np.full(len(X), self.initial)
        for tree in self.trees:
            predicted += self.learning_rate * tree.predict(X)
        return predicted


Model = GradientBoosting
