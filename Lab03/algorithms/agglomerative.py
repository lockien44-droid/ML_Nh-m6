"""Average-linkage agglomerative clustering implemented with NumPy."""
import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.isfinite(X).all() or not 2 <= self.n_clusters < len(X):
            raise ValueError("Finite data and 2 <= K < n are required.")
        n = len(X)
        distances = np.full((2*n-1, 2*n-1), np.inf)
        distances[:n, :n] = np.sqrt(np.sum((X[:, None]-X[None, :])**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        active = list(range(n))
        sizes = {i: 1 for i in active}
        members = {i: [i] for i in active}
        linkage = []
        for step in range(n-1):
            block = distances[np.ix_(active, active)]
            a, b = np.unravel_index(np.argmin(block), block.shape)
            left, right = active[a], active[b]
            parent = n + step
            sizes[parent] = sizes[left]+sizes[right]
            members[parent] = members[left]+members[right]
            linkage.append([left, right, distances[left, right], sizes[parent]])
            for other in active:
                if other not in (left, right):
                    value = (sizes[left]*distances[left, other]+sizes[right]*distances[right, other])/sizes[parent]
                    distances[parent, other] = distances[other, parent] = value
            active = [i for i in active if i not in (left, right)] + [parent]
            if len(active) == self.n_clusters:
                self.labels_ = np.empty(n, dtype=int)
                for label, group in enumerate(active):
                    self.labels_[members[group]] = label
        self.linkage_ = np.asarray(linkage)
        return self
