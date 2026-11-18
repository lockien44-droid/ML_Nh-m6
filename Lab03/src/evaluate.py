"""Compute exact Euclidean Silhouette and original-scale cluster profiles."""
import numpy as np


def silhouette_samples(X, labels):
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if X.ndim != 2 or labels.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("Finite features and aligned labels are required.")
    groups = np.unique(labels)
    if not 2 <= len(groups) < len(X):
        raise ValueError("Silhouette requires 2 to n-1 observed clusters.")
    distances = np.sqrt(np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
    scores = np.zeros(len(X))
    for index in range(len(X)):
        own = labels == labels[index]
        if own.sum() == 1:
            continue
        a = distances[index, own].sum() / (own.sum() - 1)
        b = min(distances[index, labels == group].mean()
                for group in groups if group != labels[index])
        denominator = max(a, b)
        scores[index] = (b - a) / denominator if denominator > 0 else 0
    return scores


def cluster_profiles(frame, labels, features):
    data = frame.assign(cluster=labels)
    means = data.groupby("cluster")[features].mean().add_suffix("_mean")
    means.insert(0, "customer_count", data.groupby("cluster").size())
    return means.reset_index()
