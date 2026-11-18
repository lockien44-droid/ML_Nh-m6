"""Fit one K-Means candidate and expose its diagnostics."""
from ..algorithms.kmeans import KMeans


def train_model(X, k, settings):
    return KMeans(n_clusters=int(k), max_iter=settings["max_iter"], tol=settings["tol"],
                  random_state=settings["random_state"], n_init=settings.get("n_init", 20)).fit(X)
