"""From-scratch K-Means interface for images, reusing our NumPy implementation."""
from ...algorithms.kmeans import KMeans


class KMeansScratch(KMeans):
    """Lloyd iterations, k-means++ and restarts are implemented in algorithms/kmeans.py."""

    @property
    def centroids_(self):
        return self.cluster_centers_
