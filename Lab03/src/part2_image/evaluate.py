"""Measure color reconstruction and cluster population."""
import numpy as np
import pandas as pd


def evaluate_image(X, model, original=None, reconstructed=None):
    residual = ((original.astype(float) - reconstructed.astype(float)) if original is not None
                else (X - model.cluster_centers_[model.labels_]) * 255.0)
    mse = float(np.mean(residual ** 2))
    counts = np.bincount(model.labels_, minlength=model.n_clusters)
    populations = pd.DataFrame({"cluster": np.arange(model.n_clusters),
                                "pixel_count": counts, "fraction": counts / len(X)})
    scores = dict(k=model.n_clusters, inertia=model.inertia_, rgb_mse=mse,
                  rgb_rmse=float(np.sqrt(mse)), iterations=model.n_iter_,
                  converged=model.converged_, observed_clusters=int(np.count_nonzero(counts)))
    return scores, populations



def evaluate_segmentation(X, array, k, settings):
    """Fit only our NumPy K-Means and measure color reconstruction."""
    from time import perf_counter
    from .segment import segment_image
    started = perf_counter()
    model, image = segment_image(X, array.shape, k, settings)
    elapsed = perf_counter() - started
    scores, _ = evaluate_image(X, model, array, image)
    scores.update(n_iter=model.n_iter_, time_scratch=elapsed)
    return scores, image, model
