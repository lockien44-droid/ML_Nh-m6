"""Cluster pixels with the shared NumPy K-Means and reconstruct colors."""
from .preprocessing import reconstruct_image
from .kmeans_scratch import KMeansScratch
import numpy as np
from .preprocessing import to_pixels


def segment_full_resolution(original, model, colorspace="RGB", batch_size=32768):
    """Assign original pixels in bounded batches using the learned color centers."""
    array = np.asarray(original.convert("RGB"))
    pixels = array.reshape(-1, 3)
    output = np.empty_like(pixels)
    for start in range(0, len(pixels), batch_size):
        batch = pixels[start:start + batch_size].reshape(-1, 1, 3)
        labels = model.predict(to_pixels(batch, colorspace))
        output[start:start + len(batch)] = reconstruct_image(
            model.cluster_centers_[labels], batch.shape, colorspace).reshape(-1, 3)
    return output.reshape(array.shape)


def segment_image(X, shape, k, settings):
    model = KMeansScratch(n_clusters=int(k), max_iter=settings["max_iter"],
                   tol=settings["tol"], random_state=settings["random_state"],
                   n_init=settings["n_init"]).fit(X)
    reconstructed = model.cluster_centers_[model.labels_]
    image = reconstruct_image(reconstructed, shape, settings.get("colorspace", "RGB"))
    return model, image
