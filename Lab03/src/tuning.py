"""Compare K by mean Silhouette, with smaller K as the tie-break."""
import pandas as pd
from .train import train_model
from .evaluate import silhouette_samples


def tune(X, settings):
    candidates = settings["k_values"]
    if not candidates or any(not isinstance(k, int) or not 2 <= k < len(X) for k in candidates):
        raise ValueError("K candidates must be integers between 2 and n-1.")
    rows, models = [], {}
    for k in sorted(set(candidates)):
        model = train_model(X, k, settings)
        scores = silhouette_samples(X, model.labels_)
        rows.append(dict(k=k, silhouette=float(scores.mean()), inertia=model.inertia_,
                         n_iter=model.n_iter_, converged=model.converged_))
        models[k] = model
    comparison = pd.DataFrame(rows)
    selected = comparison.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return models[int(selected["k"])], comparison
