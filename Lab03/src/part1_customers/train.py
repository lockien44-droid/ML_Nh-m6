"""Fit a selected clustering configuration and save its training partition labels."""
import json
from hashlib import sha256
import joblib
import numpy as np
import pandas as pd
from ...algorithms.kmeans import KMeans


def train_model(X, k, settings):
    return KMeans(n_clusters=int(k), max_iter=settings["max_iter"], tol=settings["tol"],
                  random_state=settings["random_state"], n_init=settings.get("n_init", 20)).fit(X)


def fit_candidate(X, algorithm, parameters, settings):
    if algorithm == "kmeans":
        model = train_model(X, parameters["k"], settings)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    model.algorithm_ = algorithm
    model.parameters_ = dict(parameters)
    return model


def load_features(run):
    config = json.loads((run / "config_used.json").read_text(encoding="utf-8"))
    data = pd.read_csv(run / "preprocessing/customers_scaled.csv", float_precision="round_trip")
    return data[config["data"]["features"]].to_numpy(dtype=float), config


def train_run(run):
    """Fit the already selected configuration; tuning is not repeated."""
    from ..utils import read_json, write_json
    X, config = load_features(run)
    selected = read_json(run / "tuning/selection.json")
    model = fit_candidate(X, selected.get("algorithm", "kmeans"), selected["parameters"], config["kmeans"])
    directory = run / "models"
    directory.mkdir(exist_ok=True)
    joblib.dump(model, directory / "clustering.joblib")
    if selected["algorithm"] == "kmeans":
        joblib.dump(model, directory / "kmeans.joblib")
    np.save(directory / "labels.npy", model.labels_)
    write_json(directory / "training.json", dict(algorithm=selected["algorithm"],
        features=config["data"]["features"], feature_sha256=sha256(X.tobytes()).hexdigest(),
        selection=selected))
    return dict(algorithm=selected["algorithm"], trained_rows=len(X))


def main():
    from ..utils import customer_step_main
    customer_step_main(train_run, "training")


if __name__ == "__main__":
    main()
