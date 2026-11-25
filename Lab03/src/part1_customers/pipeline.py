"""Prepare customers, tune, fit the chosen model, then evaluate saved artifacts."""
from hashlib import sha256
import shutil
import joblib
import pandas as pd
from ..config import LAB_DIR
from ..utils import read_json
from .preprocessing import prepare_data
from .eda import analyze
from .tuning import tune_run
from .train import train_run, train_model
from .evaluate import evaluate_run
from .analysis import stability_analysis


def run_part(run, config, log):
    for name in ("models", "results", "tuning", "figures"):
        (run / name).mkdir(exist_ok=True)
    source = LAB_DIR / config["data"]["path"]
    raw = pd.read_csv(source)
    log("eda_and_preprocessing")
    analyze(raw, config["data"]["features"], run / "figures")
    frame, X, scaling = prepare_data(config, run / "preprocessing")
    log("tuning")
    tune_run(run)
    log("final_training")
    train_run(run)
    log("evaluation")
    metadata = evaluate_run(run)
    selection = read_json(run / "tuning/selection.json")
    if selection["algorithm"] == "kmeans":
        model = joblib.load(run / "models/clustering.joblib")
        for k in (5, 9):
            if k < len(X):
                alternative = train_model(X, k, config["kmeans"])
                pd.crosstab(alternative.labels_, model.labels_, rownames=[f"K{k}"],
                    colnames=[f"K{model.n_clusters}"]).to_csv(run / "tuning" / f"k{k}_vs_selected.csv")
        log("seed_stability")
        settings = {**config["kmeans"], "algorithms": ["kmeans"]}
        stability, pairs = stability_analysis(X, settings, config["analysis"]["seeds"],
                                             model.labels_, model.n_clusters)
        stability.to_csv(run / "results/seed_stability.csv", index=False)
        pairs.to_csv(run / "results/pairwise_ari.csv", index=False)
    destination = LAB_DIR / "data/customers/processed"
    destination.mkdir(parents=True, exist_ok=True)
    for path in (run / "preprocessing").glob("*.csv"):
        shutil.copy2(path, destination / path.name)
    metadata.update(dataset_sha256=sha256(source.read_bytes()).hexdigest(),
                    raw_rows=len(raw), eligible_rows=len(frame), excluded_rows=len(raw)-len(frame))
    return metadata


def main():
    from ..pipeline import main as run
    run(default_part="1")


if __name__ == "__main__":
    main()
