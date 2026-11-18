"""Run reproducible customer clustering and retain per-run artifacts."""
import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4
import joblib
import pandas as pd
import yaml
from .preprocessing import LAB_DIR, prepare_data
from .eda import analyze
from .tuning import tune
from .evaluate import cluster_profiles, silhouette_samples
from .visualization import plot_comparison, plot_clusters


def run_pipeline():
    config = yaml.safe_load((LAB_DIR / "src/config.yaml").read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run = LAB_DIR / "results/runs" / run_id
    figures = run / "figures"
    figures.mkdir(parents=True)
    manifest = {"run_id": run_id, "status": "running", "timestamp_timezone": "UTC"}
    try:
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        source = LAB_DIR / config["data"]["path"]
        manifest["dataset_sha256"] = sha256(source.read_bytes()).hexdigest()
        raw = pd.read_csv(source)
        features = config["data"]["features"]
        analyze(raw, features, figures)
        frame, X, scaling = prepare_data(config, run / "preprocessing")
        model, comparison = tune(X, config["kmeans"])
        comparison.to_csv(run / "k_comparison.csv", index=False)
        customers = frame.assign(cluster=model.labels_, silhouette=silhouette_samples(X, model.labels_))
        customers.to_csv(run / "customer_clusters.csv", index=False)
        profiles = cluster_profiles(frame, model.labels_, features)
        profiles.to_csv(run / "cluster_profiles.csv", index=False)
        plot_comparison(comparison, figures / "silhouette_by_k.png")
        plot_clusters(customers, features, config["data"]["id_column"],
                      figures / "customer_clusters_3d.html")
        joblib.dump(model, run / "kmeans.joblib")
        # A stable convenience copy, alongside the per-run original.
        destination = LAB_DIR / "Data/preprocessing"
        destination.mkdir(parents=True, exist_ok=True)
        for path in (run / "preprocessing").glob("*.csv"):
            shutil.copy2(path, destination / path.name)
        manifest.update(status="success", selected_k=model.n_clusters,
                        silhouette=float(customers.silhouette.mean()))
        print(comparison.to_string(index=False))
        print("Selected K:", model.n_clusters)
        print("Run:", run)
        return run
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        (run / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_pipeline()
