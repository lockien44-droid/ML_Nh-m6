"""Run reproducible customer clustering and retain per-run artifacts."""
import json
import shutil
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from time import perf_counter
import joblib
import pandas as pd
import yaml
from .preprocessing import LAB_DIR, prepare_data
from .eda import analyze
from .tuning import tune
from .train import train_model
from .evaluate import cluster_profiles, silhouette_samples
from .visualization import (plot_comparison, plot_clusters, plot_income_spending,
                            plot_silhouette)
from .analysis import describe_segments, gender_distribution, stability_analysis


def run_pipeline():
    config = yaml.safe_load((LAB_DIR / "src/config.yaml").read_text(encoding="utf-8"))
    started = perf_counter()
    now = datetime.now(timezone(timedelta(hours=7)))
    base_id = now.strftime("%Y-%m-%d_%H-%M-%S")
    sequence = 1
    while True:
        run_id = base_id if sequence == 1 else f"{base_id}_{sequence:02d}"
        run = LAB_DIR / "results/runs" / run_id
        try:
            run.mkdir(parents=True)
            break
        except FileExistsError:
            sequence += 1
    figures = run / "figures"
    figures.mkdir()
    for name in ["model", "clusters", "k_selection", "validation"]:
        (run / name).mkdir()
    manifest = {"run_id": run_id, "status": "running", "timestamp_timezone": "Asia/Ho_Chi_Minh", "started_at": now.isoformat()}
    def log(stage, status="started"):
        event = {"stage": stage, "status": status,
                 "time": datetime.now(timezone(timedelta(hours=7))).isoformat()}
        with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    (run / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        log("run")
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        source = LAB_DIR / config["data"]["path"]
        manifest["dataset_sha256"] = sha256(source.read_bytes()).hexdigest()
        raw = pd.read_csv(source)
        features = config["data"]["features"]
        log("eda_and_preprocessing")
        analyze(raw, features, figures)
        frame, X, scaling = prepare_data(config, run / "preprocessing")
        log("k_selection")
        model, comparison = tune(X, config["kmeans"])
        comparison.to_csv(run / "k_selection/k_comparison.csv", index=False)
        customers = frame.assign(cluster=model.labels_, silhouette=silhouette_samples(X, model.labels_))
        profiles = cluster_profiles(frame, model.labels_, features)
        profiles = describe_segments(profiles, features)
        customers = customers.merge(profiles[["cluster","segment"]],on="cluster",validate="many_to_one")
        customers.to_csv(run / "clusters/customer_clusters.csv", index=False)
        profiles.to_csv(run / "clusters/cluster_profiles.csv", index=False)
        for alternate_k in (5,9):
            if alternate_k < len(X):
                alternative = train_model(X,alternate_k,config["kmeans"])
                pd.crosstab(alternative.labels_,model.labels_,rownames=[f"K{alternate_k}"],
                            colnames=[f"K{model.n_clusters}"]).to_csv(run / "k_selection" / f"k{alternate_k}_vs_selected.csv")
        gender_distribution(customers).to_csv(run / "clusters/gender_by_cluster.csv",index=False)
        plot_income_spending(customers,features,figures / "income_spending.png")
        plot_silhouette(customers,figures / "silhouette_clusters.png")
        log("seed_stability")
        print("Checking multiple seeds...",flush=True)
        stability, pairs = stability_analysis(X,config["kmeans"],config["analysis"]["seeds"],
                                             model.labels_,model.n_clusters)
        stability.to_csv(run / "validation/seed_stability.csv",index=False)
        pairs.to_csv(run / "validation/pairwise_ari.csv",index=False)
        log("visualization_and_export")
        plot_comparison(comparison, figures / "silhouette_by_k.png")
        plot_clusters(customers, features, config["data"]["id_column"],
                      figures / "customer_clusters_3d.html")
        joblib.dump(model, run / "model/kmeans.joblib")
        # A stable convenience copy, alongside the per-run original.
        destination = LAB_DIR / "Data/preprocessing"
        destination.mkdir(parents=True, exist_ok=True)
        for path in (run / "preprocessing").glob("*.csv"):
            shutil.copy2(path, destination / path.name)
        manifest.update(status="success", raw_rows=len(raw), eligible_rows=len(frame), excluded_rows=len(raw)-len(frame), selected_k=model.n_clusters,
                        silhouette=float(customers.silhouette.mean()))
        print(comparison.to_string(index=False))
        print("Selected K:", model.n_clusters)
        print("Run:", run)
        log("run", "success")
        return run
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        log("run", "failed")
        raise
    finally:
        manifest.update(finished_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),
                        duration_seconds=perf_counter()-started)
        (run / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_pipeline()
