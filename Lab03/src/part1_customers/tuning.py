"""Scan clustering algorithms and choose maximum covered-data Silhouette."""
import json
import pandas as pd
from .train import fit_candidate, load_features
from .evaluate import score_clustering


def candidates(settings):
    for algorithm in settings.get("algorithms", ["kmeans"]):
        if algorithm == "kmeans":
            for k in sorted(set(settings["k_values"])):
                yield algorithm, {"k": int(k)}
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


def tune(X, settings):
    if not settings["k_values"] or any(not isinstance(k, int) or not 2 <= k < len(X) for k in settings["k_values"]):
        raise ValueError("K candidates must be integers between 2 and n-1.")
    rows, models = [], []
    for algorithm, parameters in candidates(settings):
        model = fit_candidate(X, algorithm, parameters, settings)
        score = score_clustering(X, model.labels_)
        rows.append(dict(algorithm=algorithm, parameters=json.dumps(parameters, sort_keys=True),
                         n_iter=getattr(model, "n_iter_", 0),
                         converged=getattr(model, "converged_", True), **score))
        models.append(model)
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise ValueError("No candidate algorithms configured.")
    eligible = comparison.loc[comparison.silhouette.notna() &
                              comparison.coverage.ge(settings.get("min_coverage", 0.8))]
    if eligible.empty:
        raise ValueError("No candidate has valid clusters and sufficient non-noise coverage.")
    selected = eligible.sort_values(["silhouette", "coverage", "k"],
                                   ascending=[False, False, True], kind="stable").iloc[0]
    return models[int(selected.name)], comparison


def tune_run(run):
    """Scan settings on prepared customer features and save selection.json."""
    from ..utils import write_json
    if (run / "models/clustering.joblib").exists():
        raise ValueError("Create a new run to retune a configuration after final training.")
    X, config = load_features(run)
    model, comparison = tune(X, config["kmeans"])
    directory = run / "tuning"
    directory.mkdir(exist_ok=True)
    comparison.to_csv(directory / "metrics_by_k.csv", index=False)
    row = comparison.loc[(comparison.algorithm == model.algorithm_) &
        comparison.parameters.eq(json.dumps(model.parameters_, sort_keys=True))].iloc[0]
    selected = dict(algorithm=model.algorithm_, k=int(row.k),
                    parameters=json.loads(row.parameters), settings=config["kmeans"],
                    criterion="maximum_silhouette_with_minimum_coverage",
                    silhouette=float(row.silhouette), coverage=float(row.coverage))
    write_json(directory / "selection.json", selected)
    from .visualization import plot_comparison
    plot_comparison(comparison, run / "figures/silhouette_by_k.png")
    return dict(selected_algorithm=selected["algorithm"], selected_k=selected["k"])


def main():
    from ..utils import customer_step_main
    customer_step_main(tune_run, "tuning")


if __name__ == "__main__":
    main()
