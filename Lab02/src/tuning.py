"""Choose minimum validation RMSE independently for every model."""
import numpy as np
import pandas as pd
from .preprocessing import FEATURES
from .train import train_model
from .evaluate import metrics, evaluate


def tune(train, validation, config, output):
    # Expanding windows use complete dates so products on one day stay together.
    dates = np.sort(train["date"].unique())
    blocks = np.array_split(dates, 4)
    folds = []
    for index in range(1, 4):
        past = train.loc[train["date"].isin(np.concatenate(blocks[:index]))]
        future = train.loc[train["date"].isin(blocks[index])]
        folds.append((past, future))
    rows, fold_rows, selected = [], [], {}
    for name, candidates in config["models"].items():
        best = float("inf")
        best_prediction = None
        for number, parameters in enumerate(candidates, 1):
            print(f"tuning {name} {number}/{len(candidates)}", flush=True)
            model = train_model(name, parameters, config["seed"], train[FEATURES], train.sales)
            prediction = np.maximum(0, model.predict(validation[FEATURES]))
            score = metrics(validation.sales, prediction)
            rows.append({"model": name, "parameters": str(parameters), **score})
            fold_scores = []
            for fold, (past, future) in enumerate(folds, 1):
                candidate = train_model(name, parameters, config["seed"], past[FEATURES], past.sales)
                result = metrics(future.sales, np.maximum(0, candidate.predict(future[FEATURES])))
                fold_scores.append(result["rmse"])
                fold_rows.append({"model": name, "parameters": str(parameters), "fold": fold,
                                  "train_end": str(past.date.max()),
                                  "validation_start": str(future.date.min()), **result})
            # Keep the held-out validation in selection, weight periods equally.
            selection_rmse = float(np.mean([*fold_scores, score["rmse"]]))
            rows[-1]["selection_rmse"] = selection_rmse
            rows[-1]["walk_forward_rmse_std"] = float(np.std(fold_scores))
            if selection_rmse < best:
                best = selection_rmse
                selected[name] = dict(parameters)
                best_prediction = prediction.copy()
        export_columns = dict(entity_column="sku", target_column="Revenue") if config["data"].get("schema") == "fmcg_weekly" else {}
        evaluate(validation, best_prediction, output / f"{name}_validation.csv", **export_columns)
    pd.DataFrame(rows).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "walk_forward_metrics.csv", index=False)
    return selected


def save_best_config(run):
    """Publish the validation-selected configurations at one stable location."""
    import json
    from uuid import uuid4
    from .preprocessing import LAB
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if manifest["status"] not in ("tuned", "trained", "success"):
        raise ValueError("Publish the best configuration only after successful tuning.")
    selection = json.loads((run / "tuning/selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "preprocessing/summary.json").read_text(encoding="utf-8"))
    trials = pd.read_csv(run / "tuning/validation_metrics.csv")
    best = trials.sort_values("selection_rmse", kind="stable").drop_duplicates("model")
    config = {
        "run_id": run.name,
        "winner": best.iloc[0]["model"],
        "criterion": "mean_rmse_three_expanding_folds_and_validation",
        "dataset_sha256": summary["dataset_sha256"],
        "source": summary["source"],
        "frequency": summary["frequency"],
        "target": summary["target"],
        "models": selection,
        "selection_rmse": dict(zip(best.model, best.selection_rmse)),
    }
    destination = LAB / "best_config.json"
    temporary = LAB / f".best_config_{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("Best configuration:", destination, flush=True)
    return config


def tune_run(run):
    """Select configurations without reading the test split."""
    import json
    import yaml
    from .experiment_tracking import load_split, run_stage
    with run_stage(run, "tuning"):
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        selected = tune(load_split(run, "train"), load_split(run, "validation"),
                        config, run / "tuning")
        (run / "tuning/selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    save_best_config(run)
    print("Tuned run:", run.name, flush=True)


def main():
    from .experiment_tracking import stage_main
    stage_main("Choose Lab02 configurations using development data.", tune_run)


if __name__ == "__main__":
    main()
