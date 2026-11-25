"""Regression metrics and per-row predictions."""
import numpy as np
import pandas as pd
from .evaluation_layout import evaluation_path, organize_evaluation


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual and predicted must be aligned, nonempty vectors.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted must contain finite values.")
    residual = actual - predicted
    squared_error = float(np.sum(residual ** 2))
    mse = squared_error / actual.size
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    # R-squared is undefined for a single observation.
    r2 = float("nan")
    if actual.size >= 2:
        # Constant targets score 1 for perfect prediction and 0 otherwise.
        r2 = 1.0 if squared_error == 0 else 0.0
        if total_variation > 0:
            r2 = 1.0 - squared_error / total_variation
    return {"mae": float(np.mean(np.abs(residual))),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2}


def evaluate(frame, predicted, output, entity_column="StockCode", target_column="sales"):
    export_columns = {"StockCode": entity_column, "sales": target_column}
    result = frame[["StockCode", "date", "sales"]].copy()
    result["predicted"] = predicted
    result["residual"] = result["sales"] - predicted
    result.rename(columns=export_columns).to_csv(output, index=False)
    result["absolute_error"] = np.abs(result["residual"])
    result.nlargest(10, "absolute_error").rename(columns=export_columns).to_csv(
        output.with_name(output.stem + "_largest_errors.csv"), index=False)
    for group in ("StockCode", "weekday"):
        grouped = result.copy()
        if group == "weekday":
            grouped[group] = pd.to_datetime(grouped["date"]).dt.dayofweek
        rows = []
        for value, part in grouped.groupby(group):
            rows.append({group: value, "samples": len(part),
                         "mean_residual": float(part.residual.mean()),
                         **metrics(part.sales, part.predicted)})
        export_group = export_columns.get(group, group)
        pd.DataFrame(rows).rename(columns=export_columns).to_csv(output.with_name(output.stem + "_" + export_group + "_metrics.csv"), index=False)
    return metrics(frame.sales, predicted)


def evaluate_run(run):
    """Load saved models and evaluate test without fitting or tuning."""
    import json
    import shutil
    import joblib
    from .preprocessing import LAB, FEATURES
    from .experiment_tracking import load_split, run_stage
    from .report import generate

    with run_stage(run, "evaluation"):
        organize_evaluation(run / "evaluation")
        summary = json.loads((run / "preprocessing/summary.json").read_text(encoding="utf-8"))
        selection = json.loads((run / "tuning/selection.json").read_text(encoding="utf-8"))
        trials = pd.read_csv(run / "tuning/validation_metrics.csv")
        winner = trials.loc[trials.selection_rmse.idxmin(), "model"]
        metadata_path = run / "models/selection.json"
        timings = json.loads(metadata_path.read_text()).get("fit_seconds", {}) if metadata_path.exists() else {}
        # Older completed runs recorded final fit times in their metrics table.
        metrics_path = evaluation_path(run / "evaluation", "metrics.csv")
        if not timings and metrics_path.exists():
            previous = pd.read_csv(metrics_path).set_index("model")
            timings = previous.fit_seconds.to_dict()
        development = pd.concat([load_split(run, "train"), load_split(run, "validation")])
        test = load_split(run, "test")
        entity = "sku" if summary.get("revenue_definition", "").startswith("price_unit") else "StockCode"
        export_columns = dict(entity_column=entity, target_column="Revenue" if entity == "sku" else "sales")
        rows = []
        for name in selection:
            model = joblib.load(run / "models" / f"{name}.joblib")
            predicted = np.maximum(0, model.predict(test[FEATURES]))
            score = evaluate(test, predicted, evaluation_path(run / "evaluation", f"{name}.csv"), **export_columns)
            rows.append(dict(model=name, fit_seconds=timings.get(name, 0), **score))
        baselines = [
            ("mean_baseline", np.repeat(development.sales.mean(), len(test))),
            ("last_week_baseline", test.lag_7.to_numpy()),
            ("rolling_7_baseline", test.mean_7.to_numpy()),
            ("rolling_28_baseline", test.mean_28.to_numpy()),
            ("weekday_4_baseline", test.same_weekday_mean_4.to_numpy()),
            ("product_mean_baseline", test.StockCode.map(development.groupby("StockCode").sales.mean()).to_numpy()),
        ]
        for name, predicted in baselines:
            score = evaluate(test, predicted, evaluation_path(run / "evaluation", f"{name}.csv"), **export_columns)
            rows.append(dict(model=name, fit_seconds=0, **score))
        comparison = pd.read_csv(evaluation_path(run / "evaluation", f"{winner}_{entity}_metrics.csv"), dtype={entity: str})
        for baseline in ("mean_baseline", "rolling_7_baseline", "rolling_28_baseline", "weekday_4_baseline"):
            scores = pd.read_csv(evaluation_path(run / "evaluation", f"{baseline}_{entity}_metrics.csv"), dtype={entity: str})
            comparison = comparison.merge(scores[[entity, "rmse"]].rename(
                columns={"rmse": baseline + "_rmse"}), on=entity, validate="one_to_one")
            comparison[baseline + "_improvement_pct"] = 100 * (
                1 - comparison.rmse / comparison[baseline + "_rmse"].replace(0, np.nan))
        comparison.to_csv(evaluation_path(run / "evaluation", "product_baseline_comparison.csv"), index=False)
        results = pd.DataFrame(rows)
        results.to_csv(metrics_path, index=False)
        index = ["# Kết quả đánh giá", "", "| Nhóm | Tệp |", "|---|---|"]
        for path in sorted((run / "evaluation").rglob("*.csv")):
            relative = path.relative_to(run / "evaluation").as_posix()
            index.append(f"| {relative.split('/')[0]} | [{path.name}]({relative}) |")
        (run / "evaluation/README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
        report = generate(run, results, selection)
        (run / "report.md").write_text(
            report.replace(f"../results/runs/{run.name}/figures/", "figures/"), encoding="utf-8")
        processed = LAB / "data/preprocessing"
        processed.mkdir(parents=True, exist_ok=True)
        for filename in ("train.csv", "validation.csv", "test.csv", "summary.json"):
            shutil.copy2(run / "preprocessing" / filename, processed / filename)
        published = LAB / "models"
        published.mkdir(exist_ok=True)
        for path in (run / "models").glob("*.joblib"):
            shutil.copy2(path, published / path.name)
        (published / "products.json").write_text(json.dumps(summary["products"]), encoding="utf-8")
        metadata = dict(winner=winner, run_id=run.name, frequency=summary["frequency"],
                        target=summary["target"], unit=summary["unit"])
        temporary = published / "selection.tmp"
        temporary.write_text(json.dumps(metadata), encoding="utf-8")
        temporary.replace(published / "selection.json")
    print(results.to_string(index=False))
    print("Evaluated run:", run.name, flush=True)
    return results


def main():
    from .experiment_tracking import stage_main
    stage_main("Evaluate saved Lab02 models on test without tuning or fitting.", evaluate_run, use_best_config=True)


if __name__ == "__main__":
    main()
