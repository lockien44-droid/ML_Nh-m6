"""Run all Lab02 experiments from the project root."""
import json
import shutil
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import yaml
from .preprocessing import LAB, FEATURES, prepare_data
from .experiment_tracking import training_run
from .eda import analyze
from .tuning import tune
from .train import train_model
from .evaluate import evaluate
from .report import generate


def main():
    config = yaml.safe_load((LAB / "src/config.yaml").read_text())
    with training_run(LAB / "results/runs", LAB / config["data"]["path"]) as (run, log):
        for folder in ["preprocessing", "tuning", "evaluation", "figures", "models"]:
            (run / folder).mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        log("preprocessing")

        # Thư mục lưu dữ liệu đã xử lý
        processed_dir = LAB / "data" / "preprocessing"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Chia dữ liệu và lưu train.csv, validation.csv, test.csv
        train, validation, test, summary = prepare_data(config, run / "preprocessing")

        # Lưu thông tin tóm tắt cùng dữ liệu
        summary_text = json.dumps(summary, indent=2)
        (processed_dir / "summary.json").write_text(
            summary_text, encoding="utf-8"
        )

        # Giữ thông tin tóm tắt cho báo cáo của lần chạy
        (run / "preprocessing" / "summary.json").write_text(
            summary_text, encoding="utf-8"
        )
        analyze(pd.concat([train, validation, test]), run / "figures")
        log("tuning")
        selection = tune(train, validation, config, run / "tuning")
        (run / "tuning/selection.json").write_text(json.dumps(selection, indent=2))
        validation_rows = pd.read_csv(run / "tuning/validation_metrics.csv")
        winner = validation_rows.loc[validation_rows.selection_rmse.idxmin(), "model"]
        development = pd.concat([train, validation])
        rows = []
        log("final_training")
        for name, parameters in selection.items():
            started = perf_counter()
            model = train_model(name, parameters, config["seed"], development[FEATURES], development.sales)
            elapsed = perf_counter() - started
            predicted = np.maximum(0, model.predict(test[FEATURES]))
            score = evaluate(test, predicted, run / "evaluation" / f"{name}.csv")
            rows.append({"model": name, "fit_seconds": elapsed, **score})
            joblib.dump(model, run / "models" / f"{name}.joblib")
        for name, predicted in [("mean_baseline", np.repeat(development.sales.mean(), len(test))),
                                ("last_week_baseline", test.lag_7.to_numpy()),
                                ("rolling_7_baseline", test.mean_7.to_numpy()),
                                ("rolling_28_baseline", test.mean_28.to_numpy()),
                                ("weekday_4_baseline", test.same_weekday_mean_4.to_numpy()),
                                ("product_mean_baseline", test.StockCode.map(development.groupby("StockCode").sales.mean()).to_numpy())]:
            rows.append({"model": name, "fit_seconds": 0,
                         **evaluate(test, predicted, run / "evaluation" / f"{name}.csv")})
        # Compare each product with history-only baselines without selecting on test.
        comparison = pd.read_csv(run / "evaluation" / f"{winner}_StockCode_metrics.csv", dtype={"StockCode": str})
        for baseline in ("mean_baseline", "rolling_7_baseline", "rolling_28_baseline", "weekday_4_baseline"):
            baseline_scores = pd.read_csv(run / "evaluation" / f"{baseline}_StockCode_metrics.csv", dtype={"StockCode": str})
            comparison = comparison.merge(baseline_scores[["StockCode", "rmse"]].rename(columns={"rmse": baseline + "_rmse"}), on="StockCode", validate="one_to_one")
            comparison[baseline + "_improvement_pct"] = 100 * (1 - comparison.rmse / comparison[baseline + "_rmse"].replace(0, np.nan))
        comparison.to_csv(run / "evaluation/product_baseline_comparison.csv", index=False)
        results = pd.DataFrame(rows)
        results.to_csv(run / "evaluation/metrics.csv", index=False)
        log("reporting")
        report = generate(run, results, selection)
        (run / "report.md").write_text(report.replace(f"../results/runs/{run.name}/figures/", "figures/"), encoding="utf-8")
        for filename in ("train.csv", "validation.csv", "test.csv"):
            shutil.copy2(run / "preprocessing" / filename, processed_dir / filename)
        for path in (run / "models").glob("*.joblib"):
            shutil.copy2(path, LAB / "models" / path.name)
        (LAB / "models/selection.json").write_text(json.dumps({"winner": winner, "run_id": run.name}))
        (LAB / "models/products.json").write_text(json.dumps(summary["products"]))
        log("complete")
        print(results.to_string(index=False))
        print("Run:", run)


if __name__ == "__main__":
    main()
