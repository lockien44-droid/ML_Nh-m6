"""Choose minimum validation RMSE independently for every model."""
import numpy as np
import pandas as pd
from .preprocessing import FEATURES
from .train import build_model
from .evaluate import metrics


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
        for parameters in candidates:
            model = build_model(name, parameters, config["seed"])
            model.fit(train[FEATURES], train.sales)
            score = metrics(validation.sales, np.maximum(0, model.predict(validation[FEATURES])))
            rows.append({"model": name, "parameters": str(parameters), **score})
            fold_scores = []
            for fold, (past, future) in enumerate(folds, 1):
                candidate = build_model(name, parameters, config["seed"])
                candidate.fit(past[FEATURES], past.sales)
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
    pd.DataFrame(rows).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "walk_forward_metrics.csv", index=False)
    return selected
