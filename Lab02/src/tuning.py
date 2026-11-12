"""Choose minimum validation RMSE independently for every model."""
import numpy as np
import pandas as pd
from .preprocessing import FEATURES
from .train import build_model
from .evaluate import metrics


def tune(train, validation, config, output):
    rows, selected = [], {}
    for name, candidates in config["models"].items():
        best = float("inf")
        for parameters in candidates:
            model = build_model(name, parameters, config["seed"])
            model.fit(train[FEATURES], train.sales)
            score = metrics(validation.sales, np.maximum(0, model.predict(validation[FEATURES])))
            rows.append({"model": name, "parameters": str(parameters), **score})
            if score["rmse"] < best:
                best = score["rmse"]
                selected[name] = dict(parameters)
    pd.DataFrame(rows).to_csv(output / "validation_metrics.csv", index=False)
    return selected
