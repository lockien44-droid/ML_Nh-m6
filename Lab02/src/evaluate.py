"""Regression metrics and per-row predictions."""
import numpy as np
import pandas as pd


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


def evaluate(frame, predicted, output):
    result = frame[["StockCode", "date", "sales"]].copy()
    result["predicted"] = predicted
    result["residual"] = result["sales"] - predicted
    result.to_csv(output, index=False)
    return metrics(frame.sales, predicted)
