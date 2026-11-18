"""Inspect missingness, feature distributions and IQR outlier flags."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def analyze(frame, features, output):
    output.mkdir(parents=True, exist_ok=True)
    frame.isna().sum().rename("missing_count").to_csv(output / "missing_values.csv")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    numeric.describe().to_csv(output / "feature_statistics.csv")
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    flagged = (numeric < q1 - 1.5 * (q3 - q1)) | (numeric > q3 + 1.5 * (q3 - q1))
    flags = pd.DataFrame({"feature":features, "iqr_outlier_count":flagged.sum().to_numpy()})
    flags.to_csv(output / "outlier_summary.csv", index=False)
    fig = Figure(figsize=(12, 7), layout="tight")
    axes = fig.subplots(2, 3)
    for column, feature in enumerate(features):
        values = numeric[feature].dropna()
        axes[0, column].hist(values, bins=15, color="#287c8e")
        axes[0, column].set(title=feature, ylabel="Customers")
        axes[1, column].boxplot(values)
        axes[1, column].set(title=f"IQR flags: {int(flagged[feature].sum())}")
    fig.savefig(output / "eda_overview.png", dpi=160)
    return flags
