"""Validate customer records and standardize the selected numerical features."""
from pathlib import Path
import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parents[1]


def prepare_data(config, output):
    frame = pd.read_csv(LAB_DIR / config["data"]["path"])
    features = config["data"]["features"]
    identifier = config["data"]["id_column"]
    if len(features) != 3 or len(set(features)) != 3 or identifier in features:
        raise ValueError("Select three distinct numerical features, excluding CustomerID.")
    if frame[identifier].isna().any() or frame[identifier].duplicated().any():
        raise ValueError("Customer identifiers must be present and unique.")
    numeric = frame[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or infinite values. Review the EDA.")
    means = numeric.mean()
    scales = numeric.std(ddof=0).replace(0, 1)
    scaled = (numeric - means) / scales
    exported = pd.concat([frame[[identifier]], scaled], axis=1)
    output.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output / "customers_scaled.csv", index=False)
    scaling = pd.DataFrame({"feature": features, "mean": means.to_numpy(), "scale": scales.to_numpy()})
    scaling.to_csv(output / "scaling_parameters.csv", index=False)
    return frame, scaled.to_numpy(dtype=float), scaling
