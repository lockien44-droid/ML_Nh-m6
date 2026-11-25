"""Validate customer records and standardize the selected numerical features."""
import numpy as np
import pandas as pd

from ..config import LAB_DIR


def build_features(frame, config):
    """Validate records, normalize Gender for profiles and select numerical features."""
    frame = frame.copy()
    features = config["data"]["features"]
    identifier = config["data"]["id_column"]
    if len(features) != 3 or len(set(features)) != 3 or identifier in features:
        raise ValueError("Select three distinct numerical features, excluding CustomerID.")
    if frame[identifier].isna().any() or frame[identifier].duplicated().any():
        raise ValueError("Customer identifiers must be present and unique.")
    if "Gender" in frame:
        frame["Gender"] = frame["Gender"].astype("string").str.strip().str.lower().map({"male": "Male", "female": "Female"})
        if frame["Gender"].isna().any():
            raise ValueError("Gender must be Male or Female for demographic profiles.")
    numeric = frame[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Features contain missing or infinite values. Review the EDA.")
    age, income, spending = features
    valid = (numeric[age].between(1,100)
             & numeric[income].gt(0) & numeric[spending].between(1,100))
    if not valid.all():
        raise ValueError("Review invalid age, income or spending score before clustering.")
    return frame, numeric


def prepare_data(config, output):
    frame, numeric = build_features(pd.read_csv(LAB_DIR / config["data"]["path"]), config)
    features = config["data"]["features"]
    identifier = config["data"]["id_column"]
    scaler = fit_scaler(numeric)
    means, scales = scaler["mean"], scaler["scale"]
    scaled = (numeric - means) / scales
    exported = pd.concat([frame[[identifier]], scaled], axis=1)
    output.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output / "customers_scaled.csv", index=False)
    frame.to_csv(output / "customers_clean.csv", index=False)
    import joblib
    (output.parent / "models").mkdir(exist_ok=True)
    joblib.dump({"features": features, "mean": means.to_numpy(), "scale": scales.to_numpy()}, output.parent / "models/scaler.joblib")
    scaling = pd.DataFrame({"feature": features, "mean": means.to_numpy(), "scale": scales.to_numpy()})
    scaling.to_csv(output / "scaling_parameters.csv", index=False)
    return frame, scaled.to_numpy(dtype=float), scaling


def fit_scaler(features):
    """Learn population means/scales, keeping constant columns finite."""
    return {"mean": features.mean(), "scale": features.std(ddof=0).replace(0, 1)}
