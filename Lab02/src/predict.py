"""Predict next-day units from explicit past-only predictors."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from .preprocessing import LAB, FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON object containing StockCode, date and lag features")
    parser.add_argument("--model", choices=["linear_regression", "decision_tree", "random_forest", "gradient_boosting"])
    args = parser.parse_args()
    sample = json.loads(args.input)
    if sample["StockCode"] not in json.loads((LAB / "models/products.json").read_text()):
        parser.error("Product was not in the trained product scope.")
    day = pd.Timestamp(sample["date"])
    if pd.isna(day):
        parser.error("date must be a valid date.")
    sample.update(month=day.month, weekday=day.dayofweek, day_of_year=day.dayofyear)
    frame = pd.DataFrame([sample])
    values = frame[FEATURES[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] < 0).any():
        parser.error("Lag and rolling features must be finite and nonnegative.")
    selected = json.loads((LAB / "models/selection.json").read_text())
    name = args.model or selected["winner"]
    model = joblib.load(LAB / "models" / f"{name}.joblib")
    print(json.dumps({"model": name, "predicted_units": float(max(0, model.predict(frame[FEATURES])[0]))}))


if __name__ == "__main__":
    main()
