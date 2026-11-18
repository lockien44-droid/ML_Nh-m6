"""Predict next-day units from explicit past-only predictors."""
import argparse
import csv
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from .preprocessing import LAB, FEATURES


def sample_from_history(path, product, date):
    history = pd.read_csv(path, dtype={"StockCode": str})
    day = pd.Timestamp(date).normalize()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history = history.loc[(history.StockCode == product) & (history.date < day)]
    if history.duplicated(["StockCode", "date"]).any():
        raise ValueError("History must have one row per product/day.")
    dates = pd.date_range(day - pd.Timedelta(days=28), periods=28)
    sales = pd.to_numeric(history.set_index("date")["sales"], errors="coerce").reindex(dates)
    if sales.isna().any() or not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Provide 28 consecutive past days with finite nonnegative sales.")
    return {"StockCode": product, "date": str(day.date()), "lag_1": float(sales.iloc[-1]),
            "lag_7": float(sales.iloc[-7]), "mean_7": float(sales.iloc[-7:].mean()),
            "mean_28": float(sales.mean()),
            "median_7": float(sales.iloc[-7:].median()), "median_28": float(sales.median()),
            "std_7": float(sales.iloc[-7:].std(ddof=0)), "std_28": float(sales.std(ddof=0)),
            "zero_days_28": int((sales == 0).sum()),
            "same_weekday_mean_4": float(sales.iloc[[0, 7, 14, 21]].mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="JSON object containing StockCode, date and lag features")
    parser.add_argument("--model", choices=["linear_regression", "decision_tree", "random_forest", "gradient_boosting"])
    parser.add_argument("--history", help="CSV with StockCode,date,sales and 28 complete past days")
    parser.add_argument("--product")
    parser.add_argument("--date")
    args = parser.parse_args()
    if args.history:
        if args.input or not args.product or not args.date:
            parser.error("Use --history with --product and --date.")
        try:
            sample = sample_from_history(args.history, args.product, args.date)
        except (ValueError, KeyError) as error:
            parser.error(str(error))
    elif args.input:
        sample = json.loads(args.input)
    else:
        parser.error("Provide --input or --history.")
    if sample["StockCode"] not in json.loads((LAB / "models/products.json").read_text()):
        parser.error("Product was not in the trained product scope.")
    day = pd.Timestamp(sample["date"])
    if pd.isna(day):
        parser.error("date must be a valid date.")
    sample.update(month=day.month, weekday=day.dayofweek, day_of_year=day.dayofyear)
    frame = pd.DataFrame([sample]).reindex(columns=["date"] + FEATURES)
    values = frame[FEATURES[1:8]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] < 0).any():
        parser.error("Lag and rolling features must be finite and nonnegative.")
    selected = json.loads((LAB / "models/selection.json").read_text())
    name = args.model or selected["winner"]
    model = joblib.load(LAB / "models" / f"{name}.joblib")
    robust_values = frame[FEATURES[8:]].to_numpy(dtype=float)
    if model.preprocessor.feature_set == "robust" and (not np.isfinite(robust_values).all() or (robust_values < 0).any()):
        parser.error("This model needs complete rolling statistics. Use --history with 28 past days.")
    result = {"model": name, "predicted_units": float(max(0, model.predict(frame[FEATURES])[0]))}
    history = LAB / "reports/inference_history.csv"
    history.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": selected["run_id"], "StockCode": sample["StockCode"],
              "date": str(day.date()), **result}
    with history.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(record)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
