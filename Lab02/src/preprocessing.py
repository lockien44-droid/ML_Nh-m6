"""Build daily product demand and past-only predictors."""
from pathlib import Path
import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
NUMERIC = ["lag_1", "lag_7", "mean_7", "mean_28", "month", "weekday", "day_of_year"]
HISTORY = ["median_7", "median_28", "std_7", "std_28", "zero_days_28", "same_weekday_mean_4"]
FEATURES = ["StockCode"] + NUMERIC + HISTORY


def prepare_data(config, output):
    train_fraction = config["split"]["train_fraction"]
    validation_fraction = config["split"]["validation_fraction"]
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and leave a nonempty test period.")
    if not isinstance(config["data"]["top_products"], int) or config["data"]["top_products"] < 1:
        raise ValueError("top_products must be a positive integer.")
    raw = pd.read_csv(LAB / config["data"]["path"], encoding=config["data"]["encoding"],
                      dtype={"InvoiceNo": str, "StockCode": str})
    fmcg = config["data"].get("schema") == "fmcg_weekly"
    weekly = config["data"].get("schema") in ("superstore_weekly", "fmcg_weekly")
    category_data = config["data"].get("schema") == "category_orders"
    if fmcg:
        raw = raw.rename(columns={"date": "InvoiceDate", "sku": "StockCode", "price_unit": "UnitPrice"})
        raw["Quantity"] = pd.to_numeric(raw["units_sold"], errors="coerce") * pd.to_numeric(raw["UnitPrice"], errors="coerce")
    elif weekly:
        raw = raw.drop(columns=["Quantity"])
        raw = raw.rename(columns={"Order ID": "InvoiceNo", "Product ID": "StockCode",
                                  "Order Date": "InvoiceDate", "Sales": "Quantity"})
        raw["UnitPrice"] = 1.0
    if category_data:
        raw = raw.rename(columns={"order_id": "InvoiceNo", "order_date": "InvoiceDate",
                                  "product_category": "StockCode", "quantity": "Quantity",
                                  "unit_price": "UnitPrice"})
        raw["InvoiceNo"] = raw["InvoiceNo"].astype("string")
        raw["StockCode"] = raw["StockCode"].astype("string")
    original_rows = len(raw)
    date_format = "%Y-%m-%d" if fmcg else ("%m/%d/%Y" if category_data or weekly else "%m/%d/%Y %H:%M")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], format=date_format, errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    raw = raw.drop_duplicates()
    valid_codes = (raw["StockCode"].str.strip().ne("").fillna(False) if category_data or weekly
                   else raw["StockCode"].str.fullmatch(r"\d{5}[A-Za-z]*", na=False))
    valid = (raw["InvoiceDate"].notna() & valid_codes
             & (True if weekly else ~raw["InvoiceNo"].str.startswith("C", na=False))
             & (raw["Quantity"] >= 0 if fmcg else raw["Quantity"] > 0)
             & np.isfinite(raw["Quantity"]) & np.isfinite(raw["UnitPrice"])
             & (raw["UnitPrice"] > 0))
    clean = raw.loc[valid].copy()
    if clean.empty:
        raise ValueError("No valid sales transactions after cleaning.")
    clean["date"] = clean["InvoiceDate"].dt.normalize()
    # The first and last days can be partially recorded.
    trim = pd.Timedelta(days=0 if category_data else 1)
    days = pd.date_range(clean["date"].min() + trim, clean["date"].max() - trim)
    if weekly:
        first = clean["date"].min()
        last = clean["date"].max()
        start = first.to_period("W-SUN").start_time
        if start < first:
            start += pd.Timedelta(weeks=1)
        end = last.to_period("W-SUN").start_time
        if end + pd.Timedelta(days=6) > last:
            end -= pd.Timedelta(weeks=1)
        days = pd.date_range(start, end, freq="W-MON")
        clean["date"] = clean["date"].dt.to_period("W-SUN").dt.start_time
    train_end = days[int(len(days) * config["split"]["train_fraction"]) - 1]
    products = clean.loc[clean["date"] <= train_end].groupby("StockCode")["Quantity"].sum()
    if weekly:
        active = clean.loc[clean["date"].between(days[0], train_end) & (clean["Quantity"] > 0)].groupby("StockCode")["date"].nunique()
        products = products.loc[active.reindex(products.index, fill_value=0) >= config["data"].get("min_active_weeks", 6)]
        if products.empty:
            raise ValueError("No products meet the training-history requirement.")
    products = products.nlargest(config["data"]["top_products"]).index
    daily = clean.groupby(["StockCode", "date"])["Quantity"].sum()
    grid = pd.MultiIndex.from_product([products, days], names=["StockCode", "date"])
    frame = daily.reindex(grid, fill_value=0).rename("sales").reset_index()
    grouped = frame.groupby("StockCode")["sales"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_7"] = grouped.shift(7)
    frame["mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())
    frame["mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(28).mean())
    # Every statistic excludes the target day, including weekday seasonality.
    for window in (7, 28):
        frame[f"median_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window).median())
        frame[f"std_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window).std(ddof=0))
    frame["zero_days_28"] = grouped.transform(lambda s: s.shift(1).rolling(28).apply(lambda x: (x == 0).sum(), raw=True))
    frame["same_weekday_mean_4"] = sum(grouped.shift(lag) for lag in (7, 14, 21, 28)) / 4
    frame["month"] = frame["date"].dt.month
    frame["weekday"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame = frame.dropna(subset=NUMERIC).reset_index(drop=True)
    validation_end = days[int(len(days) * (config["split"]["train_fraction"] +
                                                    config["split"]["validation_fraction"])) - 1]
    train = frame.loc[frame.date <= train_end]
    validation = frame.loc[(frame.date > train_end) & (frame.date <= validation_end)]
    test = frame.loc[frame.date > validation_end]
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part.empty:
            raise ValueError(f"Empty {name} split.")
        exported = part.rename(columns={"StockCode": "sku", "sales": "Revenue"}) if fmcg else part
        exported.to_csv(output / f"{name}.csv", index=False)
    summary = {"raw_rows": original_rows, "deduplicated_rows": len(raw), "duplicate_rows_removed": original_rows - len(raw), "clean_transaction_rows": len(clean),
               "products": list(products), "train_end": str(train_end.date()),
               "validation_end": str(validation_end.date()),
               "entity": "product_category" if category_data else "product",
               "frequency": "weekly" if weekly else "daily",
               "unit": "source price_unit currency" if fmcg else ("source Sales currency" if weekly else "units"),
               "revenue_definition": "price_unit * units_sold; negative units excluded; zero units retained" if fmcg else "source Sales" if weekly else "quantity",
               "invalid_rows_removed": len(raw) - len(clean),
               "source": config["data"]["path"],
               "target": "Revenue per sku/week" if fmcg else "Sales revenue per product/week" if weekly else ("positive units sold per category/day" if category_data else "positive units sold per product/day")}
    summary["split_periods"] = {
        name: {"rows": len(part), "start": str(part.date.min().date()),
               "end": str((part.date.max() + pd.Timedelta(days=6 if weekly else 0)).date())}
        for name, part in (("train", train), ("validation", validation), ("test", test))
    }
    return train, validation, test, summary


def expand_history_features(frame):
    frame = frame.copy()
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(float)
    frame["trend_7_28"] = frame["mean_7"] - frame["mean_28"]
    frame["recent_change"] = frame["lag_1"] - frame["mean_7"]
    frame["weekly_change"] = frame["lag_1"] - frame["lag_7"]
    return frame


class Preprocessor:
    """Learn medians, scaling and product vocabulary from training rows."""

    def __init__(self, feature_set="basic"):
        if feature_set not in ("basic", "trend", "robust"):
            raise ValueError("Unknown feature set.")
        self.feature_set = feature_set

    def fit(self, frame):
        self.numeric = NUMERIC + (["trend_7_28", "recent_change", "weekly_change"]
                                  if self.feature_set in ("trend", "robust") else [])
        if self.feature_set == "robust":
            self.numeric += HISTORY + ["is_weekend"]
        values = expand_history_features(frame)[self.numeric].to_numpy(dtype=float)
        if len(values) == 0 or np.isinf(values).any():
            raise ValueError("Training features must be nonempty and contain no infinities.")
        # A completely missing column uses zero as its documented fallback.
        self.medians = np.array([
            np.median(column[~np.isnan(column)]) if np.any(~np.isnan(column)) else 0.0
            for column in values.T
        ])
        filled = np.where(np.isnan(values), self.medians, values)
        self.means = filled.mean(axis=0)
        self.scales = filled.std(axis=0)
        self.scales[self.scales == 0] = 1.0
        self.categories = np.sort(frame["StockCode"].astype(str).unique())
        return self

    def transform(self, frame):
        if not hasattr(self, "medians"):
            raise ValueError("Fit the preprocessor before transform.")
        values = expand_history_features(frame)[getattr(self, "numeric", NUMERIC)].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Features must contain no infinities.")
        filled = np.where(np.isnan(values), self.medians, values)
        scaled = (filled - self.means) / self.scales
        products = frame["StockCode"].astype(str).to_numpy()
        encoded = (products[:, None] == self.categories[None, :]).astype(float)
        return np.column_stack([scaled, encoded])

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)


def build_preprocessor(feature_set="basic"):
    return Preprocessor(feature_set)


def prepare_run(config_path=None):
    """Create an immutable config snapshot and prepare one new experiment."""
    import json
    from hashlib import sha256
    import yaml
    from .experiment_tracking import training_run
    from .eda import analyze

    config = yaml.safe_load(Path(config_path or LAB / "src/config.yaml").read_text(encoding="utf-8"))
    with training_run(LAB / "results/runs", LAB / config["data"]["path"]) as (run, log):
        for folder in ("preprocessing", "tuning", "evaluation", "figures", "models"):
            (run / folder).mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        log("preprocessing")
        train, validation, test, summary = prepare_data(config, run / "preprocessing")
        summary["run_id"] = run.name
        summary["dataset_sha256"] = sha256((LAB / config["data"]["path"]).read_bytes()).hexdigest()
        (run / "preprocessing/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        analyze(pd.concat([train, validation, test]), run / "figures", summary)
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="prepared", completed_stages=["preprocessing"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared run:", run.name, flush=True)
    return run


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data for a new Lab02 run.")
    parser.add_argument("--config")
    args = parser.parse_args()
    prepare_run(args.config)


if __name__ == "__main__":
    main()
