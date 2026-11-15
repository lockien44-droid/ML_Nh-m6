"""Build daily product demand and past-only predictors."""
from pathlib import Path
import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
NUMERIC = ["lag_1", "lag_7", "mean_7", "mean_28", "month", "weekday", "day_of_year"]
FEATURES = ["StockCode"] + NUMERIC


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
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    raw = raw.drop_duplicates()
    valid = (raw["InvoiceDate"].notna() & raw["StockCode"].str.fullmatch(r"\d{5}[A-Za-z]*", na=False)
             & ~raw["InvoiceNo"].str.startswith("C", na=False)
             & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0))
    clean = raw.loc[valid].copy()
    clean["date"] = clean["InvoiceDate"].dt.normalize()
    # The first and last days can be partially recorded.
    days = pd.date_range(clean["date"].min() + pd.Timedelta(days=1),
                         clean["date"].max() - pd.Timedelta(days=1))
    train_end = days[int(len(days) * config["split"]["train_fraction"]) - 1]
    products = clean.loc[clean["date"] <= train_end].groupby("StockCode")["Quantity"].sum()
    products = products.nlargest(config["data"]["top_products"]).index
    daily = clean.groupby(["StockCode", "date"])["Quantity"].sum()
    grid = pd.MultiIndex.from_product([products, days], names=["StockCode", "date"])
    frame = daily.reindex(grid, fill_value=0).rename("sales").reset_index()
    grouped = frame.groupby("StockCode")["sales"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_7"] = grouped.shift(7)
    frame["mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())
    frame["mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(28).mean())
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
        part.to_csv(output / f"{name}.csv", index=False)
    summary = {"raw_rows": len(raw), "clean_transaction_rows": len(clean),
               "products": list(products), "train_end": str(train_end.date()),
               "validation_end": str(validation_end.date()),
               "target": "positive units sold per product/day"}
    return train, validation, test, summary


def expand_history_features(frame):
    frame = frame.copy()
    frame["trend_7_28"] = frame["mean_7"] - frame["mean_28"]
    frame["recent_change"] = frame["lag_1"] - frame["mean_7"]
    frame["weekly_change"] = frame["lag_1"] - frame["lag_7"]
    return frame


class Preprocessor:
    """Learn medians, scaling and product vocabulary from training rows."""

    def __init__(self, feature_set="basic"):
        if feature_set not in ("basic", "trend"):
            raise ValueError("Unknown feature set.")
        self.feature_set = feature_set

    def fit(self, frame):
        self.numeric = NUMERIC + (["trend_7_28", "recent_change", "weekly_change"]
                                  if self.feature_set == "trend" else [])
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
