"""Build a custom estimator with a fitted-on-train preprocessing pipeline."""
from ..algorithms.linear_regression import Model as Linear
from ..algorithms.decision_tree import Model as Tree
from ..algorithms.random_forest import Model as Forest
from ..algorithms.gradient_boosting import Model as Boost
from .preprocessing import build_preprocessor

MODELS = {"linear_regression": Linear, "decision_tree": Tree,
          "random_forest": Forest, "gradient_boosting": Boost}


def build_model(name, parameters, seed):
    settings = dict(parameters)
    feature_set = settings.pop("feature_set", "basic")
    if name != "linear_regression":
        settings["random_state"] = seed
    return SalesModel(build_preprocessor(feature_set), MODELS[name](**settings))


class SalesModel:
    """Keep the learned preprocessing and estimator together for inference."""

    def __init__(self, preprocessor, estimator):
        self.preprocessor = preprocessor
        self.estimator = estimator

    def fit(self, X, y):
        transformed = self.preprocessor.fit_transform(X)
        self.estimator.fit(transformed, y)
        return self

    def predict(self, X):
        return self.estimator.predict(self.preprocessor.transform(X))


def train_model(name, parameters, seed, X, y):
    """Construct and fit one model on the supplied training partition."""
    model = build_model(name, parameters, seed)
    return model.fit(X, y)


def train_run(run):
    """Refit selected configurations on train plus validation and save models."""
    import json
    from time import perf_counter
    import joblib
    import pandas as pd
    import yaml
    from .preprocessing import FEATURES
    from .experiment_tracking import load_split, run_stage

    with run_stage(run, "training"):
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        selected = json.loads((run / "tuning/selection.json").read_text(encoding="utf-8"))
        development = pd.concat([load_split(run, "train"), load_split(run, "validation")])
        timings = {}
        for name, parameters in selected.items():
            started = perf_counter()
            model = train_model(name, parameters, config["seed"],
                                development[FEATURES], development.sales)
            timings[name] = perf_counter() - started
            joblib.dump(model, run / "models" / f"{name}.joblib")
        trials = pd.read_csv(run / "tuning/validation_metrics.csv")
        winner = trials.loc[trials.selection_rmse.idxmin(), "model"]
        summary = json.loads((run / "preprocessing/summary.json").read_text(encoding="utf-8"))
        metadata = dict(winner=winner, run_id=run.name, frequency=summary["frequency"],
                        target=summary["target"], unit=summary["unit"], fit_seconds=timings)
        (run / "models/selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (run / "models/products.json").write_text(json.dumps(summary["products"]), encoding="utf-8")
    print("Trained run:", run.name, flush=True)


def main():
    from .experiment_tracking import stage_main
    # Import the canonical module when invoked with -m so pickles never refer to __main__.
    from .train import train_run as operation
    stage_main("Fit final Lab02 models using the selected configurations.", operation)


if __name__ == "__main__":
    main()
