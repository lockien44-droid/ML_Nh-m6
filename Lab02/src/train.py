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
