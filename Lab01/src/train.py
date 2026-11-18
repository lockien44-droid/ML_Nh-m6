"""Shared model training and training diagnostics for Lab01."""

import numpy as np
import pandas as pd

from ..algorithms import LogisticRegression


def train_model(model, X, y, trace_path=None):
    """Fit one estimator, optionally recording iterative training diagnostics."""
    if trace_path is not None:
        return trace_fit(model, X, y, trace_path)
    return model.fit(X, y)


def trace_fit(model, X, y, path):
    rows = []

    def record(iteration, fitted):
        if iteration not in {0, 1, 10, 100, 500, fitted.n_iterations}:
            return
        score = X @ fitted.weights + fitted.bias
        norm = np.linalg.norm(fitted.weights)
        row = dict(iteration=iteration, bias=fitted.bias, weight_norm=norm,
                   predicted_spam_rate=np.mean(score >= 0))
        if isinstance(fitted, LogisticRegression):
            row['objective'] = np.mean(np.logaddexp(0, score) - y * score)
            score = fitted.sigmoid(score)
        else:
            margin = (2 * y - 1) * score
            hinge = np.maximum(0, 1 - margin).mean()
            penalty = 0.5 * norm**2
            row.update(hinge=hinge, regularization=penalty,
                       objective=penalty + fitted.C * hinge, active_rate=np.mean(margin < 1))
        for label, name in [(0, 'ham'), (1, 'spam')]:
            for quantile, value in zip(['min', 'p10', 'median', 'p90', 'max'],
                                       np.quantile(score[y == label], [0, .1, .5, .9, 1])):
                row[f'{name}_{quantile}'] = value
        rows.append(row)

    model.fit(X, y, on_iteration=record)
    pd.DataFrame(rows).to_csv(path, index=False)
    return model


def main():
    from .pipeline import main as run_pipeline

    run_pipeline()


if __name__ == "__main__":
    main()
