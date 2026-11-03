"""Compare weighted hinge and decision thresholds on validation only."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from ..algorithms import LinearSVM


def threshold_metrics(actual, scores, threshold):
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (actual == 1)))
    fp = int(np.count_nonzero(predicted & (actual == 0)))
    fn = int(np.count_nonzero(~predicted & (actual == 1)))
    tn = len(actual) - tp - fp - fn
    f1 = 2 * tp / (2 * tp + fp + fn)
    ham_f1 = 2 * tn / (2 * tn + fp + fn)
    return dict(threshold=float(threshold), macro_f1=(f1 + ham_f1) / 2,
                spam_precision=tp / (tp + fp) if tp + fp else 0.0,
                spam_recall=tp / (tp + fn), spam_f1=f1, fn=fn, fp=fp,
                predicted_spam=tp + fp, confusion_matrix=[[tn, fp], [fn, tp]])


def select_svm(X, y, V, actual, directory):
    directory.mkdir(parents=True, exist_ok=True)
    params = dict(learning_rate=0.01, n_iterations=3000, C=100)
    rows, distributions = [], []
    options = [("unweighted", None), ("balanced", "balanced"),
               ("spam_x2", {0: 1, 1: 2}), ("spam_x3", {0: 1, 1: 3}),
               ("spam_x4", {0: 1, 1: 4})]
    for name, weight in options:
        model = LinearSVM(**params, class_weight=weight).fit(X, y)
        scores = model.decision_function(V)
        ap = average_precision_score(actual, scores)
        precision, recall, thresholds = precision_recall_curve(actual, scores)
        pd.DataFrame(dict(precision=precision, recall=recall,
                          threshold=np.r_[thresholds, np.nan])).to_csv(directory / f"{name}_pr.csv", index=False)
        for label in (0, 1):
            subset = scores[actual == label]
            quantiles = np.quantile(subset, [0, .1, .5, .9, 1])
            distributions.append(dict(model=name, label=label, min=quantiles[0], p10=quantiles[1],
                median=quantiles[2], p90=quantiles[3], max=quantiles[4], nonnegative_rate=np.mean(subset >= 0)))
        choices = [threshold_metrics(actual, scores, t) for t in np.unique(
            np.r_[0.0, scores, np.nextafter(scores.max(), np.inf)])]
        best = max(choices, key=lambda r: (r['macro_f1'], -r['fp'], -abs(r['threshold'])))
        for group, metrics in zip(('A', 'B') if weight is None else ('C', 'D'),
                                  (threshold_metrics(actual, scores, 0), best)):
            rows.append(dict(group=group, model=name, **params, class_weight=weight,
                             average_precision=ap, **metrics))
    # Prefer the unweighted model on an exact macro F1 tie.
    chosen = max(rows, key=lambda r: (r['macro_f1'], r['class_weight'] is None,
                                      -r['fp'], -abs(r['threshold'])))
    pd.DataFrame(rows).to_csv(directory / 'validation.csv', index=False)
    pd.DataFrame(distributions).to_csv(directory / 'score_distribution.csv', index=False)
    summary = dict(spam_prevalence=float(actual.mean()), selected=chosen)
    (directory / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return LinearSVM(**params, class_weight=chosen['class_weight'], threshold=chosen['threshold'])
