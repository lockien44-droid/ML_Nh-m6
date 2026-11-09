"""Select Lab01 model settings on the fixed validation split."""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from ..algorithms import LinearSVM, LogisticRegression, MultinomialNaiveBayes
from .preprocessing import build_vectorizer


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


def select_svm(X, y, V, actual, directory, params):
    directory.mkdir(parents=True, exist_ok=True)
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


def select_models(train, directory):
    config_path = Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fraction = config['validation']['fraction']
    seed = config['validation']['seed']
    subtrain, validation = train_test_split(
        train, test_size=fraction, random_state=seed, stratify=train['label_num'],
    )
    subtrain.to_csv(directory / 'subtrain_clean.csv', index=False)
    validation.to_csv(directory / 'validation_clean.csv', index=False)
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(subtrain['text_clean'])
    V = vectorizer.transform(validation['text_clean'])
    y, actual = subtrain['label_num'].to_numpy(), validation['label_num'].to_numpy()
    logistic = config['logistic_regression']
    candidates = {
        'logistic_regression': (LogisticRegression, [
            dict(learning_rate=rate, n_iterations=steps)
            for rate, steps in product(logistic['learning_rate'], logistic['n_iterations'])
        ]),
        'svm': (LinearSVM, config['svm']['candidates']),
        'naive_bayes': (MultinomialNaiveBayes, [
            dict(alpha=alpha) for alpha in config['naive_bayes']['alpha']
        ]),
    }
    rows, selected = [], {}
    for name, (cls, settings) in candidates.items():
        scores = []
        for params in settings:
            predicted = cls(**params).fit(X, y).predict(V)
            macro = f1_score(actual, predicted, average='macro')
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, average='binary', zero_division=0,
            )
            scores.append(macro)
            rows.append(dict(model=name, **params, macro_f1=macro, spam_precision=precision,
                             spam_recall=recall, spam_f1=f1, predicted_spam=int(predicted.sum())))
        selected[name] = settings[int(np.argmax(scores))]
    pd.DataFrame(rows).to_csv(directory / 'validation.csv', index=False)
    summary = dict(seed=seed, validation_fraction=fraction,
                   subtrain_counts=subtrain['label_num'].value_counts().to_dict(),
                   validation_counts=validation['label_num'].value_counts().to_dict(), selected=selected)
    (directory / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    models = {name: candidates[name][0](**params) for name, params in selected.items()}
    models['svm'] = select_svm(X, y, V, actual, directory / 'svm_study', selected['svm'])
    return models
