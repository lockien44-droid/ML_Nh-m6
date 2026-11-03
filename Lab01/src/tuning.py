"""Trace default optimization and select settings using training data only."""

import json
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ..algorithms import LinearSVM, LogisticRegression
from .preprocessing import build_vectorizer
from .svm_study import select_svm


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


def select_models(train, directory):
    subtrain, validation = train_test_split(
        train, test_size=0.2, random_state=43, stratify=train['label_num'],
    )
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(subtrain['text_clean'])
    V = vectorizer.transform(validation['text_clean'])
    y, actual = subtrain['label_num'].to_numpy(), validation['label_num'].to_numpy()
    candidates = {
        'logistic_regression': (LogisticRegression, [
            dict(learning_rate=rate, n_iterations=steps)
            for rate, steps in product([0.1, 1.0, 5.0], [1000, 3000])
        ]),
        'svm': (LinearSVM, [
            dict(learning_rate=rate, n_iterations=steps, C=c)
            for rate, steps, c in [(0.01, 1000, 1), (0.05, 3000, 1), (0.05, 1000, 10),
                                   (0.1, 3000, 10), (0.01, 1000, 50), (0.05, 3000, 50),
                                   (0.01, 3000, 100)]
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
    summary = dict(seed=43, validation_fraction=0.2,
                   subtrain_counts=subtrain['label_num'].value_counts().to_dict(),
                   validation_counts=validation['label_num'].value_counts().to_dict(), selected=selected)
    (directory / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    models = {name: candidates[name][0](**params) for name, params in selected.items()}
    models['svm'] = select_svm(X, y, V, actual, directory / 'svm_study')
    return models
