"""Select Lab01 model settings on the fixed validation split."""

import json
from itertools import product
from dataclasses import dataclass
from time import perf_counter
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
from .evaluate import evaluate
from .train import train_model


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
                accuracy=(tp + tn) / len(actual), predicted_spam=tp + fp, confusion_matrix=[[tn, fp], [fn, tp]])


def select_svm(X, y, V, actual, directory, candidates):
    """Jointly select hyperparameters, class weights and threshold; retain fitted winner."""
    directory.mkdir(parents=True, exist_ok=True)
    rows, distributions, all_thresholds = [], [], []
    options = [("unweighted", None), ("balanced", "balanced"),
               ("spam_x2", {0: 1, 1: 2}), ("spam_x3", {0: 1, 1: 3}),
               ("spam_x4", {0: 1, 1: 4})]
    winner, chosen, chosen_key = None, None, None
    for index, params in enumerate(candidates):
        for name, weight in options:
            candidate_id = f"candidate_{index}_{name}"
            started = perf_counter()
            model = train_model(LinearSVM(**params, class_weight=weight), X, y)
            fit_seconds = perf_counter() - started
            scores = model.decision_function(V)
            ap = average_precision_score(actual, scores)
            precision, recall, thresholds = precision_recall_curve(actual, scores)
            pd.DataFrame(dict(precision=precision, recall=recall,
                threshold=np.r_[thresholds, np.nan])).to_csv(
                    directory / f"{candidate_id}_pr.csv", index=False)
            for label in (0, 1):
                subset = scores[actual == label]
                quantiles = np.quantile(subset, [0, .1, .5, .9, 1])
                distributions.append(dict(candidate_id=candidate_id, model=name, label=label,
                    min=quantiles[0], p10=quantiles[1], median=quantiles[2], p90=quantiles[3],
                    max=quantiles[4], nonnegative_rate=float(np.mean(subset >= 0))))
            choices = [threshold_metrics(actual, scores, t) for t in np.unique(
                np.r_[0.0, scores, np.nextafter(scores.max(), np.inf)])]
            all_thresholds.extend(dict(candidate_id=candidate_id, **params, class_weight=weight, **row) for row in choices)
            best = max(choices, key=lambda r: (r['macro_f1'], -r['fp'], -abs(r['threshold'])))
            for group, metrics in zip(('A', 'B') if weight is None else ('C', 'D'),
                                      (threshold_metrics(actual, scores, 0), best)):
                rows.append(dict(group=group, candidate_id=candidate_id, model=name,
                    **params, class_weight=weight, fit_seconds=fit_seconds,
                    average_precision=ap, **metrics))
            key = (best['macro_f1'], weight is None, -best['fp'], -abs(best['threshold']))
            if chosen_key is None or key > chosen_key:
                chosen_key = key
                model.threshold = best['threshold']
                winner = model
                chosen = dict(candidate_id=candidate_id, **params, class_weight=weight,
                              fit_seconds=fit_seconds, **best)
    pd.DataFrame(rows).to_csv(directory / 'validation.csv', index=False)
    pd.DataFrame(all_thresholds).to_csv(directory / 'all_thresholds.csv', index=False)
    pd.DataFrame(distributions).to_csv(directory / 'score_distribution.csv', index=False)
    (directory / 'selection.json').write_text(json.dumps(
        dict(spam_prevalence=float(actual.mean()), selected=chosen), indent=2) + '\n', encoding='utf-8')
    return winner, chosen


def training_record(name, variant, model, fit_seconds):
    return dict(model=name, variant=variant, fit_seconds=fit_seconds, fit_split='subtrain',
        **{key: getattr(model, key, None) for key in
           ('learning_rate', 'n_iterations', 'C', 'alpha', 'threshold', 'class_weight')})


@dataclass
class SelectionResult:
    models: dict
    baselines: dict
    vectorizer: object
    subtrain: pd.DataFrame
    validation: pd.DataFrame
    X: object
    V: object
    training: list


def select_models(train, directory, model_name=None):
    directory.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text(encoding='utf-8'))
    fraction, seed = config['validation']['fraction'], config['validation']['seed']
    subtrain, validation = train_test_split(
        train, test_size=fraction, random_state=seed, stratify=train['label_num'])
    subtrain.to_csv(directory / 'subtrain_clean.csv', index=False)
    validation.to_csv(directory / 'validation_clean.csv', index=False)
    vectorizer = build_vectorizer(config.get('features'))
    X = vectorizer.fit_transform(subtrain['text_clean'])
    V = vectorizer.transform(validation['text_clean'])
    y, actual = subtrain['label_num'].to_numpy(), validation['label_num'].to_numpy()
    baselines = dict(logistic_regression=LogisticRegression(), svm=LinearSVM(),
                     naive_bayes=MultinomialNaiveBayes())
    if model_name:
        baselines = {model_name: baselines[model_name]}
    baseline_rows, training = [], []
    for name, model in baselines.items():
        started = perf_counter()
        if name == 'naive_bayes':
            train_model(model, X, y)
        else:
            train_model(model, X, y, directory / f'{name}_trace.csv')
        training.append(training_record(name, 'baseline', model, perf_counter() - started))
        predicted = model.predict(V)
        metrics = evaluate(actual, predicted, validation['email_id'],
                           directory / 'baseline_validation' / name)
        baseline_rows.append(dict(model=name, split='validation', **metrics,
            predicted_spam=int(predicted.sum()), predicted_ham=int(len(predicted) - predicted.sum())))
    pd.DataFrame(baseline_rows).to_csv(directory / 'baseline_validation.csv', index=False)

    logistic = config['logistic_regression']
    candidates = {
        'logistic_regression': (LogisticRegression, [
            dict(learning_rate=rate, n_iterations=steps)
            for rate, steps in product(logistic['learning_rate'], logistic['n_iterations'])]),
        'naive_bayes': (MultinomialNaiveBayes, [
            dict(alpha=alpha) for alpha in config['naive_bayes']['alpha']]),
    }
    if model_name:
        candidates = {name: value for name, value in candidates.items() if name == model_name}
    rows, selected, models, logistic_threshold_rows = [], {}, {}, []
    for name, (cls, settings) in candidates.items():
        best_macro = -np.inf
        for index, params in enumerate(settings):
            started = perf_counter()
            model = train_model(cls(**params), X, y)
            fit_seconds = perf_counter() - started
            params = dict(params)
            if name == 'logistic_regression':
                probabilities = model.predict_proba(V)
                thresholds = np.unique(np.r_[0.0, 0.5, 1.0, probabilities,
                    np.nextafter(probabilities.max(), np.inf)])
                choices = [threshold_metrics(actual, probabilities, t)
                           for t in thresholds if 0 <= t <= 1]
                configured = logistic.get('threshold')
                thresholds = logistic.get('threshold_candidates')
                if configured is not None:
                    if not 0 <= float(configured) <= 1:
                        raise ValueError("Logistic threshold must be between 0 and 1.")
                    choices = [threshold_metrics(actual, probabilities, float(configured))]
                elif thresholds:
                    if any(not 0 <= float(t) <= 1 for t in thresholds):
                        raise ValueError("Logistic thresholds must be between 0 and 1.")
                    choices = [threshold_metrics(actual, probabilities, float(t)) for t in thresholds]
                best = max(choices, key=lambda r: (
                    r['macro_f1'], -r['fp'], -abs(r['threshold'] - 0.5)))
                logistic_threshold_rows.extend(dict(candidate_id=index, **params, **row) for row in choices)
                model.threshold = best['threshold']
                params['threshold'] = model.threshold
            predicted = model.predict(V)
            macro = f1_score(actual, predicted, average='macro')
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, average='binary', zero_division=0)
            rows.append(dict(model=name, candidate_id=index, **params, macro_f1=macro,
                fit_seconds=fit_seconds, spam_precision=precision, spam_recall=recall,
                spam_f1=f1, predicted_spam=int(predicted.sum())))
            if macro > best_macro:
                best_macro = macro
                selected[name] = params
                models[name] = model
                chosen_fit_seconds = fit_seconds
                chosen_index = index
        training.append(training_record(name, 'tuned', models[name], chosen_fit_seconds))
        if name == 'logistic_regression':
            for row in logistic_threshold_rows:
                row['selected_configuration'] = row['candidate_id'] == chosen_index

    if model_name in (None, 'svm'):
        models['svm'], svm_choice = select_svm(
            X, y, V, actual, directory / 'svm_study', config['svm']['candidates'])
        selected['svm'] = {key: svm_choice[key] for key in
            ('learning_rate', 'n_iterations', 'C', 'class_weight', 'threshold')}
        training.append(training_record('svm', 'tuned', models['svm'], svm_choice['fit_seconds']))
        rows.append(dict(model='svm', **svm_choice))
    pd.DataFrame(logistic_threshold_rows).to_csv(directory / 'logistic_threshold_validation.csv', index=False)
    pd.DataFrame(rows).to_csv(directory / 'validation.csv', index=False)
    summary = dict(seed=seed, validation_fraction=fraction, fit_split='subtrain', refit=False,
        subtrain_counts=subtrain['label_num'].value_counts().to_dict(),
        validation_counts=validation['label_num'].value_counts().to_dict(), selected=selected)
    (directory / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for name, model in models.items():
        output = directory / 'selected_validation' / name
        evaluate(actual, model.predict(V), validation['email_id'], output)
        from .evaluate import save_errors
        save_errors(model, V, validation, output)
    return SelectionResult(models, baselines, vectorizer, subtrain, validation, X, V, training)
