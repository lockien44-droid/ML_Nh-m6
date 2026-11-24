"""Shared model training and training diagnostics for Lab01."""

import numpy as np
import pandas as pd
import json
from time import perf_counter, sleep
from matplotlib.figure import Figure

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



def training_directory(run, model_name, fit_id):
    """Resolve grouped training artifacts, with support for historical flat runs."""
    grouped = run / 'training' / model_name / fit_id
    legacy = run / 'training' / fit_id
    return grouped if grouped.exists() or not legacy.exists() else legacy


def load_training_record(run, model_name, fit_id):
    """Read one fit from the compact table or an older per-fit directory."""
    summary = run / 'training/training_summary.csv'
    if summary.exists():
        rows = pd.read_csv(summary, float_precision='round_trip')
        rows = rows.loc[rows.fit_id.eq(fit_id) & rows.model.eq(model_name)]
        if len(rows) != 1:
            raise ValueError(f'Expected one training record for {model_name}/{fit_id}.')
        record = {key: None if pd.isna(value) else value for key,value in rows.iloc[0].to_dict().items()}
        for key in ('parameters', 'requested_parameters'):
            record[key] = json.loads(record[key])
        return record
    return json.loads((training_directory(run, model_name, fit_id) / 'training.json').read_text(encoding='utf-8'))


def save_training_record(run, record):
    """Upsert by fit_id so running/success/failed states use a single row."""
    path = run / 'training/training_summary.csv'
    row = dict(record)
    for key in ('parameters', 'requested_parameters'):
        row[key] = json.dumps(row[key], sort_keys=True)
    frame = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path, float_precision='round_trip')
        records = previous.loc[~previous.fit_id.eq(record['fit_id'])].to_dict(orient='records')
        frame = pd.DataFrame([*records, row])
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    # Windows file watchers can briefly lock the destination during replacement.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            sleep(.05)


def save_selected_training_plot(run, model_name, fit_id):
    """Keep only the selected fit's plot; retain every fit's numeric history."""
    path = run / 'training' / model_name / 'history.csv'
    if not path.exists():
        return
    frame = pd.read_csv(path, float_precision='round_trip')
    frame = frame.loc[frame.fit_id.eq(fit_id)]
    if frame.empty:
        raise ValueError(f'Missing learning history for {fit_id}.')
    record = load_training_record(run, model_name, fit_id)
    figure = Figure(figsize=(7, 4), layout='tight')
    ax = figure.subplots()
    ax.plot(frame.iteration, frame.objective)
    ax.set(title=f'{model_name} / selected {fit_id}', xlabel='Updates', ylabel=record['objective_name'])
    figure.savefig(path.parent / 'selected_training.png', dpi=130)

def fit_experiment(model, X, y, *, run, fit_id, model_name, parameters,
                   vectorizer_id, variant):
    """Fit once and record its identity, sampled objective and elapsed time.

    Threshold sweeps reuse this fitted object and never create another fit_id.
    Objectives are model-specific and are not comparable across algorithms.
    """
    destination = run / 'training' / model_name
    destination.mkdir(parents=True, exist_ok=True)
    summary = run / 'training/training_summary.csv'
    if summary.exists() and fit_id in pd.read_csv(summary).fit_id.values:
        raise ValueError(f'Fit ID already exists: {fit_id}')
    info = dict(fit_id=fit_id, model=model_name, variant=variant,
                parameters={key: getattr(model, key) for key in ('learning_rate', 'n_iterations', 'C', 'alpha', 'class_weight', 'threshold') if hasattr(model, key)},
                requested_parameters=parameters, vectorizer_id=vectorizer_id,
                fit_split='subtrain', samples=int(X.shape[0]),
                features=int(X.shape[1]), status='running')
    save_training_record(run, info)
    history = []
    started = perf_counter()

    def record(iteration, fitted):
        step = max(1, fitted.n_iterations // 20)
        if iteration not in (0, 1, fitted.n_iterations) and iteration % step:
            return
        scores = X @ fitted.weights + fitted.bias
        if model_name == 'logistic_regression':
            objective = np.mean(np.logaddexp(0, scores) - y * scores)
        else:
            margins = (2 * y - 1) * scores
            weight = getattr(fitted, 'class_weight', None)
            if weight == 'balanced':
                counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
                sample_weight = (len(y) / (2 * counts))[np.asarray(y, dtype=int)]
            elif isinstance(weight, dict):
                sample_weight = np.array([weight[int(label)] for label in y])
            else:
                sample_weight = 1.0
            objective = .5 * np.sum(fitted.weights ** 2) + fitted.C * np.mean(
                sample_weight * np.maximum(0, 1 - margins))
        history.append(dict(iteration=int(iteration), objective=float(objective),
                            bias=float(fitted.bias), weight_norm=float(np.linalg.norm(fitted.weights))))

    try:
        if model_name == 'naive_bayes':
            model.fit(X, y)
        else:
            model.fit(X, y, on_iteration=record)
        elapsed = perf_counter() - started
        objective_name = ('binary_cross_entropy' if model_name == 'logistic_regression'
                          else 'weighted_hinge_plus_l2' if model_name == 'svm' else None)
        info.update(status='success', fit_seconds=elapsed, objective_name=objective_name,
                    initial_objective=history[0]['objective'] if history else None,
                    final_objective=history[-1]['objective'] if history else None,
                    timing_scope='fit including sampled diagnostics; excludes plots and serialization')
        if history:
            path = destination / 'history.csv'
            frame = pd.DataFrame(history).assign(fit_id=fit_id)
            frame.to_csv(path, mode='a', header=not path.exists(), index=False)
        return model, info
    except BaseException as error:
        info.update(status='failed', error=str(error), fit_seconds=perf_counter() - started)
        raise
    finally:
        save_training_record(run, info)


def main():
    from .pipeline import main as run_pipeline

    run_pipeline()


if __name__ == "__main__":
    main()
