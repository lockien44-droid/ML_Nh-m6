"""Fit candidates on subtrain and select settings/thresholds on validation only."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from ..algorithms import LinearSVM, LogisticRegression, MultinomialNaiveBayes
from .preprocessing import build_vectorizer
from .train import fit_experiment, save_selected_training_plot
from .evaluate import evaluate, save_errors


def threshold_metrics(actual, scores, threshold):
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (actual == 1)))
    fp = int(np.count_nonzero(predicted & (actual == 0)))
    fn = int(np.count_nonzero(~predicted & (actual == 1)))
    tn = len(actual) - tp - fp - fn
    divide = lambda a, b: a / b if b else 0.0
    spam_f1 = divide(2 * tp, 2 * tp + fp + fn)
    ham_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return dict(threshold=float(threshold), macro_f1=(spam_f1 + ham_f1) / 2,
                spam_precision=divide(tp, tp + fp), spam_recall=divide(tp, tp + fn),
                spam_f1=spam_f1, fn=fn, fp=fp, tn=tn, tp=tp,
                accuracy=divide(tp + tn, len(actual)), predicted_spam=tp + fp,
                confusion_matrix=[[tn, fp], [fn, tp]])


def model_scores(model, X):
    """Return the score and its meaning; SVM/NB scores are not probabilities."""
    if isinstance(model, LogisticRegression):
        return model.predict_proba(X), 'spam_probability'
    if isinstance(model, LinearSVM):
        return model.decision_function(X), 'decision_score'
    scores = X @ model.feature_log_prob.T + model.class_log_prior
    return np.asarray(scores[:, 1] - scores[:, 0]).ravel(), 'log_joint_difference'


def save_threshold_search(directory, fit_id, choices, chosen, default, score_type, mode):
    destination = directory / 'threshold_search' / fit_id
    destination.mkdir(parents=True, exist_ok=False)
    rows = pd.DataFrame(choices).drop(columns='confusion_matrix').sort_values('threshold')
    rows['selected'] = rows.threshold.eq(chosen['threshold'])
    rows['is_default'] = rows.threshold.eq(default['threshold'])
    rows.to_csv(destination / 'thresholds.csv', index=False)
    metadata = dict(fit_id=fit_id, split='validation', metric='macro_f1', mode=mode,
                    score_type=score_type, default=default, selected=chosen,
                    tie_break='fewer FP, then closer to default threshold')
    (destination / 'selection.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    figure = Figure(figsize=(7, 4), layout='tight')
    ax = figure.subplots()
    ax.plot(rows.threshold, rows.macro_f1, label='Validation Macro-F1')
    ax.axvline(default['threshold'], color='gray', linestyle='--', label='Default')
    ax.axvline(chosen['threshold'], color='red', linestyle=':', label='Selected')
    ax.set(title=f'Threshold search / {fit_id}', xlabel=score_type, ylabel='Macro-F1')
    ax.legend()
    figure.savefig(destination / 'thresholds.png', dpi=130)


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
    vectorizer_id: str
    selected_fits: dict
    baseline_fits: dict


def select_models(train, directory, model_name=None):
    """One summary row per fit; a threshold sweep reuses the same fitted weights."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run = directory.parent
    snapshot = run / 'config.yaml'
    config = yaml.safe_load((snapshot if snapshot.exists() else Path(__file__).with_name('config.yaml')).read_text(encoding='utf-8'))
    subtrain, validation = train_test_split(train, test_size=config['validation']['fraction'],
        random_state=config['validation']['seed'], stratify=train.label_num)
    prep = run / 'preprocessing'
    prep.mkdir(parents=True, exist_ok=True)
    subtrain.to_csv(prep / 'subtrain_clean.csv', index=False)
    validation.to_csv(prep / 'validation_clean.csv', index=False)
    vectorizer = build_vectorizer(config.get('features'))
    X = vectorizer.fit_transform(subtrain.text_clean)
    V = vectorizer.transform(validation.text_clean)
    digest = sha256(subtrain[['email_id', 'text_clean', 'label_num']].to_csv(index=False).encode('utf-8'))
    digest.update(json.dumps(vectorizer.get_params(), default=str, sort_keys=True).encode())
    digest.update(vectorizer.idf_.tobytes())
    vectorizer_id = 'tfidf_' + digest.hexdigest()[:12]
    vectorizers = run / 'models/vectorizers'
    vectorizers.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizers / f'{vectorizer_id}.joblib')
    vector_metadata = dict(vectorizer_id=vectorizer_id, fit_split='subtrain', refit=False,
        parameters=vectorizer.get_params(), subtrain_sha256=sha256(
            subtrain[['email_id', 'text_clean', 'label_num']].to_csv(index=False).encode('utf-8')).hexdigest(),
        features=int(X.shape[1]), samples=int(X.shape[0]))
    (vectorizers / f'{vectorizer_id}.json').write_text(json.dumps(vector_metadata, indent=2, default=str), encoding='utf-8')
    y, actual = subtrain.label_num.to_numpy(), validation.label_num.to_numpy()
    classes = dict(logistic_regression=LogisticRegression, svm=LinearSVM, naive_bayes=MultinomialNaiveBayes)
    names = [model_name] if model_name else list(classes)
    if any(name not in classes for name in names):
        raise ValueError('Unknown model name.')
    training, rows, baselines, models, selected_fits, baseline_fits = [], [], {}, {}, {}, {}
    fit_root = run / 'training'
    fit_root.mkdir(exist_ok=True)
    existing_ids = [int(path.parent.name.split('_')[1])
                    for path in fit_root.rglob('training.json')
                    if path.parent.name.startswith('fit_')]
    if (fit_root / 'training_summary.csv').exists():
        existing_ids.extend(int(value.split('_')[1]) for value in pd.read_csv(fit_root / 'training_summary.csv').fit_id)
    sequence = max(existing_ids, default=0)

    def fit_one(name, parameters, variant):
        nonlocal sequence
        sequence += 1
        fit_id = f'fit_{sequence:04d}'
        print(f'{fit_id}: {variant}/{name}', flush=True)
        model, info = fit_experiment(classes[name](**parameters), X, y, run=run,
            fit_id=fit_id, model_name=name, parameters=parameters,
            vectorizer_id=vectorizer_id, variant=variant)
        training.append(info)
        return model, info

    def summary_row(name, info, default, chosen, variant):
        return dict(fit_id=info['fit_id'], model=name, variant=variant,
            parameters=json.dumps(info['parameters'], sort_keys=True), vectorizer_id=vectorizer_id,
            fit_seconds=info['fit_seconds'], default_threshold=default['threshold'] if name!='naive_bayes' else None,
            selected_threshold=chosen['threshold'] if name!='naive_bayes' else None,
            default_macro_f1=default['macro_f1'], selected_macro_f1=chosen['macro_f1'],
            macro_f1=chosen['macro_f1'], accuracy=chosen['accuracy'], spam_precision=chosen['spam_precision'],
            spam_recall=chosen['spam_recall'], spam_f1=chosen['spam_f1'], fp=chosen['fp'], fn=chosen['fn'], selected=False)

    for name in names:
        baseline, info = fit_one(name, {}, 'baseline')
        scores, _ = model_scores(baseline, V)
        default = threshold_metrics(actual, scores, .5 if name=='logistic_regression' else np.nextafter(0.,np.inf) if name=='naive_bayes' else 0.)
        rows.append(summary_row(name, info, default, default, 'baseline'))
        baselines[name], baseline_fits[name] = baseline, info['fit_id']

    logistic = config['logistic_regression']
    settings = {
        'logistic_regression': [dict(learning_rate=r, n_iterations=n) for r,n in product(logistic['learning_rate'], logistic['n_iterations'])],
        'naive_bayes': [dict(alpha=a) for a in config['naive_bayes']['alpha']],
        'svm': [dict(**params, class_weight=weight) for params in config['svm']['candidates']
                for weight in (None, 'balanced', {0:1,1:2}, {0:1,1:3}, {0:1,1:4})]}
    best_keys = {}
    for name in names:
        for parameters in settings[name]:
            model, info = fit_one(name, parameters, 'candidate')
            scores, score_type = model_scores(model, V)
            default_t = .5 if name=='logistic_regression' else np.nextafter(0.,np.inf) if name=='naive_bayes' else 0.
            default = threshold_metrics(actual, scores, default_t)
            chosen = default
            if name != 'naive_bayes':
                if name == 'logistic_regression':
                    values = np.unique(np.r_[0., .5, 1., scores, np.nextafter(scores.max(), np.inf)])
                    values = values[(values>=0)&(values<=1)]
                    mode = 'automatic'
                    if logistic.get('threshold') is not None:
                        values = [float(logistic['threshold'])]; mode = 'fixed'
                    elif logistic.get('threshold_candidates'):
                        values = list(map(float,logistic['threshold_candidates'])); mode = 'configured_candidates'
                    if any(not np.isfinite(t) or not 0<=t<=1 for t in values):
                        raise ValueError('Logistic thresholds must be finite and between 0 and 1.')
                else:
                    values = np.unique(np.r_[0., scores, np.nextafter(scores.max(), np.inf)])
                    mode = 'automatic'
                choices = [threshold_metrics(actual,scores,t) for t in values]
                chosen = max(choices,key=lambda r:(r['macro_f1'],-r['fp'],-abs(r['threshold']-default_t)))
                model.threshold = chosen['threshold']
                save_threshold_search(directory,info['fit_id'],choices,chosen,default,score_type,mode)
            row = summary_row(name,info,default,chosen,'candidate')
            rows.append(row)
            # Preserve existing selection rules, including SVM preference for unweighted fits.
            key = (chosen['macro_f1'], parameters.get('class_weight') is None,
                   -chosen['fp'], -abs(chosen['threshold'])) if name=='svm' else (chosen['macro_f1'],)
            if name not in best_keys or key > best_keys[name]:
                best_keys[name] = key
                models[name], selected_fits[name] = model,info['fit_id']
    for name, fit_id in selected_fits.items():
        save_selected_training_plot(run, name, fit_id)
    for row in rows:
        row['selected'] = row['fit_id'] == selected_fits.get(row['model'])
    pd.DataFrame(rows).to_csv(directory / 'validation_summary.csv',index=False)
    summary = dict(metric='validation_macro_f1',fit_split='subtrain',refit=False,
        vectorizer_id=vectorizer_id,selected_fits=selected_fits,baseline_fits=baseline_fits,
        validation_fraction=config['validation']['fraction'],seed=config['validation']['seed'],
        selected={name:{**next(i['requested_parameters'] for i in training if i['fit_id']==selected_fits[name]),
            **({'threshold':float(model.threshold)} if name!='naive_bayes' else {})} for name,model in models.items()})
    (directory / 'selection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for variant, fitted in [('baseline',baselines),('selected',models)]:
        for name,model in fitted.items():
            output = directory / 'validation' / name / variant
            evaluate(actual,model.predict(V),validation.email_id,output)
            save_errors(model,V,validation,output)
    return SelectionResult(models,baselines,vectorizer,subtrain,validation,X,V,training,
                           vectorizer_id,selected_fits,baseline_fits)
