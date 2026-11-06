"""Train the three Lab01 classifiers on the same TF-IDF split."""

import json
from time import perf_counter
from shutil import copytree
from itertools import product

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from ..algorithms import LinearSVM, LogisticRegression, MultinomialNaiveBayes
from .evaluate import evaluate
from .preprocessing import DATA_DIR, LAB_DIR, PREPROCESSING_DIR, build_vectorizer, prepare_data
from .run_report import training_run, save_preprocessing_figures


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


def train_run(run_dir, log):
    result_dir = run_dir / "evaluation"
    model_dir = run_dir / "models"
    prep = run_dir / "preprocessing"
    log("preprocessing")
    X_train, X_test, y_train, y_test = prepare_data(prep)
    save_preprocessing_figures(run_dir, X_train, X_test, y_train, y_test)
    train, test = (pd.read_csv(prep / f"{split}_clean.csv", keep_default_na=False)
                   for split in ("train", "test"))
    for features, labels, frame, split in ((X_train, y_train, train, "train"), (X_test, y_test, test, "test")):
        saved = pd.read_csv(prep / "09_tfidf" / f"y_{split}.csv")
        if (not isspmatrix_csr(features) or features.shape[0] != len(labels)
                or not np.array_equal(labels, frame["label_num"])
                or not saved.equals(frame[["email_id", "label_num"]].reset_index(drop=True))):
            raise ValueError(f"CSR rows, email IDs and labels do not align for {split}.")
    email_ids = test["email_id"]
    models = {
        "logistic_regression": LogisticRegression(),
        "svm": LinearSVM(),
        "naive_bayes": MultinomialNaiveBayes(),
    }

    comparison_dir = result_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    log("baseline")
    baseline = []
    for name, model in models.items():
        if name == "naive_bayes":
            model.fit(X_train, y_train)
        else:
            trace_fit(model, X_train, y_train, comparison_dir / f"{name}_trace.csv")
        predicted = model.predict(X_test)
        metrics = evaluate(y_test, predicted, email_ids, comparison_dir / "baseline" / name)
        baseline.append({"model": name, **metrics, "predicted_spam": int(predicted.sum()),
                         "predicted_ham": int(len(predicted) - predicted.sum())})
    pd.DataFrame(baseline).to_csv(comparison_dir / "baseline.csv", index=False)

    log("validation")
    models.update(select_models(train, comparison_dir))
    # Refit the shared vectorizer on the original full training split after selection.
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train["text_clean"])
    X_test = vectorizer.transform(test["text_clean"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    log("final_training_and_evaluation")
    comparison = []
    training = []
    for name, model in models.items():
        started = perf_counter()
        model.fit(X_train, y_train)
        training.append({
            "model": name, "fit_seconds": perf_counter() - started,
            **{key: getattr(model, key, None) for key in
               ("learning_rate", "n_iterations", "C", "alpha", "threshold")},
        })
        metrics = evaluate(y_test, model.predict(X_test), email_ids, result_dir / name)
        joblib.dump(model, model_dir / f"{name}.joblib")
        comparison.append({"model": name, **metrics})
        print(f"{name}: accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.2%}")

    comparison_dir = result_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(comparison_dir / "metrics.csv", index=False)
    pd.DataFrame(training).to_csv(comparison_dir / "training.csv", index=False)
    log("publish_compatibility_outputs")
    copytree(prep, PREPROCESSING_DIR, dirs_exist_ok=True)
    copytree(result_dir, LAB_DIR / "result", dirs_exist_ok=True)
    copytree(model_dir, LAB_DIR / "models", dirs_exist_ok=True)
    (LAB_DIR / "models" / "run_id.txt").write_text(run_dir.name + "\n", encoding="utf-8")


def main():
    dataset = DATA_DIR / "raw" / "spam_ham_dataset.csv"
    with training_run(LAB_DIR / "result" / "runs", dataset) as (run_dir, log):
        train_run(run_dir, log)
    from .report import generate

    generate(run_dir)


if __name__ == "__main__":
    main()
