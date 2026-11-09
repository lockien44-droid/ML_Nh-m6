"""Run the Lab01 preprocessing, training, evaluation and reporting pipeline."""

from shutil import copytree
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from ..algorithms import LinearSVM, LogisticRegression, MultinomialNaiveBayes
from .evaluate import evaluate
from .preprocessing import DATA_DIR, LAB_DIR, PREPROCESSING_DIR, build_vectorizer, prepare_data
from .run_report import save_preprocessing_figures, training_run
from .train import trace_fit
from .tuning import select_models


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
    copytree(model_dir, LAB_DIR / "models", dirs_exist_ok=True)
    (LAB_DIR / "models" / "run_id.txt").write_text(run_dir.name + "\n", encoding="utf-8")


def main():
    dataset = DATA_DIR / "raw" / "spam_ham_dataset.csv"
    with training_run(LAB_DIR / "results" / "runs", dataset) as (run_dir, log):
        train_run(run_dir, log)
    from .report import generate

    generate(run_dir)


if __name__ == "__main__":
    main()
