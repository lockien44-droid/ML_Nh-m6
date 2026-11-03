"""Train the three Lab01 classifiers on the same TF-IDF split."""

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from ..algorithms import LinearSVM, LogisticRegression, MultinomialNaiveBayes
from .evaluate import evaluate
from .preprocessing import LAB_DIR, PREPROCESSING_DIR, prepare_data
from .tuning import select_models, trace_fit


def main():
    result_dir = LAB_DIR / "result"
    model_dir = LAB_DIR / "models"
    X_train, X_test, y_train, y_test = prepare_data()
    prep = PREPROCESSING_DIR
    train, test = (pd.read_csv(prep / "08_normalize_whitespace" / f"{split}.csv", keep_default_na=False)
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

    models.update(select_models(train, comparison_dir))
    # Refit the shared vectorizer on the original full training split after selection.
    vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    X_train = vectorizer.fit_transform(train["text_clean"])
    X_test = vectorizer.transform(test["text_clean"])
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    comparison = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate(y_test, model.predict(X_test), email_ids, result_dir / name)
        joblib.dump(model, model_dir / f"{name}.joblib")
        comparison.append({"model": name, **metrics})
        print(f"{name}: accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.2%}")

    comparison_dir = result_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(comparison_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
