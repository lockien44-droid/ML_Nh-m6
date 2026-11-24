"""Evaluate binary email predictions and save per-model results."""

import json

import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


def evaluate(y_true, y_pred, email_ids, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [0, 1]
    names = ["ham", "spam"]
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "ham_as_spam": int(matrix[0, 1]),
        "spam_as_ham": int(matrix[1, 0]),
        "spam_precision": report["spam"]["precision"],
        "spam_recall": report["spam"]["recall"],
        "spam_f1": report["spam"]["f1-score"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame.from_dict({name: report[name] for name in names}, orient="index").to_csv(
        output_dir / "classification_report.csv", index_label="class",
    )
    pd.DataFrame({"email_id": email_ids, "actual": y_true, "predicted": y_pred}).to_csv(
        output_dir / "predictions.csv", index=False,
    )
    figure = Figure(figsize=(5, 4), layout="tight")
    ConfusionMatrixDisplay(matrix, display_labels=names).plot(ax=figure.subplots(), colorbar=False)
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    return {**metrics, "spam_precision": report["spam"]["precision"],
            "spam_recall": report["spam"]["recall"], "spam_f1": report["spam"]["f1-score"]}


def save_errors(model, X, frame, output):
    details = frame[["email_id", "text_clean", "label_num"]].copy().reset_index(drop=True)
    details["predicted"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        details["score"] = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        details["score"] = model.decision_function(X)
    else:
        scores = X @ model.feature_log_prob.T + model.class_log_prior
        details["score"] = scores[:, 1] - scores[:, 0]
    details["threshold"] = getattr(model, "threshold", 0.0)
    details = details.loc[details.label_num != details.predicted]
    details["error_type"] = details.predicted.map({0: "FN", 1: "FP"})
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output / "errors.csv", index=False)
