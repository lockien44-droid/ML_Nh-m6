"""Generate Lab01 report files from a completed training run."""

import argparse
import json
import re
from html import escape
from pathlib import Path
from shutil import copy2

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy.sparse import load_npz
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from .eda import analyze
from .preprocessing import DATA_DIR, LAB_DIR

MODELS = {"logistic_regression": "lr", "naive_bayes": "nb", "svm": "svm"}
DISPLAY = {"logistic_regression": "Logistic Regression", "naive_bayes": "Naive Bayes", "svm": "Linear SVM"}
REPORTS = LAB_DIR / "reports"
RUNS = LAB_DIR / "results"
LEGACY_RUNS = LAB_DIR / "results" / "runs"


def chart(path, title, labels, values, ylabel):
    fig = Figure(figsize=(7, 4), layout="tight")
    ax = fig.subplots()
    bars = ax.bar(labels, values, color="#286B93")
    ax.bar_label(bars, fmt="%.4g", padding=3)
    ax.margins(y=.16)
    ax.set(title=title, ylabel=ylabel)
    if not any(values):
        ax.set_ylim(0, 1)
    fig.savefig(path, dpi=150)


def table_figure(path, title, frame):
    fig = Figure(figsize=(10, 2.8), layout="tight")
    ax = fig.subplots()
    ax.axis("off")
    ax.set_title(title)
    table = ax.table(cellText=frame.fillna("—").values, colLabels=frame.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    fig.savefig(path, dpi=150)


def save_eda(raw, source, output):
    analyze(source, output)
    labels = pd.read_csv(output / "label_distribution.csv")
    chart(output / "label_distribution.png", "Raw label distribution",
          labels["label"], labels["count"], "Emails")
    missing = raw.isna() | raw.apply(lambda column: column.astype(str).str.strip().eq(""))
    chart(output / "missing_values.png", "Missing or blank values", raw.columns, missing.sum(), "Rows")
    fig = Figure(figsize=(7, 4), layout="tight")
    ax = fig.subplots()
    lengths = raw["text"].fillna("").str.split().str.len()
    bins = np.histogram_bin_edges(lengths, bins=30)
    for label, name in [(0, "Ham"), (1, "Spam")]:
        ax.hist(lengths[raw["label_num"] == label], bins=bins, label=name, alpha=.6)
    ax.set(title="Email length by class", xlabel="Words", ylabel="Emails")
    ax.legend()
    fig.savefig(output / "email_length_distribution.png", dpi=150)
    # The individual stage plots replace the legacy composite in this directory.
    (output / "overview.png").unlink()
    for name in ("summary.json", "features_by_label.csv"):
        (output / name).unlink()
    (output / "label_distribution.csv").replace(REPORTS / "label_distribution.csv")


def save_preprocessing(run, prep, output, X_train):
    for old, new in {
        "cleaning_counts": "preprocessing_steps",
        "split_distribution": "train_test_distribution",
        "tfidf_density": "tfidf_sparsity",
    }.items():
        copy2(run / "figures" / f"{old}.png", output / f"{new}.png")
    sample = pd.read_csv(prep / "cleaned_raw.csv", keep_default_na=False)
    sample = sample.loc[sample["text"] != sample["text_clean"]].iloc[0]
    from textwrap import fill
    fig = Figure(figsize=(9, 5), layout="tight")
    for ax, field, title in zip(fig.subplots(2), ("text", "text_clean"), ("Original", "Cleaned")):
        ax.axis("off")
        ax.set_title(title, loc="left")
        excerpt = sample[field]
        excerpt = excerpt[:430] + (" ..." if len(excerpt) > 430 else "")
        ax.text(0, .88, fill(excerpt, width=90), va="top", transform=ax.transAxes, fontsize=9)
    fig.savefig(output / "cleaning_comparison.png", dpi=150)

    names = pd.read_csv(prep / "09_tfidf/feature_names.csv")["feature_name"]
    averages = np.asarray(X_train.mean(axis=0)).ravel()
    top = np.argsort(averages)[-20:]
    fig = Figure(figsize=(7, 6), layout="tight")
    ax = fig.subplots()
    ax.barh(names.iloc[top], averages[top], color="#286B93")
    ax.set(title="Top 20 features by mean training TF-IDF", xlabel="Mean TF-IDF")
    fig.savefig(output / "tfidf_top_features.png", dpi=150)


def save_evaluation(run, output, metrics, X_test, y_test):
    fig = Figure(figsize=(12, 4), layout="tight")
    roc = Figure(figsize=(6, 5), layout="tight")
    roc_ax = roc.subplots()
    rates = []
    for ax, (name, short) in zip(fig.subplots(1, 3), MODELS.items()):
        source = run / "evaluation" / name
        copy2(source / "confusion_matrix.png", output / f"confusion_matrix_{short}.png")
        predictions = pd.read_csv(source / "predictions.csv")
        if not np.array_equal(predictions["actual"], y_test):
            raise ValueError(f"Test labels do not match saved predictions for {name}.")
        matrix = confusion_matrix(y_test, predictions["predicted"], labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["Ham", "Spam"]).plot(ax=ax, colorbar=False)
        ax.set_title(DISPLAY[name])
        tn, fp, fn, tp = matrix.ravel()
        rates.append({"Model": short.upper(), "TPR": tp / (tp + fn), "FPR": fp / (fp + tn)})
        model = joblib.load(run / "models" / f"{name}.joblib")
        if short == "lr":
            scores = model.predict_proba(X_test)
        elif short == "svm":
            scores = model.decision_function(X_test)
        else:
            scores = (X_test @ (model.feature_log_prob[1] - model.feature_log_prob[0])
                      + model.class_log_prior[1] - model.class_log_prior[0])
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_ax.plot(fpr, tpr, label=f"{short.upper()} (AUC {auc(fpr, tpr):.3f})")
    fig.savefig(output / "confusion_matrix_all.png", dpi=150)
    roc_ax.plot([0, 1], [0, 1], "--", color="grey")
    roc_ax.set(title="ROC on test", xlabel="False positive rate", ylabel="True positive rate")
    roc_ax.legend()
    roc.savefig(output / "roc_curve_all.png", dpi=150)
    for filename, frame, title in [
        ("metrics_comparison.png", metrics.set_index("model")[["accuracy", "spam_precision", "spam_recall", "spam_f1"]], "Test metrics"),
        ("tpr_fpr_comparison.png", pd.DataFrame(rates).set_index("Model"), "Test rates at deployed thresholds"),
    ]:
        fig = Figure(figsize=(8, 4.5), layout="tight")
        ax = fig.subplots()
        frame.rename(index=DISPLAY).plot.bar(ax=ax, rot=0)
        ax.set(title=title, ylim=(0, 1.08), xlabel="")
        fig.savefig(output / filename, dpi=150)


def threshold_comparison(run):
    """Report the selected configuration's saved validation sweep, without fitting."""
    path = run / 'evaluation/comparison/logistic_threshold_validation.csv'
    if not path.exists():
        raise FileNotFoundError('This run has no saved LR threshold sweep; rerun the current pipeline.')
    rows = pd.read_csv(path)
    if 'selected_configuration' in rows:
        rows = rows.loc[rows['selected_configuration'].eq(True)].copy()
    else:
        selected = json.loads((path.parent / 'selection.json').read_text(encoding='utf-8'))['selected']['logistic_regression']
        for key in ('learning_rate', 'n_iterations'):
            rows = rows.loc[rows[key].eq(selected[key])].copy()
    return rows.drop(columns=['confusion_matrix'], errors='ignore').assign(Run_ID=run.name, split='validation')


def write_reports(run, metrics, figures):
    stem = f"report_{run.name}"
    title = f"Lab01 training report - {run.name}"
    display = metrics[["model", "accuracy", "macro_f1", "spam_precision", "spam_recall", "spam_f1",
                       "spam_as_ham", "ham_as_spam"]].copy()
    display["model"] = display["model"].map(DISPLAY)
    display.columns = ["Model", "Accuracy", "Macro F1", "Precision", "Recall", "Spam F1", "FN", "FP"]
    for column in display.columns[1:-2]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    images = sorted(figures.glob("*/*.png"))
    rel = [path.relative_to(REPORTS).as_posix() for path in images]
    selection = json.loads((run / 'evaluation/comparison/selection.json').read_text(encoding='utf-8'))
    note = ("Models and TF-IDF fitted on subtrain; configurations and thresholds selected on validation; "
            "no refit before test. LR threshold comparison uses saved validation results."
            if selection.get('refit') is False else
            "Historical run. LR threshold comparison uses saved validation results.")
    cells = [display.columns.tolist(), *display.astype(str).values.tolist()]
    md_table = ["| " + " | ".join(row) + " |" for row in cells]
    md_table.insert(1, "| " + " | ".join(["---"] * len(cells[0])) + " |")
    (REPORTS / f"{stem}.md").write_text(
        f"# {title}\n\n{note}\n\n" + "\n".join(md_table)
        + "\n\n" + "\n\n".join(f"![{path.stem}]({url})" for path, url in zip(images, rel)) + "\n",
        encoding="utf-8",
    )
    gallery = "".join(f'<figure><img src="{escape(url)}" alt="{escape(path.stem)}"></figure>'
                      for path, url in zip(images, rel))
    (REPORTS / f"{stem}.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(title)}</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:2rem}'
        'td,th{padding:.5rem;border-bottom:1px solid #ddd}img{max-width:100%}</style>'
        f'<h1>{escape(title)}</h1><p>{escape(note)}</p>{display.to_html(index=False)}{gallery}</html>',
        encoding="utf-8",
    )
    from matplotlib.image import imread
    with PdfPages(REPORTS / f"{stem}.pdf", metadata={"Title": title}) as pdf:
        cover = Figure(figsize=(11.7, 8.3))
        ax = cover.subplots()
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        ax.text(.5, .76, note, ha="center", transform=ax.transAxes)
        table = ax.table(cellText=cells[1:], colLabels=cells[0], cellLoc="center", loc="center",
                         colWidths=[.23, .11, .11, .11, .11, .11, .07, .07])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        pdf.savefig(cover)
        for stage in ("eda", "preprocessing", "training", "evaluation"):
            stage_images = [path for path in images if path.parent.name == stage]
            for offset in range(0, len(stage_images), 4):
                fig = Figure(figsize=(11.7, 8.3), layout="tight")
                fig.suptitle(stage.capitalize(), fontsize=16)
                for ax, path in zip(fig.subplots(2, 2).flat, stage_images[offset:offset + 4], strict=False):
                    ax.imshow(imread(path))
                    ax.axis("off")
                for ax in fig.axes[len(stage_images[offset:offset + 4]):]:
                    ax.axis("off")
                pdf.savefig(fig)

    archive = REPORTS / "archive"
    archive.mkdir(exist_ok=True)
    reports = [path for path in REPORTS.glob("report_*.*")
               if path.suffix in {".md", ".html", ".pdf"}]
    latest = max(path.stem for path in reports)
    for old in reports:
        if old.stem == latest:
            continue
        if old.suffix in {".md", ".html"}:
            text = old.read_text(encoding="utf-8")
            old.write_text(text.replace("figures/", "../figures/"), encoding="utf-8")
        old.replace(archive / old.name)

def update_history(path, frame, run_id):
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous.loc[previous["Run_ID"] != run_id], frame], ignore_index=True)
    frame.to_csv(path, index=False)


def generate(run):
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if manifest["status"] != "success":
        raise ValueError("Reports require a successful run.")
    figures = REPORTS / "figures" / run.name
    for stage in ("eda", "preprocessing", "training", "evaluation"):
        (figures / stage).mkdir(parents=True, exist_ok=True)
    prep = run / "preprocessing"
    comparison = run / "evaluation/comparison"
    metrics = pd.read_csv(comparison / "metrics.csv")
    training = pd.read_csv(comparison / "training.csv")
    X_train, X_test = (load_npz(prep / "09_tfidf" / f"X_{split}.npz") for split in ("train", "test"))
    y_test = pd.read_csv(prep / "09_tfidf/y_test.csv")["label_num"].to_numpy()
    source = DATA_DIR / "raw/spam_ham_dataset.csv"
    from hashlib import sha256
    if sha256(source.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Current dataset differs from the one used in the run.")
    save_eda(pd.read_csv(source), source, figures / "eda")
    save_preprocessing(run, prep, figures / "preprocessing", X_train)
    training_dir = figures / "training"
    chart(training_dir / "training_time_comparison.png", "Selected model fit time (subtrain)",
          training["model"].map(MODELS).str.upper(), training["fit_seconds"], "Seconds")
    configuration = training.drop(columns="fit_seconds")
    lr_rows = configuration["model"].eq("logistic_regression")
    configuration.loc[lr_rows, "threshold"] = configuration.loc[lr_rows, "threshold"].fillna(0.5)
    table_figure(training_dir / "training_configuration.png", "Final model configuration", configuration)
    overview = pd.DataFrame([
        ["Subtrain" if (prep / "subtrain_clean.csv").exists() else "Train", X_train.shape[0], X_train.shape[1], X_train.nnz],
        ["Test", X_test.shape[0], X_test.shape[1], X_test.nnz],
    ], columns=["Split", "Emails", "Features", "Nonzero entries"])
    if (prep / '09_tfidf/X_validation.npz').exists():
        V = load_npz(prep / '09_tfidf/X_validation.npz')
        overview.loc[len(overview)] = ['Validation', V.shape[0], V.shape[1], V.nnz]
    table_figure(training_dir / "training_dataset_overview.png", "Dataset overview", overview)
    save_evaluation(run, figures / "evaluation", metrics, X_test, y_test)
    thresholds = threshold_comparison(run)
    update_history(REPORTS / "metrics_summary.csv", metrics.assign(Run_ID=run.name), run.name)
    update_history(REPORTS / "lr_threshold_comparison.csv", thresholds, run.name)
    write_reports(run, metrics, figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    current = re.fullmatch(r"\d{8}_\d{6}_baseline_tuned(?:_\d{2,})?", args.run_id)
    legacy = re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", args.run_id)
    if not (current or legacy):
        parser.error("Invalid run ID.")
    run = (RUNS if current else LEGACY_RUNS) / args.run_id
    if current and not (run / 'run.json').exists():
        previous_location = LAB_DIR / 'results_runs' / args.run_id
        if (previous_location / 'run.json').exists():
            run = previous_location
    generate(run)


if __name__ == "__main__":
    main()
