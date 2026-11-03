"""Explore the raw Lab01 email dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .preprocessing import LAB_DIR


def analyze(csv_path, output_dir):
    raw = pd.read_csv(csv_path, keep_default_na=False)
    if not {"text", "label_num"}.issubset(raw.columns):
        raise ValueError("CSV phải có hai cột text và label_num.")

    labels = pd.to_numeric(raw["label_num"], errors="coerce")
    valid_label = labels.isin([0, 1])
    texts = raw["text"].astype(str)
    has_text = raw["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    counts = labels[valid_label].map({0: "ham", 1: "spam"}).value_counts().reindex(["ham", "spam"], fill_value=0)
    distribution = counts.rename("count").to_frame()
    distribution["share"] = counts / counts.sum() if counts.sum() else 0.0

    usable = valid_label & has_text
    text = texts[usable]
    features = pd.DataFrame({
        "label": labels[usable].map({0: "ham", 1: "spam"}),
        "characters": text.str.len(),
        "words": text.str.split().str.len(),
        "url_markers": text.str.count(r"(?i)\bhttps?\s*:\s*/\s*/|(?<![\w./])\bwww\s*\."),
        "exclamation_marks": text.str.count("!"),
        "dollar_signs": text.str.count(r"\$"),
    })
    grouped = features.groupby("label")
    summary = {
        "total_rows": len(raw),
        "missing_or_blank_text": int((~has_text).sum()),
        "invalid_labels": int((~valid_label).sum()),
        "exact_duplicate_text_rows": int(texts[has_text].duplicated().sum()),
        "usable_rows": int(usable.sum()),
        "label_counts": {label: int(count) for label, count in counts.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    distribution.to_csv(output_dir / "label_distribution.csv", index_label="label")
    grouped.agg(["count", "mean", "median", "max"]).stack(level=0, future_stack=True).to_csv(
        output_dir / "features_by_label.csv", index_label=["label", "feature"],
    )

    figure = Figure(figsize=(10, 4), layout="tight")
    labels_ax, length_ax = figure.subplots(1, 2)
    labels_ax.bar(counts.index, counts.values, color=["tab:blue", "tab:orange"])
    labels_ax.set(title="Raw label distribution", ylabel="Emails")
    bins = np.histogram_bin_edges(features["words"], bins=30)
    for name, subset in grouped:
        length_ax.hist(subset["words"], bins=bins, alpha=0.55, label=name)
    if not features.empty:
        length_ax.legend()
    length_ax.set(title="Email length by label", xlabel="Whitespace-delimited words", ylabel="Emails")
    figure.savefig(output_dir / "overview.png", dpi=150)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LAB_DIR / "Data/raw/spam_ham_dataset.csv")
    parser.add_argument("--output", type=Path, default=LAB_DIR / "result/eda")
    args = parser.parse_args()
    print(json.dumps(analyze(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
