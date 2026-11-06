"""Record one Lab01 training run and generate report figures."""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

import numpy as np
from matplotlib.figure import Figure


def utc_now():
    return datetime.now(timezone.utc)


@contextmanager
def training_run(root, dataset):
    started = utc_now()
    start = time.perf_counter()
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    directory = root / run_id
    directory.mkdir(parents=True)
    print(f"Run: {directory}")
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "dataset": dataset.name,
    }

    def log(stage, status="started"):
        event = {"time": utc_now().isoformat(), "stage": stage, "status": status}
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def save_manifest():
        target = directory / "run.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    save_manifest()
    log("run")
    try:
        manifest["dataset_sha256"] = sha256(dataset.read_bytes()).hexdigest()
        yield directory, log
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        log("run", "failed")
        raise
    else:
        manifest["status"] = "success"
        log("run", "success")
    finally:
        manifest["finished_at"] = utc_now().isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - start, 3)
        save_manifest()


def save_preprocessing_figures(run_dir, X_train, X_test, y_train, y_test):
    prep = run_dir / "preprocessing"
    figures = run_dir / "figures"
    figures.mkdir()
    summary = json.loads((prep / "summary.json").read_text(encoding="utf-8"))
    splits = ("Train", "Test")
    labels = [np.bincount(y, minlength=2).tolist() for y in (y_train, y_test)]
    matrices = (X_train, X_test)
    densities = [100 * X.nnz / (X.shape[0] * X.shape[1]) for X in matrices]
    original = summary["original_rows"]
    valid = original - summary["invalid_rows_removed"]
    nonempty = valid - summary["empty_rows_removed"]
    panels = (
        ("cleaning_counts.png", ["Raw", "Valid", "Nonempty", "Unique"],
         [original, valid, nonempty, summary["clean_rows"]], "Emails through cleaning", "Emails"),
        ("tfidf_density.png", splits, densities, "Nonzero TF-IDF entries", "Nonzero entries (%)"),
    )
    for filename, categories, values, title, ylabel in panels:
        figure = Figure(figsize=(6, 4), layout="tight")
        ax = figure.subplots()
        bars = ax.bar(categories, values, color="#286B93")
        ax.bar_label(bars, fmt="%.3f" if filename == "tfidf_density.png" else "%.0f")
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.15))
        figure.savefig(figures / filename, dpi=150)

    figure = Figure(figsize=(6, 4), layout="tight")
    ax = figure.subplots()
    ham, spam = zip(*labels)
    ax.bar_label(ax.bar(splits, ham, label="Ham", color="#286B93"), label_type="center", color="white")
    ax.bar_label(ax.bar(splits, spam, bottom=ham, label="Spam", color="#B65325"),
                 label_type="center", color="white")
    ax.set(title="Train and test label distribution", ylabel="Emails")
    ax.legend()
    figure.savefig(figures / "split_distribution.png", dpi=150)

    details = {
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "tfidf": {"max_features": 50000, "ngram_range": [1, 2], "norm": "l2"},
        "sets": {
            name.lower(): {"ham": counts[0], "spam": counts[1], "shape": list(X.shape),
                           "nonzero": X.nnz, "density_percent": density}
            for name, counts, X, density in zip(splits, labels, matrices, densities)
        },
    }
    (prep / "split_tfidf_summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
