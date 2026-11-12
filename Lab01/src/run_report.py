"""Record one Lab01 training run and generate report figures."""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256

import numpy as np
from matplotlib.figure import Figure


VIETNAM_TIME = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")


def vietnam_now():
    return datetime.now(VIETNAM_TIME)


@contextmanager
def training_run(root, dataset):
    started = vietnam_now()
    start = time.perf_counter()
    base_id = f"{started:%Y%m%d_%H%M%S}_baseline_tuned"
    # Reserve the directory atomically; simultaneous runs must never overwrite.
    sequence = 1
    while True:
        run_id = base_id if sequence == 1 else f"{base_id}_{sequence:02d}"
        directory = root / run_id
        try:
            directory.mkdir(parents=True)
            break
        except FileExistsError:
            sequence += 1
    print(f"Run: {directory}")
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "dataset": dataset.name,
        "timezone": "Asia/Ho_Chi_Minh",
        "experiment": "baseline_tuned",
    }

    def log(stage, status="started"):
        event = {"time": vietnam_now().isoformat(), "stage": stage, "status": status}
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
        manifest["finished_at"] = vietnam_now().isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - start, 3)
        save_manifest()


def save_preprocessing_figures(run_dir, X_train, X_test, y_train, y_test, *,
                              X_validation=None, y_validation=None):
    prep = run_dir / "preprocessing"
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    summary = json.loads((prep / "summary.json").read_text(encoding="utf-8"))
    if X_validation is None:
        splits, matrices, targets = ('Train', 'Test'), (X_train, X_test), (y_train, y_test)
    else:
        splits = ('Subtrain', 'Validation', 'Test')
        matrices = (X_train, X_validation, X_test)
        targets = (y_train, y_validation, y_test)
    labels = [np.bincount(y, minlength=2).tolist() for y in targets]
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
    ax.set(title="Label distribution by split", ylabel="Emails")
    ax.legend()
    figure.savefig(figures / "split_distribution.png", dpi=150)

    details = {
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "tfidf": {"max_features": 50000, "ngram_range": [1, 2], "norm": "l2",
                  "fit_split": "subtrain" if X_validation is not None else "train"},
        "sets": {
            name.lower(): {"ham": counts[0], "spam": counts[1], "shape": list(X.shape),
                           "nonzero": X.nnz, "density_percent": density}
            for name, counts, X, density in zip(splits, labels, matrices, densities)
        },
    }
    (prep / "split_tfidf_summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
