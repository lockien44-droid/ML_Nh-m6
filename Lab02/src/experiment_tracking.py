"""Record an isolated run and stage events."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from time import perf_counter


@contextmanager
def training_run(root, dataset):
    started = perf_counter()
    timestamp = datetime.now(timezone(timedelta(hours=7)))
    base_id = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    sequence = 1
    while True:
        name = base_id if sequence == 1 else f"{base_id}_{sequence:02d}"
        run = root / name
        try:
            run.mkdir(parents=True)
            break
        except FileExistsError:
            sequence += 1
    manifest = {"run_id": run.name, "status": "running", "started_at": timestamp.isoformat(),
                "timezone": "Asia/Ho_Chi_Minh", "dataset_sha256": sha256(dataset.read_bytes()).hexdigest()}
    def log(stage):
        with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        print(stage, flush=True)
    try:
        yield run, log
        manifest["status"] = "success"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat()
        manifest["duration_seconds"] = perf_counter() - started
        (run / "run.json").write_text(json.dumps(manifest, indent=2))


def resolve_run(run_id):
    """Resolve one run without allowing paths outside the run root."""
    from .preprocessing import LAB
    root = (LAB / "results/runs").resolve()
    run = (root / run_id).resolve()
    if run.parent != root or not (run / "run.json").is_file():
        raise ValueError("Unknown run ID.")
    return run


def load_split(run, name):
    """Map exported FMCG columns to the model's internal representation."""
    import pandas as pd
    return pd.read_csv(run / "preprocessing" / f"{name}.csv",
                       dtype={"sku": str, "StockCode": str},
                       float_precision="round_trip").rename(
                           columns={"sku": "StockCode", "Revenue": "sales"})


@contextmanager
def run_stage(run, stage):
    """Protect stage ordering and record progress for separately invoked commands."""
    lock = run / ".stage.lock"
    try:
        stream = lock.open("x")
    except FileExistsError:
        raise ValueError("Another stage is running for this run.") from None
    try:
        stream.close()
        path = run / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        prerequisites = {
            "tuning": ["preprocessing/summary.json", "preprocessing/train.csv", "preprocessing/validation.csv"],
            "training": ["tuning/selection.json", "tuning/validation_metrics.csv"],
            "evaluation": ["models/linear_regression.joblib", "models/decision_tree.joblib",
                           "models/random_forest.joblib", "models/gradient_boosting.joblib",
                           "preprocessing/test.csv"],
        }
        if any(not (run / item).is_file() for item in prerequisites[stage]):
            hint = " Run python -m Lab02.src.train first." if stage == "evaluation" else ""
            raise ValueError(f"Required artifacts are missing before {stage}.{hint}")
        if stage == "tuning" and (run / "tuning/selection.json").exists():
            raise ValueError("Tuning is already complete. Create a new run to change configurations.")
        if stage == "training" and ("training" in manifest.get("completed_stages", [])
                or (manifest.get("status") == "success" and any((run / "models").glob("*.joblib")))):
            raise ValueError("Final models already exist. Use evaluate or create a new run.")
        def save():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(path)
        def log(status):
            with (run / "events.jsonl").open("a", encoding="utf-8") as events:
                events.write(json.dumps({"stage": stage, "status": status,
                    "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        manifest.update(status="running", current_stage=stage)
        manifest.pop("error", None)
        save()
        log("started")
        started = perf_counter()
        try:
            yield
        except BaseException as error:
            manifest.update(status="failed", error=str(error))
            log("failed")
            raise
        else:
            manifest["status"] = {"tuning": "tuned", "training": "trained", "evaluation": "success"}[stage]
            manifest.setdefault("completed_stages", [])
            if stage not in manifest["completed_stages"]:
                manifest["completed_stages"].append(stage)
            log("success")
        finally:
            manifest["finished_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat()
            manifest["duration_seconds"] = manifest.get("duration_seconds", 0) + perf_counter() - started
            save()
    finally:
        lock.unlink()


def resolve_best_run():
    """Use the fixed selection only when it matches the original tuning artifacts."""
    from .preprocessing import LAB
    path = LAB / "best_config.json"
    if not path.is_file():
        raise ValueError("best_config.json is missing. Run tuning first or provide --run-id.")
    best = json.loads(path.read_text(encoding="utf-8"))
    run = resolve_run(best["run_id"])
    selection = json.loads((run / "tuning/selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "preprocessing/summary.json").read_text(encoding="utf-8"))
    if selection != best["models"] or summary["dataset_sha256"] != best["dataset_sha256"]:
        raise ValueError("best_config.json does not match its run. Restore it from tuning artifacts.")
    return run


def stage_main(description, operation, use_best_config=False):
    """Share run selection and clear errors across stage entry points."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-id", required=not use_best_config,
                        help="Use a specific run; train/evaluate default to best_config.json")
    args = parser.parse_args()
    try:
        run = resolve_run(args.run_id) if args.run_id else resolve_best_run()
        operation(run)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
