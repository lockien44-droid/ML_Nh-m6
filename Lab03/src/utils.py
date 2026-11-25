"""Run directories, JSON artifacts, figures and execution tracking."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import json
from .config import LAB_DIR


def now():
    return datetime.now(timezone(timedelta(hours=7)))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_figure(figure, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, **kwargs)


def make_run_dir(part, root=None):
    if part not in ("customers", "image"):
        raise ValueError("Unknown exercise.")
    parent = Path(root or LAB_DIR / "runs") / part
    base = now().strftime("%Y-%m-%d_%H-%M-%S")
    sequence = 1
    while True:
        run = parent / (base if sequence == 1 else f"{base}_{sequence:02d}")
        try:
            run.mkdir(parents=True)
            return run
        except FileExistsError:
            sequence += 1


def resolve_run(part, run_id):
    parent = (LAB_DIR / "runs" / part).resolve()
    run = (parent / run_id).resolve()
    if run.parent != parent or not (run / "config_used.json").is_file():
        raise ValueError("Unknown run ID.")
    return run


def log_event(run, stage, status="started"):
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(stage=stage, status=status, time=now().isoformat())) + "\n")


@contextmanager
def track_run(run, stage):
    lock = run / ".run.lock"
    try:
        stream = lock.open("x")
    except FileExistsError:
        raise ValueError("Another operation is running in this run.") from None
    stream.close()
    manifest_path = run / "run.json"
    try:
        manifest = read_json(manifest_path) if manifest_path.exists() else dict(
            run_id=run.name, part=run.parent.name, started_at=now().isoformat(),
            timezone="Asia/Ho_Chi_Minh")
        manifest.update(status="running", current_stage=stage)
        manifest.pop("error", None)
        write_json(manifest_path, manifest)
        log_event(run, stage)
        started = perf_counter()
        try:
            yield manifest
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            log_event(run, stage, "failed")
            raise
        else:
            manifest["status"] = "success"
            log_event(run, stage, "success")
        finally:
            manifest.update(finished_at=now().isoformat(),
                duration_seconds=manifest.get("duration_seconds", 0) + perf_counter() - started)
            write_json(manifest_path, manifest)
    finally:
        lock.unlink(missing_ok=True)


def customer_step_main(operation, stage):
    import argparse
    parser = argparse.ArgumentParser(description=operation.__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        run = resolve_run("customers", args.run_id)
        with track_run(run, stage) as manifest:
            manifest.update(operation(run))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print("Run:", run)
