"""Track Lab01 experiment identity, events, timing and status."""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256



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
        save_manifest()
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
