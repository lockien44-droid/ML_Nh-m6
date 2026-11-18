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
