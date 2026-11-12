"""Record an isolated run and stage events."""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from time import perf_counter
from uuid import uuid4


@contextmanager
def training_run(root, dataset):
    started = perf_counter()
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run.mkdir(parents=True)
    manifest = {"status": "running", "dataset_sha256": sha256(dataset.read_bytes()).hexdigest()}
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
        manifest["duration_seconds"] = perf_counter() - started
        (run / "run.json").write_text(json.dumps(manifest, indent=2))
