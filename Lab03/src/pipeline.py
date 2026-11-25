"""Run each Lab03 exercise in its own runs/<part>/<run_id> directory."""
import argparse
from pathlib import Path
from .config import LAB_DIR, load_config
from .utils import make_run_dir, write_json, track_run, log_event


def run_one(part, settings, root):
    run = make_run_dir(part, root)
    write_json(run / "config_used.json", dict(part=part, **settings))
    print("Run:", run, flush=True)
    with track_run(run, "pipeline") as manifest:
        if part == "customers":
            from .part1_customers.pipeline import run_part
        else:
            from .part2_image.pipeline import run_part
        manifest.update(run_part(run, settings, lambda stage: log_event(run, stage)))
    print("Completed:", run, flush=True)
    return run


def run_pipeline(part="customers", image=None, config_path=None, output_root=None):
    config = load_config(config_path)
    part = {"1": "customers", "2": "image"}.get(str(part), part)
    parts = ("customers", "image") if part == "all" else (part,)
    if any(name not in ("customers", "image") for name in parts):
        raise ValueError("part must be customers, image or all.")
    if "image" in parts:
        paths = image if image is not None else config["image"].get("paths")
        if paths is None:
            paths = [config["image"]["path"]] if config["image"].get("path") else []
        if not paths:
            raise ValueError("Provide at least one image using --image or image.paths.")
        if isinstance(paths, (str, Path)):
            paths = [paths]
        sources = []
        for path in paths:
            source = Path(path)
            if not source.is_absolute():
                source = (Path.cwd() if image else LAB_DIR) / source
            if not source.is_file():
                raise FileNotFoundError(f"Provide an existing image using --image. Missing: {source}")
            sources.append(str(source.resolve()))
        config["image"]["paths"] = list(dict.fromkeys(sources))
        config["image"].pop("path", None)
    root = Path(output_root or LAB_DIR / "runs")
    outputs = {name: run_one(name, config[name], root) for name in parts}
    return outputs if part == "all" else outputs[part]


def main(default_part="1"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=lambda value: {"customers": "1", "image": "2"}.get(value, value),
                        choices=("1", "2", "all"), default={"customers": "1", "image": "2"}.get(default_part, default_part))
    parser.add_argument("--image", nargs="+", help="One or more input image paths")
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        run_pipeline(args.part, args.image, args.config)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
