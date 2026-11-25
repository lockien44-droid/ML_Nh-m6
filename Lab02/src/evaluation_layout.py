"""Grouped CSV outputs for algorithm evaluation, baselines and comparisons."""
from pathlib import Path
from .figure_layout import FIGURE_GROUPS

COMPARISONS = {"metrics.csv", "product_baseline_comparison.csv", "before_after.csv"}


def evaluation_relative(filename):
    if filename in COMPARISONS:
        return Path("comparison") / filename
    stem = Path(filename).stem
    for name, group in FIGURE_GROUPS.items():
        if group in ("model_evaluation", "baselines") and (stem == name or stem.startswith(name + "_")):
            return Path(group) / name / filename
    raise ValueError(f"Unknown evaluation file: {filename}")


def evaluation_path(root, filename):
    path = root / evaluation_relative(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def evaluation_input(run, filename):
    """Read both the grouped layout and previous flat runs."""
    path = run / "evaluation" / evaluation_relative(filename)
    return path if path.is_file() else run / "evaluation" / filename


def organize_evaluation(root):
    """Move known flat outputs into groups, preserving their contents."""
    root = root.resolve()
    for source in list(root.glob("*.csv")):
        try:
            relative = evaluation_relative(source.name)
        except ValueError:
            continue
        target = root / relative
        if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
            raise ValueError("Evaluation path escapes its run directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Conflicting evaluation files: {source} and {target}")
        source.replace(target)
