"""Shared layout for per-run analysis figures."""
from pathlib import Path

FIGURE_GROUPS = {
    "daily_sales": "eda",
    "eda_sales_distribution": "eda",
    "product_statistics": "eda",
    "chronological_split": "preprocessing",
    "tuning_selection": "tuning",
    "linear_regression": "model_evaluation",
    "decision_tree": "model_evaluation",
    "random_forest": "model_evaluation",
    "gradient_boosting": "model_evaluation",
    "mean_baseline": "baselines",
    "last_week_baseline": "baselines",
    "rolling_7_baseline": "baselines",
    "rolling_28_baseline": "baselines",
    "weekday_4_baseline": "baselines",
    "product_mean_baseline": "baselines",
    "metrics_comparison": "comparison",
    "product_rmse_comparison": "comparison",
}


def figure_relative(filename):
    return Path(FIGURE_GROUPS[Path(filename).stem]) / filename


def figure_path(root, filename):
    path = root / figure_relative(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def organize_figures(root):
    """Migrate known generated files within the same figures directory."""
    root = root.resolve()
    for source in list(root.rglob("*")):
        if not source.is_file() or source.suffix not in (".png", ".csv") or source.stem not in FIGURE_GROUPS:
            continue
        target = figure_path(root, source.name)
        if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
            raise ValueError("Figure path escapes its run directory.")
        if source != target:
            source.replace(target)
