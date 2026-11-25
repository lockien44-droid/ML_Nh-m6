"""Compare image reconstructions across K and implementations."""
from ..utils import save_figure
from matplotlib.figure import Figure


def plot_comparison_grid(original, scratch, values, output):
    fig = Figure(figsize=(3 * (len(values) + 1), 3), layout="tight")
    axes = fig.subplots(1, len(values) + 1, squeeze=False)[0]
    axes[0].imshow(original)
    axes[0].set_title("Original")
    for column, k in enumerate(values, 1):
        axes[column].imshow(scratch[k])
        axes[column].set_title(f"NumPy K={k}")
    for ax in axes: ax.axis("off")
    save_figure(fig, output, dpi=130)


def plot_inertia(metrics, output):
    fig = Figure(figsize=(9, 5), layout="tight")
    ax = fig.subplots()
    for image, rows in metrics.groupby("image", sort=False):
        ax.plot(rows.k, rows.inertia, marker="o", label=f"{image} — NumPy")
    ax.set(xlabel="K", ylabel="Inertia in normalized color features", title="Color reconstruction inertia by K")
    ax.legend(fontsize=8)
    save_figure(fig, output, dpi=140)
