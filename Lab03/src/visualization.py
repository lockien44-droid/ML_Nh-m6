"""Static K comparison and self-contained interactive 3D customer segments."""
from matplotlib.figure import Figure
import plotly.express as px


def plot_comparison(comparison, destination):
    fig = Figure(figsize=(10, 4), layout="tight")
    axes = fig.subplots(1, 2)
    axes[0].plot(comparison.k, comparison.silhouette, "o-")
    axes[0].set(xlabel="K", ylabel="Mean Silhouette", title="Higher is better")
    axes[1].plot(comparison.k, comparison.inertia, "o-")
    axes[1].set(xlabel="K", ylabel="Inertia", title="Within-cluster squared distances")
    fig.savefig(destination, dpi=160)


def plot_clusters(frame, features, identifier, destination):
    data = frame.copy()
    data["segment"] = "Cluster " + data["cluster"].astype(str)
    fig = px.scatter_3d(data, x=features[0], y=features[1], z=features[2],
                        color="segment", hover_data=[identifier],
                        title="Customer segments — original feature units")
    fig.update_traces(marker_size=5)
    fig.write_html(destination, include_plotlyjs=True, full_html=True)
    return fig
