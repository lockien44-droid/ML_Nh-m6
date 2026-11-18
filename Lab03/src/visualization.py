"""Static K comparison and self-contained interactive 3D customer segments."""
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


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
    if "segment" not in data:
        data["segment"] = "Cluster " + data["cluster"].astype(str)
    fig = px.scatter_3d(data, x=features[0], y=features[1], z=features[2],
                        color="segment", hover_data=[identifier],
                        title="Customer segments — original feature units")
    fig.update_traces(marker_size=5)
    centers = data.groupby(["cluster", "segment"])[features].mean().reset_index()
    fig.add_trace(go.Scatter3d(x=centers[features[0]], y=centers[features[1]],
                              z=centers[features[2]], mode="markers+text",
                              marker=dict(size=10, symbol="diamond", color="black"),
                              text=centers["segment"], name="Centroids"))
    fig.write_html(destination, include_plotlyjs=True, full_html=True)
    return fig


def plot_income_spending(customers, features, destination):
    fig = Figure(figsize=(11,6),layout="tight")
    ax=fig.subplots()
    for name, group in customers.groupby("segment"):
        ax.scatter(group[features[1]],group[features[2]],label=name,s=25)
        ax.scatter(group[features[1]].mean(),group[features[2]].mean(),marker="X",c="black",s=80)
    ax.set(xlabel=features[1],ylabel=features[2],title="Income vs spending; X = centroid")
    ax.legend(fontsize=7)
    fig.savefig(destination,dpi=160)


def plot_silhouette(customers, destination):
    fig=Figure(figsize=(10,6),layout="tight")
    ax=fig.subplots()
    start=0
    for cluster, group in customers.groupby("cluster"):
        values=np.sort(group.silhouette.to_numpy())
        ax.fill_betweenx(np.arange(start,start+len(values)),0,values,alpha=.7)
        ax.text(-.3,start+len(values)/2,f"C{cluster} (n={len(values)})",fontsize=8)
        start+=len(values)+8
    ax.axvline(customers.silhouette.mean(),color="red",linestyle="--",label="Mean")
    ax.set(xlabel="Silhouette",ylabel="Sorted customers by cluster",xlim=(-.4,1))
    ax.legend()
    fig.savefig(destination,dpi=160)
