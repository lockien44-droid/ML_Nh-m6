"""Compute exact Euclidean Silhouette and original-scale cluster profiles."""
import numpy as np


def silhouette_samples(X, labels):
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    if X.ndim != 2 or labels.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("Finite features and aligned labels are required.")
    groups = np.unique(labels)
    if not 2 <= len(groups) < len(X):
        raise ValueError("Silhouette requires 2 to n-1 observed clusters.")
    norms = np.sum(X*X,axis=1)
    distances = np.sqrt(np.maximum(norms[:,None]+norms[None,:]-2*(X @ X.T),0))
    np.fill_diagonal(distances,0)
    scores = np.zeros(len(X))
    for index in range(len(X)):
        own = labels == labels[index]
        if own.sum() == 1:
            continue
        a = distances[index, own].sum() / (own.sum() - 1)
        b = min(distances[index, labels == group].mean()
                for group in groups if group != labels[index])
        denominator = max(a, b)
        scores[index] = (b - a) / denominator if denominator > 0 else 0
    return scores


def cluster_profiles(frame, labels, features):
    data = frame.assign(cluster=labels)
    means = data.groupby("cluster")[features].mean().add_suffix("_mean")
    means.insert(0, "customer_count", data.groupby("cluster").size())
    return means.reset_index()


def davies_bouldin_score(X, labels):
    groups = [X[labels == label] for label in np.unique(labels)]
    centers = np.array([g.mean(axis=0) for g in groups])
    scatter = np.array([np.linalg.norm(g-c, axis=1).mean() for g,c in zip(groups,centers)])
    distances = np.linalg.norm(centers[:,None]-centers[None,:],axis=2)
    if np.allclose(scatter,0) or np.allclose(distances,0): return 0.0
    distances[distances == 0] = np.inf
    return np.max((scatter[:,None]+scatter[None,:])/distances,axis=1).mean()


def calinski_harabasz_score(X, labels):
    groups = [X[labels == label] for label in np.unique(labels)]
    between = sum(len(g)*np.sum((g.mean(axis=0)-X.mean(axis=0))**2) for g in groups)
    within = sum(np.sum((g-g.mean(axis=0))**2) for g in groups)
    return between*(len(X)-len(groups))/(within*(len(groups)-1)) if within else 1.0


def score_clustering(X, labels):
    """Score observed non-noise clusters and report how much data is covered."""
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    mask = labels != -1
    groups = np.unique(labels[mask])
    result = dict(k=len(groups), coverage=float(mask.mean()), noise_count=int((~mask).sum()),
                  silhouette=float("nan"), inertia=float("nan"),
                  davies_bouldin=float("nan"), calinski_harabasz=float("nan"))
    if 2 <= len(groups) < mask.sum():
        observed, assignments = X[mask], labels[mask]
        result.update(silhouette=float(silhouette_samples(observed, assignments).mean()),
                      inertia=float(sum(np.sum((observed[assignments == group] -
                                      observed[assignments == group].mean(axis=0)) ** 2)
                                      for group in groups)),
                      davies_bouldin=float(davies_bouldin_score(observed, assignments)),
                      calinski_harabasz=float(calinski_harabasz_score(observed, assignments)))
    return result


def evaluate_run(run):
    """Reload the saved clustering model and evaluate the same prepared customer data."""
    from hashlib import sha256
    import joblib
    import pandas as pd
    from .train import load_features
    from .analysis import describe_segments, gender_distribution
    from .visualization import plot_clusters, plot_income_spending, plot_silhouette
    from ..utils import read_json, write_json
    X, config = load_features(run)
    training = read_json(run / "models/training.json")
    if sha256(X.tobytes()).hexdigest() != training["feature_sha256"]:
        raise ValueError("Prepared features differ from the data used to fit this model.")
    if read_json(run / "tuning/selection.json") != training["selection"]:
        raise ValueError("The selected configuration differs from the saved model.")
    model = joblib.load(run / "models/clustering.joblib")
    labels = np.asarray(model.labels_)
    if labels.shape != (len(X),):
        raise ValueError("Saved labels do not align with customer rows.")
    frame = pd.read_csv(run / "preprocessing/customers_clean.csv")
    scaled = pd.read_csv(run / "preprocessing/customers_scaled.csv", float_precision="round_trip")
    identifier = config["data"]["id_column"]
    if not np.array_equal(frame[identifier].to_numpy(), scaled[identifier].to_numpy()):
        raise ValueError("Customer row identifiers do not align.")
    scaler = joblib.load(run / "models/scaler.joblib")
    transformed = (frame[scaler["features"]].to_numpy(dtype=float) - scaler["mean"]) / scaler["scale"]
    if not np.allclose(transformed, X, rtol=1e-12, atol=1e-12):
        raise ValueError("Original customer features do not match the stored scaling.")
    scores = score_clustering(X, labels)
    values = np.full(len(X), np.nan)
    observed = labels != -1
    if 2 <= scores["k"] < observed.sum():
        values[observed] = silhouette_samples(X[observed], labels[observed])
    customers = frame.assign(cluster=labels, silhouette=values)
    features = config["data"]["features"]
    profiles = describe_segments(cluster_profiles(frame, labels, features), features)
    noise = profiles.cluster == -1
    profiles.loc[noise, "segment"] = "Noise"
    profiles.loc[noise, "marketing_hypothesis"] = "Review noise observations before assigning a segment."
    customers = customers.merge(profiles[["cluster", "segment"]], on="cluster", validate="many_to_one")
    directory = run / "results"
    directory.mkdir(exist_ok=True)
    customers.to_csv(directory / "customers_labeled.csv", index=False)
    profiles.to_csv(directory / "cluster_profile.csv", index=False)
    gender_distribution(customers).to_csv(directory / "gender_by_cluster.csv", index=False)
    pd.DataFrame([dict(algorithm=training["algorithm"], **scores)]).to_csv(directory / "metrics.csv", index=False)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    plot_income_spending(customers, features, figures / "income_spending.png")
    plot_silhouette(customers, figures / "silhouette_clusters.png")
    plot_clusters(customers, features, identifier, figures / "customer_clusters_3d.html")
    report = ["# Phân cụm khách hàng", "", f"Thuật toán được chọn: {training['algorithm']}.",
              f"Số cụm: {scores['k']}; Silhouette: {scores['silhouette']:.4f}; coverage: {scores['coverage']:.2%}.",
              "Đánh giá trên chính dữ liệu đã phân cụm. Với DBSCAN, chỉ số loại noise và báo riêng coverage.",
              "", "![Silhouette](figures/silhouette_clusters.png)",
              "", "![Thu nhập và chi tiêu](figures/income_spending.png)"]
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(pd.DataFrame([scores]).to_string(index=False))
    return dict(algorithm=training["algorithm"], selected_k=scores["k"],
                silhouette=scores["silhouette"], coverage=scores["coverage"])


def main():
    from ..utils import customer_step_main
    customer_step_main(evaluate_run, "evaluation")


if __name__ == "__main__":
    main()
