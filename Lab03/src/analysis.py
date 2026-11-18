"""Segment descriptions, gender distribution and clustering stability."""
import numpy as np
import pandas as pd
from .tuning import tune
from ..algorithms.kmeans import KMeans
from .evaluate import silhouette_samples


def adjusted_rand_score(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Labels must be aligned vectors.")
    if len(left) < 2:
        return 1.0
    _, a = np.unique(left, return_inverse=True)
    _, b = np.unique(right, return_inverse=True)
    table = np.zeros((a.max()+1, b.max()+1), dtype=int)
    np.add.at(table, (a,b), 1)
    pairs = lambda values: np.sum(values*(values-1)/2)
    total = len(left)*(len(left)-1)/2
    observed = pairs(table)
    row_pairs, col_pairs = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = row_pairs*col_pairs/total
    denominator = (row_pairs+col_pairs)/2-expected
    return float((observed-expected)/denominator) if denominator else 1.0


def describe_segments(profiles, features):
    """Give reproducible profile-based names, independent of numeric cluster IDs."""
    profiles = profiles.copy()
    age, income, spending = [f+"_mean" for f in features]
    # Recover the ordinary customer-level mean from cluster aggregates.
    income_cutoff = np.average(profiles[income],weights=profiles["customer_count"])
    names, actions = [], []
    for row in profiles.to_dict("records"):
        if row[spending] >= 65:
            name = "Thu nhập trên TB mẫu – Chi tiêu cao" if row[income] >= income_cutoff else "Thu nhập dưới TB mẫu – Chi tiêu cao"
            action = ("Ưu tiên trải nghiệm VIP, quyền lợi thành viên và bán thêm phù hợp."
                      if row[income] >= income_cutoff else "Thử ưu đãi theo ngân sách và combo giá trị, theo dõi biên lợi nhuận.")
        elif row[spending] <= 35:
            name = "Thu nhập trên TB mẫu – Chi tiêu thấp" if row[income] >= income_cutoff else "Thu nhập dưới TB mẫu – Chi tiêu thấp"
            action = ("Khảo sát nhu cầu, thử ưu đãi cá nhân hóa và đo mức tăng chi tiêu."
                      if row[income] >= income_cutoff else "Thử sản phẩm thiết yếu, ưu đãi tiết kiệm và giới hạn chi phí tiếp cận.")
        else:
            name = "Lớn tuổi – Thu nhập và chi tiêu trung bình" if row[age] >= 40 else "Trẻ – Thu nhập và chi tiêu trung bình"
            action = ("Thử dịch vụ thuận tiện và chương trình duy trì khách hàng."
                      if row[age] >= 40 else "Thử giới thiệu sản phẩm mới và chương trình tích điểm.")
        names.append(name)
        actions.append(action)
    counts = pd.Series(names).value_counts()
    names = [f"{name} (C{cluster})" if counts[name] > 1 else name
             for name, cluster in zip(names, profiles["cluster"])]
    profiles["segment"] = names
    profiles["marketing_hypothesis"] = actions
    return profiles


def gender_distribution(customers):
    counts = pd.crosstab(customers.cluster, customers.Gender)
    shares = counts.div(counts.sum(axis=1), axis=0).add_suffix("_share")
    return counts.join(shares).reset_index()


def stability_analysis(X, settings, seeds, reference, selected_k):
    rows, pair_rows, labels = [], [], []
    for seed in seeds:
        candidate, comparison = tune(X, {**settings, "random_state": int(seed)})
        fixed = KMeans(n_clusters=selected_k, max_iter=settings["max_iter"],
                       tol=settings["tol"], n_init=settings["n_init"], random_state=int(seed)).fit(X)
        labels.append(fixed.labels_)
        rows.append(dict(seed=seed, selected_k=candidate.n_clusters,
                         silhouette=float(silhouette_samples(X,candidate.labels_).mean()),
                         ari_selected_vs_reference=adjusted_rand_score(reference,candidate.labels_),
                         fixed_k=selected_k, ari_fixed_vs_reference=adjusted_rand_score(reference,fixed.labels_)))
    for i in range(len(seeds)):
        for j in range(i+1,len(seeds)):
            pair_rows.append(dict(seed_a=seeds[i],seed_b=seeds[j],
                                 fixed_k=selected_k,ari=adjusted_rand_score(labels[i],labels[j])))
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)
