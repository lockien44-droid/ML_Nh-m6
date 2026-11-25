"""Inspect missingness, feature distributions and IQR outlier flags."""
from ..utils import save_figure
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def analyze(frame, features, output):
    output = output / "eda"
    output.mkdir(parents=True, exist_ok=True)
    frame.isna().sum().rename("missing_count").to_csv(output / "missing_values.csv")
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    numeric.describe().to_csv(output / "feature_statistics.csv")
    q1, q3 = numeric.quantile(.25), numeric.quantile(.75)
    flagged = (numeric < q1 - 1.5 * (q3 - q1)) | (numeric > q3 + 1.5 * (q3 - q1))
    flags = pd.DataFrame({"feature":features, "iqr_outlier_count":flagged.sum().to_numpy()})
    flags.to_csv(output / "outlier_summary.csv", index=False)
    plot_gender_eda(frame, numeric, output)
    plot_separate_gender_figures(frame, numeric, output)
    plot_relationships(frame, numeric, output)
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    for csv in output.glob("*.csv"):
        csv.replace(tables / csv.name)
    return flags


def plot_gender_eda(frame, numeric, output):
    """Gender-stratified distributions and correlations, inspired by the supplied notebook."""
    features = list(numeric.columns)
    corr = numeric.corr()
    corr.to_csv(output / "eda_correlations.csv")
    fig = Figure(figsize=(8, 6), layout="tight")
    ax = fig.subplots()
    heat = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features, rotation=25, ha="right")
    ax.set_yticks(range(len(features)), features)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
    ax.set_title("Pearson correlation — numeric features")
    fig.colorbar(heat, ax=ax)
    save_figure(fig, output / "eda_correlations.png", dpi=150)
    if "Gender" not in frame:
        return
    gender = frame["Gender"].astype("string").str.strip().str.title().fillna("Missing")
    groups = sorted(gender.unique())
    colors = ["#2563eb", "#db2777", "#15803d", "#b45309"]
    counts = gender.value_counts().reindex(groups)
    counts.rename_axis("Gender").rename("count").to_csv(output / "eda_gender_counts.csv")
    data = numeric.assign(Gender=gender)
    data.groupby("Gender")[features].agg(["count", "mean", "median", "std"]).to_csv(
        output / "eda_gender_statistics.csv")
    fig = Figure(figsize=(7, 5), layout="tight")
    ax = fig.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors[:len(groups)],
           wedgeprops=dict(width=.55, edgecolor="white"), startangle=90)
    ax.text(0, 0, f"N={len(frame)}", ha="center", va="center")
    ax.set_title("Customer gender distribution")
    save_figure(fig, output / "eda_gender_distribution.png", dpi=150)
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig = Figure(figsize=(15, 4.5), layout="tight")
    for ax, (left, right) in zip(fig.subplots(1, 3), pairs):
        for index, group in enumerate(groups):
            subset = numeric.loc[gender == group]
            ax.scatter(subset.iloc[:, left], subset.iloc[:, right], s=22, alpha=.7,
                       color=colors[index % len(colors)], label=group)
        ax.set(xlabel=features[left], ylabel=features[right])
        ax.legend(title="Gender")
    save_figure(fig, output / "eda_feature_pairs_by_gender.png", dpi=150)


def plot_separate_gender_figures(frame, numeric, output):
    """Separate presentation figures matching the supplied EDA examples."""
    if "Gender" not in frame:
        return
    gender = frame["Gender"].astype("string").str.strip().str.title()
    groups = ["Male", "Female"]
    names = ["Nam", "Nữ"]
    colors = ["#92b9f5", "#e5b7f6"]
    specifications = [
        ("Age", "Tuổi", np.arange(15, 76, 5), "eda_age_by_gender.png", False),
        ("Annual Income (k$)", "Thu nhập hằng năm (k$)", np.arange(10, 151, 10), "eda_income_by_gender.png", True),
        ("Spending Score (1-100)", "Điểm chi tiêu (1–100)", np.arange(0, 106, 5), "eda_spending_by_gender.png", True),
    ]
    for feature, label, bins, filename, box in specifications:
        if feature not in numeric:
            continue
        fig = Figure(figsize=(15 if box else 11, 4.5), layout="tight")
        axes = fig.subplots(1, 3 if box else 2, squeeze=False)[0]
        samples = []
        for index, group in enumerate(groups):
            values = numeric.loc[gender == group, feature].dropna()
            samples.append(values)
            ax = axes[index]
            ax.hist(values, bins=bins, color=colors[index], edgecolor="#737373")
            ax.set(title=names[index], xlabel=label, ylabel="Số khách hàng", xlim=(bins[0], bins[-1]))
            stats = f"Số lượng: {len(values)}\nTrung bình: {values.mean():.1f}"
            if box:
                stats += f"\nTrung vị: {values.median():.1f}\nĐộ lệch chuẩn: {values.std(ddof=1):.1f}"
            ax.text(.97, .97, stats, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        ymax = max(ax.get_ylim()[1] for ax in axes[:2])
        for ax in axes[:2]: ax.set_ylim(0, ymax * 1.12)
        if box:
            artists = axes[2].boxplot(samples, tick_labels=names, patch_artist=True)
            for patch, color in zip(artists["boxes"], colors): patch.set_facecolor(color)
            axes[2].set(title=f"Boxplot {label.lower()}", xlabel="Giới tính", ylabel=label)
        save_figure(fig, output / filename, dpi=150)
    if not {"Age", "Annual Income (k$)"}.issubset(numeric.columns):
        return
    ages = pd.cut(numeric["Age"], bins=np.arange(15, 71, 5), right=True)
    table = numeric.assign(Gender=gender, age_group=ages).groupby(
        ["age_group", "Gender"], observed=False)["Annual Income (k$)"].agg(["mean", "count"])
    table.to_csv(output / "eda_income_by_age_gender.csv")
    means = table["mean"].unstack("Gender").reindex(columns=groups)
    fig = Figure(figsize=(13, 5), layout="tight")
    ax = fig.subplots()
    positions = np.arange(len(means))
    for index, group in enumerate(groups):
        ax.bar(positions + (index - .5) * .38, means[group], width=.38,
               label=names[index], color=colors[index], edgecolor="#555555", linewidth=.5)
    ax.set_xticks(positions, [str(interval) for interval in means.index], rotation=25, ha="right")
    ax.set(title="Thu nhập trung bình hằng năm theo nhóm tuổi và giới tính",
           xlabel="Nhóm tuổi", ylabel="Thu nhập trung bình (k$/năm)")
    ax.legend(title="Giới tính")
    save_figure(fig, output / "eda_income_by_age_gender.png", dpi=150)


def plot_relationships(frame, numeric, output):
    """Joint distributions and descriptive least-squares trends from the EDA reference.

    Regression lines summarize association; they are not clustering models or forecasts.
    """
    age, income, spending = "Age", "Annual Income (k$)", "Spending Score (1-100)"
    if not {age, income, spending}.issubset(numeric.columns):
        return
    pairs = numeric[[age, spending]].dropna()
    fig = Figure(figsize=(8, 7), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1])
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    right = fig.add_subplot(grid[1, 1], sharey=ax)
    x, y = pairs[age].to_numpy(), pairs[spending].to_numpy()
    ax.scatter(x, y, s=25, alpha=.65, color="#287c8e")
    if len(x) > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line = np.linspace(x.min(), x.max(), 100)
        ax.plot(line, slope * line + intercept, color="#dc2626", label="Xu hướng tuyến tính")
        ax.legend(loc="lower left")
    r = pairs.corr().iloc[0, 1]
    ax.set(xlabel="Tuổi", ylabel="Điểm chi tiêu (1–100)", title=f"Tuổi – chi tiêu | Pearson r = {r:.2f}")
    top.hist(x, bins=12, color="#92b9f5", edgecolor="white")
    top.set_ylabel("Số khách")
    top.tick_params(labelbottom=False)
    right.hist(y, bins=12, orientation="horizontal", color="#92b9f5", edgecolor="white")
    right.set_xlabel("Số khách")
    right.tick_params(labelleft=False)
    save_figure(fig, output / "eda_age_spending_joint.png", dpi=150)
    if "Gender" not in frame:
        return
    gender = frame["Gender"].astype("string").str.strip().str.title()
    fig = Figure(figsize=(10, 6), layout="tight")
    ax = fig.subplots()
    statistics = []
    for group, name, color in [("Male", "Nam", "#6097e3"), ("Female", "Nữ", "#c878dc")]:
        subset = numeric.loc[gender == group, [age, income]].dropna()
        r = subset.corr().iloc[0, 1]
        ax.scatter(subset[age], subset[income], color=color, alpha=.7, s=28,
                   label=f"{name} (n={len(subset)}, r={r:.2f})")
        if len(subset) > 1 and subset[age].nunique() > 1:
            slope, intercept = np.polyfit(subset[age], subset[income], 1)
            line = np.linspace(subset[age].min(), subset[age].max(), 100)
            ax.plot(line, slope * line + intercept, color=color, linewidth=2)
        statistics.append(dict(Gender=group, n=len(subset), pearson_age_income=r))
    pd.DataFrame(statistics).to_csv(output / "eda_relationship_statistics.csv", index=False)
    ax.set(title="Tuổi và thu nhập theo giới tính", xlabel="Tuổi", ylabel="Thu nhập hằng năm (k$)")
    ax.legend(title="Giới tính và tương quan Pearson")
    save_figure(fig, output / "eda_age_income_regression.png", dpi=150)


def main():
    """Generate the documentation EDA figures without tuning or training."""
    import argparse
    from pathlib import Path
    from ..config import LAB_DIR, load_config
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ EDA khách hàng Lab03.")
    parser.add_argument("--input", type=Path, help="CSV khách hàng; mặc định theo config.yaml")
    parser.add_argument("--output", type=Path, default=LAB_DIR / "docs/images",
                        help="Thư mục cha; hình lưu trong eda/, bảng trong eda/tables/")
    args = parser.parse_args()
    settings = load_config()["customers"]["data"]
    source = args.input if args.input is not None else LAB_DIR / settings["path"]
    analyze(pd.read_csv(source), settings["features"], args.output)
    print(f"EDA saved: {(args.output / 'eda').resolve()}")


if __name__ == "__main__":
    main()
