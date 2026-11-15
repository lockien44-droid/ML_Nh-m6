"""Export plots and a report from actual experimental outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate(run, results, selection):
    for name in results.model:
        data = pd.read_csv(run / "evaluation" / f"{name}.csv")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(data.sales, data.predicted, alpha=0.25, s=8)
        limit = max(data.sales.max(), data.predicted.max())
        axes[0].plot([0, limit], [0, limit], "--")
        axes[0].set(xlabel="Actual units", ylabel="Predicted units", title=name)
        axes[1].hist(data.residual, bins=40)
        axes[1].set(xlabel="Actual - predicted", ylabel="Count", title="Residual distribution")
        fig.tight_layout()
        fig.savefig(run / "figures" / f"{name}.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    results.plot.bar(x="model", y=["mae", "rmse"], ax=axes[0], rot=35)
    results.plot.bar(x="model", y="fit_seconds", ax=axes[1], rot=35)
    fig.tight_layout()
    fig.savefig(run / "figures/metrics_comparison.png")
    plt.close(fig)
    headers = results.columns.tolist()
    table = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in results.itertuples(index=False, name=None):
        table.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    lines = ["# Lab02 — Kết quả thực nghiệm", "",
             "Dự đoán số lượng bán dương theo sản phẩm/ngày. Chọn mô hình bằng RMSE trung bình của ba cửa sổ mở rộng trên train và validation cuối.",
             "", "## Kết quả test", "", "\n".join(table), "",
             "## Phạm vi đánh giá", "",
             "Đánh giá từng ngày với lịch sử bán thực tế đến hết ngày trước đó. "
             "Các ngày không ghi nhận giao dịch được xem là 0. "
             "Chọn 20 sản phẩm bán nhiều nhất từ train. "
             "Mô hình chỉ áp dụng cho nhóm sản phẩm này và dữ liệu lịch sử cùng nguồn.",
             "", "MAE là sai số tuyệt đối trung bình. MSE là sai số bình phương trung bình. "
             "RMSE là căn MSE, cùng đơn vị số lượng. R² so sánh sai số với dự đoán trung bình của tập đánh giá.",
             "", "## Biểu đồ", ""]
    for name in results.model:
        lines.extend([f"![{name}](../results/runs/{run.name}/figures/{name}.png)", ""])
    lines.extend(["", "## Phân tích sai số", "",
                  "Các bảng *_largest_errors.csv lưu 10 lỗi lớn nhất. Các bảng *_StockCode_metrics.csv và *_weekday_metrics.csv phân tích theo sản phẩm và ngày trong tuần.",
                  "Residual dương là dự báo thiếu, residual âm là dự báo dư.",
                  f"![So sánh](../results/runs/{run.name}/figures/metrics_comparison.png)"])
    return "\n".join(lines)
