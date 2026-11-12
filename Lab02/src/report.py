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
    lines = ["# Lab02 — Kết quả thực nghiệm", "",
             "Dự đoán số lượng bán dương theo sản phẩm/ngày. Chọn mô hình bằng RMSE validation.",
             "", "## Kết quả test", "", results.to_csv(index=False), "",
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
    return "\n".join(lines)
