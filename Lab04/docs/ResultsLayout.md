# Lab04 — Bố cục kết quả và nguồn ảnh

## Mở đúng nơi theo nhu cầu

| Cần xem | Đường dẫn trong Lab04 |
|---|---|
| Báo cáo chi tiết và giải thích kết quả | docs/Lab04_Report.md |
| Giải thích cấu hình và tham số học | docs/ReadmeForParameters.md |
| Hình minh họa cố định của báo cáo | docs/images/ |
| Toàn bộ đầu ra một lượt | results/<run_id>/ |
| Tình trạng, hash dữ liệu, phiên bản | results/<run_id>/run.json |
| Diễn biến các giai đoạn | results/<run_id>/events.jsonl |
| Train, validation, test và mảng chuẩn hóa | results/<run_id>/preprocessing/ |
| Điểm từng ứng viên trên validation | results/<run_id>/evaluation/validation_metrics.csv |
| Cấu hình và checkpoint được chọn | results/<run_id>/evaluation/selection.json |
| MSE/MAE mỗi epoch | results/<run_id>/evaluation/mlp_history.csv |
| Điểm cuối cùng trên test | results/<run_id>/evaluation/metrics.csv |
| Dự đoán và phần dư từng mẫu | results/<run_id>/evaluation/*_predictions.csv |
| Mô hình, scaler và schema của lượt | results/<run_id>/models/ |
| Biểu đồ sinh tự động | results/<run_id>/figures/ |
| Báo cáo tự động của lượt | results/<run_id>/report.md |

## Nguồn hình trong tài liệu

Lượt tham chiếu là [20261123_135712_mlp_comparison](../results/20261123_135712_mlp_comparison/). Sáu ảnh learning_curve và *_predictions trong docs/images là bản sao nguyên vẹn từ figures của lượt này, giúp tài liệu có hình minh họa cố định.

metrics_comparison.png được dựng từ evaluation/metrics.csv. data_overview.png dùng split_counts trong run.json và target của preprocessing/train.csv. Hai ảnh bổ sung dùng số liệu đã lưu.

Mỗi lượt pipeline mới tạo figures riêng trong results. Tài liệu docs hiện tham chiếu lượt nêu trên. Khi cập nhật báo cáo sang lượt khác, cần đồng bộ số liệu, lựa chọn, nguồn hình và tên lượt.

## Đọc lịch sử học

mlp_candidate_0_history.csv và mlp_candidate_1_history.csv ghi hai ứng viên. mlp_history.csv là lịch sử ứng viên được chọn. Checkpoint mlp.pt có thể thuộc epoch trước hàng cuối history vì early stopping phục hồi epoch có validation MSE thấp nhất.

## Tạo lại hình và báo cáo tự động từ kết quả đã lưu

Chạy từ thư mục gốc repository:

```powershell
python -c "from pathlib import Path; from Lab04.src.report import generate; generate(Path('Lab04/results/20261123_135712_mlp_comparison'))"
```

Lệnh đọc các tệp evaluation để ghi lại figures và report.md của lượt tương ứng.

[Quay lại báo cáo](Lab04_Report.md)
