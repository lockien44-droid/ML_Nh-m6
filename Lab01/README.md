# Lab01 — Phân loại email spam

Lab01 dùng TF-IDF và ba mô hình tự cài đặt để phân loại email thành ham (0) hoặc spam (1). Xem [báo cáo Lab01](docs/report.md) để đọc phương pháp, cấu hình và phân tích lỗi.

## Chạy nhanh

Mở PowerShell tại thư mục gốc repository. Cài thư viện rồi chạy huấn luyện.

```powershell
python -m pip install -r requirements.txt
python -m Lab01.src.train
```

Lệnh huấn luyện làm sạch và loại trùng, chia train/test, tạo TF-IDF, chạy baseline, chọn cấu hình trên validation, rồi fit lại và đánh giá mô hình cuối. Để xem từng bước tiền xử lý, chạy ba lệnh theo thứ tự sau.

```powershell
python -m Lab01.src.preprocessing
python -m Lab01.src.preprocessing --split
python -m Lab01.src.preprocessing --tfidf
```

Sau khi huấn luyện, chạy thử một email thường và một email nhận thưởng.

```powershell
python -m Lab01.src.predict --model logistic_regression --text "Please review the attached agenda for tomorrow team meeting"
python -m Lab01.src.predict --model logistic_regression --text "Congratulations! Claim your free cash prize now"
```

Có thể đổi `--model` thành `svm` hoặc `naive_bayes`. Mô hình cuối và vectorizer được sinh trong `Lab01/models/`.

## Theo dõi từng lần huấn luyện

`python -m Lab01.src.train` in đường dẫn `Lab01/result/runs/<run_id>/`. Mỗi lần chạy có mã riêng và lưu các đầu ra của chính nó.

| Tệp hoặc thư mục trong lượt chạy | Nội dung |
|---|---|
| `run.json` và `events.jsonl` | Trạng thái, thời gian, SHA-256 của CSV đầu vào và nhật ký các giai đoạn |
| `preprocessing/` | Dữ liệu sạch, train/test, TF-IDF, nhãn và thống kê |
| `figures/` | Biểu đồ số dòng qua làm sạch, phân bố nhãn train/test và tỷ lệ ô TF-IDF khác 0 |
| `evaluation/` | Baseline, thử tham số, kết quả test và ma trận nhầm lẫn |
| `models/` | Ba mô hình cuối và bộ TF-IDF của lượt chạy |

Sau khi đánh giá xong, chương trình sao chép đầu ra sang các vị trí quen thuộc để lệnh dự đoán và liên kết báo cáo tiếp tục dùng được. Khi so sánh nhiều lần chạy, mở thư mục theo `run_id`. Git bỏ qua các bản lưu theo lượt vì chúng chứa cả ma trận và mô hình.

## Báo cáo tự động

Lệnh huấn luyện xuất báo cáo Markdown, HTML và PDF trong `Lab01/reports/`. Mỗi báo cáo lấy số liệu từ lượt chạy tương ứng. Các báo cáo cũ chuyển sang `archive/`, còn hình theo lượt nằm ở `figures/<run_id>/{eda,preprocessing,training,evaluation}/`.

| Tệp | Nội dung |
|---|---|
| `metrics_summary.csv` | Kết quả test của ba mô hình, có `Run_ID` để so sánh giữa các lần chạy |
| `inference_history.csv` | Thời gian, mô hình, nhãn và điểm của mỗi lần chạy `predict.py`; không lưu nội dung email |
| `label_distribution.csv` | Số lượng và tỷ lệ nhãn của dữ liệu gốc từ EDA |
| `lr_threshold_comparison.csv` | So sánh năm ngưỡng LR trên validation. Ngưỡng triển khai vẫn là 0,5 |

Nếu cần tạo lại báo cáo từ một lượt đã hoàn tất và có `evaluation/comparison/training.csv`, chạy:

```powershell
python -m Lab01.src.report --run-id <run_id>
```

Mở trực tiếp tệp HTML hoặc PDF khi demo. Bốn ảnh ma trận ở gốc `reports/` là bản tương thích của lượt vừa xuất báo cáo.

## Tài liệu Lab01

- [Mô tả hệ thống và kết quả](docs/Lab01_descript.md)
- [Giải thích tham số](docs/ReadmeForParameters.md)
- [Lịch sử thực nghiệm](docs/experiment_logs.csv)
- [Báo cáo chi tiết](docs/report.md)

## Đầu ra

| Nội dung | Tệp |
|---|---|
| Dữ liệu sạch và tập chia | [cleaned_raw.csv](Data/preprocessing/cleaned_raw.csv), [train_clean.csv](Data/preprocessing/train_clean.csv), [test_clean.csv](Data/preprocessing/test_clean.csv) |
| TF-IDF, nhãn và tên đặc trưng | [09_tfidf/](Data/preprocessing/09_tfidf/) |
| Baseline | [baseline.csv](result/comparison/baseline.csv) |
| Cấu hình và ngưỡng chọn | [selection.json](result/comparison/selection.json), [ngưỡng SVM](result/comparison/svm_study/selection.json) |
| Kết quả test | [metrics.csv](result/comparison/metrics.csv) |

Chạy `python -m Lab01.src.eda` để khảo sát dữ liệu gốc và tạo biểu đồ tổng quan tại `result/eda/overview.png`. Xem [ảnh phân bố nhãn trong tài liệu](docs/images/label_distribution.png).
