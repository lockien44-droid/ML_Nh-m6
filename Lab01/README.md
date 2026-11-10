# Lab01 — Phân loại email spam

## Tổng quan

Lab01 dùng TF-IDF và ba thuật toán tự cài đặt để phân loại email thành ham (`0`) hoặc spam (`1`).

## Thành viên nhóm

1. Nguyễn Thị Thanh Thùy
2. Nguyễn Văn Minh
3. Lê Hồng Sơn
4. Phan Đình Hưng
5. Phạm Minh Tiến
6. Võ Thị Thanh Tú

## Cấu trúc thư mục

```text
Lab01/
├── algorithms/                        # Cài đặt ba thuật toán phân loại
│   ├── logistic_regression.py         # Học bằng gradient descent, dự đoán từ xác suất spam
│   ├── svm.py                         # Linear SVM, dự đoán bằng điểm quyết định và ngưỡng
│   ├── naive_bayes.py                 # Multinomial Naive Bayes với hệ số làm trơn alpha
│   └── __init__.py                    # Xuất các lớp mô hình để pipeline import
├── Data/
│   ├── raw/
│   │   └── spam_ham_dataset.csv       # Email gốc và nhãn ham/spam
│   └── preprocessing/                # Bản dữ liệu tiền xử lý được cập nhật khi chạy pipeline
│       ├── cleaned_raw.csv            # Email đã làm sạch và loại trùng
│       ├── train_clean.csv            # Tập train ban đầu, dùng để tách validation
│       ├── test_clean.csv             # Tập test dùng để đánh giá mô hình cuối
│       ├── summary.json               # Số email qua các bước lọc và làm sạch
│       ├── split_tfidf_summary.json   # Phân bố nhãn, kích thước và mật độ ma trận TF-IDF
│       └── 09_tfidf/                  # Ma trận thưa .npz, nhãn và tên đặc trưng
├── src/
│   ├── pipeline.py                    # Điều phối tiền xử lý, baseline, tuning, đánh giá và báo cáo
│   ├── config.yaml                    # Tỷ lệ, seed chia validation và siêu tham số cần thử
│   ├── preprocessing.py              # Làm sạch, kiểm tra nhãn, chia train/test và tạo TF-IDF
│   ├── tuning.py                     # Chia validation, chọn cấu hình và ngưỡng SVM theo Macro-F1
│   ├── train.py                      # Ghi trace huấn luyện baseline và giữ lệnh chạy cũ
│   ├── evaluate.py                   # Tính chỉ số, lưu dự đoán và vẽ ma trận nhầm lẫn
│   ├── run_report.py                 # Tạo mã lần chạy, ghi nhật ký và xuất hình tiền xử lý
│   ├── report.py                     # Tổng hợp kết quả và xuất Markdown, HTML, PDF
│   ├── eda.py                        # Khảo sát nhãn, độ dài email và dữ liệu thiếu
│   ├── predict.py                    # Nạp mô hình và TF-IDF để dự đoán email mới
│   └── __init__.py                   # Khai báo package src
├── models/                           # Mô hình dùng cho lệnh predict
│   ├── logistic_regression.joblib     # Logistic Regression đã huấn luyện
│   ├── svm.joblib                     # Linear SVM đã huấn luyện, kèm ngưỡng quyết định
│   ├── naive_bayes.joblib             # Multinomial Naive Bayes đã huấn luyện
│   ├── tfidf_vectorizer.joblib        # Bộ từ vựng và IDF đã học
│   └── run_id.txt                     # Mã lần chạy đã tạo các mô hình đang lưu
├── results/
│   └── runs/<run_id>/                 # Dữ liệu, mô hình và kết quả của từng lần huấn luyện
├── reports/                          # Báo cáo tự động, hình ảnh và các bảng lịch sử
├── docs/
│   ├── Lab01_Report.md                # Báo cáo chi tiết theo 12 bước
│   ├── ReadmeForParameters.md         # Giải thích các tham số
│   ├── experiment_logs.csv           # Nhật ký thực nghiệm trong tài liệu
│   └── images/                       # Hình minh họa trong tài liệu
├── notebook/                         # Thư mục dành cho notebook thực nghiệm
└── README.md                         # Cấu trúc và cách chạy Lab01
```

### Đầu ra của một lần chạy

Các mục dưới đây được tạo khi chạy pipeline. `<run_id>` là mã được in trong Terminal.

```text
Lab01/results/runs/<run_id>/
├── run.json                           # Thời gian, trạng thái và SHA-256 của dữ liệu đầu vào
├── events.jsonl                       # Nhật ký các giai đoạn của lần chạy
├── preprocessing/                    # Dữ liệu sạch, train/test và TF-IDF của lần chạy
├── figures/                          # Biểu đồ làm sạch, chia tập và mật độ TF-IDF
├── models/                           # Bản mô hình và vectorizer của lần chạy
└── evaluation/
    ├── comparison/
    │   ├── baseline.csv               # Kết quả test của các cấu hình mặc định
    │   ├── subtrain_clean.csv         # Email dùng để huấn luyện thử
    │   ├── validation_clean.csv       # Email dùng để chọn cấu hình và khảo sát ngưỡng
    │   ├── validation.csv             # Chỉ số validation của các cấu hình đã thử
    │   ├── selection.json             # Cách chia validation và siêu tham số đã chọn
    │   ├── svm_study/                 # Khảo sát trọng số lớp và ngưỡng SVM trên validation
    │   ├── training.csv               # Cấu hình và thời gian fit của mô hình cuối
    │   └── metrics.csv                # So sánh chỉ số test của ba mô hình cuối
    └── <tên_mô_hình>/
        ├── metrics.json               # Chỉ số tổng hợp và số lỗi FP/FN
        ├── classification_report.csv  # Precision, Recall và F1 theo lớp
        ├── predictions.csv            # Mã email, nhãn thật và nhãn dự đoán trên test
        └── confusion_matrix.png       # Ma trận nhầm lẫn trên test
```

Trong `svm_study/validation.csv`, cột `confusion_matrix` ghi ma trận trên **validation** ở ngưỡng 0 và ngưỡng tốt nhất của từng cấu hình trọng số lớp. Ảnh `confusion_matrix.png` của mô hình cuối dùng **test**.

## Luồng xử lý

1. Làm sạch email, kiểm tra nhãn và loại trùng, rồi chia train/test theo tỷ lệ 80/20.
2. Tạo TF-IDF từ tập train và chạy các mô hình mặc định để lấy kết quả baseline.
3. Tách validation từ train theo tỷ lệ và seed trong `config.yaml`. Fit TF-IDF trên phần train thử rồi dùng bộ từ vựng đó để biến đổi validation.
4. So sánh cấu hình theo Macro-F1 trên validation. Logistic Regression thử learning rate và số vòng lặp, SVM thử thêm `C`, trọng số lớp và ngưỡng điểm, còn Naive Bayes thử hệ số làm trơn `alpha`. Logistic Regression dùng ngưỡng dự đoán 0,5.
5. Fit lại TF-IDF và ba mô hình trên toàn bộ train ban đầu. Đánh giá trên test, lưu mô hình, ma trận nhầm lẫn và báo cáo.

## Cài đặt và chạy

Mở PowerShell tại thư mục gốc của dự án:

```powershell
python -m pip install -r requirements.txt
python -m Lab01.src.pipeline
```

Lệnh `python -m Lab01.src.train` cũng chạy cùng pipeline.

## Kết quả và demo

Kết quả trên 977 email test từ lần chạy đã lưu `20261106T172203Z-a2e0309c`. Naive Bayes trong lần chạy này dùng `alpha=1`. Khi chạy lại pipeline hiện tại, `alpha` được chọn trên validation.

| Mô hình | Accuracy | Precision spam | Recall spam | F1 spam | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 99,18% | 98,29% | 98,97% | 98,63% | 99,02% |
| Linear SVM | 96,93% | 96,44% | 93,13% | 94,76% | 96,29% |
| Multinomial Naive Bayes | 95,70% | 98,82% | 86,60% | 92,31% | 94,66% |

Sau khi chạy pipeline, mở `evaluation/comparison/metrics.csv` trong thư mục lần chạy mới để xem kết quả tương ứng.

Thử dự đoán một email mới:

```powershell
python -m Lab01.src.predict --model svm --text "Congratulations! Claim your free cash prize now"
```

Tham số `--model` nhận `logistic_regression`, `svm` hoặc `naive_bayes`.

## Tài liệu chi tiết
Báo cáo tự động nằm trong `reports/`, hình theo lần chạy nằm trong `reports/figures/<run_id>/`. File `metrics_summary.csv` tổng hợp chỉ số test theo mã lần chạy, còn `inference_history.csv` ghi lịch sử dự đoán.

Để tạo lại báo cáo từ một lần chạy đã hoàn tất, thay `RUN_ID` bằng mã thư mục tương ứng:

```powershell
python -m Lab01.src.report --run-id RUN_ID
```

Xem thêm [báo cáo chi tiết theo 12 bước](docs/Lab01_Report.md) và [giải thích các tham số](docs/ReadmeForParameters.md).
