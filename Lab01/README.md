# Lab01 — Phân loại email spam

## Tổng quan

TF-IDF và ba thuật toán tự cài Logistic Regression, Linear SVM, Multinomial Naive Bayes. Nhãn 0 là ham, 1 là spam. Scikit-learn hỗ trợ TF-IDF, chia tập và chỉ số; thuật toán phân loại được tự cài.

## Thành viên nhóm

- Vũ Thị Thanh Thùy
- Nguyễn Văn Minh
- Lê Hồng Sơn
- Phan Đình Hưng
- Phạm Minh Tiến
- Võ Thị Anh Tú

## Cấu trúc thư mục

```text
Lab01/
├── algorithms/                   # Mỗi thuật toán một lớp, fit và predict
├── Data/
│   ├── raw/                      # Email gốc có nhãn
│   └── preprocessing/            # Bản dữ liệu đã xử lý dùng chung
├── src/
│   ├── config.yaml               # Chia tập, TF-IDF, cấu hình cần thử
│   ├── preprocessing.py          # Làm sạch, loại trùng và chia train/test
│   ├── train.py                  # Fit và ghi training/training_summary.csv
│   ├── tuning.py                 # Chia validation, chọn cấu hình/ngưỡng
│   ├── evaluate.py               # Chỉ số, ma trận và lỗi từng email
│   ├── experiment_tracking.py    # Run ID, trạng thái, phiên bản và nhật ký
│   ├── eda.py                    # Khảo sát và báo cáo MD/HTML/PDF
│   ├── predict.py                # Nạp cặp mô hình + TF-IDF qua model_info
│   └── pipeline.py               # Điều phối các bước và lưu kết quả
├── models/                       # baseline/, tuned/, vectorizers/, model_info.json
├── results/<run_id>/             # Đầu ra riêng của mỗi lần thực nghiệm
├── reports/                      # Báo cáo, hình và bảng lịch sử tổng hợp
├── docs/                         # Báo cáo 12 bước, tham số, cấu trúc đầu ra
└── notebooks/                    # Trình bày và kiểm chứng kết quả từng bước
```

### Cấu trúc một lần chạy

```text
results/<run_id>/
├── config.yaml                         # Cấu hình của lần chạy
├── run.json                            # Trạng thái, hash, phiên bản và selected_fits
├── events.jsonl                        # Nhật ký các giai đoạn
├── preprocessing/
│   ├── cleaned_raw.csv                 # Email sạch, đã loại trùng
│   ├── train_clean.csv                 # Train ban đầu = subtrain + validation
│   ├── subtrain_clean.csv
│   ├── validation_clean.csv
│   ├── test_clean.csv
│   ├── summary.json
│   ├── split_tfidf_summary.json
│   └── tfidf/<vectorizer_id>/
│       ├── X_subtrain.npz              # Ma trận CSR
│       ├── X_validation.npz
│       ├── X_test.npz
│       ├── y_subtrain.csv              # email_id và nhãn, theo đúng thứ tự X
│       ├── y_validation.csv
│       ├── y_test.csv
│       ├── feature_names.csv
│       └── metadata.json
├── figures/                            # EDA, làm sạch, chia tập, mật độ TF-IDF
├── training/
│   ├── training_summary.csv           # Một dòng cho mỗi fit_id, đủ cả 49 lần học
│   ├── logistic_regression/
│   │   ├── history.csv                # Lịch sử mọi fit Logistic, có cột fit_id
│   │   └── selected_training.png      # Chỉ biểu đồ cấu hình được chọn
│   └── svm/
│       ├── history.csv                # Lịch sử mọi fit SVM, có cột fit_id
│       └── selected_training.png
├── tuning/
│   ├── validation_summary.csv          # Mỗi lần fit một dòng, gồm baseline
│   ├── selection.json                  # Cấu hình/ngưỡng và fit_id được chọn
│   ├── validation/<model>/<variant>/   # baseline hoặc selected, trên validation
│   └── threshold_search/<fit_id>/
│       ├── thresholds.csv              # Một ngưỡng một dòng
│       ├── selection.json              # So ngưỡng mặc định và ngưỡng được chọn
│       └── thresholds.png
├── models/
│   ├── baseline/<model>.joblib
│   ├── tuned/<model>.joblib
│   ├── vectorizers/<vectorizer_id>.joblib
│   ├── vectorizers/<vectorizer_id>.json # Thiết lập thực tế, dấu vết subtrain
│   └── model_info.json                 # Model + variant → fit_id + vectorizer_id
└── evaluation/
    ├── <model>/<variant>/              # baseline hoặc tuned, chỉ trên test
    │   ├── test_metrics.json
    │   ├── classification_report.csv
    │   ├── confusion_matrix.png
    │   ├── predictions.csv             # ID, nhãn thật/dự đoán, score, score_type
    │   └── errors.csv
    └── comparison/test_summary.csv     # Sáu hàng cho ba thuật toán × hai phiên bản

```

## Luồng xử lý

1. Khảo sát, làm sạch và loại trùng email.
2. Chia train ban đầu/test theo 80/20, rồi tách 20% train ban đầu làm validation.
3. Học TF-IDF chỉ trên subtrain. Ghi vectorizer_id và ma trận từng tập.
4. Fit baseline và các cấu hình. Mỗi lần fit có fit_id, lịch sử và thời gian riêng.
5. Thử ngưỡng trên validation với cùng trọng số; chọn Macro-F1 cao nhất theo quy tắc đã ghi.
6. Giữ nguyên mô hình, TF-IDF và ngưỡng. Đánh giá sáu trường hợp baseline/tuned trên cùng test.
7. Lưu models/model_info.json để liên kết đúng mô hình, TF-IDF, fit và ngưỡng; xuất báo cáo.

## Cài đặt và chạy

Từ thư mục gốc repository:

```powershell
python -m pip install -r requirements.txt
python -m Lab01.src.pipeline
python -m Lab01.src.pipeline --model svm
python -m Lab01.src.predict --model svm --text "Claim your free prize now"
```

Chạy --model chỉ tạo thực nghiệm riêng, không thay bộ mô hình mặc định. Muốn đọc một run cụ thể, thêm --run-id vào predict. Cấu trúc mới không fit lại trên train cộng validation. Trong schema 2, ma trận được đặt tên X_subtrain.npz, không nhầm với train ban đầu.

## Kết quả và demo

Lần chạy tham chiếu `20261119_015148_baseline_tuned`. Kết quả test:

| model | variant | accuracy | macro_f1 | ham_as_spam | spam_as_ham |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | baseline | 0.702149 | 0.412508 | 0 | 291 |
| svm | baseline | 0.702149 | 0.412508 | 0 | 291 |
| naive_bayes | baseline | 0.942682 | 0.927334 | 1 | 55 |
| logistic_regression | tuned | 0.991812 | 0.990231 | 5 | 3 |
| svm | tuned | 0.984647 | 0.981807 | 12 | 3 |
| naive_bayes | tuned | 0.982600 | 0.979302 | 11 | 6 |

Ngưỡng Logistic là xác suất, SVM là điểm quyết định có thể âm. So sánh ngưỡng trên cùng fit_id ở tuning/validation_summary.csv; không dò lại ngưỡng bằng test.

```powershell
python -m Lab01.src.predict --run-id 20261119_015148_baseline_tuned --model logistic_regression --text "Team meeting tomorrow"
python -m Lab01.src.eda --run-id 20261119_015148_baseline_tuned
```

## Tài liệu chi tiết

- [Báo cáo 12 bước](docs/Lab01_Report.md)
- [Tham số](docs/ReadmeForParameters.md)
- [Cách đọc cấu trúc results](docs/ResultsLayout.md)
- [Notebook](notebooks/Lab01_Spam_Classification.ipynb)

Mỗi lần chạy tạo thư mục mới theo giờ Việt Nam, không ghi đè run cũ. Các bản dùng chung được cập nhật sau khi chạy toàn bộ thành công. Notebook đọc run hoàn tất và kiểm chứng, không ghi lại vào run tham chiếu.
