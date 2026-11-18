# Lab01 — Phân loại email spam

## Tổng quan

Lab01 dùng TF-IDF và ba thuật toán tự cài đặt để phân loại email thành ham (`0`) hoặc spam (`1`).

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
│   ├── tuning.py                     # Chia validation, chọn cấu hình và ngưỡng LR/SVM theo Macro-F1
│   ├── train.py                      # Huấn luyện dùng chung và ghi diễn biến huấn luyện
│   ├── evaluate.py                   # Tính chỉ số, lưu dự đoán và vẽ ma trận nhầm lẫn
│   ├── experiment_tracking.py                 # Tạo mã lần chạy, ghi thời gian, trạng thái và nhật ký
│   ├── eda.py                     # Khảo sát dữ liệu, vẽ hình và xuất báo cáo Markdown, HTML, PDF
│   ├── eda.py                        # Khảo sát nhãn, độ dài email và dữ liệu thiếu
│   ├── predict.py                    # Nạp mô hình và TF-IDF để dự đoán email mới
│   └── __init__.py                   # Khai báo package src
├── models/                           # Mô hình dùng cho lệnh predict
│   ├── logistic_regression.joblib     # Logistic Regression đã huấn luyện
│   ├── svm.joblib                     # Linear SVM đã huấn luyện, kèm ngưỡng quyết định
│   ├── naive_bayes.joblib             # Multinomial Naive Bayes đã huấn luyện
│   ├── tfidf_vectorizer.joblib        # Bộ từ vựng và IDF đã học
│   └── run_id.txt                     # Mã lần chạy đã tạo các mô hình đang lưu
├── results_runs/
│   └── <run_id>/                 # Dữ liệu, mô hình và kết quả của từng lần huấn luyện
├── reports/                          # Báo cáo tự động, hình ảnh và các bảng lịch sử
├── docs/
│   ├── Lab01_Report.md                # Báo cáo chi tiết theo 12 bước
│   ├── ReadmeForParameters.md         # Giải thích các tham số
│   ├── experiment_logs.csv           # Nhật ký thực nghiệm trong tài liệu
│   └── images/                       # Hình minh họa trong tài liệu
├── notebooks/                        # Khảo sát dữ liệu và thử nghiệm validation
└── README.md                         # Cấu trúc và cách chạy Lab01
```

### Đầu ra của một lần chạy

Các mục dưới đây được tạo khi chạy pipeline. `<run_id>` dùng giờ Việt Nam (UTC+7), ví dụ `20261118_212649_baseline_tuned`, và được in trong Terminal. Nếu trùng đến giây, hệ thống thêm `_02`, `_03`… để không ghi đè. Timestamp trong nhật ký và `run.json` có múi giờ `+07:00`. Các lần chạy cũ trong `results/runs/` được giữ nguyên.

```text
Lab01/results_runs/<run_id>/
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

1. Làm sạch email, kiểm tra nhãn và loại trùng, rồi chia train ban đầu/test theo tỷ lệ 80/20.
2. Tách validation từ train ban đầu theo `config.yaml` (mặc định 20% của train). Học TF-IDF duy nhất trên subtrain; validation và test chỉ được biến đổi bằng bộ này.
3. Huấn luyện baseline mặc định trên subtrain, đánh giá validation (`baseline_validation.csv`). Giữ lại chính các mô hình baseline đã fit.
4. Logistic Regression thử từng cấu hình learning rate/số vòng và chọn ngưỡng xác suất theo Macro-F1 validation. SVM thử toàn bộ tổ hợp learning rate/số vòng/`C`/trọng số lớp, tìm ngưỡng tốt nhất cho mỗi tổ hợp rồi chọn chung. Naive Bayes chọn `alpha` bằng validation. Tất cả mô hình học trên cùng subtrain và TF-IDF.
5. Giữ nguyên mô hình đã fit, TF-IDF và ngưỡng được chọn; **không fit lại trên train + validation**. Đánh giá cả baseline và mô hình tinh chỉnh trên test ở cuối. `baseline.csv`/`metrics.csv` lưu từng nhóm; `before_after.csv` ghép hai nhóm. `baseline_training.csv`/`training.csv` ghi thời gian fit trên subtrain của từng mô hình được giữ (không phải tổng thời gian tìm kiếm).
6. Lưu mô hình tinh chỉnh trong `models/`, baseline trong `models/baseline/`; demo dùng mô hình tinh chỉnh. Báo cáo ngưỡng đọc kết quả validation đã lưu, không huấn luyện lại.

`train_clean.csv` vẫn chứa train ban đầu (subtrain + validation). Trong `09_tfidf/`, `X_train.npz` và `y_train.csv` tương ứng với **subtrain_clean.csv**; `X_validation.npz`/`y_validation.csv` tương ứng validation. `metadata.json` ghi rõ vai trò này. Mọi ma trận dùng đúng bộ TF-IDF đã học trên subtrain.

`logistic_threshold_validation.csv` lưu mọi ngưỡng LR đã thử và đánh dấu cấu hình được chọn. `svm_study/validation.csv` lưu kết quả ngưỡng 0 và ngưỡng tốt nhất cho từng tổ hợp; mỗi tổ hợp có `candidate_id` riêng. `selection.json` lưu đầy đủ cấu hình và ngưỡng cuối, cùng `fit_split=subtrain`, `refit=false`.

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

Để demo đúng mô hình của một lần chạy đã chốt, thêm `--run-id`:

```powershell
python -m Lab01.src.predict --run-id 20261118_212649_baseline_tuned --model svm --text "Congratulations! Claim your free cash prize now"
```

Lệnh đọc mô hình và TF-IDF trực tiếp trong lần chạy, giữ nguyên ngưỡng đã chọn, không huấn luyện lại. Không có `--run-id`, lệnh vẫn dùng `Lab01/models/`. Kết quả và lịch sử dự đoán ghi mã lần chạy được sử dụng.

## Tài liệu chi tiết
Báo cáo tự động nằm trong `reports/`, hình theo lần chạy nằm trong `reports/figures/<run_id>/`. File `metrics_summary.csv` tổng hợp chỉ số test theo mã lần chạy, còn `inference_history.csv` ghi lịch sử dự đoán.

Để tạo lại báo cáo từ một lần chạy đã hoàn tất, thay `RUN_ID` bằng mã thư mục tương ứng:

```powershell
python -m Lab01.src.eda --run-id 20261118_212649_baseline_tuned
```

Xem thêm [báo cáo chi tiết theo 12 bước](docs/Lab01_Report.md) và [giải thích các tham số](docs/ReadmeForParameters.md).


EDA chạy ở đầu pipeline và lưu bảng thống kê cùng hình tại `results/<run_id>/figures/eda/`. Khi xuất báo cáo, chương trình dùng lại các đầu ra này và sao chép sang `reports/figures/<run_id>/eda/`.


## Cải tiến thực nghiệm

Chạy riêng một thuật toán bằng python -m Lab01.src.pipeline --model naive_bayes
(cũng nhận logistic_regression hoặc svm). Kết quả và mô hình của lần chạy riêng nằm
trong thư mục lần chạy. Dùng predict --run-id để dự đoán với phiên bản đó.

logistic_regression.threshold trong src/config.yaml nhận số từ 0 đến 1 để cố định
ngưỡng. Đặt null để chọn trên validation. threshold_candidates là danh sách ngưỡng
cần thử, để rỗng sẽ quét các điểm xác suất. Ngưỡng đã chọn được lưu cùng mô hình.

svm_study/all_thresholds.csv ghi mọi ngưỡng đã thử, Accuracy, Macro-F1, Precision,
Recall, F1, FP/FN và ma trận. Mỗi cấu hình có candidate_id riêng.
selected_validation/<model>/errors.csv và evaluation/<model>/errors.csv ghi email_id,
text_clean, nhãn thật, nhãn dự đoán, điểm, ngưỡng và loại lỗi FP/FN.
Nội dung là văn bản sau làm sạch, có thể truy nguyên bản gốc bằng email_id.

features trong YAML cấu hình TF-IDF của pipeline. analyzer có thể là word hoặc
char_wb, ngram_range khai báo khoảng độ dài từ/cụm ký tự.

## Trách nhiệm các module

train.py cung cấp train_model() để huấn luyện một mô hình và trace_fit() để ghi
diễn biến. tuning.py gọi train_model() cho baseline và các ứng viên.
experiment_tracking.py quản lý mã lần chạy, thời gian, trạng thái và nhật ký.
eda.py phụ trách hình tiền xử lý và báo cáo. pipeline.py điều phối toàn bộ.
Lệnh python -m Lab01.src.train tiếp tục gọi pipeline.


## Notebook khảo sát và thử nghiệm

- notebooks/Lab01_Spam_Classification.ipynb khảo sát dữ liệu, làm sạch và thử nghiệm đặc trưng, ngưỡng, tuning trên validation.

Mở notebook trong VS Code, chọn kernel Python của dự án và chạy các ô theo thứ tự.
Cài ipykernel nếu kernel yêu cầu. Phần thử nghiệm validation cần train_clean.csv từ bước tiền xử lý.
Các kết quả được lưu vào results/notebook_eda_<mã> hoặc results/notebook_validation_<mã>.


Các lần chạy được lưu cục bộ và bỏ qua trong Git. Repository giữ một lần chạy tham chiếu đầy đủ để đối chiếu báo cáo. Tham chiếu hiện tại là `20261118_212649_baseline_tuned`. Chạy pipeline sẽ tạo thư mục mới, không ghi đè thư mục tham chiếu.
