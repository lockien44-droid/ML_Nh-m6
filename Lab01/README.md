# Lab01 — Phân loại email spam

Lab01 so sánh ba mô hình tự cài đặt trên cùng đặc trưng TF-IDF dạng thưa gồm Logistic Regression, Linear SVM và Multinomial Naive Bayes. Nhãn `0` là ham và `1` là spam.

## Chạy nhanh

Mở PowerShell ở thư mục gốc repository. Lần chạy đối chiếu dùng Python 3.13.9. Dữ liệu nằm trong [spam_ham_dataset.csv](Data/raw/spam_ham_dataset.csv).

Nếu muốn tách thư viện khỏi Python đang dùng, tạo và kích hoạt môi trường ảo trước khi cài đặt.

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
python -m Lab01.src.train
```

Lệnh `train` chạy tiền xử lý, tạo TF-IDF, chọn cấu hình trên validation, huấn luyện, đánh giá và lưu cả ba mô hình. Để chỉ xem đầu ra tiền xử lý trước khi huấn luyện, có thể chạy `python -m Lab01.src.preprocessing`.

```powershell
python -m Lab01.src.predict --model logistic_regression --text "Please review the attached agenda for tomorrow team meeting"
python -m Lab01.src.predict --model logistic_regression --text "Congratulations! Claim your free cash prize now"
```

CLI trả `model`, `label` và `spam_probability` cho Logistic Regression. Xác suất từ 0,5 trở lên được gán `spam`. Có thể dùng `--model svm` để xem `decision_score` và `threshold`, hoặc `--model naive_bayes` để xem nhãn.

Để khám phá CSV gốc, chạy `python -m Lab01.src.eda`. Bảng thống kê và biểu đồ nằm trong [result/eda/](result/eda/).

## Đầu ra

| Nội dung | Vị trí |
|---|---|
| Các bước tiền xử lý và TF-IDF | [Data/preprocessing/](Data/preprocessing/) |
| Bảng tổng hợp ba mô hình | [metrics.csv](result/comparison/metrics.csv) |
| Dự đoán từng email | [Logistic Regression](result/logistic_regression/predictions.csv), [SVM](result/svm/predictions.csv), [Naive Bayes](result/naive_bayes/predictions.csv) |
| Ma trận nhầm lẫn | [Logistic Regression](result/logistic_regression/confusion_matrix.png), [SVM](result/svm/confusion_matrix.png), [Naive Bayes](result/naive_bayes/confusion_matrix.png) |
| Mô hình và vectorizer sinh sau khi train | [models/](models/), gồm `logistic_regression.joblib`, `svm.joblib`, `naive_bayes.joblib` và `tfidf_vectorizer.joblib` |
| Phương pháp, kết quả và phân tích lỗi | [Báo cáo Lab01](docs/report.md) |
