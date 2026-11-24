# Tham số Lab01

Lần chạy tham chiếu `20261119_015148_baseline_tuned`. [Cấu hình được chọn](../results/20261119_015148_baseline_tuned/tuning/selection.json).

| Nhóm | Tham số | Vai trò |
| --- | --- | --- |
| Chia tập | split.test_size, split.random_state | Tỷ lệ test và seed; chia phân tầng theo nhãn |
| Validation | validation.fraction, validation.seed | Tách validation từ train ban đầu |
| TF-IDF | features.analyzer, ngram_range, max_features | Kiểu đặc trưng, n-gram, tối đa số cột |
| Logistic | learning_rate, n_iterations | Bước và số lần cập nhật gradient |
| Logistic | threshold, threshold_candidates | Ép ngưỡng, danh sách thử hoặc tự quét khi null/rỗng |
| SVM | learning_rate, n_iterations, C | Tối ưu và mức phạt hinge loss |
| SVM | class_weight | Thử đều, balanced và spam ×2/×3/×4 trong tuning.py |
| Naive Bayes | alpha | Làm trơn thống kê TF-IDF theo lớp |

Learning rate quá nhỏ có thể học chậm, quá lớn có thể dao động. Số vòng lớn hơn không bảo đảm validation tốt hơn. Logistic/SVM dừng theo số vòng, chưa có dừng sớm. Ngưỡng Logistic nằm trong [0,1]; SVM có thể âm vì không phải xác suất. Naive Bayes chọn lớp có điểm lớn nhất, không quét ngưỡng trong bài này.

TF-IDF và thuật toán chỉ fit trên subtrain. Validation dùng chọn Macro-F1. Test chỉ đánh giá cuối. Mọi cấu hình được so trên cùng cách chia; không refit sau chọn ngưỡng.

Cấu hình được chọn thực tế:

| model | fit_id | parameters | selected_threshold |
| --- | --- | --- | --- |
| logistic_regression | fit_0008 | {"learning_rate": 5.0, "n_iterations": 1000, "threshold": 0.5} | 0.450945 |
| svm | fit_0040 | {"C": 100, "class_weight": null, "learning_rate": 0.01, "n_iterations": 3000, "threshold": 0.0} | -0.662012 |
| naive_bayes | fit_0046 | {"alpha": 0.01} | nan |

## Đọc kết quả

- training/training_summary.csv lưu tham số thực tế, requested_parameters và thời gian fit.
- tuning/validation_summary.csv so ngưỡng mặc định và đã chọn trên cùng fit_id.
- tuning/threshold_search/<fit_id>/thresholds.csv lưu các ngưỡng thực sự được xét.
- models/model_info.json xác định mô hình dùng bộ TF-IDF nào.
- evaluation/comparison/test_summary.csv chứa cả sáu trường hợp test.

Xem [cấu trúc đầy đủ](ResultsLayout.md). Thời gian fit có kèm chẩn đoán lấy mẫu; Naive Bayes không có loss theo vòng. experiment_logs.csv cũ là lịch sử tài liệu, không phải bảng tự động của cấu trúc mới; dùng training_summary.csv và run.json để xem thời gian hiện tại.
