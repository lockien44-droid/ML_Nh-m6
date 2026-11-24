# Cách đọc kết quả Lab01

Schema 2 tách quá trình học, lựa chọn và đánh giá. Dữ liệu, thuật toán và quy tắc chọn cấu hình được giữ như trước.

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

## Ba mã định danh

run_id là một lần thực nghiệm; fit_id là một lần gọi fit với dữ liệu/cấu hình xác định; vectorizer_id định danh bộ TF-IDF học từ subtrain. ID TF-IDF được tạo từ dấu vết dữ liệu subtrain, cấu hình thực tế và IDF. Không trộn mô hình với một vectorizer chỉ vì chúng có cùng số cột.

Một lần fit có thể thử hàng trăm ngưỡng mà không fit lại. Những lần fit bị loại vẫn có dòng trong training_summary.csv và kết quả validation; chỉ baseline và mô hình thắng từng thuật toán được lưu joblib. Naive Bayes không có lịch sử loss theo vòng.

## Truy ngược một kết quả

1. Mở evaluation/comparison/test_summary.csv, lấy model, variant và fit_id.
2. Mở models/model_info.json, tìm đúng model và variant để lấy đường dẫn mô hình/TF-IDF.
3. Mở training/training_summary.csv, lọc theo fit_id để xem thông số thực tế và thời gian fit.
4. Nếu có thử ngưỡng, đọc tuning/threshold_search/<fit_id>/selection.json và thresholds.csv.
5. Kiểm tra metadata TF-IDF và danh sách email_id của các tập trong preprocessing.

## So sánh ngưỡng

Trong validation_summary.csv, so default_macro_f1 với selected_macro_f1 trên cùng fit_id. Điều này giữ nguyên trọng số để đo tác động của ngưỡng. Cột selected đánh dấu cấu hình thắng trong từng họ thuật toán. Baseline luôn có selected=false vì là mốc đối chiếu riêng.

thresholds.csv lưu các ngưỡng thực sự được xét. Khi YAML ép một ngưỡng hoặc một danh sách không chứa mặc định, selection.json vẫn ghi kết quả ngưỡng mặc định để đối chiếu; không bảo đảm ngưỡng ép tốt hơn mặc định. Trong chế độ automatic, ngưỡng mặc định được xét cùng các ngưỡng khác.

Macro-F1 dùng chọn cấu hình trên validation. Test không được dùng chọn lại mô hình/ngưỡng. SVM dùng decision_score, Logistic dùng spam_probability, Naive Bayes lưu chênh lệch log-joint, không phải xác suất.

## Thời gian và loss

fit_seconds đo fit có kèm tính chẩn đoán tại một số vòng, không gồm vẽ hình/lưu file. Thời lượng toàn run nằm trong run.json. Loss Logistic là binary cross-entropy; mục tiêu SVM gồm hinge có trọng số và L2, không so độ lớn hai hàm mục tiêu để xếp hạng thuật toán. Biểu đồ dùng mẫu khoảng 20 mốc cộng đầu/cuối, không phải log mọi vòng.

## Các bản dùng chung và lịch sử

models/ và Data/preprocessing/ ở gốc Lab01 được cập nhật khi chạy đủ ba mô hình thành công. Chạy --model chỉ tạo run riêng. predict.py nạp đúng cặp đối tượng qua model_info.json; vẫn đọc được cấu trúc phẳng của run cũ. Notebook hiện đọc và kiểm chứng run hoàn tất, không ghi đè nó. Các run lịch sử không bị chuyển đổi hoặc xóa.


## Training gọn

training_summary.csv có 49 dòng của lần chạy tham chiếu. parameters và requested_parameters là chuỗi JSON trong CSV; mỗi hàng vẫn giữ fit_id, vectorizer_id, thông số, trạng thái, thời gian và mục tiêu đầu/cuối. Đọc bằng pandas rồi json.loads nếu cần truy cập từng tham số.

Hai file history.csv giữ lịch sử tất cả các fit Logistic/SVM. Lọc cột fit_id để xem một lần học; mỗi cặp (fit_id, iteration) xác định một mốc. selected_training.png chỉ vẽ fit thắng validation của thuật toán đó. Naive Bayes chỉ có các dòng trong bảng tổng hợp, không cần thư mục riêng vì không có vòng cập nhật.

Việc gom file không đổi số lần học, trọng số, ngưỡng hoặc kết quả test. Các fit không được chọn vẫn giữ đầy đủ số liệu nhưng không tạo biểu đồ riêng.
