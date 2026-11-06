# Lab01 — Giải thích tham số

Tài liệu này tách ba nhóm giá trị dùng trong Lab01. **Tham số huấn luyện** ảnh hưởng đến cách mô hình học, **tham số tiền xử lý** xác định đặc trưng đầu vào, còn **ngưỡng quyết định** chuyển điểm thành nhãn. Trọng số mô hình và bias do thuật toán học từ dữ liệu.

## 1. Chia dữ liệu và TF-IDF

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `test_size` | 0,2 | Dành khoảng 20% dữ liệu sạch làm test. Khi chọn tham số, tách riêng 20% của train thành validation. |
| `random_state` | 42 cho train/test, 43 cho validation | Giữ cách chia có thể tái tạo trên cùng dữ liệu và thứ tự dòng. |
| `stratify` | Theo nhãn | Giữ tỷ lệ ham/spam gần nhau giữa các tập. |
| `ngram_range` | `(1, 2)` | Dùng từ đơn và cụm hai từ, ví dụ `free` và `free prize`. |
| `max_features` | 50.000 | Giới hạn số cột của từ vựng TF-IDF. Số cột thực tế phụ thuộc dữ liệu dùng để fit. |
| `norm` | L2 | Chuẩn hóa vector khác 0 về độ dài 1. |

Khi tuning, vectorizer học từ train con và chỉ biến đổi validation. Sau khi chốt cấu hình, vectorizer học lại từ toàn bộ train rồi biến đổi test. Số cột cùng thứ tự đặc trưng phải khớp với mô hình ở thời điểm dự đoán. [Mã tiền xử lý](../src/preprocessing.py)

## 2. Logistic Regression

| Giá trị | Baseline | Cấu hình cuối |
|---|---:|---:|
| `learning_rate` | 0,1 | 5 |
| `n_iterations` | 1.000 | 1.000 |
| Ngưỡng xác suất | 0,5 | 0,5 |

`learning_rate` nhân với gradient để xác định độ lớn bước cập nhật. Tốc độ quá nhỏ có thể cần nhiều vòng mới giảm loss đủ. Tốc độ quá lớn có thể khiến loss dao động. `n_iterations` là số vòng cập nhật toàn bộ tập đang fit.

Lab01 thử tốc độ 0,1, 1 và 5 kết hợp 1.000 hoặc 3.000 vòng. Tốc độ 5 cho cùng Macro F1 validation ở hai số vòng, nên cấu hình cuối dùng 1.000 vòng. `predict_proba()` đổi điểm tuyến tính thành xác suất lớp spam bằng sigmoid. `predict()` gán spam khi xác suất ít nhất 0,5. Bảng thử ngưỡng phục vụ phân tích validation, còn cấu hình triển khai giữ 0,5. [Mã Logistic Regression](../algorithms/logistic_regression.py)

## 3. Linear SVM

| Giá trị | Baseline | Cấu hình cuối |
|---|---:|---:|
| `learning_rate` | 0,01 | 0,01 |
| `n_iterations` | 1.000 | 3.000 |
| `C` | 1 | 100 |
| `class_weight` | Đều | Đều |
| `threshold` | 0 | Khoảng −0,662012 |

`C` điều chỉnh mức ưu tiên giảm Hinge Loss so với regularization L2. Tăng `C` làm lỗi margin có ảnh hưởng lớn hơn trong hàm mục tiêu. `class_weight` thay đổi mức đóng góp của mẫu từng lớp vào Hinge Loss trong lúc học. Lab01 đã thử trọng số đều, cân bằng theo tần suất lớp và tăng trọng số spam. Mô hình cuối giữ trọng số đều.

`threshold` được áp dụng sau khi tính điểm tuyến tính. Một email điểm −0,587 được gán spam ở ngưỡng −0,662 vì điểm cao hơn ngưỡng. Lab01 chọn ngưỡng theo Macro F1 trên validation. Sau khi fit lại TF-IDF và SVM, thang điểm có thể đổi, nên ngưỡng cần kiểm chứng trên dữ liệu mới. [Mã Linear SVM](../algorithms/svm.py)

## 4. Multinomial Naive Bayes

`alpha=1` là hệ số làm trơn trong cả baseline và cấu hình cuối. Khi tính xác suất đặc trưng theo lớp, chương trình cộng alpha vào tổng trọng số của mỗi đặc trưng trước khi chuẩn hóa. Điều này tránh xác suất bằng 0 cho một đặc trưng chưa xuất hiện trong lớp. Mô hình tính điểm log cho ham và spam rồi chọn lớp có điểm lớn hơn. Thời gian fit chủ yếu dành cho việc cộng thống kê, vì thuật toán không dùng gradient descent. [Mã Naive Bayes](../algorithms/naive_bayes.py)

## 5. Đọc lịch sử thực nghiệm

[experiment_logs.csv](experiment_logs.csv) có một dòng cho mỗi mô hình cuối trong từng lượt có dữ liệu thời gian. `Run_ID` xác định lượt chạy, `Started_UTC` là thời điểm bắt đầu và `fit_seconds` chỉ đo lời gọi `fit()` cuối, không gồm tiền xử lý, tuning, dự đoán hay tạo báo cáo. Ô trống biểu thị tham số không áp dụng cho mô hình. Ngưỡng LR trong CSV được đối chiếu với hằng số 0,5 trong `predict()`. CSV là bản tổng hợp tại thời điểm lập tài liệu. Nguồn của các lượt mới là `result/runs/<run_id>/evaluation/comparison/training.csv`.


## 6. So sánh khi đổi tham số

- Đổi `max_features` hoặc `ngram_range` làm thay đổi tập cột đầu vào. Khi so sánh hai cấu hình, giữ cùng cách chia và fit lại cả vectorizer lẫn mô hình.
- Đổi `learning_rate`, `n_iterations`, `C` hoặc `alpha` ảnh hưởng đến quá trình học. Có thể đối chiếu Macro F1, recall spam và thời gian fit để thấy chi phí của từng cấu hình.
- Đổi ngưỡng trên cùng mô hình giữ nguyên điểm dự đoán liên tục và chỉ thay nhãn. Ví dụ, với xác suất spam 0,48, ngưỡng 0,5 cho ham, còn ngưỡng 0,4 cho spam. Khi hạ ngưỡng, số spam bị bỏ lọt có thể giảm nhưng số email thường bị chặn nhầm có thể tăng.
- Chọn tham số và ngưỡng bằng validation. Sau khi đã chốt cấu hình, đánh giá trên test và ghi riêng FP, FN. Kết quả hiện có dựa trên một lần chia, nên khi thay cấu hình cần ghi rõ tập dùng để đo.
