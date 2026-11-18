# Lab01 — Giải thích tham số

Các giá trị cấu hình cuối trong tài liệu này thuộc lần chạy **`20261118_212649_baseline_tuned`** (13/11/2026, giờ Việt Nam). Nguồn: [selection.json](../results/20261118_212649_baseline_tuned/evaluation/comparison/selection.json).

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

Vectorizer chỉ học từ subtrain; validation và test dùng cùng bộ từ vựng/IDF. Sau khi chốt cấu hình và ngưỡng, giữ nguyên vectorizer và mô hình đã fit trên subtrain, không huấn luyện lại. Số cột cùng thứ tự đặc trưng phải khớp với mô hình ở thời điểm dự đoán. [Mã tiền xử lý](../src/preprocessing.py)

## 2. Logistic Regression

| Giá trị | Baseline | Cấu hình cuối |
|---|---:|---:|
| `learning_rate` | 0,1 | 5 |
| `n_iterations` | 1.000 | 1.000 |
| Ngưỡng xác suất | 0,5 | 0,4509449095570793 |

`learning_rate` nhân với gradient để xác định độ lớn bước cập nhật. Tốc độ quá nhỏ có thể cần nhiều vòng mới giảm loss đủ. Tốc độ quá lớn có thể khiến loss dao động. `n_iterations` là số vòng cập nhật toàn bộ tập đang fit.

Lab01 thử tốc độ 0,1, 1 và 5 kết hợp 1.000 hoặc 3.000 vòng. Hai cấu hình tốc độ 5 đạt cùng Macro-F1 validation 99,54% sau khi chọn ngưỡng riêng; code giữ cấu hình 1.000 vòng xuất hiện trước. `predict_proba()` đổi điểm tuyến tính thành xác suất lớp spam bằng sigmoid. `predict()` gán spam khi xác suất ít nhất bằng `threshold`. Với mỗi cấu hình, pipeline thử các ngưỡng trên xác suất validation và chọn theo Macro-F1; khi hòa điểm ngưỡng, ưu tiên ít FP hơn rồi gần 0,5 hơn. Cấu hình và ngưỡng tốt nhất được lưu trong `selection.json`; mô hình `.joblib` giữ ngưỡng này khi đánh giá test và dự đoán email mới. Ngưỡng 0,5 chỉ còn là mặc định cho baseline và mô hình cũ. [Mã Logistic Regression](../algorithms/logistic_regression.py)

## 3. Linear SVM

| Giá trị | Baseline | Cấu hình cuối |
|---|---:|---:|
| `learning_rate` | 0,01 | 0,01 |
| `n_iterations` | 1.000 | 3.000 |
| `C` | 1 | 100 |
| `class_weight` | Đều | Đều |
| `threshold` | 0 | Khoảng −0,662012 |

`C` điều chỉnh mức ưu tiên giảm Hinge Loss so với regularization L2. Tăng `C` làm lỗi margin có ảnh hưởng lớn hơn trong hàm mục tiêu. `class_weight` thay đổi mức đóng góp của mẫu từng lớp vào Hinge Loss trong lúc học. Lab01 đã thử trọng số đều, cân bằng theo tần suất lớp và tăng trọng số spam. Mô hình cuối giữ trọng số đều.

`threshold` được áp dụng sau khi tính điểm tuyến tính. Một email điểm −0,587 được gán spam ở ngưỡng −0,662 vì điểm cao hơn ngưỡng. Lab01 chọn ngưỡng theo Macro F1 trên validation. Mỗi tổ hợp learning rate, số vòng, C và trọng số lớp được tìm ngưỡng riêng trên validation rồi so sánh Macro-F1. Hệ thống lưu chính mô hình đã fit và ngưỡng tương ứng, không fit lại TF-IDF hay SVM sau lựa chọn. Với 7 cấu hình cơ sở và 5 cách đặt trọng số lớp, có 35 tổ hợp được thử; chọn chung cấu hình/trọng số/ngưỡng theo validation. Xem selection.json của từng lần chạy để biết cấu hình tương ứng. [Mã Linear SVM](../algorithms/svm.py)

## 4. Multinomial Naive Bayes

`alpha=1` là hệ số làm trơn của baseline; mô hình tuned dùng `alpha=0.01`, được chọn trên validation từ lưới `[1.0, 0.01, 0.1, 0.5, 2.0]`. Khi tính xác suất đặc trưng theo lớp, chương trình cộng alpha vào tổng trọng số của mỗi đặc trưng trước khi chuẩn hóa. Điều này tránh xác suất bằng 0 cho một đặc trưng chưa xuất hiện trong lớp. Mô hình tính điểm log cho ham và spam rồi chọn lớp có điểm lớn hơn. Thời gian fit chủ yếu dành cho việc cộng thống kê, vì thuật toán không dùng gradient descent. [Mã Naive Bayes](../algorithms/naive_bayes.py)

## 5. Đọc lịch sử thực nghiệm

[experiment_logs.csv](experiment_logs.csv) có một dòng cho mỗi mô hình tuned được ghi nhận của từng lần chạy. Các dòng cũ được giữ làm lịch sử; ba dòng của `20261118_212649_baseline_tuned` dùng cấu hình và thời gian từ `results/20261118_212649_baseline_tuned/evaluation/comparison/training.csv`.

`Run_ID` xác định lần chạy. `Started_UTC` giữ thời điểm bắt đầu quy đổi sang UTC để thống nhất với dữ liệu cũ; `run.json` của lần chạy mới lưu thời gian gốc UTC+7. `fit_seconds` đo duy nhất lời gọi fit của mô hình được chọn, không gồm tiền xử lý, dò ngưỡng, các cấu hình bị loại, dự đoán hay xuất báo cáo. Lần chạy hiện tại fit trên subtrain; lần lịch sử `20261106T172203Z-a2e0309c` dùng quy trình fit lại trên train ban đầu nên không nên so sánh thời gian như cùng điều kiện.

Ngưỡng LR và SVM lấy từ mô hình/cấu hình được chọn, không gán cố định 0,5 cho mô hình tuned. Ô trống là tham số không áp dụng; `class_weight=unweighted` nghĩa là trọng số đều. Naive Bayes dùng quy tắc chọn lớp có điểm log lớn nhất. CSV này là bản tổng hợp tài liệu, không tự cập nhật mỗi khi chạy pipeline.

Baseline học cùng subtrain và TF-IDF với tuned. Kết quả validation nằm ở `baseline_validation.csv`; test chỉ đánh giá ở cuối. `baseline.csv`/`metrics.csv` là chỉ số test hai nhóm, `before_after.csv` ghép hai nhóm, `baseline_training.csv`/`training.csv` là cấu hình và thời gian fit. Tổng thời gian toàn lần chạy ở `run.json`, khác với thời gian fit từng mô hình.

Mọi mô hình được giữ nguyên sau lựa chọn, không fit lại trên train + validation. Dự đoán bằng mô hình đã chốt:

```powershell
python -m Lab01.src.predict --run-id 20261118_212649_baseline_tuned --model svm --text "Claim your free prize now"
```


## 6. So sánh khi đổi tham số

- Đổi `max_features` hoặc `ngram_range` làm thay đổi tập cột đầu vào. Khi so sánh hai cấu hình, giữ cùng cách chia và fit lại cả vectorizer lẫn mô hình.
- Đổi `learning_rate`, `n_iterations`, `C` hoặc `alpha` ảnh hưởng đến quá trình học. Có thể đối chiếu Macro F1, recall spam và thời gian fit để thấy chi phí của từng cấu hình.
- Đổi ngưỡng trên cùng mô hình giữ nguyên điểm dự đoán liên tục và chỉ thay nhãn. Ví dụ, với xác suất spam 0,48, ngưỡng 0,5 cho ham, còn ngưỡng 0,4 cho spam. Khi hạ ngưỡng, số spam bị bỏ lọt có thể giảm nhưng số email thường bị chặn nhầm có thể tăng.
- Chọn tham số và ngưỡng bằng validation. Sau khi đã chốt cấu hình, đánh giá trên test và ghi riêng FP, FN. Kết quả hiện có dựa trên một lần chia, nên khi thay cấu hình cần ghi rõ tập dùng để đo.
