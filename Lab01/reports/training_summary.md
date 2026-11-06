# Lab01 — Tóm tắt huấn luyện

[Mã huấn luyện](../src/train.py) so sánh Logistic Regression, Linear SVM và Multinomial Naive Bayes trên cùng dữ liệu. Xem [tóm tắt tiền xử lý](preprocessing_summary.md) để biết cách tạo TF-IDF và chia train/test.

## Quy trình huấn luyện

1. **Baseline.** Dùng TF-IDF fit trên toàn bộ 3.907 email train, huấn luyện ba mô hình với tham số mặc định và đánh giá trên 977 email test.
2. **Chọn cấu hình.** Tách train phân tầng thành 3.125 email train con và 782 email validation với seed 43. Fit TF-IDF mới trên train con và chỉ biến đổi validation. Chọn tham số Logistic Regression và SVM theo Macro F1, chỉ số cho hai lớp trọng số bằng nhau.
3. **Chọn ngưỡng SVM.** So sánh các phương án trọng số lớp và ngưỡng trên validation. Phương án được chọn dùng trọng số đều, đạt Macro F1 98,33% trên validation. Naive Bayes giữ cấu hình mặc định.
4. **Fit lại và đánh giá.** Fit TF-IDF và các mô hình cuối trên toàn bộ train, biến đổi test bằng vectorizer đã fit và tính chỉ số. Ngưỡng SVM đã chọn được áp dụng cho mô hình cuối. Mô hình và vectorizer được lưu trong `Lab01/models/`.

## Cấu hình cuối

| Mô hình | Tốc độ học | Số vòng | Tham số khác |
|---|---:|---:|---|
| Logistic Regression | 5 | 1.000 | Ngưỡng xác suất 0,5 |
| Linear SVM | 0,01 | 3.000 | C = 100, trọng số đều, ngưỡng điểm ≈ −0,662012 |
| Multinomial Naive Bayes | — | — | Alpha = 1 |

## Kết quả trên 977 email test

| Mô hình | Accuracy | Macro F1 | Recall spam | FN | FP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 99,18% | 99,02% | 98,97% | 3 | 5 |
| Linear SVM | 96,93% | 96,29% | 93,13% | 20 | 10 |
| Multinomial Naive Bayes | 95,70% | 94,66% | 86,60% | 39 | 3 |

FN là spam bị bỏ lọt, FP là ham bị chặn nhầm. Kết quả phản ánh một lần chia dữ liệu. Tập test đã được xem qua nhiều vòng. Sau khi fit lại, thang điểm SVM có thể thay đổi, nên ngưỡng cần được kiểm chứng trên dữ liệu mới. Stratified 5-Fold chưa được chạy.

## Chạy lại và xem đầu ra

Từ thư mục gốc repository, chạy một lần để tạo cả kết quả baseline và kết quả hiện tại.

```powershell
python -m Lab01.src.train
```

| Đầu ra | Nội dung |
|---|---|
| [baseline.csv](../result/comparison/baseline.csv) | Chỉ số của cấu hình mặc định |
| [validation.csv](../result/comparison/validation.csv) và [selection.json](../result/comparison/selection.json) | Các lượt thử tham số và cấu hình được chọn |
| [Kết quả thử ngưỡng](../result/comparison/svm_study/validation.csv) và [ngưỡng đã lưu](../result/comparison/svm_study/selection.json) | Kết quả theo trọng số lớp và ngưỡng trên validation |
| [metrics.csv](../result/comparison/metrics.csv) | Chỉ số test của cấu hình cuối |

Xem [báo cáo Lab01](../docs/report.md) để đọc ma trận nhầm lẫn và phân tích lỗi.
