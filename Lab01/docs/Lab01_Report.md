# Báo cáo Lab01 — Phân loại email spam bằng TF-IDF

Tài liệu đối chiếu với lần chạy **`20261113_024910_baseline_tuned`**, bắt đầu lúc **02:49:10 ngày 13/11/2026, giờ Việt Nam (UTC+7)**. Trạng thái: `success`. Các bảng và hình dưới đây cùng thuộc lần chạy này; số hiển thị được làm tròn, còn file kết quả giữ giá trị đầy đủ.

## 1. Xác định bài toán và mục tiêu

Lab01 phân loại email thành ham (`0`) hoặc spam (`1`) bằng Logistic Regression, Linear SVM và Multinomial Naive Bayes. Ba mô hình dùng chung cách chia dữ liệu và bộ TF-IDF. Macro-F1, trung bình F1 của hai lớp, là tiêu chí chọn cấu hình và ngưỡng trên validation. Phần thuyết trình tập trung vào Logistic Regression và SVM; Naive Bayes là mô hình đối chiếu.

Hai lỗi cần theo dõi là ham bị nhận nhầm thành spam (FP) và spam bị bỏ sót thành ham (FN). Vì số ham nhiều hơn spam, accuracy cần được đọc cùng Macro-F1, Precision và Recall spam.

## 2. Dữ liệu đầu vào và định danh email

Nguồn là [spam_ham_dataset.csv](../Data/raw/spam_ham_dataset.csv). `text` chứa nội dung, `label_num` chứa nhãn. Tiền xử lý tạo `email_id` từ chỉ số dòng gốc để đối chiếu lỗi; `text_clean` là văn bản đưa vào TF-IDF. Các CSV dự đoán giữ `email_id`, nhãn thật và nhãn dự đoán.

## 3. Khảo sát và kiểm tra dữ liệu

| Kiểm tra | Số email |
| --- | --- |
| Số dòng ban đầu | 5.171 |
| Dòng không hợp lệ bị loại | 0 |
| Dòng rỗng sau làm sạch bị loại | 0 |
| Bản trùng bị loại | 287 |
| Email còn lại | 4.884 |

Sau làm sạch còn **3.427 ham và 1.457 spam**. Loại trùng theo văn bản sạch trước khi chia tập để tránh cùng nội dung xuất hiện ở nhiều tập. Nếu cùng văn bản sạch có nhãn mâu thuẫn, chương trình báo lỗi. [Thống kê làm sạch của lần chạy](../results/20261113_024910_baseline_tuned/preprocessing/summary.json).

![Phân bố nhãn dữ liệu thô](../reports/figures/20261113_024910_baseline_tuned/eda/label_distribution.png)

Hình EDA mô tả dữ liệu thô, được xuất khi tạo báo cáo. Bảng trên mô tả dữ liệu sau bước làm sạch.

## 4. Chia subtrain, validation và test

Chia 80% train ban đầu và 20% test, phân tầng theo nhãn, seed `42`. Tách 20% của train ban đầu làm validation, seed `43`. Tỷ lệ toàn bộ xấp xỉ 64% subtrain / 16% validation / 20% test. Đây là một lần chia validation cố định, chưa phải cross-validation.

| Tập | Ham | Spam | Tổng | Vai trò |
| --- | --- | --- | --- | --- |
| Train ban đầu | 2.741 | 1.166 | 3.907 | Gồm subtrain + validation, không dùng để fit lại |
| Subtrain | 2.192 | 933 | 3.125 | Học TF-IDF và trọng số của mọi mô hình |
| Validation | 549 | 233 | 782 | Chọn cấu hình và ngưỡng |
| Test | 686 | 291 | 977 | Đánh giá cả baseline và tuned ở cuối |

**Không fit lại trên train + validation sau khi chọn cấu hình/ngưỡng.** Cả baseline và tuned giữ nguyên trọng số học trên subtrain. [Code chia và chọn mô hình](../src/tuning.py).

![Phân bố subtrain, validation và test](../reports/figures/20261113_024910_baseline_tuned/preprocessing/train_test_distribution.png)

## 5. Làm sạch email

`clean_text()` giải mã HTML entity, bỏ thẻ HTML và nội dung script/style, chuyển chữ thường, thay dấu câu bằng khoảng trắng và loại từ dừng tiếng Anh. Ví dụ `<b>FREE</b> Offer!!!` trở thành `free offer`. Các dấu `!`, `$` không trở thành đặc trưng trực tiếp. Cùng hàm được dùng khi dự đoán email mới. [Mã tiền xử lý](../src/preprocessing.py).

![Văn bản trước và sau làm sạch](../reports/figures/20261113_024910_baseline_tuned/preprocessing/cleaning_comparison.png)

## 6. Biểu diễn TF-IDF

Học bộ từ vựng và IDF **duy nhất trên subtrain**, dùng từ đơn/cụm hai từ (`ngram_range=(1,2)`), tối đa 50.000 cột, chuẩn hóa L2. Validation, test và email mới chỉ được `transform` bằng đúng bộ đã học. Không fit lại TF-IDF sau tuning.

| Tập | Kích thước | Phần tử khác 0 | Mật độ |
| --- | --- | --- | --- |
| subtrain | 3.125 × 50.000 | 312.835 | 0,200% |
| validation | 782 × 50.000 | 70.784 | 0,181% |
| test | 977 × 50.000 | 85.955 | 0,176% |

Ma trận thưa CSR giữ các giá trị khác 0 và chỉ số tương ứng. `09_tfidf/X_train.npz` và `y_train.csv` tương ứng **subtrain_clean.csv**, còn `train_clean.csv` chứa train ban đầu. `X_validation.npz`/`y_validation.csv` và `X_test.npz`/`y_test.csv` giữ hai tập còn lại. [Thống kê TF-IDF](../results/20261113_024910_baseline_tuned/preprocessing/split_tfidf_summary.json) và [Vai trò các ma trận](../results/20261113_024910_baseline_tuned/preprocessing/09_tfidf/metadata.json).

![Mật độ TF-IDF theo từng tập](../reports/figures/20261113_024910_baseline_tuned/preprocessing/tfidf_sparsity.png)

## 7. Baseline và luồng thực nghiệm

Baseline là cấu hình mặc định trong code: LR có learning rate `0.1`, 1.000 vòng, ngưỡng `0.5`; SVM có learning rate `0.01`, 1.000 vòng, `C=1`, trọng số đều, ngưỡng `0`; Naive Bayes có `alpha=1`.

Luồng chạy:

1. Làm sạch, chia subtrain/validation/test và học TF-IDF trên subtrain.
2. Huấn luyện baseline trên subtrain, đánh giá validation để có mốc ban đầu.
3. Thử cấu hình và chọn ngưỡng bằng validation.
4. Giữ nguyên TF-IDF cùng các mô hình đã fit; không huấn luyện lại.
5. Đánh giá cả baseline và tuned trên test ở cuối, rồi xuất bảng trước/sau.

[Baseline trên validation](../results/20261113_024910_baseline_tuned/evaluation/comparison/baseline_validation.csv) được tạo trước tuning. Bảng dưới là baseline trên **test**, được đánh giá sau khi đã chốt lựa chọn:

| Baseline | Accuracy test | Macro-F1 test | F1 spam | FN | FP |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 70,21% | 41,25% | 0,00% | 291 | 0 |
| Linear SVM | 70,21% | 41,25% | 0,00% | 291 | 0 |
| Multinomial Naive Bayes | 94,27% | 92,73% | 89,39% | 55 | 1 |

LR và SVM mặc định dự đoán toàn bộ test là ham. Điều này phản ánh cấu hình mặc định và quá trình tối ưu trong lần chạy, không chứng minh hai thuật toán không phù hợp. [Baseline test](../results/20261113_024910_baseline_tuned/evaluation/comparison/baseline.csv).

## 8. Huấn luyện ba thuật toán

### 8.1 Logistic Regression

Với vector TF-IDF của một email, mô hình tính điểm $z = w^T x + b$ rồi dùng sigmoid để tính xác suất spam $p = 1/(1 + e^{-z})$. Trong mỗi vòng huấn luyện, gradient descent cập nhật trọng số và bias dựa trên sai lệch giữa xác suất dự đoán và nhãn thật. Ngưỡng phân loại được áp dụng sau khi tính xác suất. [Mã Logistic Regression](../algorithms/logistic_regression.py)

Trọng số và bias bắt đầu từ 0. Mỗi vòng dùng toàn bộ dữ liệu huấn luyện để tính xác suất $p$ và sai số $p-y$. Với $n$ email, gradient của trọng số là $X^T(p-y)/n$, còn gradient của bias là trung bình của $p-y$. Chương trình nhân các gradient với learning rate rồi trừ vào tham số hiện tại. Đây là cập nhật theo gradient của binary cross-entropy. Learning rate quyết định độ lớn của mỗi bước cập nhật, còn số vòng lặp quyết định số lần cập nhật.

### 8.2 Linear SVM

SVM học một mặt phân cách tuyến tính qua điểm $s=w^Tx+b$. Trong huấn luyện, nhãn được ánh xạ thành $y\in\{-1,+1\}$. Mô hình tối ưu tổng của phần phạt L2 trên trọng số và hinge loss nhân với $C$. Hinge loss $\max(0,1-ys)$ phạt mẫu sai phía hoặc nằm trong vùng biên. Chỉ mẫu có $ys<1$ đóng góp vào gradient của hinge loss. Nếu bật trọng số lớp, lỗi ở lớp có trọng số cao sẽ đóng góp nhiều hơn.

Sau huấn luyện, SVM dự đoán spam khi điểm $s$ đạt ngưỡng. Điểm này có thể âm và được dùng trực tiếp. [Mã Linear SVM](../algorithms/svm.py)

### 8.3 Multinomial Naive Bayes

Mô hình ước lượng xác suất tiên nghiệm từ tỷ lệ ham và spam trong train. Với mỗi lớp, nó cộng trọng số TF-IDF theo từng cột, cộng thêm $\alpha$ để làm trơn rồi chuẩn hóa. Cộng $\alpha$ giúp đặc trưng vắng mặt ở một lớp vẫn có xác suất dương.

Với lớp $c$, mô hình tính điểm $\log P(c)+\sum_j x_j\log P(j\mid c)$. Giả định Naive Bayes xem các đặc trưng là độc lập có điều kiện theo lớp. Chương trình dùng điểm log để cộng đóng góp của các cột rồi chọn lớp có điểm cao nhất. Việc học dựa trên thống kê theo lớp và không cần gradient descent. [Mã Multinomial Naive Bayes](../algorithms/naive_bayes.py)

## 9. Chọn cấu hình và ngưỡng bằng validation

### 9.1 Logistic Regression

Với mỗi trong sáu cặp learning rate/số vòng, huấn luyện trên subtrain, tính xác suất validation, rồi tìm ngưỡng có Macro-F1 tốt nhất. Khi hòa điểm giữa các ngưỡng, ưu tiên ít FP hơn, rồi ngưỡng gần 0,5 hơn. Sau đó so sánh sáu cấu hình theo Macro-F1; khi hòa, giữ cấu hình xuất hiện trước.

| Learning rate | Số vòng | Ngưỡng tốt nhất | Macro-F1 validation | Recall spam |
| --- | --- | --- | --- | --- |
| 0,1 | 1.000 | 0,338152 | 96,25% | 97,85% |
| 0,1 | 3.000 | 0,381724 | 96,69% | 98,28% |
| 1,0 | 1.000 | 0,427161 | 98,03% | 99,14% |
| 1,0 | 3.000 | 0,470298 | 99,24% | 99,57% |
| 5,0 | 1.000 | 0,450945 | 99,54% | 100,00% |
| 5,0 | 3.000 | 0,351801 | 99,54% | 100,00% |

Chọn **learning rate 5, 1.000 vòng, ngưỡng `0.4509449095570793`**. Hai cấu hình learning rate 5 cùng Macro-F1 tốt nhất nên giữ cấu hình 1.000 vòng xuất hiện trước. [Bảng validation](../results/20261113_024910_baseline_tuned/evaluation/comparison/validation.csv).

### 9.2 Linear SVM

Thử **7 cấu hình learning rate/số vòng/C × 5 cách đặt trọng số lớp = 35 tổ hợp**. Trọng số gồm đều, `balanced`, spam ×2, ×3 và ×4. Với từng tổ hợp, huấn luyện trên subtrain, tính điểm validation và tìm ngưỡng tốt nhất; sau đó so sánh toàn bộ tổ hợp. Không loại cấu hình chỉ dựa vào ngưỡng mặc định 0.

Năm tổ hợp có Macro-F1 cao nhất sau chọn ngưỡng (mã tổ hợp thể hiện cách đặt trọng số):

| candidate_id | Learning rate | Số vòng | C | Ngưỡng | Macro-F1 validation |
| --- | --- | --- | --- | --- | --- |
| candidate_0_unweighted | 0,01 | 1.000 | 1 | -0,997026 | 98,33% |
| candidate_4_unweighted | 0,01 | 1.000 | 50 | -0,858412 | 98,33% |
| candidate_6_unweighted | 0,01 | 3.000 | 100 | -0,662012 | 98,33% |
| candidate_6_spam_x3 | 0,01 | 3.000 | 100 | 0,700963 | 97,44% |
| candidate_0_spam_x3 | 0,01 | 1.000 | 1 | 0,994033 | 97,13% |

Khi hòa Macro-F1 giữa các tổ hợp, ưu tiên trọng số đều, ít FP hơn, rồi ngưỡng gần 0 hơn. Cấu hình được chọn là **learning rate 0,01; 3.000 vòng; C=100; trọng số đều; ngưỡng `-0.6620115722742366`**. [Toàn bộ tổ hợp SVM](../results/20261113_024910_baseline_tuned/evaluation/comparison/svm_study/validation.csv) và [Lựa chọn SVM](../results/20261113_024910_baseline_tuned/evaluation/comparison/svm_study/selection.json).

### 9.3 Multinomial Naive Bayes

Thử hệ số làm trơn trên cùng subtrain và validation; chọn `alpha=0.01` theo Macro-F1. Quy tắc dự đoán vẫn là chọn lớp có điểm log cao nhất, không có bước dò ngưỡng riêng.

| Alpha | Macro-F1 validation | Recall spam |
| --- | --- | --- |
| 1,0 | 93,38% | 82,40% |
| 0,01 | 98,48% | 98,28% |
| 0,1 | 98,47% | 97,85% |
| 0,5 | 96,70% | 90,99% |
| 2,0 | 87,21% | 67,81% |

![Cấu hình các mô hình được giữ lại](../reports/figures/20261113_024910_baseline_tuned/training/training_configuration.png)

## 10. Quy tắc dự đoán và tác động của ngưỡng

### 10.1 Logistic Regression

`predict()` gán spam khi xác suất **lớn hơn hoặc bằng ngưỡng đã chọn**. Ngưỡng 0,5 là mặc định cho baseline; tuned dùng `0.4509449095570793`. Trên cùng mô hình LR được chọn, kết quả validation đã lưu là:

| Ngưỡng | Precision spam | Recall spam | Macro-F1 validation | FN | FP |
| --- | --- | --- | --- | --- | --- |
| 0,500000000 | 99,13% | 98,28% | 99,08% | 4 | 2 |
| 0,450944910 | 98,73% | 100,00% | 99,54% | 0 | 3 |

[Bảng quét ngưỡng LR](../results/20261113_024910_baseline_tuned/evaluation/comparison/logistic_threshold_validation.csv) lưu toàn bộ ngưỡng đã thử, có cột đánh dấu cấu hình được chọn. `report.py` chỉ đọc bảng này để tạo báo cáo, không fit lại mô hình.

### 10.2 Linear SVM

SVM dự đoán spam khi `decision_score >= threshold`. Điểm SVM không phải xác suất. Ví dụ điểm −0,587 vẫn được gán spam vì lớn hơn ngưỡng −0,662012. Cấu hình được chọn có ma trận validation `[[541,8],[3,230]]`: FP=8, FN=3, Macro-F1=98,33%. Giữ nguyên mô hình đã tạo ra các điểm này để ngưỡng khớp với thang điểm khi đánh giá test.

### 10.3 Đối chiếu lỗi

Các tệp `selected_validation/<model>/predictions.csv` và `evaluation/<model>/predictions.csv` giữ lỗi validation và test riêng. Dùng `email_id` để đối chiếu với email gốc; không suy đoán nguyên nhân lỗi chỉ từ số đếm. FP là ham bị chặn nhầm; FN là spam bị bỏ sót.

## 11. Đánh giá test và so sánh

Sau khi chốt cấu hình/ngưỡng trên validation, dùng nguyên mô hình và TF-IDF đã học trên subtrain để đánh giá **977 email test (686 ham, 291 spam)**. Không huấn luyện thêm trên validation. Baseline và tuned dùng chung subtrain, TF-IDF và test.

| Mô hình tuned | Accuracy | Macro-F1 | Precision spam | Recall spam | F1 spam | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 99,18% | 99,02% | 98,29% | 98,97% | 98,63% | 3 | 5 |
| Linear SVM | 98,46% | 98,18% | 96,00% | 98,97% | 97,46% | 3 | 12 |
| Multinomial Naive Bayes | 98,26% | 97,93% | 96,28% | 97,94% | 97,10% | 6 | 11 |

[Kết quả tuned test](../results/20261113_024910_baseline_tuned/evaluation/comparison/metrics.csv) và [So sánh trước/sau tinh chỉnh](../results/20261113_024910_baseline_tuned/evaluation/comparison/before_after.csv).

LR dự đoán đúng 681 ham và 288 spam, sai tổng cộng 8 email. SVM cùng bỏ sót 3 spam nhưng chặn nhầm 12 ham, nhiều hơn LR 7 email. Naive Bayes bỏ sót 6 spam và chặn nhầm 11 ham. Vì vậy LR đạt Macro-F1 cao nhất trong lần chia này. Đây là kết quả thực nghiệm, chưa chứng minh LR luôn tốt hơn các thuật toán khác.

![Ma trận nhầm lẫn LR trên test, ngưỡng 0,450945](../reports/figures/20261113_024910_baseline_tuned/evaluation/confusion_matrix_lr.png)

![Ma trận nhầm lẫn SVM trên test, ngưỡng −0,662012](../reports/figures/20261113_024910_baseline_tuned/evaluation/confusion_matrix_svm.png)

![Ma trận nhầm lẫn Naive Bayes trên test, alpha 0,01](../reports/figures/20261113_024910_baseline_tuned/evaluation/confusion_matrix_nb.png)

| Mô hình | Thời gian fit mô hình được chọn (giây) | Macro-F1 baseline test | Macro-F1 tuned test |
| --- | --- | --- | --- |
| Logistic Regression | 0,552280 | 41,25% | 99,02% |
| Linear SVM | 1,879152 | 41,25% | 98,18% |
| Multinomial Naive Bayes | 0,002207 | 92,73% | 97,93% |

Thời gian fit trên subtrain lấy từ [training.csv](../results/20261113_024910_baseline_tuned/evaluation/comparison/training.csv), chỉ đo lời gọi fit của mô hình được chọn, không gồm TF-IDF, các cấu hình bị loại, dò ngưỡng, dự đoán hay xuất báo cáo. Đây không phải tổng thời gian tuning và phụ thuộc môi trường chạy. So sánh baseline/tuned đo tác động tổng thể của cả thay đổi cấu hình lẫn ngưỡng, chưa tách riêng từng yếu tố.

![So sánh các chỉ số test](../reports/figures/20261113_024910_baseline_tuned/evaluation/metrics_comparison.png)

![Đường ROC trên test](../reports/figures/20261113_024910_baseline_tuned/evaluation/roc_curve_all.png)

ROC quét qua các ngưỡng điểm để mô tả khả năng xếp hạng; bảng test dùng ngưỡng đã chốt trên validation. Kết quả chỉ dựa trên một lần chia, một nguồn email và phạm vi cấu hình hữu hạn. Cần các lần chia hoặc dữ liệu độc lập để đánh giá độ ổn định; không điều chỉnh tham số chỉ để nâng kết quả của test đã xem.

## 12. Lưu kết quả và demo email mới

Bản đầy đủ của lần thực nghiệm nằm trong `Lab01/results/20261113_024910_baseline_tuned/`: `run.json`, `events.jsonl`, `preprocessing/`, `models/`, `evaluation/`, `figures/`. Mô hình tuned và TF-IDF nằm trong `models/`; baseline nằm trong `models/baseline/`. `selection.json` ghi `fit_split=subtrain`, `refit=false`. Nhật ký thời gian của lần chạy dùng UTC+7.

Báo cáo tự động và hình tổng hợp nằm trong `Lab01/reports/`, với hình tại `reports/figures/20261113_024910_baseline_tuned/`. [Báo cáo PDF](../reports/report_20261113_024910_baseline_tuned.pdf). `Data/preprocessing/` và `Lab01/models/` là các bản được cập nhật khi pipeline chạy thành công; để tái hiện đúng thực nghiệm, dùng thư mục riêng của lần chạy.

Luồng dự đoán: **email mới → clean_text → TF-IDF đã lưu → xác suất/điểm → so với ngưỡng đã lưu → ham/spam**. Chạy từ thư mục gốc:

```powershell
python -m Lab01.src.predict --run-id 20261113_024910_baseline_tuned --model logistic_regression --text "Congratulations! Claim your free cash prize now"
python -m Lab01.src.predict --run-id 20261113_024910_baseline_tuned --model svm --text "Congratulations! Claim your free cash prize now"
python -m Lab01.src.predict --run-id 20261113_024910_baseline_tuned --model naive_bayes --text "Congratulations! Claim your free cash prize now"
```

Lệnh dùng mô hình của đúng lần chạy, không huấn luyện lại. Không truyền `--run-id` thì dùng `Lab01/models/`. Lịch sử dự đoán ghi trong `reports/inference_history.csv` kèm `Run_ID`. Demo vài email chỉ minh họa hành vi, không thay thế đánh giá định lượng trên tập test.

```powershell
python -m Lab01.src.report --run-id 20261113_024910_baseline_tuned
```

Lệnh trên tạo lại báo cáo từ dữ liệu và kết quả validation đã lưu. [Code pipeline](../src/pipeline.py) · [Code tuning](../src/tuning.py) · [Code dự đoán](../src/predict.py).
