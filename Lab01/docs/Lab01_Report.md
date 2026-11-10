# Báo cáo Lab01 — Phân loại email spam bằng TF-IDF với Logistic Regression, Linear SVM và Naive Bayes

## 1. Xác định bài toán và mục tiêu

Lab01 phân loại email dựa trên nội dung. Ba mô hình dùng chung vector TF-IDF và trả về nhãn ham (`0`) hoặc spam (`1`). Logistic Regression dùng xác suất spam, Linear SVM dùng điểm quyết định và Naive Bayes so sánh điểm của hai lớp. Hai lỗi cần theo dõi là spam bị bỏ lọt (FN) và ham bị đánh dấu nhầm (FP). Lab01 dùng Macro-F1 để chọn cấu hình, tức trung bình F1 của hai lớp.

Báo cáo trình bày quy trình của lần chạy `Thứ 2`. Ba thuật toán được so sánh trên cùng cách chia dữ liệu, cùng đầu vào TF-IDF và cùng bộ chỉ số. Bài toán có hai yêu cầu cần cân bằng. Recall spam cao giúp giảm số thư rác lọt vào hộp thư đến. Precision spam cao giúp các thư bị gắn cờ có ít email hợp lệ hơn.

## 2. Thu thập và gán nhãn dữ liệu

Dữ liệu đầu vào là CSV đã gán nhãn `spam_ham_dataset.csv`. Cột `text` chứa nội dung email, `label_num` chứa nhãn thật. Quy trình tiền xử lý tạo `email_id` từ chỉ số dòng gốc để truy lại email khi phân tích lỗi.

| Trường | Nguồn | Vai trò |
| --- | --- | --- |
| `email_id` | Chỉ số dòng trong CSV gốc | Đối chiếu dự đoán với email gốc |
| `text` | CSV gốc | Nội dung đưa vào tiền xử lý |
| `label_num` | CSV gốc | Nhãn dùng để huấn luyện và đánh giá |
| `text_clean` | Kết quả của `clean_text` | Văn bản đầu vào của TF-IDF |

## 3. Khảo sát và kiểm tra dữ liệu

Khảo sát phân bố ham/spam, độ dài email và dữ liệu thiếu để hiểu chất lượng đầu vào. Trước khi chia tập, mã loại dòng có nhãn không hợp lệ, nội dung rỗng và bản trùng theo văn bản sạch. Nếu cùng văn bản sạch có hai nhãn khác nhau, chương trình báo lỗi. Việc loại bản trùng trước khi chia giúp tránh đưa cùng một nội dung vào cả train và test. [Mã tiền xử lý](../src/preprocessing.py)

| Kiểm tra | Số email |
| --- | ---: |
| Số dòng ban đầu | 5.171 |
| Dòng không hợp lệ bị loại | 0 |
| Dòng rỗng sau làm sạch bị loại | 0 |
| Bản trùng theo văn bản sạch bị loại | 287 |
| Số email còn lại | 4.884 |

Sau khi làm sạch có 3.427 ham và 1.457 spam, tương ứng khoảng 70,2% và 29,8%. Chênh lệch này giải thích vì sao accuracy cần được đọc cùng recall spam và Macro-F1. [Thống kê tiền xử lý](../Data/preprocessing/summary.json)

![Phân bố nhãn ham và spam](../reports/figures/20261106T172203Z-a2e0309c/eda/label_distribution.png)

*Hình 1. Phân bố nhãn trong khảo sát dữ liệu.*

Biểu đồ EDA được tạo ở giai đoạn xuất báo cáo sau huấn luyện và mô tả dữ liệu thô. Bảng số lượng sau làm sạch ở trên đến từ bước tiền xử lý trước khi chia tập. [Mã xuất EDA](../src/report.py#L259)

## 4. Phân chia train, validation và test

Lab01 chia dữ liệu sạch theo tỷ lệ **80% train và 20% test**, phân tầng theo nhãn với seed `42`. Khi chọn siêu tham số, chương trình tách tiếp **20% của tập train** làm validation với seed `43`, cũng phân tầng theo nhãn. Tỷ lệ trên toàn bộ dữ liệu xấp xỉ 64% huấn luyện thử, 16% validation và 20% test.

Đây là cách dùng **một tập validation cố định**. Với 5-fold, phần dữ liệu phát triển sẽ được chia thành năm phần và luân phiên dùng từng phần để lấy dự đoán validation. Lab01 hiện dùng cách chia cố định trong bước chọn cấu hình. [Mã chia validation](../src/train.py#L104)

| Tập dữ liệu | Ham | Spam | Tổng | Mục đích |
| --- | ---: | ---: | ---: | --- |
| Train ban đầu | 2.741 | 1.166 | 3.907 | Chia để thử cấu hình, sau đó dùng toàn bộ để fit lại |
| Phần huấn luyện thử | 2.192 | 933 | 3.125 | Fit TF-IDF và từng cấu hình |
| Validation | 549 | 233 | 782 | So sánh cấu hình và khảo sát ngưỡng |
| Test | 686 | 291 | 977 | Đo kết quả của mô hình cuối |

Hai dòng huấn luyện thử và validation là hai phần của train ban đầu. Tỷ lệ spam xấp xỉ 30% trong cả ba tập sử dụng khi chọn và đánh giá mô hình.

![Phân bố train và test](../reports/figures/20261106T172203Z-a2e0309c/preprocessing/train_test_distribution.png)

*Hình 2. Phân bố nhãn sau khi chia train và test.*

## 5. Làm sạch và tiền xử lý email

Hàm `clean_text` giải mã HTML entity, bỏ thẻ HTML và nội dung script/style, chuyển chữ thường, thay dấu câu bằng khoảng trắng và loại từ dừng tiếng Anh. Cùng hàm này được áp dụng cho email mới khi dự đoán. [Hàm làm sạch](../src/preprocessing.py#L16)

Ví dụ, đầu vào `<b>FREE</b> Offer!!!` cho văn bản sạch `free offer`. Bước này giữ từ mang nội dung và giảm biến thể do kiểu chữ hoặc dấu câu. Các dấu như `!` và `$` bị loại, nên chúng không tạo thành đặc trưng trực tiếp trong TF-IDF hiện tại.

![So sánh văn bản trước và sau làm sạch](../reports/figures/20261106T172203Z-a2e0309c/preprocessing/cleaning_comparison.png)

*Hình 3. Minh họa tác động của tiền xử lý lên nội dung email.*

## 6. Biểu diễn văn bản bằng TF-IDF

Mỗi hàng trong ma trận TF-IDF biểu diễn một email. Mỗi cột tương ứng với một từ đơn hoặc cụm hai từ trong bộ từ vựng, với tối đa 50.000 đặc trưng. Trong lúc chọn cấu hình, `fit_transform` học bộ từ vựng và IDF từ phần huấn luyện thử. `transform` dùng lại bộ từ vựng và IDF đó cho validation. [Cấu hình TF-IDF](../src/preprocessing.py#L83)

TF-IDF kết hợp tần suất xuất hiện trong từng email với độ hiếm của từ hoặc cụm từ trong tập train. Vector của mỗi email được chuẩn hóa L2. Từ ngoài bộ từ vựng đã học không sinh thêm cột khi chuyển đổi validation, test hoặc email mới.

| Ma trận khi đánh giá cuối | Kích thước | Số phần tử khác 0 | Mật độ |
| --- | ---: | ---: | ---: |
| Train | 3.907 × 50.000 | 387.045 | 0,198% |
| Test | 977 × 50.000 | 86.379 | 0,177% |

Hầu hết email chỉ chứa một phần nhỏ bộ từ vựng. Ma trận thưa CSR lưu các giá trị khác 0 và chỉ số tương ứng, phù hợp với phép nhân ma trận trong quá trình huấn luyện. [Thống kê TF-IDF](../Data/preprocessing/split_tfidf_summary.json)

![Mật độ ma trận TF-IDF](../reports/figures/20261106T172203Z-a2e0309c/preprocessing/tfidf_sparsity.png)

*Hình 4. Mật độ giá trị khác 0 của ma trận train và test.*

## 7. Lập mô hình mốc

Cấu hình Logistic Regression mặc định là mốc để đối chiếu với cấu hình đã qua lựa chọn. Một mốc minh họa khác là luôn đoán ham. Trên tập giả định có 85% ham, quy tắc này đạt accuracy 85% nhưng recall spam bằng 0%. Ví dụ cho thấy vì sao cần đọc thêm các chỉ số riêng cho lớp spam.

Trong lần chạy đã lưu, baseline Logistic Regression với `learning_rate=0.1` và `n_iterations=1000` dự đoán cả 977 email test là ham. Accuracy đạt 70,21% do có 686 ham, nhưng recall spam bằng 0% và Macro-F1 chỉ đạt 41,25%. Kết quả này là mốc mô tả trên cùng test. Việc chọn cấu hình dựa vào bảng validation ở bước 9. [Kết quả baseline](../results/runs/20261106T172203Z-a2e0309c/evaluation/comparison/baseline.csv)

Baseline của ba mô hình trên cùng 977 email test:

| Mô hình mặc định | Accuracy | Macro-F1 | F1 spam | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 70,21% | 41,25% | 0,00% | 291 | 0 |
| Linear SVM | 70,21% | 41,25% | 0,00% | 291 | 0 |
| Multinomial Naive Bayes | 95,70% | 94,66% | 92,31% | 39 | 3 |

Linear SVM mặc định cũng dự đoán toàn bộ test là ham. Naive Bayes dùng cùng cấu hình mặc định cho kết quả cuối, nên hai hàng kết quả của mô hình này trùng nhau giữa bảng baseline và bảng test cuối. [Bảng baseline gốc](../results/runs/20261106T172203Z-a2e0309c/evaluation/comparison/baseline.csv)

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

## 9. So sánh cấu hình và chọn siêu tham số

### 9.1 Logistic Regression

Lab01 thử sáu cặp từ ba giá trị `learning_rate` (0,1, 1 và 5) và hai giá trị `n_iterations` (1.000 và 3.000). Mỗi cặp được huấn luyện trên cùng phần huấn luyện thử, sau đó dự đoán trên cùng tập validation với ngưỡng 0,5. Cặp có Macro-F1 cao nhất được chọn và ghi trong `selection.json`. [Mã chọn cấu hình](../src/train.py#L112)

| Learning rate | Số vòng | Macro-F1 | Recall spam | F1 spam |
| ---: | ---: | ---: | ---: | ---: |
| 0,1 | 1.000 | 41,25% | 0,00% | 0,00% |
| 0,1 | 3.000 | 64,16% | 26,61% | 41,89% |
| 1 | 1.000 | 94,80% | 87,98% | 92,55% |
| 1 | 3.000 | 98,77% | 97,42% | 98,27% |
| **5** | **1.000** | **99,08%** | **98,28%** | **98,71%** |
| 5 | 3.000 | 99,08% | 98,28% | 98,71% |

Hai cấu hình learning rate 5 có cùng Macro-F1. Mã chọn giá trị cực đại xuất hiện đầu tiên, nên cấu hình được lưu là **5 và 1.000 vòng**. [Bảng validation gốc](../results/runs/20261106T172203Z-a2e0309c/evaluation/comparison/validation.csv)

### 9.2 Linear SVM

Lab01 thử bảy bộ learning rate, số vòng và $C$ trên cùng validation. Sáu cấu hình chỉ dự đoán ham tại ngưỡng điểm 0, đạt Macro-F1 41,25%. Bộ **0,01, 3.000 vòng, $C=100$** đạt 42,16% và bắt được 2 trong 233 spam tại ngưỡng 0. Cấu hình này được chuyển sang bước khảo sát ngưỡng và trọng số lớp. [Bảng cấu hình đã thử](../results/runs/20261106T172203Z-a2e0309c/evaluation/comparison/validation.csv)

### 9.3 Multinomial Naive Bayes

Lần chạy dùng $\alpha=1$ theo mặc định. Bảng validation của Lab01 ghi lưới thử cho Logistic Regression và Linear SVM. Naive Bayes được fit với $\alpha=1$ và đánh giá trên test.

![Cấu hình huấn luyện của ba mô hình](../reports/figures/20261106T172203Z-a2e0309c/training/training_configuration.png)

*Hình 5. Cấu hình của ba mô hình khi huấn luyện cuối.*

## 10. Quy tắc phân loại, chọn ngưỡng và phân tích lỗi

### 10.1 Logistic Regression

Trong mô hình Logistic Regression hiện tại, `predict()` dùng **ngưỡng cố định 0,5**. Mã tạo báo cáo có một bước riêng để so sánh các ngưỡng 0,3 đến 0,7 trên validation. Kết quả này dùng để quan sát sự thay đổi của các chỉ số và được ghi vào `lr_threshold_comparison.csv`. So sánh này chạy khi tạo báo cáo sau khi mô hình cuối đã được huấn luyện. Hàm dự đoán dùng 0,5. [Mã so sánh ngưỡng](../src/report.py#L148) · [Hàm dự đoán](../algorithms/logistic_regression.py#L67)

Kết quả thực tế trên **782 email validation** của lần chạy đã lưu:

| Ngưỡng | Precision spam | Recall spam | Macro-F1 | Spam bỏ lọt (FN) | Ham đánh dấu nhầm (FP) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,3 | 92,09% | 100,00% | 97,01% | 0 | 20 |
| 0,4 | 98,31% | 100,00% | 99,39% | 0 | 4 |
| 0,5 | 99,13% | 98,28% | 99,08% | 4 | 2 |
| 0,6 | 99,07% | 91,42% | 96,56% | 20 | 2 |
| 0,7 | 99,46% | 78,97% | 91,84% | 49 | 1 |

Trong năm ngưỡng khảo sát, **0,4 có Macro-F1 cao nhất**. So với 0,5, ngưỡng này phát hiện thêm 4 spam và đánh dấu nhầm thêm 2 ham. Với 549 ham trên validation, tỷ lệ ham bị đánh dấu nhầm tăng từ 0,36% lên 0,73%. Các con số trên mô tả lần thử ngưỡng ở validation. Kết quả test ở bước 11 được tính với ngưỡng 0,5 của hàm dự đoán. [Dữ liệu đối chiếu ngưỡng](../reports/lr_threshold_comparison.csv)

Để phân tích định tính, có thể đối chiếu `email_id` của từng lỗi với CSV gốc và kiểm tra xác suất dự đoán. Các email sát ngưỡng giúp giải thích tác động của việc dịch ngưỡng. Các email sai với xác suất cách xa ngưỡng có thể gợi ý kiểm tra lại nội dung sau làm sạch hoặc nhãn thật. Bảng trên tổng hợp số lỗi, còn việc phân nhóm nội dung lỗi trên validation cần kiểm tra từng email.

### 10.2 Linear SVM

Lab01 đánh giá SVM với trọng số lớp mặc định, `balanced` và các mức trọng số spam gấp 2, 3 hoặc 4 lần ham. Với mỗi cách đặt trọng số, mã ghi kết quả ở ngưỡng 0 và thử các ngưỡng từ điểm validation. Cấu hình cuối được chọn theo Macro-F1 trên validation.

| Trọng số lớp | Ngưỡng chọn theo từng cấu hình | Macro-F1 | Recall spam | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mặc định | **−0,6620** | **98,33%** | **98,71%** | **3** | **8** |
| Balanced | 0,5293 | 96,51% | 96,14% | 9 | 14 |
| Spam ×2 | 0,2655 | 96,67% | 97,00% | 7 | 15 |
| Spam ×3 | 0,7010 | 97,44% | 98,71% | 3 | 14 |
| Spam ×4 | 0,9074 | 96,99% | 98,28% | 4 | 16 |

Với trọng số mặc định, hạ ngưỡng từ 0 xuống khoảng **−0,6620** tăng số spam phát hiện từ 2 lên 230 trên validation và tăng số ham bị đánh dấu nhầm từ 0 lên 8. Ngưỡng đầy đủ `-0.6620115722742366` được lưu trong mô hình cuối. [Kết quả khảo sát SVM](../results/runs/20261106T172203Z-a2e0309c/evaluation/comparison/svm_study/validation.csv)

### 10.3 Multinomial Naive Bayes

Naive Bayes dự đoán lớp có điểm log cao hơn. Lần chạy giữ quy tắc này khi đánh giá và demo, nên bảng thử ngưỡng riêng áp dụng cho Logistic Regression và SVM. Khi phân tích lỗi test của Naive Bayes, có thể dùng `email_id` để truy lại 39 spam bị bỏ lọt và 3 ham bị đánh dấu nhầm trong tệp dự đoán. [Dự đoán Naive Bayes trên test](../results/runs/20261106T172203Z-a2e0309c/evaluation/naive_bayes/predictions.csv)

## 11. Chốt mô hình và đánh giá trên test

Sau khi chọn cấu hình, Lab01 fit lại TF-IDF và ba mô hình trên toàn bộ tập train ban đầu. Email test được chuyển sang vector bằng TF-IDF đã học rồi được phân loại theo quy tắc của từng mô hình. Kết quả phản ánh một tập test đã được xem trong lần chạy đã lưu. [Báo cáo lần chạy](../reports/report_20261106T172203Z-a2e0309c.md)

Với 686 ham và 291 spam, mô hình dự đoán đúng 681 ham và 288 spam. Ba spam bị bỏ lọt và năm ham bị gắn nhãn spam. Tỷ lệ ham bị đánh dấu nhầm là $5/686$, khoảng **0,73%**.

![Ma trận nhầm lẫn Logistic Regression](../reports/figures/20261106T172203Z-a2e0309c/evaluation/confusion_matrix_lr.png)

*Hình 6. Ma trận nhầm lẫn trên tập test tại ngưỡng 0,5.*

| Mô hình | Accuracy | Macro-F1 | Precision spam | Recall spam | F1 spam | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 99,18% | 99,02% | 98,29% | 98,97% | 98,63% | 3 | 5 |
| Linear SVM | 96,93% | 96,29% | 96,44% | 93,13% | 94,76% | 20 | 10 |
| Naive Bayes | 95,70% | 94,66% | 98,82% | 86,60% | 92,31% | 39 | 3 |

Logistic Regression có Macro-F1 cao nhất trong lần chạy này. Naive Bayes đánh dấu nhầm ít ham hơn, nhưng bỏ lọt 39 spam. Các kết quả mô tả hiệu năng trên một cách chia dữ liệu và một bộ email cụ thể. Hiệu năng với email từ nguồn hoặc thời điểm khác cần được đo trên dữ liệu tương ứng.

![Ma trận nhầm lẫn Linear SVM](../reports/figures/20261106T172203Z-a2e0309c/evaluation/confusion_matrix_svm.png)

*Hình 7. Ma trận của Linear SVM với ngưỡng đã chọn trên validation.*

![Ma trận nhầm lẫn Naive Bayes](../reports/figures/20261106T172203Z-a2e0309c/evaluation/confusion_matrix_nb.png)

*Hình 8. Ma trận của Multinomial Naive Bayes với quy tắc chọn lớp có điểm lớn nhất.*

| Mô hình | Ham dự đoán đúng | Spam dự đoán đúng | Tỷ lệ ham đánh dấu nhầm | Thời gian fit |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 681 | 288 | 0,73% | 5,314 giây |
| Linear SVM | 676 | 271 | 1,46% | 7,903 giây |
| Multinomial Naive Bayes | 683 | 252 | 0,44% | 0,0054 giây |

Thời gian là phép đo của lần fit cuối, tách khỏi chi phí tạo TF-IDF và thử cấu hình. Naive Bayes fit nhanh nhất trong lần chạy, phù hợp với cách cộng thống kê thay cho cập nhật qua nhiều vòng. [Thời gian fit gốc](../results/runs/20261106T172203Z-a2e0309c/evaluation/comparison/training.csv)

![So sánh chỉ số test của ba mô hình](../reports/figures/20261106T172203Z-a2e0309c/evaluation/metrics_comparison.png)

*Hình 9. So sánh các chỉ số test ở quy tắc dự đoán đã chốt của mỗi mô hình.*

![Đường ROC của ba mô hình](../reports/figures/20261106T172203Z-a2e0309c/evaluation/roc_curve_all.png)

*Hình 10. ROC mô tả tỷ lệ phát hiện spam và tỷ lệ đánh dấu nhầm ham khi quét qua các ngưỡng điểm. Bảng test ở trên báo cáo kết quả tại quy tắc đã dùng để dự đoán.*

Bộ test này cũng đã được dùng để báo cáo baseline. Điểm test mô tả hiệu năng trên split đã lưu. Các bước chọn cấu hình và chọn ngưỡng SVM trong mã dùng validation.

## 12. Lưu mô hình, dự đoán và giám sát

Lab01 lưu mô hình và TF-IDF dưới dạng `.joblib`. Chương trình `predict.py` nạp hai thành phần này để chạy luồng **email thô → văn bản sạch → vector TF-IDF → điểm hoặc xác suất theo mô hình → nhãn**. [Mã dự đoán](../src/predict.py)

Phần đã thực hiện là demo dự đoán. Nếu triển khai thực tế, có thể theo dõi tỷ lệ email bị gắn cờ và dùng phản hồi đã xác minh để đo spam bỏ lọt và ham bị đánh dấu nhầm.

Chạy demo từ thư mục gốc của dự án:

```bash
python -m Lab01.src.predict --model logistic_regression --text "Congratulations! Claim your free cash prize now"
python -m Lab01.src.predict --model svm --text "Congratulations! Claim your free cash prize now"
python -m Lab01.src.predict --model naive_bayes --text "Congratulations! Claim your free cash prize now"
```

Để đo chất lượng sau khi triển khai, cần có nhãn đã xác minh cho mẫu email được theo dõi. Khi có nhãn, có thể tính lại recall spam và tỷ lệ ham bị gắn nhầm theo từng giai đoạn để so sánh với kết quả trong báo cáo.
