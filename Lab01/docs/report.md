# Lab01 — Phân loại email spam

## 1. Tổng quan và mục tiêu bài toán

Phân loại email thành `ham` (email thường, nhãn `0`) hoặc `spam` (nhãn `1`). Tệp [spam_ham_dataset.csv](../Data/raw/spam_ham_dataset.csv) có 5.171 email. Quy trình dùng `text` làm đầu vào và `label_num` làm nhãn. CSV không có cột riêng cho người gửi hoặc người nhận.

Đầu vào khi dự đoán là văn bản của một email. Đầu ra chính là nhãn ham hoặc spam. Logistic Regression trả thêm xác suất spam, còn Linear SVM trả điểm quyết định và ngưỡng. Mục tiêu thực nghiệm là so sánh hai mô hình trên cùng cách chia dữ liệu và đặc trưng, đồng thời đo cả spam bị bỏ lọt (FN) và ham bị chặn nhầm (FP).



## 2. Dữ liệu

Nhãn `0` biểu thị ham và nhãn `1` biểu thị spam. CSV gốc có 5.171 email, gồm 3.672 ham và 1.499 spam. Sau khi loại 287 email trùng theo văn bản sạch, còn 4.884 email. Chia phân tầng 80/20 với seed 42 để giữ tỷ lệ lớp gần nhau.

| Tập | Ham | Spam | Tổng |
|---|---:|---:|---:|
| Train | 2.741 | 1.166 | 3.907 |
| Test | 686 | 291 | 977 |

Trong CSV gốc, ham chiếm **71,01%** và spam chiếm **28,99%**. Trên test, một mô hình luôn đoán ham vẫn đạt 686/977 = **70,21% accuracy** nhưng recall spam bằng 0. Chia phân tầng giữ tỷ lệ lớp gần nhau giữa train và test, đồng thời giúp các tập có đủ mẫu spam để tính chỉ số. Cách chia này vẫn giữ mức mất cân bằng của dữ liệu.

### Khám phá dữ liệu gốc
CSV gốc có 178 dòng trùng nội dung trước khi chuẩn hóa. Trung vị số từ tách bằng khoảng trắng là 123 ở ham và 114 ở spam. Spam có trung bình 1,64 dấu chấm than và 1,03 dấu hiệu URL mỗi email, so với 0,32 và 0,11 ở ham. Đây là thống kê mô tả. Các số đếm ký tự và URL trong EDA chưa được đưa thành cột đầu vào riêng. Xem [bảng EDA theo lớp](../result/eda/features_by_label.csv).

## 3. Quy trình xử lý

```text
CSV → tiền xử lý → TF-IDF → huấn luyện cấu hình thử
    → chọn tham số trên validation → fit lại trên toàn bộ train → đánh giá trên test

Email mới → cùng bước làm sạch → TF-IDF đã lưu → mô hình đã lưu → nhãn ham/spam
```

CSV gốc có 3.672 ham và 1.499 spam. Quy trình kiểm tra văn bản và nhãn, bỏ HTML, chuyển chữ thường, bỏ dấu câu và từ dừng tiếng Anh, rồi chuẩn hóa khoảng trắng. Sau khi kiểm tra văn bản rỗng và nhãn xung đột, quy trình loại trùng theo văn bản sạch và chia train/test phân tầng 80/20 với seed 42. Lần chạy hiện tại loại 287 email trùng. Xem [thống kê dữ liệu gốc](../result/eda/overview.png) và [tóm tắt tiền xử lý](../Data/preprocessing/summary.json).

Cả ba mô hình dùng TF-IDF của từ đơn và cụm hai từ, chuẩn hóa L2, với tối đa 50.000 cột. Ma trận được lưu ở dạng thưa CSR. Xem [danh sách đặc trưng](../Data/preprocessing/09_tfidf/feature_names.csv).

| Đặc trưng trong đề bài | Cách sử dụng hiện tại |
|---|---|
| Tần suất từ | Dùng làm thành phần của TF-IDF cho từ đơn và cụm hai từ. |
| Tiêu đề và thân email | Xử lý chung trong cột `text`, chưa tạo hai nhóm đặc trưng riêng. |
| Tần suất ký tự như `!` và `$` | Chưa trích xuất riêng. Bước làm sạch loại các ký tự này trước TF-IDF. |

`label_num` chỉ dùng làm nhãn. `email_id` dùng để đối chiếu các đầu ra. Cả hai đều nằm ngoài vector đặc trưng.


### Tác dụng của từng bước làm sạch

| Bước | Mục đích | Thông tin có thể mất |
|---|---|---|
| Bỏ HTML | Giữ chữ hiển thị và giảm nhiễu từ thẻ định dạng, script và style | Cấu trúc HTML hoặc URL chỉ nằm trong thuộc tính của thẻ |
| Chuyển chữ thường | Gộp các biến thể như `FREE` và `free` | Dấu hiệu viết hoa toàn bộ |
| Bỏ dấu câu | Giảm khác biệt về định dạng và ngắt token ở dấu phân cách | Tần suất `!`, ký hiệu tiền hoặc cấu trúc của URL |
| Bỏ từ dừng | Giảm số từ phổ biến ít mang tính phân biệt | Từ phủ định như `not` và một phần ngữ cảnh |
| Chuẩn hóa khoảng trắng | Thu gọn dấu cách và xuống dòng thành một dấu cách | Bố cục dòng, kể cả ranh giới giữa tiêu đề và thân thư nếu nằm trong cùng cột |

Ví dụ minh họa, `<b>FREE!!!</b> Click now for the prize.` thành `free click prize`. Nội dung chính còn lại, trong khi kiểu viết hoa, ba dấu chấm than và từ dừng bị loại.

### Công thức TF-IDF

Với từ hoặc cụm từ `t` trong email `d`, TF là số lần `t` xuất hiện trong `d`. Với thiết lập mặc định của vectorizer trong Lab01, `IDF(t) = ln((1 + N)/(1 + df(t))) + 1`, trong đó `N` là số email dùng để fit và `df(t)` là số email trong tập fit chứa `t`. Trọng số ban đầu là `TF(t,d) × IDF(t)`. IDF hạ mức ảnh hưởng tương đối của từ xuất hiện trong nhiều email.

Unigram là một từ, chẳng hạn `free`. Bigram là cặp từ liên tiếp sau xử lý, chẳng hạn `free prize`. Bigram giữ thêm ngữ cảnh nhưng làm từ vựng lớn hơn và có thể ít xuất hiện. Sau khi tính trọng số, chuẩn hóa L2 chia mỗi vector khác 0 cho căn bậc hai của tổng bình phương các phần tử. CSR lưu các giá trị khác 0, chỉ số cột và con trỏ bắt đầu từng hàng. Nhờ đó, ma trận 50.000 cột không cần lưu số 0 cho mọi từ vắng mặt.

### Pipeline và phòng tránh rò rỉ dữ liệu

Pipeline hiện tại được ghép qua các hàm xử lý và hai thành phần được lưu riêng là vectorizer và classifier. Khi dự đoán, dùng lại `clean_text()`, gọi `transform()` với vectorizer đã lưu và chuyển ma trận vào classifier. Quy trình chọn cấu hình chỉ fit từ vựng và IDF trên train con. Khi fit mô hình cuối, vectorizer học lại trên toàn bộ train. Validation và test chỉ được biến đổi bằng vectorizer của giai đoạn tương ứng.

Trong thiết kế 5-Fold, mỗi fold tạo và fit pipeline riêng. Mọi bước học từ dữ liệu như từ vựng, IDF hoặc chuẩn hóa đặc trưng bổ sung đều chỉ fit trên phần train của fold. Loại trùng trước khi chia giúp tránh đưa cùng văn bản sạch sang hai tập.

#### Sơ đồ luồng dữ liệu

```text
CSV gốc → kiểm tra nhãn → làm sạch để lọc trùng → chia phân tầng
  ├─ Train → train con: fit TF-IDF và mô hình thử
  │        → validation: transform → chọn tham số và ngưỡng
  │        → toàn bộ train: fit lại TF-IDF và mô hình cuối
  └─ Test  → transform bằng TF-IDF fit trên toàn bộ train → đánh giá
```

Ví dụ, một cụm từ chỉ xuất hiện trong validation sẽ không được thêm vào từ vựng của lượt chọn cấu hình. Tương tự, từ chỉ có trong test không được dùng để tính IDF. Việc đã xem điểm trên test qua nhiều vòng là một giới hạn riêng. Cần tập mới để đánh giá độc lập sau khi đã chốt thiết kế.

### Cấu trúc thư mục

```text
Lab01/
├── algorithms/   # Các class phân loại email
├── src/          # Tiền xử lý, huấn luyện, đánh giá và dự đoán
├── Data/         # CSV gốc và dữ liệu trung gian
├── models/       # Vectorizer và mô hình sinh khi chạy
└── result/       # Chỉ số, dự đoán và kết quả thực nghiệm
```

Xem [cây mã nguồn chi tiết](#phụ-lục-a--cấu-trúc-mã-nguồn-và-đầu-ra).

## 4. Thuật toán

### 4.1. Logistic Regression

- `fit()` khởi tạo vector trọng số và bias bằng 0. Với mỗi vòng, tính điểm `z = X @ w + b` và xác suất spam `p = sigmoid(z)`. Sigmoid chặn đầu vào trong khoảng −500 đến 500 để tránh tràn khi tính hàm mũ.
- Sai số là `p − y`. Gradient binary cross-entropy theo trọng số là `X.T @ (p − y) / n`, còn gradient theo bias là trung bình của `p − y`. Hai tham số được cập nhật theo hướng ngược gradient trên toàn bộ tập đang fit.
- `predict_proba()` trả một xác suất lớp spam cho mỗi hàng. `predict()` gán spam từ 0,5 trở lên. Trọng số dương làm tăng điểm spam trước sigmoid khi đặc trưng tương ứng xuất hiện.

Hàm mất mát là `BCE = −mean(y log(p) + (1−y) log(1−p))`. Mô hình cho xác suất và dễ đọc dấu của hệ số. Với ranh giới tuyến tính, tương tác phức tạp chỉ được thể hiện qua đặc trưng đã xây dựng. Cấu hình tốc độ học và số vòng ảnh hưởng đến mức hội tụ.

Trong `z = X @ w + b`, mỗi hàng của `X` là vector TF-IDF, `w` chứa một hệ số cho mỗi cột và `b` dịch chuyển ranh giới. Với tốc độ học `η`, gradient descent cập nhật `w ← w − η∇w` và `b ← b − η∇b`. Tốc độ quá nhỏ có thể khiến mô hình tiến chậm trong số vòng cho trước. Tốc độ quá lớn có thể làm loss dao động. Lab01 chạy số vòng cố định. Bảng trace ở mục 5 cho biết loss đã giảm đến đâu, còn điểm validation dùng để chọn cấu hình.

### 4.2. Linear SVM

#### Nguyên lý siêu phẳng và margin

Siêu phẳng gốc là `w·x + b = 0`. Khi `w` khác 0, khoảng cách có dấu tới siêu phẳng là `(w·x+b)/‖w‖`. Hai biên `w·x+b = ±1` cách nhau `2/‖w‖`. Trong Linear SVM chuẩn, các mẫu nằm trên hoặc vi phạm biên có vai trò xác định nghiệm tối ưu. Lab01 tối ưu trực tiếp trọng số và bias, dùng mẫu có margin nhỏ hơn 1 để tính thành phần gradient của Hinge Loss. Mô hình lưu trọng số và bias.

#### Hàm mất mát Hinge Loss

`fit()` đổi nhãn sang −1/+1. Với điểm `s = X @ w + b`, margin có dấu là `y_signed × s`. Hinge Loss bằng `max(0, 1 − margin)`. Hàm mục tiêu là `½‖w‖² + C × mean(sample_weight × hinge)`. Số hạng L2 hạn chế độ lớn của trọng số, còn `C` điều chỉnh ảnh hưởng của lỗi margin.

#### Cập nhật trọng số và bias

Với `a = sample_weight × y_signed × (margin < 1)`, gradient theo trọng số là `w − C × X.T @ a / n`, còn gradient theo bias là `−C × mean(a)`. Mỗi vòng trừ tốc độ học nhân gradient tương ứng khỏi `w` và `b`.

#### Trọng số lớp và ngưỡng quyết định

Trọng số mặc định là 1 cho mỗi lớp. Chế độ `balanced` dùng `n / (2 × số mẫu của lớp)`. Các phương án spam ×2, ×3, ×4 giữ trọng số ham bằng 1. Sau khi học, `predict()` gán spam nếu điểm đạt ngưỡng đã lưu. Với ngưỡng −0,662, điểm −0,3 được gán spam và −0,8 được gán ham. Điểm này dùng để xếp hạng và chưa được hiệu chuẩn thành xác suất.

### 4.3. Multinomial Naive Bayes

Theo Bayes, `P(c|x) ∝ P(c) × P(x|c)`. Giả định Naive Bayes coi các lần xuất hiện của đặc trưng là độc lập có điều kiện theo lớp. Mô hình dùng điểm `log P(c) + Σⱼ xⱼ log θ(c,j)`, với `θ(c,j) = (S(c,j)+α)/(Σₖ S(c,k)+αV)`. Ở đây `S(c,j)` là tổng trọng số TF-IDF của cột `j` trong lớp `c`, `V` là số đặc trưng và `α=1` là hệ số làm trơn. Mã dùng TF-IDF không âm thay cho số đếm từ nguyên. Naive Bayes là mô hình tham chiếu vì học từ thống kê theo lớp mà không cần nhiều vòng gradient.

### 4.4. So sánh các thuật toán

| Thuật toán | Nguyên lý | Đầu ra trong Lab01 | Ưu điểm | Hạn chế |
|---|---|---|---|---|
| Logistic Regression | Tối ưu binary cross-entropy qua sigmoid | Nhãn và xác suất spam | Có ngưỡng dễ diễn giải, đọc được hướng đóng góp của hệ số | Cần chọn tốc độ học và số vòng phù hợp |
| Linear SVM | Tối ưu hinge loss và regularization L2 | Nhãn, điểm quyết định và ngưỡng | Phù hợp với đặc trưng văn bản thưa nhiều chiều | Điểm cần ngưỡng phù hợp và không có ý nghĩa xác suất trực tiếp |
| Multinomial Naive Bayes | Prior và thống kê đặc trưng theo lớp trong miền log | Nhãn | Fit nhanh, cấu hình đơn giản | Giả định độc lập hạn chế khả năng mô hình hóa đặc trưng tương quan |

## 5. Huấn luyện và tối ưu

### Thực nghiệm đã chạy

| Thuật toán | Cách triển khai |
|---|---|
| [Logistic Regression](../algorithms/logistic_regression.py) | Dùng sigmoid và gradient của binary cross-entropy để học trọng số. Đã kiểm tra huấn luyện và dự đoán trực tiếp trên CSR của Lab01. |
| [Linear SVM](../algorithms/svm.py) | Cập nhật trọng số và bias từ hinge loss có regularization L2. Điểm phân loại được so với ngưỡng để quyết định nhãn. Đã kiểm tra với TF-IDF mẫu dạng dense và CSR. |
| [Multinomial Naive Bayes](../algorithms/naive_bayes.py) | Cộng trọng số đặc trưng theo lớp, làm trơn bằng `alpha` và chọn lớp có điểm log cao hơn. Đã đối chiếu kết quả mẫu dạng dense và CSR với scikit-learn. |

- **Baseline.** Fit TF-IDF và huấn luyện các mô hình mặc định trên toàn bộ 3.907 email train, sau đó biến đổi và đánh giá trên test.
- **Validation.** Tách phân tầng 20% từ train với seed 43, được 3.125 email train con và 782 email validation. Fit TF-IDF trên train con, rồi dùng cùng bộ biến đổi cho validation. Chọn tham số theo macro F1 validation. Logistic Regression chọn tốc độ học 5, 1.000 vòng và giữ ngưỡng 0,5. SVM chọn tốc độ học 0,01, 3.000 vòng và C=100, sau đó chọn ngưỡng trên cùng validation. Naive Bayes giữ alpha=1. Xem [bảng thử tham số](../result/comparison/validation.csv) và [cấu hình đã chọn](../result/comparison/selection.json).
- **Mô hình cuối.** Fit lại TF-IDF và các mô hình trên toàn bộ 3.907 email train, rồi biến đổi và đánh giá trên cùng 977 email test. SVM áp dụng ngưỡng đã chọn trên validation.

Macro F1 được chọn vì lấy trung bình F1 của ham và spam với trọng số bằng nhau. Điều này làm lộ trường hợp accuracy khá cao nhờ dự đoán lớp ham phổ biến nhưng bỏ lọt toàn bộ spam. Với mỗi phương án trọng số SVM, mã thử ngưỡng 0, mọi điểm validation phân biệt và một ngưỡng ngay trên điểm cao nhất để xét cả trường hợp không dự đoán spam. Mỗi ngưỡng được chấm bằng macro F1. Nếu hòa trong cùng phương án, ưu tiên ít FP rồi ngưỡng gần 0 hơn. Khi so sánh giữa các phương án có cùng macro F1, ưu tiên mô hình không trọng số trước các tiêu chí phụ.

### Cấu hình mô hình cuối

| Mô hình | Learning rate | Số vòng | Tham số khác |
|---|---:|---:|---|
| Logistic Regression | 5 | 1.000 | Ngưỡng xác suất 0,5 |
| Linear SVM | 0,01 | 3.000 | C = 100, trọng số đều, ngưỡng điểm ≈ −0,662 |
| Multinomial Naive Bayes | — | — | α = 1 |

Logistic Regression đạt cùng Macro F1 validation với 1.000 và 3.000 vòng ở tốc độ 5. Cấu hình 1.000 vòng được chọn. SVM dùng ngưỡng chọn trên validation sau khi thử trọng số lớp. Naive Bayes giữ cấu hình mặc định. Xem [lưới tham số đầy đủ](#phụ-lục-b--thí-nghiệm-validation).

### Chẩn đoán baseline từ trace

| Mô hình | Vòng 0 | Vòng 100 | Vòng 1.000 | Điểm spam cao nhất ở vòng cuối |
|---|---:|---:|---:|---:|
| Logistic Regression (BCE) | 0,6931 | 0,5970 | 0,5143 | 0,4569 |
| Linear SVM (hàm mục tiêu) | 1,0000 | 0,8328 | 0,5964 | −0,9909 |

BCE vẫn giảm ở cuối baseline nhưng xác suất spam cao nhất chưa vượt 0,5. Với SVM mặc định, điểm cao nhất còn dưới 0 và hàm mục tiêu thay đổi rất ít từ vòng 500 đến 1.000. Xem [trace LR](../result/comparison/logistic_regression_trace.csv) và [trace SVM](../result/comparison/svm_trace.csv).

### Thiết kế thực nghiệm tiếp theo với Stratified 5-Fold

**Chưa chạy trong phiên bản hiện tại.** Chia 3.907 email train thành 5 fold phân tầng theo nhãn, dùng cùng các fold cho mọi mô hình và cấu hình. Mỗi lượt huấn luyện trên 4 fold và đánh giá trên fold còn lại. Báo cáo trung bình và độ lệch chuẩn của macro F1, precision và recall spam. Giữ test 977 email ngoài quá trình chọn cấu hình. Nếu tối ưu ngưỡng SVM, có thể dùng dự đoán ngoài fold trên train để chọn ngưỡng và cần kiểm tra lại sau fit cuối. Nếu muốn dùng điểm CV làm ước lượng độc lập cho cả bước chọn ngưỡng, chọn ngưỡng ở vòng chia bên trong từng fold đánh giá.

| Cấu hình đề xuất | Đặc trưng | Trạng thái |
|---|---|---|
| A | TF-IDF từ văn bản sạch, gồm từ đơn và cụm hai từ | Đã dùng với một lần chia validation. Chưa có kết quả 5-Fold. |
| B | A cộng TF-IDF ký tự từ văn bản | Chưa triển khai và chưa đo. |
| C | B cộng số đếm dấu câu, URL và thống kê độ dài từ văn bản gốc | Chưa triển khai và chưa đo. |

A/B/C ở bảng này là **cấu hình đặc trưng**. Các ký hiệu A–D ở bảng thí nghiệm SVM bên dưới chỉ các phương án trọng số lớp và ngưỡng.

## 6. Đánh giá mô hình

### Cách tính chỉ số

Ma trận nhầm lẫn có dạng `[[TN, FP], [FN, TP]]`. Với spam là lớp dương, accuracy là `(TP+TN)/N`, precision là `TP/(TP+FP)`, recall là `TP/(TP+FN)` và F1 là `2 × precision × recall / (precision+recall)`. Macro F1 lấy trung bình F1 của hai lớp. Các điểm dưới đây lấy từ tập test, không phải trung bình 5-Fold.

### Kết quả hiện tại

| Mô hình | Accuracy | Macro F1 | Recall spam | FN | FP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 99,18% | 99,02% | 98,97% | 3 | 5 |
| Linear SVM | 96,93% | 96,29% | 93,13% | 20 | 10 |
| Multinomial Naive Bayes | 95,70% | 94,66% | 86,60% | 39 | 3 |

Kết quả tính trên 977 email test. FN là spam bị bỏ lọt, FP là ham bị chặn nhầm. Linear SVM dùng ngưỡng điểm khoảng **−0,662**, chọn trên validation. Xem [bảng chỉ số](../result/comparison/metrics.csv) và ma trận nhầm lẫn của [Logistic Regression](../result/logistic_regression/confusion_matrix.png), [Linear SVM](../result/svm/confusion_matrix.png) và [Naive Bayes](../result/naive_bayes/confusion_matrix.png). Trong mỗi ma trận, hàng là nhãn thật và cột là nhãn dự đoán, theo thứ tự ham, spam.

| Mô hình | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 681 | 5 | 3 | 288 |
| Linear SVM | 676 | 10 | 20 | 271 |
| Multinomial Naive Bayes | 683 | 3 | 39 | 252 |

Ví dụ, Logistic Regression tìm được 288 trong 291 spam thật và gán nhầm 5 ham. Do đó recall spam là 288/291 = 98,97% và precision spam là 288/293 = 98,29%.

### Chỉ số theo lớp trên test

| Mô hình | Lớp | Precision | Recall | F1 | Số mẫu |
|---|---|---:|---:|---:|---:|
| Logistic Regression | ham | 99,56% | 99,27% | 99,42% | 686 |
| Logistic Regression | spam | 98,29% | 98,97% | 98,63% | 291 |
| Linear SVM | ham | 97,13% | 98,54% | 97,83% | 686 |
| Linear SVM | spam | 96,44% | 93,13% | 94,76% | 291 |
| Multinomial Naive Bayes | ham | 94,60% | 99,56% | 97,02% | 686 |
| Multinomial Naive Bayes | spam | 98,82% | 86,60% | 92,31% | 291 |

Xem [tệp kết quả và thông tin tái lập](#phụ-lục-c--đầu-ra-và-tái-lập-thực-nghiệm).

## 7. So sánh và phân tích

### Các mốc so sánh

| Mô hình và mốc | Accuracy test | Macro F1 test | FN | FP |
|---|---:|---:|---:|---:|
| Logistic Regression, baseline | 70,21% | 41,25% | 291 | 0 |
| Linear SVM, baseline C=1, ngưỡng 0 | 70,21% | 41,25% | 291 | 0 |
| Linear SVM, sau chọn C=100, ngưỡng 0 (vòng trước) | 70,32% | 41,62% | 290 | 0 |
| Linear SVM, hiện tại C=100, ngưỡng −0,662 | 96,93% | 96,29% | 20 | 10 |

Naive Bayes giữ cùng cấu hình và kết quả giữa baseline và lần chạy hiện tại. Xem [bảng baseline](../result/comparison/baseline.csv), [trace Logistic Regression](../result/comparison/logistic_regression_trace.csv) và [trace SVM](../result/comparison/svm_trace.csv).

Trên tập test hiện tại, Logistic Regression có macro F1 99,02% so với 96,29% của SVM. Logistic Regression cũng có ít FP hơn (5 so với 10) và ít FN hơn (3 so với 20). Với tiêu chí giảm cả hai loại lỗi, Logistic Regression là lựa chọn phù hợp hơn trên lần chia này.

### Vì sao đổi ngưỡng SVM?

SVM không trọng số đạt **AP 98,73%** trên validation, cho thấy điểm quyết định xếp hạng spam tốt. Tuy vậy, ở ngưỡng 0, mô hình chỉ phát hiện **2/233 spam** và chặn nhầm 0 ham. Hạ ngưỡng xuống khoảng −0,662 giúp cùng mô hình phát hiện **230/233 spam** và chặn nhầm **8 ham**. AP được tính từ điểm liên tục nên giữ nguyên khi chỉ đổi ngưỡng.

Các thử nghiệm so sánh trọng số đều, balanced và trọng số spam ×2, ×3, ×4. Ngưỡng được chọn theo macro F1 validation, khi hòa ưu tiên ít FP hơn rồi ngưỡng gần 0 hơn. Khi macro F1 hòa giữa phương án có và không có trọng số, ưu tiên phương án không trọng số. Phương án trọng số đều với ngưỡng −0,662 đạt macro F1 validation cao nhất, 98,33%. Xem [10 phương án đánh giá trọng số và ngưỡng](../result/comparison/svm_study/validation.csv), [phân vị điểm](../result/comparison/svm_study/score_distribution.csv), [đường precision–recall](../result/comparison/svm_study/unweighted_pr.csv) và [ngưỡng đã lưu](../result/comparison/svm_study/selection.json).

Trên validation, hạ ngưỡng từ 0 xuống khoảng −0,662 làm recall spam tăng từ **0,86% lên 98,71%** và precision giảm từ **100% xuống 96,64%**. Precision 100% ban đầu chỉ dựa trên hai dự đoán spam. Trên test, Logistic Regression có cả precision lẫn recall spam cao hơn SVM tại các ngưỡng đang dùng. Kết quả validation cho thấy điểm SVM xếp hạng tốt nhưng vị trí ngưỡng ảnh hưởng lớn đến số lỗi. Chênh lệch trên test phản ánh các cấu hình và ngưỡng đã thử trên lần chia này.

### Phân vị điểm SVM không trọng số trên validation

| Lớp thật | Min | P10 | Median | P90 | Max | Điểm ≥ 0 |
|---|---:|---:|---:|---:|---:|---:|
| Ham | −1,22629 | −1,11343 | −0,97302 | −0,75939 | −0,35596 | 0,00% |
| Spam | −0,68437 | −0,61493 | −0,49246 | −0,30748 | 0,02295 | 0,86% |

AP tham chiếu theo tỷ lệ spam là 233/782 = 29,80%. Trung vị spam cao hơn ham dù cả hai đều âm. Sự dịch chuyển ngưỡng khai thác khác biệt về điểm giữa hai lớp. Ma trận validation đổi từ `[[549, 0], [231, 2]]` sang `[[541, 8], [3, 230]]`. Trên test sau fit lại, hai mốc lịch sử và hiện tại tương ứng với `[[686, 0], [290, 1]]` và `[[676, 10], [20, 271]]`.

#### Các phương án tiêu biểu trên validation

| Trọng số lớp | Ngưỡng | Macro F1 | FN | FP |
|---|---:|---:|---:|---:|
| Đều | 0 | 42,16% | 231 | 0 |
| Đều | −0,662012 | 98,33% | 3 | 8 |
| Balanced | 0 | 76,99% | 0 | 174 |
| Spam ×3 | 0,700963 | 97,44% | 3 | 14 |

Xem [đủ 10 phương án](#10-phương-án-trọng-số-lớp-và-ngưỡng-svm).

### Phân tích lỗi Logistic Regression

Ghép dự đoán với dữ liệu test theo `email_id`. Cả 977 ID và nhãn đều khớp. Mô hình gán spam khi xác suất từ 0,5 trở lên. Xem [dự đoán từng email](../result/logistic_regression/predictions.csv).

Đóng góp `x_j × w_j` được tính trước sigmoid. Dấu dương đẩy điểm về spam và dấu âm đẩy về ham.

| ID | Lỗi | Xác suất spam | Nội dung và thay đổi sau làm sạch | Đóng góp và khoảng cách đến ngưỡng |
|---:|---|---:|---|---|
| 4447 | FN | 0,406211 | Quảng cáo vay dùng chữ chèn như `zfree` và `qquote`. Dấu chấm than và `%` mất, chữ chèn vẫn còn. | `flow` −0,449 và `subject new` −0,264. Dưới ngưỡng 0,094. |
| 534 | FN | 0,488380 | Thân thư gồm các trích dẫn rời, ít dấu hiệu quảng cáo rõ. Dấu phân cách mất, các từ nội dung vẫn còn. | `know` −0,281 và `need` −0,169, đối lại `world` +0,225. Sát ngưỡng, thấp hơn 0,012. |
| 1012 | FN | 0,443805 | Thư rất ngắn với tiêu đề và đoạn kết khó hiểu. Dấu ba chấm mất, `know` và `face` vẫn còn. | `know` −0,614 lớn hơn `face` +0,233. Dưới ngưỡng 0,056. |
| 26 | FP | 0,722264 | Xác nhận đăng ký dịch vụ, có nội dung miễn phí và nhiều liên kết. Dấu phân cách URL mất, `http`, `www` và `free` còn lại. | `http` +0,457 và `email` +0,353. Trên ngưỡng 0,222. |
| 3423 | FP | 0,502311 | Thư mời tham gia chương trình thưởng. Ký hiệu tiền và dấu chấm than mất, `free`, `click`, `earn` vẫn còn. | `free` +0,158 và `click` +0,072, đối lại `mail` −0,112. Sát ngưỡng, cao hơn 0,002. |
| 5025 | FP | 0,663749 | Thư mời ăn tối có hướng dẫn bật và tắt nhạc. Biểu tượng cảm xúc mất, `click`, `stop` và `dinner` vẫn còn. | `click` +0,382 và `stop` +0,292, đối lại `dinner` −0,085. Trên ngưỡng 0,164. |
| 3212 | FP | 0,666559 | Thư hỏi thăm có chân trang quảng bá dịch vụ email. Dấu phân cách URL mất, `http`, `www` và `free` vẫn còn. | `http` +0,544 và `free` +0,117. Trên ngưỡng 0,167. |
| 4288 | FP | 0,675421 | Chuyện cười chuyển tiếp, có dấu trích dẫn và dấu chấm than lặp. Các dấu mất, từ chỉ vai trò gia đình lặp lại vẫn còn. | `wife` +0,427 và `husband` +0,296. Trên ngưỡng 0,175. |

Bốn trong năm FP có từ liên quan đến liên kết, thao tác nhấp hoặc nội dung miễn phí. Hai lỗi 534 và 3423 cách ngưỡng dưới 0,012 điểm xác suất. Việc chữ chèn làm yếu tín hiệu quảng cáo vẫn là giả thuyết từ nội dung.

### Demo với cấu hình hiện tại

| Mô hình | Email họp | Email nhận thưởng |
|---|---|---|
| Logistic Regression | ham, xác suất spam 0,08875 | spam, xác suất spam 0,78093 |
| Linear SVM | ham, điểm −0,88302 | spam, điểm −0,58738 |
| Multinomial Naive Bayes | ham | spam |

Xem [lệnh chạy và nội dung hai email mẫu](../README.md#chạy-nhanh).

## 8. Kết luận và hướng phát triển

Lab01 đã xây dựng được luồng phân loại từ email thô đến dự đoán, dùng chung TF-IDF để so sánh mô hình và lưu kết quả theo từng email. Thực nghiệm cho thấy Logistic Regression đạt kết quả cao nhất trên lần chia hiện tại. Với SVM, thay đổi ngưỡng giúp chuyển khả năng xếp hạng spam thành số dự đoán spam đúng cao hơn.

Các bước tiếp theo là chạy Stratified 5-Fold cho A/B/C trên train, so sánh độ ổn định qua các fold và xác nhận mô hình cùng ngưỡng trên dữ liệu mới. Nếu có dữ liệu cấu trúc phù hợp, có thể tách tiêu đề và thân thư hoặc khai thác trường người gửi và người nhận.

Kế hoạch kiểm chứng là chạy cùng năm fold phân tầng trên tập train cho A/B/C, tổng hợp trung bình và độ lệch chuẩn theo fold, rồi chốt cấu hình trước khi đánh giá trên dữ liệu mới. Dữ liệu mới nên có nhãn ham/spam và được kiểm tra trùng với dữ liệu đã dùng. So sánh lại FP, FN và phân bố điểm SVM để xác nhận ngưỡng sau fit lại.

### Giới hạn

Tập test 977 email đã được xem qua nhiều vòng. Khi kết quả test ảnh hưởng đến quyết định tiếp tục điều chỉnh, tập này mất tính độc lập với quá trình phát triển. **Accuracy 99,18% của Logistic Regression là kết quả trên tập test hiện tại. Hiệu quả trên email thực tế hoàn toàn mới vẫn cần được đo riêng.** Khi fit lại TF-IDF và SVM trên toàn bộ train, thang điểm có thể thay đổi. Ngưỡng chọn trên mô hình validation cần được xác nhận trên dữ liệu mới khi áp dụng cho mô hình cuối.

## Phụ lục A — Cấu trúc mã nguồn và đầu ra

```text
Lab01/                              # Bài lab phân loại email spam
├── README.md                       # Hướng dẫn cài đặt, chạy và xem đầu ra
├── docs/report.md                  # Báo cáo phương pháp, kết quả và phân tích lỗi
├── algorithms/                     # Các class phân loại email
│   ├── __init__.py                 # Đưa ba class ra cấp package để import
│   ├── logistic_regression.py      # Huấn luyện và dự đoán bằng Logistic Regression
│   ├── naive_bayes.py              # Huấn luyện và dự đoán bằng Multinomial Naive Bayes
│   └── svm.py                      # Huấn luyện Linear SVM và dự đoán theo ngưỡng
├── src/                            # Các module xử lý, huấn luyện và dự đoán
│   ├── __init__.py                 # Đánh dấu src là package Python
│   ├── eda.py                      # Thống kê dữ liệu gốc và vẽ biểu đồ
│   ├── preprocessing.py            # Làm sạch, chia dữ liệu và tạo TF-IDF
│   ├── train.py                    # Chạy baseline, chọn cấu hình và huấn luyện lại
│   ├── evaluate.py                 # Tính chỉ số và xuất ma trận nhầm lẫn
│   └── predict.py                  # Nạp mô hình và phân loại email mới
├── notebook/lab1.ipynb             # Notebook minh họa dự đoán với ba mô hình
├── Data/                           # Dữ liệu đầu vào và dữ liệu đã xử lý
│   ├── raw/spam_ham_dataset.csv    # CSV gốc chứa nội dung và nhãn email
│   └── preprocessing/              # Lưu CSV trung gian, ma trận TF-IDF và nhãn
├── models/                         # Sinh khi chạy tiền xử lý và huấn luyện
└── result/                         # Đầu ra thống kê, so sánh và đánh giá
    ├── eda/                        # Bảng thống kê và biểu đồ dữ liệu
    ├── comparison/                 # Bảng so sánh và kết quả thử tham số
    │   ├── baseline/               # Kết quả chi tiết của cấu hình mặc định
    │   └── svm_study/              # Thử nghiệm trọng số lớp và ngưỡng SVM
    ├── logistic_regression/        # Kết quả test của Logistic Regression
    ├── naive_bayes/                # Kết quả test của Multinomial Naive Bayes
    └── svm/                        # Kết quả test của Linear SVM
```

## Phụ lục B — Thí nghiệm validation

### Lưới tham số trên validation

| Mô hình | Tốc độ học | Số vòng | C | Macro F1 | Recall spam |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,1 | 1000 | — | 41,25% | 0,00% |
| Logistic Regression | 0,1 | 3000 | — | 64,16% | 26,61% |
| Logistic Regression | 1,0 | 1000 | — | 94,80% | 87,98% |
| Logistic Regression | 1,0 | 3000 | — | 98,77% | 97,42% |
| Logistic Regression | 5,0 | 1000 | — | 99,08% | 98,28% |
| Logistic Regression | 5,0 | 3000 | — | 99,08% | 98,28% |
| Linear SVM | 0,01 | 1000 | 1 | 41,25% | 0,00% |
| Linear SVM | 0,05 | 3000 | 1 | 41,25% | 0,00% |
| Linear SVM | 0,05 | 1000 | 10 | 41,25% | 0,00% |
| Linear SVM | 0,1 | 3000 | 10 | 41,25% | 0,00% |
| Linear SVM | 0,01 | 1000 | 50 | 41,25% | 0,00% |
| Linear SVM | 0,05 | 3000 | 50 | 41,25% | 0,00% |
| Linear SVM | 0,01 | 3000 | 100 | 42,16% | 0,86% |

Ở tốc độ 5, Logistic Regression đạt cùng macro F1 với 1.000 và 3.000 vòng. Lưới SVM trên đây dùng ngưỡng 0. Sáu cấu hình đầu dự đoán toàn ham trên validation.

### 10 phương án trọng số lớp và ngưỡng SVM

| Nhóm | Trọng số lớp | Ngưỡng | AP | Macro F1 | Precision spam | Recall spam | F1 spam | FN | FP | Dự đoán spam |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Đều | 0 | 98,73% | 42,16% | 100,00% | 0,86% | 1,70% | 231 | 0 | 2 |
| B | Đều | −0,662012 | 98,73% | 98,33% | 96,64% | 98,71% | 97,66% | 3 | 8 | 238 |
| C | Balanced | 0 | 97,88% | 76,99% | 57,25% | 100,00% | 72,81% | 0 | 174 | 407 |
| D | Balanced | 0,529344 | 97,88% | 96,51% | 94,12% | 96,14% | 95,12% | 9 | 14 | 238 |
| C | Spam ×2 | 0 | 97,71% | 90,07% | 76,90% | 100,00% | 86,94% | 0 | 70 | 303 |
| D | Spam ×2 | 0,265490 | 97,71% | 96,67% | 93,78% | 97,00% | 95,36% | 7 | 15 | 241 |
| C | Spam ×3 | 0 | 98,56% | 75,54% | 55,61% | 100,00% | 71,47% | 0 | 186 | 419 |
| D | Spam ×3 | 0,700963 | 98,56% | 97,44% | 94,26% | 98,71% | 96,44% | 3 | 14 | 244 |
| C | Spam ×4 | 0 | 97,82% | 68,29% | 48,64% | 100,00% | 65,45% | 0 | 246 | 479 |
| D | Spam ×4 | 0,907420 | 97,82% | 96,99% | 93,47% | 98,28% | 95,82% | 4 | 16 | 245 |

A và B dùng cùng mô hình trọng số đều. Mỗi cặp C và D dùng cùng mô hình có trọng số lớp, lần lượt đánh giá tại ngưỡng 0 và ngưỡng chọn trên validation.

## Phụ lục C — Đầu ra và tái lập thực nghiệm

| Tệp | Cách sử dụng |
|---|---|
| `metrics.json` | Kiểm tra accuracy, macro F1 và số lỗi hai chiều của từng mô hình. |
| `classification_report.csv` | Xem precision, recall, F1 và số mẫu thật cho từng lớp. |
| `predictions.csv` | Ghép `email_id` với CSV test để xem từng dự đoán và truy vết lỗi. |
| `confusion_matrix.png` | Đọc số lượng theo dạng `[[TN, FP], [FN, TP]]`. |
| `comparison/baseline.csv` và `comparison/metrics.csv` | So sánh baseline mặc định với cấu hình cuối trên cùng tập test. |
| `comparison/selection.json` | Tham số chọn ở vòng validation ban đầu, kích thước chia tập và seed. |
| `comparison/svm_study/selection.json` | Trọng số lớp, ngưỡng chính xác và các chỉ số validation của SVM đã chọn. |
| `comparison/svm_study/*_pr.csv` | Các điểm precision–recall theo từng phương án trọng số. |

Accuracy là tỷ lệ nhãn đúng. Precision spam là tỷ lệ spam thật trong nhóm bị gán spam, còn recall spam là tỷ lệ spam thật được tìm thấy. F1 cân bằng precision và recall qua trung bình điều hòa. Macro F1 lấy trung bình F1 của ham và spam, nên cả hai lớp có trọng số bằng nhau. Với lớp không có dự đoán, báo cáo quy ước precision bằng 0.

### Phạm vi kiểm tra tái lập

Một lần chạy `python -m Lab01.src.train` sinh cả [baseline mặc định](../result/comparison/baseline.csv) và [kết quả hiện tại](../result/comparison/metrics.csv), không cần tham số riêng. Mốc SVM C=100, ngưỡng 0 trên test là số liệu từ vòng trước. Lệnh hiện tại vẫn ghi kết quả ngưỡng 0 **trên validation** trong [bảng thử SVM](../result/comparison/svm_study/validation.csv). Xem [README](../README.md) để biết vị trí dữ liệu, thứ tự chạy và đầu ra từng nhóm.

Lần kiểm tra trước đã tái tạo kết quả trên môi trường Windows có sẵn với Python 3.13.9. SHA-256 trùng cho bảng chỉ số, ba bảng dự đoán, nhãn test và tệp chọn ngưỡng SVM. Cài đặt từ môi trường ảo mới chưa được kiểm chứng.
