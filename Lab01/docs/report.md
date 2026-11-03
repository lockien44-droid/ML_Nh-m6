# Lab01 — Phân loại email spam

## Mục tiêu và dữ liệu

Phân loại email thành `ham` (email thường, nhãn `0`) hoặc `spam` (nhãn `1`). Tệp [spam_ham_dataset.csv](../Data/spam_ham_dataset.csv) có 5.171 email. Quy trình dùng `text` làm đầu vào và `label_num` làm nhãn. CSV không có cột riêng cho người gửi hoặc người nhận.

## Tiền xử lý

Chạy từ thư mục gốc bằng `python -m Lab01.src.preprocessing`. Toàn bộ quy trình nằm trong [preprocessing.py](../src/preprocessing.py).

1. Đọc CSV và kiểm tra văn bản, nhãn.
2. Bỏ HTML, chuyển chữ thường, bỏ dấu câu và từ dừng tiếng Anh, rồi chuẩn hóa khoảng trắng.
3. Lọc văn bản rỗng sau khi làm sạch. Kiểm tra các email có cùng văn bản sạch nhưng khác nhãn, rồi loại trùng theo văn bản sạch trước khi chia tập.
4. Chia train/test theo tỷ lệ 80/20, giữ tỷ lệ nhãn và dùng `random_state=42`.
5. Học từ vựng và IDF trên train, rồi dùng cùng bộ TF-IDF để biến đổi test. Đặc trưng gồm từ đơn và cụm hai từ, giới hạn 50.000 cột.

| Tập | Ham | Spam | Tổng |
|---|---:|---:|---:|
| Train | 2.741 | 1.166 | 3.907 |
| Test | 686 | 291 | 977 |

Lần chạy hiện tại loại 287 email trùng sau làm sạch. Không có dòng không hợp lệ, văn bản sạch rỗng hoặc nhãn xung đột. Hai tập không có văn bản sạch trùng nhau.

Các CSV trung gian, ma trận CSR `.npz`, nhãn và danh sách đặc trưng được lưu trong `result/preprocessing/`. Tệp [summary.json](../result/preprocessing/summary.json) ghi số liệu của lần chạy. Bộ TF-IDF đã học được lưu tại `models/tfidf_vectorizer.joblib`.

## Đặc trưng đầu vào

Ba thuật toán dùng chung vector TF-IDF tạo từ cột `text` sau khi làm sạch. Mỗi cột của ma trận tương ứng với một từ hoặc cụm hai từ. Trọng số TF-IDF kết hợp tần suất xuất hiện của từ hoặc cụm từ trong email với độ hiếm của nó trong tập huấn luyện. Sau đó, vector của mỗi email được chuẩn hóa L2. Lần chạy hiện tại có 50.000 đặc trưng. Danh sách cột theo đúng thứ tự được lưu trong [feature_names.csv](../result/preprocessing/09_tfidf/feature_names.csv).

| Đặc trưng trong đề bài | Cách sử dụng hiện tại |
|---|---|
| Tần suất từ | Dùng làm thành phần của TF-IDF cho từ đơn và cụm hai từ. |
| Tiêu đề và thân email | Xử lý chung trong cột `text`, chưa tạo hai nhóm đặc trưng riêng. |
| Tần suất ký tự như `!` và `$` | Chưa trích xuất riêng. Bước làm sạch loại các ký tự này trước TF-IDF. |
| Người gửi và người nhận | CSV không có cột riêng. Quy trình hiện tại không trích xuất địa chỉ từ văn bản. |

`label_num` chỉ dùng làm nhãn. `email_id` dùng để đối chiếu các đầu ra. Cả hai đều nằm ngoài vector đặc trưng.

TF-IDF là bộ đặc trưng cơ sở để so sánh ba thuật toán trên cùng đầu vào. Nếu đề bài yêu cầu triển khai riêng tất cả các nhóm trong bảng, phiên bản hiện tại mới đáp ứng nhóm tần suất từ thông qua TF-IDF.

## Ba thuật toán

| Thuật toán | Cách triển khai |
|---|---|
| [Logistic Regression](../algorithms/logistic_regression.py) | Dùng sigmoid và gradient của binary cross-entropy để học trọng số. Đã kiểm tra huấn luyện và dự đoán trực tiếp trên CSR của Lab01. |
| [Linear SVM](../algorithms/svm.py) | Cập nhật trọng số và bias từ hinge loss có regularization L2. Dấu của điểm phân loại quyết định nhãn. Đã kiểm tra với TF-IDF mẫu dạng dense và CSR. |
| [Multinomial Naive Bayes](../algorithms/naive_bayes.py) | Cộng trọng số đặc trưng theo lớp, làm trơn bằng `alpha` và chọn lớp có điểm log cao hơn. Đã đối chiếu kết quả mẫu dạng dense và CSR với scikit-learn. |

## Tái hiện cấu hình mặc định

Chạy từ thư mục gốc với `python -m Lab01.src.train`. Lệnh tái hiện baseline trước khi chọn tham số. Tập test gồm 977 email, trong đó có 686 ham và 291 spam. Kết quả trùng với báo cáo trước.

| Mô hình | Cấu hình | Dự đoán ham / spam | Accuracy | Macro F1 | Bỏ lọt spam |
|---|---|---:|---:|---:|---:|
| Logistic Regression | learning rate 0,1, 1.000 vòng | 977 / 0 | 70,21% | 41,25% | 291 |
| Linear SVM | learning rate 0,01, 1.000 vòng, C=1 | 977 / 0 | 70,21% | 41,25% | 291 |
| Multinomial Naive Bayes | alpha=1 | 722 / 255 | 95,70% | 94,66% | 39 |

Accuracy 70,21% của hai mô hình đầu bằng đúng tỷ lệ ham. Recall spam bằng 0 vì cả 291 email spam đều bị gán ham. Macro F1 cho hai lớp trọng số bằng nhau. Precision của lớp không có dự đoán được quy ước bằng 0.

## Chẩn đoán

Nhãn 0 là ham và 1 là spam. Thứ tự nhãn và ID trong các CSV khớp, số hàng khớp với hai ma trận CSR có 50.000 cột. Kiểm tra sai phân hữu hạn trên CSR cho thấy dấu cập nhật trọng số và bias ở bước đầu khớp với hàm mục tiêu của cả hai thuật toán. Logistic Regression dự đoán spam từ xác suất 0,5 trở lên. SVM dự đoán spam từ điểm 0 trở lên, sau khi dùng nhãn −1/+1 để huấn luyện. Không cần đổi công thức huấn luyện.

Trace trên toàn bộ tập train ở các mốc 0, 1, 10, 100, 500 và 1.000 ghi bias, chuẩn trọng số, tỷ lệ dự đoán spam và các phân vị theo nhãn thật.

| Mô hình | Loss hoặc mục tiêu ở vòng 0 → 100 → 1.000 | Bias cuối | Chuẩn trọng số cuối | Trung vị ham / spam | Điểm cao nhất của spam | Spam dự đoán trên train ở vòng cuối |
|---|---|---:|---:|---|---:|---:|
| Logistic Regression | BCE 0,6931 → 0,5970 → 0,5143 | −0,6752 | 3,0961 | 0,2734 / 0,3684 | 0,4569 | 0% |
| Linear SVM | Mục tiêu 1,0000 → 0,8328 → 0,5964 | −0,9981 | 0,03375 | −1,0010 / −0,9963 | −0,9909 | 0% |

Logistic Regression đã học tín hiệu phân biệt. Trung vị xác suất spam thật cao hơn ham, nhưng xác suất lớn nhất vẫn dưới 0,5. BCE vẫn giảm từ 0,5558 ở vòng 500 xuống 0,5143 ở vòng 1.000. Kết quả khi tăng tốc độ học hoặc số vòng trên validation củng cố kết luận rằng cấu hình mặc định tiến quá chậm để vượt ngưỡng.

SVM tối ưu ½‖w‖² + C × hinge trung bình. Ở vòng cuối, hinge là 0,59585, regularization là 0,00057 và 46,89% mẫu còn margin dưới 1. Bias âm gần −1 và chuẩn trọng số nhỏ khiến cả hai nhóm nằm dưới 0. Mục tiêu đã thay đổi rất ít từ vòng 500 đến 1.000. Với các cấu hình đã thử, tăng số vòng đơn thuần chưa giải quyết được hiện tượng này.

## Chọn tham số trên validation

Tách phân tầng 20% từ 3.907 email train với seed 43. Train con có 3.125 email (2.192 ham, 933 spam). Validation có 782 email (549 ham, 233 spam). Bộ TF-IDF dùng cho thử nghiệm chỉ fit trên văn bản sạch của train con. Tập test gốc được giữ nguyên và chỉ dùng để báo cáo baseline và kết quả cuối.

- Logistic Regression thử tổ hợp tốc độ học 0,1 / 1 / 5 với 1.000 / 3.000 vòng. Tốc độ 5 cho macro F1 validation cao nhất là 99,08% ở cả hai số vòng. Chọn 1.000 vòng vì cho kết quả bằng 3.000 vòng. Recall spam validation là 98,28%. Ngưỡng giữ ở 0,5.
- SVM thử các bộ (tốc độ học, số vòng, C) gồm (0,01, 1.000, 1), (0,05, 3.000, 1), (0,05, 1.000, 10), (0,1, 3.000, 10), (0,01, 1.000, 50), (0,05, 3.000, 50), (0,01, 3.000, 100). Bộ cuối có macro F1 cao nhất, 42,16%, nhưng chỉ tìm được 2/233 spam. Sáu bộ còn lại đều dự đoán toàn ham.
- Naive Bayes giữ alpha=1. Toàn bộ kết quả thử và tiêu chí chọn được lưu trong [validation.csv](../result/comparison/validation.csv) và [selection.json](../result/comparison/selection.json).

## Kết quả vòng chọn tham số ban đầu

Sau khi chọn cấu hình, fit lại TF-IDF trên toàn bộ train gốc và huấn luyện lại cả ba mô hình. Trong ma trận nhầm lẫn, hàng là nhãn thật và cột là nhãn dự đoán, cùng theo thứ tự ham rồi spam.

| Mô hình | Accuracy | Macro F1 | Ma trận nhầm lẫn | Spam bỏ lọt trước → sau |
|---|---:|---:|---|---:|
| Logistic Regression | 99,18% | 99,02% | `[[681, 5], [3, 288]]` | 291 → 3 |
| Linear SVM | 70,32% | 41,62% | `[[686, 0], [290, 1]]` | 291 → 290 |
| Multinomial Naive Bayes | 95,70% | 94,66% | `[[683, 3], [39, 252]]` | 39 → 39 |

| Mô hình | Lớp | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Logistic Regression | ham | 99,56% | 99,27% | 99,42% | 686 |
| Logistic Regression | spam | 98,29% | 98,97% | 98,63% | 291 |
| Linear SVM | ham | 70,29% | 100,00% | 82,55% | 686 |
| Linear SVM | spam | 100,00% | 0,34% | 0,68% | 291 |
| Multinomial Naive Bayes | ham | 94,60% | 99,56% | 97,02% | 686 |
| Multinomial Naive Bayes | spam | 98,82% | 86,60% | 92,31% | 291 |

SVM chỉ gán nhãn spam cho một email, nên precision 100% đi kèm recall 0,34%. Kết quả dựa trên một lần chia dữ liệu cố định.

## Chạy lại và dự đoán

```powershell
python -m Lab01.src.train
python -m Lab01.src.predict --model logistic_regression --text "Please review the attached agenda for tomorrow team meeting"
python -m Lab01.src.predict --model logistic_regression --text "Congratulations! Claim your free cash prize now"
```

Có thể đổi `--model` thành `svm` hoặc `naive_bayes`. Lệnh dự đoán dùng lại hàm làm sạch và bộ TF-IDF đã lưu, rồi truyền ma trận thưa vào mô hình.

| Mô hình | Email họp | Email nhận thưởng |
|---|---|---|
| Logistic Regression | ham, xác suất spam 0,08875 | spam, xác suất spam 0,78093 |
| Linear SVM | ham, điểm −0,88302 | ham, điểm −0,58738 |
| Multinomial Naive Bayes | ham | spam |

Điểm SVM là decision score. Bảng ví dụ trên ghi kết quả của vòng chọn tham số ban đầu. Kết quả dự đoán SVM hiện tại nằm ở phần thí nghiệm bên dưới. Các mô hình và vectorizer nằm trong `models/`. Các thư mục theo thuật toán trong `result/` chứa ma trận nhầm lẫn PNG, chỉ số theo lớp và dự đoán gắn ID email. [Bảng baseline mặc định](../result/comparison/baseline.csv) và [bảng kết quả hiện tại](../result/comparison/metrics.csv) nằm cùng các trace trong `result/comparison/`. Báo cáo chi tiết baseline nằm trong thư mục con `baseline/`.

## Thí nghiệm SVM với ngưỡng và trọng số lớp

Cấu hình SVM ở vòng trước dùng learning rate 0,01, 3.000 vòng, `C=100`, trọng số đều và ngưỡng 0. Cấu hình này được chọn bằng macro F1 trên validation ở seed 43. Nó phát hiện 1/291 spam trên tập đối chiếu đã xem. Vòng này tái sử dụng đúng train con 3.125 email (2.192 ham, 933 spam) và validation 782 email (549 ham, 233 spam). TF-IDF chỉ fit trên train con và tất cả cấu hình SVM dùng chung hai ma trận CSR. Mốc AP tham chiếu là tỷ lệ spam trong validation, 233/782 = 29,80%.

Điểm quyết định là `X @ weights + bias`. Ví dụ, spam có điểm −0,3 vẫn xếp trên ham có điểm −0,8. Ở ngưỡng 0 cả hai được gán ham, còn ở ngưỡng −0,5 chỉ email đầu được gán spam. Quy tắc trong `predict()` là điểm lớn hơn hoặc bằng ngưỡng thì gán spam.

Hàm mục tiêu có trọng số là `½‖w‖² + C × mean(sample_weight × max(0, 1 − y_signed × score))`. Với chế độ `balanced`, trọng số lớp k là `n / (2 × count(k))`, tính từ nhãn của phần đang fit. Trong train con, trọng số xấp xỉ 0,713 cho ham và 1,675 cho spam. [LinearSVM](../algorithms/svm.py) nhân trọng số vào `active_labels` cho các mẫu có margin dưới 1, nên cả cập nhật trọng số mô hình và bias đều nhận hệ số này. Các mức ×2, ×3 và ×4 khảo sát quanh tỷ lệ ham/spam 2,35 với trọng số ham bằng 1. Các lượt thử giữ cùng learning rate, số vòng và C.

Với mỗi mô hình, thử các điểm validation, ngưỡng 0 và một ngưỡng ngay trên điểm lớn nhất. Chọn macro F1 cao nhất, khi hòa ưu tiên ít ham bị chặn nhầm hơn rồi ngưỡng gần 0 hơn. Khi macro F1 hòa giữa phương án có và không có trọng số, ưu tiên phương án không có trọng số. AP tính từ điểm liên tục và giữ nguyên khi đổi ngưỡng của cùng một mô hình.

| Lớp thật | Min | P10 | Median | P90 | Max | Điểm ≥ 0 |
|---|---:|---:|---:|---:|---:|---:|
| Ham | −1,22629 | −1,11343 | −0,97302 | −0,75939 | −0,35596 | 0,00% |
| Spam | −0,68437 | −0,61493 | −0,49246 | −0,30748 | 0,02295 | 0,86% |

SVM không trọng số đạt AP 98,73%, cao hơn tỷ lệ spam 29,80%. Trung vị điểm spam cao hơn ham nhưng chỉ hai mẫu spam vượt ngưỡng 0. Kết quả cho thấy ngưỡng 0 bỏ qua phần lớn tín hiệu xếp hạng đã học.

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

Ma trận validation theo thứ tự `[ham, spam]` là `[[549, 0], [231, 2]]` cho A và `[[541, 8], [3, 230]]` cho B. Chọn B vì có macro F1 cao nhất. B phát hiện thêm 228 spam và chặn nhầm thêm 8 ham so với A. Phương án có trọng số tốt nhất (spam ×3, ngưỡng đã chọn) bỏ lọt cùng 3 spam nhưng chặn nhầm 14 ham. AP của cả bốn cách đặt trọng số đều thấp hơn SVM không trọng số trong lượt thử này. Xem [bảng đầy đủ](../result/comparison/svm_study/validation.csv), [phân bố điểm](../result/comparison/svm_study/score_distribution.csv) và [đường precision–recall](../result/comparison/svm_study/unweighted_pr.csv).

Sau khi chốt B trên validation, fit lại TF-IDF và SVM trên 3.907 email train gốc, rồi áp dụng ngưỡng đã lưu `−0,6620115722742366`. Trên tập đối chiếu đã xem, ma trận chuyển từ `[[686, 0], [290, 1]]` sang `[[676, 10], [20, 271]]`. Accuracy tăng từ 70,32% lên 96,93%, macro F1 từ 41,62% lên 96,29% và recall spam từ 0,34% lên 93,13%. Có thêm 270 spam được phát hiện và 10 ham bị chặn nhầm. Fit lại có thể đổi thang điểm. Việc xác nhận độc lập ngưỡng đang chờ dữ liệu mới chưa được dùng để ra quyết định.

Tái hiện từ thư mục gốc:

```powershell
python -m Lab01.src.train
python -m Lab01.src.predict --model svm --text "Please review the attached agenda for tomorrow team meeting"
python -m Lab01.src.predict --model svm --text "Congratulations! Claim your free cash prize now"
```

Email họp được gán `ham` với decision score −0,88302. Email nhận thưởng được gán `spam` với decision score −0,58738. Cả hai dùng ngưỡng −0,6620115722742366. `predict.py` nạp `models/svm.joblib` cùng `models/tfidf_vectorizer.joblib`; ngưỡng nằm trong mô hình và [selection.json](../result/comparison/svm_study/selection.json). [Kết quả đối chiếu hiện tại](../result/comparison/metrics.csv) dùng cùng hàm `predict()`.
## Phân tích lỗi Logistic Regression

Trên tập đối chiếu đã xem, mô hình bỏ lọt **3 spam (FN)** và chặn nhầm **5 ham (FP)**. Ghép bảng dự đoán với văn bản test theo `email_id` theo quan hệ một-một. Cả 977 ID đều khớp và nhãn thật trùng `label_num`. Khi tính lại từ mô hình và vectorizer đã lưu, toàn bộ nhãn khớp bảng dự đoán. Điểm dưới đây là xác suất lớp spam từ sigmoid, với ngưỡng **0,5**. Cột cuối ghi đóng góp có dấu `x_j × w_j` vào điểm trước sigmoid. Dấu dương đẩy về spam và dấu âm đẩy về ham.

| ID | Thật | Dự đoán | Lỗi | Xác suất spam | Nội dung gốc và thay đổi sau làm sạch | Đóng góp và khoảng cách đến ngưỡng |
|---:|---|---|---|---:|---|---|
| 4447 | spam | ham | FN | 0,406211 | Quảng cáo vay dùng chữ chèn như `zfree` và `qquote`. Dấu chấm than và `%` mất, chữ chèn vẫn còn. | `flow` −0,449 và `subject new` −0,264. Dưới ngưỡng 0,094. |
| 534 | spam | ham | FN | 0,488380 | Thân thư gồm các trích dẫn rời, ít dấu hiệu quảng cáo rõ. Dấu phân cách mất, các từ nội dung vẫn còn. | `know` −0,281 và `need` −0,169, đối lại `world` +0,225. Sát ngưỡng, thấp hơn 0,012. |
| 1012 | spam | ham | FN | 0,443805 | Thư rất ngắn với tiêu đề và đoạn kết khó hiểu. Dấu ba chấm mất, `know` và `face` vẫn còn. | `know` −0,614 lớn hơn `face` +0,233. Dưới ngưỡng 0,056. |
| 26 | ham | spam | FP | 0,722264 | Xác nhận đăng ký dịch vụ, có nội dung miễn phí và nhiều liên kết. Dấu phân cách URL mất, `http`, `www` và `free` còn lại. | `http` +0,457 và `email` +0,353. Trên ngưỡng 0,222. |
| 3423 | ham | spam | FP | 0,502311 | Thư mời tham gia chương trình thưởng. Ký hiệu tiền và dấu chấm than mất, `free`, `click`, `earn` vẫn còn. | `free` +0,158 và `click` +0,072, đối lại `mail` −0,112. Sát ngưỡng, cao hơn 0,002. |
| 5025 | ham | spam | FP | 0,663749 | Thư mời ăn tối có hướng dẫn bật và tắt nhạc. Biểu tượng cảm xúc mất, `click`, `stop` và `dinner` vẫn còn. | `click` +0,382 và `stop` +0,292, đối lại `dinner` −0,085. Trên ngưỡng 0,164. |
| 3212 | ham | spam | FP | 0,666559 | Thư hỏi thăm có chân trang quảng bá dịch vụ email. Dấu phân cách URL mất, `http`, `www` và `free` vẫn còn. | `http` +0,544 và `free` +0,117. Trên ngưỡng 0,167. |
| 4288 | ham | spam | FP | 0,675421 | Chuyện cười chuyển tiếp, có dấu trích dẫn và dấu chấm than lặp. Các dấu mất, từ chỉ vai trò gia đình lặp lại vẫn còn. | `wife` +0,427 và `husband` +0,296. Trên ngưỡng 0,175. |

Hai FN là thư ngắn và một FN gồm các trích dẫn rời. Trong năm FP, bốn email có từ liên quan đến liên kết, thao tác nhấp hoặc nội dung miễn phí. Hai lỗi 534 và 3423 nằm cách ngưỡng dưới 0,012 điểm xác suất. Các đóng góp cho thấy đặc trưng đã đẩy điểm theo hướng nào. Việc chữ chèn làm yếu tín hiệu quảng cáo là giả thuyết từ nội dung, chưa có thí nghiệm thay thế từ để đo tác động.

## Kiểm tra tái tạo theo README

Từ thư mục gốc trên PowerShell, đã chạy `python -m pip install -r requirements.txt`, `python -m Lab01.src.preprocessing`, `python -m Lab01.src.train` và hai lệnh dự đoán Logistic Regression trong [hướng dẫn chạy nhanh](../README.md). Lần kiểm tra dùng môi trường Windows có sẵn với Python 3.13.9, pandas 2.3.3, scikit-learn 1.7.2, NumPy 2.4.6, SciPy 1.15.2, matplotlib 3.10.1 và joblib 1.5.2. Pip xác nhận đã có đủ thư viện. Việc cài đặt trong môi trường ảo mới chưa được kiểm tra.

SHA-256 trước và sau trùng nhau cho bảng chỉ số tổng hợp, ba bảng dự đoán, tệp nhãn test và tệp chọn ngưỡng SVM. Mỗi bảng dự đoán có 977 ID duy nhất. Ma trận tính lại từ các bảng đều có tổng 977 và cả ba PNG đọc được. Hàng là nhãn thật, cột là nhãn dự đoán, cùng thứ tự `[ham, spam]`.

| Mô hình | Accuracy | Macro F1 | Ma trận trước và sau |
|---|---:|---:|---|
| Logistic Regression | 99,18% | 99,02% | `[[681, 5], [3, 288]]` |
| Linear SVM | 96,93% | 96,29% | `[[676, 10], [20, 271]]` |
| Multinomial Naive Bayes | 95,70% | 94,66% | `[[683, 3], [39, 252]]` |

Logistic Regression vẫn dùng tốc độ học 5, 1.000 vòng và ngưỡng 0,5. Demo trả `ham` với xác suất spam 0,08875 cho email họp và `spam` với xác suất 0,78093 cho email nhận thưởng. Các liên kết tương đối trong README và báo cáo đều trỏ tới đầu ra hiện có. Repository chưa có bộ kiểm thử riêng được phát hiện. Lần chạy này kiểm tra khả năng tái tạo trên cùng tập đối chiếu và cùng môi trường.

