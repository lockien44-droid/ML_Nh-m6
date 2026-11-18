# Báo cáo Lab02 — Dự đoán số lượng sản phẩm bán theo ngày

Lần chạy thực nghiệm 2026-11-18_21-14-12. Tiền xử lý và lớp ghép mô hình được tự cài bằng NumPy và pandas. Các chỉ số MAE, MSE, RMSE và R² cũng được tự tính bằng NumPy.

## 1. Xác định bài toán và mục tiêu

Mục tiêu sales là tổng số lượng bán dương của một sản phẩm trong một ngày. Mỗi hàng sau tổng hợp là một cặp StockCode và date. Kết quả là số thực không âm, đơn vị sản phẩm/ngày. Mô hình được chọn theo selection_rmse, trung bình RMSE của ba cửa sổ mở rộng trên train và validation cuối. Dự đoán hỗ trợ ước lượng nhu cầu cho sản phẩm đã có lịch sử
## 2. Thu thập và hiểu dữ liệu

Nguồn là data/raw/E-Commerc.csv, mã hóa cp1252, gồm 541.909 dòng giao dịch.

| Cột | Ý nghĩa và cách dùng |
| --- | --- |
| InvoiceNo | Mã hóa đơn, chữ C đầu mã biểu thị giao dịch hủy |
| StockCode | Mã sản phẩm để nhóm giao dịch và tạo đặc trưng phân loại |
| Description | Mô tả hàng hóa, giữ trong nguồn, chưa dùng trong mô hình |
| Quantity | Số lượng giao dịch, dùng tổng hợp mục tiêu |
| InvoiceDate | Thời gian giao dịch, dùng tạo ngày và chia tập |
| UnitPrice | Giá đơn vị, dùng kiểm tra giao dịch hợp lệ |
| CustomerID | Mã khách hàng, có dữ liệu thiếu |
| Country | Quốc gia giao dịch |


## 3. Khảo sát và kiểm tra dữ liệu

Kiểm tra trùng dòng, ngày lỗi, số lượng và giá không hợp lệ. Lần chạy có 522504 dòng giao dịch sau lọc. EDA lưu thống kê theo sản phẩm và biểu đồ tổng số lượng từng ngày.

![Doanh số theo ngày](../results/runs/2026-11-18_21-14-12/figures/daily_sales.png)

## 4. Phân chia train, validation và test

Chia theo thứ tự ngày với cấu hình 60/20/20. Các sản phẩm trong cùng ngày thuộc cùng một tập. Chọn 20 sản phẩm có lượng bán lớn nhất trong giai đoạn train. Loại 28 ngày đầu thiếu lịch sử khiến số mẫu cuối lệch tỷ lệ danh nghĩa.

| split | rows |
| --- | --- |
| train | 3900 |
| validation | 1480 |
| test | 1500 |

Train kết thúc 2011-07-12. Validation kết thúc 2011-09-24. Test nằm sau mốc validation. Các tập CSV được lưu trong preprocessing của từng lần chạy.

## 5. Làm sạch và tiền xử lý

Loại bản trùng và giữ mã hàng dạng năm chữ số có thể kèm chữ cái. Loại giao dịch hủy, số lượng hoặc giá không dương, ngày không đọc được. Ngày đầu và ngày cuối bị loại vì có thể ghi nhận chưa đủ ngày. Giao dịch hợp lệ số lượng lớn được giữ lại vì có thể là đơn bán buôn. Đây là chính sách xử lý ngoại lệ, chưa winsorize doanh số. Tạo lưới đầy đủ sản phẩm/ngày và điền ngày không có giao dịch bằng 0. Giả định này chưa phân biệt hết hàng với thiếu nhu cầu.

## 6. Trích xuất và lựa chọn đặc trưng

Bộ basic có các đặc trưng số lag_1, lag_7, mean_7, mean_28, month, weekday và day_of_year. Lag lấy số lượng ngày trước. Trung bình trượt đều shift một ngày trước khi rolling nên không dùng sales của chính ngày cần dự đoán.

Bộ trend bổ sung trend_7_28 = mean_7 − mean_28, recent_change = lag_1 − mean_7 và weekly_change = lag_1 − lag_7. Các đặc trưng này chỉ dùng dữ liệu quá khứ.

StockCode được one-hot encoding. Imputer dùng trung vị và scaler học trên phần train tương ứng. Khi tuning, mọi bộ tiền xử lý chỉ fit trên train thử. Khi huấn luyện cuối, chúng được fit lại trên train cộng validation.

Test được đánh giá theo rolling one-step, dùng số bán thực tế của những ngày trước đó. Dự báo nhiều ngày liên tiếp sẽ cần cách cập nhật lịch sử riêng.

Bộ robust bổ sung trung vị, độ lệch chuẩn 7/28 ngày, số ngày bằng 0, trung bình cùng thứ bốn tuần và cuối tuần. Các thống kê chỉ dùng quá khứ.

## 7. Xây dựng mô hình mốc

Baseline mean dự đoán trung bình sales trên train cộng validation cho mọi mẫu test. Baseline last_week lấy lag_7 của từng sản phẩm. Baseline product_mean dự đoán trung bình theo từng sản phẩm trên train cộng validation. Thêm rolling_7, rolling_28 và weekday_4 từ lịch sử trước ngày mục tiêu. Sáu baseline dùng cùng tập test và chỉ số với bốn thuật toán.

| model | fit_seconds | mae | mse | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| linear_regression | 0.2675 | 76.3529 | 45557.5353 | 213.4421 | 0.1087 |
| decision_tree | 0.1081 | 77.5818 | 47927.7569 | 218.9241 | 0.0623 |
| random_forest | 8.3417 | 78.4745 | 46317.0759 | 215.2140 | 0.0938 |
| gradient_boosting | 1.7710 | 87.9482 | 50810.1504 | 225.4111 | 0.0059 |
| mean_baseline | 0.0000 | 95.7001 | 51121.7215 | 226.1011 | -0.0002 |
| last_week_baseline | 0.0000 | 96.2567 | 77078.8940 | 277.6309 | -0.5081 |
| rolling_7_baseline | 0.0000 | 89.0090 | 50427.5084 | 224.5607 | 0.0134 |
| rolling_28_baseline | 0.0000 | 82.1792 | 46341.1126 | 215.2699 | 0.0933 |
| weekday_4_baseline | 0.0000 | 85.3017 | 52048.3475 | 228.1411 | -0.0183 |
| product_mean_baseline | 0.0000 | 95.4789 | 52808.5168 | 229.8010 | -0.0332 |

## 8. Huấn luyện bốn thuật toán

### 8.1 Hồi quy tuyến tính Ridge

Class LinearRegression tự học bằng full-batch gradient descent. Khởi tạo trọng số và bias bằng 0. Mỗi vòng tính dự đoán Xw + b rồi lấy sai số e = dự đoán − y.

Hàm mục tiêu là (tổng bình phương sai số + alpha nhân tổng bình phương trọng số)/(2n). Gradient trọng số là (Xᵀe + alpha w)/n. Gradient bias là trung bình e. Cập nhật w bằng w − learning_rate nhân gradient trọng số và cập nhật b tương tự. Bias được miễn phạt. Số vòng lặp, learning rate và alpha khai báo trong YAML.

### 8.2 Decision Tree

Class DecisionTree tự xây cây đệ quy. Mỗi nút thử từng đặc trưng và tối đa 15 ngưỡng phân vị từ tập mẫu của nút. Chọn phép chia giảm tổng bình phương sai số nhiều nhất và đảm bảo số mẫu tối thiểu ở mỗi lá. Cây dừng theo max_depth, min_samples_leaf hoặc khi không còn cải thiện. Giá trị lá là trung bình mục tiêu.

Đây là cây dùng ứng viên ngưỡng phân vị để giới hạn chi phí tính toán. Kết quả có thể khác cây duyệt mọi điểm chia.

### 8.3 Random Forest

Class RandomForest tạo n_estimators cây. Mỗi cây học trên mẫu bootstrap có hoàn lại. Mỗi nút chọn ngẫu nhiên khoảng căn bậc hai số đặc trưng để tìm phép chia. Dự đoán là trung bình dự đoán các cây. Random seed được khai báo trong YAML.

### 8.4 Gradient Boosting

Class GradientBoosting khởi tạo bằng trung bình mục tiêu train. Mỗi vòng tính phần dư y − dự đoán hiện tại, học một cây nông để dự đoán phần dư, rồi cộng learning_rate nhân dự đoán của cây vào mô hình. Khi dự đoán, cộng lại toàn bộ cây theo đúng thứ tự.

Cả bốn thuật toán đều có fit(X, y) và predict(X). Model là alias trong từng module để giữ import hiện có.

## 9. So sánh cấu hình và tuning

Các ứng viên trong YAML so sánh basic, trend và robust; mỗi dòng là một cấu hình được chạy. Ba cửa sổ mở rộng dùng các khối ngày trên train, giữ mọi sản phẩm trong cùng ngày ở cùng một tập. Mỗi bộ tiền xử lý chỉ fit trên phần quá khứ của cửa sổ. Sau đó mỗi ứng viên được đánh giá trên validation cuối. selection_rmse là trung bình bằng trọng số của RMSE tại bốn giai đoạn. Mỗi họ thuật toán chọn ứng viên có selection_rmse thấp nhất, khi hòa giữ ứng viên đầu.

| model | parameters | mae | mse | rmse | r2 | selection_rmse | walk_forward_rmse_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_regression | {'alpha': 1.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'basic'} | 83.2180 | 34860.5312 | 186.7098 | -0.3385 | 372.1868 | 243.2016 |
| linear_regression | {'alpha': 1.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'trend'} | 83.2181 | 34860.6099 | 186.7100 | -0.3385 | 372.2190 | 243.2584 |
| linear_regression | {'alpha': 10.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'basic'} | 82.2357 | 34166.5627 | 184.8420 | -0.3119 | 316.4679 | 154.2479 |
| linear_regression | {'alpha': 10.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'trend'} | 82.2368 | 34167.2978 | 184.8440 | -0.3119 | 316.6194 | 154.5030 |
| linear_regression | {'alpha': 1.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'robust'} | 77.4598 | 32410.9339 | 180.0304 | -0.2444 | 365.6835 | 242.2041 |
| linear_regression | {'alpha': 10.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'robust'} | 76.3766 | 31762.9277 | 178.2216 | -0.2196 | 318.6611 | 171.0785 |
| linear_regression | {'alpha': 100.0, 'learning_rate': 0.1, 'n_iterations': 3000, 'feature_set': 'robust'} | 71.3946 | 29095.4855 | 170.5740 | -0.1171 | 232.7586 | 54.6310 |
| decision_tree | {'max_depth': 5, 'min_samples_leaf': 10, 'feature_set': 'basic'} | 68.7246 | 26221.5029 | 161.9305 | -0.0068 | 300.5313 | 221.2619 |
| decision_tree | {'max_depth': 5, 'min_samples_leaf': 10, 'feature_set': 'trend'} | 70.3721 | 25710.8340 | 160.3460 | 0.0128 | 300.3223 | 221.4524 |
| decision_tree | {'max_depth': 10, 'min_samples_leaf': 20, 'feature_set': 'basic'} | 69.7299 | 27077.6793 | 164.5530 | -0.0397 | 339.7603 | 291.0905 |
| decision_tree | {'max_depth': 10, 'min_samples_leaf': 20, 'feature_set': 'trend'} | 73.9568 | 27168.1592 | 164.8277 | -0.0431 | 340.5073 | 290.9251 |
| decision_tree | {'max_depth': 4, 'min_samples_leaf': 20, 'feature_set': 'trend'} | 69.5591 | 25056.5097 | 158.2925 | 0.0379 | 334.6278 | 292.3239 |
| decision_tree | {'max_depth': 4, 'min_samples_leaf': 20, 'feature_set': 'robust'} | 70.1623 | 25076.3709 | 158.3552 | 0.0372 | 337.5225 | 292.6232 |
| decision_tree | {'max_depth': 8, 'min_samples_leaf': 10, 'feature_set': 'trend'} | 73.6412 | 28867.4266 | 169.9042 | -0.1084 | 307.9640 | 220.7252 |
| decision_tree | {'max_depth': 8, 'min_samples_leaf': 10, 'feature_set': 'robust'} | 78.0278 | 29965.1076 | 173.1043 | -0.1505 | 308.7620 | 220.3973 |
| decision_tree | {'max_depth': 12, 'min_samples_leaf': 5, 'feature_set': 'trend'} | 85.6428 | 35616.9534 | 188.7245 | -0.3675 | 527.7478 | 600.7355 |
| decision_tree | {'max_depth': 12, 'min_samples_leaf': 5, 'feature_set': 'robust'} | 91.2283 | 39741.4187 | 199.3525 | -0.5259 | 531.4290 | 607.1370 |
| random_forest | {'n_estimators': 60, 'max_depth': 10, 'min_samples_leaf': 5, 'feature_set': 'basic'} | 70.4484 | 24833.6040 | 157.5868 | 0.0465 | 262.0461 | 141.7661 |
| random_forest | {'n_estimators': 60, 'max_depth': 10, 'min_samples_leaf': 5, 'feature_set': 'trend'} | 71.1534 | 24865.9357 | 157.6894 | 0.0452 | 217.1435 | 57.2539 |
| random_forest | {'n_estimators': 40, 'max_depth': 6, 'min_samples_leaf': 10, 'feature_set': 'trend'} | 71.2359 | 24720.5310 | 157.2276 | 0.0508 | 236.6064 | 86.5302 |
| random_forest | {'n_estimators': 40, 'max_depth': 6, 'min_samples_leaf': 10, 'feature_set': 'robust'} | 67.7605 | 24323.3091 | 155.9593 | 0.0661 | 283.9577 | 181.4093 |
| random_forest | {'n_estimators': 60, 'max_depth': 10, 'min_samples_leaf': 5, 'feature_set': 'robust'} | 70.1827 | 24991.0933 | 158.0857 | 0.0404 | 332.7542 | 267.7400 |
| random_forest | {'n_estimators': 80, 'max_depth': 14, 'min_samples_leaf': 10, 'feature_set': 'trend'} | 70.2387 | 24735.8505 | 157.2764 | 0.0502 | 237.6342 | 91.3649 |
| random_forest | {'n_estimators': 80, 'max_depth': 14, 'min_samples_leaf': 10, 'feature_set': 'robust'} | 69.2344 | 24671.2590 | 157.0709 | 0.0527 | 265.5779 | 144.7280 |
| gradient_boosting | {'n_estimators': 60, 'max_depth': 2, 'learning_rate': 0.05, 'feature_set': 'basic'} | 74.6473 | 26271.5073 | 162.0849 | -0.0087 | 346.3855 | 254.2840 |
| gradient_boosting | {'n_estimators': 60, 'max_depth': 2, 'learning_rate': 0.05, 'feature_set': 'trend'} | 74.6019 | 26252.2459 | 162.0254 | -0.0080 | 346.3706 | 254.2840 |
| gradient_boosting | {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05, 'feature_set': 'basic'} | 74.8542 | 27136.3293 | 164.7311 | -0.0419 | 366.3808 | 259.4874 |
| gradient_boosting | {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05, 'feature_set': 'trend'} | 75.2657 | 27492.5343 | 165.8087 | -0.0556 | 366.6506 | 259.4870 |
| gradient_boosting | {'n_estimators': 60, 'max_depth': 2, 'learning_rate': 0.05, 'feature_set': 'robust'} | 74.8039 | 26574.5479 | 163.0170 | -0.0204 | 367.5493 | 297.5806 |
| gradient_boosting | {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05, 'feature_set': 'robust'} | 72.3719 | 26837.1194 | 163.8204 | -0.0304 | 363.9557 | 261.3573 |
| gradient_boosting | {'n_estimators': 80, 'max_depth': 3, 'learning_rate': 0.05, 'feature_set': 'robust'} | 72.2290 | 25544.1587 | 159.8254 | 0.0192 | 646.2126 | 833.7195 |
| gradient_boosting | {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.1, 'feature_set': 'robust'} | 66.6809 | 27362.7603 | 165.4169 | -0.0506 | 404.9957 | 319.0951 |

Cấu hình chọn được lưu trong tuning/selection.json. Xem [điểm từng cửa sổ](../results/runs/2026-11-18_21-14-12/tuning/walk_forward_metrics.csv).

## 10. Cấu hình chọn và phân tích sai số

| model | alpha | learning_rate | n_iterations | feature_set | max_depth | min_samples_leaf | n_estimators |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_regression | 100.0000 | 0.1000 | 3000.0000 | robust | — | — | — |
| decision_tree | — | — | — | trend | 5.0000 | 10.0000 | — |
| random_forest | — | — | — | trend | 10.0000 | 5.0000 | 60.0000 |
| gradient_boosting | — | 0.0500 | — | trend | 2.0000 | — | 60.0000 |

Bảng validation ghi chỉ số từng ứng viên. Với test, mỗi mô hình có bảng 10 lỗi lớn nhất, bảng chỉ số theo StockCode và bảng theo weekday. Residual bằng thực tế trừ dự đoán. Residual dương là dự báo thiếu và âm là dự báo dư. Các bảng giúp tìm sản phẩm hoặc ngày có sai số lớn. Dự đoán validation của cấu hình thắng từng họ thuật toán được lưu trong tuning/*_validation.csv.

## 11. Chốt mô hình và đánh giá trên test

Fit lại mỗi cấu hình thắng trên train cộng validation. Test được biến đổi bằng bộ tiền xử lý vừa fit. Mô hình demo mặc định được chọn theo selection_rmse trước khi đánh giá test. So sánh các thuật toán trên test nhằm báo cáo hiệu năng.

| model | fit_seconds | mae | mse | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| linear_regression | 0.2675 | 76.3529 | 45557.5353 | 213.4421 | 0.1087 |
| decision_tree | 0.1081 | 77.5818 | 47927.7569 | 218.9241 | 0.0623 |
| random_forest | 8.3417 | 78.4745 | 46317.0759 | 215.2140 | 0.0938 |
| gradient_boosting | 1.7710 | 87.9482 | 50810.1504 | 225.4111 | 0.0059 |
| mean_baseline | 0.0000 | 95.7001 | 51121.7215 | 226.1011 | -0.0002 |
| last_week_baseline | 0.0000 | 96.2567 | 77078.8940 | 277.6309 | -0.5081 |
| rolling_7_baseline | 0.0000 | 89.0090 | 50427.5084 | 224.5607 | 0.0134 |
| rolling_28_baseline | 0.0000 | 82.1792 | 46341.1126 | 215.2699 | 0.0933 |
| weekday_4_baseline | 0.0000 | 85.3017 | 52048.3475 | 228.1411 | -0.0183 |
| product_mean_baseline | 0.0000 | 95.4789 | 52808.5168 | 229.8010 | -0.0332 |

MAE cùng đơn vị số lượng. MSE có đơn vị bình phương. RMSE là căn MSE. R² có thể âm khi sai số lớn hơn mốc dự đoán trung bình tập đánh giá. R² của các mô hình còn thấp, nên hiệu năng giải thích biến động doanh số còn hạn chế.

![linear_regression — dự đoán và phần dư](../results/runs/2026-11-18_21-14-12/figures/linear_regression.png)

![decision_tree — dự đoán và phần dư](../results/runs/2026-11-18_21-14-12/figures/decision_tree.png)

![random_forest — dự đoán và phần dư](../results/runs/2026-11-18_21-14-12/figures/random_forest.png)

![gradient_boosting — dự đoán và phần dư](../results/runs/2026-11-18_21-14-12/figures/gradient_boosting.png)

![mean_baseline — dự đoán và phần dư](../results/runs/2026-11-18_21-14-12/figures/mean_baseline.png)

![last_week_baseline — dự đoán và phần dư](../results/runs/2026-11-18_21-14-12/figures/last_week_baseline.png)

## 12. Lưu mô hình, demo và giám sát

Mỗi lần chạy lưu hash dữ liệu, YAML, tập chia, bảng tuning, cấu hình chọn, dự đoán test và hình. Joblib chứa pipeline tiền xử lý và thuật toán tự cài. models/selection.json ghi tên mô hình demo và mã lần chạy. predict.py nhận StockCode, date, lag_1, lag_7, mean_7 và mean_28. Các đặc trưng lịch được tạo từ date. Chỉ các sản phẩm có trong models/products.json được hỗ trợ.

Chạy toàn bộ bằng python -m Lab02.src.pipeline từ thư mục gốc repository. Dùng python -m Lab02.src.predict --input với JSON chứa các trường trên để demo.

Hệ thống hiện có dự đoán cục bộ. API production, nguồn lịch sử tự cập nhật, dự báo nhiều bước, giám sát drift và phản hồi tồn kho là các phần phát triển tiếp. Khi có nhãn thực tế sau triển khai, có thể theo dõi MAE và RMSE theo ngày, theo sản phẩm và so với baseline.

Bộ xử lý tự cài học trung vị, trung bình, độ lệch chuẩn và danh sách mã sản phẩm từ train. Cột số thiếu hoàn toàn được điền 0. Cột hằng có hệ số chia bằng 1. Mã sản phẩm lạ khi transform cho vector one-hot toàn 0.


## Kiểm chứng theo thời gian và phân tích lỗi

Pipeline bổ sung ba cửa sổ train mở rộng, giữ nguyên thứ tự ngày. Điểm chọn cấu hình là RMSE trung bình của ba cửa sổ và validation cuối. Test chỉ dùng đánh giá sau khi chọn. Bảng walk_forward_metrics.csv lưu từng cửa sổ. Mỗi lượt lưu đủ CSV train/validation/test. Baseline bổ sung trung bình theo sản phẩm. Bảng lỗi lớn nhất và chỉ số theo sản phẩm/ngày trong tuần giúp tìm nơi dự báo thiếu hoặc dư.


## Đối chiếu kết quả mới nhất

Random Forest với bộ trend đạt RMSE 215,2140 và R² 0,0938. So với lượt basic trước có RMSE 216,1292 và MAE 76,3146, RMSE giảm khoảng 0,42% nhưng MAE tăng lên 78,4745. Việc giảm lỗi bình phương chưa cải thiện sai số tuyệt đối trung bình. Mô hình được chọn theo các giai đoạn phát triển, không dựa trên điểm test. Test đã được xem qua nhiều vòng, nên cần giai đoạn dữ liệu mới để xác nhận.

CSV lịch sử có cột StockCode,date,sales có thể đưa vào predict.py qua --history, --product và --date. Chương trình yêu cầu đủ 28 ngày liên tiếp trước ngày dự đoán và bỏ qua các dòng từ ngày đó trở đi. Lịch sử dự đoán lưu tại reports/inference_history.csv. Báo cáo Markdown, HTML và PDF đã được tự động xuất. Dự báo nhiều bước và khoảng dự báo vẫn là hướng phát triển.


## Kết luận và giới hạn

Mô hình được chọn bằng dữ liệu phát triển là **random_forest**. RMSE test 215.2140, R² 0.0938. Mức giảm RMSE so với baseline trung bình là 4.82%.
R² thấp cho thấy mô hình còn hạn chế trong giải thích biến động doanh số. Không dùng bảng test để chọn lại cấu hình.
Top sản phẩm được cố định từ toàn bộ train trước walk-forward. Các fold kiểm chứng dự báo trong phạm vi sản phẩm này, chưa mô phỏng việc chọn lại danh mục tại từng mốc quá khứ.
Không có giao dịch được quy ước là 0, nhưng dữ liệu không phân biệt hết hàng, đóng cửa và không có nhu cầu. Hai ngày biên bị loại theo giả định dữ liệu có thể chưa đủ ngày.
Các đặc trưng robust gồm trung vị, độ lệch chuẩn 7/28 ngày, số ngày doanh số 0, trung bình cùng thứ bốn tuần và cuối tuần. Tất cả chỉ dùng lịch sử trước ngày dự báo.

## Các lỗi lớn nhất của mô hình được chọn

| Sản phẩm | Ngày | Thực tế | Dự đoán | Sai số tuyệt đối |
| --- | --- | --- | --- | --- |
| 84077 | 2011-10-27 | 4848.00 | 146.15 | 4701.85 |
| 22197 | 2011-12-08 | 3111.00 | 174.30 | 2936.70 |
| 22197 | 2011-10-28 | 2065.00 | 124.80 | 1940.20 |
| 22197 | 2011-11-07 | 2012.00 | 147.10 | 1864.90 |
| 22197 | 2011-11-14 | 1767.00 | 186.99 | 1580.01 |

Mô hình cải thiện RMSE so với baseline trung bình trên 17/20 sản phẩm. Xem ../results/runs/2026-11-18_21-14-12/evaluation/product_baseline_comparison.csv để đối chiếu từng sản phẩm với các baseline lịch sử.

Baseline trung bình 28 ngày đạt RMSE 215,2699, gần bằng Random Forest 215,2140. Ridge với alpha=100 và bộ robust đạt RMSE 213,4421, giảm 9,29% so với cấu hình Ridge cũ. Random Forest vẫn là lựa chọn theo dữ liệu phát triển; không đổi mô hình mặc định dựa vào test. So sánh trước/sau trên cùng test chỉ mang tính mô tả, cần dữ liệu tương lai độc lập để xác nhận mức cải thiện.
