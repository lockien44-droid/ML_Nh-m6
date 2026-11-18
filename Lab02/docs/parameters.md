# Cấu hình Lab02

data.path là đường dẫn tương đối so với Lab02. data.encoding mặc định cp1252. top_products là số mã hàng được chọn bằng lượng bán trong train.

train_fraction và validation_fraction xác định mốc chia ngày. Phần thời gian còn lại làm test. seed cố định ngẫu nhiên trong các thuật toán cây.

Linear Ridge dùng alpha để kiểm soát chính quy hóa. Decision Tree dùng max_depth và min_samples_leaf. Random Forest thêm n_estimators. Mã tự cài hiện không có tham số n_jobs. Gradient Boosting dùng n_estimators, max_depth và learning_rate.

Tuning tối thiểu hóa selection_rmse, trung bình RMSE trên ba cửa sổ mở rộng của train và validation cuối. Dự đoán nhận StockCode, date, lag_1, lag_7, mean_7, mean_28. Bốn thống kê lịch sử phải được tính từ dữ liệu trước ngày dự đoán.


## So sánh đặc trưng và dự đoán từ lịch sử

Các ứng viên so sánh ba bộ basic, trend và robust trên cùng cửa sổ thời gian. Bộ trend thêm chênh lệch trung bình 7/28 ngày, chênh lệch lag_1 với mean_7 và chênh lệch lag_1 với lag_7. Các đặc trưng được tính trong bộ xử lý lưu cùng mô hình. Bộ basic vẫn là phương án đối chiếu khi chọn cấu hình.

Có thể dùng CSV doanh số theo ngày với cột StockCode,date,sales. Cần đủ 28 ngày liên tiếp trước ngày dự đoán, mỗi sản phẩm/ngày một dòng. Ngày có doanh số 0 phải ghi rõ. Chương trình bỏ qua các dòng từ ngày dự đoán trở đi.

```powershell
python -m Lab02.src.predict --history history.csv --product 23166 --date 2011-12-10
```

Cách nhập JSON cũ được hỗ trợ cho basic/trend. Với robust, JSON cần thêm các thống kê HISTORY hoặc dùng --history để tự tính đầy đủ. Kết quả đánh giá hiện tại là dự báo từng ngày với lịch sử thực tế đã biết.


## Cấu hình đang triển khai

Lượt chạy 2026-11-18_21-14-12.

| model | alpha | learning_rate | n_iterations | feature_set | max_depth | min_samples_leaf | n_estimators |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_regression | 100.0000 | 0.1000 | 3000.0000 | robust | — | — | — |
| decision_tree | — | — | — | trend | 5.0000 | 10.0000 | — |
| random_forest | — | — | — | trend | 10.0000 | 5.0000 | 60.0000 |
| gradient_boosting | — | 0.0500 | — | trend | 2.0000 | — | 60.0000 |


## Đặc trưng robust và phạm vi kiểm chứng

Bộ robust giữ bộ trend và thêm median_7, median_28, std_7, std_28 (độ lệch chuẩn tổng thể, ddof=0), zero_days_28, same_weekday_mean_4 và is_weekend. Trung bình cùng thứ lấy các ngày t−7, t−14, t−21, t−28. Tất cả thống kê loại ngày mục tiêu.

YAML khai báo danh sách ứng viên, mỗi phần tử là một cấu hình thực sự được chạy. selection_rmse cho bốn giai đoạn trọng số bằng nhau, không phải trung bình theo số dòng. walk_forward_rmse_std biểu diễn độ dao động giữa ba fold, không phải khoảng tin cậy.

Danh mục top sản phẩm được cố định từ toàn bộ train. Kiểm chứng walk-forward áp dụng trong danh mục cố định này, chưa đánh giá quy trình chọn lại danh mục tại mỗi mốc quá khứ. Không sử dụng test để chọn đặc trưng, siêu tham số hoặc mô hình thắng.

Thêm baseline trung bình 7 ngày, 28 ngày và trung bình cùng thứ bốn tuần. evaluation/product_baseline_comparison.csv đối chiếu RMSE từng sản phẩm và phần trăm cải thiện. Giá trị âm nghĩa là mô hình kém baseline.
