# Cấu hình Lab02

data.path là đường dẫn tương đối so với Lab02. data.encoding mặc định cp1252. top_products là số mã hàng được chọn bằng lượng bán trong train.

train_fraction và validation_fraction xác định mốc chia ngày. Phần thời gian còn lại làm test. seed cố định ngẫu nhiên trong các thuật toán cây.

Linear Ridge dùng alpha để kiểm soát chính quy hóa. Decision Tree dùng max_depth và min_samples_leaf. Random Forest thêm n_estimators và n_jobs. Gradient Boosting dùng n_estimators, max_depth và learning_rate.

Tuning tối thiểu hóa RMSE trên validation. Dự đoán nhận StockCode, date, lag_1, lag_7, mean_7, mean_28. Bốn thống kê lịch sử phải được tính từ dữ liệu trước ngày dự đoán.


## So sánh đặc trưng và dự đoán từ lịch sử

Mỗi cấu hình thử hai bộ basic và trend trên cùng cửa sổ thời gian. Bộ trend thêm chênh lệch trung bình 7/28 ngày, chênh lệch lag_1 với mean_7 và chênh lệch lag_1 với lag_7. Các đặc trưng được tính trong bộ xử lý lưu cùng mô hình. Bộ basic vẫn là phương án đối chiếu khi chọn cấu hình.

Có thể dùng CSV doanh số theo ngày với cột StockCode,date,sales. Cần đủ 28 ngày liên tiếp trước ngày dự đoán, mỗi sản phẩm/ngày một dòng. Ngày có doanh số 0 phải ghi rõ. Chương trình bỏ qua các dòng từ ngày dự đoán trở đi.

```powershell
python -m Lab02.src.predict --history history.csv --product 23166 --date 2011-12-10
```

Cách nhập JSON cũ vẫn được hỗ trợ. Kết quả đánh giá hiện tại là dự báo từng ngày với lịch sử thực tế đã biết.
