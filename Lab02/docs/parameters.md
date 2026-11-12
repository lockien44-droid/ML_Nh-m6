# Cấu hình Lab02

data.path là đường dẫn tương đối so với Lab02. data.encoding mặc định cp1252. top_products là số mã hàng được chọn bằng lượng bán trong train.

train_fraction và validation_fraction xác định mốc chia ngày. Phần thời gian còn lại làm test. seed cố định ngẫu nhiên trong các thuật toán cây.

Linear Ridge dùng alpha để kiểm soát chính quy hóa. Decision Tree dùng max_depth và min_samples_leaf. Random Forest thêm n_estimators và n_jobs. Gradient Boosting dùng n_estimators, max_depth và learning_rate.

Tuning tối thiểu hóa RMSE trên validation. Dự đoán nhận StockCode, date, lag_1, lag_7, mean_7, mean_28. Bốn thống kê lịch sử phải được tính từ dữ liệu trước ngày dự đoán.
