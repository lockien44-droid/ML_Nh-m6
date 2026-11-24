# Lab02 — Tham số và đặc trưng FMCG

## Cấu hình dữ liệu

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| data.path | data/raw/FMCG_2022_2024.csv | Nguồn CSV |
| data.encoding | utf-8 | Mã hóa |
| data.schema | fmcg_weekly | Tổng hợp doanh thu SKU/tuần |
| data.top_products | 20 | Số SKU tối đa, chọn theo Revenue train |
| data.min_active_weeks | 6 | Số tuần Revenue dương tối thiểu trong train |
| split.train_fraction | 0,6 | Mốc cuối train theo số tuần đầy đủ |
| split.validation_fraction | 0,2 | Khoảng validation sau train |
| seed | 42 | Nguồn ngẫu nhiên cho thuật toán cây |

Mục tiêu Revenue = tổng price_unit × units_sold theo sku/tuần, loại số lượng âm và giữ số lượng 0. Giữ tuần đủ từ thứ Hai đến Chủ nhật. Chỉ dùng lịch sử trước tuần dự báo.

## Bộ đặc trưng

| Tên nội bộ | Ý nghĩa theo tuần |
|---|---|
| StockCode | sku, được one-hot theo từ vựng học từ train |
| sales | Revenue, mục tiêu |
| lag_1, lag_7 | Revenue 1 và 7 tuần trước |
| mean_7, mean_28 | Trung bình 7 và 28 tuần trước |
| month, weekday, day_of_year | Lịch ngày thứ Hai bắt đầu tuần |
| trend_7_28 | mean_7 − mean_28 |
| recent_change | lag_1 − mean_7 |
| weekly_change | lag_1 − lag_7 |
| median_7, median_28 | Trung vị 7 và 28 tuần trước |
| std_7, std_28 | Độ lệch chuẩn 7 và 28 tuần trước, ddof=0 |
| zero_days_28 | Số tuần Revenue=0 trong 28 tuần trước |
| same_weekday_mean_4 | Trung bình các mốc 7,14,21,28 tuần trước |
| is_weekend | Cờ cuối tuần của ngày bắt đầu kỳ, luôn bằng 0 |

basic dùng SKU và bảy cột số lag/mean/lịch. trend thêm ba chênh lệch. robust thêm HISTORY và is_weekend. weekday và is_weekend là cột hằng trong dữ liệu tuần. Các tên kế thừa không ngụ ý độ trễ theo ngày.

Bộ xử lý học trung vị để điền thiếu, trung bình/độ lệch chuẩn để chuẩn hóa và từ vựng SKU từ train. Cột hằng có hệ số chia bằng 1. SKU lạ được one-hot bằng 0, nhưng CLI giới hạn SKU trong products.json.

## Thuật toán và lựa chọn

| Mô hình | Cấu hình được chọn |
|---|---|
| Ridge | alpha=10, learning_rate=0,1, n_iterations=3000, robust |
| Decision Tree | max_depth=10, min_samples_leaf=20, trend |
| Random Forest | n_estimators=80, max_depth=14, min_samples_leaf=10, robust |
| Gradient Boosting | n_estimators=100, max_depth=2, learning_rate=0,05, trend |

alpha kiểm soát regularization Ridge. max_depth giới hạn độ sâu cây, min_samples_leaf giới hạn số mẫu tại lá. n_estimators là số cây. learning_rate của boosting điều khiển phần đóng góp của từng cây.

Mỗi cấu hình dùng bộ xử lý fit riêng. selection_rmse là RMSE trung bình của ba cửa sổ mở rộng và validation cuối. Sau lựa chọn, fit lại trên train + validation. Gradient Boosting được chọn bằng validation, Random Forest có điểm test tốt nhất.

## Baseline và tên kế thừa

mean_baseline lấy trung bình train + validation. product_mean_baseline lấy trung bình theo SKU. rolling_7/28 dùng Revenue trung bình 7/28 tuần trước. last_week_baseline hiện dùng lag_7, nghĩa là 7 tuần trước. weekday_4_baseline lấy các mốc cách 7 tuần, chưa phải baseline mùa vụ năm. Cần đọc theo công thức thực tế khi so sánh.

## Đầu ra và lịch sử dự đoán

CSV dữ liệu tiền xử lý và đánh giá FMCG dùng sku,date,Revenue. Nội bộ mô hình ánh xạ về StockCode,date,sales. CLI nhận CSV lịch sử theo cả hai cách đặt tên để hỗ trợ các dữ liệu cũ.

Cần 28 tuần liên tiếp trước tuần dự báo, ghi rõ Revenue=0 cho tuần trống. Có thể truyền JSON với sku,date và các lag/mean. Bộ robust cần đầy đủ các thống kê HISTORY, nên ưu tiên --history.

[Hướng dẫn chạy](../README.md) · [Báo cáo chi tiết](Lab02_Report.md)
