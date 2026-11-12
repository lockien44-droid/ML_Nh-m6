# Lab02 — Dự đoán số lượng sản phẩm bán theo ngày

## Tổng quan

Triển khai bài Predicting Product Sales bằng bốn thuật toán tự cài NumPy. Tiền xử lý và ghép bộ xử lý với mô hình được tự cài bằng NumPy và pandas. Các chỉ số MAE, MSE, RMSE và R² cũng được tự tính bằng NumPy. Mỗi mẫu là một sản phẩm trong một ngày. Mục tiêu sales là tổng số đơn vị bán dương, loại giao dịch hủy và trả hàng. Đề gợi ý bốn họ thuật toán và cho phép lựa chọn. Phiên bản này dùng Ridge (hồi quy tuyến tính có chính quy hóa), Decision Tree, Random Forest và Gradient Boosting.

## Thành viên nhóm

- Nguyễn Thị Thanh Thùy
- Nguyễn Văn Minh
- Lê Hồng Sơn
- Phan Đình Hưng
- Phạm Minh Tiến
- Võ Thị Thanh Tú

## Cấu trúc thư mục

```text
Lab02/
├── algorithms/             # Mỗi file có một class thuật toán tự cài NumPy
├── data/
│   ├── raw/E-Commerc.csv   # 541.909 dòng giao dịch nguồn, mã hóa cp1252
│   └── preprocessing/      # Dành cho đầu ra tiền xử lý độc lập
├── src/
│   ├── config.yaml        # Nguồn dữ liệu, số sản phẩm, tỷ lệ chia, seed và cấu hình thử
│   ├── preprocessing.py   # Lọc giao dịch, tạo bảng ngày và đặc trưng lịch sử
│   ├── eda.py             # Thống kê theo sản phẩm và biểu đồ doanh số theo ngày
│   ├── tuning.py          # Chọn cấu hình từng thuật toán theo RMSE validation
│   ├── train.py           # Tạo pipeline tiền xử lý và estimator
│   ├── evaluate.py        # MAE, MSE, RMSE, R² và dự đoán từng mẫu
│   ├── run_report.py      # Mã lần chạy, hash dữ liệu, nhật ký và trạng thái
│   ├── report.py          # Báo cáo Markdown, scatter và histogram phần dư
│   ├── predict.py         # Demo dự đoán một sản phẩm/ngày từ JSON
│   └── pipeline.py        # Điều phối toàn bộ thực nghiệm
├── models/                # Bốn pipeline joblib, sản phẩm và lựa chọn theo validation
├── results/runs/          # Cấu hình, dữ liệu chia, nhật ký, số liệu và hình từng lần chạy
├── reports/               # Báo cáo Markdown từng lần chạy
├── docs/                  # Phương pháp, giả định và hướng dẫn đọc tham số
└── README.md              # Cách chạy và kết quả
```

## Luồng xử lý

1. Loại dòng trùng, ngày lỗi, giao dịch hủy, số lượng hoặc giá không dương. Chỉ giữ mã hàng dạng năm chữ số có thể kèm chữ cái.
2. Bỏ ngày đầu và cuối vì có thể chỉ ghi một phần ngày. Chọn 20 sản phẩm đứng đầu về số lượng bán trong giai đoạn train.
3. Tạo đủ ngày cho nhóm sản phẩm được chọn. Ngày thiếu giao dịch được xem là bán 0.
4. Tạo lag 1, lag 7, trung bình 7 và 28 ngày trước, cùng đặc trưng lịch. Bỏ 28 ngày đầu thiếu lịch sử.
5. Chia thời gian theo 60/20/20, rồi fit imputer, scaler và encoder trên train của từng lần thử.
6. Chọn cấu hình theo RMSE validation. Fit lại trên train và validation, đánh giá test, xuất báo cáo và lưu model.

## Cài đặt và chạy

Chạy từ thư mục gốc repository.

```powershell
python -m pip install -r requirements.txt
python -m Lab02.src.pipeline
```

Đầu ra được lưu theo mã lần chạy trong results/runs. Mô hình demo mặc định được chọn bằng validation, mọi mô hình đều có kết quả test.

## Kết quả và demo

Lần chạy tự cài 20261111T193054Z-f51c7f83.

| model | fit_seconds | mae | mse | rmse | r2 |
| --- | --- | --- | --- | --- | --- |
| linear_regression | 0.0131 | 87.6791 | 55371.9789 | 235.3125 | -0.0834 |
| decision_tree | 0.0844 | 77.0006 | 47906.8425 | 218.8763 | 0.0627 |
| random_forest | 6.2645 | 76.3146 | 46711.8420 | 216.1292 | 0.0861 |
| gradient_boosting | 1.2706 | 87.9482 | 50810.1504 | 225.4111 | 0.0059 |
| mean_baseline | 0.0000 | 95.7001 | 51121.7215 | 226.1011 | -0.0002 |
| last_week_baseline | 0.0000 | 96.2567 | 77078.8940 | 277.6309 | -0.5081 |

Xem bảng có cấu trúc tại evaluation/metrics.csv và hình tại figures trong thư mục lần chạy. Demo nhận mã sản phẩm đã được huấn luyện, ngày dự đoán và bốn thống kê lịch sử tính đến trước ngày đó.

```powershell
python -m Lab02.src.predict --input '{"StockCode":"23166","date":"2011-12-10","lag_1":10,"lag_7":15,"mean_7":12,"mean_28":11}'
```

Thay 23166 bằng mã có trong models/products.json. Có thể thêm --model random_forest hoặc tên mô hình khác trong config.yaml.

## Tài liệu chi tiết

- [Phương pháp và giới hạn](docs/Lab02_Report.md)
- [Tham số và đặc trưng](docs/parameters.md)

Bộ xử lý tự cài học trung vị, trung bình, độ lệch chuẩn và danh sách mã sản phẩm từ train. Cột số thiếu hoàn toàn được điền 0. Cột hằng có hệ số chia bằng 1. Mã sản phẩm lạ khi transform cho vector one-hot toàn 0.
