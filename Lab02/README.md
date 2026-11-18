# Lab02 — Dự đoán số lượng bán sản phẩm

## Tổng quan

Dự đoán số lượng bán dương theo sản phẩm/ngày từ lịch sử giao dịch E-Commerc.csv.
Ridge, Decision Tree, Random Forest, Gradient Boosting cùng tiền xử lý và các chỉ số
được tự cài bằng NumPy/pandas.

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
├── algorithms/                         # Mỗi thuật toán một file và class tự cài
├── data/
│   ├── raw/E-Commerc.csv               # Dữ liệu giao dịch gốc
│   └── preprocessing/                  # Bản train/validation/test và summary mới nhất
├── src/
│   ├── config.yaml                     # Dữ liệu, chia thời gian, bộ đặc trưng, siêu tham số
│   ├── preprocessing.py                # Làm sạch, tổng hợp ngày, lag/rolling và chuẩn hóa
│   ├── train.py                        # Tạo mô hình và hàm train_model dùng chung
│   ├── tuning.py                       # Thử cấu hình trên walk-forward và validation
│   ├── evaluate.py                     # MAE, MSE, RMSE, R² và bảng phân tích lỗi
│   ├── eda.py                          # Khảo sát doanh số theo ngày và sản phẩm
│   ├── report.py                       # Hình dự đoán/phần dư và báo cáo Markdown
│   ├── predict.py                      # Dự đoán một mẫu từ JSON hoặc lịch sử
│   ├── experiment_tracking.py          # Mã lần chạy, hash, thời gian, trạng thái và sự kiện
│   └── pipeline.py                     # Điều phối từ dữ liệu gốc đến mô hình và báo cáo
├── notebooks/
│   └── Lab02_Product_Sales_Prediction.ipynb  # Đọc và phân tích một lần chạy hoàn tất
├── models/                             # Phiên bản đang dùng cho demo
├── results/runs/<run_id>/
│   ├── run.json                        # Trạng thái, hash dữ liệu và thời lượng
│   ├── config.yaml                     # Snapshot cấu hình
│   ├── events.jsonl                    # Sự kiện từng giai đoạn
│   ├── preprocessing/                  # Train, validation, test và summary
│   ├── models/                         # Bốn pipeline đã huấn luyện
│   ├── tuning/
│   │   ├── validation_metrics.csv      # Chỉ số ứng viên và selection_rmse
│   │   ├── walk_forward_metrics.csv    # Điểm từng cửa sổ quá khứ
│   │   └── selection.json              # Cấu hình được chọn cho từng thuật toán
│   ├── evaluation/                     # Chỉ số test, dự đoán, lỗi lớn, lỗi theo nhóm
│   ├── figures/                        # EDA và biểu đồ đánh giá
│   └── report.md                       # Báo cáo của chính lần chạy
├── reports/                            # Báo cáo từng lần chạy, lịch sử dự đoán
├── docs/                               # Báo cáo 12 bước và giải thích tham số
└── README.md
```

## Luồng xử lý

1. Làm sạch giao dịch và chọn 20 sản phẩm theo số lượng bán trong giai đoạn train.
2. Tổng hợp ngày, tạo lag/rolling từ các ngày trước và chia train/validation/test theo thời gian.
3. So sánh basic và trend với lưới tham số. Mỗi lần thử fit riêng bộ tiền xử lý từ phần train tương ứng.
4. Chọn theo selection_rmse, trung bình RMSE của ba cửa sổ mở rộng và validation cuối.
5. Huấn luyện lại trên train cộng validation, đánh giá test và lưu mô hình, số liệu, báo cáo.

Bộ sản phẩm được chọn từ train tổng trước các cửa sổ. Điểm walk-forward đánh giá trên
danh mục đã chốt, chưa mô phỏng việc lựa chọn lại danh mục theo từng mốc.

## Cài đặt và chạy

Chạy tại thư mục gốc repository.

```powershell
python -m pip install -r requirements.txt
python -m Lab02.src.pipeline
```

Notebook đọc run_id đang công bố trong models/selection.json. Chạy các ô từ trên xuống.
Notebook đọc cấu hình và mô hình từ cùng một lần chạy và không ghi vào thư mục đó.
Mở bằng VS Code/Jupyter với kernel đã cài các thư viện của dự án.

## Kết quả và demo

Lần chạy tham chiếu **2026-11-18_21-14-12**, giờ Việt Nam. Mô hình demo được chọn theo dữ liệu phát triển là Random Forest.

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

```powershell
python -m Lab02.src.predict --model random_forest --input '{"StockCode":"84077","date":"2011-12-10","lag_1":10,"lag_7":15,"mean_7":12,"mean_28":11}'
```

Các thống kê trong ví dụ là minh họa. Có thể dùng --history với CSV chứa 28 ngày
liên tiếp trước ngày dự đoán, kèm --product và --date để tính đặc trưng từ lịch sử.

Giả định ngày không có giao dịch là 0, chưa phân biệt hết hàng hoặc ngày nghỉ.
Phạm vi chỉ gồm 20 sản phẩm đã chọn. Đây là dự báo từng ngày với lịch sử thực tế
của ngày trước, chưa phải dự báo nhiều ngày liên tiếp không có dữ liệu cập nhật.

## Tài liệu chi tiết

- [Báo cáo theo 12 bước](docs/Lab02_Report.md)
- [Tham số và đặc trưng](docs/parameters.md)
- [Notebook](notebooks/Lab02_Product_Sales_Prediction.ipynb)
- [Báo cáo lần chạy tham chiếu](results/runs/2026-11-18_21-14-12/report.md)


## Báo cáo tự động

    reports/
    ├── report_<run_id>.md       # Báo cáo Markdown hiện hành
    ├── report_<run_id>.html     # Trang kết quả và hình
    ├── report_<run_id>.pdf      # Bản PDF để trình bày
    ├── figures/<run_id>/
    │   ├── eda/
    │   ├── training/
    │   └── evaluation/
    ├── metrics_summary.csv     # Kết quả test theo Run_ID
    ├── validation_summary.csv  # Điểm ứng viên theo Run_ID
    ├── inference_history.csv   # Lịch sử dự đoán sản phẩm qua CLI
    └── archive/                # Báo cáo của các lượt trước

Pipeline tự xuất ba định dạng. Các báo cáo cũ chuyển vào archive, hình giữ theo Run_ID.
inference_history.csv ở Lab02 chỉ ghi dự đoán sản phẩm. Để tạo lại báo cáo mà không
huấn luyện lại, chạy từ thư mục gốc:

    python -m Lab02.src.report --run-id 2026-11-18_21-14-12

Lệnh này cập nhật báo cáo hiện hành cho lần chạy được chọn và cập nhật bảng lịch sử
theo Run_ID, không nhân đôi các dòng của cùng một lần chạy.


Bộ đặc trưng robust thêm thống kê phân bố và mùa vụ từ 28 ngày quá khứ. Khi mô hình chọn robust, nên dùng --history để tự tính đầy đủ đặc trưng. Bảng so sánh từng sản phẩm nằm trong evaluation/product_baseline_comparison.csv của lần chạy.


Các lần chạy được lưu cục bộ và bỏ qua trong Git. Repository giữ một lần chạy tham chiếu đầy đủ để đối chiếu báo cáo. Tham chiếu hiện tại là `2026-11-18_21-14-12`. Chạy pipeline sẽ tạo thư mục mới, không ghi đè thư mục tham chiếu.
