# Lab03 — Customer Clustering

## Tổng quan

K-Means và Silhouette tự cài bằng NumPy. Ba đặc trưng gồm tuổi, thu nhập năm và điểm chi tiêu. Bộ CSV hiện có 200 khách hàng. Nguồn công bố/giấy phép cần bổ sung vì file chưa kèm thông tin đó.

## Cấu trúc

```text
Lab03/
├── algorithms/kmeans.py             # Gán cụm, cập nhật tâm, hội tụ, nhiều khởi tạo
├── Data/raw/Mall_Customers.csv      # CSV gốc
├── Data/preprocessing/              # CSV chuẩn hóa và tham số scaling mới nhất
├── src/
│   ├── config.yaml                 # Đặc trưng, K và tham số
│   ├── preprocessing.py            # Kiểm tra dữ liệu và chuẩn hóa
│   ├── eda.py                      # Thiếu, ngoại lệ, phân bố
│   ├── train.py                    # Huấn luyện một K
│   ├── tuning.py                   # So sánh K theo Silhouette
│   ├── evaluate.py                 # Silhouette và profile
│   ├── visualization.py            # Biểu đồ K và Plotly 3D
│   └── pipeline.py                 # Thực nghiệm và lưu kết quả
├── notebooks/01_customer_clustering.ipynb
├── results/runs/<run_id>/           # Kết quả, cấu hình, mô hình và hình từng lần chạy
├── docs/
│   ├── Lab03_Report.md             # Phương pháp và kết quả
│   ├── ReadmeForParameters.md      # Giải thích tham số
│   └── assignments_Lab03/          # Bản đề bài theo cấu trúc yêu cầu
└── README.md
```

## Cài đặt và chạy

Chạy từ thư mục gốc repository.

```powershell
python -m pip install -r requirements.txt
python -m Lab03.src.pipeline
```

Lệnh in thư mục kết quả. Mở figures/customer_clusters_3d.html trong thư mục đó bằng trình duyệt, dùng chuột để xoay, cuộn để thu phóng và hover xem khách hàng. HTML nhúng thư viện nên có thể mở ngoại tuyến.

## Kết quả

Lần chạy 20261118T074359Z-36241066 chọn K=6, Silhouette 0,4274. [Báo cáo và phân khúc](docs/Lab03_Report.md) có bảng profile và hình. Notebook gọi lại các module trong src và class KMeans.

Bản đề gốc hiện có được sao chép thành docs/assignments_Lab03/Lab03_Assignment(2).pdf. Phần bổ sung phân đoạn ảnh trong PDF nằm ngoài cấu trúc phân cụm khách hàng được yêu cầu.
