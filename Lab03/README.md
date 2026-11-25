# Lab03 — Phân cụm khách hàng và phân đoạn ảnh

## Thành viên nhóm

- Vũ Thị Thanh Thùy
- Nguyễn Văn Minh
- Lê Hồng Sơn
- Phan Đình Hưng
- Phạm Minh Tiến
- Võ Thị Anh Tú

## Hai phần thực hành

Phần 1 dùng Mall_Customers.csv để phân cụm 200 khách hàng theo Age, Annual Income (k$) và Spending Score (1-100). Gender dùng mô tả nhóm. K được chọn bằng Silhouette, inertia hỗ trợ Elbow và ARI kiểm tra độ ổn định theo seed.

Phần 2 nhận một ảnh, thu nhỏ giữ tỷ lệ và gom các pixel RGB thành K nhóm màu bằng cùng K-Means tự cài NumPy. Ảnh phân đoạn thay màu mỗi pixel bằng màu tâm cụm. Có thể thử nhiều K trên cùng ảnh.

## Cấu trúc

```text
Lab03/
├── config.yaml
├── algorithms/
│   └── kmeans.py
├── data/
│   ├── customers/
│   │   ├── raw/
│   │   └── processed/
│   └── images/raw/
├── src/
│   ├── config.py             Đường dẫn và cấu hình mặc định
│   ├── utils.py              JSON, hình ảnh, run_id và trạng thái
│   ├── part1_customers/
│   │   ├── eda.py
│   │   ├── preprocessing.py
│   │   ├── tuning.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── analysis.py
│   │   ├── visualization.py
│   │   └── pipeline.py
│   ├── part2_image/
│   │   ├── kmeans_scratch.py
│   │   ├── preprocessing.py
│   │   ├── segment.py
│   │   ├── evaluate.py
│   │   ├── visualization.py
│   │   └── pipeline.py
│   └── pipeline.py
├── runs/
│   ├── customers/<run_id>/
│   │   ├── config_used.json
│   │   ├── run.json
│   │   ├── preprocessing/
│   │   ├── models/             kmeans.joblib, scaler.joblib
│   │   ├── tuning/             metrics_by_k.csv, selection.json
│   │   ├── results/            customers_labeled.csv, cluster_profile.csv
│   │   ├── figures/
│   │   └── report.md
│   └── image/<run_id>/
│       ├── config_used.json
│       ├── run.json
│       ├── metrics.csv
│       ├── report.md
│       └── figures/            <ảnh>_k2_4_8_16.png, inertia_vs_k.png
├── notebooks/
│   ├── 01_customer_clustering.ipynb
│   └── 02_image_segmentation.ipynb
└── docs/
```

src chỉ chứa pipeline tổng, __init__.py và hai thư mục chức năng. Các file phân tích, huấn luyện, đánh giá khách hàng nằm trong part1_customers. Phần ảnh nằm trong part2_image và dùng chung thuật toán trong algorithms. Các lượt lịch sử trong results/runs giữ bố cục cũ để đối chiếu báo cáo.

## Cài đặt và chạy

Chạy tại thư mục gốc repository:

```powershell
python -m pip install -r Lab03/requirements.txt

# Phần 1, cũng là mặc định của lệnh cũ
python -m Lab03.src.pipeline
python -m Lab03.src.part1_customers.pipeline

# Phần 2, thay đường dẫn bằng ảnh thật
python -m Lab03.src.part2_image.pipeline --image "E:/du_lieu/landscape.jpg"

# Cả hai phần trong một lượt
python -m Lab03.src.pipeline --part all --image "E:/du_lieu/landscape.jpg"
```

Pipeline tổng nhận --part 1, --part 2 hoặc --part all. --config nhận một YAML theo schema config.yaml. Đường dẫn dữ liệu trong cấu hình tương đối với Lab03, còn --image tương đối với thư mục terminal hiện tại. Nếu bỏ --image, phần ảnh đọc image.path trong config.yaml qua image.paths. Đặt ảnh thật vào data/images/raw/input.jpg hoặc thay đường dẫn trước khi chạy phần 2.

## Cấu hình và kết quả

config.yaml có hai nhóm customers và image. Nhóm customers giữ dải K=2–10, n_init=20 và kiểm tra 10 seed. Kết quả kiểm chứng sau chuyển cấu trúc vẫn chọn K=6, Silhouette 0,4274, inertia 133,8683.

Nhóm image mặc định thử K=2,4,8,16, max_size=192, n_init=3 và colorspace=RGB. Các màu chuẩn hóa về [0,1] khi fit. rgb_mse và rgb_rmse báo sai số màu theo thang RGB 0–255. Chúng đo tái tạo màu, chưa đo độ đúng theo nhãn vật thể. Nếu K lớn hơn số màu khác nhau của ảnh sau resize, chương trình báo lỗi để chọn K phù hợp.

Mỗi phần lưu cấu hình thực tế config_used.json, hash nguồn và trạng thái trong run.json. Chạy --part all tạo hai lượt riêng, một dưới runs/customers và một dưới runs/image. Mô hình khách hàng lưu kmeans.joblib. scaler.joblib chứa danh sách đặc trưng cùng mảng mean/scale để tái tạo chuẩn hóa.

Phần ảnh chỉ sử dụng K-Means NumPy tự cài. Mỗi hình gồm ảnh gốc và kết quả K=2,4,8,16 trên một hàng; metrics.csv lưu inertia, RGB MSE/RMSE, số vòng và thời gian chạy.

## Tài liệu

- [Báo cáo khách hàng](docs/Lab03_Report.md)
- [Bài bổ sung phân đoạn ảnh](docs/Image_Segmentation_Report.md)
- [Tham số](docs/ReadmeForParameters.md)
- [Notebook khách hàng](notebooks/01_customer_clustering.ipynb)
- [Notebook ảnh](notebooks/02_image_segmentation.ipynb)
- [Dữ liệu khách hàng](data/customers/raw/Mall_Customers.csv)


## Chạy nhiều ảnh

```powershell
python -m Lab03.src.part2_image.pipeline --image "E:/du_lieu/photo_seascape.jpg" "E:/du_lieu/painting_beach_palm.jpg"
```

Phần ảnh chỉ sử dụng K-Means NumPy tự cài. Mỗi hình gồm ảnh gốc và kết quả K=2,4,8,16 trên một hàng; metrics.csv lưu inertia, RGB MSE/RMSE, số vòng và thời gian chạy.


## Chạy theo nhiệm vụ của phần khách hàng

```powershell
python -m Lab03.src.pipeline --part 1

# Đánh giá lại lượt đã có mô hình, thay RUN_ID bằng mã thật
python -m Lab03.src.part1_customers.evaluate --run-id RUN_ID
```

tuning.py, train.py và evaluate.py có điểm chạy riêng theo --run-id. Tuning cần dữ liệu đã chuẩn bị trong lượt. Train đọc tuning/selection.json để fit cấu hình đã chọn. Evaluate nạp mô hình và nhãn của dữ liệu đã phân cụm, kiểm tra hash đặc trưng và cấu hình, rồi xuất lại các chỉ số và hình. Đánh giá lại sử dụng cùng tập khách hàng đã phân cụm, chưa phải kiểm định trên khách hàng mới.

Muốn thay không gian tham số, chỉnh config.yaml rồi tạo lượt mới. Trường customers.kmeans.algorithms nhận kmeans, dbscan, hierarchical. Mặc định chỉ có kmeans để giữ đối chiếu thực nghiệm cũ. K-Means quét k_values; DBSCAN quét eps_values và min_samples_values; Hierarchical quét k_values và linkages. Tuning chọn Silhouette cao nhất trong các ứng viên có coverage ít nhất min_coverage, mặc định 0,8. Silhouette/DB/CH của DBSCAN loại noise và luôn báo thêm coverage, noise_count.

## Chức năng chung và K-Means ảnh

src/config.py đọc config.yaml và bổ sung các mặc định seed, K, eps và kích thước ảnh. src/utils.py tạo runs theo phần, ghi JSON nguyên tử, lưu hình và ghi trạng thái có khóa theo lượt.

Phần ảnh chỉ sử dụng K-Means NumPy tự cài. Mỗi hình gồm ảnh gốc và kết quả K=2,4,8,16 trên một hàng; metrics.csv lưu inertia, RGB MSE/RMSE, số vòng và thời gian chạy.

Phần ảnh hỗ trợ colorspace RGB, HSV hoặc LAB qua image.colorspace. OpenCV dùng để chuyển đổi màu, các kênh được đưa về [0,1] trước khi phân cụm. Với HSV, khoảng cách Euclid chưa biểu diễn đầy đủ tính vòng của hue. RGB MSE/RMSE được tính sau chuyển ảnh tái tạo về RGB, cùng thang 0–255.

Lab03 hiện chạy K-Means tự cài, không phụ thuộc scikit-learn. Silhouette, Davies–Bouldin và Calinski–Harabasz được tính bằng NumPy. Các tùy chọn DBSCAN/Hierarchical trước đây đã bỏ khỏi luồng chạy.

[Tài liệu Part 1 theo 5 bước K-Means](docs/Part1_KMeans_5_steps.md).
