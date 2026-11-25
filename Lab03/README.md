# Lab03 — Phân cụm khách hàng và phân đoạn ảnh

## Thành viên nhóm

- Vũ Thị Thanh Thùy
- Nguyễn Văn Minh
- Lê Hồng Sơn
- Phan Đình Hưng
- Phạm Minh Tiến
- Võ Thị Anh Tú

## Mục tiêu

- **Part 1:** phân cụm 200 khách hàng của Mall_Customers.csv dựa trên sự tương đồng, xác định và mô tả các phân khúc khác biệt.
- **Part 2:** phân đoạn ảnh bằng cách gom các pixel có màu gần nhau và thay mỗi pixel bằng màu tâm cụm. Đây là phân cụm màu, không phải nhận diện vật thể.

Cả hai phần dùng K-Means NumPy tự cài trong [algorithms/kmeans.py](algorithms/kmeans.py). Không phụ thuộc scikit-learn. OpenCV dùng chuyển đổi không gian màu.

## Cài đặt và chạy

Chạy từ thư mục gốc repository, chọn môi trường Python đã cài các thư viện:

```powershell
python -m pip install -r Lab03/requirements.txt

# Part 1 — mặc định nếu không có --part
python -m Lab03.src.pipeline
python -m Lab03.src.pipeline --part 1

# Part 2 — hai ảnh có sẵn trong config.yaml
python -m Lab03.src.pipeline --part 2

# Chạy cả hai phần, tạo hai run riêng
python -m Lab03.src.pipeline --part all

# Dùng ảnh khác hoặc nhiều ảnh
python -m Lab03.src.pipeline --part 2 --image "E:/du_lieu/anh1.jpg" "E:/du_lieu/anh2.jpg"

# Chỉ vẽ EDA khách hàng, không tuning/huấn luyện
python -m Lab03.src.part1_customers.eda

# Đánh giá lại mô hình khách hàng đã lưu; thay RUN_ID bằng mã thực tế
python -m Lab03.src.part1_customers.evaluate --run-id RUN_ID
```

`--config` nhận đường dẫn YAML theo cấu trúc [config.yaml](config.yaml). Đường dẫn trong cấu hình tương đối với `Lab03/`; đường dẫn `--image` tương đối với terminal hiện tại. Khi không truyền `--image`, đọc `image.paths`, mặc định là `draw_beach.jpg` và `photo_flower.jpg` trong `data/images/raw/`.

## Cấu trúc thư mục

```text
Lab03/
├── README.md
├── requirements.txt
├── config.yaml
├── algorithms/kmeans.py       Class KMeans tự cài bằng NumPy
├── data/
│   ├── customers/raw/         CSV gốc
│   ├── customers/processed/   Dữ liệu khách hàng đã xử lý
│   └── images/raw/            Ảnh đầu vào
├── src/
│   ├── __init__.py
│   ├── config.py              Đọc cấu hình, đường dẫn, mặc định
│   ├── utils.py               Tạo run, JSON, lưu hình, trạng thái và log
│   ├── pipeline.py            Điều phối --part 1 | 2 | all
│   ├── part1_customers/       Khảo sát, phân cụm và đánh giá khách hàng
│   └── part2_image/           Xử lý pixel, phân đoạn và đánh giá ảnh
├── runs/
│   ├── customers/<run_id>/
│   └── image/<run_id>/
├── notebooks/                Demo và giải thích từng bước
└── docs/
    ├── Lab03_Report.md
    ├── Part1_KMeans_5_steps.md
    └── images/
        ├── eda/tables/         Hình EDA và CSV thống kê riêng
        ├── preprocessing/
        ├── tuning/
        └── evaluation/
```

## Thuật toán dùng chung: K-Means

1. Chọn K; một lần `fit` giữ K cố định.
2. Khởi tạo K tâm bằng k-means++.
3. Gán mỗi mẫu vào tâm gần nhất theo khoảng cách Euclid bình phương.
4. Cập nhật mỗi tâm bằng trung bình các vector thành viên; xử lý cụm rỗng.
5. Lặp gán/cập nhật đến khi `shift <= tol` hoặc đạt `max_iter`.

`shift = sqrt(sum((updated - centers)**2))` là độ dịch chuyển tổng thể của tất cả tâm. Dừng vì đạt max_iter chưa chắc hội tụ; xem `converged_`. Mỗi K thử `n_init` lần khởi tạo và giữ nghiệm có inertia nhỏ nhất. `fit` học tâm, `predict` gán mẫu vào tâm đã học, không huấn luyện lại.

## Part 1 — Phân cụm khách hàng

### Dữ liệu, chuẩn hóa và chọn K

Mỗi khách hàng có ba đặc trưng dùng cho khoảng cách: `Age`, `Annual Income (k$)`, `Spending Score (1-100)`. `CustomerID` dùng đối chiếu; `Gender` dùng EDA và mô tả cơ cấu từng cụm.

Standardize bằng z-score từng cột: `(x - mean) / std`, `ddof=0`. Trung bình sau chuẩn hóa gần 0, độ lệch chuẩn gần 1; không giới hạn trong [0,1]. Phân cụm mô tả toàn bộ khách hàng, không đánh giá dự báo trên tập test.

Mặc định thử K=2…10, n_init=20, max_iter=300, tol=0.0001, random_state=42. Tuning tự chọn K theo Silhouette cao nhất trong các cấu hình hợp lệ đáp ứng coverage. Inertia phục vụ quan sát Elbow; Elbow chưa tự chọn K. Davies–Bouldin (thấp hơn tốt hơn) và Calinski–Harabasz (cao hơn tốt hơn) là đánh giá bổ sung. Không diễn giải Silhouette thành phần trăm chính xác.

Lần chạy tham chiếu chọn K=6, Silhouette khoảng 0.4274 và inertia khoảng 133.8683; đây là kết quả thực nghiệm, không phải giá trị bị gán sẵn. ARI dùng so sánh phân cụm qua nhiều seed sau khi chọn K. Mã hiện chỉ hỗ trợ thuật toán `kmeans`; các trường eps/linkage còn trong cấu hình không kích hoạt DBSCAN/Hierarchical.

### Chức năng các file Part 1

| File trong src/part1_customers | Chức năng |
|---|---|
| `eda.py` | Thống kê thiếu dữ liệu/ngoại lệ, phân phối theo giới tính, quan hệ và tương quan; sinh 9 hình EDA |
| `preprocessing.py` | Kiểm tra dữ liệu, chọn ba cột số, z-score, lưu dữ liệu và scaler |
| `tuning.py` | Thử K, tính chỉ số và lưu selection.json; không tự chọn đặc trưng |
| `train.py` | Fit cấu hình đã chọn, lưu mô hình và thông tin huấn luyện |
| `evaluate.py` | Nạp mô hình đã lưu, đánh giá dữ liệu đã phân cụm, xuất nhãn/hồ sơ cụm |
| `analysis.py` | Diễn giải các nhóm, thống kê giới tính, kiểm tra độ ổn định qua seed |
| `visualization.py` | Biểu đồ Silhouette/Elbow, phân cụm 2D/3D và Silhouette từng khách |
| `pipeline.py` | EDA → chuẩn hóa → tuning → train → evaluate → kiểm tra seed |

`tuning.py`, `train.py`, `evaluate.py` có lệnh riêng nhận `--run-id`. Tuning cần dữ liệu đã chuẩn bị; muốn tuning lại sau khi đã lưu mô hình phải tạo run mới. Đánh giá lại kiểm tra hash dữ liệu và cấu hình trước khi xuất kết quả.

### Đầu ra Part 1

```text
runs/customers/<run_id>/
├── config_used.json
├── run.json, events.jsonl
├── preprocessing/             CSV sạch, chuẩn hóa, tham số scale
├── models/                    scaler.joblib, kmeans.joblib, clustering.joblib,
│                              labels.npy, training.json
├── tuning/                    metrics_by_k.csv, selection.json, so sánh K
├── results/
│   ├── customers_labeled.csv  Mỗi khách hàng kèm nhãn cụm
│   ├── cluster_profile.csv    Số khách, trung bình, tên nhóm, gợi ý tiếp thị
│   ├── gender_by_cluster.csv  Số lượng/tỷ lệ giới tính mỗi cụm
│   ├── metrics.csv            Chỉ số phân cụm cuối
│   ├── seed_stability.csv     Kết quả theo seed
│   └── pairwise_ari.csv       ARI giữa từng cặp seed
├── figures/
│   ├── eda/                   9 hình EDA; CSV trong eda/tables/
│   ├── silhouette_by_k.png    Silhouette và Elbow
│   ├── silhouette_clusters.png
│   ├── income_spending.png
│   └── customer_clusters_3d.html
└── report.md
```

`marketing_hypothesis` là gợi ý sau phân cụm, không phải nhãn cụm hoặc điều K-Means tự học. Số khách mỗi cụm không bắt buộc bằng nhau. EDA chạy riêng mặc định lưu hình tại `docs/images/eda/`; khi chạy pipeline lưu tại `figures/eda/` của run mới. Các run cũ không tự được cập nhật.

## Part 2 — Phân đoạn ảnh

### Luồng hoạt động

```text
Đọc ảnh, sửa hướng EXIF, chuyển RGB
→ thu nhỏ giữ tỷ lệ
→ chuyển không gian màu theo cấu hình
→ flatten thành N × 3 và chuẩn hóa màu
→ fit KMeansScratch cho từng K
→ thay màu pixel bằng màu tâm và tái tạo ảnh thu nhỏ
→ tính chỉ số, vẽ hình
→ dùng tâm đã học gán cụm cho pixel ảnh gốc theo từng lô
→ thay màu và reshape về kích thước ảnh gốc, lưu PNG
```

Mỗi ảnh được học độc lập; các K được thử độc lập. Thứ tự pixel được giữ nguyên khi flatten, nên reshape khôi phục đúng vị trí. Mô hình học các tâm màu, không học nhãn hoa/cây/biển.

### Chức năng các file Part 2

| File trong src/part2_image | Hàm/class chính | Chức năng |
|---|---|---|
| `preprocessing.py` | `load_image`, `resize_image`, `to_pixels`, `prepare_image` | Đọc ảnh, EXIF/RGB, thu nhỏ, chuyển RGB/HSV/LAB và chuẩn hóa pixel |
| `preprocessing.py` | `reconstruct_image` | Reshape vector màu, nhân thang màu, làm tròn, clip, uint8 và chuyển về RGB để hiển thị |
| `kmeans_scratch.py` | `KMeansScratch` | Kế thừa lõi NumPy trong algorithms/kmeans.py; cung cấp centroids_ ngoài cluster_centers_ |
| `segment.py` | `segment_image` | Fit K-Means trên pixel ảnh thu nhỏ và thay pixel bằng tâm cụm |
| `segment.py` | `segment_full_resolution` | Predict từng lô pixel ảnh gốc bằng tâm đã học; không fit lại |
| `evaluate.py` | `evaluate_image`, `evaluate_segmentation` | Inertia, RGB MSE/RMSE, dân số cụm, số vòng, hội tụ, thời gian fit/tái tạo |
| `visualization.py` | `plot_comparison_grid`, `plot_inertia` | Ảnh gốc cạnh kết quả từng K, đồ thị inertia theo K |
| `pipeline.py` | `run_part`, `main` | Chạy các ảnh/các K, lưu ảnh kích thước gốc, bảng chỉ số và báo cáo |
| `__init__.py` | — | Đánh dấu package Python |

### Cấu hình và cách đọc kết quả

Mặc định K=2,4,8,16, max_size=192, n_init=3, max_iter=100, tol=0.0001, seed=42, colorspace=RGB. Không tự động chọn K tốt nhất. K nhỏ tạo ít màu, mất chi tiết; K lớn thường giữ nhiều sắc độ và tốn tính toán hơn. K vượt số màu khác nhau sau thu nhỏ sẽ báo lỗi.

RGB và LAB encoded chia các kênh cho 255; HSV encoded chia cho [179,255,255]. HSV dùng Euclid nên chưa phản ánh đầy đủ tính vòng của hue. `reconstruct_image` hoàn nguyên thang màu và chuyển HSV/LAB về RGB; hàm này không huấn luyện.

Inertia tính trên không gian màu chuẩn hóa. RGB MSE/RMSE tính giữa ảnh thu nhỏ và ảnh tái tạo ở thang RGB 0–255, không phải độ chính xác nhận diện vật thể. Chỉ so inertia giữa các K khi cùng ảnh, kích thước và không gian màu. `time_scratch` gồm fit và tái tạo ảnh thu nhỏ; `time_full_resolution` là thời gian gán màu cho ảnh gốc. Toàn bộ pixel ảnh gốc dùng cùng tâm đã học từ ảnh thu nhỏ, không phải học lại trên toàn ảnh.

### Đầu ra Part 2

```text
runs/image/<run_id>/
├── config_used.json
├── run.json, events.jsonl      Cấu hình nguồn, hash ảnh, kích thước, trạng thái
├── metrics.csv                image, k, colorspace, inertia, rgb_mse, rgb_rmse,
│                              n_iter, converged, observed_clusters,
│                              time_scratch, time_full_resolution, source
├── segmented/                 <ảnh>_k2.png, ..._k4.png, ..._k8.png, ..._k16.png
│                              Ảnh phân đoạn đúng kích thước nguồn
├── figures/
│   ├── <ảnh>_k2_4_8_16.png     Ảnh gốc thu nhỏ cạnh các kết quả K, một hàng
│   └── inertia_vs_k.png
└── report.md
```

Không có bước so sánh sklearn. Pipeline ảnh hiện không xuất mô hình joblib hay nhãn pixel riêng; xuất ảnh và chỉ số của từng cấu hình. Hai ảnh cùng tên ở các thư mục khác nhau được thêm mã phân biệt.

## Notebook để thuyết trình và demo

- [Part 1: phân cụm khách hàng](notebooks/01_customer_clustering.ipynb): EDA, chuẩn hóa, giải thích 5 bước, chọn K, train, đánh giá, hồ sơ cụm, demo khách hàng mới. Run All tạo run mới. Kiểm tra seed mặc định tắt để demo nhanh; bật RUN_STABILITY nếu cần. Khách mới dùng đúng scaler/thứ tự cột rồi predict, không fit lại.
- [Part 2: phân đoạn ảnh](notebooks/02_image_segmentation.ipynb): giải thích pixel, học/tái tạo màu, che cụm, thử nhiều K và xuất ảnh. Kết quả notebook được trình bày ngay trong cell; ảnh xuất riêng ở notebook_exports.

## Tài liệu và dữ liệu

- [Báo cáo Part 1](docs/Lab03_Report.md)
- [Part 2: phân đoạn ảnh — 8 phần](docs/Image_Segmentation_Report.md)
- [Part 1 theo 5 bước K-Means](docs/Part1_KMeans_5_steps.md)
- [Giải thích tham số](docs/ReadmeForParameters.md)
- [Cấu hình](config.yaml)
- [Dữ liệu khách hàng](data/customers/raw/Mall_Customers.csv)
