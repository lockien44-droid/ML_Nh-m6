# Lab03 — Mall Customer Clustering

Dùng Mall_Customers.csv gồm 200 khách và năm cột. K-Means học từ Age,
Annual Income (k$) và Spending Score (1-100). Gender dùng mô tả sau phân cụm.

## Thành viên nhóm

- Vũ Thị Thanh Thùy
- Nguyễn Văn Minh
- Lê Hồng Sơn
- Phan Đình Hưng
- Phạm Minh Tiến
- Võ Thị Anh Tú

## Cài đặt và chạy

    python -m pip install -r Lab03/requirements.txt
    python -m Lab03.src.pipeline

Thuật toán K-Means, chuẩn hóa, Silhouette và ARI tự cài bằng NumPy.
Matplotlib vẽ biểu đồ tĩnh, Plotly vẽ biểu đồ 3D tương tác.

## Cấu trúc

- algorithms/ chứa các class thuật toán.
- Data/raw/Mall_Customers.csv là dữ liệu gốc, SOURCE.md và DATA_DICTIONARY.md mô tả nguồn.
- Data/preprocessing/ chứa 200 dòng chuẩn hóa và tham số scaling mới nhất.
- src/ chứa config, EDA, preprocessing, train, tuning, evaluate, analysis,
  visualization và pipeline.
- notebooks/01_customer_clustering.ipynb trình bày phân tích và kết quả.
- results/runs/<run_id>/ chứa kết quả theo phiên bản.
- docs/ chứa báo cáo, tham số và đề bài.

## Kết quả tham chiếu

K=6, Silhouette 0,4274, inertia 133,8683. 9/10 seed chọn K=6, seed 8 chọn K=8.

[Notebook](notebooks/01_customer_clustering.ipynb)
[Báo cáo](docs/Lab03_Report.md)
[Biểu đồ 3D](results/runs/2026-11-18_22-05-16/figures/customer_clusters_3d.html)
[Nguồn và giấy phép](Data/raw/SOURCE.md)

Git giữ một lần chạy tham chiếu 2026-11-18_22-05-16. Các lượt chạy mới được lưu cục bộ và gitignore.
Mở HTML 3D bằng trình duyệt để xoay, thu phóng, xem tên nhóm và tâm cụm.


## Thư mục kết quả mỗi lần chạy

    results/runs/<run_id>/
    ├── run.json
    ├── config.yaml
    ├── preprocessing/
    ├── figures/
    ├── model/
    │   └── kmeans.joblib
    ├── clusters/
    │   ├── customer_clusters.csv
    │   ├── cluster_profiles.csv
    │   └── gender_by_cluster.csv
    ├── k_selection/
    │   ├── k_comparison.csv
    │   ├── k5_vs_selected.csv
    │   └── k9_vs_selected.csv
    └── validation/
        ├── seed_stability.csv
        └── pairwise_ari.csv

validation/ là các phân tích kiểm chứng cụm, không phải tập validation có nhãn
như bài học có giám sát. Các bảng trong thư mục này đánh giá độ ổn định của K-Means khi thay đổi seed.

Notebook Run All tạo thư mục mới có hậu tố notebook và mã riêng; không ghi kết quả vào lần chạy tham chiếu. config.yaml và run.json trong thư mục mới ghi cấu hình và hash dữ liệu.


Các lần chạy được lưu cục bộ và bỏ qua trong Git. Repository giữ một lần chạy tham chiếu đầy đủ để đối chiếu báo cáo. Tham chiếu hiện tại là `2026-11-18_22-05-16`. Chạy pipeline sẽ tạo thư mục mới, không ghi đè thư mục tham chiếu.
