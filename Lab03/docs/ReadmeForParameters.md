# Tham số Lab03

| Tham số | Vai trò |
| --- | --- |
| data.path | Đường dẫn Mall_Customers.csv tương đối so với Lab03 |
| id_column | Mã khách hàng để truy nguyên, không đưa vào khoảng cách |
| features | Tuổi, thu nhập nghìn USD và điểm chi tiêu |
| k_values | Các K thử từ 2 đến 10 |
| max_iter | Tối đa số vòng cập nhật tâm mỗi khởi tạo |
| tol | Ngưỡng độ dịch chuyển toàn bộ tâm trong không gian chuẩn hóa |
| random_state | Seed cho khởi tạo tâm K-Means |
| n_init | Số lần khởi tạo mỗi K, giữ nghiệm inertia nhỏ nhất |
| analysis.seeds | Các seed thử lại việc chọn K và so sánh ARI |

Z-score dùng ddof=0.
Cột hằng có hệ số chia bằng 1. K-Means mô tả toàn bộ 200 khách đã chuẩn hóa.

## Cấu hình hai phần

Cấu hình hiện tại nằm tại Lab03/config.yaml. Nhóm customers chứa data, kmeans và analysis của bài khách hàng. Nhóm image chứa path, max_size, k_values, max_iter, tol, random_state và n_init. Hai phần dùng chung thuật toán K-Means và giữ kết quả trong thư mục riêng dưới runs/customers/<run_id> hoặc runs/image/<run_id>.

Phần ảnh dùng image.paths (danh sách), K=2,4,8,16 và colorspace=RGB. Mỗi ảnh/K được so sánh với sklearn theo metrics.csv trong runs/image/<run_id>.

## Tuning nhiều thuật toán và đánh giá lại

customers.kmeans.algorithms mặc định [kmeans], có thể thêm dbscan và hierarchical. DBSCAN dùng eps_values, min_samples_values; Hierarchical dùng k_values và linkages. min_coverage mặc định 0,8 giới hạn tỷ lệ mẫu không phải noise. tuning/selection.json lưu algorithm và parameters; train đọc lựa chọn này. evaluate --run-id chỉ nạp mô hình, đối chiếu dữ liệu chuẩn hóa và đánh giá cùng tập khách hàng đã phân cụm.

image.colorspace nhận RGB, HSV hoặc LAB. Inertia giữa các không gian màu có thang đo và ý nghĩa khác nhau, nên so sánh K trong cùng không gian. rgb_rmse dùng ảnh RGB tái tạo để diễn giải sai số màu.
