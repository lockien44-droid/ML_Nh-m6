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
