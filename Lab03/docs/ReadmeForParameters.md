# Tham số Lab03

| Tham số | Vai trò |
| --- | --- |
| features | Ba cột số dùng phân cụm |
| id_column | Mã khách hàng, không đưa vào khoảng cách |
| k_values | Các số cụm được thử, từ 2 đến n−1 |
| max_iter | Giới hạn vòng cập nhật tâm cho một khởi tạo |
| tol | Ngưỡng hội tụ của độ dịch chuyển toàn bộ tâm trong không gian chuẩn hóa |
| random_state | Seed cho bộ sinh ngẫu nhiên |
| n_init | Số khởi tạo mỗi K, giữ nghiệm inertia thấp nhất |

Silhouette càng cao cho thấy cấu trúc cụm càng tách biệt theo metric đang dùng.
Inertia thường giảm khi tăng K, nên chưa đủ để tự chọn K.
Đánh giá Silhouette chính xác cần ma trận khoảng cách n × n, phù hợp bộ 200 khách.
