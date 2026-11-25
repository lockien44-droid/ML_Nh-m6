# Lab03 — Part 2: Phân đoạn ảnh bằng K-means

## 1. Mục tiêu

Áp dụng K-Means để gom các pixel có màu tương đồng thành K cụm, từ đó tạo ảnh phân đoạn bằng cách thay màu từng pixel bằng màu đại diện của cụm.Kết quả là các nhóm màu; không mặc định mỗi cụm tương ứng một vật thể.

## 2. Đầu vào

Hai ảnh mặc định:

- [draw_beach.jpg](../data/images/raw/draw_beach.jpg): tranh bãi biển.
- [photo_flower.jpg](../data/images/raw/photo_flower.jpg): ảnh hoa.

Một ảnh RGB có dạng `(H, W, 3)`. Mỗi pixel là một mẫu `x_i = [R_i, G_i, B_i]`, ba kênh màu là ba đặc trưng. K là số cụm màu, không phải số kênh hoặc số đặc trưng.

Mỗi ảnh và mỗi giá trị K được huấn luyện độc lập. Có thể thay ảnh bằng tham số `--image`; đường dẫn phải trỏ tới file tồn tại.

## 3. Tiền xử lý

Mã tại [preprocessing.py](../src/part2_image/preprocessing.py):

| Hàm | Xử lý |
|---|---|
| `load_image` | Đọc bằng OpenCV imdecode, chuyển BGR → RGB |
| `resize_image` | Thu nhỏ giữ tỷ lệ, cạnh dài không vượt `max_size=192` |
| `to_pixels` | Chuyển không gian màu theo cấu hình, reshape về `(N, 3)` và chuẩn hóa |
| `prepare_image` | Gọi các bước trên, trả dữ liệu pixel, ảnh thu nhỏ và ảnh gốc |


## 4. Thuật toán

[KMeansScratch](../src/part2_image/kmeans_scratch.py) kế thừa lõi [KMeans NumPy ](../algorithms/kmeans.py).

Với mỗi K, thuật toán thực hiện:

1. **Khởi tạo tâm bằng k-means++:** chọn tâm đầu ngẫu nhiên; các tâm sau được chọn theo xác suất tỷ lệ với bình phương khoảng cách tới tâm gần nhất đã có.
2. **Gán pixel:** chọn tâm gần nhất theo bình phương khoảng cách Euclid:

   `label_i = argmin_j sum_f (X[i,f] - center[j,f])²`.

3. **Cập nhật tâm:** lấy trung bình vector màu của các pixel trong cụm:

   `center[j] = mean(X[labels == j], axis=0)`.

4. **Lặp và kiểm tra hội tụ:** tính

   `shift = sqrt(sum((updated - centers)²))`.

   Dừng khi `shift <= tol=0.0001`, hoặc đạt `max_iter=100`. Nếu chỉ hết giới hạn vòng mà shift vẫn lớn hơn tol, `converged_` là False.

Cụm rỗng được xử lý bằng cách đặt tâm tại điểm đang xa tâm gần nhất nhất. Với mỗi K, chạy `n_init=3` lần khởi tạo và giữ nghiệm có inertia nhỏ nhất. Đây không phải ba vòng cập nhật; mỗi lần khởi tạo có thể chạy tối đa 100 vòng. `random_state=42` giúp tái lập kết quả trên cùng dữ liệu và môi trường.

Hàm `segment_image` trong [segment.py](../src/part2_image/segment.py) gọi `fit` trên pixel của ảnh thu nhỏ. Mô hình học được K tâm màu, không học vị trí hoặc ý nghĩa vật thể.

## 5. Tái tạo ảnh

Sau khi phân cụm, mỗi pixel được thay bằng màu tâm tương ứng:

```python
new_pixels = model.cluster_centers_[labels]
```

Hàm `reconstruct_image` trong preprocessing.py thực hiện:

```python
encoded = np.clip(
    np.rint(features.reshape(shape) * scales),
    0, scales
).astype("uint8")
```

- `reshape(shape)` đưa về chiều cao × chiều rộng × ba kênh.
- Nhân `scales` hoàn nguyên thang màu; RGB dùng 255.
- Làm tròn, giới hạn phạm vi hợp lệ và chuyển sang uint8.
- Với HSV/LAB, chuyển về RGB để hiển thị.

Ví dụ màu chuẩn hóa `[0.2, 0.5, 0.8]` được đổi về khoảng `[51, 128, 204]`. Vị trí pixel giữ nguyên; chỉ màu thay đổi.

Để tạo ảnh đúng kích thước gốc, `segment_full_resolution` dùng tâm đã học từ ảnh thu nhỏ, gán cụm cho pixel ảnh gốc theo từng lô rồi tái tạo. **Không fit lại trên ảnh gốc.** Xử lý theo lô giảm lượng bộ nhớ cần cho ma trận khoảng cách.

## 6. Thực nghiệm

Cấu hình mặc định:

| Tham số | Giá trị |
|---|---|
| K | 2, 4, 8, 16 |
| Không gian màu | RGB |
| Cạnh dài ảnh dùng học | Tối đa 192 pixel |
| `n_init` | 3 |
| `max_iter` | 100 |
| `tol` | 0.0001 |
| `random_state` | 42 |



Luồng mã: `pipeline.py` → chuẩn bị pixel → `evaluate_segmentation` gọi `segment_image` và tính chỉ số → gán màu cho ảnh gốc → `visualization.py` vẽ và lưu báo cáo.

Kết quả mỗi lần chạy:

```text
Lab03/runs/image/<run_id>/
├── config_used.json
├── run.json, events.jsonl
├── metrics.csv
├── segmented/                 Ảnh phân đoạn đúng kích thước nguồn theo từng K
├── figures/
│   ├── <ảnh>_k2_4_8_16.png     Ảnh gốc thu nhỏ đặt cạnh các kết quả
│   └── inertia_vs_k.png
└── report.md
```

Ảnh so sánh có một hàng, gồm ảnh gốc thu nhỏ và các kết quả K. Đọc notebook [02_image_segmentation.ipynb](../notebooks/02_image_segmentation.ipynb) để demo từng bước, xem tâm màu và minh họa che một cụm.

## 7. Đánh giá

### Đánh giá thị giác

Đặt ảnh gốc cạnh các kết quả để xem vùng màu bị gộp, chi tiết còn giữ được, và khác biệt giữa các K. Không kết luận “nhận diện đúng vật thể” chỉ từ hình phân cụm màu. Phần 2 hiện thử các K, chưa tự động chọn K tốt nhất.

### Chỉ số trong metrics.csv

| Chỉ số | Ý nghĩa |
|---|---|
| `inertia` | Tổng bình phương khoảng cách pixel tới tâm trong không gian màu chuẩn hóa |
| `rgb_mse` | Trung bình bình phương sai khác các kênh RGB giữa ảnh thu nhỏ và ảnh tái tạo |
| `rgb_rmse` | Căn MSE, cùng thang giá trị RGB 0–255 |
| `n_iter` | Số vòng của nghiệm khởi tạo được giữ lại |
| `converged` | Có đạt điều kiện dịch chuyển tâm <= tol hay không |
| `observed_clusters` | Số cụm thực tế có pixel trong ảnh dùng học |
| `time_scratch` | Giây để fit và tái tạo ảnh thu nhỏ |
| `time_full_resolution` | Giây để gán màu cho pixel ảnh gốc; không gồm fit lại |

Với N pixel và tâm c của từng pixel:

```text
Inertia = sum_i ||X[i] - center[label_i]||²
MSE_RGB = mean((RGB_original - RGB_reconstructed)²)
RMSE_RGB = sqrt(MSE_RGB)
```

So sánh inertia giữa các K trên cùng ảnh, cùng kích thước và cách chuẩn hóa. Tăng K thường làm sai số giảm nhưng không bảo đảm từng lần tối ưu cục bộ luôn đơn điệu. Inertia/RMSE đo tái tạo màu, không phải tỷ lệ phân đoạn đúng và không đánh giá trực tiếp nhãn vật thể. RMSE trong bảng được tính trên ảnh thu nhỏ, không phải ảnh đúng kích thước gốc.

## 8. Kết luận

K-Means chuyển bài toán ảnh thành phân cụm vector màu: pixel là mẫu, RGB là đặc trưng, tâm cụm là màu đại diện. Tái tạo ảnh bằng cách giữ vị trí pixel và thay màu bằng màu tâm.

- **K nhỏ:** ảnh ít màu, các vùng dễ quan sát nhưng mất chi tiết.
- **K lớn:** giữ nhiều sắc độ và thường giảm sai số màu; số phép tính tăng.
- **Giới hạn:** chỉ dùng màu nên các pixel ở xa nhau hoặc thuộc các vật thể khác nhau vẫn có thể nằm cùng cụm. Không có nhãn chuẩn để tuyên bố độ chính xác nhận diện vật thể.

Khi thuyết trình: “Phần 2 sử dụng K-Means tự cài để tìm các màu đại diện trong ảnh. Nhóm học tâm trên ảnh thu nhỏ, sau đó dùng các tâm đó để gán màu cho ảnh gốc. Các K được so sánh bằng hình ảnh và sai số tái tạo màu, không phải độ chính xác phân loại vật thể.”

[Quay lại README Lab03](../README.md)

### Xử lý ảnh bằng OpenCV

Part 2 dùng imdecode để đọc, cvtColor chuyển BGR → RGB, resize với INTER_AREA để thu nhỏ giữ tỷ lệ, và imencode để lưu PNG. Dùng fromfile/tofile hỗ trợ đường dẫn tiếng Việt. Ảnh gốc và ảnh thu nhỏ là NumPy array RGB; thuật toán K-Means vẫn là class tự cài, không dùng cv2.kmeans. Cách resize khác PIL nên chỉ số có thể thay đổi so với run cũ.
