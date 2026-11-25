# Lab03 — Part 1: Phân cụm khách hàng theo 5 bước K-Means

## Mục tiêu và dữ liệu đầu vào

Phân nhóm khách hàng có đặc điểm tương đồng trong `Mall_Customers.csv`. Mỗi khách hàng là một điểm gồm ba đặc trưng: `Age`, `Annual Income (k$)` và `Spending Score (1-100)`. `CustomerID` dùng để đối chiếu khách hàng; `Gender` dùng mô tả cụm sau khi học, không đưa vào khoảng cách.

Đây là học không giám sát, không có nhãn cụm đúng cho trước. Part 1 phân tích toàn bộ tập khách hàng và đánh giá cấu trúc cụm nội bộ; không chia train/validation/test như bài dự đoán doanh thu.

Trước 5 bước, kiểm tra dữ liệu và chuẩn hóa z-score: `z = (x - mean) / std`, với độ lệch chuẩn `ddof=0`. Chuẩn hóa giúp ba đặc trưng có thang đo tương đương khi tính khoảng cách. Tham số chuẩn hóa được lưu để dùng lại.

## Bước 1. Chọn số cụm K

Trong một lần gọi `fit`, K được cố định. Pipeline thực nghiệm lần lượt các K từ **2 đến 10**, theo `customers.kmeans.k_values` trong [config.yaml](../config.yaml).

Ví dụ K=4 nghĩa là yêu cầu thuật toán chia khách hàng thành bốn nhóm. Tên và ý nghĩa kinh doanh của các nhóm chỉ được xác định sau khi xem đặc điểm từng cụm.

Mỗi K thực hiện các bước 2–5 rồi được đánh giá. Pipeline chọn cấu hình có Silhouette lớn nhất trong các cấu hình hợp lệ, đáp ứng mức coverage cấu hình. Inertia và biểu đồ Elbow hỗ trợ nhận xét; không chọn K chỉ vì inertia nhỏ nhất vì chỉ số này thường giảm khi tăng K.

Mã phụ trách: [tuning.py](../src/part1_customers/tuning.py), hàm `candidates`, `tune`, `tune_run`.

## Bước 2. Khởi tạo K tâm

Class tự viết dùng **k-means++**:

1. Chọn ngẫu nhiên một khách hàng làm tâm đầu tiên.
2. Với mỗi điểm, tính khoảng cách bình phương tới tâm gần nhất đã chọn.
3. Chọn tâm tiếp theo theo xác suất tỷ lệ với khoảng cách bình phương đó.
4. Tiếp tục đến khi đủ K tâm.

Các điểm ở xa tâm hiện tại có xác suất được chọn cao hơn, giúp tâm ban đầu trải rộng hơn. Đây là cải tiến của bước chọn ngẫu nhiên K điểm trong bài giảng, vẫn thuộc bước khởi tạo K-Means. `random_state=42` giúp tái lập kết quả.

Mã phụ trách: [kmeans.py](../algorithms/kmeans.py), hàm `_initialize`.

## Bước 3. Gán mỗi điểm vào tâm gần nhất

Với mỗi khách hàng đã chuẩn hóa, tính khoảng cách tới toàn bộ K tâm và chọn tâm gần nhất:

```text
label_i = argmin_j sum_f (X[i, f] - center[j, f])²
```

Mã dùng bình phương khoảng cách Euclid để tránh tính căn; thứ tự gần–xa không thay đổi. Khách hàng có vector tuổi, thu nhập và chi tiêu gần nhau có xu hướng được gán cùng cụm.

Ví dụ khoảng cách bình phương của một khách hàng đến ba tâm là `[0.4, 2.1, 1.3]`: khách hàng được gán vào cụm có chỉ số 0. Số 0 chỉ là mã cụm, không phải mức chất lượng hay tên nhóm kinh doanh.

Mã phụ trách: `_distances` và `labels = distances.argmin(axis=1)` trong `fit`.

## Bước 4. Cập nhật tâm bằng trung bình

Sau khi gán tất cả khách hàng, tính lại tâm từng cụm bằng trung bình vector các thành viên:

```text
center[j] = mean(X[labels == j], axis=0)
```

Ví dụ hai điểm đã chuẩn hóa trong một cụm là `[0, 1, 2]` và `[2, 3, 4]`, tâm mới là `[1, 2, 3]`. Tâm có thể không trùng với khách hàng thực tế nào.

Nếu một cụm không có thành viên, mã đưa tâm đó đến điểm hiện đang được đại diện kém nhất, tức có khoảng cách tới tâm gần nhất lớn nhất. Cách này tránh lấy trung bình tập rỗng.

Mã phụ trách: vòng lặp cập nhật `updated[cluster]` trong `fit`.

## Bước 5. Lặp bước 3–4 đến khi dừng

Sau mỗi lần cập nhật, tính độ dịch chuyển tổng thể của toàn bộ ma trận tâm:

```text
shift = sqrt(sum((updated - centers)²))
```

Dừng khi `shift <= tol`, với **tol=0.0001**, hoặc đã chạy **max_iter=300** vòng. Đạt giới hạn vòng lặp không đồng nghĩa đã hội tụ; thuộc tính `converged_` ghi rõ điều này.

Sau vòng cuối, mã gán lại nhãn theo tâm cuối cùng và tính inertia:

```text
inertia = sum_i ||X[i] - center[label_i]||²
```

Inertia đo độ tập trung quanh tâm trên dữ liệu đã chuẩn hóa, không phải tỷ lệ dự đoán đúng.

Mã phụ trách: vòng lặp trong `fit`; kết quả gồm `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `converged_`.

## Chạy nhiều lần khởi tạo: n_init=20

Với mỗi K, lặp quy trình khởi tạo–gán cụm–cập nhật–dừng **20 lần**, rồi giữ lần có inertia nhỏ nhất. Đây là bước bao quanh quá trình học nhằm giảm ảnh hưởng của tâm ban đầu, không phải bước thứ sáu của một vòng K-Means.

Phân biệt hai phép chọn: **cùng K**, chọn lần khởi tạo bằng inertia; **giữa các K**, chọn cấu hình bằng Silhouette. `n_iter_` và `converged_` thuộc lần khởi tạo được giữ lại.

## Từ thuật toán đến kết quả Part 1

```text
eda.py → preprocessing.py → tuning.py → train.py → evaluate.py
                              ↓           ↓             ↓
                          chọn K      lưu mô hình   chỉ số, biểu đồ,
                                                    mô tả từng cụm
```

Các bước trên do [pipeline.py](../src/part1_customers/pipeline.py) điều phối; xem thứ tự thực tế tại file này. Năm bước K-Means nằm trong class thuật toán và được gọi khi tuning hoặc huấn luyện. `train.py` huấn luyện theo cấu hình đã chọn; `evaluate.py` nạp mô hình để đánh giá, không tuning lại. Các chỉ số Silhouette, Davies–Bouldin và Calinski–Harabasz được tính bằng NumPy.

`fit(X)` học tâm; `predict(X_new)` gán khách hàng mới vào các tâm đã học. Khách hàng mới phải dùng đúng thứ tự đặc trưng và tham số chuẩn hóa đã lưu. Nhãn cụm được diễn giải bằng trung bình đặc trưng ở đơn vị gốc và số lượng khách hàng; gợi ý kinh doanh là giả thuyết cần kiểm chứng.

## Chạy và đọc đầu ra

Tại thư mục gốc dự án:

```powershell
python -m Lab03.src.pipeline --part 1
```

Kết quả nằm trong `Lab03/runs/customers/<run_id>/`:

| Đầu ra | Nội dung |
|---|---|
| `tuning/metrics_by_k.csv` | Chỉ số của từng cấu hình K |
| `tuning/selection.json` | K và cấu hình được chọn |
| `models/kmeans.joblib` | Mô hình K-Means đã học |
| `models/scaler.joblib` | Tham số chuẩn hóa |
| `results/metrics.csv` | Chỉ số đánh giá cấu hình cuối |
| `results/customers_labeled.csv` | Khách hàng kèm nhãn cụm |
| `results/cluster_profile.csv` | Thống kê và mô tả từng cụm |
| `figures/` | Biểu đồ phục vụ phân tích |

## Cách trình bày ngắn khi thuyết trình

“Part 1 sử dụng K-Means tự cài để phân nhóm khách hàng theo tuổi, thu nhập và điểm chi tiêu sau chuẩn hóa. Thuật toán gồm năm bước: chọn K, khởi tạo K tâm bằng k-means++, gán khách hàng vào tâm gần nhất, cập nhật tâm bằng trung bình và lặp đến điều kiện dừng. Mỗi K được khởi tạo 20 lần và giữ nghiệm có inertia thấp nhất. Sau đó, nhóm so sánh các K từ 2 đến 10 bằng Silhouette để chọn cấu hình, rồi mô tả đặc điểm của từng nhóm khách hàng.”

[Xem báo cáo Part 1](Lab03_Report.md) · [Hướng dẫn Lab03](../README.md)
