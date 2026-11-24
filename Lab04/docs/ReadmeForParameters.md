# Lab04 — Giải thích tham số

[Cấu hình mặc định](../src/config.yaml) mô tả không gian thử. [selection.json](../results/20261123_135712_mlp_comparison/evaluation/selection.json) ghi lựa chọn của lượt tham chiếu.

| Tham số | Giá trị cấu hình | Vai trò |
|---|---|---|
| seed | 42 | Chia test và khởi tạo mô hình, RNG PyTorch/NumPy/Python |
| test_fraction | 0,2 | Tỷ lệ test trên toàn dữ liệu |
| validation_fraction | 0,2 | Tỷ lệ validation trong phần còn lại sau tách test |
| hidden | [32,16] hoặc [64,32] | Số neuron ở hai lớp ẩn |
| learning_rate | 0,001 | Bước cập nhật của Adam |
| epochs | 100 | Số epoch tối đa |
| batch_size | 32 | Số mẫu mỗi lần tính gradient, batch cuối có thể nhỏ hơn |
| patience | 15 | Số epoch liên tiếp validation MSE không cải thiện trước khi dừng |
| max_depth của cây | 6 hoặc 12 | Giới hạn độ sâu Decision Tree |
| min_samples_leaf của cây | 5 | Số mẫu tối thiểu tại lá |
| n_estimators | 100 | Số cây trong Random Forest |
| max_depth của rừng | 16 hoặc null | null cho phép mở rộng cây theo điều kiện dừng khác |
| min_samples_leaf của rừng | 2 | Số mẫu tối thiểu tại lá mỗi cây |

## Các giá trị cần phân biệt

Tỷ lệ train/validation/test toàn bộ dữ liệu xấp xỉ 64/16/20. Seed chia validation là seed + 1, tức 43. Validation chọn cấu hình và epoch. Test dùng để đo chất lượng sau lựa chọn.

MSELoss là hàm mục tiêu huấn luyện. Adam là optimizer. ReLU là activation ở lớp ẩn. Đầu ra Linear cuối giữ giá trị thực. Các lựa chọn này được đặt trong mã model.py và train.py.

Mỗi ứng viên MLP được khởi tạo theo seed cấu hình. DataLoader xáo trộn train bằng generator có seed đó. StandardScaler chỉ học từ train, còn target giữ nguyên thang đo.

Lượt tham chiếu chọn MLP [64,32], epoch tốt nhất 79 và dừng tại epoch 94. best_epoch là vị trí checkpoint được chọn. Số dòng history là số epoch thực sự đã chạy. fit_seconds tính toàn thời gian fit của ứng viên, bao gồm đo train/validation mỗi epoch.

## Các đại lượng được học

| Thành phần | Giá trị được học và nơi lưu |
|---|---|
| StandardScaler | Trung bình và độ lệch chuẩn từng cột, lưu scaler.joblib |
| MLP | Trọng số và bias các lớp Linear, lưu mlp.pt |
| Linear Regression | Hệ số và intercept, lưu linear_regression.joblib |
| Decision Tree | Cột chia, ngưỡng chia và giá trị lá, lưu decision_tree.joblib |
| Random Forest | Tập các cây đã học, lưu random_forest.joblib |

learning_rate, hidden và patience là tham số cấu hình. Trọng số và bias là giá trị thay đổi qua huấn luyện. mlp_history.csv lưu các chỉ số train/validation mỗi epoch. mlp.pt lưu checkpoint được chọn. Mã hiện tại chưa xuất từng bộ trọng số sau mỗi batch hoặc epoch.

## Chỉ số đánh giá

- MSE = trung bình của (thực tế − dự đoán)².
- RMSE = căn bậc hai MSE.
- MAE = trung bình của |thực tế − dự đoán|.
- R² = 1 − tổng bình phương sai số / tổng bình phương độ lệch target so với trung bình tập đánh giá.

RMSE và MAE nhân 100.000 sẽ đổi sang USD. MSE nhân 100.000² sẽ đổi sang USD². Mẫu số của R² và các trường hợp target hằng được xử lý trong [evaluate.py](../src/evaluate.py).

## Chế độ kiểm tra nhanh

`--quick` dùng MLP [32,16], learning rate 0,001, 5 epoch, batch size 64 và patience 3. Decision Tree dùng ứng viên đầu. Random Forest dùng 20 cây, độ sâu 12 và ít nhất 2 mẫu/lá. Cấu hình thực tế được ghi lại vào config.yaml trong lượt chạy.

[Quay lại báo cáo](Lab04_Report.md)
