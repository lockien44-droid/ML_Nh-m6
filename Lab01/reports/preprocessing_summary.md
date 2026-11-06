# Lab01 — Tóm tắt tiền xử lý

Đầu vào là [CSV email gốc](../Data/raw/spam_ham_dataset.csv). Cột `text` chứa nội dung email, `label_num` chứa nhãn (0 là ham, 1 là spam), còn `email_id` dùng để truy vết dòng gốc.

## Kết quả lọc dữ liệu

| Chỉ tiêu | Số email |
|---|---:|
| Dòng gốc | 5.171 |
| Loại do văn bản hoặc nhãn không hợp lệ | 0 |
| Loại do rỗng sau khi làm sạch | 0 |
| Loại do trùng văn bản sau khi làm sạch | 287 |
| Còn lại sau lọc | 4.884 |

## Quy trình và phân chia

[Mã tiền xử lý](../src/preprocessing.py) bỏ HTML, chuyển chữ thường, bỏ dấu câu và từ dừng tiếng Anh, rồi chuẩn hóa khoảng trắng. Sau khi làm sạch, chương trình kiểm tra xung đột nhãn và loại trùng trước khi chia để tránh đưa cùng văn bản sạch vào cả hai tập.

Chia phân tầng **80/20** với seed **42** để giữ tỷ lệ ham/spam gần nhau giữa train và test.

| Tập | Ham | Spam | Tổng |
|---|---:|---:|---:|
| Train | 2.741 | 1.166 | 3.907 |
| Test | 686 | 291 | 977 |

## TF-IDF và đầu ra

Vectorizer dùng từ đơn và cụm hai từ, chuẩn hóa L2 và giữ tối đa 50.000 đặc trưng. Từ vựng và IDF được fit trên train. Test chỉ gọi `transform()` với cùng vectorizer. Ma trận thưa CSR của train có kích thước **3.907 × 50.000**, còn test là **977 × 50.000**.

| Đầu ra | Nội dung |
|---|---|
| [summary.json](../Data/preprocessing/summary.json) | Số dòng lọc, số mẫu mỗi lớp và kích thước ma trận |
| [Các bước tiền xử lý](../Data/preprocessing/) | CSV kiểm tra, loại trùng, chia tập và từng bước làm sạch |
| [09_tfidf](../Data/preprocessing/09_tfidf/) | Ma trận `.npz`, nhãn theo ID và danh sách đặc trưng |

Chạy từ thư mục gốc repository để tạo lại các đầu ra tiền xử lý.

```powershell
python -m Lab01.src.preprocessing
```
