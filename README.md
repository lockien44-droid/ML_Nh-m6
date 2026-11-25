# Machine Learning — Nhóm 6

Repository bài thực hành Machine Learning. Các lab tách riêng dữ liệu, thuật toán,
tiền xử lý, thực nghiệm, đánh giá và tài liệu. Thuật toán tự cài được ghi rõ trong README từng lab.

## Thành viên

1. Vũ Thị Thanh Thùy
2. Nguyễn Văn Minh
3. Lê Hồng Sơn
4. Phan Đình Hưng
5. Phạm Minh Tiến
6. Võ Thị Anh Tú

## Nội dung

| Lab | Bài toán | Trạng thái và tài liệu |
| --- | --- | --- |
| Lab01 | Phân loại email spam bằng Logistic Regression, Linear SVM, Naive Bayes | [Hướng dẫn và notebook](Lab01/README.md) |
| Lab02 | Dự báo doanh thu FMCG theo SKU/tuần bằng bốn thuật toán hồi quy | [Hướng dẫn, kết quả và giới hạn](Lab02/README.md) |
| Lab03 | Phân cụm khách hàng và phân đoạn ảnh bằng K-Means | [Hướng dẫn, 3D và báo cáo](Lab03/README.md) |
| Lab04 | Hồi quy giá nhà bằng PyTorch MLP và ba mô hình đối chiếu | [Hướng dẫn chạy và đầu ra](Lab04/README.md) |

## Cài đặt và chạy

Mở terminal tại thư mục gốc repository. Python 3.10 trở lên được đề xuất.

    python -m pip install -r requirements.txt
    python -m Lab01.src.pipeline
    python -m Lab02.src.pipeline
    python -m Lab03.src.pipeline

requirements.txt gốc là môi trường dùng chung, gồm scikit-learn cho Lab01.
Lab02 tự cài tiền xử lý, thuật toán và chỉ số. Lab03 có bộ phụ thuộc riêng:

    python -m pip install -r Lab03/requirements.txt

## Kết quả và notebook

Mỗi lab lưu đầu ra thực nghiệm theo mã lần chạy. README từng lab giải thích đường dẫn
và lệnh chạy. Mở các file ipynb trong notebooks/ bằng VS Code hoặc Jupyter, chọn
kernel Python đã cài các thư viện tương ứng.

## Nguồn dữ liệu

Các nguồn và giới hạn được ghi theo từng lab. Lab03 hiện dùng Mall_Customers.csv200 khách, đã đối chiếu nguồn Kaggle. Giấy phép khai báo Other. [Thông tin nguồn và giấy phép Lab03](Lab03/Data/raw/SOURCE.md).
