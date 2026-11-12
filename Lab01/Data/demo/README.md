# Bộ 15 email mô phỏng để demo Lab01

Các email được viết bằng tiếng Anh để phù hợp với bước xử lý văn bản của Lab01. Tất cả là nội dung mô phỏng, chưa được dùng để huấn luyện và chưa được kiểm tra dự đoán khi soạn bộ này. Nhãn là nhãn dự kiến theo tình huống mô phỏng, không phải kết quả của mô hình. Bộ demo không thay thế tập test chính thức.

Có 5 ham rõ ràng, 5 spam rõ ràng và 5 trường hợp khó phân loại (3 ham, 2 spam). Tổng cộng 8 ham và 7 spam.

Khi dự đoán, chỉ đưa cột `text` vào clean_text/TF-IDF; không đưa `group`, `expected_label` hay `context_vi` vào mô hình. Một số trường hợp khó cần ngữ cảnh mà mô hình chỉ đọc nội dung không có được. Giữ cả kết quả đúng và sai để thảo luận.

CSV: `demo_emails.csv` (UTF-8 có BOM, 15 dòng dữ liệu; nội dung nhiều dòng được đặt trong dấu nháy theo chuẩn CSV). `expected_label`: 0=ham, 1=spam. `source=synthetic`.

## 1. Ham rõ ràng

### H01 — Ham dự kiến

Trao đổi nội bộ về lịch họp dự án.

```text
Subject: Project meeting on Thursday

Hi team,
Our project meeting is scheduled for Thursday at 10 AM in room 204. Please review the experiment results and bring your notes about the presentation.
Thanks,
Alex
```

### H02 — Ham dự kiến

Đồng nghiệp gửi nội dung báo cáo cần duyệt.

```text
Subject: Updated report for review

Hi Morgan,
I have updated the report with the latest figures and corrected the table captions. Please review section three and let me know if any explanation is unclear.
Best,
Taylor
```

### H03 — Ham dự kiến

Tin nhắn cá nhân từ người quen.

```text
Subject: Lunch plans for tomorrow

Hi Sam,
Would you like to have lunch after class tomorrow? I finish at noon and can meet you outside the library. Let me know what time works for you.
See you,
Jamie
```

### H04 — Ham dự kiến

Thông báo tương ứng với yêu cầu đặt sách đã thực hiện.

```text
Subject: Your library book is ready

Hello,
The book you reserved is now available at the campus library. Please collect it at the service desk before Friday and bring your student card.
Library Services
```

### H05 — Ham dự kiến

Giảng viên phản hồi bài tập đã nộp.

```text
Subject: Feedback on your assignment

Hi Jordan,
Your assignment has been reviewed. The introduction is clear, but the evaluation section needs more explanation. Please compare precision and recall and submit your revision next week.
Regards,
Chris
```

## 2. Spam rõ ràng

### S01 — Spam dự kiến

Thông báo trúng thưởng không đăng ký, đòi phí và thông tin ngân hàng.

```text
Subject: You have won a cash prize

Congratulations!
Your email address has been selected for a cash prize of one million dollars. Send your bank account details and pay the processing charge today to receive your reward. Act immediately before your prize expires.
```

### S02 — Spam dự kiến

Chào mời thu nhập phi thực tế, yêu cầu trả tiền trước.

```text
Subject: Guaranteed income with no experience

Earn thousands of dollars every day from home! No skills, no effort, and no experience are required. Send your registration payment now and we guarantee you will become rich within a week. Limited places available!
```

### S03 — Spam dự kiến

Giả mạo ngân hàng yêu cầu mật khẩu và mã xác thực.

```text
Subject: Urgent account verification required

Your bank account will be permanently closed in two hours. Reply with your password, card number, and verification code so our support team can restore access. Do not contact your branch because this must be handled by email.
```

### S04 — Spam dự kiến

Quảng cáo điều trị phi thực tế gửi ngoài mong muốn.

```text
Subject: Exclusive miracle treatment

Buy our miracle capsules today! They cure every illness in just three days with guaranteed results. No medical consultation is needed. Order ten bottles now and receive an exclusive secret discount. Reply immediately to secure your supply.
```

### S05 — Spam dự kiến

Đầu tư cam kết lợi nhuận phi thực tế và yêu cầu chuyển tiền.

```text
Subject: Private investment opportunity

You have been chosen for a confidential investment that doubles your money in 24 hours with zero risk. Transfer your deposit today to reserve your position. Keep this opportunity secret and do not ask anyone else for advice.
```

## 3. Trường hợp khó phân loại

### D01 — Ham dự kiến

Ham mô phỏng từ nội bộ; có các từ free, register, payment dễ gây nhầm.

```text
Subject: Free Python workshop for employees

Hi team,
The company is offering a free Python workshop next Friday. Employees can register through the usual internal learning portal. There is no payment required. Please check with your manager if you need time away from your regular tasks.
Learning and Development
```

### D02 — Ham dự kiến

Ham: người nhận đã yêu cầu hoàn tiền cho đơn hàng này; từ bank details và processing fee xuất hiện trong câu phủ định.

```text
Subject: Refund approved for your recent order

Hello Casey,
We have approved the refund you requested for order 4821. The amount will be returned to your original payment method within five business days. You do not need to send your bank details or pay a processing fee.
Customer Support
```

### D03 — Ham dự kiến

Ham: hướng dẫn bảo mật nội bộ; có password, verification code nhưng không yêu cầu gửi chúng.

```text
Subject: Security reminder for your work account

Hi everyone,
Please review the security settings for your work account this week. Open the company portal using your saved bookmark and enable two-factor authentication if it is not already active. The IT team will never ask you to email your password or verification code.
IT Services
```

### D04 — Spam dự kiến

Spam/lừa đảo mô phỏng: không có hợp đồng hay hóa đơn tương ứng, người gửi mạo danh. Văn phong giống email công việc; ngữ cảnh này không nằm trong đầu vào mô hình.

```text
Subject: Following up on invoice 7318

Hello,
I am following up on invoice 7318 for the consulting work completed last month. Our payment account has changed. Please use the replacement bank details in my next message and arrange the transfer today. There is no need to check with the previous contact.
Accounts Team
```

### D05 — Spam dự kiến

Spam/lừa đảo tuyển dụng mô phỏng: người nhận không ứng tuyển, bị yêu cầu đóng phí trước. Có từ vựng công việc bình thường.

```text
Subject: Next step in your application

Hello,
We reviewed your profile and would like to offer you a remote assistant position. No interview is necessary. Before receiving your contract, please pay the equipment reservation fee by bank transfer. This payment is required today to keep the position available.
Recruitment Office
```

## Chạy thử một email

Lệnh hiện tại nhận từng email qua `--text`; chưa có tùy chọn đọc cả tệp CSV. Ví dụ:

```powershell
python -m Lab01.src.predict --run-id 20261113_024910_baseline_tuned --model logistic_regression --text "Subject: Free Python workshop for employees. Hi team, the company is offering a free Python workshop next Friday. Employees can register through the usual internal learning portal. There is no payment required."
```

Đổi `--model logistic_regression` thành `--model svm` để so sánh trên cùng nội dung.
