# Bài 02 — Dữ liệu và cách máy học: train, test, overfitting và các thước đo

**Mục tiêu của bài:** hiểu một bài toán học máy được "ra đề" và "chấm
điểm" như thế nào — trước khi học bất kỳ thuật toán nào.

## 1. Dữ liệu là gì trong học máy?

Một bộ dữ liệu thường là một **bảng**:

- Mỗi **hàng** là một ví dụ (một email, một học sinh, một giao dịch...).
- Mỗi **cột** là một **đặc trưng (feature)** — thông tin dùng để dự
  đoán: số từ trong email, số giờ học, số tiền giao dịch...
- Cột đặc biệt cần dự đoán gọi là **nhãn (label)**: "rác/không rác",
  "qua môn/trượt", giá nhà...

Ví dụ: dự đoán qua môn từ `[giờ học mỗi ngày, số buổi nghỉ]`, nhãn là
`1` (qua) hoặc `0` (trượt) — chính là dữ liệu trong
`examples/demo_decision_tree.py`.

Hai họ bài toán lớn:

- **Hồi quy (regression):** dự đoán một *con số* — giá nhà, nhiệt độ.
- **Phân loại (classification):** dự đoán một *nhãn* — rác/không rác.

Ngoài ra còn **phân cụm (clustering)** — không có nhãn, máy tự gom nhóm
(bài 07).

## 2. Vì sao phải chia train/test?

Nếu cho học sinh ôn đúng đề thi rồi chấm bằng chính đề đó, điểm cao
không nói lên điều gì. Tương tự, ta chia dữ liệu làm hai phần:

- **Tập huấn luyện (train):** để máy học.
- **Tập kiểm tra (test):** giữ riêng, chỉ dùng để *đánh giá* lần cuối.

Trong gói code đi kèm, hàm `train_test_split` trong `aibasics.data`
làm việc này (xáo trộn rồi chia theo tỉ lệ, mặc định 25% cho test).

```python
from aibasics import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.25)
```

## 3. Overfitting — "học vẹt"

Một mô hình **overfitting (quá khớp)** khi nó thuộc lòng cả những chi
tiết ngẫu nhiên của tập train: điểm train rất cao nhưng gặp dữ liệu mới
thì sai nhiều. Ngược lại, **underfitting (chưa khớp)** là khi mô hình
quá đơn giản, đến tập train cũng làm không tốt.

Dấu hiệu nhận biết overfitting:

- Độ chính xác trên train gần 100% nhưng trên test thấp hơn hẳn.
- Mô hình quá phức tạp so với lượng dữ liệu (ví dụ cây quyết định mọc
  quá sâu — bài 05, hay k-NN với k = 1 — bài 04).

Cách chống: thêm dữ liệu, giảm độ phức tạp của mô hình, và luôn đánh
giá trên tập test chưa từng dùng khi học.

## 4. Các thước đo cơ bản

**Với phân loại — accuracy (độ chính xác):**

```
accuracy = số dự đoán đúng / tổng số dự đoán
```

Đơn giản nhưng có thể gây hiểu lầm khi dữ liệu lệch: nếu 99% email là
không rác, một mô hình "đoán bừa tất cả không rác" cũng đạt 99%. Khi đó
người ta dùng thêm **precision** (trong số đoán là rác, bao nhiêu đúng)
và **recall** (trong số rác thật, bắt được bao nhiêu).

**Với hồi quy — MSE (mean squared error, sai số bình phương trung bình):**

```
MSE = trung bình của (giá trị thật − giá trị dự đoán)²
```

MSE càng nhỏ càng tốt; bình phương khiến các lỗi lớn bị phạt nặng hơn.

Cả hai đều có sẵn trong `aibasics.data`: `accuracy_score` và
`mean_squared_error`.

## Tự kiểm tra

1. Vì sao không nên đánh giá mô hình trên chính dữ liệu đã dùng để học?
2. Một mô hình đạt accuracy 100% trên train và 62% trên test đang gặp
   hiện tượng gì? Nêu hai cách khắc phục.
3. Khi nào accuracy là thước đo gây hiểu lầm?

## Tóm tắt

- Dữ liệu = bảng gồm đặc trưng (đầu vào) và nhãn (cần dự đoán).
- Luôn tách train/test; học trên train, chấm trên test.
- Overfitting = học vẹt; thước đo khởi đầu: accuracy cho phân loại,
  MSE cho hồi quy.
- Bài tiếp theo: thuật toán đầu tiên — hồi quy tuyến tính.
