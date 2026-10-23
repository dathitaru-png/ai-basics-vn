# Bài 03 — Hồi quy tuyến tính: dự đoán con số bằng một đường thẳng

**Code đi kèm:** `src/aibasics/linear_regression.py` ·
**Chạy thử:** `python examples/demo_regression.py` hoặc `aibasics demo regression`

## 1. Ý tưởng

Giả sử số giờ học và điểm thi có quan hệ gần như đường thẳng:

```
điểm ≈ w × giờ_học + b
```

Hồi quy tuyến tính (linear regression) là việc **tìm đường thẳng khớp
nhất với dữ liệu**: tìm hệ số góc `w` và hệ số chặn `b` sao cho đường
thẳng đi "gần" các điểm dữ liệu nhất. "Gần" được đo bằng **MSE** (bài
02): trung bình bình phương khoảng cách dọc từ điểm tới đường thẳng.

## 2. Máy tìm đường thẳng bằng cách nào? Gradient descent

Hãy tưởng tượng hàm lỗi (loss) là một thung lũng, và ta đang đứng trên
sườn núi trong sương mù. Cách xuống đáy: cảm nhận độ dốc dưới chân
(đạo hàm — gradient) rồi bước ngược hướng dốc. Lặp lại nhiều lần sẽ tới
đáy — nơi loss nhỏ nhất.

Với mỗi vòng lặp (epoch), thuật toán:

1. Dự đoán bằng `w`, `b` hiện tại.
2. Tính lỗi và gradient của MSE theo `w` và `b`.
3. Cập nhật: `w ← w − learning_rate × gradient`.

**Learning rate (tốc độ học)** là độ dài bước chân: quá nhỏ thì học
chậm mãi không tới, quá lớn thì nhảy qua nhảy lại, thậm chí văng khỏi
thung lũng.

Trong code đi kèm, dữ liệu được **chuẩn hoá** (đưa mỗi đặc trưng về
trung bình 0, độ lệch chuẩn 1) trước khi học, để một learning rate dùng
được cho nhiều thang đo khác nhau; hệ số cuối cùng được quy đổi lại về
thang gốc để bạn đọc được.

## 3. Chạy thử

```python
from aibasics import LinearRegression, make_regression_data

# Dữ liệu hư cấu sinh từ quy luật y = 3x + 2 (cộng nhiễu nhỏ)
X, y = make_regression_data(n=60, seed=42)
model = LinearRegression(learning_rate=0.1, epochs=800).fit(X, y)

print(model.weights, model.bias)   # ≈ [2.97], ≈ 2.03 — gần đúng (3, 2)
print(model.predict_one([4.0]))    # ≈ 13.9 (kỳ vọng 3×4 + 2 = 14)
print(model.loss_history[0], model.loss_history[-1])  # loss giảm mạnh
```

Việc loss giảm dần và hệ số học được tiến gần giá trị thật cho thấy mô
hình đã "học" đúng quy luật ẩn trong dữ liệu.

## 4. Giới hạn

- Quan hệ thật mà cong (không phải đường thẳng) thì đường thẳng khớp
  kém — cần mô hình khác hoặc biến đổi đặc trưng.
- Điểm ngoại lai (outlier) kéo đường thẳng lệch mạnh vì MSE phạt lỗi
  lớn theo bình phương.
- **Ngoại suy nguy hiểm:** đường thẳng học trên x trong [0, 10] không
  đảm bảo đúng ở x = 1000.

## Tự kiểm tra

1. Learning rate quá lớn gây hiện tượng gì khi huấn luyện?
2. Vì sao ta nói gradient descent "đi xuống thung lũng"? Đại lượng nào
   cho biết hướng đi?
3. Chạy demo với `epochs=5` rồi `epochs=800`: hệ số thay đổi thế nào?

## Tóm tắt

- Hồi quy tuyến tính: tìm `w`, `b` tối thiểu hoá MSE.
- Gradient descent: lặp lại việc "đo độ dốc — bước ngược dốc".
- Bài tiếp theo chuyển sang phân loại với k-NN — thuật toán đơn giản
  nhất có thể.
