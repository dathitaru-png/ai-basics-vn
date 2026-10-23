# Bài 08 — Mạng nơ-ron nhỏ: từ perceptron tới MLP

**Code đi kèm:** `src/aibasics/perceptron.py` ·
**Chạy thử:** `python examples/demo_perceptron.py` hoặc `aibasics demo perceptron` / `aibasics demo mlp`

## 1. Một nơ-ron nhân tạo làm gì?

Lấy cảm hứng (rất thô sơ) từ neuron sinh học, một **perceptron** làm
ba việc:

1. Nhân mỗi đầu vào với một **trọng số (weight)** rồi cộng thêm **bias**.
2. Cộng tất cả thành một điểm số `z = w₁x₁ + w₂x₂ + ... + b`.
3. Qua **hàm kích hoạt**: nếu `z >= 0` dự đoán 1, ngược lại dự đoán 0.

Học = mỗi lần đoán sai, nudge các trọng số một chút theo hướng sửa lỗi:

```
w ← w + tốc độ_học × (đáp án đúng − dự đoán) × x
```

## 2. Perceptron học được AND nhưng bó tay với XOR

| x₁ | x₂ | AND | XOR |
|----|----|-----|-----|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |

Vẽ bốn điểm lên mặt phẳng: AND tách được bằng **một đường thẳng** (ba
điểm 0 một phía, điểm 1 một phía). XOR thì không — hai điểm 1 nằm chéo
nhau, không đường thẳng nào chia đúng. Perceptron chỉ vẽ được đúng
một đường thẳng nên XOR là giới hạn cứng của nó. Chạy demo để thấy tận
mắt: perceptron học AND hoàn hảo và thất bại đúng ở XOR.

## 3. Thêm một lớp ẩn: MLP

**MLP (multi-layer perceptron)** xếp thêm một **lớp ẩn (hidden layer)**
giữa đầu vào và đầu ra. Mỗi nơ-ron ẩn vẽ một đường thẳng riêng; lớp
đầu ra kết hợp các đường đó lại — đủ sức bao quanh vùng chéo của XOR.

Hai thay đổi kỹ thuật quan trọng trong `TinyMLP` của gói code:

- Hàm kích hoạt trơn **tanh** cho lớp ẩn và **sigmoid** cho đầu ra,
  thay cho bước nhảy bậc thang — nhờ đó tính được đạo hàm.
- **Lan truyền ngược (backpropagation):** lấy gradient descent của
  bài 03 áp dụng qua nhiều lớp — lỗi ở đầu ra được "chia trách nhiệm"
  ngược về từng trọng số theo quy tắc dây chuyền (chain rule). Đây
  chính là cơ chế huấn luyện của mọi mạng nơ-ron hiện đại, chỉ khác
  quy mô.

```python
from aibasics import TinyMLP

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [0, 1, 1, 0]  # XOR

mlp = TinyMLP(hidden_dim=4, epochs=4000, seed=42).fit(X, y)
print(mlp.predict(X))   # [0, 1, 1, 0] — học được!
```

## 4. Từ mạng nhỏ tới deep learning

Mạng trong sản phẩm thật khác bản đồ chơi này ở quy mô, không khác
gốc rễ: nhiều lớp hơn (deep = sâu), hàng triệu đến hàng tỷ trọng số, dữ
liệu khổng lồ, và phần cứng GPU/TPU để huấn luyện. Hiểu perceptron +
backpropagation là đã nắm được "động cơ" của deep learning — và của
cả các mô hình ngôn ngữ lớn trong bài tiếp theo.

## Tự kiểm tra

1. Vì sao perceptron không học được XOR? Vẽ hình để tự giải thích.
2. Lớp ẩn giúp ích gì mà lớp đầu ra không tự làm được?
3. Backpropagation dùng lại ý tưởng nào của bài 03?

## Tóm tắt

- Perceptron: tổng có trọng số + ngưỡng; chỉ học được bài toán tách
  tuyến tính.
- MLP thêm lớp ẩn + backpropagation → học được XOR và các mẫu cong.
- Bài tiếp theo: bước nhảy tới mô hình ngôn ngữ lớn và cách dùng
  prompt hiệu quả.
