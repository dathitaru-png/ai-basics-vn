# Bài 11 — Hồi quy logistic: từ dự đoán con số sang dự đoán xác suất

**Code đi kèm:** `src/aibasics/logistic_regression.py` ·
**Chạy thử:** `python examples/demo_logistic.py` hoặc `aibasics demo logistic`

## 1. Ý tưởng

Bài 03 dự đoán một con số (điểm thi) bằng đường thẳng `y ≈ w·x + b`.
Nhưng nhiều bài toán cần câu trả lời có/không: email này có phải spam
không? Học sinh này có đỗ không? Bệnh nhân này có mắc bệnh không?

Hồi quy logistic (logistic regression) tái sử dụng đúng đường thẳng
đó, nhưng thêm một bước: **bóp đầu ra thành xác suất** từ 0 đến 1.
Đường thẳng cho ra một điểm số `z = w·x + b` (càng lớn càng nghiêng
về "có"); xác suất là cách đọc điểm số đó theo ngôn ngữ "khả năng
bao nhiêu phần trăm".

Ví dụ: số giờ ôn tập càng nhiều, xác suất đỗ càng cao — nhưng xác
suất không bao giờ vượt quá 100% hay rơi xuống dưới 0%, dù giờ ôn
tăng mãi. Đường thẳng thì không làm được điều đó: nó cứ tăng mãi.

## 2. Hàm sigmoid: chiếc cầu từ điểm số sang xác suất

Hàm sigmoid biến bất kỳ số nào thành một số trong khoảng (0, 1):

```
sigmoid(z) = 1 / (1 + e^(−z))
```

- `z = 0` → xác suất 0.5 (ngay ranh giới, không nghiêng bên nào).
- `z` càng lớn → xác suất tiến dần tới 1, nhưng không bao giờ chạm 1.
- `z` càng nhỏ → xác suất tiến dần tới 0, nhưng không bao giờ chạm 0.

Hình dạng của nó là chữ S nằm ngang: ở giữa dốc, hai đầu bẹt dần.
Vì vậy mô hình rất "quyết đoán" ở hai đầu (gần 0 hoặc gần 1) và do dự
nhất ở giữa. **Ngưỡng quyết định** mặc định là 0.5: xác suất ≥ 0.5 thì
dự đoán "có", dưới 0.5 thì dự đoán "không". Ngưỡng này có thể dời đi
— bài 12 sẽ cho thấy việc dời ngưỡng đánh đổi thứ gì lấy thứ gì.

## 3. Máy học bằng cách nào? Vẫn là gradient descent

Giống bài 03, máy vẫn "đo độ dốc — bước ngược dốc" (gradient descent).
Thứ duy nhất thay đổi là **hàm lỗi**: thay vì MSE, hồi quy logistic
dùng **binary cross-entropy** (entropy chéo nhị phân, còn gọi là log
loss). Ý tưởng của nó rất trực quan: mô hình càng tự tin mà sai thì
càng bị phạt nặng. Đoán 90% "có" trong khi đáp án là "không" thì lỗi
lớn hơn nhiều so với đoán 55% "có" mà sai.

Trong code đi kèm, dữ liệu cũng được **chuẩn hoá** trước khi học như
bài 03, và hệ số cuối cùng được quy đổi lại về thang gốc để bạn đọc
được.

## 4. Chạy thử

Dữ liệu hư cấu: 8 học sinh, số giờ ôn tập từ 1 đến 8, đỗ khi ôn từ
5 giờ trở lên (quy luật tự đặt ra để minh hoạ, không phải số liệu
thật):

```python
from aibasics import LogisticRegression

# Dữ liệu hư cấu: giờ ôn -> đỗ (1) / trượt (0)
X = [[float(h)] for h in range(1, 9)]
y = [0, 0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X, y)
print(model.weights, model.bias)   # ≈ [2.28], ≈ -10.27
print(model.loss_history[0], model.loss_history[-1])  # 0.693 -> 0.078

print(model.predict_proba_one([4.0]))   # ≈ 0.24 — ôn 4 giờ, khả năng đỗ thấp
print(model.predict_proba_one([4.5]))   # ≈ 0.50 — ngay ranh giới
print(model.predict_proba_one([6.0]))   # ≈ 0.97 — ôn 6 giờ, gần như chắc đỗ
print(model.predict_one([4.0]))         # 0 — dưới ngưỡng 0.5 nên đoán "trượt"
```

Lưu ý: loss ban đầu ≈ 0.693 — đó là loss của việc "đoán bừa 50/50"
cho mọi ví dụ. Loss giảm xuống ≈ 0.078 cho thấy mô hình đã học được
ranh giới thật của dữ liệu. Và quan trọng: mô hình trả về **xác
suất**, không chỉ nhãn — bạn thấy được nó chắc chắn tới đâu, thay vì
chỉ nghe một câu trả lời cụt lủn.

## 5. Giới hạn

- **Ranh giới thẳng:** giống hồi quy tuyến tính, mô hình chỉ tách được
  hai nhóm bằng một đường thẳng (trong không gian nhiều chiều là một
  mặt phẳng). Dữ liệu xoắn xuýt kiểu XOR thì nó bó tay.
- **Xác suất chưa chắc đã "chuẩn":** xác suất mô hình đưa ra phản ánh
  độ tự tin của chính nó, không phải tần suất thật ngoài đời, trừ khi
  đã được kiểm chứng cẩn thận.
- **Nhạy với thang đo và ngoại lai:** vì dùng gradient descent trên
  dữ liệu chuẩn hoá, điểm ngoại lai vẫn có thể kéo ranh giới lệch —
  giống bài 03.

## Tự kiểm tra

1. Vì sao không dùng thẳng đường `y = w·x + b` để dự đoán xác suất?
   Sigmoid sửa được vấn đề gì?
2. Một ví dụ có xác suất 0.51 và một ví dụ có xác suất 0.99 đều được
   dự đoán là "có" ở ngưỡng 0.5. Hai dự đoán này khác nhau ở điểm
   nào, và vì sao sự khác biệt đó có ích?
3. Thử dời ngưỡng quyết định từ 0.5 lên 0.8 trong đầu: số dự đoán
   "có" tăng hay giảm? Điều đó tốt hay xấu thì còn tuỳ bài toán —
   bài 12 sẽ nói rõ.

## Tóm tắt

- Hồi quy logistic = điểm số tuyến tính `z = w·x + b` qua hàm sigmoid
  thành xác suất trong (0, 1).
- Học bằng gradient descent với hàm lỗi binary cross-entropy: tự tin
  mà sai thì phạt nặng.
- Ngưỡng 0.5 biến xác suất thành nhãn; xác suất cho biết mô hình chắc
  chắn tới đâu.
- Bài tiếp theo: accuracy thôi chưa đủ — đo classifier bằng ma trận
  nhầm lẫn, precision, recall và F1.
