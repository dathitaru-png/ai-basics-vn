# Bài 11 — Hồi quy logistic: từ dự đoán con số sang dự đoán xác suất

**Code đi kèm:** `src/aibasics/logistic_regression.py` ·
**Chạy thử:** `python examples/demo_logistic.py` hoặc `aibasics demo logistic`

## 1. Ý tưởng: không phải "được mấy điểm" mà là "khả năng đỗ bao nhiêu phần trăm"

Bài 03 dự đoán một con số: ôn bao nhiêu giờ thì được mấy điểm.

Nhưng nhiều câu hỏi trong đời không cần con số. Chỉ cần có hoặc
không:

- Email này có phải spam không?
- Học sinh này có đỗ không?
- Bệnh nhân này có mắc bệnh không?

Với những câu hỏi đó, ta không cần biết "mấy điểm". Ta cần biết
"khả năng bao nhiêu phần trăm".

Ví dụ quen thuộc: ôn càng nhiều giờ thì khả năng đỗ càng cao. Nhưng
khả năng không bao giờ vượt quá 100%, cũng không bao giờ rơi xuống
dưới 0%. Dù bạn ôn thêm bao nhiêu giờ nữa, khả năng đỗ cũng chỉ tiến
gần tới 100% chứ không vượt qua nó.

Đường thẳng của bài 03 không làm được điều đó. Nó cứ tăng mãi, vượt
cả 100%.

Vì vậy bài này giữ lại đường thẳng của bài 03, nhưng thêm một bước
cuối: bóp đầu ra thành phần trăm.

Cách làm này có tên là hồi quy logistic (logistic regression). Bạn
cứ đọc tiếp — cái tên nghe ghê gớm vậy thôi, ý tưởng thì rất đời
thường.

## 2. Từ điểm số sang phần trăm: cái "máy bóp" sigmoid

Trước hết, đường thẳng cho ra một điểm số. Điểm này càng lớn thì
càng nghiêng về "có".

Điểm số đó có tên là `z`, tính đúng như bài 03:

```
z = w·x + b
```

Bạn ôn nhiều giờ thì `z` lớn. Bạn ôn ít giờ thì `z` nhỏ.

Giờ ta cần biến `z` thành phần trăm. Việc này do một "cái máy bóp"
đảm nhận. Nó có tên là hàm sigmoid.

Máy bóp này làm một việc duy nhất: nhận vào bất kỳ con số nào, trả
ra một phần trăm từ 0% đến 100%. Không bao giờ vượt quá 100%, không
bao giờ rơi xuống dưới 0%.

Công thức của nó là:

```
sigmoid(z) = 1 / (1 + e^(−z))
```

Bạn không cần tính tay công thức này — code lo phần đó. Chỉ cần nhớ
cách đọc nó, từng mảnh một:

- `z` là điểm số vừa nói ở trên.
- `e` là một con số cố định, khoảng 2,718. Nó chỉ là một hằng số,
  giống như số pi.
- Dấu trừ (−) đảo ngược: `z` càng lớn thì `e^(−z)` càng nhỏ.
- Cả biểu thức `1 / (1 + ...)` có tác dụng "bóp" kết quả vào khoảng
  từ 0 đến 1, tức là từ 0% đến 100%.

Và đây là cách máy bóp cư xử:

- `z = 0` → 50%. Ngay ranh giới, không nghiêng bên nào.
- `z` càng lớn → phần trăm tiến dần tới 100%, nhưng không bao giờ
  chạm 100%.
- `z` càng nhỏ → phần trăm tiến dần tới 0%, nhưng không bao giờ
  chạm 0%.

Hình dạng của nó giống chữ S nằm ngang: ở giữa dốc, hai đầu bẹt dần.
Vì vậy mô hình rất "quyết đoán" ở hai đầu (gần 0% hoặc gần 100%) và
do dự nhất ở giữa.

Cuối cùng, ta cần một vạch chia để biến phần trăm thành câu trả lời
có/không. Vạch chia mặc định là 50%:

- Từ 50% trở lên → dự đoán "có".
- Dưới 50% → dự đoán "không".

Vạch chia này có tên là ngưỡng quyết định. Nó có thể dời đi — bài 12
sẽ cho thấy dời vạch chia thì được cái này, mất cái kia.

## 3. Máy học bằng cách nào? Vẫn là "đo dốc, bước ngược dốc"

Giống bài 03, máy vẫn học bằng cách đo độ dốc rồi bước ngược hướng
dốc, lặp đi lặp lại cho đến khi dự đoán khớp thực tế. Cách học này
có tên là gradient descent. Bạn đã gặp nó ở bài 03, ở đây nó không
đổi.

Thứ duy nhất thay đổi là luật phạt — tức là cách máy tính xem mình
sai bao nhiêu.

Ở bài 03, luật phạt là MSE: đo khoảng cách giữa số dự đoán và số
thật.

Ở đây, luật phạt khác, vì ta dự đoán phần trăm chứ không phải con
số. Luật phạt mới làm theo một nguyên tắc rất đời thường: càng tự
tin mà sai thì phạt càng nặng.

Ví dụ: máy đoán 90% "có" trong khi đáp án là "không" thì bị phạt rất
nặng. Máy chỉ đoán 55% "có" mà sai thì bị phạt nhẹ hơn nhiều — vì nó
vốn đã không chắc chắn.

Luật phạt này có tên là binary cross-entropy (entropy chéo nhị
phân, còn gọi là log loss). Bạn chỉ cần nhớ nguyên tắc "tự tin mà
sai thì phạt nặng" là đủ.

Trong code đi kèm, dữ liệu cũng được chuẩn hoá trước khi học như
bài 03 (đưa mọi con số về cùng một thang đo để máy học ổn định), và
hệ số cuối cùng được quy đổi lại về thang gốc để bạn đọc được.

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

Giải thích kết quả, từng dòng một:

- Lúc đầu, loss ≈ 0.693. Đó là loss của việc "đoán bừa 50/50" cho
  mọi ví dụ — tức là máy chưa biết gì, cứ đoán mỗi bên một nửa.
- Sau khi học, loss giảm xuống ≈ 0.078. Loss giảm nghĩa là máy đã
  học được ranh giới thật của dữ liệu.
- Ôn 4 giờ → khả năng đỗ ≈ 24%. Thấp, vì 4 giờ còn dưới mốc đỗ.
- Ôn 4,5 giờ → khả năng đỗ ≈ 50%. Ngay ranh giới.
- Ôn 6 giờ → khả năng đỗ ≈ 97%. Gần như chắc đỗ.

Và điều quan trọng nhất: mô hình trả về phần trăm, không chỉ nhãn.
Bạn thấy được nó chắc chắn tới đâu, thay vì chỉ nghe một câu trả lời
cụt lủn.

## 5. Giới hạn

- **Ranh giới thẳng.** Giống hồi quy tuyến tính, mô hình chỉ tách
  được hai nhóm bằng một đường thẳng. Nếu hai nhóm xoắn xuýt vào nhau
  kiểu bàn cờ (ví dụ bài toán XOR), một đường thẳng duy nhất không
  tách nổi — nó bó tay.
- **Phần trăm chưa chắc đã "chuẩn".** Phần trăm mô hình đưa ra phản
  ánh độ tự tin của chính nó. Nó chỉ khớp với tần suất thật ngoài
  đời khi đã được kiểm chứng cẩn thận.
- **Nhạy với điểm lạ.** Vì học trên dữ liệu chuẩn hoá, một điểm dữ
  liệu quá khác biệt vẫn có thể kéo ranh giới lệch — giống bài 03.

## Tự kiểm tra

1. Vì sao không dùng thẳng đường `y = w·x + b` để dự đoán phần trăm?
   Cái "máy bóp" sigmoid sửa được vấn đề gì?
2. Một ví dụ có xác suất 0,51 và một ví dụ có xác suất 0,99 đều được
   dự đoán là "có" ở ngưỡng 0,5. Hai dự đoán này khác nhau ở điểm
   nào, và vì sao sự khác biệt đó có ích? (Gợi ý: nghĩ về độ tự
   tin.)
3. Thử dời ngưỡng quyết định từ 0,5 lên 0,8 trong đầu: số dự đoán
   "có" tăng hay giảm? Điều đó tốt hay xấu thì còn tuỳ bài toán —
   bài 12 sẽ nói rõ.

## Tóm tắt

- Hồi quy logistic = đường thẳng cho ra điểm số, rồi "máy bóp"
  sigmoid biến điểm số thành phần trăm từ 0% đến 100%.
- Máy học bằng cách đo dốc, bước ngược dốc, với luật phạt "tự tin mà
  sai thì phạt nặng".
- Ngưỡng 0,5 biến phần trăm thành nhãn có/không. Phần trăm cho biết
  mô hình chắc chắn tới đâu.
- Bài tiếp theo: accuracy thôi chưa đủ — đo classifier bằng ma trận
  nhầm lẫn, precision, recall và F1.
