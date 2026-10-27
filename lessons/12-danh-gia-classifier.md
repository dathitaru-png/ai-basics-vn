# Bài 12 — Đánh giá classifier: accuracy thôi chưa đủ

**Code đi kèm:** `src/aibasics/metrics.py` ·
**Chạy thử:** `python examples/demo_metrics.py` hoặc `aibasics demo metrics`

## 1. Ý tưởng: một con số "đúng bao nhiêu phần trăm" có thể đánh lừa

Bài 02 đã cảnh báo: giả sử 99% thư là "thư thường", chỉ 1% là thư
rác. Một mô hình lười biếng đoán tất cả là "thư thường" cũng đạt
accuracy 99%. Nghe thì giỏi, nhưng thực ra nó chẳng bắt được lá thư
rác nào.

Accuracy — tỉ lệ đoán đúng trên tổng số — chỉ kể một nửa câu chuyện.
Nó gộp chung mọi loại sai lầm vào một con số.

Muốn đánh giá cho ra hồn, ta phải tách riêng các loại đúng và sai.
Bài này dùng ví dụ lọc thư rác (bài 06).

Trước hết, ta gọi tên hai phía cho gọn:

- Thư rác là thứ ta muốn bắt. Ta gọi nó là "dương" (positive).
- Thư thường là thứ ta muốn giữ lại. Ta gọi nó là "âm" (negative).

"Dương" và "âm" ở đây chỉ là tên gọi, không có nghĩa tốt hay xấu.

## 2. Bốn ô kể hết câu chuyện: bắt đúng, báo nhầm, bỏ lọt, hay đúng thật?

Hãy tưởng tượng máy lọc thư như một người gác cổng. Với mỗi email,
người gác cổng hoặc gắn cờ "rác", hoặc cho qua là "thường". Mỗi
quyết định có thể đúng hoặc sai. Tất cả chỉ có 4 trường hợp.

Lấy 100 email hư cấu: 10 thư rác thật, 90 thư thường. Một mô hình
gắn cờ 8 email là rác, trong đó 6 đúng là rác, 2 thực ra là thư
thường; đồng thời nó bỏ sót 4 thư rác.

|  | Thực tế: rác | Thực tế: thường |
|---|---|---|
| **Máy báo: rác** | TP = 6 (bắt đúng) | FP = 2 (báo nhầm) |
| **Máy báo: thường** | FN = 4 (bỏ sót) | TN = 88 (đúng) |

Giải thích từng ô bằng lời thường:

- **Bắt đúng:** máy báo rác, đúng là rác. Có 6 email như vậy.
- **Báo nhầm:** máy báo rác, hoá ra là thư thường. Người ta còn gọi
  là "báo động giả". Có 2 email như vậy.
- **Bỏ lọt:** máy báo thường, hoá ra là rác. Có 4 email như vậy.
- **Đúng là thường:** máy báo thường, đúng là thường. Có 88 email
  như vậy.

Bốn ô này có tên viết tắt:

- Bắt đúng = TP (true positive).
- Báo nhầm = FP (false positive).
- Bỏ lọt = FN (false negative).
- Đúng là thường = TN (true negative).

Đừng cố nhớ tên viết tắt ngay. Chỉ cần nhớ 4 trường hợp bằng lời
thường là đủ đọc tiếp.

Giờ tính accuracy của mô hình này: số đoán đúng là 6 + 88 = 94, trên
tổng 100 email. Accuracy = 94%. Nghe rất tốt.

Nhưng nhìn vào bốn ô mới thấy vấn đề: nó bỏ lọt tới 4 trên 10 thư
rác. Gần một nửa số rác vẫn lọt vào hộp thư đến.

Hai thước đo sau đo đúng hai nỗi lo đó.

## 3. Hai câu hỏi khác nhau: "báo có chuẩn không" và "bắt được bao nhiêu"

### Câu hỏi 1: trong số những gì máy báo là rác, bao nhiêu phần trăm đúng thật?

Đây là câu hỏi của người dùng: "Mỗi lần máy báo rác, tôi tin được
bao nhiêu?"

```
precision = TP / (TP + FP) = 6 / (6 + 2) = 75%
```

Hãy đọc công thức từng mảnh:

- Tử số TP = 6: số email bắt đúng.
- Mẫu số TP + FP = 6 + 2 = 8: tổng số email máy đã báo là rác.
- 6 trên 8 email máy báo là rác đúng thật → 75%.

Thước đo này có tên là precision (độ chính xác dương).

Precision thấp nghĩa là báo động giả nhiều. Người dùng thấy thư quan
trọng cứ rơi vào thùng rác thì sẽ mất lòng tin.

### Câu hỏi 2: trong số thư rác thật, máy bắt được bao nhiêu phần trăm?

Đây là câu hỏi của người dọn hộp thư: "Rác thật ngoài kia, máy bắt
được bao nhiêu?"

```
recall = TP / (TP + FN) = 6 / (6 + 4) = 60%
```

Đọc công thức từng mảnh:

- Tử số TP = 6: số email bắt đúng.
- Mẫu số TP + FN = 6 + 4 = 10: tổng số thư rác thật.
- 6 trên 10 thư rác thật bị bắt → 60%.

Thước đo này có tên là recall (độ phủ, còn gọi là sensitivity).

Recall thấp nghĩa là bỏ lọt nhiều. Thùng thư đến vẫn đầy rác.

Chú ý sự khác nhau: precision đếm theo "máy đã báo", recall đếm
theo "thực tế có". Cùng tử số là 6, nhưng mẫu số khác nhau nên kết
quả khác nhau.

Hai thước đo này thường đánh đổi nhau. Hãy tưởng tượng ta hạ vạch
chia (ngưỡng quyết định ở bài 11) xuống thấp: máy báo rác nhiều hơn.
Nhờ đó nó bắt được thêm rác thật — recall tăng. Nhưng đồng thời nó
báo nhầm thêm thư thường — precision giảm.

Không có ngưỡng nào cho cả hai cùng 100%, trừ khi dữ liệu tách biệt
hoàn hảo.

## 4. Một con số khi cần cả hai: F1

Có lúc ta cần một con số duy nhất để so sánh nhanh hai mô hình,
nhưng vẫn muốn bắt mô hình phải tốt cả hai mặt. Lúc đó người ta dùng
F1.

F1 là một loại trung bình của precision và recall. Nhưng không phải
trung bình cộng thông thường. Nó là loại trung bình "kéo về phía số
nhỏ hơn": chỉ cần một trong hai tệ thì kết quả cũng tệ theo.

Loại trung bình này có tên là trung bình điều hoà. Công thức của nó
là:

```
F1 = 2 × precision × recall / (precision + recall)
   = 2 × 0.75 × 0.60 / (0.75 + 0.60) ≈ 0.67
```

Vì sao không dùng trung bình cộng cho tiện? Hãy thử một ví dụ cực
đoan: precision 100% mà recall 0% — tức là máy báo rất chuẩn, nhưng
chẳng bắt được gì cả.

- Trung bình cộng: (100% + 0%) / 2 = 50%. Nghe như tạm được — trong
  khi mô hình đó vô dụng.
- F1 (trung bình điều hoà): 2 × 1,0 × 0,0 / (1,0 + 0,0) = 0. Đúng
  với thực tế: mô hình chẳng bắt được gì thì điểm phải bằng 0.

Đó là lý do F1 dùng trung bình điều hoà: nó không "tha thứ" cho mặt
yếu.

## 5. Chọn thước đo theo bài toán: loại sai lầm nào đắt hơn?

Không có thước đo "đúng nhất" cho mọi bài toán. Chọn thước đo chính
là quyết định xem loại sai lầm nào đắt hơn trong bài toán của bạn.

- **Lọc thư rác:** báo nhầm thư quan trọng thành rác rất tai hại, vì
  bạn có thể mất thư. Vì vậy coi trọng precision — báo thì phải
  chuẩn.
- **Sàng lọc bệnh:** bỏ sót một ca bệnh nguy hiểm hơn nhiều so với
  báo nhầm một người khoẻ. Vì vậy coi trọng recall — thà báo nhầm
  còn hơn bỏ lọt.
- **So sánh hai mô hình chung chung:** dùng F1 khi cần cả hai mặt
  đều khá.

## 6. Chạy thử

```python
from aibasics import accuracy_score
from aibasics.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 100 email hư cấu: 10 rác thật, 90 thường; máy gắn cờ 8 email là rác
y_true = [1] * 10 + [0] * 90
y_pred = [1] * 6 + [0] * 4 + [1] * 2 + [0] * 88

print(confusion_matrix(y_true, y_pred))   # {'tp': 6, 'fp': 2, 'tn': 88, 'fn': 4}
print(accuracy_score(y_true, y_pred))     # 0.94 — nghe rất tốt...
print(precision_score(y_true, y_pred))    # 0.75
print(recall_score(y_true, y_pred))       # 0.60 — ...nhưng bỏ lọt 40% số rác
print(f1_score(y_true, y_pred))           # ≈ 0.67
```

## 7. Giới hạn

- **Thước đo không sửa được mô hình dở.** Precision và recall chỉ
  giúp bạn nhìn rõ mô hình sai ở đâu. Chúng không tự làm mô hình tốt
  lên.
- **Mọi con số trên chỉ có nghĩa khi tính trên tập kiểm tra giữ
  riêng** (bài 02). Đo trên tập huấn luyện thì mô hình học vẹt cũng
  cho số đẹp.
- **Với bài toán nhiều lớp** (không chỉ hai nhãn), mỗi lớp có một bộ
  precision/recall riêng. Bài này chỉ xét trường hợp hai lớp cho dễ
  hiểu.

## Tự kiểm tra

1. Một mô hình lọc rác đạt accuracy 99% trên tập có 99% thư thường.
   Nó có thể đang bỏ lọt toàn bộ thư rác không? Vì sao? (Gợi ý: thử
   tưởng tượng mô hình đoán tất cả là "thư thường".)
2. Precision và recall khác nhau ở mẫu số. Mẫu số của precision đếm
   theo cái gì, của recall đếm theo cái gì? (Gợi ý: một cái đếm theo
   "máy đã báo", một cái đếm theo "thực tế có".)
3. Vì sao F1 dùng trung bình điều hoà thay vì trung bình cộng? Lấy ví
   dụ precision = 1,0 và recall = 0,0, tính cả hai loại trung bình
   rồi so sánh.

## Tóm tắt

- Bốn ô (bắt đúng, báo nhầm, bỏ lọt, đúng là thường) tách sai lầm ra
  rõ hơn accuracy rất nhiều.
- Precision: trong số máy báo là rác, bao nhiêu phần trăm đúng thật.
- Recall: trong số rác thật, máy bắt được bao nhiêu phần trăm.
- F1: trung bình điều hoà, chỉ cao khi cả hai đều cao.
- Chọn thước đo theo loại sai lầm nào đắt hơn trong bài toán của
  bạn.
- Đây là bài cuối của sprint v0.2; xem ROADMAP.md để biết các bài
  tiếp theo (phụ lục toán, case study chuỗi thuật toán).
