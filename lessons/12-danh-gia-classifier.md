# Bài 12 — Đánh giá classifier: accuracy thôi chưa đủ

**Code đi kèm:** `src/aibasics/metrics.py` ·
**Chạy thử:** `python examples/demo_metrics.py` hoặc `aibasics demo metrics`

## 1. Ý tưởng

Bài 02 đã cảnh báo: với 99% thư "không rác", một mô hình đoán toàn
bộ "không rác" cũng đạt accuracy 99% mà chẳng bắt được lá thư rác
nào. Accuracy — tỉ lệ đoán đúng trên tổng số — chỉ kể một nửa câu
chuyện: nó gộp chung mọi loại sai lầm vào một con số.

Để đánh giá classifier cho ra hồn, cần tách riêng các loại đúng/sai.
Bài này dùng ví dụ lọc thư rác (bài 06): nhãn "dương" (positive) là
thư rác — thứ ta muốn bắt; nhãn "âm" (negative) là thư thường.

## 2. Ma trận nhầm lẫn: bốn ô kể hết câu chuyện

Lấy 100 email hư cấu: 10 thư rác thật, 90 thư thường. Một mô hình
gắn cờ 8 email là rác, trong đó 6 đúng là rác, 2 thực ra là thư
thường; đồng thời nó bỏ sót 4 thư rác.

|  | Thực tế: rác | Thực tế: thường |
|---|---|---|
| **Máy báo: rác** | TP = 6 (bắt đúng) | FP = 2 (báo nhầm) |
| **Máy báo: thường** | FN = 4 (bỏ sót) | TN = 88 (đúng) |

- **TP (true positive):** báo rác, đúng là rác.
- **FP (false positive):** báo rác, hoá ra thư thường — "báo động giả".
- **FN (false negative):** báo thường, hoá ra là rác — "bỏ lọt".
- **TN (true negative):** báo thường, đúng là thường.

Accuracy của mô hình này là (6 + 88) / 100 = 94% — nghe rất tốt. Nhưng
nhìn vào bốn ô mới thấy: nó bỏ lọt tới 4/10 thư rác. Hai thước đo sau
đo đúng hai nỗi lo đó.

## 3. Precision và recall: hai câu hỏi khác nhau

**Precision (độ chính xác dương):** trong số những gì máy báo là rác,
bao nhiêu phần trăm đúng thật?

```
precision = TP / (TP + FP) = 6 / (6 + 2) = 75%
```

Precision thấp nghĩa là báo động giả nhiều — người dùng thấy thư quan
trọng cứ rơi vào thùng rác thì sẽ mất lòng tin.

**Recall (độ phủ, còn gọi là sensitivity):** trong số thư rác thật,
máy bắt được bao nhiêu phần trăm?

```
recall = TP / (TP + FN) = 6 / (6 + 4) = 60%
```

Recall thấp nghĩa là bỏ lọt nhiều — thùng thư đến vẫn đầy rác.

Hai thước đo này thường **đánh đổi nhau**: hạ ngưỡng quyết định (bài
11) thì máy báo rác nhiều hơn — recall tăng vì bắt được thêm rác thật,
nhưng precision giảm vì báo nhầm thêm thư thường. Không có ngưỡng nào
cho cả hai cùng 100% trừ khi dữ liệu tách biệt hoàn hảo.

## 4. F1: một con số khi cần cả hai

Khi cần một con số duy nhất mà vẫn bắt mô hình phải tốt cả hai mặt,
người ta dùng **F1** — trung bình điều hoà của precision và recall:

```
F1 = 2 × precision × recall / (precision + recall)
   = 2 × 0.75 × 0.60 / (0.75 + 0.60) ≈ 0.67
```

Vì sao là trung bình điều hoà chứ không phải trung bình cộng? Vì
trung bình cộng "tha thứ": precision 100% mà recall 0% thì trung bình
cộng vẫn 50%, nghe như tạm được — trong khi mô hình đó chẳng bắt được
gì. Trung bình điều hoà kéo mạnh về phía giá trị nhỏ hơn: một trong
hai mà tệ thì F1 tệ theo.

## 5. Chọn thước đo theo bài toán

- **Lọc thư rác:** báo nhầm thư quan trọng thành rác rất tai hại (mất
  thư) → coi trọng **precision**.
- **Sàng lọc bệnh:** bỏ sót một ca bệnh nguy hiểm hơn báo nhầm một
  người khoẻ → coi trọng **recall**.
- **So sánh hai mô hình chung chung:** dùng **F1** khi cần cả hai mặt
  đều khá.

Không có thước đo "đúng nhất" cho mọi bài toán — chọn thước đo chính
là một quyết định về việc **loại sai lầm nào đắt hơn**.

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

- Thước đo không sửa được mô hình dở: precision/recall chỉ giúp bạn
  **nhìn rõ** mô hình sai ở đâu, không tự làm nó tốt lên.
- Mọi con số trên chỉ có nghĩa khi tính trên **tập kiểm tra giữ
  riêng** (bài 02) — đo trên tập huấn luyện thì mô hình học vẹt cũng
  cho số đẹp.
- Với bài toán nhiều lớp (không chỉ hai nhãn), mỗi lớp có một bộ
  precision/recall riêng; bài này chỉ xét trường hợp hai lớp cho dễ
  hiểu.

## Tự kiểm tra

1. Một mô hình lọc rác đạt accuracy 99% trên tập có 99% thư thường.
   Nó có thể đang bỏ lọt toàn bộ thư rác không? Vì sao?
2. Precision và recall khác nhau ở mẫu số: mẫu số của precision là
   gì, của recall là gì? (Gợi ý: một cái đếm theo "máy báo", một cái
   đếm theo "thực tế".)
3. Vì sao F1 dùng trung bình điều hoà thay vì trung bình cộng?
   Cho ví dụ precision = 1.0, recall = 0.0 và tính cả hai loại trung
   bình để thấy sự khác biệt.

## Tóm tắt

- Ma trận nhầm lẫn tách sai lầm thành TP, FP, TN, FN — nhìn rõ hơn
  accuracy rất nhiều.
- Precision: trong số máy báo dương, bao nhiêu là đúng thật.
- Recall: trong số dương thật, máy bắt được bao nhiêu.
- F1: trung bình điều hoà, chỉ cao khi cả hai đều cao.
- Chọn thước đo theo loại sai lầm nào đắt hơn trong bài toán của bạn.
- Đây là bài cuối của sprint v0.2; xem ROADMAP.md để biết các bài
  tiếp theo (phụ lục toán, case study chuỗi thuật toán).
