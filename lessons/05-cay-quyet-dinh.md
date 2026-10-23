# Bài 05 — Cây quyết định: máy ra quyết định bằng câu hỏi có/không

**Code đi kèm:** `src/aibasics/decision_tree.py` ·
**Chạy thử:** `python examples/demo_decision_tree.py` hoặc `aibasics demo tree`

## 1. Ý tưởng

Hãy tưởng tượng trò chơi "20 câu hỏi": mỗi câu hỏi chia tập hợp ứng
viên làm đôi, và ta chọn câu hỏi giúp thu hẹp nhanh nhất. Cây quyết
định (decision tree) làm y hệt với dữ liệu:

```
Điểm thi >= 5?
├── Đúng -> Đạt
└── Sai  -> Chưa đạt
```

Mỗi **nút** là một câu hỏi dạng "đặc trưng X ≤ ngưỡng nào đó?", mỗi
**lá** là một dự đoán. Dự đoán cho một điểm mới = đi từ gốc xuống lá
theo các câu trả lời — con người đọc và kiểm chứng được toàn bộ logic,
đây là ưu điểm lớn nhất của cây quyết định.

## 2. Máy chọn câu hỏi bằng cách nào? Độ "lẫn lộn" Gini

Một nhóm toàn nhãn giống nhau là "thuần khiết" — không cần hỏi thêm.
Một nhóm nửa nọ nửa kia là lẫn lộn tối đa. **Độ tạp Gini** đo mức lẫn
lộn:

```
Gini = 1 − Σ (tỉ lệ của mỗi nhãn)²
```

Gini = 0 khi nhóm thuần khiết; càng gần 0.5 (với hai nhãn) càng lẫn
lộn. Khi học, cây thử mọi câu hỏi ứng viên, chấm điểm bằng Gini trung
bình có trọng số của hai nhóm con, và chọn câu hỏi làm Gini **giảm
nhiều nhất**. Sau đó lặp lại đệ quy cho từng nhóm con.

## 3. Khi nào cây ngừng lớn?

Cây mọc không giới hạn sẽ thuộc lòng từng điểm học — overfitting kinh
điển (bài 02). Hai phanh phổ biến, đều có trong code đi kèm:

- `max_depth`: độ sâu tối đa của cây.
- `min_samples_split`: nút có ít điểm hơn mức này thì không tách nữa.

Cây nông, ít lá = mô hình đơn giản, dễ hiểu, khái quát tốt hơn.

## 4. Chạy thử

```python
from aibasics import DecisionTreeClassifier

# Quy tắc ẩn: điểm >= 5 thì Đạt
X = [[float(s)] for s in range(0, 11)]
y = [1 if s >= 5 else 0 for s in range(0, 11)]

tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
print(tree.predict_one([7.0]))   # 1 — Đạt
print(tree.predict_one([3.0]))   # 0 — Chưa đạt
```

Trong `examples/demo_decision_tree.py` là ví dụ hai đặc trưng (giờ học
+ số buổi nghỉ) — cây tự tìm ra câu hỏi tốt mà không cần ta chỉ định.

## 5. Ưu và nhược

- **Ưu:** dễ giải thích nhất trong các mô hình; không cần chuẩn hoá
  thang đo; xử lý được quan hệ phi tuyến theo từng bậc.
- **Nhược:** cây đơn lẻ dễ overfitting và "mong manh" — đổi dữ liệu
  một chút cây có thể khác hẳn. Đây là lý do thực tế người ta dùng
  **rừng ngẫu nhiên (random forest)**: hàng trăm cây bỏ phiếu chung —
  ý tưởng bỏ phiếu giống k-NN ở bài 04.

## Tự kiểm tra

1. Gini của nhóm gồm 8 nhãn A và 2 nhãn B bằng bao nhiêu? (Gợi ý:
   1 − 0.8² − 0.2².)
2. Vì sao cây rất sâu dễ overfitting?
3. Nêu một tình huống thật mà "đọc được logic của mô hình" quan trọng
   hơn vài phần trăm độ chính xác.

## Tóm tắt

- Cây quyết định: chuỗi câu hỏi có/không; chọn câu hỏi giảm Gini nhất.
- Kiểm soát độ sâu là cách chính chống overfitting.
- Bài tiếp theo: Naive Bayes — phân loại văn bản bằng xác suất.
