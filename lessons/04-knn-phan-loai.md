# Bài 04 — k-NN: phân loại bằng cách "hỏi ý kiến hàng xóm"

**Code đi kèm:** `src/aibasics/knn.py` ·
**Chạy thử:** `python examples/demo_knn.py` hoặc `aibasics demo knn`

## 1. Ý tưởng

k-NN (k-Nearest Neighbours — k hàng xóm gần nhất) hoạt động đúng như
câu "gần mực thì đen, gần đèn thì sáng": muốn biết một điểm mới thuộc
nhóm nào, hãy nhìn **k điểm đã biết gần nó nhất** và lấy nhãn chiếm đa
số.

Ví dụ: một khách hàng mới có thói quen mua sắm gần giống nhóm khách
"tiết kiệm" hơn nhóm "chi tiêu mạnh" — dự đoán khách này thuộc nhóm
tiết kiệm.

Điểm đặc biệt: k-NN gần như **không có giai đoạn huấn luyện** — nó chỉ
lưu lại toàn bộ dữ liệu học. Toàn bộ "trí tuệ" nằm ở lúc dự đoán: đo
khoảng cách tới mọi điểm đã lưu.

## 2. Khoảng cách đo thế nào?

Phổ biến nhất là **khoảng cách Euclid** — độ dài đường thẳng giữa hai
điểm (định lý Pythagoras mở rộng):

```
d(a, b) = √( (a₁−b₁)² + (a₂−b₂)² + ... )
```

Hệ quả quan trọng: nếu một đặc trưng có thang đo lớn (thu nhập tính
bằng đồng) và một đặc trưng thang nhỏ (số con), đặc trưng lớn sẽ "át"
hoàn toàn khoảng cách. Khi dùng k-NN cần **chuẩn hoá thang đo** các đặc
trưng về cùng cỡ.

## 3. Chọn k bao nhiêu?

- **k nhỏ (k = 1):** quyết định theo đúng một hàng xóm — rất nhạy với
  nhiễu, dễ overfitting (bài 02).
- **k lớn:** mượt mà hơn nhưng có thể "nuốt" cả nhóm nhỏ; k quá lớn thì
  mọi dự đoán đều thành nhãn đa số toàn cục.
- Thường chọn k lẻ (tránh hoà phiếu) và thử vài giá trị k, chọn cái tốt
  nhất trên tập kiểm tra.

## 4. Chạy thử

```python
from aibasics import KNeighborsClassifier, accuracy_score, make_blobs, train_test_split

X, y = make_blobs(n_per_cluster=25, seed=42)   # hai cụm điểm hư cấu
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)

model = KNeighborsClassifier(k=3).fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))   # ≈ 1.0
print(model.predict_one([5.8, 6.2]))                    # 1 — gần cụm 1
```

## 5. Ưu và nhược

- **Ưu:** cực kỳ dễ hiểu, không giả định dạng của quy luật, thêm dữ
  liệu mới không cần học lại.
- **Nhược:** dự đoán chậm khi dữ liệu lớn (phải đo tới mọi điểm);
  kém đi khi số đặc trưng rất nhiều ("lời nguyền số chiều" — ở không
  gian nhiều chiều, mọi điểm đều "xa nhau" như nhau).

## Tự kiểm tra

1. Vì sao cần chuẩn hoá thang đo trước khi dùng k-NN?
2. k = 1 và k = 101 khác nhau thế nào về nguy cơ overfitting?
3. k-NN "học" vào lúc nào: lúc fit hay lúc predict?

## Tóm tắt

- k-NN: lưu dữ liệu, dự đoán bằng phiếu bầu của k điểm gần nhất.
- Khoảng cách Euclid + chuẩn hoá thang đo là hai việc không thể quên.
- Bài tiếp theo: cây quyết định — mô hình ra quyết định bằng các câu
  hỏi có/không, đọc được bằng mắt thường.
