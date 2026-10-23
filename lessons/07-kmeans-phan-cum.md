# Bài 07 — k-Means: phân cụm khi không có nhãn

**Code đi kèm:** `src/aibasics/kmeans.py` ·
**Chạy thử:** `python examples/demo_kmeans.py` hoặc `aibasics demo kmeans`

## 1. Khi không ai cho nhãn thì sao?

Các bài trước đều là **học có giám sát**: mỗi ví dụ đều kèm đáp án.
Nhưng nhiều bài toán đời thực không có đáp án sẵn — ví dụ một cửa hàng
có 10.000 khách và muốn tự tìm ra "có mấy nhóm khách điển hình?". Đây
là **học không giám sát**, và thuật toán kinh điển là **k-Means (phân
cụm k trung tâm)**.

## 2. Thuật toán trong 4 bước

Cho trước số cụm `k`:

1. **Chọn k tâm ban đầu** — code đi kèm dùng chiến lược k-means++:
   chọn tâm đầu ngẫu nhiên, các tâm sau ưu tiên điểm *xa* các tâm đã
   chọn (để các cụm khởi đầu rải đều, tránh dồn cục).
2. **Gán:** mỗi điểm thuộc về tâm gần nó nhất.
3. **Cập nhật:** mỗi tâm dời về **trung bình** (mean) của các điểm
   thuộc về nó — chữ "Means" trong tên thuật toán là đây.
4. Lặp lại bước 2–3 cho đến khi tâm không còn di chuyển.

Chất lượng phân cụm được đo bằng **inertia**: tổng bình phương khoảng
cách từ mỗi điểm tới tâm của nó — càng nhỏ cụm càng "gọn".

## 3. Chạy thử

```python
from aibasics import KMeans, make_blobs

# Ba cụm điểm hư cấu quanh (0,0), (6,6), (0,8) — máy không được biết điều này
X, y_that = make_blobs(n_per_cluster=20, centers=((0, 0), (6, 6), (0, 8)), seed=7)

model = KMeans(k=3, seed=42).fit(X)
print(model.centroids_)   # gần với ba tâm thật, thứ tự có thể khác
```

Lưu ý: số thứ tự cụm (0, 1, 2) do máy tự đặt, không mang ý nghĩa sẵn —
khớp tâm tìm được với tâm thật phải so theo vị trí, không theo số.

## 4. Khó khăn thật sự: chọn k và diễn giải

- **Chọn k:** k-Means không tự biết có mấy cụm. Cách làm phổ biến:
  chạy với nhiều k, vẽ inertia theo k và tìm "điểm khuỷu tay" (elbow)
  — chỗ mà tăng k nữa cũng không giảm inertia bao nhiêu.
- **Nhạy với thang đo:** giống k-NN (bài 04), đặc trưng thang lớn sẽ
  chi phối khoảng cách — cần chuẩn hoá.
- **Cụm không tròn thì khó:** k-Means giả định cụm dạng khối cầu quanh
  tâm; cụm hình trăng khuyết, vòng tròn đồng tâm sẽ bị cắt sai.
- **Diễn giải là việc của người:** máy chỉ gom nhóm; gọi tên và hiểu ý
  nghĩa từng cụm ("khách săn sale", "khách mua đều") là phần con người
  phải làm — và phải kiểm chứng, đừng vội tin mọi cụm máy tìm ra đều có
  ý nghĩa thật.

## Tự kiểm tra

1. Phân cụm khác phân loại ở điểm gốc rễ nào?
2. Vì sao tâm ban đầu đặt dồn cục một chỗ lại cho kết quả kém?
3. Inertia giảm khi tăng k — vậy vì sao không chọn k thật lớn luôn?

## Tóm tắt

- k-Means: lặp "gán điểm vào tâm gần nhất — dời tâm về trung bình".
- Phải tự chọn k, chuẩn hoá thang đo, và tự diễn giải cụm.
- Bài tiếp theo: viên gạch cuối — mạng nơ-ron.
