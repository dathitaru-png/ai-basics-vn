# Bài 06 — Naive Bayes: phân loại văn bản bằng xác suất

**Code đi kèm:** `src/aibasics/naive_bayes.py` ·
**Chạy thử:** `python examples/demo_naive_bayes.py` hoặc `aibasics demo bayes`

## 1. Bài toán: lọc tin nhắn rác

Cho một tin nhắn mới, làm sao biết nó là **spam** (rác) hay **ham**
(bình thường)? Con người nhìn vào các từ: "trúng thưởng", "miễn phí",
"khuyến mãi sốc" → nghi là rác; "họp nhóm", "thư viện" → chắc bình
thường. Naive Bayes biến trực giác đó thành phép tính xác suất.

## 2. Định lý Bayes trong một câu

```
P(nhãn | nội dung) tỉ lệ với P(nhãn) × tích các P(từ | nhãn)
```

Đọc xuôi: khả năng tin nhắn là spam = mức độ phổ biến của spam nói
chung × với mỗi từ xuất hiện, từ đó "hay gặp trong spam đến mức nào"
so với "hay gặp trong tin thường đến mức nào". Từ "miễn phí" xuất
hiện trong 6/8 tin spam nhưng 0/8 tin thường là bằng chứng rất mạnh.

Phần **"naive" (ngây thơ)** nằm ở giả định các từ **độc lập** với nhau
— ngoài đời không đúng ("New" và "York" hay đi cùng nhau), nhưng mô
hình vẫn chạy tốt đáng ngạc nhiên cho phân loại văn bản.

Chi tiết kỹ thuật trong code:

- **Làm mịn Laplace (alpha):** cộng thêm một lượng nhỏ vào mọi từ, để
  từ chưa từng xuất hiện không làm xác suất của cả lớp thành 0.
- **Tính bằng log:** nhân nhiều xác suất nhỏ sẽ tràn số (underflow),
  nên code cộng log-xác suất thay vì nhân xác suất.

## 3. Dữ liệu minh hoạ

`aibasics.data.make_text_dataset()` chứa 16 tin nhắn **hư cấu** viết
riêng cho bài học (8 spam, 8 ham) — không lấy từ hộp thư thật của bất
kỳ ai.

## 4. Chạy thử

```python
from aibasics import MultinomialNaiveBayes, make_text_dataset

texts, labels = make_text_dataset()
model = MultinomialNaiveBayes().fit(texts, labels)

print(model.predict_one("Trúng thưởng lớn, nhận quà miễn phí ngay!"))  # spam
print(model.predict_one("Chiều mai họp nhóm ở thư viện nhé bạn."))    # ham
print(model.predict_proba("Khuyến mãi giảm giá sốc hôm nay"))         # {'ham': ~0.0, 'spam': ~1.0}
```

## 5. Ưu và nhược

- **Ưu:** huấn luyện và dự đoán rất nhanh; hoạt động tốt với dữ liệu
  học ít; dễ triển khai (bộ lọc spam đời đầu của nhiều hệ thống email
  dùng chính kỹ thuật này).
- **Nhược:** giả định độc lập giữa các từ là sai về lý thuyết; không
  hiểu ngữ cảnh, phủ định ("không miễn phí") hay châm biếm.

## Tự kiểm tra

1. Vì sao cần làm mịn Laplace? Điều gì xảy ra nếu một từ trong tin mới
   chưa từng xuất hiện ở lớp spam?
2. Giả định "ngây thơ" của Naive Bayes là gì?
3. Vì sao code cộng log-xác suất thay vì nhân xác suất trực tiếp?

## Tóm tắt

- Naive Bayes chấm điểm mỗi lớp bằng xác suất tiên nghiệm nhân với
  "bằng chứng" của từng từ.
- Ngây thơ nhưng nhanh và hiệu quả cho văn bản ngắn.
- Bài tiếp theo: học khi *không có nhãn* — phân cụm bằng k-Means.
