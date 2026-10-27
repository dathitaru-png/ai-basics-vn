# AI Basics VN

**Basic AI knowledge for absolute beginners — in Vietnamese.**

AI Basics VN is an open learning kit with two halves:

1. A **structured beginner curriculum in Vietnamese** (the `lessons/`
   folder): twelve short lessons that explain AI, machine learning, deep
   learning and LLMs in plain language with everyday examples.
2. **From-scratch, dependency-free Python implementations** of the
   classic algorithms behind the lessons — linear regression, logistic
   regression, k-NN, decision trees, Naive Bayes, k-Means, a perceptron
   and a tiny MLP — plus classifier evaluation (confusion matrix,
   precision, recall, F1) — so learners can read, run and modify real
   code with nothing but Python installed.

> Tóm tắt: Bộ tài liệu "Kiến thức AI cơ bản cho người mới" bằng tiếng
> Việt, kèm code Python thuần (không thư viện ngoài) cài đặt lại các
> thuật toán học máy kinh điển để bạn vừa đọc lý thuyết vừa chạy thử
> được ngay.

## Lộ trình học (Learning path)

| Bài | Chủ đề | Code đi kèm |
|---|---|---|
| [01](lessons/01-ai-la-gi.md) | AI là gì? AI / ML / DL / GenAI khác nhau thế nào | — |
| [02](lessons/02-du-lieu-va-cach-may-hoc.md) | Dữ liệu, train/test, overfitting, thước đo | `aibasics.data` |
| [03](lessons/03-hoi-quy-tuyen-tinh.md) | Hồi quy tuyến tính & gradient descent | `linear_regression.py` |
| [04](lessons/04-knn-phan-loai.md) | k-NN — phân loại theo hàng xóm | `knn.py` |
| [05](lessons/05-cay-quyet-dinh.md) | Cây quyết định & độ tạp Gini | `decision_tree.py` |
| [06](lessons/06-naive-bayes-van-ban.md) | Naive Bayes — lọc tin nhắn rác | `naive_bayes.py` |
| [07](lessons/07-kmeans-phan-cum.md) | k-Means — phân cụm không nhãn | `kmeans.py` |
| [08](lessons/08-mang-no-ron-nho.md) | Mạng nơ-ron nhỏ: perceptron & MLP | `perceptron.py` |
| [09](lessons/09-llm-va-prompt-co-ban.md) | LLM & prompt cơ bản (không cần API key) | — |
| [10](lessons/10-dao-duc-va-gioi-han-ai.md) | Đạo đức & giới hạn của AI | — |
| [11](lessons/11-hoi-quy-logistic.md) | Hồi quy logistic & hàm sigmoid | `logistic_regression.py` |
| [12](lessons/12-danh-gia-classifier.md) | Đánh giá classifier: ma trận nhầm lẫn, precision, recall, F1 | `metrics.py` |

Tài liệu phụ: [Thuật ngữ EN–VI](lessons/thuat-ngu.md) ·
[Câu hỏi ôn tập (có đáp án)](lessons/cau-hoi-on-tap.md)

## Bắt đầu nhanh (Quick start)

Yêu cầu: Python 3.10 trở lên. Không cần cài thêm thư viện nào khác.

```bash
pip install -e .
aibasics demo regression     # chạy thử demo bài 03
aibasics demo all            # chạy mọi demo
python examples/demo_knn.py  # hoặc chạy từng file trong examples/
```

```python
from aibasics import KNeighborsClassifier, make_blobs, train_test_split, accuracy_score

X, y = make_blobs(n_per_cluster=25, seed=42)        # dữ liệu hư cấu
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)
model = KNeighborsClassifier(k=3).fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))  # ≈ 1.0
```

Chạy kiểm thử:

```bash
pip install -e ".[test]"
python -m pytest
```

## Nguyên tắc của dự án

- **Dễ đọc hơn tối ưu.** Code viết cho người mới học: ít dòng, đặt tên
  rõ, không thủ thuật khó hiểu. Đây là tài liệu học, không phải thư
  viện dùng cho sản phẩm thật.
- **Không phụ thuộc.** Toàn bộ code chỉ dùng thư viện chuẩn của Python.
- **Dữ liệu hư cấu, ghi rõ hư cấu.** Mọi ví dụ và bộ dữ liệu minh hoạ
  đều là dữ liệu tổng hợp được sinh ra cho việc học — không có dữ liệu
  thật của bất kỳ ai.
- **Trung thực về giới hạn.** Các thuật toán ở đây là bản rút gọn cho
  việc học; bài học nào cũng nêu rõ điểm yếu của thuật toán.

## Trạng thái dự án (Project status)

AI Basics VN là dự án **mới ra mắt — v0.1.0, tháng 10/2026**, đang
phát triển theo ROADMAP (xem mục Unreleased trong CHANGELOG). Nội dung
hiện gồm 12 bài học, 2 tài liệu phụ, 9 module thuật toán và thước đo,
các ví dụ chạy được và bộ kiểm thử tự động; CI chạy trên Python
3.10–3.12. Dự án chưa có số liệu sử dụng nào (không lượt tải, không
học viên) — chúng tôi không kê khai con số chưa tồn tại. Xem
[ROADMAP.md](ROADMAP.md) cho kế hoạch tiếp theo và
[CONTRIBUTING.md](CONTRIBUTING.md) nếu bạn muốn đóng góp bài học, bản
dịch hoặc sửa lỗi.

## License

MIT — see [LICENSE](LICENSE).
