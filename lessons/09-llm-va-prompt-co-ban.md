# Bài 09 — LLM và prompt cơ bản: trò chuyện với mô hình ngôn ngữ lớn

**Bài này không cần code và không cần API key** — chỉ cần một chatbot
bất kỳ (ChatGPT, Gemini, Claude...) để thực hành khi đọc.

## 1. LLM là gì?

**LLM (Large Language Model — mô hình ngôn ngữ lớn)** là một mạng
nơ-ron rất lớn (bài 08) được huấn luyện trên khối lượng văn bản khổng
lồ với một nhiệm vụ nghe đơn giản đến bất ngờ: **đoán từ tiếp theo**.

Đoán từ tiếp theo ở quy mô đủ lớn buộc mô hình phải học ngữ pháp, kiến
thức, cách lập luận bề mặt và cả phong cách — từ đó nảy sinh khả năng
trả lời câu hỏi, tóm tắt, dịch, viết code... mà không ai lập trình
riêng từng kỹ năng.

## 2. Ba khái niệm cần nắm

**Token** — đơn vị văn bản mà mô hình xử lý. Token không phải là từ:
một token có thể là một từ ngắn, một phần của từ dài, hoặc dấu câu.
Tiếng Anh trung bình 1 token ≈ ¾ từ; tiếng Việt thường tốn nhiều token
hơn cho cùng một câu vì dấu và âm tiết bị tách nhỏ. Giới hạn độ dài của
mô hình luôn tính bằng token.

**Context window (cửa sổ ngữ cảnh)** — tổng số token mô hình "nhìn
thấy" cùng lúc, gồm cả câu hỏi của bạn và câu trả lời của nó. Nội dung
nằm ngoài cửa sổ này (phần đầu của cuộc trò chuyện rất dài, tài liệu
không được đưa vào) thì mô hình **không biết** — nó sẽ không tự nhận
là không biết trong mọi trường hợp, nên hãy chủ động nhắc lại thông
tin quan trọng.

**Temperature (nhiệt độ)** — mức "ngẫu hứng" khi chọn từ tiếp theo:

- Thấp (gần 0): chọn từ chắc chắn nhất — ổn định, hợp việc cần chính
  xác, tra cứu, phân loại.
- Cao (gần 1 trở lên): đa dạng, sáng tạo hơn nhưng dễ lạc đề và sai.

## 3. Viết prompt hiệu quả: 5 nguyên tắc

1. **Nói rõ nhiệm vụ và định dạng đầu ra.** "Tóm tắt văn bản sau
   thành đúng 3 gạch đầu dòng, mỗi dòng dưới 15 từ" tốt hơn nhiều
   "tóm tắt giúp tôi".
2. **Cho ngữ cảnh cần thiết.** Người đọc là ai, mục đích gì, giới hạn
   nào. Mô hình không đọc được suy nghĩ của bạn.
3. **Cho ví dụ mẫu (few-shot).** Một, hai ví dụ đầu vào → đầu ra đúng ý
   thường hiệu quả hơn mô tả dài.
4. **Chia việc lớn thành bước nhỏ.** Bảo mô hình "liệt kê ý chính
   trước, sau đó mới viết bài" cho kết quả kiểm soát được hơn một yêu
   cầu khổng lồ.
5. **Kiểm chứng đầu ra.** Đặc biệt với số liệu, trích dẫn, tên riêng:
   yêu cầu mô hình nêu nguồn/cách tính, và tự kiểm tra lại bằng nguồn
   độc lập (bài 10 nói rõ vì sao).

Một khung prompt gọn để tái sử dụng:

```
Vai trò: bạn là [trợ lý/biên tập viên/...].
Nhiệm vụ: [việc cụ thể].
Ngữ cảnh: [thông tin nền cần thiết].
Định dạng: [độ dài, cấu trúc, giọng văn].
Ràng buộc: [điều phải tránh].
Nội dung: [văn bản/dữ liệu đầu vào].
```

## 4. Hiểu đúng giới hạn để dùng đúng

LLM dự đoán văn bản **hợp lý**, không tra cứu sự thật — nó có thể bịa
ra một trích dẫn nghe rất thuyết phục (ảo giác). LLM cũng không tự
biết hôm nay là ngày nào, không xem được link bạn không dán nội dung,
và kiến thức của nó dừng ở thời điểm huấn luyện trừ khi có công cụ tra
cứu kèm theo.

## Tự kiểm tra

1. Vì sao cùng một câu hỏi, hai lần hỏi có thể cho hai câu trả lời
   khác nhau?
2. Cuộc trò chuyện rất dài bị "quên" đoạn đầu — khái niệm nào giải
   thích việc này?
3. Viết lại yêu cầu "viết bài giới thiệu sản phẩm" thành một prompt
   theo khung 5 phần ở trên.

## Tóm tắt

- LLM = mạng nơ-ron khổng lồ học đoán từ tiếp theo; token, context
  window, temperature là ba khái niệm vận hành cốt lõi.
- Prompt tốt = rõ nhiệm vụ, đủ ngữ cảnh, có ví dụ, chia bước, và luôn
  kiểm chứng đầu ra.
- Bài cuối: đạo đức và giới hạn của AI — dùng AI có trách nhiệm.
