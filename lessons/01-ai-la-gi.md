# Bài 01 — AI là gì? AI, Machine Learning, Deep Learning và GenAI khác nhau thế nào?

**Mục tiêu của bài:** phân biệt được bốn khái niệm hay bị dùng lẫn lộn:
AI, Machine Learning (học máy), Deep Learning (học sâu) và GenAI (AI tạo
sinh) — và nhận ra chúng quanh mình mỗi ngày.

## 1. Một cách hiểu đơn giản

- **AI (Artificial Intelligence — trí tuệ nhân tạo):** tên gọi chung cho
  mọi kỹ thuật giúp máy làm những việc thường cần trí thông minh của
  người: nhận diện khuôn mặt, dịch ngôn ngữ, gợi ý phim, trò chuyện...
- **Machine Learning (ML — học máy):** một cách làm AI rất phổ biến.
  Thay vì con người viết sẵn mọi quy tắc, ta cho máy **dữ liệu mẫu** và
  máy tự rút ra quy tắc. Ví dụ: cho máy xem 10.000 email đã gắn nhãn
  "rác / không rác", máy tự học cách phân loại email mới.
- **Deep Learning (DL — học sâu):** một nhánh của ML dùng **mạng nơ-ron
  nhiều lớp** (bài 08). DL đứng sau hầu hết thành tựu AI gần đây: nhận
  diện giọng nói, dịch máy, xe tự lái.
- **GenAI (Generative AI — AI tạo sinh):** AI có khả năng **tạo ra nội
  dung mới** — văn bản, hình ảnh, âm thanh, mã lập trình. ChatGPT,
  Gemini, Claude, Midjourney là GenAI. Hầu hết GenAI hiện nay dựa trên
  mô hình ngôn ngữ lớn (LLM — bài 09).

Quan hệ giữa chúng giống như búp bê Nga: **GenAI ⊂ DL ⊂ ML ⊂ AI**.

## 2. Ví dụ đời thường

| Việc quen thuộc | Thuộc loại nào? |
|---|---|
| Lọc thư rác trong Gmail | ML (phân loại văn bản — bài 06) |
| Gợi ý video tiếp theo trên YouTube | ML/DL |
| Mở khoá điện thoại bằng khuôn mặt | DL (nhận diện hình ảnh) |
| Trợ lý ảo trả lời câu hỏi, viết email | GenAI / LLM |
| Điều hoà tự chỉnh nhiệt độ theo giờ cố định | Tự động hoá thường, *không* phải AI |

Điểm khác biệt cốt lõi của ML: **quy tắc do máy học từ dữ liệu**, không
phải do lập trình viên viết tay từng dòng "nếu... thì...".

## 3. AI không phải phép màu

Dù rất mạnh, AI hiện nay về bản chất vẫn là những mô hình toán học học
mẫu (pattern) từ dữ liệu. AI:

- Giỏi ở việc được huấn luyện, kém ở việc ngoài dữ liệu (bài 10).
- Không "hiểu" như con người — nó dự đoán dựa trên mẫu đã học.
- Sai một cách rất tự tin đôi khi — gọi là **ảo giác (hallucination)**.

## 4. Lộ trình của khoá học này

Bạn sẽ tự tay cài đặt (bằng Python thuần, không thư viện lạ) những
thuật toán nền tảng nhất: hồi quy tuyến tính, k-NN, cây quyết định,
Naive Bayes, k-Means và một mạng nơ-ron nhỏ. Khi đã hiểu các "viên gạch"
này, các hệ thống AI lớn sẽ bớt bí ẩn hơn rất nhiều.

## Tự kiểm tra

1. GenAI và ML khác nhau ở điểm nào chính?
2. "Máy học tự viết quy tắc từ dữ liệu" khác gì phần mềm truyền thống?
3. Kể thêm hai ví dụ AI trong đời sống của bạn và đoán chúng thuộc ML,
   DL hay GenAI.

## Tóm tắt

- AI là khái niệm chung; ML là cách cho máy học từ dữ liệu; DL là ML
  dùng mạng nơ-ron nhiều lớp; GenAI là AI biết tạo nội dung mới.
- Bài tiếp theo: máy học *từ cái gì* — dữ liệu và cách đánh giá.
- Thuật ngữ: xem thêm [Thuật ngữ EN–VI](thuat-ngu.md).
