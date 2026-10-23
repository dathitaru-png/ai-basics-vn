# Bài 10 — Đạo đức và giới hạn của AI: dùng AI có trách nhiệm

**Mục tiêu của bài:** nhận diện các rủi ro thật của AI — thiên kiến,
ảo giác, quyền riêng tư, an toàn — và có thói quen dùng AI một cách có
trách nhiệm. Đây là bài kết của lộ trình cơ bản.

## 1. Thiên kiến (bias): dữ liệu lệch, kết quả lệch

Mô hình học từ dữ liệu con người tạo ra — và dữ liệu chứa sẵn định
kiến của xã hội: nghề nghiệp gắn với giới tính, vùng miền gắn với thu
nhập... Nếu không kiểm soát, mô hình sẽ **khuếch đại** các mẫu đó:
hệ thống tuyển dụng học từ lịch sử tuyển dụng thiên vị sẽ tiếp tục
thiên vị, và tệ hơn, khoác lên nó vẻ "khách quan vì là máy tính".

Bài học thực hành: trước khi tin một mô hình, hãy hỏi *nó học từ dữ
liệu nào, của ai, thiếu nhóm nào?* — dữ liệu quyết định phần lớn chất
lượng và tính công bằng của AI (bài 02).

## 2. Ảo giác (hallucination): sai một cách tự tin

Mô hình ngôn ngữ được tối ưu để tạo văn bản **nghe hợp lý**, không
phải để nói thật (bài 09). Hệ quả: nó có thể bịa ra bài báo không tồn
tại, điều luật bịa, hay một phép tính sai — với giọng văn chắc chắn
như đúng rồi.

Thói quen phòng vệ:

- Coi đầu ra của AI là **bản nháp cần kiểm chứng**, nhất là số liệu,
  tên riêng, trích dẫn, và mọi thứ có hậu quả nếu sai.
- Đối chiếu bằng nguồn độc lập, đáng tin — không dùng chính AI đó để
  "xác nhận" lại.
- Việc càng quan trọng (y tế, pháp lý, tài chính), càng phải có
  chuyên gia con người chịu trách nhiệm cuối cùng.

## 3. Quyền riêng tư và dữ liệu cá nhân

Những gì bạn dán vào một dịch vụ AI có thể được lưu lại và — tuỳ điều
khoản — dùng để cải thiện hệ thống. Nguyên tắc an toàn:

- Không dán dữ liệu cá nhân nhạy cảm (số giấy tờ, mật khẩu, hồ sơ y
  tế, dữ liệu khách hàng thật) vào các dịch vụ công cộng.
- Trong lập trình và học tập: dùng **dữ liệu tổng hợp hư cấu** như gói
  code của khoá này vẫn làm — mọi ví dụ đều là bịa, ghi rõ là bịa.
- Khi xây dựng sản phẩm: thu thập tối thiểu, nói rõ mục đích, và tôn
  trọng quy định pháp luật về dữ liệu cá nhân nơi bạn hoạt động.

## 4. Giới hạn năng lực cần nhớ

- AI hiện nay giỏi **nhận mẫu trong dữ liệu đã có**; nó không có kinh
  nghiệm sống, không có ý thức, và khả năng suy luận của nó mong manh
  khi bài toán nằm ngoài phân phối dữ liệu huấn luyện.
- AI có thể làm tốt bài kiểm tra chuẩn nhưng vẫn sai những điều hiển
  nhiên với con người. Đừng dùng một con số benchmark làm bằng chứng
  cho mọi tình huống.
- Tự động hoá không xoá trách nhiệm: "máy nó quyết vậy" không phải lý
  do miễn trừ cho người triển khai và sử dụng.

## 5. Một khung dùng AI có trách nhiệm

1. **Minh bạch:** nói rõ khi nội dung có AI tham gia tạo ra, nhất là
   khi công bố cho người khác.
2. **Kiểm chứng:** kiểm tra đầu ra theo mức độ hậu quả nếu sai.
3. **Con người quyết định cuối:** việc ảnh hưởng tới người khác (tuyển
   dụng, chấm điểm, y tế...) phải có người chịu trách nhiệm xem lại.
4. **Tôn trọng dữ liệu và tác quyền:** không dùng dữ liệu không có
   quyền dùng; ghi nguồn khi cần.
5. **Cập nhật liên tục:** lĩnh vực này thay đổi rất nhanh — kể cả các
   giới hạn trong bài này cũng cần được xem lại theo thời gian.

## Tự kiểm tra

1. Một chatbot trả lời rất trôi chảy — vì sao độ trôi chảy không phải
   bằng chứng của độ đúng?
2. Kể một tình huống thiên kiến dữ liệu có thể làm AI bất công.
3. Bạn định dán hồ sơ khách hàng thật vào chatbot để nhờ tóm tắt —
   theo bài này, nên làm gì thay thế?

## Tóm tắt

- AI phản chiếu dữ liệu của nó: lệch dữ liệu → lệch kết quả.
- Ảo giác là tính chất cố hữu của LLM hiện nay — kiểm chứng là bắt
  buộc, không phải tuỳ chọn.
- Dùng AI có trách nhiệm = minh bạch, kiểm chứng, giữ con người ở
  quyết định cuối, tôn trọng dữ liệu.

**Hết lộ trình cơ bản.** Tiếp theo bạn có thể: làm [Câu hỏi ôn
tập](cau-hoi-on-tap.md), tra [Thuật ngữ](thuat-ngu.md), hoặc đóng góp
bài học và bản dịch mới cho dự án — xem `CONTRIBUTING.md` ở thư mục
gốc.
