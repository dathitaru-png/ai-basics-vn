# Thuật ngữ EN–VI — Glossary

Tra nhanh các thuật ngữ dùng trong bộ bài học. Cột "Bài" chỉ nơi khái
niệm được giải thích đầy đủ.

| English | Tiếng Việt | Nghĩa ngắn gọn | Bài |
|---|---|---|---|
| Artificial Intelligence (AI) | Trí tuệ nhân tạo | Tên chung cho kỹ thuật giúp máy làm việc cần trí thông minh | 01 |
| Machine Learning (ML) | Học máy | Máy tự rút quy tắc từ dữ liệu thay vì được lập trình tay | 01 |
| Deep Learning (DL) | Học sâu | Học máy bằng mạng nơ-ron nhiều lớp | 01 |
| Generative AI (GenAI) | AI tạo sinh | AI tạo ra nội dung mới: văn bản, ảnh, âm thanh, code | 01 |
| Large Language Model (LLM) | Mô hình ngôn ngữ lớn | Mạng nơ-ron lớn học đoán từ tiếp theo trong văn bản | 09 |
| Feature | Đặc trưng | Một cột thông tin đầu vào dùng để dự đoán | 02 |
| Label | Nhãn | Giá trị cần dự đoán của mỗi ví dụ | 02 |
| Training set | Tập huấn luyện | Phần dữ liệu dùng để máy học | 02 |
| Test set | Tập kiểm tra | Phần dữ liệu giữ riêng để đánh giá cuối | 02 |
| Overfitting | Quá khớp ("học vẹt") | Thuộc lòng tập học, kém trên dữ liệu mới | 02 |
| Underfitting | Chưa khớp | Mô hình quá đơn giản, học cũng kém | 02 |
| Regression | Hồi quy | Bài toán dự đoán một con số | 02, 03 |
| Classification | Phân loại | Bài toán dự đoán một nhãn | 02 |
| Clustering | Phân cụm | Tự gom nhóm khi không có nhãn | 07 |
| Accuracy | Độ chính xác | Tỉ lệ dự đoán đúng trên tổng số | 02 |
| Mean Squared Error (MSE) | Sai số bình phương trung bình | Trung bình bình phương sai lệch dự đoán (hồi quy) | 02, 03 |
| Linear regression | Hồi quy tuyến tính | Dự đoán bằng đường thẳng `y = wx + b` | 03 |
| Gradient descent | Hạ gradient | Cách tối ưu: lặp lại "đo độ dốc, bước ngược dốc" | 03 |
| Loss function | Hàm mất mát / hàm lỗi | Con số đo mô hình đang sai bao nhiêu | 03 |
| Learning rate | Tốc độ học | Độ dài bước trong mỗi vòng gradient descent | 03 |
| Epoch | Vòng học | Một lượt duyệt qua toàn bộ tập huấn luyện | 03 |
| k-Nearest Neighbours (k-NN) | k hàng xóm gần nhất | Phân loại bằng phiếu bầu của k điểm gần nhất | 04 |
| Euclidean distance | Khoảng cách Euclid | Độ dài đường thẳng giữa hai điểm | 04 |
| Decision tree | Cây quyết định | Mô hình gồm chuỗi câu hỏi có/không | 05 |
| Gini impurity | Độ tạp Gini | Thước đo mức "lẫn lộn" nhãn trong một nhóm | 05 |
| Naive Bayes | Bayes ngây thơ | Phân loại xác suất với giả định các đặc trưng độc lập | 06 |
| Prior | Xác suất tiên nghiệm | Mức phổ biến của một lớp trước khi xem bằng chứng | 06 |
| Laplace smoothing | Làm mịn Laplace | Cộng một lượng nhỏ để tránh xác suất bằng 0 | 06 |
| k-Means | Phân cụm k trung tâm | Lặp "gán vào tâm gần nhất — dời tâm về trung bình" | 07 |
| Centroid | Tâm cụm | Điểm trung bình của một cụm | 07 |
| Inertia | Quán tính cụm | Tổng bình phương khoảng cách tới tâm; càng nhỏ cụm càng gọn | 07 |
| Neuron | Nơ-ron (nhân tạo) | Đơn vị tính: tổng có trọng số qua hàm kích hoạt | 08 |
| Weight / Bias | Trọng số / hệ số chệch | Hệ số mô hình học được cho mỗi đầu vào / hằng số cộng thêm | 08 |
| Activation function | Hàm kích hoạt | Hàm biến đổi đầu ra của nơ-ron (step, tanh, sigmoid...) | 08 |
| Hidden layer | Lớp ẩn | Lớp nơ-ron nằm giữa đầu vào và đầu ra | 08 |
| Backpropagation | Lan truyền ngược | Cách chia "trách nhiệm" của lỗi về từng trọng số | 08 |
| Perceptron | Perceptron | Nơ-ron đơn giản nhất; chỉ tách được bằng một đường thẳng | 08 |
| Token | Token | Đơn vị văn bản nhỏ mà LLM xử lý (không hẳn là một từ) | 09 |
| Context window | Cửa sổ ngữ cảnh | Lượng token tối đa mô hình thấy cùng lúc | 09 |
| Temperature | Nhiệt độ | Mức ngẫu hứng khi mô hình chọn từ tiếp theo | 09 |
| Prompt | Câu lệnh / lời nhắc | Nội dung người dùng đưa cho mô hình để yêu cầu việc | 09 |
| Few-shot | Học qua vài ví dụ | Đưa kèm vài ví dụ mẫu ngay trong prompt | 09 |
| Hallucination | Ảo giác | Mô hình bịa thông tin sai nhưng trình bày rất tự tin | 09, 10 |
| Bias | Thiên kiến / lệch | Xu hướng sai lệch hệ thống do dữ liệu hoặc thiết kế | 10 |
| Synthetic data | Dữ liệu tổng hợp | Dữ liệu hư cấu sinh bằng chương trình, ghi rõ là hư cấu | 02 |
| Supervised learning | Học có giám sát | Học từ ví dụ đã có nhãn/đáp án | 02 |
| Unsupervised learning | Học không giám sát | Học cấu trúc trong dữ liệu không nhãn | 07 |
