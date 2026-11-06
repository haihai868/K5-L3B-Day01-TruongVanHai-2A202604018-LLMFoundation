# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604018
- **Họ và tên:** Trương Văn Hải
- **SEED** (in ở ô đầu notebook): 4514

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | v   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | v   |

## 2. Kết quả chính

| Model              | F1 test thường (SUB) | F1 50 case khó | Accuracy 50 case khó |
| ------------------ | -------------------- | -------------- | -------------------- |
| 1. BoW + LogReg    | 0.857                | 0.341          | 0.509                |
| 2. BoW + MLP       | 0.947                | 0.341          | 0.509                |
| 3a. Emb + kNN      | 0.947                | 0.719          | 0.673                |
| 3b. Emb + LogReg   | 0.947                | 0.582          | 0.582                |
| 3c. Emb + MLP head | 0.947                | 0.667          | 0.655                |

## 3. Dự đoán và thực tế

1. emb_tricky – Dự đoán Embedding + LogReg xử lý tốt hơn BoW

- Đoán: Encoder + Logistic Regression sẽ phân loại đúng nhiều hơn BoW vì hiểu ngữ nghĩa và ngữ cảnh tốt hơn.
- Kết quả: BoW + MLP đạt F1 = 0.341, Embedding + LogReg đạt 0.582 trên 50 case khó. Tuy nhiên, Embedding + kNN đạt 0.719, cao nhất.
- Vì sao lệch: Embedding không đảm bảo luôn phân loại đúng. Khả năng biểu diễn ngữ nghĩa tốt hơn không đồng nghĩa với khả năng phân loại tốt hơn trong mọi trường hợp.

2. hard_eval – Dự đoán về khả năng tổng quát hóa

- Đoán: Accuracy của tất cả model sẽ giảm đáng kể trên 50 case khó, BoW khó nhất.
- Kết quả: BoW + LogReg và BoW + MLP đạt accuracy 0.509, trong khi Embedding + kNN đạt 0.673.
- Vì sao lệch: Tôi chưa đánh giá đúng mức độ khác biệt giữa các phương pháp embedding. kNN có khả năng xử lý những câu diễn đạt mới tốt hơn dự đoán ban đầu.

3. tradeoff – Dự đoán mô hình nằm trên Pareto

- Đoán: BoW + Logistic Regression và Embedding + Logistic Regression có khả năng nằm trên đường Pareto.
- Kết quả: Chỉ BoW + MLP được xác định nằm trên Pareto.
- Vì sao lệch: Tôi đánh giá dựa trên sự cân bằng latency và F1 một cách chủ quan, chưa xét đầy đủ việc một mô hình có bị mô hình khác vượt trội đồng thời về cả tốc độ và chất lượng hay không.

## 4. Phân tích 50 case khó

_Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai._
Model tụt nhiều nhất: BoW + MLP.
BoW + Logistic Regression cũng giảm từ 0.857 xuống 0.341. Trong khi đó, Embedding + kNN đạt F1 = 0.719 trên 50 case khó, tốt nhất trong các model được thử nghiệm.

Nhóm case dễ gây lỗi:

1. Ham looks like spam: Tin nhắn chứa từ khóa thường gặp trong spam nhưng thực chất là hội thoại bình thường.
2. Natural spam: Spam được viết tự nhiên, không sử dụng những từ khóa quảng cáo quen thuộc.
3. Talks about spam: Tin nhắn thảo luận hoặc cảnh báo về spam nhưng bản thân nó không phải spam.
   Hai case cụ thể và cơ chế khiến model sai:
   Case 1 – Tin nhắn ham chứa từ khóa spam:
   I received a message saying I won a cash prize and need to verify my bank details. Is this a scam?

- Nhãn thực tế: Ham.
- Cơ chế dễ gây lỗi: BoW có thể chú ý đến các từ như cash prize, verify, bank details và xem chúng là dấu hiệu spam. Tuy nhiên, model không hiểu rằng người gửi đang hỏi về một tin nhắn đáng ngờ.
  Case 2 – Spam diễn đạt tự nhiên:
  I know we haven't talked in a while, but I found a great opportunity that could help you earn extra income. The first week is completely free.

- Nhãn thực tế: Spam.
- Cơ chế dễ gây lỗi: Tin nhắn không sử dụng những từ khóa spam quá rõ ràng như WIN, FREE PRIZE hoặc CLAIM NOW. BoW có thể không nhận diện được ý đồ quảng cáo vì chỉ dựa vào sự xuất hiện của các từ riêng lẻ.

## 5. Thử thách cá nhân

_Đề theo SEED của bạn, cách làm, kết quả, giải thích._
Đề #3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ? 2. Cách làm

- Sử dụng cùng tập dữ liệu huấn luyện và kiểm thử (X_tr, X_te).
- Sử dụng cùng kiến trúc MLP, chỉ thay đổi số neuron trong hidden layer.
- Sử dụng BCEWithLogitsLoss làm hàm mất mát.
- Sử dụng Adam optimizer với learning rate = 0.001.
- Huấn luyện mỗi mô hình trong 30 epochs.
- Đánh giá bằng F1-score trên tập test.

3. Kết quả
   Chỉ số MLP (4 neurons) MLP (512 neurons)
   Parameters 8,009 1,025,025
   F1 test 0.904 0.951
   Training time (s) 1.33 16.01

4. Giải thích

- Số lượng tham số: MLP 512 neurons có khoảng 128 lần số tham số so với MLP 4 neurons. Vì vậy, mô hình lớn có khả năng học được nhiều đặc trưng và quan hệ phức tạp hơn.
- F1-score: MLP 512 neurons đạt F1 = 0.951, cao hơn MLP 4 neurons (0.904). Điều này cho thấy mô hình lớn phân loại tốt hơn trên tập test trong thí nghiệm này.
- Thời gian huấn luyện: MLP lớn mất khoảng 16 giây, trong khi MLP nhỏ chỉ mất khoảng 1.33 giây. Mô hình lớn cần nhiều tài nguyên tính toán hơn.

5. Kết luận
   MLP với 512 neurons đạt F1-score cao hơn nhưng có số lượng tham số và thời gian huấn luyện lớn hơn đáng kể. MLP với 4 neurons phù hợp khi cần tiết kiệm tài nguyên và thời gian huấn luyện. Ngược lại, MLP lớn phù hợp khi ưu tiên chất lượng dự đoán và có đủ tài nguyên tính toán.

   Tuy nhiên, mô hình lớn không phải lúc nào cũng tốt hơn vì có nguy cơ overfitting. Cần sử dụng validation set để theo dõi khả năng tổng quát hóa trước khi lựa chọn mô hình cuối cùng.

## 6. Phương án cho sản phẩm thật

Ưu tiên ML/DL chạy local vì có độ trễ thấp và chi phí rẻ. Chọn mô hình có F1 cao nhất trong nhóm đạt p95 dưới 50 ms. Không nên dùng LLM cho toàn bộ 10 triệu SMS/ngày vì chi phí cao. 2. Thiết kế lai Kết hợp Logistic Regression và LLM:

Tin có xác suất spam cao hoặc thấp: ML tự phân loại.
Tin có xác suất từ 0.3–0.7: chuyển cho LLM xử lý. Cách này giúp giảm chi phí, duy trì tốc độ và tận dụng khả năng hiểu ngữ cảnh của LLM.

## 7. AI log và điều đã học

Hỏi: Phân tích các case khó
Trả lời: Giải thích tại sao model có thể nhầm lẫn giữa spam và ham khi ngữ cảnh phức tạp
Áp dụng: phân tích các case khó và bổ sung ví dụ riêng để kiểm tra khả năng tổng quát hóa.

Hiểu rằng Accuracy cao không đồng nghĩa với mô hình tốt, đặc biệt khi dữ liệu mất cân bằng; cần sử dụng F1-score và các tập dữ liệu khó để đánh giá chính xác hơn. Tăng số lượng neurons có thể cải thiện khả năng dự đoán nhưng đồng thời làm tăng số tham số và thời gian huấn luyện.
