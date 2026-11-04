# Zalo AI Challenge 2026 — hồ sơ thông tin

**Ngày thu thập:** 04/11/2026 (GMT+7)
**Nguồn:** trang chính thức Zalo AI Challenge và PDF thể lệ chính thức.

**Đội đã chọn LifeMate.** Xem [kế hoạch triển khai chi tiết cho 5 người](implementation_plan.md), gồm phân công, lịch 04–23/11, đánh giá, GPU, Docker và checklist nộp bài. Đây là kế hoạch đề xuất, chưa phải kết quả triển khai.

## Các tệp trong thư mục

- Zalo_AI_Challenge_2026_The_le_VI.pdf — thể lệ tiếng Việt, 14 trang.
- Zalo_AI_Challenge_2026_The_le_VI.txt — bản trích xuất văn bản để tìm kiếm.
- Zalo_AI_Challenge_2026_Regulations_EN.pdf — regulations tiếng Anh, 15 trang.
- Zalo_AI_Challenge_2026_Regulations_EN.txt — bản trích xuất văn bản tiếng Anh.
- README.md — bản tổng hợp tiếng Việt dưới đây.

## 1. Cuộc thi là gì?

Zalo AI Challenge 2026 là cuộc thi kỹ năng do Zalo AI tổ chức nhằm thúc đẩy khoa học dữ liệu và trí tuệ nhân tạo tại Việt Nam. Năm 2026 có hai bài toán theo hướng AI agent: **LifeMate** (nghe yêu cầu nói và thực hiện hành động qua công cụ) và **DataMate** (trả lời câu hỏi tiếng Việt trên dữ liệu dạng bảng).

Mục tiêu chung là xây hệ thống giải quyết tình huống thực tế, chấm bằng kết quả đầu ra và trạng thái cuối của hệ thống; không chỉ tạo câu trả lời nghe hợp lý.

## 2. Hai nhánh thử thách

### LifeMate — Agentic AI That Turns Your Voice Into Action

- **Bài toán:** biến yêu cầu nói bằng tiếng Việt thành chuỗi hành động qua các ứng dụng/dịch vụ được kết nối, chẳng hạn nhà thông minh, ngân hàng, gọi xe, giao đồ ăn và email.
- **Đầu vào:** âm thanh yêu cầu hiện tại; ngữ cảnh như thời gian, múi giờ, vị trí, hồ sơ người dùng; các ứng dụng và bộ công cụ khả dụng; lịch sử hội thoại; môi trường mô phỏng có dữ liệu ứng dụng.
- **Đầu ra:** danh sách có thứ tự các lệnh gọi công cụ, kèm tham số và mã băm trạng thái cuối (state_hash). Agent phải hỏi bổ sung nếu thiếu thông tin; nếu không có ứng dụng phù hợp thì không thực hiện hành động.
- **Điểm khó:** nhận dạng nhiều giọng/phương ngữ Bắc–Trung–Nam, hiểu ý định, lập kế hoạch nhiều bước, chọn đúng công cụ và đối số, rồi để ứng dụng ở đúng trạng thái mong muốn.

### DataMate — Talking To Your Data Through Agentic AI

- **Bài toán:** trả lời câu hỏi đời thường bằng tiếng Việt trực tiếp trên dữ liệu bảng lộn xộn, như dữ liệu bán hàng, kho, điểm danh, hồ sơ phòng khám hoặc công nợ.
- **Đầu vào:** gói dữ liệu của một doanh nghiệp (một hay nhiều tệp CSV/XLSX/SQLite và README mô tả); câu hỏi tiếng Việt; ngày tham chiếu (as_of_date); rulebook và lịch dương–âm dùng chung.
- **Đầu ra:** một giá trị có kiểu (số, phần trăm, chuỗi, ngày, boolean, danh sách, bảng), unanswerable khi dữ liệu không đủ để suy ra, hoặc abstain khi hệ thống chọn không trả lời.
- **Nguyên tắc cốt lõi:** trả lời đúng khi có căn cứ; biết từ chối khi dữ liệu không hỗ trợ. Câu trả lời sai bị tính điểm bất lợi, còn im lặng có thể không bị phạt theo mô tả tổng quan.
- **Ví dụ câu hỏi:** “Tháng này bán chạy nhất mặt hàng gì?”, “Tuần rồi bán được bao nhiêu?”, “Còn ai nợ tiền chưa trả?”. Câu hỏi có thể viết tắt, dùng cách nói địa phương, và dữ liệu có nhiều sheet/tên hàng không đồng nhất.

## 3. Dữ liệu, đánh giá và bảng xếp hạng

- Website thông báo bộ phát triển, dữ liệu huấn luyện, public test và phần mềm đánh giá được phát hành từ **26/10/2026**.
- Cuộc thi có vòng public test và private test. Bảng công khai chỉ mang tính tham khảo/tạm thời; kết quả cuối cùng được xác minh bằng bộ private test riêng.
- Bảng xếp hạng hiển thị ba nhóm mô hình: **APIs** (API bên thứ ba), **LLM** (mô hình lớn trên 15B tham số), **SLM** (mô hình nhỏ không quá 15B). Trang leaderboard nói thứ hạng công khai được tính theo SLM; LLM và APIs/Others chỉ để tham khảo.
- Cổng portal có các mục Description, Evaluation, Data, Rule và Submission Guidance. Lần kiểm tra bổ sung ngày 04/11/2026 đã đọc được Rule và Submission Guidance của LifeMate; riêng Data yêu cầu đăng nhập. Thư mục chưa chứa dataset, development kit, evaluation software hoặc submission package.
- LifeMate yêu cầu toàn hệ thống dưới 6B tham số, LLM dưới 5B, context tối đa 16.384 token; inference offline trong 30 giây/mẫu trên 1 GPU A30, CPU 16 core, RAM 64 GB. Chỉ truy cập dữ liệu ứng dụng qua API `zaic_tools` được phép, đúng một ToolEnv mỗi mẫu; không đọc trực tiếp `<split>_environment/`.

## 4. Lịch cuộc thi

| Mốc | Nội dung |
|---|---|
| 26/10/2026, 09:00 | Bắt đầu; công bố development kit, train data, public test và evaluation software |
| 16/11/2026, 23:59 | Hạn nộp giải pháp: mã huấn luyện, mã kiểm thử và mô hình |
| 20/11/2026 | Công bố private test; bảng xếp hạng lúc đó là tạm thời |
| 23/11/2026, 23:59 | Hạn nộp kết quả cuối cùng trên private test |
| 02/12/2026 | Công bố top 2 mỗi nhánh |
| 12/12/2026 | Công bố thứ hạng chung cuộc |

**Có lệch giờ giữa nguồn chính thức:** trang chủ ghi mốc phát hành private test lúc **11:00 GMT+7**, trong khi PDF thể lệ ghi **09:00 GMT+7**. Nên theo dõi thông báo mới nhất trên portal hoặc email từ Ban Tổ chức trước mốc này.

## 5. Giải thưởng

- Tổng giá trị giải thưởng công bố trên trang chủ: **12.000 USD**, tiền mặt và quà tặng.
- Mỗi nhánh có giải Nhất tương đương **3.500 USD** và giải Nhì tương đương **2.500 USD**; top 5 mỗi nhánh có gói quà ZAIC.
- PDF thể lệ quy đổi mức tiền mặt tương ứng thành 90.412.000 VND và 64.580.000 VND, theo tỷ giá bình quân mua/bán ngày 23/10/2026. Thuế được khấu trừ theo quy định. Người thắng cần nộp hồ sơ trong 2 tuần từ khi được thông báo; giải được trao trong 30 ngày sau khi nhận đủ hồ sơ và không chuyển nhượng.

## 6. Điều kiện tham gia và đội thi

- Theo thể lệ: người tham gia cần có quốc tịch Việt Nam, tài khoản Zalo hợp lệ, và từ 18 tuổi hoặc đạt tuổi trưởng thành theo nơi cư trú; trường hợp chưa đủ tuổi cần có chấp thuận hợp lệ của phụ huynh/người giám hộ.
- Mỗi người chỉ tham gia một đội hoặc thi cá nhân, dùng một tài khoản. Đội tối đa 5 thành viên, tính cả đội trưởng.
- Nhân viên/cộng tác viên/người làm việc theo hợp đồng cho VNG và công ty thuộc tập đoàn không đủ điều kiện theo PDF thể lệ.
- Trang FAQ khuyến nghị có kiến thức lập trình cơ bản; cuộc thi mở cho người Việt quan tâm AI.
- FAQ trực tuyến được tra cứu bổ sung ngày 04/11 ghi chức năng thêm/xóa thành viên bị khóa từ **17/11**. Bản tổng hợp trước đó ghi 16/11; nên chốt đội sớm và theo dõi thông báo hiện hành trên portal.

## 7. Quy định dữ liệu và bài nộp cần nhớ

- Chỉ dùng dữ liệu cuộc thi cho mục đích học thuật/phi thương mại trong phạm vi cuộc thi; khi sử dụng phải trích dẫn challenge.zalo.ai.
- Không chia sẻ riêng dữ liệu cuộc thi ra ngoài đội trong thời gian thi; không sao chép/phân phối cho bên thứ ba. Thể lệ yêu cầu xóa/hủy dữ liệu cá nhân sau khi cuộc thi kết thúc.
- Không dùng nhãn thủ công hoặc dự đoán của con người trên validation/test để tạo bài dự thi. Dữ liệu ngoài chỉ được dùng nếu nhánh cho phép và phải tuân thủ pháp luật/quy định về dữ liệu cá nhân.
- Mã nguồn mở sử dụng cần có giấy phép OSI hợp lệ, trừ trường hợp có chấp thuận bằng văn bản khác.
- Mỗi đội/cá nhân được nộp tối đa 5 bài mỗi ngày; chỉ chọn một bài cuối cùng để chấm chính thức. Trước private test, Ban Tổ chức yêu cầu đóng băng mô hình và nộp mã huấn luyện, mã kiểm thử, mô hình, mô tả phương pháp và hướng dẫn chạy.
- Riêng Submission Guidance của LifeMate: public/private result là JSONL nộp qua Submission. Hạn giải pháp 16/11, 23:59: nộp Docker package qua Solution Submission, cung cấp checksum và link Google Drive công khai để BTC tải. Gói giải pháp được nộp tối đa **2 lần**, BTC chỉ đánh giá **lần cuối**. Cần đối chiếu hướng dẫn Docker riêng và làm rõ phần “data” phải kèm trước khi công khai gói có dữ liệu hạn chế.
- Tác giả giữ quyền sở hữu bài dự thi nhưng cấp cho Ban Tổ chức quyền sử dụng không độc quyền, miễn phí, không giới hạn thời gian cho mục đích nghiên cứu, quảng bá, giáo dục và truyền thông liên quan cuộc thi.

Xem PDF tiếng Việt trong thư mục để đọc nguyên văn toàn bộ điều khoản và ngoại lệ. Phần trên là tóm tắt, không thay thế thể lệ chính thức.

## 8. Ban cố vấn và thông tin trang chủ

- Prof. Nguyen Le Minh — Trưởng phòng thí nghiệm Machine Learning and Natural Language Understanding, JAIST.
- Assoc. Prof. Dr. Quan Thanh Tho — Dean, Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology, VNU-HCM.
- Mr. Nguyen Minh Tu — CTO Zalo, Trưởng Ban Tổ chức Zalo AI Challenge 2026.
- Dr. Nguyen Truong Son — Senior Research Science Manager, Zalo AI.
- MSc. Nguyen Canh Hung — Technical Manager, Zalo AI.
- MSc. Tran Van Hoa — AI Manager, Zalo AI.

Trang chủ nêu ba lợi ích: giải quyết bài toán thực tế, trao đổi với chuyên gia, và trải nghiệm đánh giá/leaderboard theo chuẩn quốc tế. Nhà tài trợ hiển thị là Leadtek; đối tác truyền thông là Znews. Email hỗ trợ chính thức: [aichallenge@zalo.me](mailto:aichallenge@zalo.me).

## 9. Liên kết nguồn chính thức

- [Trang chủ và mô tả thử thách](https://challenge.zalo.ai/#challenges)
- [Portal LifeMate](https://challenge.zalo.ai/portal/lifemate)
- [Portal DataMate](https://challenge.zalo.ai/portal/datamate)
- [FAQ](https://challenge.zalo.ai/#faq)
- [Quy định cuộc thi](https://challenge.zalo.ai/regulations)
- [PDF thể lệ tiếng Việt](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf)
- [PDF regulations tiếng Anh](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-en.pdf)

> Lưu ý: trang web và leaderboard là nguồn cập nhật trực tiếp; mốc thời gian và nội dung có thể được Ban Tổ chức điều chỉnh. Hãy kiểm tra portal chính thức trước khi nộp bài.
