Theo các tác giả, dù các mô hình như GPT-4o, Claude hay các phương pháp SFT (Supervised Fine-Tuning) đã tiến bộ, việc ứng dụng tool-use trong thực tế vẫn đối mặt với hai rào cản lớn:

Sự giòn vã trong suy luận đa bước/đa lượt (Brittle multi-step & multi-turn reasoning):

Khi thực hiện chuỗi lệnh gọi công cụ phụ thuộc lẫn nhau (nested/dependent calls), mô hình thường xuyên thất bại trong việc ánh xạ đầu ra của tool này vào schema đầu vào mong đợi của tool tiếp theo do các API lộ diện dưới dạng dữ liệu không đồng nhất (heterogeneous schemas, diverse output formats).

Vấn đề này càng trầm trọng hơn ở các cuộc hội thoại nhiều lượt (multi-turn) khi ngữ cảnh ngày càng phình to, làm suy giảm khả năng tập trung của mô hình. Các benchmark như NESTFUL cho thấy các mô hình mạnh nhất cũng chỉ đạt 28% độ chính xác khớp toàn bộ chuỗi (full-sequence match).
Sự suy giảm hiệu năng khi danh mục công cụ quá lớn (Large tool catalogues degradation):

Khi số lượng API tăng lên (như ToolBench có hơn 16.000 API), việc ném toàn bộ raw API schemas vào input context của mô hình làm giảm hiệu năng nghiêm trọng (hiện tượng "lost in the middle" hoặc quá tải token, với mức suy giảm độ chính xác từ 7% đến 85% khi scale từ 8K lên 120K token).
A. Tool Primitives (Đổi mới giao diện gọi công cụ)Ý tưởng đột phá: "Thay vì bắt mô hình nói ngôn ngữ của API (schema phức tạp), tại sao không bắt công cụ nói ngôn ngữ của mô hình?"Mỗi công cụ không còn lộ diện schema thô trước LLM, mà được bọc bên trong một Tool Primitive (một trình bao bọc dựa trên LLM).Thay vì gọi hàm bằng JSON schema cứng nhắc, model gọi công cụ bằng ngôn ngữ tự nhiên (natural language). Bản thân Tool Primitive sẽ tự phân giải schema, thực thi lệnh bên trong và trả về kết quả cấu trúc, giúp việc giao tiếp giữa các tool với nhau (inter-tool communication) trở nên đồng nhất và mượt mà hơn.B. ToolFace (Kho lưu trữ công cụ trung tâm quy mô lớn)Là một cơ sở dữ liệu tập trung chứa tới 25,519 hàm (tổng hợp từ ToolBench, NESTFUL, $\tau^2$-Bench, ACEBench, BFCLv4 và các công cụ thủ công).Thay vì nạp toàn bộ danh mục tool vào ngữ cảnh, LLM sẽ thực hiện truy xuất động (dynamic retrieval) để chỉ lôi ra đúng các Tool Primitive liên quan tại thời điểm suy luận, loại bỏ hoàn toàn việc làm ô nhiễm context bằng raw schemas
Để điều phối Tool Primitives và ToolFace một cách đáng tin cậy, HEART xây dựng một hệ thống đa tác nhân (Multi-Agent System) gồm 5 giai đoạn:

Planner (Hoạch định): Phân tích ý đồ truy vấn, kiểm tra xem ngữ cảnh hiện tại đã đủ thông tin chưa (nếu thiếu sẽ tự động hỏi lại người dùng để tránh đoán mò), sau đó lập ra một kế hoạch gọi công cụ theo thứ tự.

Router (Định tuyến): Ánh xạ tham số, liên kết đối số (parameter mapping) và gửi yêu cầu ngôn ngữ tự nhiên đến đúng Tool Primitive mục tiêu kèm các siêu tham số thực thi (h-params như retry, priority).

Tools (Thực thi Tool Primitives): Chạy công cụ bên trong lớp bọc độc lập.

Execution (Chạy và Trả kết quả): Thu thập dữ liệu đầu ra.

Verifier (Kiểm định & Phục hồi): Đánh giá kết quả thực thi dựa trên 4 tiêu chí: hoàn thành tác vụ (task completion), tính nhất quán của đối số (argument consistency), tính hợp lệ của việc thực thi (execution validity) và thỏa mãn ràng buộc (constraint satisfaction). Nếu phát hiện lỗi, Verifier sẽ sinh phản hồi có cấu trúc (structured feedback) đẩy ngược về cho Planner để tái lập kế hoạch (re-planning) thay vì buông xuôi hay lặp vô hạn.


---

## PHẦN 1: Làm rõ Mã giả Algorithm 1 (Vòng lặp điều phối & Ngân sách $B$)

Mã giả **Algorithm 1** trong Phụ lục C mô tả chính xác cách 4 tác nhân (`Planner`, `Router`, `Tool Primitive`, `Verifier`) phối hợp nhịp nhàng theo một chu trình khép kín 5 giai đoạn: **Planning $\rightarrow$ Routing $\rightarrow$ Tools (Primitives) $\rightarrow$ Execution $\rightarrow$ Verification**.

### 4 Bước vận hành chi tiết trong vòng đời một truy vấn:
1. **Giai đoạn Kiểm tra đầy đủ thông tin (Information Sufficiency Loop — Dòng 2–5):**
   - Khi nhận câu hỏi $q$ từ người dùng, `Planner` phân tích xem thông tin đã đủ để lập kế hoạch chưa.
   - Nếu thiếu (`insufficient`), `Planner` tự động sinh ra câu hỏi làm rõ (`clarification request`) để hỏi ngược lại người dùng. Vòng lặp này tiếp tục cho đến khi ngữ cảnh đủ (`sufficient`).
2. **Giai đoạn Lập kế hoạch (Plan Generation — Dòng 6):**
   - `Planner` bẻ nhỏ truy vấn thành một chuỗi các bước gọi công cụ có thứ tự: $\Pi = (\pi_1, \pi_2, \dots, \pi_K)$.
3. **Giai đoạn Thực thi & Kiểm định tuần tự (Execution & Verification Loop — Dòng 7–17):**
   - Vòng lặp chạy qua từng bước $\pi_k$:
     - `Router` lấy ngữ cảnh từ lịch sử, ánh xạ tham số và đóng gói thành yêu cầu ngôn ngữ tự nhiên $\beta_k$ kèm siêu tham số (`retry`, `priority`).
     - `Tool Primitive` nhận $\beta_k$, tự giải quyết schema nội bộ, chạy hàm thực thi và trả về kết quả $r_k$.
     - `Verifier` chấm điểm kết quả $r_k$ theo 4 tiêu chí.
4. **Giai đoạn Xử lý lỗi & Ngân sách tái lập kế hoạch ($B$) (Dòng 11–16):**
   - **Nếu Verifier báo `pass`:** Chuyển sang bước tiếp theo $\pi_{k+1}$ cho đến hết chuỗi $K$ bước và trả kết quả cho người dùng.
   - **Nếu Verifier báo `fail`:**
     - `Verifier` viết một bản phản hồi chẩn đoán lỗi cụ thể (structured feedback $\phi_k$).
     - Ngữ cảnh $C$ được nạp thêm phản hồi lỗi này.
     - `Planner` kích hoạt quá trình tái lập kế hoạch (`re-planning`) để sửa đổi lộ trình.
     - Biến đếm ngân sách $b$ tăng lên 1 ($b \leftarrow b + 1$).
     - **Giới hạn ngân sách $B$ (Mặc định $B = 3$):** Nếu hệ thống lặp lại quá 3 lần tái lập kế hoạch mà vẫn không qua được Verifier ($b > B$), hệ thống sẽ dừng lại và trả về báo cáo lỗi chi tiết (`FAILUREREPORT`), tránh việc agent sa vào vòng lặp vô hạn.

---

## PHẦN 2: Làm rõ Hệ thống Prompt (Phụ lục D)

Phụ lục D cung cấp các System Prompts nguyên bản định hình hành vi cho từng tác nhân trong hệ thống HEART, tách bạch rõ ràng nhiệm vụ của từng agent:

### 1. Planner Prompt (Tác nhân Hoạch định & Kiểm tra thông tin)
- **Nhiệm vụ:** Phân tích ý đồ (*Intent Analysis*) và kiểm tra xem thông tin đã đủ chưa.
- **Cơ chế hoạt động:**
  - Nếu thiếu thông tin, bắt buộc xuất ra JSON có trạng thái `"INSUFFICIENT"` kèm câu hỏi làm rõ.
  - Nếu đủ, xuất ra danh sách các bước gọi công cụ theo định dạng JSON (`"status": "SUFFICIENT"`, kèm mảng `plan` gồm các bước có `dependencies`).
  - Khi nhận feedback lỗi từ `Verifier`, `Planner` đọc lỗi đó để viết lại kế hoạch mới được đánh dấu cờ `"replanned": true`.

### 2. Router Prompt (Tác nhân Định tuyến & Ánh xạ tham số)
- **Nhiệm vụ:** Giải quyết bài toán liên kết tham số (*parameter mapping*) từ ngữ cảnh và đóng gói thành yêu cầu dạng ngôn ngữ tự nhiên.
- **Cơ chế hoạt động:**
  - Không bắt `Router` phải nhớ raw schema phức tạp, mà yêu cầu `Router` trích xuất giá trị từ ngữ cảnh (truy vấn người dùng, kết quả bước trước như `$step_j`) và nhúng trực tiếp vào yêu cầu dạng ngôn ngữ tự nhiên (`invocation_request`).
  - Gắn kèm các siêu tham số thực thi (`h_params`) như số lần thử lại (`retry` từ 1 đến 5), độ ưu tiên (`priority`: `low`/`normal`/`high`) và thời gian chờ (`timeout_s`).

### 3. Tool Primitive Prompt (Lớp bọc công cụ thông minh)
- **Nhiệm vụ:** Đóng vai trò là cầu nối dịch từ ngôn ngữ tự nhiên sang schema API thô bên trong từng công cụ.
- **Cơ chế hoạt động:**
  - Nhận yêu cầu ngôn ngữ tự nhiên từ `Router` $\rightarrow$ tự phân giải schema, ép kiểu dữ liệu (*type coercion*), chuẩn hóa ngày tháng, kiểm tra ràng buộc tham số.
  - Gọi hàm thực thi bên dưới và trả về JSON chuẩn hóa gồm `status: SUCCESS` kèm kết quả và một đoạn tóm tắt (`summary`) bằng ngôn ngữ tự nhiên để làm ngữ cảnh cho bước tiếp theo.
  - **Nguyên tắc cô lập lỗi (Execution Isolation):** `Tool Primitive` tuyệt đối không tự ý retry hay gọi tool khác khi lỗi xảy ra, mà bắt buộc phải trả về `status: FAILURE` để đẩy việc xử lý lỗi về cho `Verifier`.

### 4. Verifier Prompt (Tác nhân Kiểm định & Phản hồi)
- **Nhiệm vụ:** Đóng vai trò "trọng tài" đánh giá kết quả thực thi $r_k$ và sinh feedback nếu có lỗi.
- **Cơ chế hoạt động:** Chấm điểm độc lập qua **4 tiêu chí**:
  1. **Task Completion** (Hoàn thành tác vụ).
  2. **Argument Consistency** (Tính nhất quán của tham số so với ý định ban đầu).
  3. **Execution Validity** (Tính hợp lệ khi thực thi không có lỗi runtime).
  4. **Constraint Satisfaction** (Thỏa mãn các quy tắc kinh doanh/giới hạn hệ thống).
- Chỉ khi cả 4 tiêu chí đạt `pass`, `Verifier` mới cho phép đi tiếp. Nếu có bất kỳ tiêu chí nào `fail`, `Verifier` xuất ra JSON phản hồi mang tính hành động cao, giúp `Planner` có dữ liệu chính xác để sửa sai ở vòng kế hoạch tiếp theo.


---

## PHẦN 1: Làm rõ Mã giả Algorithm 1 (Vòng lặp điều phối & Ngân sách $B$)

Mã giả **Algorithm 1** trong Phụ lục C mô tả chính xác cách 4 tác nhân (`Planner`, `Router`, `Tool Primitive`, `Verifier`) phối hợp nhịp nhàng theo một chu trình khép kín 5 giai đoạn: **Planning $\rightarrow$ Routing $\rightarrow$ Tools (Primitives) $\rightarrow$ Execution $\rightarrow$ Verification**.

### 4 Bước vận hành chi tiết trong vòng đời một truy vấn:
1. **Giai đoạn Kiểm tra đầy đủ thông tin (Information Sufficiency Loop — Dòng 2–5):**
   - Khi nhận câu hỏi $q$ từ người dùng, `Planner` phân tích xem thông tin đã đủ để lập kế hoạch chưa.
   - Nếu thiếu (`insufficient`), `Planner` tự động sinh ra câu hỏi làm rõ (`clarification request`) để hỏi ngược lại người dùng. Vòng lặp này tiếp tục cho đến khi ngữ cảnh đủ (`sufficient`).
2. **Giai đoạn Lập kế hoạch (Plan Generation — Dòng 6):**
   - `Planner` bẻ nhỏ truy vấn thành một chuỗi các bước gọi công cụ có thứ tự: $\Pi = (\pi_1, \pi_2, \dots, \pi_K)$.
3. **Giai đoạn Thực thi & Kiểm định tuần tự (Execution & Verification Loop — Dòng 7–17):**
   - Vòng lặp chạy qua từng bước $\pi_k$:
     - `Router` lấy ngữ cảnh từ lịch sử, ánh xạ tham số và đóng gói thành yêu cầu ngôn ngữ tự nhiên $\beta_k$ kèm siêu tham số (`retry`, `priority`).
     - `Tool Primitive` nhận $\beta_k$, tự giải quyết schema nội bộ, chạy hàm thực thi và trả về kết quả $r_k$.
     - `Verifier` chấm điểm kết quả $r_k$ theo 4 tiêu chí.
4. **Giai đoạn Xử lý lỗi & Ngân sách tái lập kế hoạch ($B$) (Dòng 11–16):**
   - **Nếu Verifier báo `pass`:** Chuyển sang bước tiếp theo $\pi_{k+1}$ cho đến hết chuỗi $K$ bước và trả kết quả cho người dùng.
   - **Nếu Verifier báo `fail`:**
     - `Verifier` viết một bản phản hồi chẩn đoán lỗi cụ thể (structured feedback $\phi_k$).
     - Ngữ cảnh $C$ được nạp thêm phản hồi lỗi này.
     - `Planner` kích hoạt quá trình tái lập kế hoạch (`re-planning`) để sửa đổi lộ trình.
     - Biến đếm ngân sách $b$ tăng lên 1 ($b \leftarrow b + 1$).
     - **Giới hạn ngân sách $B$ (Mặc định $B = 3$):** Nếu hệ thống lặp lại quá 3 lần tái lập kế hoạch mà vẫn không qua được Verifier ($b > B$), hệ thống sẽ dừng lại và trả về báo cáo lỗi chi tiết (`FAILUREREPORT`), tránh việc agent sa vào vòng lặp vô hạn.

---

## PHẦN 2: Làm rõ Hệ thống Prompt (Phụ lục D)

Phụ lục D cung cấp các System Prompts nguyên bản định hình hành vi cho từng tác nhân trong hệ thống HEART, tách bạch rõ ràng nhiệm vụ của từng agent:

### 1. Planner Prompt (Tác nhân Hoạch định & Kiểm tra thông tin)
- **Nhiệm vụ:** Phân tích ý đồ (*Intent Analysis*) và kiểm tra xem thông tin đã đủ chưa.
- **Cơ chế hoạt động:**
  - Nếu thiếu thông tin, bắt buộc xuất ra JSON có trạng thái `"INSUFFICIENT"` kèm câu hỏi làm rõ.
  - Nếu đủ, xuất ra danh sách các bước gọi công cụ theo định dạng JSON (`"status": "SUFFICIENT"`, kèm mảng `plan` gồm các bước có `dependencies`).
  - Khi nhận feedback lỗi từ `Verifier`, `Planner` đọc lỗi đó để viết lại kế hoạch mới được đánh dấu cờ `"replanned": true`.

### 2. Router Prompt (Tác nhân Định tuyến & Ánh xạ tham số)
- **Nhiệm vụ:** Giải quyết bài toán liên kết tham số (*parameter mapping*) từ ngữ cảnh và đóng gói thành yêu cầu dạng ngôn ngữ tự nhiên.
- **Cơ chế hoạt động:**
  - Không bắt `Router` phải nhớ raw schema phức tạp, mà yêu cầu `Router` trích xuất giá trị từ ngữ cảnh (truy vấn người dùng, kết quả bước trước như `$step_j`) và nhúng trực tiếp vào yêu cầu dạng ngôn ngữ tự nhiên (`invocation_request`).
  - Gắn kèm các siêu tham số thực thi (`h_params`) như số lần thử lại (`retry` từ 1 đến 5), độ ưu tiên (`priority`: `low`/`normal`/`high`) và thời gian chờ (`timeout_s`).

### 3. Tool Primitive Prompt (Lớp bọc công cụ thông minh)
- **Nhiệm vụ:** Đóng vai trò là cầu nối dịch từ ngôn ngữ tự nhiên sang schema API thô bên trong từng công cụ.
- **Cơ chế hoạt động:**
  - Nhận yêu cầu ngôn ngữ tự nhiên từ `Router` $\rightarrow$ tự phân giải schema, ép kiểu dữ liệu (*type coercion*), chuẩn hóa ngày tháng, kiểm tra ràng buộc tham số.
  - Gọi hàm thực thi bên dưới và trả về JSON chuẩn hóa gồm `status: SUCCESS` kèm kết quả và một đoạn tóm tắt (`summary`) bằng ngôn ngữ tự nhiên để làm ngữ cảnh cho bước tiếp theo.
  - **Nguyên tắc cô lập lỗi (Execution Isolation):** `Tool Primitive` tuyệt đối không tự ý retry hay gọi tool khác khi lỗi xảy ra, mà bắt buộc phải trả về `status: FAILURE` để đẩy việc xử lý lỗi về cho `Verifier`.

### 4. Verifier Prompt (Tác nhân Kiểm định & Phản hồi)
- **Nhiệm vụ:** Đóng vai trò "trọng tài" đánh giá kết quả thực thi $r_k$ và sinh feedback nếu có lỗi.
- **Cơ chế hoạt động:** Chấm điểm độc lập qua **4 tiêu chí**:
  1. **Task Completion** (Hoàn thành tác vụ).
  2. **Argument Consistency** (Tính nhất quán của tham số so với ý định ban đầu).
  3. **Execution Validity** (Tính hợp lệ khi thực thi không có lỗi runtime).
  4. **Constraint Satisfaction** (Thỏa mãn các quy tắc kinh doanh/giới hạn hệ thống).
- Chỉ khi cả 4 tiêu chí đạt `pass`, `Verifier` mới cho phép đi tiếp. Nếu có bất kỳ tiêu chí nào `fail`, `Verifier` xuất ra JSON phản hồi mang tính hành động cao, giúp `Planner` có dữ liệu chính xác để sửa sai ở vòng kế hoạch tiếp theo.
