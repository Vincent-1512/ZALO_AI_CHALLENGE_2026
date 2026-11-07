không gian tối ưu hóa là Harness — lớp điều khiển thời gian chạy bao quanh mô hình cố định. Bài báo giới hạn bề mặt tìm kiếm vào 2 thành phần:

System Prompts: Giải quyết các chính sách tổng quát, luồng hội thoại và quy tắc nghiệp vụ.

Tool-Boundary Middleware (Middleware tại ranh giới công cụ): Các hàm chặn (intercepts) có kiểm soát ngay trước/sau khi gọi API để kiểm tra điều kiện cục bộ, chuẩn hóa dữ liệu mà không làm thay đổi cấu trúc thực thi cốt lõi của agent.
Prompt + Tool-Boundary Middleware (các hàm chặn kiểm tra/sửa lỗi ngay tại ranh giới gọi API theo 3 mẫu an toàn cố định) và dùng cơ chế định tuyến lỗi (routing) để quyết định lỗi nào cần sửa ở Prompt, lỗi nào cần chặn ở Middleware.
ối ưu hóa lớp vỏ điều khiển (Harness) xung quanh một mô hình nhỏ/nhanh cố định (như gpt-5.4-mini) dưới các giới hạn nghiêm ngặt về số lượt gọi, thời gian phản hồi và lượng token, giúp mô hình nhỏ đạt được độ chính xác và độ ổn định vượt qua cả mô hình lớn chưa tối ưu Harness với chi phí vận hành chỉ bằng một phần nhỏ
PRISM sinh ra các quy tắc Silent Correction (tự động sửa ngầm trước khi lệnh gọi chạm tới hàm thực thi):

Đổi tên tham số bị sai lệch (Argument Name Remapping):

Lỗi của LLM: Khi gọi công cụ tính toán, mô hình thường tự suy diễn và truyền tên biến là expression (một tên biến rất phổ biến trong dữ liệu huấn luyện) thay vì tên chuẩn theo yêu cầu của API là math_expression.

Cách Middleware xử lý: Chặn gói tin JSON ngay tại biên công cụ, kiểm tra nếu tồn tại khóa expression nhưng thiếu math_expression, middleware tự động ánh xạ giá trị sang math_expression trước khi đẩy vào hàm đích.

Ép kiểu dữ liệu và gỡ vỏ bọc JSON (Type Coercion & JSON Unwrapping):

Lỗi của LLM: Truyền một số nguyên dưới dạng chuỗi ký tự (ví dụ: "5" thay vì 5), hoặc bọc tham số trong một lớp chuỗi JSON dư thừa (nested JSON string).

Cách Middleware xử lý: Tự động phân tích cú pháp (parse) để gỡ lớp bọc JSON thừa và ép kiểu dữ liệu về đúng định dạng schema quy định.

Hiệu quả thực tế: Đây là các lỗi cấu trúc hoàn toàn xác định (deterministic). Việc sửa thẳng bằng code tại Middleware giúp PRISM-MW tăng +14.2 điểm phần trăm trên BFCL, bỏ xa mức +5.8 điểm phần trăm của phiên bản chỉ sửa câu lệnh (PRISM-PO).
, lỗi không còn nằm ở cú pháp JSON đơn giản mà nằm ở sự bất nhất về trạng thái dữ liệu qua nhiều bước hội thoại.
Bối cảnh lỗi (xuất hiện ở 8% quỹ đạo baseline): Khi một khách hàng có nhiều đơn hàng đã giao, mô hình rất dễ lấy nhầm mã sản phẩm (item_id) của đơn hàng này gắn vào lệnh thao tác (đổi/trả/hủy) của đơn hàng khác. Ví dụ: lấy sản phẩm 5996159312 thuộc đơn hàng #W4689314 nhưng lại truyền vào hàm xử lý cho đơn hàng #W8855135.

Tại sao Prompt thất bại? Khi xảy ra lỗi, hệ thống backend chỉ trả về một thông báo lỗi chung chung (opaque error), khiến LLM không hiểu nguyên nhân và rơi vào vòng lặp thử lại (retry loop) vô ích. Dù Prompt có dặn "Hãy kiểm tra kỹ sản phẩm có thuộc đơn hàng không", LLM chỉ nhìn thấy văn bản hội thoại dài dòng phía trước nên vẫn bị nhầm lẫn.
Cách Middleware xử lý (Prerequisite + Error Blocking):

Theo dõi trạng thái (State Tracking): Mỗi khi hàm get_order_details được gọi và trả về kết quả ở các bước trước, Middleware âm thầm trích xuất dữ liệu và xây dựng một bảng ánh xạ chuẩn xác trong bộ nhớ: {order_id -> danh_sách_item_ids}.

Chặn và phản hồi tường minh: Khi LLM gọi hàm thao tác trên đơn hàng #W8855135 với sản phẩm 5996159312, Middleware đối chiếu với bảng ánh xạ, chặn đứng lệnh gọi ngay lập tức (không cho gửi tới backend) và trả về cho LLM một thông báo lỗi chi tiết: "Lỗi: Sản phẩm 5996159312 không thuộc đơn hàng #W8855135 mà thuộc đơn hàng #W4689314". Nhờ thông tin cụ thể này, Agent sửa lại đúng tham số ngay trong lượt tiếp theo.

Tình huống B: Quy tắc tra cứu chủ động khi khách hàng quên mã đơn (Proactive Lookup Rule)
Bối cảnh lỗi: Khách hàng yêu cầu hỗ trợ nhưng không nhớ mã đơn hàng (order_id). Ở nhánh chỉ tối ưu Prompt (PRISM-PO trên seed 1), mọi lần chạy đều dậm chân tại chỗ ở mức 7/20 ca thành công trên tập holdout (ngang bằng baseline).

Cách PRISM-MW xử lý: Bằng cách kết hợp cả hai bề mặt (Prompt hướng dẫn luồng hỏi thông tin định danh + Middleware kiểm tra điều kiện tiên quyết xem đã tra cứu danh sách đơn hàng của user trước khi gọi hàm chi tiết hay chưa), PRISM-MW buộc Agent phải thực hiện bước tra cứu hồ sơ khách hàng để lấy danh sách order_id hợp lệ trước khi thực hiện các bước tiếp theo.
Tóm tắt sự phân vai giữa Prompt và Middleware trong PRISMBề mặt can thiệpVí dụ lỗi điển hình trong bài báoBản chất xử lýMiddleware (p_2)Truyền expression thay vì math_expression (BFCL); truyền item_id của đơn #W4689314 vào đơn #W8855135 (Retail).Bất biến cục bộ (Local Invariants): Kiểm tra và sửa lỗi chính xác 100% bằng code tại thời điểm gọi hàm.Prompt (p_1)Chưa biết khi nào nên tự xử lý tiếp (refuel_data), khi nào nên chuyển tiếp cho nhân viên hỗ trợ trên $\tau^2$-Telecom (Mục D.3).Chính sách toàn cục (Global Policy): Cung cấp danh sách kiểm tra (checklist) theo thứ tự để định hướng suy luận của LLM.


---

## Cơ chế tiến hóa của PRISM: Analyst LLM, 3 Khe Đột Biến ($p_0, p_1, p_2$) & Lai Ghép (Crossover)

### 1. Cơ chế Analyst LLM: Phân cụm theo nguyên nhân gốc rễ và Định tuyến bề mặt
Ở mỗi thế hệ (generation), sau khi chọn ra cá thể cha (parent harness) tốt nhất từ tập biên Pareto, PRISM thu thập toàn bộ các ca thất bại trên tập sửa lỗi ($\mathcal{D}_{\text{repair}}$) và đưa vào Analyst LLM (`claude-opus-4.7`).

- **Đầu vào (Input) của Analyst LLM:** Bản tóm tắt có cấu trúc gồm 4 trường cho mỗi ca thất bại:
  - `case_id`: Mã định danh của ca kiểm thử.
  - `stuck_type`: Loại vi phạm tài nguyên nếu có (`timeout`, vượt số lượt tối đa `max_steps_exceeded`, lặp vô hạn `infinite_tool_loop`, hoặc `crash`).
  - `tool_call_sequence`: Chuỗi các hàm và tham số mà Agent đã thực sự gọi.
  - `truncated_state_diff` / `error_message`: Sự sai lệch giữa trạng thái môi trường cuối cùng của Agent so với trạng thái đích (oracle state), hoặc thông báo lỗi trực tiếp từ hệ thống.
- **Quy tắc chẩn đoán (Prompting Rules):**
  1. **Phân cụm theo nguyên nhân gốc rễ (Root Cause)**, không phân cụm theo biểu hiện bề mặt.
  2. **Loại bỏ nhiễu hạ tầng:** Tự động bỏ qua các ca lỗi do sập API hoặc lỗi mạng (`infrastructure-error cases`).
  3. **Ưu tiên bề mặt đơn (Single-surface preference):** Chỉ định sửa trên một bề mặt duy nhất (`prompt_only` hoặc `middleware_only`) nếu có thể; chỉ dùng `prompt_middleware_both` khi lỗi thực sự đòi hỏi sự phối hợp của cả hai.
- **Đầu ra (Output JSON):** Đúng 4 cụm lỗi (clusters), sắp xếp theo số lượng ca lỗi giảm dần:
```json
[
  {
    "root_cause": "Mô tả nguyên nhân gốc rễ gây ra thất bại của nhóm ca này",
    "repair_surface": "prompt_only | middleware_only | prompt_middleware_both",
    "case_ids": ["case_12", "case_45", "case_78"],
    "reason": "Giải thích trong 1 câu tại sao lại chọn bề mặt sửa chữa này"
  }
]
```

### 2. Cơ chế hoạt động của 3 Khe đột biến ($p_0, p_1, p_2$)
PRISM khởi chạy 3 khe đột biến ($p_0, p_1, p_2$) song song trong mỗi thế hệ (`mutations_per_gen = 3`), mỗi khe bị ràng buộc bởi một **Surface Manifest**:

| Khe đột biến | Bề mặt được phép sửa (Surface Manifest) | Cụm lỗi đầu vào được định tuyến (Target Pattern) | Quy tắc & Ràng buộc thực thi (Access & Safety Rules) |
|---|---|---|---|
| **Khe $p_0$ (Joint Slot)** | `system_prompt.txt` VÀ `middleware.py` | Toàn bộ các ca lỗi, bao gồm các cụm `prompt_middleware_both`. | Tìm và áp dụng 1 bản vá an toàn có giá trị cao nhất xuyên qua một hoặc cả hai bề mặt. |
| **Khe $p_1$ (Prompt Slot)** | Chỉ `system_prompt.txt` | Nhận `case_ids` từ các cụm `prompt_only` (vẫn chạy nếu không có cụm `prompt_only` để duy trì đa dạng). | Chỉ viết quy tắc hành vi phổ quát; giới hạn thêm tối đa 1–2 câu mỗi thế hệ để chống phình ngữ cảnh (prompt bloat). |
| **Khe $p_2$ (Middleware Slot)** | Chỉ `middleware.py` | Nhận `case_ids` từ các cụm `middleware_only`. | Khóa chặt vào 3 mẫu can thiệp tại biên công cụ: (1) Silent correction, (2) Error blocking, (3) Prerequisite blocking. |

### 3. Không gian làm việc (Mutator Workspace) và Phanh an toàn chống suy thoái
Trước khi mỗi khe chạy, PRISM kết xuất Workspace gồm:
- **Trạng thái hiện tại:** `system_prompt` và code `middleware` của cá thể cha.
- **Ma trận lỗi xuyên thế hệ (Cross-generation Failure Matrix):** Đánh dấu từng ca qua các đời (`NEW`, `RECURRING`, `PERSISTENT`, `FIXED`).
- **Lịch sử tiến hóa:** Các ứng viên trên biên Pareto và danh sách bản vá đã được chấp nhận (`accepted`) hoặc bị từ chối (`rejected`).
- **Quỹ đạo đối chiếu (Trajectories):** Cả quỹ đạo đang lỗi (`target failure trajectories`) VÀ quỹ đạo đang đúng (`passing trajectories`) để kiểm tra hồi quy.

**4 Quy tắc an toàn bắt buộc:**
1. **Ưu tiên lỗi tái diễn (`RECURRING`):** Tập trung giải quyết các lỗi lặp đi lặp lại.
2. **Bỏ qua lỗi cố hữu (`PERSISTENT`):** Không cố sửa các ca thất bại liên tục qua nhiều thế hệ trừ khi có hướng tiếp cận hoàn toàn mới.
3. **Không lặp lại hướng đã thất bại:** Đọc lịch sử các ứng viên bị từ chối.
4. **Bảo vệ tuyệt đối ca đã sửa (`FIXED`):** Kiểm tra chéo với `passing trajectories` để đảm bảo không gây hồi quy (`NEW` failure).

### 4. Đánh giá & Lai ghép có điều kiện (Conditional Crossover)
Sau khi $p_0, p_1, p_2$ sinh ra 3 cá thể con và đánh giá trên $\mathcal{D}_{\text{repair}}$ và $\mathcal{D}_{\text{gate}}$:
- Nếu cá thể con tốt nhất (ví dụ từ khe $p_2$ - Middleware) đạt điểm cao nhất nhưng thất bại ở ca $X$, trong khi cá thể con từ khe $p_1$ (Prompt) giải quyết thành công ca $X$, PRISM kích hoạt **Crossover** để hợp nhất bản vá Prompt của $p_1$ và bản vá Middleware của $p_2$ thành một cá thể lai mới trước khi cập nhật tập biên Pareto.


---

## Cơ chế tiến hóa của PRISM: Analyst LLM, 3 Khe Đột Biến ($p_0, p_1, p_2$) & Lai Ghép (Crossover)

### 1. Cơ chế Analyst LLM: Phân cụm theo nguyên nhân gốc rễ và Định tuyến bề mặt
Ở mỗi thế hệ (generation), sau khi chọn ra cá thể cha (parent harness) tốt nhất từ tập biên Pareto, PRISM thu thập toàn bộ các ca thất bại trên tập sửa lỗi ($\mathcal{D}_{\text{repair}}$) và đưa vào Analyst LLM (`claude-opus-4.7`).

- **Đầu vào (Input) của Analyst LLM:** Bản tóm tắt có cấu trúc gồm 4 trường cho mỗi ca thất bại:
  - `case_id`: Mã định danh của ca kiểm thử.
  - `stuck_type`: Loại vi phạm tài nguyên nếu có (`timeout`, vượt số lượt tối đa `max_steps_exceeded`, lặp vô hạn `infinite_tool_loop`, hoặc `crash`).
  - `tool_call_sequence`: Chuỗi các hàm và tham số mà Agent đã thực sự gọi.
  - `truncated_state_diff` / `error_message`: Sự sai lệch giữa trạng thái môi trường cuối cùng của Agent so với trạng thái đích (oracle state), hoặc thông báo lỗi trực tiếp từ hệ thống.
- **Quy tắc chẩn đoán (Prompting Rules):**
  1. **Phân cụm theo nguyên nhân gốc rễ (Root Cause)**, không phân cụm theo biểu hiện bề mặt.
  2. **Loại bỏ nhiễu hạ tầng:** Tự động bỏ qua các ca lỗi do sập API hoặc lỗi mạng (`infrastructure-error cases`).
  3. **Ưu tiên bề mặt đơn (Single-surface preference):** Chỉ định sửa trên một bề mặt duy nhất (`prompt_only` hoặc `middleware_only`) nếu có thể; chỉ dùng `prompt_middleware_both` khi lỗi thực sự đòi hỏi sự phối hợp của cả hai.
- **Đầu ra (Output JSON):** Đúng 4 cụm lỗi (clusters), sắp xếp theo số lượng ca lỗi giảm dần:
```json
[
  {
    "root_cause": "Mô tả nguyên nhân gốc rễ gây ra thất bại của nhóm ca này",
    "repair_surface": "prompt_only | middleware_only | prompt_middleware_both",
    "case_ids": ["case_12", "case_45", "case_78"],
    "reason": "Giải thích trong 1 câu tại sao lại chọn bề mặt sửa chữa này"
  }
]
```

### 2. Cơ chế hoạt động của 3 Khe đột biến ($p_0, p_1, p_2$)
PRISM khởi chạy 3 khe đột biến ($p_0, p_1, p_2$) song song trong mỗi thế hệ (`mutations_per_gen = 3`), mỗi khe bị ràng buộc bởi một **Surface Manifest**:

| Khe đột biến | Bề mặt được phép sửa (Surface Manifest) | Cụm lỗi đầu vào được định tuyến (Target Pattern) | Quy tắc & Ràng buộc thực thi (Access & Safety Rules) |
|---|---|---|---|
| **Khe $p_0$ (Joint Slot)** | `system_prompt.txt` VÀ `middleware.py` | Toàn bộ các ca lỗi, bao gồm các cụm `prompt_middleware_both`. | Tìm và áp dụng 1 bản vá an toàn có giá trị cao nhất xuyên qua một hoặc cả hai bề mặt. |
| **Khe $p_1$ (Prompt Slot)** | Chỉ `system_prompt.txt` | Nhận `case_ids` từ các cụm `prompt_only` (vẫn chạy nếu không có cụm `prompt_only` để duy trì đa dạng). | Chỉ viết quy tắc hành vi phổ quát; giới hạn thêm tối đa 1–2 câu mỗi thế hệ để chống phình ngữ cảnh (prompt bloat). |
| **Khe $p_2$ (Middleware Slot)** | Chỉ `middleware.py` | Nhận `case_ids` từ các cụm `middleware_only`. | Khóa chặt vào 3 mẫu can thiệp tại biên công cụ: (1) Silent correction, (2) Error blocking, (3) Prerequisite blocking. |

### 3. Không gian làm việc (Mutator Workspace) và Phanh an toàn chống suy thoái
Trước khi mỗi khe chạy, PRISM kết xuất Workspace gồm:
- **Trạng thái hiện tại:** `system_prompt` và code `middleware` của cá thể cha.
- **Ma trận lỗi xuyên thế hệ (Cross-generation Failure Matrix):** Đánh dấu từng ca qua các đời (`NEW`, `RECURRING`, `PERSISTENT`, `FIXED`).
- **Lịch sử tiến hóa:** Các ứng viên trên biên Pareto và danh sách bản vá đã được chấp nhận (`accepted`) hoặc bị từ chối (`rejected`).
- **Quỹ đạo đối chiếu (Trajectories):** Cả quỹ đạo đang lỗi (`target failure trajectories`) VÀ quỹ đạo đang đúng (`passing trajectories`) để kiểm tra hồi quy.

**4 Quy tắc an toàn bắt buộc:**
1. **Ưu tiên lỗi tái diễn (`RECURRING`):** Tập trung giải quyết các lỗi lặp đi lặp lại.
2. **Bỏ qua lỗi cố hữu (`PERSISTENT`):** Không cố sửa các ca thất bại liên tục qua nhiều thế hệ trừ khi có hướng tiếp cận hoàn toàn mới.
3. **Không lặp lại hướng đã thất bại:** Đọc lịch sử các ứng viên bị từ chối.
4. **Bảo vệ tuyệt đối ca đã sửa (`FIXED`):** Kiểm tra chéo với `passing trajectories` để đảm bảo không gây hồi quy (`NEW` failure).

### 4. Đánh giá & Lai ghép có điều kiện (Conditional Crossover)
Sau khi $p_0, p_1, p_2$ sinh ra 3 cá thể con và đánh giá trên $\mathcal{D}_{\text{repair}}$ và $\mathcal{D}_{\text{gate}}$:
- Nếu cá thể con tốt nhất (ví dụ từ khe $p_2$ - Middleware) đạt điểm cao nhất nhưng thất bại ở ca $X$, trong khi cá thể con từ khe $p_1$ (Prompt) giải quyết thành công ca $X$, PRISM kích hoạt **Crossover** để hợp nhất bản vá Prompt của $p_1$ và bản vá Middleware của $p_2$ thành một cá thể lai mới trước khi cập nhật tập biên Pareto.


---

## Phân vai giữa Prompt và Middleware, Chế độ `-PO` vs `-MW` & Ràng buộc an toàn Khe $p_1$

### 1. Vai trò phân chia công việc: Prompt xử lý "Chính sách toàn cục", Middleware xử lý "Ràng buộc cục bộ"
Prompt không phải là chiếc đũa thần có thể giải quyết mọi loại lỗi của agent. Sự phân vai rõ ràng giữa hai bề mặt được quy định như sau:
- **Prompt (Bề mặt chính sách toàn cục - Global Policy):** Phù hợp để giải quyết các lỗi liên quan đến chiến lược tổng thể của tác vụ, tư duy suy luận, cách hướng dẫn agent thu thập thông tin, tuân thủ quy trình dịch vụ, hoặc khi nào cần chuyển tiếp/hỏi lại người dùng (ví dụ: danh sách kiểm tra policy checklist trong bài toán chăm sóc khách hàng viễn thông $\tau^2$-Telecom, hoặc `sys.ask_missing_info` trong LifeMate).
- **Middleware (Bề mặt ranh giới công cụ - Local Constraints):** Phù hợp hơn cho các lỗi cơ học mang tính xác định tại runtime như sai tên tham số, sai kiểu dữ liệu JSON, hoặc nhầm lẫn mã sản phẩm/đơn hàng (`x-ref`). Cố gắng bắt prompt gánh vác các lỗi này thường kém hiệu quả hơn việc dùng code middleware chặn lại.

### 2. Hai chế độ tối ưu hóa: Prompt-Only (`-PO`) và Prompt + Middleware (`-MW`)
- **Các biến thể Prompt-Only (`-PO`):** Như `PRISM-PO`, `BH-PO`, `GEPA-PO`, `MIPROv2-PO`. Không gian tìm kiếm bị giới hạn hoàn toàn trong việc biên tập lại `system_prompt.txt` (tiến hóa văn bản, phản hồi quỹ đạo — trajectory reflection, hoặc tối ưu hóa hướng dẫn).
- **Các biến thể Prompt + Middleware (`-MW`):** Cho phép tác động đồng thời vào cả prompt và middleware (`PRISM-MW`). Kết quả thực nghiệm cho thấy kết hợp cả hai luôn đem lại độ tin cậy và mức cải thiện ($\text{RelLift}_{95}$) vượt trội so với chỉ dùng prompt đơn thuần, đặc biệt ở các benchmark đòi hỏi gọi hàm chính xác như BFCL.

### 3. Ràng buộc an toàn đối với Prompt trong khe đột biến $p_1$ của PRISM
Khi bộ định tuyến giao việc cho khe sửa prompt riêng biệt ($p_1$), PRISM áp dụng các quy tắc kiểm soát ngặt nghèo để tránh tình trạng prompt bị "phình to" (prompt bloat) hoặc làm suy thoái mô hình:
- **Giới hạn dung lượng:** Khe $p_1$ chỉ được phép viết thêm tối đa **1 đến 2 câu** (hoặc quy tắc hành vi phổ quát ngắn gọn) vào system prompt trong mỗi thế hệ đột biến.
- **Chống suy thoái:** Việc sửa prompt phải vượt qua các bài kiểm tra hồi quy bằng cách đối chiếu với các quỹ đạo chạy đúng trước đó (`passing trajectories`), đảm bảo việc thêm quy tắc mới không làm hỏng các ca xử lý đã thành công từ trước (`FIXED` / `STABLE_PASS`).
