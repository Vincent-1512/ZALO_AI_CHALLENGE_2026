# Tổng hợp Ứng dụng của Từng Paper vào Hệ thống LifeMate (Zalo AI Challenge 2026)

Tài liệu này phân tích chi tiết **từng bài báo (paper) đã tham khảo**, **vấn đề thực tế trong LifeMate mà nó giải quyết**, **những gì đã áp dụng trực tiếp vào code**, và **những gì chủ động lược bỏ/biến đổi** để tuân thủ giới hạn `< 5B` tham số LLM và `< 30 giây/mẫu` trên GPU NVIDIA A30.

---

## Tổng quan: Sự phân vai của 3 Paper trong Pipeline LifeMate

Ba bài báo không chồng chéo nhau mà ghép nối thành **3 tầng bảo vệ liên tiếp** quanh mô hình `Qwen3-4B-Instruct-2507`:

```text
[TẦNG 1: TRƯỚC KHI ĐƯA VÀO LLM] ──► Paper 3: Agent-as-a-Graph
  • Dựng đồ thị 2 phần App–Tool, xếp hạng wRRF (α_A=1.5, α_T=1.0), duyệt ngược Tool -> App.
  • Đưa đúng cụm ứng dụng cần dùng (read đứng trước write) lên đầu prompt, cắt tỉa đuôi khi tràn token.
        │
        ▼
[TẦNG 2: ĐIỀU PHỐI CHUỖI ĐA BƯỚC & HẬU KIỂM] ──► Paper 2: HEART
  • Nén gọn JSON Schema (ToolFace), trích xuất sẵn bảng ID "resolved_refs" giữa các bước.
  • Phân giải tên tự nhiên ("Họp nhóm") sang mã ID kỹ thuật ("evt_1") (Router / Tool Primitives).
  • Chấm điểm kết quả thực thi theo 4 tiêu chí (Verifier) & khóa ngân sách tái lập kế hoạch B = 3.
        │
        ▼
[TẦNG 3: KIỂM SOÁT BIÊN CÔNG CỤ & CHẨN ĐOÁN LỖI] ──► Paper 1: PRISM
  • Silent Correction: tự sửa kiểu dữ liệu ("5" -> 5), gỡ JSON lồng, ánh xạ tên tham số, xóa trường null.
  • Local Invariants: chặn đoán mò ID (x-ref), chặn ghép nhầm ID cha-con giữa các thực thể, chặn gọi lặp.
  • Analyst & Failure Matrix: tự động gom 4 cụm lỗi gốc rễ và theo dõi FIXED / NEW / RECURRING trên tập dev.
```

---

## 1. Paper 1: PRISM — *Harness Engineering in LLM Tool Use*
- **Tài liệu gốc:** [Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives.md](Harness%20Engineering%20in%20LLM%20Tool%20Use%20via%20Agent-Native%20Reusable%20Tool%20Primitives.md) & thư mục [agent-harness-optimizer/](agent-harness-optimizer/README.md)
- **File code đã áp dụng:** [`src/lifemate/middleware.py`](src/lifemate/middleware.py), [`src/lifemate/prism_analyst.py`](src/lifemate/prism_analyst.py), [`training/evaluate_dev.py`](training/evaluate_dev.py)

### Những gì bạn áp dụng được từ PRISM:
1. **Phân tách 2 bề mặt tối ưu (Prompt $p_1$ vs. Middleware $p_2$):**
   - **System Prompt ($p_1$ — Global Policy):** Chỉ giữ các quy tắc suy luận toàn cục (luồng `read` trước `write`, khi nào gọi `sys.ask_missing_info`, khi nào trả `stop`) và kiểm soát chống phình ngữ cảnh ($\le 2$ câu thêm mới mỗi thế hệ qua `validate_p1_prompt_mutation()`).
   - **Tool-Boundary Middleware ($p_2$ — Local Invariants):** Dùng code Python xác định 100% để bắt các lỗi cơ học mà Prompt thường thất bại.
2. **Silent Correction (Tự động sửa ngầm trước khi chạm `env.validate()`):**
   - **Argument Name Remapping (`_remap_argument_keys`):** Tự động sửa tên biến viết sai chuẩn (`order-id`, `orderId` $\rightarrow$ `order_id`) hoặc ánh xạ `id` $\leftrightarrow$ `<entity>_id`.
   - **Type Coercion & JSON Unwrapping (`coerce_value_by_schema`):** Tự động gỡ chuỗi JSON bị bọc kép; ép kiểu `"5"` $\rightarrow$ `5`, `"true"` $\rightarrow$ `True`, bọc phần tử đơn thành mảng `[val]`, chuẩn hóa Unicode NFC và khớp `enum` không phân biệt hoa/thường.
   - **Null Optional Stripping:** Xóa các tham số không bắt buộc mà LLM điền `"param": null` để không bị trượt JSON Schema.
3. **Prerequisite & Cross-Entity State Blocking (`check_local_invariants`):**
   - **Chặn đoán mò ID (`x-ref` Prerequisite Rule):** Nếu LLM truyền một mã ID vào tham số có cờ `"x-ref"` mà mã đó chưa từng xuất hiện trong `prior_results` hay `history`, Middleware chặn đứng trước khi gọi `env.call()` và nhắc LLM gọi công cụ `read` của ứng dụng đó trước.
   - **Chặn lấy nhầm ID giữa các thực thể (`_check_cross_entity_consistency`):** Khi có nhiều bản ghi (ví dụ đơn hàng `#W4689314` và `#W8855135`), nếu LLM lấy `item_id` của đơn hàng A gắn vào `order_id` của đơn hàng B, Middleware phát hiện và báo rõ cặp ID đúng để sửa ngay.
4. **PRISM Analyst & Ma trận lỗi xuyên thế hệ (`src/lifemate/prism_analyst.py`):**
   - Tự động gom các ca sai trên tập `train-dev` thành **4 cụm nguyên nhân gốc rễ (`prism_analyst_clusters`)** và định tuyến về `prompt_only`, `middleware_only`, hoặc `prompt_middleware_both`.
   - Lập ma trận **`FIXED`, `NEW`, `RECURRING`, `PERSISTENT`** và phát hiện cơ hội **Lai ghép (`evaluate_crossover_opportunity`)** giữa nhánh sửa Prompt và nhánh sửa Middleware.

---

## 2. Paper 2: HEART — *LLM Tool Use via Agent-Native Reusable Tool Primitives*
- **Tài liệu gốc:** [LLM Tool Use via Agent-Native Reusable Tool Primitives.md](LLM%20Tool%20Use%20via%20Agent-Native%20Reusable%20Tool%20Primitives.md)
- **File code đã áp dụng:** [`src/lifemate/middleware.py`](src/lifemate/middleware.py), [`src/lifemate/prompts.py`](src/lifemate/prompts.py), [`src/lifemate/runner.py`](src/lifemate/runner.py)

### Những gì bạn áp dụng được từ HEART:
1. **Nén gọn Schema công cụ — cảm hứng từ ToolFace (`_compact_schema`):**
   - Lược bỏ các trường boilerplate dư thừa của JSON Schema (`additionalProperties`, `title`, `$schema`, các trường rỗng) giúp giảm 15–25% lượng token đầu vào mà giữ nguyên 100% thông tin ngữ nghĩa.
2. **Định tuyến & Phân giải Tham số Liên bước — cảm hứng từ HEART Router & Tool Primitives:**
   - **Phân giải Tên tự nhiên sang ID (`_resolve_xref_labels_to_ids`):** Khi gọi chuỗi công cụ phụ thuộc (`read` $\rightarrow$ `write`), nếu LLM truyền tên hiển thị tiếng Việt (như `"Họp nhóm"`, `"Báo thức sáng"`) vào trường yêu cầu ID (`event_id`, `alarm_id`), hệ thống tự động tra cứu trong kết quả bước trước và đổi sang đúng mã ID kỹ thuật (`"evt_1"`, `"alm_01"`).
   - **Chuẩn hóa Ngày/Giờ tự động (`_coerce_scalar`):** Tự động chuẩn hóa `"6:30"` $\rightarrow$ `"06:30"` và `"2026/10/25"` $\rightarrow$ `"2026-10-25"`.
   - **Trích xuất bảng ánh xạ ID phẳng (`extract_inter_tool_bindings`):** Tự động rút trích mọi khóa `id` / `*_id` từ kết quả JSON lồng sâu của bước trước ra trường `"resolved_refs"` ngay trong `steps_this_request`.
3. **Bộ kiểm định 4 tiêu chí sau thực thi — HEART Verifier (`evaluate_4_criteria_verifier`):**
   - Sau mỗi lệnh `env.call()`, tự động kiểm tra 4 tiêu chí (*Task Completion, Argument Consistency, Execution Validity, Constraint Satisfaction*). Nếu gặp lỗi nghiệp vụ hoặc công cụ `read` trả về danh sách rỗng `[]`, tự động gắn `"verifier_note"` nhắc Agent không được bịa ID và hướng dẫn đổi hướng tra cứu hoặc gọi `sys.ask_missing_info`.
4. **Ngân sách tái lập kế hoạch $B = 3$ — HEART Algorithm 1 (`max_replan_budget = 3`):**
   - Khóa trần tổng số lần sửa lỗi / tái lập kế hoạch trong một mẫu ở mức $B = 3$ (`replan_state` trong `_run_actions`). Khi $b > 3$, Agent dừng an toàn ngay lập tức để chống lặp vô hạn và đảm bảo không bao giờ vượt quá 30 giây/mẫu trên A30.

> **Điểm tinh chỉnh quan trọng so với bản gốc HEART:** Bản gốc HEART dùng 4 LLM riêng biệt (Planner, Router, Primitive, Verifier). Để không vi phạm luật `< 5B` tham số và `< 30s` trên A30 của LifeMate, chúng ta chuyển hóa Router, Primitive và Verifier thành các thuật toán xác định (**Zero-LLM-Overhead**), chỉ giữ duy nhất 1 LLM (`Qwen3-4B`) đóng vai trò Planner.

---

## 3. Paper 3: Agent-as-a-Graph — *Tool and Agent Retrieval for LLM Multi-Agent Systems*
- **Tài liệu gốc:** [Agent-as-a-GraphTool and Ag Retrieval for LLM Multi-Agent Systems.md](Agent-as-a-GraphTool%20and%20Ag%20Retrieval%20for%20LLM%20Multi-Agent%20Systems.md)
- **File code đã áp dụng:** [`src/lifemate/graph_retrieval.py`](src/lifemate/graph_retrieval.py), [`src/lifemate/prompts.py`](src/lifemate/prompts.py), [`training/train_lora.py`](training/train_lora.py)

### Những gì bạn áp dụng được từ Agent-as-a-Graph:
1. **Đồ thị Tri thức Hai phần $\mathcal{G} = (\mathcal{A}, \mathcal{T}, \mathcal{E})$ (`BipartiteToolGraph`):**
   - Biểu diễn các ứng dụng được cài đặt làm tập nút Agent/App ($\mathcal{A}$ — `AppNode`) và các công cụ cụ thể làm tập nút Tool ($\mathcal{T}$ — `ToolNode`), nối với nhau qua cạnh sở hữu $\text{owner}(T) = A$.
   - Tự động dựng lại theo từng mẫu từ `catalog` động, nhờ đó **xử lý zero-shot hoàn toàn cho cả 11 ứng dụng mới (34 công cụ mới)** ở tập Public/Private Test.
2. **Thuật toán xếp hạng hợp nhất có trọng số loại wRRF (`rank_tools_wrrf`):**
   - Xếp hạng truy vấn (unigram + bigram tiếng Việt chuẩn hóa NFC từ `request` và `history`) độc lập trên cả kho App ($\mathcal{C}_A$) và kho Tool ($\mathcal{C}_T$).
   - Áp dụng đúng công thức wRRF với bộ trọng số tối ưu từ bài báo: **$\alpha_A = 1.5, \alpha_T = 1.0$**:
     $$s_{\text{wRRF}}(e) = \begin{cases} \dfrac{1.0}{k + r_T(e)}, & e \in \mathcal{C}_T \\[6pt] \dfrac{1.5}{k + r_A(e)}, & e \in \mathcal{C}_A \end{cases}$$
3. **Duyệt đồ thị ngược `Tool -> App` (Graph Traversal):**
   - Khi câu nói của người dùng khớp với một công cụ thao tác chi tiết $t \in \mathcal{T}$ (ví dụ `orders.return_item`), phép duyệt ngược $t \rightarrow \text{owner}(t) = a$ kéo toàn bộ cụm công cụ của ứng dụng $a$ lên đầu danh sách `available_tools`, và sắp xếp **các công cụ `kind="read"` đứng ngay trước `kind="write"`**.
   - Giúp mô hình không bao giờ bị mất công cụ tra cứu tiền đề và giải quyết triệt để hiện tượng *"lost in the middle"* khi mẫu cài tới 18 ứng dụng (62 công cụ).
4. **Cắt tỉa đuôi an toàn theo ngân sách Token (`_pop_lowest_wrrf_tool`):**
   - Khi chạy trên các mẫu quá dài hoặc khi train QLoRA trên GPU nhỏ (như Colab T4 với `max_length = 2048`), nếu đã cắt hết `history` mà vẫn vượt ngân sách token, hệ thống tự động cắt từ công cụ có điểm wRRF thấp nhất ở đáy danh sách lên, bảo vệ tuyệt đối các cụm ứng dụng đứng đầu và `sys.ask_missing_info`.

