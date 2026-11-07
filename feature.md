# Tích hợp Tool-Boundary Middleware (PRISM Harness Engineering) cho LifeMate

Tài liệu này ghi lại các nâng cấp đã thực hiện dựa trên phương pháp **Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives (PRISM)** nhằm tối ưu hóa độ chính xác, tính ổn định và khả năng tổng quát hóa (zero-shot schema) cho mô hình nhỏ (`Qwen3-4B-Instruct-2507`) trong bài toán **LifeMate — Zalo AI Challenge 2026**.

---

## 1. Động lực & Bài toán trong LifeMate

Trong cuộc thi LifeMate:
- **Giới hạn tham số nghiêm ngặt:** Toàn hệ thống `< 6B` tham số, riêng LLM `< 5B` tham số và thời gian chạy `< 30 giây/mẫu` trên 1 GPU NVIDIA A30.
- **Cơ chế chấm điểm nghiêm ngặt:**
  - Sai `state_hash` cuối cùng $\rightarrow$ **0 điểm** cho cả mẫu.
  - Mọi lệnh gọi vào `env.call()` (kể cả các lệnh gọi lỗi nghiệp vụ) đều bị ghi vào `env.get_tool_calls()`, làm giảm trực tiếp **Tool-Call Precision / F1**.
  - Tập Public/Private Test xuất hiện các ứng dụng và công cụ hoàn toàn mới (ví dụ 11 app mới và 34 tool mới ở Public Test) không có trong tập Train.
- **Giải pháp theo hướng PRISM:** Tối ưu hóa lớp vỏ điều khiển thời gian chạy (**Harness**) trên 2 bề mặt:
  1. **System Prompt ($p_1$ — Global Policy):** Định hướng luồng suy luận tổng quát (tra cứu trước khi ghi, xử lý phản hồi lỗi).
  2. **Tool-Boundary Middleware ($p_2$ — Local Invariants):** Chặn ngay tại ranh giới trước khi gọi `env.validate()` và `env.call()` để tự động sửa lỗi cấu trúc xác định (**Silent Correction**) và kiểm tra điều kiện tiên quyết / trạng thái dữ liệu (**Prerequisite & State Tracking**).

---

## 2. Kiến trúc luồng xử lý mới (`src/lifemate/runner.py`)

```text
LLM Output (raw text)
   │
   ▼
parse_action() [src/lifemate/models.py]
   │  (Bóc tách JSON, mặc định arguments={}, gỡ chuỗi JSON bọc ngoài arguments)
   ▼
silent_correct_action() [src/lifemate/middleware.py]
   │  1. Chuẩn hóa tên tool (tự ghép prefix <app>. nếu duy nhất trong installed_apps)
   │  2. Ánh xạ lại tên tham số sai lệch (Argument Name Remapping)
   │  3. Xóa các tham số optional có giá trị null/None
   │  4. Gỡ lớp bọc JSON lồng nhau & Ép kiểu dữ liệu theo Schema (Type Coercion)
   │  5. Chuẩn hóa Unicode NFC & khớp giá trị Enum (Enum Snapping)
   ▼
check_local_invariants() [src/lifemate/middleware.py]
   │  1. Kiểm tra Whitelist: Tool có thuộc available_tools của mẫu không?
   │  2. Chặn gọi lặp (Anti-Retry-Loop): Lệnh gọi đã thực thi ở bước trước chưa?
   │  3. Kiểm tra điều kiện tiên quyết (x-ref): ID truyền vào đã được tra cứu chưa?
   │  4. Kiểm tra bất nhất chéo (Cross-Entity Consistency): Cặp ID cha-con có khớp cùng thực thể không?
   ▼
env.validate(name, arguments) [zaic_tools]
   │
   ├── Nếu bị chặn bởi Middleware hoặc Schema Error:
   │     Trả về thông báo lỗi tường minh qua `validation_feedback` để LLM thử lại 1 lần.
   │     (Tuyệt đối không chạm vào env.call() nên không làm bẩn lịch sử gọi hàm).
   │
   └── Nếu hợp lệ 100%:
         Gọi env.call(name, arguments) ──► Lưu kết quả vào steps_this_request
```

---

## 3. Chi tiết các tính năng đã triển khai

### 3.1. Lớp Silent Correction (`src/lifemate/middleware.py`)
Tự động sửa ngầm các lỗi định dạng/cấu trúc của LLM mà không tốn thêm lượt sinh token:

- **Chuẩn hóa tên công cụ (`silent_correct_action`):**
  - Nếu LLM quên ghi tiền tố ứng dụng (ví dụ gọi `"create_alarm"` thay vì `"clock.create_alarm"`) và trong `available_tools` của mẫu chỉ có duy nhất một công cụ khớp hậu tố `.create_alarm`, Middleware tự động ánh xạ về đúng tên đầy đủ.
- **Ánh xạ tên tham số bị lệch (`_remap_argument_keys`):**
  - Chuẩn hóa chữ thường và dấu gạch ngang/gạch dưới (ví dụ: `"order-id"` hoặc `"orderId"` $\rightarrow$ `"order_id"`).
  - Tự động ánh xạ giữa `"id"` và `"<entity>_id"` khi schema chỉ có một trường ID duy nhất chưa được điền.
- **Loại bỏ tham số `null` không bắt buộc (`coerce_arguments_by_schema`):**
  - LLM thường sinh các trường không bắt buộc với giá trị `null` (ví dụ `"note": null`), gây lỗi khi kiểm tra JSON Schema nghiêm ngặt. Middleware tự động lược bỏ các trường `None` nếu chúng không nằm trong danh sách `required`.
- **Gỡ vỏ bọc JSON và Ép kiểu theo Schema (`coerce_value_by_schema`, `_coerce_scalar`):**
  - **JSON Unwrapping:** Tự động `json.loads` nếu tham số kiểu `object` hoặc `array` bị truyền dưới dạng chuỗi JSON.
  - **Type Coercion:**
    - `integer`: `"5"` hoặc `5.0` $\rightarrow$ `5`.
    - `number`: `"5.5"` $\rightarrow$ `5.5`.
    - `boolean`: `"true"`, `"1"`, `"yes"` $\rightarrow$ `True`; `"false"`, `"0"`, `"no"` $\rightarrow$ `False`.
    - `string`: `5996159312` $\rightarrow$ `"5996159312"`, đồng thời chuẩn hóa **Unicode NFC** (đảm bảo khớp chính xác và tăng điểm `token_f1` cho các trường `x-match: "fuzzy"`).
    - `array`: Tự động bọc giá trị đơn lẻ `val` thành `[val]` nếu schema yêu cầu mảng, và ép kiểu đệ quy cho từng phần tử bên trong.
  - **Enum Snapping (`_snap_enum`):** Khớp không phân biệt hoa/thường và chuẩn hóa NFC với danh sách `enum` trong schema để trả về đúng chuỗi chuẩn.

---

### 3.2. Lớp Prerequisite & State Tracking (`src/lifemate/middleware.py`)
Ngăn chặn các lỗi ngữ nghĩa và trạng thái qua nhiều bước hội thoại trước khi lệnh gọi chạm tới `env.call()`:

- **Kiểm tra Whitelist ứng dụng được cài:**
  - Chặn ngay nếu LLM gọi tool không nằm trong `context.installed_apps` (hoặc `sys.ask_missing_info`), đồng thời liệt kê các tool hợp lệ của cùng ứng dụng trong thông báo lỗi để LLM chọn lại.
- **Chặn vòng lặp gọi trùng (Duplicate Call Blocking):**
  - Phát hiện khi LLM gọi lại chính xác cùng một `name` và `arguments` đã được thực thi thành công ở các bước trước trong cùng lượt yêu cầu (`prior_calls`), buộc LLM sử dụng kết quả đã có trong `steps_this_request` hoặc trả về `{"action": "stop"}`.
- **Kiểm tra điều kiện tiên quyết `x-ref` (Proactive Lookup Rule):**
  - Thu thập toàn bộ các giá trị/ID đã xuất hiện trong `context`, `history` (cả `tool_calls` và `tool_results`), `transcript` và `prior_results` của các bước trước.
  - Khi LLM gọi một công cụ có tham số đánh dấu `"x-ref"` trong catalog nhưng giá trị truyền vào chưa từng xuất hiện trong dữ liệu quan sát được (tức LLM tự bịa/đoán ID mà chưa gọi công cụ tra cứu), Middleware chặn lệnh gọi và trả về hướng dẫn tường minh: liệt kê các công cụ `kind="read"` của ứng dụng đó để LLM gọi tra cứu trước.
- **Kiểm tra bất nhất dữ liệu chéo giữa các thực thể (`_check_cross_entity_consistency`):**
  - Giải quyết bài toán LLM lấy nhầm ID con của thực thể này gắn vào ID cha của thực thể khác (tương tự lỗi ghép nhầm `item_id` của đơn hàng `#W4689314` vào đơn hàng `#W8855135` trong bài báo PRISM).
  - Khi lệnh gọi có từ 2 tham số `x-ref` trở lên, Middleware đối chiếu các bản ghi (records) đã trả về từ các bước tra cứu trước. Nếu phát hiện tham số con không nằm trong bản ghi của tham số cha mà thuộc về một bản ghi anh em khác, Middleware chặn lại và chỉ rõ ID thực thể cha đúng trong phản hồi lỗi.

---

### 3.3. Nâng cấp Parser và System Prompt

- **`src/lifemate/models.py` (`parse_action`):**
  - Hỗ trợ trường hợp LLM lược bỏ khóa `"arguments"` khi gọi các hàm không có tham số (tự động gán `{}`).
  - Hỗ trợ gỡ chuỗi JSON nếu LLM xuất `"arguments"` dưới dạng string thay vì object.
- **`src/lifemate/prompts.py` (`SYSTEM_PROMPT`):**
  - Bổ sung chính sách toàn cục ($p_1$): yêu cầu gọi công cụ tra cứu (`kind="read"`) trước công cụ thao tác (`kind="write"`) đối với các tham số có `x-ref`, không tự đoán ID, và ưu tiên sửa lỗi theo `validation_feedback` khi có phản hồi từ Middleware/Schema.

---

## 4. Danh sách các file đã thay đổi

| File | Trạng thái | Mô tả thay đổi |
|---|---|---|
| `src/lifemate/middleware.py` | **Tạo mới** | Cài đặt toàn bộ `silent_correct_action`, `coerce_arguments_by_schema`, `check_local_invariants`, và `_check_cross_entity_consistency`. |
| `src/lifemate/runner.py` | **Cập nhật** | Thêm `_intercept_and_validate()` vào `_action_for_step()` để kết nối Middleware với vòng lặp suy luận và cơ chế retry 1 lần qua `validation_feedback`. |
| `src/lifemate/models.py` | **Cập nhật** | Nâng cấp `parse_action()` xử lý `arguments` rỗng hoặc bị bọc chuỗi JSON. |
| `src/lifemate/prompts.py` | **Cập nhật** | Cập nhật `SYSTEM_PROMPT` đồng bộ với cơ chế kiểm tra `x-ref` và `validation_feedback`. |


---

### 3.4. Lớp PRISM Analyst, Định tuyến Bề mặt & Ma trận Lỗi Xuyên Thế Hệ (`src/lifemate/prism_analyst.py` & `training/evaluate_dev.py`)

Triển khai cơ chế chẩn đoán tự động và chống suy thoái xuyên thế hệ của PRISM ngay trong bộ đánh giá cục bộ (`training/evaluate_dev.py`):

1. **Bản tóm tắt ca lỗi chuẩn PRISM (`build_case_diagnostic`):**
   - Với mỗi mẫu trong `train-dev`, tự động trích xuất 4 trường cốt lõi:
     - `case_id`: ID mẫu (`sample["id"]`).
     - `stuck_type`: Tự động phát hiện `timeout` (`> 30s`), `max_steps_exceeded`, `infinite_tool_loop` (gọi trùng hàm + tham số), `crash`, hoặc `none`.
     - `tool_call_sequence` & `gold_tool_call_sequence`: Chuỗi lệnh gọi thực tế so với nhãn chuẩn.
     - `truncated_state_diff`: So sánh chi tiết trạng thái các ứng dụng bị lệch (`actual` vs `expected` được cắt gọn 400 ký tự) khi `state_match = False`.
2. **Phân cụm nguyên nhân gốc rễ & Định tuyến bề mặt sửa lỗi (`cluster_failures_for_prism`):**
   - Tự động gom các ca thất bại (loại bỏ lỗi crash hạ tầng) thành tối đa **4 cụm lỗi lớn nhất** và định tuyến về 3 bề mặt tương ứng với 3 khe đột biến ($p_0, p_1, p_2$):
     - `middleware_only` (Khe $p_2$): Lỗi lặp vòng lặp gọi tool, gọi đúng tên tool nhưng sai kiểu dữ liệu/tên trường/ID thực thể.
     - `prompt_only` (Khe $p_1$): Lỗi gọi tool dư thừa ở các mẫu ngoài phạm vi (`Out-of-Scope`), hoặc chọn sai công cụ so với ngữ cảnh.
     - `prompt_middleware_both` (Khe $p_0$): Lỗi thiếu bước tra cứu chủ động (`proactive lookup`) trước khi ghi hoặc quên gọi `sys.ask_missing_info` khi thiếu thông tin bắt buộc.
3. **Ma trận lỗi xuyên thế hệ (`compute_cross_generation_matrix`):**
   - Khi chạy `training/evaluate_dev.py --prev-report reports/runs/dev_base.json ...`, hệ thống tự động đối chiếu từng `case_id` qua các thế hệ và phân loại vào 4 nhóm:
     - `FIXED`: Ca lỗi ở thế hệ trước đã được sửa thành công (cần bảo vệ tuyệt đối).
     - `NEW`: Ca trước đó đang đúng nhưng bị hỏng ở bản vá hiện tại (cảnh báo hồi quy — `regression_safe = False`).
     - `RECURRING`: Ca lỗi tái diễn cần ưu tiên sửa ở thế hệ tiếp theo.
     - `PERSISTENT`: Ca lỗi cố hữu qua nhiều thế hệ liên tiếp.

| File bổ sung | Trạng thái | Mô tả thay đổi |
|---|---|---|
| `src/lifemate/prism_analyst.py` | **Tạo mới** | Cài đặt `build_case_diagnostic`, `cluster_failures_for_prism` (4 cụm root-cause + routing `prompt_only` / `middleware_only` / `prompt_middleware_both`) và `compute_cross_generation_matrix` (`FIXED`, `NEW`, `RECURRING`, `PERSISTENT`). |
| `training/evaluate_dev.py` | **Cập nhật** | Tích hợp `prism_analyst_clusters`, `cross_generation_matrix` và cờ `--prev-report` vào báo cáo đánh giá `dev_metrics.json`. |


---

### 3.5. Hỗ trợ Ablation `PRISM-PO` vs `PRISM-MW`, Kiểm soát Prompt Bloat (Khe $p_1$) & Lai ghép Crossover

1. **Chuyển đổi linh hoạt giữa `PRISM-MW` và `PRISM-PO` (`configs/baseline.json`, `src/lifemate/runner.py`, `training/evaluate_dev.py`):**
   - Bổ sung cấu hình `"harness": {"mode": "prism_mw", "enable_middleware": true, "max_p1_added_sentences": 2}` trong `configs/baseline.json`.
   - Bổ sung cờ `--disable-middleware` trong `training/evaluate_dev.py` để dễ dàng chạy thực nghiệm đối chứng (Ablation Study) giữa:
     - **Chế độ `PRISM-PO` (Prompt-Only):** `python training/evaluate_dev.py --disable-middleware --report reports/runs/dev_prism_po.json`
     - **Chế độ `PRISM-MW` (Prompt + Middleware):** `python training/evaluate_dev.py --report reports/runs/dev_prism_mw.json`
2. **Hàm kiểm tra giới hạn câu của Khe $p_1$ (`validate_p1_prompt_mutation` trong `src/lifemate/prism_analyst.py`):**
   - Kiểm tra số câu tăng thêm giữa `parent_prompt` và `candidate_prompt`, đảm bảo mỗi thế hệ đột biến ở khe $p_1$ chỉ thêm tối đa $\le 2$ câu quy tắc hành vi phổ quát, chống hiện tượng phình ngữ cảnh (prompt bloat).
3. **Hàm phát hiện cơ hội Lai ghép (`evaluate_crossover_opportunity` trong `src/lifemate/prism_analyst.py`):**
   - Tự động đối chiếu tập `case_id` giải được giữa ứng viên khe $p_1$ (`p1_prompt_report`) và ứng viên khe $p_2$ (`p2_middleware_report`). Khi cả hai có các ca giải đúng độc lập bổ sung cho nhau (`uniquely_solved_by_p1_prompt` và `uniquely_solved_by_p2_middleware`), hàm trả về `should_trigger_crossover = True` cùng danh sách ca được hưởng lợi để hợp nhất bản vá.


---

### 3.6. Tích hợp 3 Kỹ thuật từ bài báo HEART (Tool Primitives, ToolFace & Verifier)

Để giải quyết 2 rào cản lớn được chỉ ra trong bài báo **HEART (`LLM Tool Use via Agent-Native Reusable Tool Primitives.md`)** — gồm *suy luận đa bước dễ gãy khi truyền kết quả giữa các tool* và *suy giảm hiệu năng khi danh mục công cụ lớn (lên tới 62 tool / 18 app mỗi mẫu trong LifeMate)* — hệ thống đã bổ sung 3 cơ chế **Zero-LLM-Overhead** (không tốn thêm lượt gọi LLM, giữ vững ngân sách `< 30s` trên A30):

1. **Nén Schema & Sắp xếp Ưu tiên theo Độ liên quan — cảm hứng từ ToolFace (`_compact_schema` & `_rank_tool_specs` trong `src/lifemate/prompts.py`):**
   - **Nén Schema (`_compact_schema`):** Tự động loại bỏ các trường boilerplate dư thừa của JSON Schema (`additionalProperties`, `title`, `$schema`, các trường rỗng `""`/`[]`/`{}`) nhưng bảo toàn 100% thông tin ngữ nghĩa (`properties`, `required`, `enum`, `x-ref`, `x-one-of`, `x-match`). Giúp giảm lượng token đầu vào và tăng tốc độ prefill trên A30.
   - **Sắp xếp chống "Lost in the Middle" (`_rank_tool_specs`):** Thay vì liệt kê 60 công cụ theo thứ tự bảng chữ cái cứng nhắc khiến công cụ quan trọng bị chìm ở giữa prompt, hệ thống tính điểm khớp từ khóa giữa `(request + history + prior_calls)` với tên/mô tả của từng ứng dụng, đưa các ứng dụng liên quan nhất lên đầu `available_tools` và xếp các công cụ `kind="read"` đứng trước `kind="write"` trong cùng ứng dụng.

2. **Bộ định tuyến tham số liên bước — cảm hứng từ HEART Router (`_resolve_xref_labels_to_ids` & `extract_inter_tool_bindings` trong `src/lifemate/middleware.py`):**
   - **Tự động phân giải Tên tự nhiên sang ID (`_resolve_xref_labels_to_ids`):** Khi gọi chuỗi công cụ phụ thuộc nhau, nếu LLM truyền tên/nhãn ngôn ngữ tự nhiên (ví dụ `"Báo thức sáng"` hoặc `"Họp nhóm"`) vào tham số `x-ref` thay vì mã ID kỹ thuật (`"alm_01"`, `"evt_1"`), Middleware tự động đối chiếu với kết quả của các bước `read` trước đó (`prior_results` và `history`) và ánh xạ chính xác sang ID duy nhất tương ứng.
   - **Trích xuất sẵn bảng ánh xạ ID (`extract_inter_tool_bindings`):** Tự động rút trích các cặp `id` / `*_id` từ kết quả trả về không đồng nhất của từng bước vào trường `"resolved_refs"` ngay trong `steps_this_request`, giúp Qwen-4B nhìn thấy ngay các ID khả dụng cho bước kế tiếp mà không bị lạc trong JSON lồng sâu.

3. **Bộ kiểm định sau thực thi — cảm hứng từ HEART Verifier (`verify_step_execution` trong `src/lifemate/middleware.py`):**
   - Ngay sau khi `env.call()` trả về kết quả, Verifier đánh giá tính hợp lệ của bước thực thi và gắn trường `"verifier_note"` có cấu trúc vào `steps_this_request` trong các tình huống:
     - **Lỗi thực thi nghiệp vụ (`status == "error"` / `ok == False`):** Nhắc Agent kiểm tra lại tham số, đổi công cụ tra cứu hoặc gọi `sys.ask_missing_info`, tuyệt đối không gọi lặp lại y hệt.
     - **Tra cứu trả về rỗng (`read` trả về `[]`):** Cảnh báo Agent không được tự bịa ID cho công cụ `write` ở bước tiếp theo, hướng dẫn mở rộng điều kiện tìm kiếm hoặc hỏi thêm thông tin từ người dùng.


---

### 3.7. Ngân sách Tái lập kế hoạch $B = 3$ (HEART Algorithm 1) & Bộ kiểm định 4 tiêu chí (Phụ lục D)

- **Kiểm soát ngân sách tái lập kế hoạch $B = 3$ (`max_replan_budget` trong `src/lifemate/runner.py` & `configs/baseline.json`):**
  - Áp dụng đúng cơ chế $b \leftarrow b + 1$ và điều kiện dừng $b > B$ ($B = 3$) của **Algorithm 1**: mỗi lần Agent phải lập lại hành động do lỗi chặn ở Middleware/Schema hoặc lỗi hậu kiểm ở Verifier, bộ đếm `replan_state["used"]` tăng lên 1 kèm cờ `"replanned": true`. Nếu vượt quá $B = 3$ lần trong một mẫu, Agent dừng ngay lập tức để tránh vòng lặp vô hạn và đảm bảo thời gian chạy `< 30s` trên A30.
- **Chuẩn hóa Ngày/Giờ tự động trong Tool Primitive (`_coerce_scalar` trong `src/lifemate/middleware.py`):**
  - Tự động chuẩn hóa định dạng giờ `"6:30"` $\rightarrow$ `"06:30"` và ngày `"2026/10/25"` $\rightarrow$ `"2026-10-25"` theo đúng mô tả của Tool Primitive trong Phụ lục D.3.
- **Chấm điểm Verifier theo 4 tiêu chí độc lập (`evaluate_4_criteria_verifier` trong `src/lifemate/middleware.py`):**
  - Đánh giá kết quả thực thi $r_k$ trên 4 tiêu chí: `task_completion`, `argument_consistency`, `execution_validity`, `constraint_satisfaction` và xuất phản hồi chẩn đoán có cấu trúc $\phi_k$ (`feedback_phi`).


---

### 3.8. Đồ thị Tri thức Hai phần (Bipartite Knowledge Graph) & Thuật toán wRRF (`src/lifemate/graph_retrieval.py`)

Triển khai toàn bộ 3 trụ cột từ bài báo **[Agent-as-a-Graph](Agent-as-a-GraphTool%20and%20Ag%20Retrieval%20for%20LLM%20Multi-Agent%20Systems.md)** để tối ưu hóa việc truy xuất và sắp xếp công cụ động:

1. **Đồ thị hai phần $\mathcal{G} = (\mathcal{A}, \mathcal{T}, \mathcal{E})$ (`BipartiteToolGraph`):**
   - Biểu diễn các ứng dụng được cài đặt làm tập nút Agent/App ($\mathcal{A}$ — `AppNode`) và các công cụ làm tập nút Tool ($\mathcal{T}$ — `ToolNode`), liên kết qua cạnh sở hữu $\text{owner}(T) = A$.
2. **Xếp hạng hợp nhất có trọng số loại wRRF (`rank_tools_wrrf`):**
   - Tính điểm xếp hạng riêng biệt trên kho App ($\mathcal{C}_A$) và kho Tool ($\mathcal{C}_T$) bằng độ tương đồng Cosine từ vựng (unigram + bigram tiếng Việt chuẩn hóa NFC) cộng điểm duy trì ngữ cảnh hội thoại, sau đó hợp nhất bằng công thức wRRF với bộ trọng số chuẩn của bài báo ($\alpha_A = 1.5, \alpha_T = 1.0$).
3. **Duyệt đồ thị ngược `Tool -> App` & Cắt tỉa theo ngân sách token (`_pop_lowest_wrrf_tool`):**
   - Lan truyền điểm wRRF từ các nút Tool con về nút App chủ quản ($T \rightarrow A$) để gom trọn cụm công cụ của ứng dụng liên quan nhất lên đầu `available_tools` (luôn xếp `read` trước `write`).
   - Hỗ trợ cắt tỉa an toàn các công cụ ở đáy bảng xếp hạng wRRF trong `fit_messages()` (`src/lifemate/prompts.py`) và `trim_history()` (`training/train_lora.py`) khi vượt ngân sách token trên GPU nhỏ.
