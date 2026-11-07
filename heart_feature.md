# Tích hợp Kiến trúc HEART (Tool Primitives, ToolFace & Verifier) cho LifeMate

Tài liệu này mô tả chi tiết các nâng cấp được thiết kế và tích hợp từ bài báo **[LLM Tool Use via Agent-Native Reusable Tool Primitives](LLM%20Tool%20Use%20via%20Agent-Native%20Reusable%20Tool%20Primitives.md)** (kiến trúc **HEART**) vào hệ thống **LifeMate — Zalo AI Challenge 2026**.

---

## 1. Bối cảnh & Hai nút thắt từ bài báo HEART trong LifeMate

Theo bài báo HEART, ngay cả các mô hình được SFT tốt vẫn gặp 2 điểm nghẽn lớn khi gọi công cụ thực tế:

1. **Suy luận đa bước/đa lượt dễ gãy (Brittle multi-step & multi-turn reasoning):**
   - Trong LifeMate, phần lớn mẫu cần nhiều bước gọi tool phụ thuộc nhau (ví dụ: gọi tool `read` để tìm sự kiện/báo thức/đơn hàng, sau đó lấy ID từ kết quả để truyền vào tool `write`).
   - Do kết quả trả về từ các ứng dụng khác nhau có định dạng JSON lồng sâu không đồng nhất, mô hình nhỏ (`Qwen3-4B`) dễ truyền nhầm **tên/tiêu đề ngôn ngữ tự nhiên** vào trường yêu cầu **mã định danh (`x-ref` ID)** hoặc bị lạc mất ID khi ngữ cảnh hội thoại dài tới 24 lượt.
2. **Suy giảm hiệu năng khi danh mục công cụ lớn (Large tool catalogues degradation):**
   - Mỗi mẫu trong LifeMate có thể cài tới **18 ứng dụng (62 công cụ)**. Việc đưa toàn bộ raw JSON Schema của 62 công cụ theo thứ tự bảng chữ cái vào prompt gây ra hiện tượng *"lost in the middle"* (công cụ cần dùng bị chìm ở giữa prompt) và làm tăng thời gian prefill trên GPU NVIDIA A30.

> **Nguyên tắc thiết kế cho LifeMate:** Chắt lọc tinh hoa của 3 thành phần trong HEART (**ToolFace**, **Router / Tool Primitives**, và **Verifier**) dưới dạng **Zero-LLM-Overhead** (xử lý 100% bằng thuật toán xác định ở runtime, không gọi thêm lượt LLM phụ) nhằm giữ vững giới hạn `< 30 giây/mẫu` trên A30 và `< 5B` tham số.

---

## 2. Sơ đồ kiến trúc HEART kết hợp trong LifeMate

```text
Yêu cầu người dùng (Transcript) + History + Context
   │
   ▼
[1. ToolFace Dynamic Compaction & Ranking] (src/lifemate/prompts.py)
   ├── _compact_schema(): Lược bỏ boilerplate JSON Schema, giữ 100% ngữ nghĩa
   └── _rank_tool_specs(): Xếp ứng dụng khớp ngữ cảnh lên đầu + ưu tiên read trước write
   │
   ▼
Qwen3-4B Policy (Planner) ──► Sinh quyết định JSON: call(name, arguments) | stop
   │
   ▼
[2. HEART Router & Tool Primitive Adapter] (src/lifemate/middleware.py)
   └── _resolve_xref_labels_to_ids():
         Nếu LLM truyền tên/nhãn tự nhiên ("Báo thức sáng") vào tham số x-ref,
         tự động tra cứu trong prior_results/history để đổi sang đúng ID ("alm_01")
   │
   ▼
env.validate() & env.call() (Thực thi trên ToolEnv)
   │
   ▼
[3. HEART Post-Execution Verifier & Binding Extractor] (src/lifemate/middleware.py)
   ├── extract_inter_tool_bindings(): Rút trích phẳng các cặp {id, *_id} vào "resolved_refs"
   └── verify_step_execution(): Kiểm định lỗi thực thi / kết quả rỗng -> gắn "verifier_note"
   │
   └──► Đưa "resolved_refs" + "verifier_note" vào steps_this_request cho bước kế tiếp
```

---

## 3. Chi tiết kỹ thuật các thành phần đã triển khai

### 3.1. ToolFace-Inspired: Nén Schema & Sắp xếp Ưu tiên theo Ngữ cảnh (`src/lifemate/prompts.py`)

- **Nén Schema công cụ (`_compact_schema`):**
  - Loại bỏ các trường siêu dữ liệu dư thừa trong JSON Schema (`additionalProperties`, `title`, `$schema`) và các trường có giá trị rỗng (`""`, `[]`, `{}`).
  - Bảo toàn tuyệt đối các trường quyết định hành vi gọi hàm: `properties`, `type`, `description`, `required`, `enum`, `items`, `x-ref`, `x-one-of`, `x-match`.
  - **Tác dụng:** Giảm kích thước prompt từ 15–25% trên các mẫu có nhiều ứng dụng, giảm độ trễ prefill trên A30 và giúp mô hình tập trung vào các ràng buộc thực sự.

- **Xếp hạng động chống "Lost in the Middle" (`_rank_tool_specs`):**
  - Tính điểm độ liên quan của từng ứng dụng dựa trên:
    1. Độ trùng khớp từ khóa (sau khi chuẩn hóa Unicode NFC) giữa `(request + 4 lượt history gần nhất)` với tên và mô tả công cụ.
    2. Cộng điểm ưu tiên (+5) cho các ứng dụng đang hoạt động (`active_apps`) đã xuất hiện trong `prior_calls` của lượt hiện tại hoặc 2 lượt history gần nhất.
  - Sắp xếp danh sách `available_tools` theo thứ tự:
    - Các ứng dụng có điểm liên quan cao nhất đứng đầu tiên.
    - Trong cùng một ứng dụng, các công cụ tra cứu (`kind="read"`) luôn được đặt trước công cụ thao tác (`kind="write"`), định hướng tự nhiên cho mô hình thực hiện chuỗi **Read $\rightarrow$ Write**.
    - Công cụ hệ thống `sys.ask_missing_info` luôn giữ ở vị trí cố định cuối danh sách để dễ chú ý.

---

### 3.2. HEART Router: Phân giải Ngôn ngữ Tự nhiên sang ID & Liên kết Tham số (`src/lifemate/middleware.py`)

Hiện thực hóa triết lý của **Tool Primitives** (*"Thay vì bắt mô hình nói ngôn ngữ ID thô của API, lớp bọc tự phân giải ngôn ngữ tự nhiên sang tham số chuẩn"*):

- **Tự động phân giải Tên/Nhãn sang Mã định danh (`_resolve_xref_labels_to_ids`):**
  - Khi công cụ đích có tham số gắn cờ `"x-ref"` (ví dụ `alarm_id`, `event_id`, `contact_id`, `item_id`), LLM đôi khi truyền tên hiển thị mà người dùng vừa nhắc tới (như `"Họp nhóm"`, `"Báo thức sáng"`) thay vì mã ID kỹ thuật (`"evt_1"`, `"alm_01"`).
  - Hàm `_resolve_xref_labels_to_ids` quét toàn bộ các bản ghi thực thể trong `prior_results` và `history`. Nếu giá trị truyền vào chưa phải là ID nhưng khớp duy nhất với một trường tên/nhãn (`name`, `title`, `label`, `summary`, `contact_name`, `recipient`, `subject`, `item_name`) của một bản ghi đã tra cứu, hệ thống tự động thay thế bằng mã ID chính xác của bản ghi đó trước khi đưa vào `env.validate()` và `env.call()`.

- **Trích xuất phẳng bảng ánh xạ ID liên bước (`extract_inter_tool_bindings`):**
  - Sau mỗi bước gọi hàm trong lượt hiện tại, hàm tự động quét đệ quy kết quả trả về (`result`) và gom tất cả các khóa định danh (`id` hoặc `*_id`) thành một từ điển phẳng `"resolved_refs"` đính kèm ngay trong mục tương ứng của `steps_this_request`.
  - Ví dụ:
    ```json
    {
      "call": {"name": "calendar.get_events", "arguments": {"date": "2026-10-25"}},
      "result": {"status": "ok", "events": [{"id": "evt_1", "title": "Họp nhóm"}]},
      "resolved_refs": {"id": ["evt_1"]}
    }
    ```
  - Giúp Qwen3-4B nhìn thấy ngay mã ID cần điền cho bước tiếp theo mà không bị nhầm lẫn khi `result` có cấu trúc lồng nhiều tầng.

---

### 3.3. HEART Verifier: Kiểm định Sau Thực Thi & Phục hồi Có Cấu Trúc (`src/lifemate/middleware.py`)

Sau khi `env.call()` thực thi xong, hàm `verify_step_execution(call, result, catalog)` kiểm tra kết quả dựa trên các tiêu chí của HEART Verifier và bổ sung trường `"verifier_note"` vào `steps_this_request` nếu phát hiện bất thường:

1. **Kiểm định tính hợp lệ thực thi (Execution Validity):**
   - Khi công cụ trả về trạng thái lỗi (`status` là `"error"` / `"failed"`, `ok == False`, hoặc có trường `"error"`), Verifier sinh cảnh báo nhắc Planner kiểm tra lại đối số, đổi công cụ tra cứu hoặc gọi `sys.ask_missing_info`, ngăn chặn hành vi gọi lặp lại lệnh lỗi.
2. **Kiểm định tra cứu rỗng (Empty Read Recovery):**
   - Khi một công cụ tra cứu (`kind="read"`) trả về danh sách rỗng `[]` (không tìm thấy bản ghi nào khớp điều kiện), Verifier lập tức gắn cảnh báo: *"Tuyệt đối không tự bịa ID cho bước tiếp theo; hãy thử mở rộng điều kiện tìm kiếm hoặc gọi sys.ask_missing_info."*
   - Cơ chế này phối hợp chặt chẽ với hàng rào chặn `x-ref` trước thực thi để triệt tiêu hoàn toàn lỗi **Hallucinated ID** trong chuỗi suy luận đa bước.

---

## 4. Bảng tổng hợp các file liên quan

| File | Các hàm chính thuộc kiến trúc HEART | Vai trò |
|---|---|---|
| [`src/lifemate/prompts.py`](src/lifemate/prompts.py) | `_compact_schema()`, `_rank_tool_specs()`, `_build_step_entry()` | Nén JSON Schema, xếp hạng động công cụ theo ngữ cảnh (ToolFace) và đính kèm `resolved_refs` / `verifier_note` vào prompt. |
| [`src/lifemate/middleware.py`](src/lifemate/middleware.py) | `_resolve_xref_labels_to_ids()`, `extract_inter_tool_bindings()`, `verify_step_execution()` | Phân giải tên tự nhiên sang ID cho tham số `x-ref` (Router), rút trích ID liên bước (Tool Primitives) và kiểm định kết quả sau thực thi (Verifier). |
| [`src/lifemate/runner.py`](src/lifemate/runner.py) | `_intercept_and_validate()` | Kết nối `prior_results` vào luồng tiền xử lý để Router phân giải tham số giữa các bước trong cùng lượt yêu cầu. |



---

## 5. Cập nhật theo Algorithm 1 (Ngân sách Tái lập kế hoạch $B=3$) & Phụ lục D

Bổ sung các cơ chế điều phối từ **Algorithm 1** và **Phụ lục D** của bài báo HEART vào runtime của LifeMate:

1. **Ngân sách tái lập kế hoạch $B = 3$ (`max_replan_budget: 3` trong `configs/baseline.json` & `_run_actions` trong `src/lifemate/runner.py`):**
   - Duy trì biến đếm ngân sách lỗi/tái lập kế hoạch $b$ (`replan_state["used"]`) xuyên suốt vòng đời xử lý một mẫu với trần $B = 3$.
   - Khi một bước bị chặn trước thực thi (bởi Middleware hoặc Schema) hoặc bị đánh trượt sau thực thi bởi Verifier (`not verdict["passed"]`), hệ thống tăng $b \leftarrow b + 1$ và đính kèm cờ `"replanned": true`, `"replan_budget_used": "b/3"` cùng phản hồi lỗi $\phi_k$ vào `validation_feedback`.
   - Ngay khi $b > B$ ($b > 3$), vòng lặp lập tức dừng an toàn (`return`), bảo vệ tuyệt đối ngân sách thời gian `< 30 giây/mẫu` trên GPU A30 và tránh lặp vô hạn.

2. **Chuẩn hóa Ngày/Giờ trong Tool Primitive (`_coerce_scalar` trong `src/lifemate/middleware.py`):**
   - Hiện thực hóa vai trò chuẩn hóa ngày tháng của **Tool Primitive (Phụ lục D.3)**: tự động chuẩn hóa chuỗi giờ thiếu số 0 ở đầu (`"6:30"` $\rightarrow$ `"06:30"`) và chuẩn hóa định dạng ngày dùng dấu gạch chéo (`"2026/10/25"` $\rightarrow$ `"2026-10-25"`).

3. **Bộ chấm điểm 4 tiêu chí độc lập của Verifier (`evaluate_4_criteria_verifier` trong `src/lifemate/middleware.py`):**
   - Chấm điểm từng bước thực thi qua đúng 4 tiêu chí của **Phụ lục D.4**:
     - `task_completion`
     - `argument_consistency`
     - `execution_validity`
     - `constraint_satisfaction`
   - Trả về cấu trúc `{passed, criteria, feedback_phi}` để cả `prompts.py` (tạo `verifier_note`) và `runner.py` (quản lý ngân sách tái lập kế hoạch $B$) cùng sử dụng thống nhất.
