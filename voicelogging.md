# Hướng dẫn Kỹ thuật: Kiến trúc Xử lý Giọng nói & Điều phối Luồng Hội thoại Đa lượt trong LifeMate

Tài liệu này chuẩn hóa kiến trúc **Voice-to-Action Pipeline** của hệ thống **LifeMate — Zalo AI Challenge 2026**, bao gồm: nhận dạng giọng nói theo lô (**Batched PhoWhisper ASR**), quản lý ngữ cảnh hội thoại đa lượt (**Multi-Turn Audio History**), chuẩn hóa thực thể tiếng Việt nói (**Spoken Vietnamese Normalization**), nguyên tắc không đoán mò tham số (**Zero Default Guessing qua `sys.ask_missing_info`**), và kiểm soát chống gọi lặp (**Idempotent Tool Execution**) dưới ngân sách `< 30 giây/mẫu` trên GPU NVIDIA A30.

---

## 1. Sơ đồ Kiến trúc Toàn diện (End-to-End LifeMate Voice-to-Action Flow)

```text
          Audio hiện tại (.wav) + Chuỗi Audio Lịch sử (history/*.wav)
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │        SpeechRecognizer (models.py)         │
               │ • Model: vinai/PhoWhisper-medium (769M)     │
               │ • Resample 16kHz + Mono Normalization       │
               │ • Batched Inference (batch_size = 4)        │
               │ • Greedy Decoding (do_sample = False)       │
               └──────────────────────┬──────────────────────┘
                                      │ transcript hiện tại + transcripts history
                                      ▼
               ┌─────────────────────────────────────────────┐
               │    BipartiteToolGraph + Prompt Builder      │
               │ • Nén JSON Schema (_compact_schema)         │
               │ • Xếp hạng wRRF (α_A = 1.5, α_T = 1.0)      │
               │ • Duyệt ngược Tool -> App (read trước write)│
               │ • Lịch sử chỉ tham chiếu, cấm Replay        │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │   ToolAgent Policy: Qwen3-4B-Instruct-2507  │
               │ • Sinh JSON: {"action": "call" | "stop"}    │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │    Tool-Boundary Middleware (middleware.py) │
               │ • Chuẩn hóa giờ/ngày nói ("6h30" -> "06:30")│
               │ • Ép kiểu & Ánh xạ tên tham số theo Schema  │
               │ • Phân giải Tên tự nhiên -> ID (x-ref)      │
               │ • Zero Default Guessing & Chặn gọi trùng    │
               └──────────────────────┬──────────────────────┘
                         ┌────────────┴────────────┐
                         ▼                         ▼
             [Đủ thông tin & Hợp lệ]     [Thiếu thông tin bắt buộc]
                         │                         │
                         ▼                         ▼
                env.validate() -> ok     Gọi sys.ask_missing_info
                env.call(name, args)     (Không tự bịa ID / đối số)
                         └────────────┬────────────┘
                                      │
                                      ▼
               ┌─────────────────────────────────────────────┐
               │       Submission Output (runner.py)         │
               │ • Đúng 1 phiên ToolEnv / mẫu                │
               │ • env.get_tool_calls() + env.get_state_hash │
               │ • Xuất 1 dòng JSONL vào submission.jsonl    │
               └─────────────────────────────────────────────┘
```

---

## 2. Nhận dạng Giọng nói Đa lượt & Quản lý Ngân sách 30 Giây (`src/lifemate/models.py` & `runner.py`)

### 2.1. Bài toán Audio Lịch sử Cộng dồn (Cumulative Audio History)
Theo kiểm kê dữ liệu trong [`docs/analysis/dataset_audit.md`](docs/analysis/dataset_audit.md):
- **66,1% mẫu Public Test** có lịch sử hội thoại (`history`), với số lượt tối đa lên tới **24 lượt**.
- Mỗi lượt `role == "user"` trong `history` đều là một tệp `.wav` cần được nhận dạng ở tập Test (vì tập Test không có sẵn `audio_text`).
- Thời lượng audio yêu cầu cuối ở mức p95 là `13,46 giây`, nhưng **tổng thời lượng audio hiện tại + history ở mức p95 lên tới `38,31 giây` (cao nhất `82,83 giây`)**.

### 2.2. Giải pháp Batched ASR & Ưu tiên Lượt Gần Nhất (`_audio_inputs` & `transcribe_many`)
Triển khai tại [`src/lifemate/runner.py`](src/lifemate/runner.py) và [`src/lifemate/models.py`](src/lifemate/models.py):

| Cơ chế | Chi tiết cài đặt | Mục đích thực tế trên GPU A30 |
| :--- | :--- | :--- |
| **Thứ tự ưu tiên Audio** | `refs = [current] + list(reversed(history))` | Nhận dạng yêu cầu hiện tại trước, sau đó đến các lượt lịch sử gần nhất. |
| **Gom lô động (Batching)** | `batch_size = 4`, `padding = "longest"` | Xử lý song song 4 tệp `.wav` mỗi lượt trên GPU A30 (FP16), giảm thời gian chạy ASR của mẫu 38 giây audio xuống chỉ còn vài giây thực thi. |
| **Chuẩn hóa Tín hiệu** | `sf.read(..., always_2d=True)` $\rightarrow$ `mean(axis=1)` $\rightarrow$ `resample_poly` về `16000 Hz` | Đảm bảo mọi tệp `.wav` (dù mono hay stereo, sample rate khác nhau) đều về đúng chuẩn đầu vào của `PhoWhisper-medium`. |
| **Giải mã Xác định** | `do_sample=False`, `num_beams=1`, `language="vi"` | Tốc độ giải mã nhanh nhất và tái lập kết quả 100% giữa các lần chạy Docker. |

---

## 3. Kiểm soát Ngữ cảnh Lịch sử & Chống Thực thi Lại (History Reference vs. Replay Prevention)

Trong hội thoại giọng nói nhiều lượt, người dùng thường dùng đại từ chỉ định hoặc nối tiếp câu trước (ví dụ: lượt trước hỏi *"Sáng mai tôi có lịch họp nào không?"*, lượt hiện tại nói *"Dời lịch đó sang 9 giờ rưỡi giúp tôi"*).

### 3.1. Phân tách rõ Lịch sử (`history`) và Lượt hiện tại (`request`)
Triển khai tại [`src/lifemate/prompts.py`](src/lifemate/prompts.py):
- **Không bao giờ Replay hành động cũ:** Các `tool_calls` trong `history` đã được môi trường `ToolEnv` của BTC áp dụng sẵn vào trạng thái khởi tạo. Hệ thống chỉ đưa `history` vào trường `"history"` để giải quyết đồng tham chiếu (*"lịch đó"*, *"đơn hàng vừa nãy"*, *"anh ấy"*), tuyệt đối chỉ sinh lệnh gọi tool mới cho `"request"` cuối cùng.
- **Bảo전 Thời gian Ngữ cảnh (`context.current_time`):** Mọi biểu thức thời gian tương đối trong lời nói (*"sáng mai"*, *"thứ Sáu tuần này"*, *"30 phút nữa"*) đều được tính từ `context.current_time` và `context.timezone` của mẫu, tuyệt đối không lấy giờ hệ thống của máy chủ chấm thi.

---

## 4. Chuẩn hóa Ngôn ngữ Nói Tiếng Việt & Nguyên tắc "Zero Default Guessing" (`src/lifemate/middleware.py`)

Lời nói tự nhiên sau khi qua ASR thường chứa các cách diễn đạt giờ giấc/số lượng đời thường hoặc bị khuyết thông tin bắt buộc.

### 4.1. Chuẩn hóa Giờ giấc & Ký hiệu Nói Tiếng Việt (`_normalize_vietnamese_datetime_str`)
Triển khai tại [`src/lifemate/middleware.py`](src/lifemate/middleware.py):
- Khi LLM trích xuất chuỗi thời gian từ câu nói tiếng Việt vào đối số công cụ, Middleware tự động chuẩn hóa về định dạng chuẩn ISO / `HH:MM`:
  - `"6:30"` $\rightarrow$ `"06:30"` (bổ sung số 0 ở đầu).
  - `"6h30"`, `"6 giờ 30"`, `"6 giờ 30 phút"` $\rightarrow$ `"06:30"`.
  - `"6h rưỡi"`, `"6 giờ rưỡi"` $\rightarrow$ `"06:30"`.
  - `"2026/10/25"` $\rightarrow$ `"2026-10-25"`.
- Chuẩn hóa toàn bộ chuỗi tiếng Việt về dạng **Unicode NFC** để đạt điểm khớp `token_f1 > 0.6` tối đa trên các đối số đánh dấu `x-match: "fuzzy"`.

### 4.2. Nguyên tắc Tuyệt đối: Zero Default Guessing (`sys.ask_missing_info` & `x-ref` Guard)
- **Không bao giờ tự đoán ID hoặc tham số bắt buộc:**
  - Nếu câu nói của người dùng thiếu thông tin bắt buộc (ví dụ: *"Chuyển tiền cho Minh"* nhưng chưa nói số tiền, hoặc trong danh bạ chưa tra cứu `contact_id`), Agent **cấm tự điền giá trị mặc định** hay tự bịa mã ID (`"contact_1"`).
- **Quy trình 2 bước bắt buộc:**
  1. Nếu thông tin có thể tra cứu trong ứng dụng (như tìm `contact_id`, `alarm_id`, `event_id`, `order_id`), bắt buộc gọi công cụ `kind="read"` trước; sau đó [`extract_inter_tool_bindings()`](src/lifemate/middleware.py) sẽ rút trích ID thật vào `"resolved_refs"`.
  2. Nếu thông tin bắt buộc do người dùng quên cung cấp và không thể tự tra cứu (hoặc công cụ `read` trả về danh sách rỗng `[]`), bắt buộc gọi công cụ hệ thống **`sys.ask_missing_info`**.

---

## 5. Phân định: Chống Gọi Trùng (Idempotency) vs. Kiểm soát Ngân sách Tái lập Kế hoạch ($B = 3$)

| Tiêu chí | Chống Gọi Trùng (Duplicate Call Blocking) | Kiểm soát Ngân sách Tái lập Kế hoạch ($B = 3$) |
| :--- | :--- | :--- |
| **Nơi triển khai** | [`check_local_invariants()`](src/lifemate/middleware.py) trong `src/lifemate/middleware.py` | [`_run_actions()`](src/lifemate/runner.py) trong `src/lifemate/runner.py` |
| **Cơ chế** | Đối chiếu `(action["name"], action["arguments"])` với danh sách `prior_calls` đã thực thi trên `ToolEnv` trong cùng lượt yêu cầu. | Đếm tổng số lần bị chặn bởi Middleware/Schema hoặc bị đánh trượt bởi Verifier 4 tiêu chí (`replan_state["used"] <= 3`). |
| **Mục đích** | Ngăn LLM gọi lặp lại y hệt một lệnh `read` hoặc `write` đã chạy thành công ở bước trước (tránh làm hỏng `state_hash` và giảm Precision). | Ngăn Agent sa vào vòng lặp sửa sai vô hạn khi gặp mẫu quá khó hoặc lỗi nghiệp vụ liên tiếp, bảo vệ giới hạn `< 30 giây/mẫu`. |
| **Hành động khi kích hoạt** | Chặn trước khi chạm `env.call()` và nhắc Agent dùng kết quả đã có trong `steps_this_request` hoặc trả về `{"action": "stop"}`. | Khi `used > 3` ($b > B$), dừng vòng lặp thực thi ngay lập tức và xuất kết quả an toàn. |
