# Thư mục `promt/` — Bộ Prompt & Tín hiệu Ngữ nghĩa Hội thoại cho LifeMate

Thư mục này ban đầu chứa các prompt từ bài toán **Meeting Intelligence & Neo4j Knowledge Graph**. Sau khi rà soát theo đúng đặc thù của **Zalo AI Challenge 2026 — LifeMate (Vietnamese Voice Multi-Turn Tool-Calling Agent)**, thư mục đã được **dọn dẹp và viết lại hoàn toàn** để phục vụ trực tiếp cho pipeline của dự án.

---

## 1. Các file đã XÓA (Không phù hợp với LifeMate)

| File đã xóa | Lý do xóa |
|---|---|
| `meeting_extraction_system.txt` | Dành cho bóc tách biên bản họp doanh nghiệp (`SUMMARY`, `DECISION`, `RISK` tiếng Anh), không khớp với 13 ứng dụng cá nhân của LifeMate. |
| `text_to_cypher.txt` | Sinh truy vấn cơ sở dữ liệu đồ thị Neo4j (`MATCH (n)...`). LifeMate gọi tool qua `zaic_tools` JSON Schema, không dùng Neo4j. |
| `graph_grounded_qa.txt` | Prompt trả lời câu hỏi tự luận kèm trích dẫn `[Source: Meeting_ID]`, trái ngược với yêu cầu xuất `<tool_call>` hoặc `final` ngắn gọn của LifeMate. |
| `hub_node_profiling.txt` | Viết tiểu sử nhân sự trong công ty bằng tiếng Anh, hoàn toàn không liên quan. |
| `multihop_graph_qa.txt` | Suy luận tự luận 5 đoạn văn cho Neo4j QA, gây tốn token và vi phạm cấu trúc output của `Qwen3-4B`. |

---

## 2. Các file ĐƯỢC GIỮ LẠI & CHỈNH SỬA CHO LIFEMATE

Trong tập dữ liệu của BTC, **66% mẫu có lịch sử hội thoại nhiều lượt (`history` dài tới 24 turns)** và đầu vào là **giọng nói tiếng Việt qua ASR (`PhoWhisper`)**. Do đó, 3 kỹ thuật quan trọng nhất từ bộ prompt gốc (gồm **DUR — Incomplete Utterance Restoration**, **Temporal Intent Shift — `SUPERSEDES`**, và **Entity Resolution & Ambiguity Guard**) đã được chuyển hóa thành 3 file chuẩn cho LifeMate:

### 2.1. [`lifemate_system.txt`](lifemate_system.txt)
- **Mục đích**: System prompt tham chiếu cho Agent LifeMate (`Qwen3-4B`), kết hợp quy tắc của 3 bài báo (**Agent-as-a-Graph**, **HEART**, **PRISM**) với xử lý hội thoại đa lượt:
  - **DUR (Khôi phục đại từ & câu tỉnh lược)**: Giải quyết các cụm từ chỉ định trong giọng nói tiếng Việt (*"đơn đó"*, *"lịch vừa nãy"*, *"người đó"*, *"dời sang 7 giờ"*) bằng cách đối chiếu ngược `history` và `steps_this_request`.
  - **Temporal Intent Shift (`SUPERSEDES`)**: Khi người dùng đổi ý giữa chừng (*"À thôi đổi lại..."*, *"Khoan đã, hủy giúp tôi..."*), ưu tiên tuyệt đối ý định ở lượt cuối cùng và gọi đúng tool cập nhật/hủy với `x-ref` ID thực.
  - **Null-Safety & `sys.ask_missing_info`**: Cấm bịa tham số hoặc ID khi thiếu thông tin bắt buộc.

### 2.2. [`dur_restoration.txt`](dur_restoration.txt)
- **Mục đích**: Prompt chuyên dụng cho bước **Offline Data Enrichment / SFT Augmentation** hoặc phân tích lỗi hội thoại đa lượt:
  - Khôi phục câu nói tỉnh lược (`rewritten_utterance`) từ `history` + `raw_transcript`.
  - Sửa lỗi đồng âm ASR tiếng Việt theo ngữ cảnh `installed_apps`.
  - Trích xuất các cạnh đồ thị hội thoại (`anaphora_bindings` $\rightarrow$ `COREFERS_TO` và `supersedes_intent` $\rightarrow$ `SUPERSEDES`).

### 2.3. [`entity_resolution.txt`](entity_resolution.txt)
- **Mục đích**: Quy tắc phân giải tên thực thể tự nhiên trong câu nói sang khóa ngoại `x-ref` ID (`contact_id`, `order_id`, `event_id`, `alarm_id`, `item_id`, ...):
  - **Exact/Unique Match**: Ánh xạ tên thực thể sang ID duy nhất tìm được từ `read` tool.
  - **Ambiguity Guard (Chống nhập nhằng thực thể)**: Nếu tìm thấy **$\ge 2$ thực thể trùng tên** (ví dụ: 2 liên hệ cùng tên *"Tuấn"* hoặc 2 đơn hàng cùng trạng thái), bắt buộc dừng lệnh `write` và gọi `sys.ask_missing_info` để hỏi lại người dùng.
  - **Cross-Entity Integrity**: Kiểm tra tính nhất quán cha-con (`item_id` phải thuộc đúng `order_id`).

### 2.4. [`prompts.py`](prompts.py)
- Cung cấp class `PromptManager` để load 3 file prompt trên.
- Cung cấp hàm deterministic `extract_dialogue_graph_signals(transcript, history)` giúp phát hiện tức thì (~0ms, không tốn thêm LLM call) các tín hiệu `has_anaphora`, `has_supersedes`, `prior_tool_apps`, và `history_entity_ids` để hỗ trợ tầng Graph Retrieval (`src/lifemate/graph_retrieval.py`) và Middleware (`src/lifemate/middleware.py`).
