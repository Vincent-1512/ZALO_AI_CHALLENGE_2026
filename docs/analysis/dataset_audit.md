# Kiểm kê dữ liệu LifeMate

Ngày kiểm kê: **04/11/2026**. Nguồn là kit **v3.2.0** đã tải về trong workspace. Chỉ đọc JSONL, catalog, metadata WAV và mã `zaic_tools`; không mở, đọc, sao chép hay phân tích bất cứ file nào dưới `*_environment/`.

## Phân chia dữ liệu

| Thuộc tính | Train | Public test |
|---|---:|---:|
| Số mẫu | 1.545 | 634 |
| Audio của yêu cầu hiện tại | 1.545 | 634 |
| Tổng file WAV gồm audio history | 5.036 | 1.933 |
| Có history | 1.018 (65,9%) | 419 (66,1%) |
| Số lượt history lớn nhất | 24 | 24 |
| App cài trung bình / lớn nhất | 9,20 / 18 | 8,53 / 17 |
| Tool có thể dùng trung bình / lớn nhất | 32,53 / 62 | 30,66 / 59 |
| Độ dài audio hiện tại p50 / p95 / max | 6,94 / 11,82 / 22,33 giây | 7,20 / 13,46 / 20,24 giây |
| Tổng audio hiện tại + history p50 / p95 / max | 13,00 / 40,79 / 69,41 giây | 12,71 / 38,31 / 82,83 giây |

Mỗi history có các lượt user và assistant xen kẽ. User turn có audio; assistant turn có nội dung, tool calls và tool results. Vì vậy public inference phải nhận dạng cả audio history, không chỉ audio hiện tại. Mẫu public p95 chứa khoảng 38 giây audio cộng dồn, còn mẫu dài nhất khoảng 83 giây; cần đo batching ASR trên A30 để đáp ứng hạn 30 giây/mẫu.

## Nhãn train và độ khó

- `audio_text` có sẵn cho yêu cầu hiện tại trong train; các user turn history của train cũng có transcript. Public test không có transcript.
- Gold `tool_calls` phân bố: 0 call: 111; 1: 369; 2: 428; 3: 359; 4: 161; 5: 107; 6: 10. Tổng cộng 3.541 call và 5.086 nhãn bước kế tiếp nếu tính thêm một tín hiệu dừng cho mỗi mẫu.
- 111 mẫu train có danh sách tool call rỗng. 217 mẫu có `sys.ask_missing_info`.
- `label.tool_results` có cùng số phần tử với `label.tool_calls` ở cả 1.545 mẫu. Có thể tạo trajectory: prompt tại bước i gồm yêu cầu + history + các call/result trước đó; target là call thứ i. Sau call cuối, target là dừng.
- `label.final_state` là nhãn kiểm tra train, không được đưa vào prompt hay target sinh. Dữ liệu đầu vào huấn luyện dùng `audio_text`, `context`, `history`, catalog, `tool_calls` và `tool_results`.

## Chênh lệch catalog cần tính đến

| Thuộc tính | Train catalog | Public catalog |
|---|---:|---:|
| App | 46 | 36 |
| Tool | 147 | 122 |
| App giao nhau | 25 | 25 |
| Tool giao nhau | 88 | 88 |
| App chỉ có ở public | — | 11 |
| Tool chỉ có ở public | — | 34 |

Các app mới của public là `ecommerce`, `event_tickets`, `grocery`, `habit_tracker`, `pantry`, `parking`, `public_transit`, `recipes`, `stocks`, `train_bus_tickets`, `translator`. Một số app dùng chung cũng có tool mới; do đó số tool giao nhau chỉ là 88. Agent cần đọc schema động từ catalog và danh sách tool cài trong từng mẫu. Không ánh xạ cứng theo ID, app hay tên tool train.

## Hệ quả cho baseline

1. ASR nhận dạng theo nhóm audio của một mẫu, theo batch nhỏ; lưu ý audio history làm p95 cộng dồn cao hơn đáng kể audio hiện tại.
2. Qwen nhận transcript, thời điểm/múi giờ/profile, history, schema các tool được cài và các observation sinh ra trong lượt hiện tại.
3. Sinh mỗi lần một hành động có cấu trúc: `call` hoặc `stop`. OOS học từ các mẫu dừng; hỏi thiếu thông tin vẫn là `sys.ask_missing_info`.
4. `env.validate` chặn call sai schema trước khi thực thi. Call hợp lệ mới được chuyển cho `env.call`; tool result thật trở lại prompt ở bước kế tiếp.
5. Một mẫu tạo đúng một `ToolEnv`. Inference chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`; không gọi `get_state` hay đọc file môi trường.
6. Đánh giá tốc độ phải dùng workload đủ history trên A30, không ngoại suy từ audio đơn. Tổng context tối đa khóa ở 16.384 token; cấu hình khởi điểm 8.192 để giảm latency, rồi điều chỉnh theo rollout.

## Phạm vi và giới hạn kiểm kê

Đây là thống kê cấu trúc, độ dài audio và nhãn call; chưa phải kết quả ASR, điểm baseline hay phép đo chính thức. Bộ `zaic_tools` có validator schema, nhưng chưa xác nhận được scorer đầy đủ cho công thức Evaluation. Không dùng public test để gán nhãn, tune prompt hoặc tạo dữ liệu train.
