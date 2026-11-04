# Yêu cầu LifeMate

Tóm tắt Data, Evaluation, Rule và Submission Guidance do đội cung cấp ngày 04/11/2026. [Portal BTC](https://challenge.zalo.ai/portal/lifemate) và [thể lệ tiếng Việt](regulations/Zalo_AI_Challenge_2026_The_le_VI.pdf) là nguồn gốc; theo dõi thay đổi trước mỗi lần phát hành.

## Dữ liệu và đầu ra

Input gồm audio yêu cầu cuối, context, history và schema app/tool. Train có `audio_text`, gold tool calls, tool results và final state. Public/private không có nhãn để agent sử dụng. Các hành động trong history đã được áp dụng sẵn: không replay chúng.

Agent dùng schema động của split và chỉ gọi tool thuộc app được cài, cộng `sys.ask_missing_info`. Mỗi mẫu có đúng một ToolEnv; output lấy từ `get_tool_calls()` và `get_state_hash()`. JSONL UTF-8 phải có đúng một dòng/mẫu với `id`, `tool_calls`, `state_hash`, bao gồm cả call thất bại.

## Luật inference

- Toàn hệ thống dưới 6B tham số; riêng LLM dưới 5B; context tối đa 16.384 token.
- Chạy offline, không API/dịch vụ ngoài; 30 giây/mẫu trên 1 A30, CPU Xeon Gold 6442Y 16 core, RAM 64 GB. Timeout/lỗi thực thi được 0.
- Không truy cập trực tiếp `<split>_environment/`; chỉ dùng `validate`, `call`, `get_tool_calls`, `get_state_hash`. `get_state` chỉ dành cho train, không dùng lúc inference.
- Cố định seed, giải mã xác định. Nếu tái lập khác, BTC chạy Docker 5 lần và lấy trung bình.
- Public data/model nguồn mở phải phát hành trước 26/10/2026, giấy phép phù hợp. Có thể tạo synthetic data trong phát triển theo quyền sử dụng; không dùng dịch vụ tạo đó lúc inference, không gán nhãn public/private test để train hoặc tune.

## Evaluation

Sai final state hash thì mẫu 0. Hash do môi trường sinh; BTC replay call theo thứ tự bằng đúng phiên bản kit và kết quả BTC quyết định nếu có khác biệt.

Call match yêu cầu tên tool và mọi đối số không fuzzy giống nhau. Đối số fuzzy dùng token-F1 >0.6 sau lowercase và Unicode NFC. Mỗi call ghép tối đa một lần; thứ tự không dùng để ghép call, nhưng vẫn quyết định kết quả thực thi. Call dư hoặc bị schema từ chối ảnh hưởng precision. `sys.ask_missing_info` được chấm như tool khác. Gold calls rỗng chỉ được điểm khi dự đoán cũng rỗng.

Hình công thức tổng thể chưa có trong văn bản đội gửi. Không tự suy diễn trọng số hoặc gọi CLI kiểm tra schema là scorer đầy đủ. Cần bổ sung công thức/scorer trước khi xác nhận điểm local tương đương BTC.

Hoà điểm: BTC tính F1 call bỏ kiểm tra hash; vẫn hoà thì latency trung bình thấp hơn thắng.

## Submission

Public round 26/10–16/11: chạy public test, nộp JSONL qua Submission; 5 lần/ngày, reset cuối ngày. Leaderboard public chỉ tham khảo.

Ngày 16/11, hạn 23:59 GMT+7: nộp giải pháp Docker gồm training code, inference code và data được yêu cầu qua Solution Submission, kèm checksum và link Drive công khai. Tối đa 2 lần; BTC chỉ xét lần cuối.

Docker phải chạy offline. BTC mount private test tại `/data`, chạy inference, lấy `result/submission.jsonl`. Entrypoint, working directory và cách đóng gói cụ thể theo tài liệu Docker riêng của BTC.

Ngày 20/11 lúc 09:00 theo Submission Guidance: phát hành private test. Chạy đúng giải pháp đóng băng, nộp JSONL trước 23/11 lúc 23:59. Private leaderboard hiển thị theo giờ nộp; BTC chạy lại Docker, kiểm tra code và xác minh điểm. Top 2 công bố 02/12.

Trang chủ từng hiển thị 11:00 cho mốc private, khác với Guidance 09:00; đội chuẩn bị theo mốc sớm hơn và theo dõi thông báo mới nhất.
