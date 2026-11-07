# Sửa baseline Colab sau lỗi context — 05/11/2026

Phạm vi đã được yêu cầu: rà soát, sửa baseline/notebook và kiểm tra để upload lại.

1. Tái hiện lỗi fit prompt/target; sửa việc dành ngân sách cho cả đáp án.
2. Preflight toàn bộ train/dev bằng tokenizer pin; chọn context theo dữ liệu, không theo VRAM, không bỏ record hoặc cắt schema/đáp án.
3. Cấu hình QLoRA single-GPU rõ ràng, checkpoint/log/khả năng resume; giữ model, split và recipe ban đầu.
4. Notebook kiểm tra dependency/GPU trước train; log subprocess đầy đủ vào Drive; chỉ full train sau smoke và GPU check.
5. Test hồi quy, kiểm tra tokenizer trên dữ liệu thật, validate notebook; đóng gói allowlist với checksum.

Giới hạn: máy local không có NVIDIA CUDA. Lỗi exit 1 trong Colab chưa có traceback gốc; không suy đoán nguyên nhân hoặc tuyên bố đã xác minh T4.

## Kết quả

- Đã sửa fit prompt + target và kiểm tra cả train/dev không bỏ record.
- Notebook v2 có kiểm tra CUDA NF4/optimizer, log Drive, context theo dữ liệu, checkpoint định kỳ và resume bỏ qua checkpoint thiếu file.
- Loss đáp án đã được so loss/gradient với Qwen gốc và kiểm tra PEFT train/eval/save/resume trên CPU. Tổng 9 test đạt.
- Bundle v2 đã kiểm checksum, schema/cú pháp notebook và tính tái lập khi build; tạo lại SFT và preflight từ bundle giải nén sạch.
- Chưa chạy được model 4B/QLoRA trên T4 vì local không có CUDA; lỗi Colab exit 1 trước đây chưa có traceback gốc để kết luận nguyên nhân.

## OOM attention thực tế trên T4

- Log mới xác nhận allocation5.20GiB trong SDPA tại6605token; dấu vết khớp math backend ma trận32heads float32.
- Đã thêm context T4 scoped repeat-KV + efficient-only, probe forward/backward trước nạp4B, giữ nguyên dữ liệu/context/recipe.
- 12 kiểm tra CPU đạt. Chưa có profiler hoặc chạy CUDA local; cần xác minh GPU bằng probe và smoke ở Colab.
