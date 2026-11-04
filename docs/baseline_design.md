# Baseline LifeMate: ASR → schema-aware agent → ToolEnv

## Mục tiêu và giới hạn

Baseline đầu tiên dùng PhoWhisper-medium để phiên âm tiếng Việt và Qwen3-4B-Instruct-2507 để quyết định hành động. Tổng tham số ước tính **4,79B**, dưới giới hạn 6B; LLM 4,02B, dưới 5B. Qwen chạy trong tối đa 8.192 token ở cấu hình đầu, thấp hơn giới hạn cuộc thi 16.384. Mô hình và revision được pin trong [`configs/baseline.json`](../configs/baseline.json).

Model cards: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache-2.0; 4,02B tham số; phát hành năm 2025) và [PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium) (BSD-3-Clause; model tiếng Việt từ 2024). Trước khi đóng gói, đếm lại tham số từ artifacts đã tải và lưu checksum/revision.

Đây là baseline thiết kế, chưa có số điểm hay số đo A30. PhoWhisper-medium là ứng viên chất lượng; vì tổng audio p95 của public khoảng 38 giây/mẫu, nếu không đạt 30 giây end-to-end thì so với PhoWhisper-small hoặc backend ASR nhanh hơn trên cùng dev. Chỉ giữ thay đổi khi điểm state/tool và latency đều được đo.

## Dòng xử lý một mẫu

```text
JSONL + audio hiện tại + history audio
                  │
                  ▼
       PhoWhisper, batch tối đa 4 audio
                  │ transcript hiện tại + transcript history
                  ▼
 Qwen: yêu cầu + context + history + tool schemas đã cài
                  │ một JSON: stop | call(name, arguments)
                  ▼
            env.validate()
             │ hợp lệ
             ▼
 env.call() ── kết quả thật trở lại Qwen ── lặp tối đa 8 bước
                  │
                  ▼
 env.get_tool_calls() + env.get_state_hash()
                  │
                  ▼
            submission.jsonl
```

Một `ToolEnv` được tạo đúng một lần cho mỗi mẫu. History chỉ giúp hiểu tham chiếu; các hành động history không được gọi lại. Mỗi call mới phải hợp lệ theo schema trước khi thực thi. Lỗi schema được gửi lại cho agent một lần để sửa; nếu vẫn sai, agent dừng và không ghi call lỗi vào danh sách nộp. Kết quả nghiệp vụ như ID không tồn tại vẫn là observation bình thường do tool trả về.

Tool schema lấy từ catalog của split và lọc theo `context.installed_apps` của mẫu, có thêm `sys.ask_missing_info`. Cách này cho phép agent nhìn thấy 34 tool public không có trong train catalog. SFT học hành vi chọn tool và điền đối số; tên/schema cụ thể vẫn được cấp động lúc inference.

## Dữ liệu SFT

[`training/prepare_sft.py`](../training/prepare_sft.py) tạo một ví dụ cho mỗi bước gold và một ví dụ dừng cuối trajectory. Bước i chỉ thấy request, context, history và gold call/result đến bước i−1; target là call i. Nhãn dừng được thêm cả cho mẫu OOS và sau chuỗi hành động. `final_state` không được đưa vào input hay target.

Split cố định 80/10/10 bằng seed 2026; các mẫu có cùng transcript yêu cầu đã chuẩn hóa được xếp cùng một split. `holdout.jsonl` không dùng train. Manifest ghi số mẫu, số bước và SHA-256 danh sách ID. Trong vòng đầu chỉ train trên train split, theo dõi eval loss trên dev và chọn checkpoint bằng rollout/state-tool score riêng; không dùng loss làm điểm thi.

[`training/train_lora.py`](../training/train_lora.py) chạy QLoRA NF4, LoRA rank 16, batch 1, gradient accumulation 16, tối đa 2 epoch. Đây là cấu hình khởi điểm. Trên Vax.ai cần chạy kiểm tra 50–100 mẫu trước, xem VRAM/loss/gradient và lưu checkpoint; sau đó mới chạy đủ. Không fine-tune ASR lúc đầu: trước tiên đo riêng transcript chuẩn và transcript ASR để xác định ASR có phải nút thắt.

## Chạy trên Vax.ai

Từ gốc repo, sau khi chuyển repo và bộ dữ liệu hợp lệ sang máy GPU:

```bash
python -m pip install -r requirements-train.txt
python scripts/download_models.py
python training/prepare_sft.py --data-root data/raw/train+public_test
python training/evaluate_dev.py --report reports/runs/dev_base.json
python training/train_lora.py --train-jsonl data/processed/sft/train.jsonl --dev-jsonl data/processed/sft/dev.jsonl --limit-train-samples 64 --limit-dev-samples 16 --epochs 1 --output-dir models/lifemate-qwen3-4b-smoke
python training/evaluate_dev.py --adapter models/lifemate-qwen3-4b-smoke/final --report reports/runs/dev_smoke.json
# Chỉ chạy đủ 2 epoch sau khi smoke hoàn tất và model nạp/lưu/evaluate được.
python training/train_lora.py --train-jsonl data/processed/sft/train.jsonl --dev-jsonl data/processed/sft/dev.jsonl
python training/evaluate_dev.py --adapter models/lifemate-qwen3-4b-lora/final --report reports/runs/dev_lora.json
```

Chạy inference public với checkpoint base trước để có baseline pretrained, rồi chạy lại với adapter để so sánh:

```bash
python scripts/run_inference.py --data data/raw/train+public_test --split public_test --output result/base.jsonl
python scripts/run_inference.py --data data/raw/train+public_test --split public_test --output result/lora.jsonl --adapter models/lifemate-qwen3-4b-lora/final
```

Trước khi nộp, tải đúng weights/dependencies vào Docker image và chạy offline. Lệnh tải model chỉ dùng ở giai đoạn chuẩn bị; inference tắt tải từ Hub bằng `local_files_only=True`.

## Tiêu chí chốt baseline

1. Đủ tất cả ID, JSONL hợp schema, mỗi mẫu có hash/call list do đúng phiên `ToolEnv` tạo.
2. Đo state match, tool precision/recall/F1 theo evaluator BTC khi có đủ công thức/scorer. Validator schema chỉ kiểm cấu trúc, không phải điểm thi.
3. So base với LoRA trên cùng grouped dev; bổ sung app-held-out để đo khả năng dùng schema/tool mới.
4. Chạy toàn bộ workload trên A30; không mẫu nào quá 30 giây, không OOM. Ghi p50/p95/p99/max, VRAM và thời gian ASR/LLM/tool riêng.
5. Nếu LoRA không tăng state/tool score hoặc làm app-held-out xấu rõ, giữ base. Nếu ASR medium chậm, so small trên cùng mẫu và cùng agent.

`training/evaluate_dev.py` chạy rollout trên ID của dev và dùng `get_state()` chỉ trong tập train để so các app đã gắn nhãn thay đổi. Call P/R/F1 là chẩn đoán cục bộ; do thiếu toàn bộ công thức Evaluation ở dạng văn bản, không gọi đây là điểm BTC.
