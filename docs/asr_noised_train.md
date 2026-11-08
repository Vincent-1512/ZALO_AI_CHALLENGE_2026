# Chuẩn bị SFT với transcript ASR

`build_asr_train_data.py` tạo file train riêng có đầu vào gần inference hơn: với xác suất `p` (mặc định `0.5`), một sample dùng transcript PhoWhisper thay cho `audio_text` chuẩn; target tool call, kết quả lịch sử, nhãn và cách chia sample vẫn giữ như `prepare_sft.py`. Chọn ASR/gold theo `seed` và `sample_id`, nên chạy lại cùng tham số sẽ chọn nhất quán. Script chỉ chuẩn bị JSONL, không train.

Trên Vast, sau khi data, PhoWhisper và môi trường offline đã sẵn sàng:

```bash
python scripts/build_asr_train_data.py \
  --data-root data/raw/train+public_test \
  --output-dir data/processed/asr_noisy_sft \
  --asr-model models/PhoWhisper-medium \
  --asr-cache lifemate_runs/asr_cache.json \
  --asr-probability 0.5 --seed 2026
```

Sau đó trỏ `--train-jsonl` của lần train thử sang `data/processed/asr_noisy_sft/train.jsonl`; giữ dev gốc để so sánh công bằng:

```bash
python training/train_lora.py \
  --train-jsonl data/processed/asr_noisy_sft/train.jsonl \
  --dev-jsonl data/processed/sft/dev.jsonl \
  --model-path models/Qwen3-4B-Instruct-2507 \
  --output-dir lifemate_runs/asr-noisy-experiment/full
```

Không resume optimizer từ checkpoint cũ khi dữ liệu train đã đổi. Đây là một run mới, không phải resume tương đương.

## Schema-order augmentation

Mặc định script giữ augmentation thứ tự tool tương tự notebook: mỗi trajectory record được ghi nguyên bản và thêm tối đa một bản có `available_tools` đảo thứ tự, cùng target. Chỉ thêm bản sao nếu sample có ít nhất hai tool khả dụng. Vì vậy số record chỉ gấp đôi khi mọi record đủ điều kiện; không mặc định khẳng định số cụ thể như 4062 → 8124. Dùng `--no-schema-permutation` để tắt và cô lập tác động ASR noise.

Dev/holdout không được tạo lại hoặc đổi transcript bởi script; tiếp tục dùng dev hiện có. ASR cache chỉ hợp lệ với fingerprint model và byte nội dung audio tương ứng.

## Đo sửa transcript theo ngữ cảnh

Các cờ correction mặc định tắt. Chạy rollout dev có log thay thế:

```bash
python training/evaluate_dev.py --asr-correct \
  --asr-cache lifemate_runs/asr_cache.json \
  --per-sample-log lifemate_runs/dev_corrected_samples.jsonl \
  --report lifemate_runs/dev_corrected.json
```

Đo WER/CER trước/sau mà không nạp ASR model, từ cache khớp chính xác cùng dữ liệu, thứ tự audio và fingerprint:

```bash
python scripts/eval_asr_correct_offline.py \
  --data-root data/raw/train+public_test \
  --sft-dev data/processed/sft/dev.jsonl \
  --asr-cache lifemate_runs/asr_cache.json \
  --asr-model models/PhoWhisper-medium
```

Từ vựng lấy từ `context.user_profile`, tên app/tool đã cài, trường catalog của tool, tên/description tham số và enum. Không đọc state ẩn của môi trường. Ngưỡng mặc định 0.88 là heuristic cần đánh giá; bật correction chỉ sau khi xem các mẫu bị WER tệ hơn và log thay thế.
