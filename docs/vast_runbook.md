# Chạy baseline LifeMate trên Vast.ai

Runbook này dành cho **train/dev và public test**, chưa phải Docker package cuối gửi BTC. Bundle Vast gồm source và ZIP dataset BTC trong cùng một archive; ZIP được giữ nguyên, không đưa dataset lên GitHub.

## Chuẩn bị instance

- Chọn Linux x86_64 có NVIDIA CUDA, **tối thiểu 24 GB VRAM** và khoảng **50 GB ổ đĩa trống** để chứa bộ dữ liệu, weights, cache và checkpoint.
- Dùng image có CUDA-compatible PyTorch sẵn; `requirements-train.txt` không cài thay PyTorch CUDA.
- Xác nhận trước khi cài:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
```

## Upload và giải nén

Tải lên một file `lifemate-vast-bundle-<commit>.tar.gz`. Tên file có mã commit ngắn để biết chính xác source đang chạy. Giải nén sẽ tạo thư mục `lifemate/` gồm code và `data/archives/train+public_test.zip`. Trên instance:

```bash
tar -xzf lifemate-vast-bundle-<commit>.tar.gz
cd lifemate
sha256sum -c SHA256SUMS
mkdir -p data/raw
unzip -n -q data/archives/train+public_test.zip 'train+public_test/*' -d data/raw
```

Checksum trong `SHA256SUMS` xác nhận ZIP dữ liệu nguyên vẹn. Dataset và thư mục giải nén chỉ ở instance của đội; không commit chúng lên GitHub.

## Cài môi trường, tải model và tạo SFT

```bash
python -m pip install -r requirements-train.txt
python scripts/download_models.py
python training/prepare_sft.py --data-root data/raw/train+public_test
```

Model tải theo revision cố định trong `configs/baseline.json`. Việc tải model cần mạng trong giai đoạn chuẩn bị; inference dùng weights local và chế độ offline.

## Đo baseline và chạy smoke train

Đo model gốc trước, sau đó train 64 mẫu trong một epoch để xác nhận CUDA, VRAM, loss, lưu adapter và rollout dev đều chạy được:

```bash
python training/evaluate_dev.py --report reports/runs/dev_base.json
python training/train_lora.py \
  --train-jsonl data/processed/sft/train.jsonl \
  --dev-jsonl data/processed/sft/dev.jsonl \
  --limit-train-samples 64 --limit-dev-samples 16 --epochs 1 \
  --output-dir models/lifemate-qwen3-4b-smoke
python training/evaluate_dev.py \
  --adapter models/lifemate-qwen3-4b-smoke/final \
  --report reports/runs/dev_smoke.json
```

Nếu smoke chạy hết và có report, mới train toàn bộ train split:

```bash
python training/train_lora.py \
  --train-jsonl data/processed/sft/train.jsonl \
  --dev-jsonl data/processed/sft/dev.jsonl
python training/evaluate_dev.py \
  --adapter models/lifemate-qwen3-4b-lora/final \
  --report reports/runs/dev_lora.json
```

Chọn adapter theo state/tool diagnostics và latency trên cùng dev; eval loss chỉ là tín hiệu trong train. Nếu OOM hoặc quá chậm, giữ model gốc, giảm `--max-sequence-length`/batching có đo lường hoặc benchmark PhoWhisper-small trước khi chạy train dài.

## Sinh kết quả public

Sau khi chọn ứng viên:

```bash
python scripts/run_inference.py \
  --data data/raw/train+public_test --split public_test \
  --output result/submission.jsonl
```

Với LoRA, thêm `--adapter models/lifemate-qwen3-4b-lora/final`. Kiểm tra đủ 634 dòng và nộp file JSONL qua portal. Mỗi mẫu phải dưới 30 giây trên máy chấm A30; Vast GPU nhanh hơn không thay thế benchmark A30. Script log latency từng mẫu ở stderr.

## Cấu trúc sau khi bung bundle

```text
lifemate/
  data/archives/train+public_test.zip  # ZIP gốc BTC, giữ nguyên
  data/raw/train+public_test/          # tạo bởi lệnh unzip ở trên
  configs/
  docs/vast_runbook.md
  scripts/
  src/
  training/
  SHA256SUMS
```
