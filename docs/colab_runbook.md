# Runbook Colab LifeMate v3

Notebook: [`notebooks/lifemate_colab_train.ipynb`](../notebooks/lifemate_colab_train.ipynb). Upload bundle code và dataset thành hai tệp riêng.

## 1. Bố trí Google Drive

```text
MyDrive/AI-ZALO/
├── dataset/
│   └── train+public_test/train/
│       ├── train.jsonl
│       └── train_tools_catalog.json
├── bundles/
│   └── lifemate-colab-bundle-v3-<hash>.tar.gz
└── lifemate_runs/
    └── <RUN_ID>/
        ├── run_info.json
        ├── *_<ATTEMPT_ID>.log
        ├── lifemate-qwen3-4b-smoke-training/ # adapter smoke
        └── lifemate-qwen3-4b-full-training/  # adapter và checkpoints
```

Notebook cũng nhận adapter smoke từ cấu trúc run v2 `lifemate-qwen3-4b-smoke-adapter/`.

- Upload `train.jsonl` và `train_tools_catalog.json` từ thư mục `train/` của kit vào `dataset/train+public_test/train/` **một lần**. Hai file này là toàn bộ dữ liệu đầu vào mà baseline cần.
- Tạo `bundles/` và upload bundle v3 vào đó.
- Import notebook v3 vào Colab, chọn GPU. Khi mở runtime mới, chạy notebook từ đầu bằng **Runtime → Run all**.

Notebook chỉ đọc `train.jsonl` và `train_tools_catalog.json` từ Drive. Nó không đọc public/private labels, audio hoặc thư mục environment. Bundle v3 không chứa dataset hoặc weights.

## 2. Chọn action trong cell đầu

| `ACTION` | Mục đích |
|---|---|
| `prepare` | Dựng source/SFT, kiểm GPU và preflight; không train. Mặc định an toàn. |
| `smoke` | Chạy 64 mẫu train / 16 dev, 1 epoch; lưu adapter thẳng vào Drive. |
| `full` | Bắt đầu full train mới trong run đã chọn; cần smoke adapter của run đó. |
| `resume` | Tiếp tục full train từ checkpoint đầy đủ mới nhất trong cùng run. |

### Run mới

Để `RUN_ID = ""`, chọn `ACTION = "smoke"` rồi chạy toàn notebook. Ghi lại ID được in ra. Khi smoke đạt và muốn train đủ dữ liệu, nhập ID đó, đặt `ACTION = "full"` rồi chạy toàn notebook.

### Tiếp tục run hiện tại

Smoke đã lưu ở `MyDrive/AI-ZALO/lifemate_runs/20261105-114926`. Để bắt đầu full train từ trạng thái đó, đặt:

```python
ACTION = "full"
RUN_ID = "20261105-114926"
```

Nếu phiên full train bị ngắt sau khi có checkpoint, đặt `ACTION = "resume"` và giữ nguyên `RUN_ID`. Khi resume, giữ nguyên seed, learning rate, epochs, accumulation và context của lần train đó.

Smoke chỉ xác nhận pipeline/GPU chạy được; full train khởi tạo từ Qwen gốc, không tiếp tục trọng số từ smoke adapter.

## 3. Siêu tham số và dữ liệu đầu ra

Thay đổi siêu tham số tại cell `Cấu hình`: `SEED`, `LEARNING_RATE`, `FULL_EPOCHS`, `GRADIENT_ACCUMULATION_STEPS`, `SAVE_STEPS`, `CONTEXT_CEILING`, `SMOKE_TRAIN_SAMPLES`, `SMOKE_DEV_SAMPLES`.

Mỗi run lưu `run_info.json`, `dataset_info.json`, split manifest, preflight report, log và adapter/checkpoint trong `lifemate_runs/<RUN_ID>/`. Adapter/checkpoint được ghi lên Drive; source, SFT và model nằm trên `/content` và được dựng/tải lại khi runtime mới.

Trên T4, 8.192 token là ceiling theo dữ liệu, không phải cam kết vừa VRAM. Smoke thật vẫn là kiểm tra bắt buộc trước full train. `SAVE_STEPS` mặc định là 5 để giảm lượng công việc mất khi Colab ngắt; checkpoint thường xuyên làm tăng ghi dữ liệu lên Drive.

Train split cố định hiện có 1.236 mẫu / 4.062 next-action records; dev có 154 / 521; holdout có 155 / 503. Smoke 64/16 mẫu không thay thế đánh giá rollout trên ToolEnv. Notebook này không train ASR, không dùng nhãn test và không chạy scorer cuộc thi.
