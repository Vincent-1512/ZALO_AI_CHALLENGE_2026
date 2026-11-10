# Dựng và nộp image (làm trên máy có GPU; tham chiếu: Docker Guide ZAC2026 LifeMate)

Hạn nộp: **16/11/2026 23:59 GMT+7**. Tối đa 2 lần nộp, chỉ lần cuối được xét. Máy chấm: RTX 5090 32 GB, CUDA 13.2, không có mạng.

## 1. Chuẩn bị trọng số (máy bất kỳ có ~20 GB RAM)
```bash
python scripts/merge_for_submission.py \
  --llm-base models/Qwen3-4B-Instruct-2507 \
  --adapter <thư mục adapter cuối cùng> \
  --asr-ft models/PhoWhisper-large-ft \
  --output docker_submission/saved_models
```
Script gộp LoRA vào bản fp16 của Qwen, chép ASR fine-tune, đếm tham số từ header safetensors và ghi `saved_models/MANIFEST.json` (SHA256 từng file). Dừng nếu tổng ≥ 6B hoặc LLM ≥ 5B.

Cấu trúc kết quả: `saved_models/asr/*.safetensors` và `saved_models/llm/*.safetensors`.
Guide ghi mẫu `saved_models/models.safetensors`; chưa xác nhận được BTC có bắt buộc đúng một file hay không (xem mục Rủi ro).

## 2. Kiểm tra điểm mô hình đã gộp (BẮT BUỘC, cần GPU)
Chạy `training/evaluate_dev.py` trên dev với `--llm-model docker_submission/saved_models/llm` (không adapter) và `--asr-model docker_submission/saved_models/asr`, so với điểm adapter chưa gộp (mốc dev của adapter `train_noise` khi chưa gộp: 53.46; adapter 1524 cũ: 54.62). Chênh lệch lớn nghĩa là việc gộp làm đổi hành vi: dừng lại, không dựng image.

## 3. Dựng container (cần `--network host` chỉ lúc dựng)
```bash
sudo docker run --gpus '"device=0"' --network host -it --name zac2026 nvidia/cuda:12.8.1-devel-ubuntu22.04 /bin/bash
# trong container:
apt-get update && apt-get install -y python3 python3-pip libsndfile1 ffmpeg
# Cài torch hỗ trợ RTX 5090 (sm_120) + CUDA 13 TRƯỚC. Chọn wheel đúng ở pytorch.org rồi kiểm tra:
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```
Từ terminal khác trên máy host:
```bash
sudo docker cp docker_submission/. zac2026:/code/
```
Trong container:
```bash
cd /code && pip install -r requirements.txt
```

## 4. Thử như BTC (host)
```bash
sudo docker commit zac2026 zac2026-teamname:v1
sudo docker run --gpus '"device=0"' -v /mnt/zac2026/data:/data:ro -v /mnt/zac2026/result:/result \
  zac2026-teamname:v1 /bin/bash /code/predict.sh
# Jupyter:
sudo docker run -it --gpus '"device=0"' -p 9777:9777 -v /mnt/zac2026/data:/data:ro -v /mnt/zac2026/result:/result \
  zac2026-teamname:v1 /bin/bash /code/start_jupyter.sh    # http://localhost:9777 , mật khẩu zac2026
```
Kiểm tra: `submission.jsonl` và `jupyter_submission.jsonl` cho **cùng** tool_calls/state_hash; `time` mỗi mẫu < 25 000 ms (giới hạn 30 s, giữ biên); chạy khi tắt mạng (`--network none`) vẫn thành công.

## 5. Xuất và nộp
```bash
sudo docker save zac2026-teamname:v1 | gzip > zac2026_TeamName.tar.gz
md5sum zac2026_TeamName.tar.gz            # macOS: md5 <file>
sudo docker load -i zac2026_TeamName.tar.gz   # kiểm tra lại trên máy khác rồi chạy lại bước 4
```
Upload Google Drive (Anyone with link), nộp link + MD5. Thử qua tab "Test Solution" của portal trước khi nộp lần cuối.

## Rủi ro đang mở
- Tên/số lượng file trọng số (`models.safetensors` hay thư mục `asr/` + `llm/`): chưa xác nhận với BTC.
- Wheel PyTorch cho sm_120 + CUDA 13.2 chưa kiểm chứng; `requirements.txt` không ghim torch.
- Điểm mô hình đã gộp so với chưa gộp chưa đo (mục 2).
- Thời gian/mẫu trên 5090 chưa đo; số liệu trên 3090: p95 10.35 s, max 12.7 s (ASR large fine-tune).
