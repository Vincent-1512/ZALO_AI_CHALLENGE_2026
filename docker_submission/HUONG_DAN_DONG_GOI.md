# Hướng dẫn đóng gói Docker — ZAC2026 LifeMate

Mục tiêu: dựng **một image Docker** chứa mô hình đã huấn luyện, chạy thử không mạng trên GPU, xuất file `zac2026_TeamName.tar.gz` + MD5, upload Google Drive và gửi lại link + MD5 + log cho chủ dự án.
**Không nộp lên cổng BTC** nếu chủ dự án chưa nói (Test Solution chỉ có 1 lần; nộp chính thức tối đa 2 lần). Không dán token hay khóa riêng vào chat.

Hạn nộp cuối: **16/11/2026 23:59 (GMT+7)**.

## 1. Cần gì

| Mục | Yêu cầu |
|---|---|
| Hệ điều hành | Linux x86_64 (Ubuntu 22.04), hoặc Windows 10/11 + WSL2 (Ubuntu) + Docker Desktop |
| GPU | NVIDIA 16 GB VRAM trở lên (RTX 3090/4090/5090…), driver hỗ trợ CUDA 12.8 trở lên (driver 570+) |
| Docker | Docker Engine + **nvidia-container-toolkit** (Docker phải dùng được `--gpus all`) |
| Ổ đĩa | ≥ 100 GB trống (script chặn nếu < 90 GB) |
| RAM | ≥ 32 GB |
| Mạng | Có mạng lúc dựng (tải ~11 GB model từ HuggingFace, ~1.2 GB dữ liệu thử từ GitHub release) |

Không có GPU thì chỉ dựng được image, không thử được → **không nên**: lần Test Solution của BTC chỉ có 1 lần.

Chưa có GPU nhà: thuê trên vast.ai template **"Ubuntu 22.04 VM"** (không dùng template Jupyter/Docker thường: container không chạy được Docker bên trong), RTX 3090/4090, ổ đĩa 150 GB, thêm SSH key vào tài khoản Vast. Khoảng 2–4 giờ, ~$2–3.

## 2. Các bước

```bash
git clone https://github.com/Vincent-1512/zalo_ai_challenge_2026.git
cd zalo_ai_challenge_2026
git checkout docker-submission
```

Nếu `docker run --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi -L` báo lỗi "could not select device driver", cài toolkit (Ubuntu):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
```

Chạy build (nền, mất SSH vẫn tiếp tục):

```bash
nohup bash docker_submission/vm_build.sh > build.log 2>&1 &
tail -f build.log
```

Script chạy lần lượt: `preflight` (kiểm tra Docker/GPU/ổ đĩa) → `download` (3 repo HuggingFace) → `cudacheck` (CUDA có chạy thật không, tối đa 120 s) → `merge` (gộp adapter LoRA vào Qwen, ghi `MANIFEST.json`) → `data` (dữ liệu thử) → `build` (dựng image `zac2026-teamname:v1`) → `test` (chạy 30 mẫu với `--network none`) → `export` (`zac2026_TeamName.tar.gz` + `.md5`).
Chạy lại cùng lệnh sau khi sửa lỗi: bước đã xong tự bỏ qua (dấu trong `.done/`). Làm lại một bước: `rm .done/<bước>`.

## 3. Kết quả phải đạt (gửi lại cho chủ dự án)

1. Dòng in cuối bước `merge`: tổng tham số (< 6B, LLM < 5B).
2. Bước `build`: dòng `torch ... cuda True [...]` (danh sách kiến trúc có `sm_120` hoặc `sm_89`/`sm_86` tùy GPU của bạn).
3. Bước `test`: dòng `mẫu: 30 | time ms: p50 ... max ...` — **max phải < 30000 ms**, mong đợi dưới ~25000.
4. Cuối cùng: tên file, kích thước và MD5 trong `zac2026_TeamName.md5`.
5. Upload `zac2026_TeamName.tar.gz` lên Google Drive, chia sẻ "Anyone with the link", gửi link + MD5.

## 4. Lỗi thường gặp

- `LỖI: CUDA không chạy được` ở bước `cudacheck`: GPU/driver của máy hỏng (nvidia-smi bình thường nhưng CUDA treo); đổi máy.
- `docker không dùng được GPU`: thiếu nvidia-container-toolkit, hoặc driver cũ hơn CUDA 12.8.
- `Ổ đĩa ... cần >= 90 GB`: giải phóng ổ đĩa.
- Merge chậm bất thường: script dùng GPU nếu có (`device_map="cuda"`); trên CPU gộp half-precision rất chậm.
- Hết giờ lúc `test` (một mẫu ≥ 30 s): gửi log, **không tự sửa code**.

## 5. Không làm

- Không commit trọng số hay file `.tar.gz` vào git (`.gitignore` đã chặn).
- Không đổi code, Dockerfile, URL HuggingFace, hay adapter nếu chưa báo chủ dự án.
- Không đọc/mở thư mục `*_environment/` trong dữ liệu.

## 6. Ghi chú kỹ thuật

- Image dùng `nvidia/cuda:12.8.1-devel-ubuntu22.04` + `torch==2.9.1` (cu128). Máy chấm của BTC: RTX 5090, CUDA 13.2, không mạng; image CUDA thấp hơn vẫn hợp lệ.
- Trọng số trong image: `saved_models/asr/` (PhoWhisper-large đã fine-tune) và `saved_models/llm/` (Qwen3-4B-Instruct-2507 đã gộp LoRA). Docker Guide mẫu ghi `saved_models/models.safetensors`; chưa xác nhận với BTC.
- Ngân sách vòng lặp agent mặc định 18 s (`LIFEMATE_LOOP_BUDGET_S` trong `zac_runtime.py`) để tổng thời gian mỗi mẫu dưới 30 s.
- Nguồn trọng số (đều công khai): `Qwen/Qwen3-4B-Instruct-2507`, `vincent1512/lifemate-phowhisper-large-ft`, `vincent1512/lifemate-qwen-adapter`.
