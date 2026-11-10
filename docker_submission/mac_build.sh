#!/usr/bin/env bash
# Dựng image trên macOS (chip Apple/Intel) KHÔNG cần GPU. Image build cho linux/amd64 (máy chấm x86_64 + RTX 5090).
# Cách dùng (từ thư mục gốc kit hoặc repo):  bash docker_submission/mac_build.sh 2>&1 | tee mac_build.log
# Cần: Docker Desktop đang chạy (Settings > General: bật "Use Rosetta for x86_64/amd64 emulation" nếu có),
#      >= 100 GB ổ đĩa trống, RAM >= 16 GB (Docker Desktop: Resources > Memory >= 8 GB), python3.
# Chạy lại được: bước nào xong sẽ ghi dấu vào .done_mac/
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$KIT"
mkdir -p .done_mac work
LLM_REPO="${LLM_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
ASR_REPO="${ASR_REPO:-vincent1512/lifemate-phowhisper-large-ft}"
ADAPTER_REPO="${ADAPTER_REPO:-vincent1512/lifemate-qwen-adapter}"
TAG="${TAG:-zac2026-teamname:v1}"
OUT_NAME="${OUT_NAME:-zac2026_TeamName}"
step() { [ -f ".done_mac/$1" ] && { echo "== bỏ qua $1 (đã xong)"; return 1; }; echo "== $(date +%H:%M:%S) bắt đầu $1"; return 0; }
ok()   { touch ".done_mac/$1"; echo "== $(date +%H:%M:%S) xong $1"; }

if step preflight; then
  command -v docker >/dev/null || { echo "LỖI: chưa cài Docker Desktop"; exit 1; }
  docker info >/dev/null 2>&1 || { echo "LỖI: Docker Desktop chưa chạy"; exit 1; }
  FREE=$(df -g . | awk 'NR==2{print $4}'); echo "Ổ đĩa trống: ${FREE} GB"; [ "$FREE" -ge 80 ] || { echo "LỖI: cần >= 80 GB trống"; exit 1; }
  ok preflight
fi

if step venv; then
  python3 -m venv .venv_mac
  ./.venv_mac/bin/pip install -q -U pip
  ./.venv_mac/bin/pip install -q huggingface_hub transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 safetensors==0.6.2 sentencepiece torch
  ok venv
fi
PY=./.venv_mac/bin/python

if step download; then
  $PY - <<PYX
from huggingface_hub import snapshot_download
for repo, dest in (("$LLM_REPO","work/qwen"),("$ASR_REPO","work/asr_ft"),("$ADAPTER_REPO","work/adapter")):
    snapshot_download(repo_id=repo, local_dir=dest); print("ok", repo, flush=True)
PYX
  ok download
fi

if step merge; then
  # Gộp trên CPU (fp32 -> fp16), cần ~20 GB RAM trống; mất vài phút.
  $PY scripts/merge_for_submission.py --llm-base work/qwen --adapter work/adapter --asr-ft work/asr_ft --output docker_submission/saved_models
  head -20 docker_submission/saved_models/MANIFEST.json
  ok merge
fi

if step build; then
  # amd64 = kiến trúc máy chấm. Trên Mac ARM chạy qua giả lập: chậm (30-90 phút), bình thường.
  docker buildx build --platform linux/amd64 -t "$TAG" --load docker_submission
  ok build
fi

if step smoke; then
  # Kiểm tra nhẹ KHÔNG cần GPU: import được thư viện, trọng số có mặt, mã nguồn biên dịch được, đủ file BTC yêu cầu.
  docker run --rm --platform linux/amd64 "$TAG" /bin/bash -c '
    set -e; cd /code
    ls predict.py preprocessing.py predict.sh start_jupyter.sh requirements.txt predict_notebook.ipynb training_code/README.md >/dev/null
    ls saved_models/asr/*.safetensors saved_models/llm/*.safetensors >/dev/null
    python3 -m py_compile predict.py preprocessing.py zac_runtime.py
    python3 -c "import torch, transformers, peft, soundfile, scipy; print(\"imports OK torch\", torch.__version__, \"arch_list\", torch.cuda.get_arch_list())"
    python3 predict.py --help >/dev/null && echo "predict.py --help OK"
    du -sh saved_models'
  ok smoke
fi

if step export; then
  docker save "$TAG" | gzip > "${OUT_NAME}.tar.gz"
  md5 "${OUT_NAME}.tar.gz" | tee "${OUT_NAME}.md5"
  ls -la "${OUT_NAME}.tar.gz"
  ok export
fi
echo "HOÀN TẤT. File: $KIT/${OUT_NAME}.tar.gz (MD5 trong ${OUT_NAME}.md5). CHƯA chạy thử trên GPU."
