#!/usr/bin/env bash
# Chạy TRONG MỘT MÁY VM CÓ DOCKER + GPU (Vast: template "Ubuntu 22.04 VM"; instance Jupyter/Docker thường KHÔNG chạy được docker).
#
# Cách dùng (trên VM, thư mục /root):
#   tar xzf zac_build_kit.tar.gz && cd zac_build_kit
#   nohup bash docker_submission/vm_build.sh > build.log 2>&1 &
#   tail -f build.log          # Ctrl+C chỉ thoát tail, tiến trình vẫn chạy dù mất SSH
#
# Chạy lại an toàn: mỗi bước xong đều ghi dấu vào .done/ và sẽ được bỏ qua. Muốn làm lại một bước: rm .done/<bước>
# Bước: preflight -> download -> cudacheck -> merge -> data -> build -> test -> export
set -euo pipefail
KIT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$KIT"
mkdir -p .done work
LLM_REPO="${LLM_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
ASR_REPO="${ASR_REPO:-vincent1512/lifemate-phowhisper-large-ft}"
ADAPTER_REPO="${ADAPTER_REPO:-vincent1512/lifemate-qwen-adapter}"
TAG="${TAG:-zac2026-teamname:v1}"
OUT_NAME="${OUT_NAME:-zac2026_TeamName}"
TEST_LIMIT="${TEST_LIMIT:-30}"      # số mẫu chạy thử (0 = toàn bộ public_test, ~2 giờ trên 3090)
DATA_BUNDLE_URL="${DATA_BUNDLE_URL:-https://github.com/Vincent-1512/ZALO_AI_CHALLENGE_2026/releases/download/lifemate-resume-508-20261107/lifemate-vast-resume-clean-508-20261107.tar.gz}"

step() { [ -f ".done/$1" ] && { echo "== bỏ qua $1 (đã xong)"; return 1; }; echo "== $(date +%H:%M:%S) bắt đầu $1"; return 0; }
ok()   { touch ".done/$1"; echo "== $(date +%H:%M:%S) xong $1"; }

if step preflight; then
  command -v docker >/dev/null || { echo "LỖI: không có docker. Đây phải là máy VM (template Ubuntu 22.04 VM), không phải container."; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv
  FREE=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
  echo "Ổ đĩa trống: ${FREE} GB"; [ "$FREE" -ge 90 ] || { echo "LỖI: cần >= 90 GB trống"; exit 1; }
  docker info >/dev/null 2>&1 || { sudo systemctl start docker 2>/dev/null || true; docker info >/dev/null; }
  # Docker có thấy GPU không (nvidia-container-toolkit)?
  docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi -L \
    || { echo "LỖI: docker không dùng được GPU (thiếu nvidia-container-toolkit hoặc driver thấp hơn CUDA 13.2)"; exit 1; }
  ok preflight
fi

if step download; then
  python3 -m pip install -q -U huggingface_hub transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 safetensors==0.6.2 sentencepiece
  python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 || python3 -m pip install -q torch
  python3 - <<PY
from huggingface_hub import snapshot_download
for repo, dest in (("$LLM_REPO","work/qwen"),("$ASR_REPO","work/asr_ft"),("$ADAPTER_REPO","work/adapter")):
    snapshot_download(repo_id=repo, local_dir=dest)
    print("ok", repo, flush=True)
PY
  ok download
fi

if step cudacheck; then
  # CUDA thật sự chạy được trên máy này không? (một số máy thuê: nvidia-smi bình thường nhưng CUDA treo -> đổi máy ngay)
  timeout 120 python3 -c "import torch; x=torch.zeros(1).cuda(); print('CUDA OK', torch.cuda.get_device_name(0), torch.__version__)" \
    || { echo "LỖI: CUDA không chạy được (treo hoặc lỗi). Máy này hỏng GPU: hủy và thuê VM khác."; exit 1; }
  ok cudacheck
fi

if step merge; then
  python3 scripts/merge_for_submission.py --llm-base work/qwen --adapter work/adapter --asr-ft work/asr_ft --output docker_submission/saved_models
  head -20 docker_submission/saved_models/MANIFEST.json
  ok merge
fi

# Dữ liệu thử (chỉ để test trong VM; không đưa vào image): lấy public_test từ bản release GitHub công khai của repo.
if step data; then
  if [ ! -f /data/public_test/public_test.jsonl ]; then
    curl -fL --retry 3 -o work/bundle.tar.gz "$DATA_BUNDLE_URL"
    mkdir -p work/bundle
    tar xzf work/bundle.tar.gz -C work/bundle --wildcards '*train+public_test/public_test/*' '*train+public_test/VERSION' || true
    SRC=$(dirname "$(find work/bundle -name public_test.jsonl | head -1)")
    [ -n "$SRC" ] && [ -f "$SRC/public_test.jsonl" ] || { echo "LỖI: không tìm thấy public_test.jsonl trong bundle"; exit 1; }
    sudo mkdir -p /data && sudo cp -r "$SRC" /data/public_test
    rm -rf work/bundle work/bundle.tar.gz
  fi
  ls /data/public_test | head
  ok data
fi

if step build; then
  docker build -t "$TAG" docker_submission
  docker run --rm --gpus all "$TAG" python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_arch_list())"
  ok build
fi

if step test; then
  mkdir -p /tmp/zac_result && rm -f /tmp/zac_result/*
  # --network none = giống máy chấm (không mạng). Chạy ${TEST_LIMIT} mẫu đầu để kiểm thời gian và chạy-được.
  LIM=""; [ "$TEST_LIMIT" -gt 0 ] && LIM="--limit $TEST_LIMIT"
  docker run --rm --gpus '"device=0"' --network none -v /data:/data:ro -v /tmp/zac_result:/result "$TAG" \
    /bin/bash -c "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=2026; cd /code && python3 predict.py $LIM"
  python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("/tmp/zac_result/submission.jsonl",encoding="utf-8") if l.strip()]
t=sorted(r["time"] for r in rows)
print("mẫu:", len(rows), "| time ms: p50", t[len(t)//2], "max", t[-1], "| rỗng:", sum(1 for r in rows if not r["tool_calls"]))
print("khoá:", sorted(rows[0].keys()))
assert t[-1] < 30000, "có mẫu >= 30 s"
PY
  ok test
fi

if step export; then
  docker save "$TAG" | gzip > "${OUT_NAME}.tar.gz"
  md5sum "${OUT_NAME}.tar.gz" | tee "${OUT_NAME}.md5"
  ls -la "${OUT_NAME}.tar.gz"
  ok export
fi
echo "HOÀN TẤT. File nộp: $KIT/${OUT_NAME}.tar.gz (md5 trong ${OUT_NAME}.md5)"
