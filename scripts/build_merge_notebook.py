"""Builds notebooks/lifemate_vast_merge.ipynb: merge LoRA -> saved_models, re-score merged model on dev. Run on Vast."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
merge_src = (ROOT / "scripts/merge_for_submission.py").read_text(encoding="utf-8")
assert "'''" not in merge_src and not merge_src.rstrip().endswith("\\")

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}
def code(text): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}

cells = [
md("""
# LifeMate — gộp adapter Qwen vào model đầy đủ và đo lại dev (chạy trên Vast, máy ĐANG CÓ `/workspace/lifemate`)
Mục đích: tạo `saved_models/{asr,llm}` cho image nộp bài và kiểm tra bản đã gộp cho điểm dev gần bản chưa gộp (53.46).
Không train, không tải model. Cần ~20 GB RAM trống và 1 GPU (để đo dev). Chạy lần lượt từ trên xuống.
"""),
code("""
import json, os, subprocess, sys, statistics, time
from pathlib import Path

PROJECT_DIR = Path("/workspace/lifemate")
os.chdir(PROJECT_DIR)
RUNS = sorted((PROJECT_DIR / "lifemate_runs").glob("vast-*"))
assert RUNS, "không thấy lifemate_runs/vast-*"
RUN_DIR = RUNS[-1]
ADAPTER = RUN_DIR / "stage_noise/final"           # adapter Qwen mới (train_noise)
LLM_BASE = PROJECT_DIR / "models/Qwen3-4B-Instruct-2507"
ASR_FT = PROJECT_DIR / "models/PhoWhisper-large-ft"
OUT = PROJECT_DIR / "saved_models"
DEV_UNMERGED = 53.46                               # điểm dev của adapter này khi chưa gộp
TOLERANCE = 2.0                                    # giả thuyết: chênh > 2 điểm là bất thường (dev 154 mẫu có sai số ~±4)

HF_ADAPTER_REPO = "vincent1512/lifemate-qwen-adapter"   # nếu chưa có adapter trong run này thì tải từ HuggingFace
if not (ADAPTER / "adapter_model.safetensors").is_file():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_ADAPTER_REPO, local_dir=str(ADAPTER))
    print("Đã tải adapter từ", HF_ADAPTER_REPO)
for p in (ADAPTER / "adapter_model.safetensors", LLM_BASE / "config.json", ASR_FT / "config.json"):
    assert p.is_file(), f"thiếu {p}"
print("RUN_DIR:", RUN_DIR); print("adapter:", ADAPTER)
print(subprocess.run(["bash","-lc","free -g | head -2; df -h /workspace | tail -1; nvidia-smi --query-gpu=name,memory.used --format=csv"],capture_output=True,text=True).stdout)
"""),
code("Path('scripts').mkdir(exist_ok=True)\nPath('scripts/merge_for_submission.py').write_text(r'''" + merge_src + "''', encoding='utf-8')\nprint('đã ghi scripts/merge_for_submission.py')"),
code("""
# Gộp LoRA vào bản fp16 của Qwen, chép ASR fine-tune, đếm tham số (<6B tổng, LLM <5B) và ghi MANIFEST.json.
r = subprocess.run([sys.executable, "scripts/merge_for_submission.py", "--llm-base", str(LLM_BASE),
                    "--adapter", str(ADAPTER), "--asr-ft", str(ASR_FT), "--output", str(OUT)])
assert r.returncode == 0, "merge lỗi hoặc vượt ngân sách tham số: xem log trên"
print(subprocess.run(["bash","-lc",f"du -sh {OUT}/llm {OUT}/asr; ls {OUT}/llm {OUT}/asr | head -40"],capture_output=True,text=True).stdout)
"""),
code("""
# Đo dev bằng model ĐÃ GỘP (không adapter), cùng ASR fine-tune. Không dùng cache ASR cũ để phép đo sạch.
report = RUN_DIR / "dev_merged.json"
samples = RUN_DIR / "dev_merged_samples.jsonl"
r = subprocess.run([sys.executable, "training/evaluate_dev.py", "--data-root", "data/raw/train+public_test",
                    "--llm-model", str(OUT / "llm"), "--asr-model", str(OUT / "asr"),
                    "--limit-samples", "0", "--report", str(report), "--per-sample-log", str(samples)])
assert r.returncode == 0, "evaluate_dev lỗi"
rep = json.loads(report.read_text())
score = rep["official_score_100"]
print(f"ĐIỂM DEV bản đã gộp = {score:.2f}  (chưa gộp: {DEV_UNMERGED})  chênh = {score-DEV_UNMERGED:+.2f}")
print("hash_match =", round(rep["full_hash_match_rate"], 3), "| dự đoán rỗng =", rep["empty_prediction_samples"], "| ngoài phạm vi đúng =", rep["oos_exact_rate"])
times = sorted(json.loads(l)["total_seconds"] for l in samples.open(encoding="utf-8") if l.strip())
print(f"thời gian/mẫu (3090): p50={statistics.median(times):.1f}s p95={times[int(0.95*(len(times)-1))]:.1f}s max={times[-1]:.1f}s (giới hạn chấm 30s)")
print("=> OK để dựng image" if abs(score-DEV_UNMERGED) <= TOLERANCE else "=> CHÊNH LỚN: DỪNG, gửi log cho Claude, chưa dựng image")
"""),
md("""
## Lưu kết quả
Cell dưới nén `MANIFEST.json`, báo cáo dev và adapter (nhỏ, ~130 MB). **Không** nén `saved_models/` (~11 GB): máy dựng image sẽ gộp lại từ adapter + hai model gốc bằng `scripts/merge_for_submission.py`.
Tải file `.tar.gz` và `.sha256` về máy trước khi huỷ Vast.
"""),
code("""
import tarfile, hashlib
stamp = time.strftime("%Y%m%d-%H%M%S")
pack = PROJECT_DIR / f"lifemate-merge-{stamp}.tar.gz"
with tarfile.open(pack, "w:gz") as ar:
    ar.add(OUT / "MANIFEST.json", arcname="saved_models/MANIFEST.json")
    for p in (RUN_DIR/"dev_merged.json", RUN_DIR/"dev_merged_samples.jsonl"):
        ar.add(p, arcname=f"reports/{p.name}")
    ar.add(ADAPTER, arcname="adapter_stage_noise_final")
d = hashlib.sha256(pack.read_bytes()).hexdigest()
(pack.parent / (pack.name + ".sha256")).write_text(f"{d}  {pack.name}\\n")
print("Gói:", pack, f"({pack.stat().st_size/1024**2:.0f} MiB)"); print("SHA-256:", d)
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks/lifemate_vast_merge.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
