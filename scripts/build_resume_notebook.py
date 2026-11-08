"""Generate notebooks/lifemate_vast_resume_1524.ipynb.

The notebook embeds the patched code files (gzip+base64, SHA-256 checked) so that the only file
to upload to a new Vast instance is the notebook itself (plus the 1524 handoff archive).
Re-run this script after changing any file in OVERLAY_FILES to keep the notebook in sync.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY_FILES = (
    "training/train_lora.py",
    "training/evaluate_dev.py",
    "src/lifemate/runner.py",
    "src/lifemate/models.py",
    "scripts/run_inference.py",
)
SOURCE_NOTEBOOK = ROOT / "notebooks/lifemate_vast_handoff_1016.ipynb"
OUTPUT_NOTEBOOK = ROOT / "notebooks/lifemate_vast_resume_1524.ipynb"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def overlay() -> dict:
    result = {}
    for rel in OVERLAY_FILES:
        new = (ROOT / rel).read_bytes()
        old = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, check=True, capture_output=True).stdout
        result[rel] = {
            "old": [sha256(old)],
            "new": sha256(new),
            "data": base64.b64encode(gzip.compress(new, 9, mtime=0)).decode("ascii"),
        }
    return result


def source_cell(notebook: dict, index: int, must_contain: str) -> str:
    text = "".join(notebook["cells"][index]["source"])
    assert must_contain in text, (index, must_contain)
    return text


MD_TITLE = """# LifeMate Vast — tiếp tục từ checkpoint-1524 (nhanh hơn, có đo dev trước khi nộp)

Chỉ cần upload **notebook này** và file `lifemate-resume-latest.tar.gz` (checkpoint 1524) lên Vast, chọn kernel Python 3.12+ rồi Run All từ trên xuống.

**Notebook làm gì**
1. Tải bundle code + dữ liệu cũ, nhập checkpoint 1524, **sao lưu 1524 sang thư mục riêng** trước mọi thứ khác.
2. Áp bản code đã sửa (có SHA-256 kiểm tra, giữ bản gốc `.orig`).
3. Chạy theo `MODE` ở cell cấu hình:
   - `warmstart` *(mặc định)*: run **mới và ngắn** từ trọng số adapter 1524, optimizer/scheduler mới, LR thấp giảm dần về 0. Ghi ra thư mục riêng, không đụng `full/`.
   - `continue`: tiếp tục đúng run cũ từ 1524 (giống notebook trước). Run này sẽ tự dừng sau ~1–2 lần eval vì early stopping vẫn so với loss tốt nhất 0,03678.
   - `eval_only`: chỉ đo dev và (tuỳ chọn) tạo public submission cho 1524.
4. Đo dev rollout cho 1524 và adapter mới, **chọn adapter theo dev**, rồi chỉ tạo public submission cho adapter được chọn.
5. Đóng gói kết quả để tải về.

**Lưu ý trung thực:** `warmstart` với LR 2e-5 là *giả thuyết* (lịch LR cũ gần như không giảm). Chưa có bằng chứng nó tăng điểm; bước đo dev ở giữa là để kiểm chứng. Dev chỉ có 154 mẫu nên sai số khoảng ±4 điểm phần trăm; chênh lệch nhỏ không đủ kết luận.

**Thời gian (ước tính, chưa đo trên GPU của bạn):** các checkpoint đầu cho thấy khoảng 30 giây/bước tối ưu hoá, tức ~4–5 giờ cho 508 bước. Đo dev 154 mẫu mỗi adapter thêm một khoảng nữa; cache ASR giúp lần đo thứ hai và public nhanh hơn.
"""

CELL_CONFIG = r'''from pathlib import Path
import os

# --- Bundle code + dữ liệu cơ sở (đã dùng ở các notebook trước) ---
RELEASE_REPO = "Vincent-1512/ZALO_AI_CHALLENGE_2026"
RELEASE_TAG = "lifemate-resume-508-20261107"
BUNDLE_NAME = "lifemate-vast-resume-clean-508-20261107.tar.gz"
BUNDLE_SHA256 = "b7413e49c002a7afb6fea52889da370b7961a3e9c847201dac0ff0202f764bbf"
WORKSPACE_DIR = Path("/workspace")  # nếu đã có thư mục lifemate/ khác, chọn thư mục mới

# --- Gói checkpoint-1524 ---
# Upload file này lên /workspace (hoặc điền tag Release nếu đã đưa lên GitHub Release).
HANDOFF_ARCHIVE = Path("/workspace/lifemate-resume-latest.tar.gz")
HANDOFF_NAME = "lifemate-resume-latest.tar.gz"
HANDOFF_SHA256 = "8849e69703fc2a62dd084b15b8ea2f1803b6fd4b8b193d2cf224518f27048b4c"
HANDOFF_RELEASE_TAG = ""

# --- Chế độ chạy ---
MODE = "warmstart"      # "warmstart" | "continue" | "eval_only"
WARM_LR = 2e-5          # giả thuyết cần đo; đổi MỘT thông số mỗi lần nếu thử lại
WARM_MAX_STEPS = 508    # 1 epoch dữ liệu augmentation; cosine giảm về ~0 trong đúng số bước này
WARM_SEED = 2027        # khác 2026 để không lặp lại đúng thứ tự dữ liệu epoch đầu của run cũ

EVAL_BASELINE = True    # đo dev cho 1524 (cần để so sánh; dùng cache ASR nên lần sau nhanh hơn)
RUN_PUBLIC = True       # tạo public submission cho adapter được chọn
PUBLIC_CHOICE = "auto"  # "auto" (chọn theo dev) | "baseline" (1524) | "candidate" (adapter mới)
MIN_GAIN = 0.0          # adapter mới phải vượt 1524 về state-match thêm ít nhất chừng này mới được chọn

# --- Thông số run gốc (không đổi: dùng để tái tạo đúng dữ liệu và kiểm hash) ---
RUN_ID = "vast-20261106-124816"
SEED = 2026
LR_ORIG = 5e-5
MAX_EPOCHS = 30
MAX_SEQUENCE_LENGTH = 8192
GRAD_ACCUMULATION = 16
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
EVALS_PER_EPOCH = 2
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_THRESHOLD = 0.001
SCHEMA_PERMUTATIONS = 1

assert MODE in {"warmstart", "continue", "eval_only"}
assert PUBLIC_CHOICE in {"auto", "baseline", "candidate"}
print("MODE =", MODE)
'''

CELL_BOOTSTRAP_CALL = r'''PROJECT_DIR = bootstrap_github_release(
    RELEASE_REPO, RELEASE_TAG, BUNDLE_NAME, BUNDLE_SHA256, WORKSPACE_DIR
)
os.chdir(PROJECT_DIR)
print("Project:", PROJECT_DIR)
'''

CELL_HANDOFF = r'''import json, shutil, tarfile
from pathlib import PurePosixPath

if HANDOFF_RELEASE_TAG:
    asset = get_release_asset(RELEASE_REPO, HANDOFF_RELEASE_TAG, HANDOFF_NAME, "")
    download_asset(RELEASE_REPO, asset, HANDOFF_ARCHIVE, HANDOFF_SHA256, "")
if not HANDOFF_ARCHIVE.is_file():
    raise FileNotFoundError(f"Upload {HANDOFF_NAME} lên {HANDOFF_ARCHIVE.parent} trước khi chạy cell này.")
if file_sha256(HANDOFF_ARCHIVE) != HANDOFF_SHA256:
    raise RuntimeError("SHA-256 của gói handoff không khớp; upload lại file.")

RUN_DIR = PROJECT_DIR / "lifemate_runs" / RUN_ID
FULL_DIR = RUN_DIR / "full"
BASE_CKPT = FULL_DIR / "checkpoint-1524"

# Chỉ lấy file của run này. Không ghi đè code trong gói (code cũ hơn); bỏ file augmentation vì sẽ tạo lại và kiểm hash.
prefix = f"lifemate/lifemate_runs/{RUN_ID}/"
skip = {prefix + "train_schema_augmented.jsonl"}
if not BASE_CKPT.exists():
    with tarfile.open(HANDOFF_ARCHIVE, "r:gz") as archive:
        members = []
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Đường dẫn không an toàn trong handoff: {member.name}")
            if member.name.startswith(prefix) and member.name not in skip:
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Loại file không được phép: {member.name}")
                members.append(member)
        archive.extractall(WORKSPACE_DIR, members=members, filter="data")

required = ["adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
            "rng_state.pth", "trainer_state.json"]
missing = [str(BASE_CKPT / name) for name in required if not (BASE_CKPT / name).is_file()]
if missing or not (FULL_DIR / "preflight.json").is_file():
    raise FileNotFoundError("Handoff thiếu file:\n" + "\n".join(missing + [str(FULL_DIR / "preflight.json")]))
state = json.loads((BASE_CKPT / "trainer_state.json").read_text())
if state.get("global_step") != 1524:
    raise RuntimeError(f"Checkpoint không đúng bước 1524: {state.get('global_step')}")

# Bản sao lưu nằm ngoài full/ nên không bị save_total_limit xoá.
BACKUP_CKPT = RUN_DIR / "backup_checkpoint-1524"
if not BACKUP_CKPT.exists():
    shutil.copytree(BASE_CKPT, BACKUP_CKPT)
print("Checkpoint 1524 OK; đã sao lưu tại", BACKUP_CKPT)
'''

CELL_OVERLAY = r'''import base64, gzip, hashlib, json, shutil

OVERLAY = json.loads(__OVERLAY_JSON__)

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

for rel, spec in OVERLAY.items():
    target = PROJECT_DIR / rel
    current = _sha(target.read_bytes())
    if current == spec["new"]:
        print("đã áp dụng:", rel)
        continue
    if current not in spec["old"]:
        raise RuntimeError(f"{rel} không phải bản gốc đã biết (sha {current[:12]}); dừng để tránh ghi đè nhầm.")
    new_bytes = gzip.decompress(base64.b64decode(spec["data"]))
    if _sha(new_bytes) != spec["new"]:
        raise RuntimeError(f"Dữ liệu nhúng của {rel} bị hỏng.")
    backup = target.with_name(target.name + ".orig")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_bytes(new_bytes)
    print("đã cập nhật:", rel, "(bản gốc:", backup.name + ")")
'''

CELL_PATHS = r'''import hashlib, json, os, random, shutil, subprocess, sys, time

# Giữ thứ tự tool theo bảng chữ cái như baseline (không xếp theo transcript ASR).
os.environ["LIFEMATE_TOOL_RETRIEVAL"] = "off"

DATA_ROOT = PROJECT_DIR / "data/raw/train+public_test"
MODEL_PATH = PROJECT_DIR / "models/Qwen3-4B-Instruct-2507"
WARM_DIR = RUN_DIR / "warm_from_1524"
ASR_CACHE = RUN_DIR / "asr_cache.json"   # chỉ memoize transcript; không đổi kết quả
PUBLIC_DIR = RUN_DIR / "public"

def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=PROJECT_DIR, check=True)

def latest_checkpoint(folder):
    found = [p for p in folder.glob("checkpoint-*") if p.is_dir()]
    return max(found, key=lambda p: int(p.name.rsplit("-", 1)[1])) if found else None

print("Run:", RUN_DIR)
'''

CELL_CHECKS = r'''subprocess.run(["nvidia-smi"], check=True)
free_gib = shutil.disk_usage(PROJECT_DIR).free / 1024**3
print(f"Free disk: {free_gib:.1f} GiB")
if free_gib < 18:
    raise RuntimeError("Ổ đĩa còn dưới 18 GiB; cần dọn hoặc thuê instance disk lớn hơn.")

try:
    import torch
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu130"])
    import torch
if not torch.cuda.is_available():
    raise RuntimeError("Kernel không thấy CUDA GPU. Chọn kernel Python của /venv/main.")
props = torch.cuda.get_device_properties(0)
print(f"GPU={props.name}; VRAM={props.total_memory/1024**3:.1f} GiB")
if props.total_memory < 23 * 1024**3:
    raise RuntimeError("Cần GPU 24 GB VRAM trở lên.")

if not (DATA_ROOT / "train/train.jsonl").is_file():
    dataset_archive = PROJECT_DIR / "data/archives/train+public_test.zip"
    if not dataset_archive.is_file():
        raise FileNotFoundError(dataset_archive)
    (PROJECT_DIR / "data/raw").mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-n", "-q", str(dataset_archive), "train+public_test/*", "-d", str(PROJECT_DIR / "data/raw")], check=True)

for path in (DATA_ROOT / "train/train.jsonl", DATA_ROOT / "train/train_tools_catalog.json",
             DATA_ROOT / "public_test/public_test.jsonl", BASE_CKPT / "adapter_model.safetensors"):
    if not path.is_file():
        raise FileNotFoundError(path)
print("GPU, dataset và checkpoint: OK")
'''

CELL_PREFLIGHT = r'''common = [
    "--train-jsonl", TRAIN_FOR_RUN, "--dev-jsonl", DEV_JSONL,
    "--model-path", MODEL_PATH,
    "--gradient-accumulation-steps", GRAD_ACCUMULATION,
    "--lora-rank", LORA_RANK, "--lora-alpha", LORA_ALPHA, "--lora-dropout", LORA_DROPOUT,
    "--evals-per-epoch", EVALS_PER_EPOCH,
]
preflight = RUN_DIR / "resume_preflight.json"
run([sys.executable, "training/train_lora.py", *common, "--seed", SEED,
     "--max-sequence-length", MAX_SEQUENCE_LENGTH, "--learning-rate", LR_ORIG,
     "--preflight-only", "--auto-context", "--preflight-report", preflight,
     "--output-dir", RUN_DIR / "preflight"])
selected_context = json.loads(preflight.read_text())["selected_context"]
print("Preflight context:", selected_context)

# Dữ liệu train/dev phải trùng SHA-256 với run gốc, nếu không thì không phải tiếp tục đúng run.
original = json.loads((FULL_DIR / "preflight.json").read_text())
current = json.loads(preflight.read_text())
for split in ("train", "dev"):
    if current["splits"][split]["data_sha256"] != original["splits"][split]["data_sha256"]:
        raise RuntimeError(f"Dữ liệu {split} khác run gốc. Dừng trước khi nạp checkpoint.")
print("Train/dev SHA-256 khớp run gốc.")
'''

CELL_TRAIN = r'''CAND_ADAPTER = None
CAND_DIR = None

if MODE == "continue":
    CAND_DIR = FULL_DIR
    if (FULL_DIR / "final/adapter_config.json").is_file():
        print("full/final đã có; bỏ qua train.")
    else:
        resume_from = latest_checkpoint(FULL_DIR)
        run([sys.executable, "training/train_lora.py", *common, "--seed", SEED,
             "--max-sequence-length", selected_context, "--learning-rate", LR_ORIG,
             "--early-stopping-patience", EARLY_STOPPING_PATIENCE,
             "--early-stopping-threshold", EARLY_STOPPING_THRESHOLD,
             "--epochs", MAX_EPOCHS, "--save-total-limit", 5,
             "--output-dir", FULL_DIR, "--resume-from-checkpoint", resume_from])
    CAND_ADAPTER = FULL_DIR / "final"

elif MODE == "warmstart":
    CAND_DIR = WARM_DIR
    if (WARM_DIR / "final/adapter_config.json").is_file():
        print("warm_from_1524/final đã có; bỏ qua train.")
    else:
        interrupted = latest_checkpoint(WARM_DIR)
        # Run warm-start bị ngắt giữa chừng thì tiếp tục đúng run đó; nếu không, khởi tạo từ adapter 1524.
        start = ["--resume-from-checkpoint", interrupted] if interrupted else ["--warm-start-adapter", BASE_CKPT]
        run([sys.executable, "training/train_lora.py", *common, "--seed", WARM_SEED,
             "--max-sequence-length", selected_context, "--learning-rate", WARM_LR,
             "--epochs", MAX_EPOCHS, "--max-steps", WARM_MAX_STEPS,
             "--disable-early-stopping", "--no-load-best", "--save-total-limit", 4,
             "--output-dir", WARM_DIR, *start])
    CAND_ADAPTER = WARM_DIR / "final"

if CAND_ADAPTER is not None and not (CAND_ADAPTER / "adapter_config.json").is_file():
    raise FileNotFoundError(f"Train kết thúc nhưng không thấy {CAND_ADAPTER}")
print("Adapter ứng viên:", CAND_ADAPTER)
'''

CELL_DEV = r'''def dev_eval(tag, adapter):
    report = RUN_DIR / f"dev_{tag}.json"
    weights = Path(adapter) / "adapter_model.safetensors"
    if report.is_file() and report.stat().st_mtime > weights.stat().st_mtime:
        print(f"Dùng lại báo cáo dev mới hơn adapter: {report.name}")
    else:
        run([sys.executable, "training/evaluate_dev.py",
             "--data-root", DATA_ROOT, "--adapter", adapter, "--limit-samples", 0,
             "--report", report, "--per-sample-log", RUN_DIR / f"dev_{tag}_samples.jsonl",
             "--asr-cache", ASR_CACHE])
    return json.loads(report.read_text())

BASE_REPORT = dev_eval("ckpt1524", BASE_CKPT) if EVAL_BASELINE else None
CAND_REPORT = dev_eval(f"cand_{MODE}", CAND_ADAPTER) if (CAND_ADAPTER and MODE != "eval_only") else None
'''

CELL_COMPARE = r'''def row(name, r):
    if r is None:
        return f"{name:<22} (không đo)"
    return (f"{name:<22} state_match={r['changed_app_state_match_rate']:.4f} ({r['state_match_count']}/{r['samples']})  "
            f"call_F1_diag={r['tool_call_f1_diagnostic']:.4f}  empty_pred={r['empty_prediction_samples']} "
            f"(gold rỗng={r['empty_gold_samples']})  end={r['end_reason_counts']}")

print(row("1524 (baseline)", BASE_REPORT))
print(row(f"ứng viên [{MODE}]", CAND_REPORT))

SELECTED = "baseline"
if CAND_REPORT is not None and BASE_REPORT is not None:
    cand_key = (CAND_REPORT["changed_app_state_match_rate"], CAND_REPORT["tool_call_f1_diagnostic"])
    base_key = (BASE_REPORT["changed_app_state_match_rate"] + MIN_GAIN, BASE_REPORT["tool_call_f1_diagnostic"])
    SELECTED = "candidate" if cand_key > base_key else "baseline"
    print("\nDev có 154 mẫu: chênh vài điểm phần trăm có thể chỉ là nhiễu.")
elif CAND_REPORT is not None:
    SELECTED = "candidate"
if PUBLIC_CHOICE != "auto":
    SELECTED = PUBLIC_CHOICE
if SELECTED == "candidate" and CAND_ADAPTER is None:
    raise RuntimeError("Không có adapter ứng viên (MODE=eval_only); chọn PUBLIC_CHOICE='baseline'.")
SELECTED_ADAPTER = CAND_ADAPTER if SELECTED == "candidate" else BASE_CKPT
print("\nAdapter được chọn cho public:", SELECTED, "->", SELECTED_ADAPTER)
'''

CELL_PUBLIC = r'''submission_path = None
if RUN_PUBLIC:
    public_input = DATA_ROOT / "public_test/public_test.jsonl"
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = PUBLIC_DIR / f"submission-{SELECTED}-{MODE}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    run([sys.executable, "scripts/run_inference.py",
         "--data", DATA_ROOT, "--split", "public_test", "--output", submission_path,
         "--adapter", SELECTED_ADAPTER, "--asr-cache", ASR_CACHE])

    with public_input.open(encoding="utf-8") as source:
        expected_ids = [json.loads(line)["id"] for line in source if line.strip()]
    with submission_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    predicted_ids = [r.get("id") for r in rows]
    if len(rows) != len(expected_ids) or len(set(predicted_ids)) != len(expected_ids):
        raise RuntimeError(f"Số dòng/ID không khớp: output={len(rows)}, input={len(expected_ids)}")
    if set(predicted_ids) != set(expected_ids):
        raise RuntimeError("ID trong submission không khớp public test.")
    if any(set(r) != {"id", "tool_calls", "state_hash"} for r in rows):
        raise RuntimeError("Submission sai schema; cần id, tool_calls, state_hash.")
    print(f"Submission hợp lệ: {submission_path} ({len(rows)} dòng; rỗng={sum(1 for r in rows if not r['tool_calls'])})")
else:
    print("RUN_PUBLIC=False: bỏ qua public inference.")
'''

CELL_PACK = r'''import tarfile

stamp = time.strftime("%Y%m%d-%H%M%S")
pack = WORKSPACE_DIR / f"lifemate-handoff-{MODE}-{stamp}.tar.gz"
include = [p for p in RUN_DIR.glob("dev_*") if p.is_file()]
if submission_path:
    include.append(submission_path)
include.append(RUN_DIR / "resume_preflight.json")
if CAND_DIR is not None:
    for name in ("final",):
        if (CAND_DIR / name).exists():
            include.append(CAND_DIR / name)
    newest = latest_checkpoint(CAND_DIR)
    if newest is not None:
        include.append(newest)   # có optimizer/scheduler để còn resume tiếp
    if (CAND_DIR / "preflight.json").is_file():
        include.append(CAND_DIR / "preflight.json")
with tarfile.open(pack, "w:gz") as archive:
    for path in include:
        archive.add(path, arcname=str(Path("lifemate") / path.relative_to(PROJECT_DIR)))
digest = hashlib.sha256(pack.read_bytes()).hexdigest()
(pack.parent / (pack.name + ".sha256")).write_text(f"{digest}  {pack.name}\n")
print("Gói kết quả:", pack, f"({pack.stat().st_size / 1024**2:.0f} MiB)")
print("SHA-256:", digest)
print("Tải file này về qua trình duyệt file của Jupyter trước khi huỷ instance.")
'''

MD_DONE = """## Hoàn tất

- Adapter mới (nếu có): `lifemate_runs/vast-20261106-124816/warm_from_1524/final/` (hoặc `full/final/` ở chế độ `continue`).
- Báo cáo dev: `lifemate_runs/.../dev_*.json`; từng mẫu: `dev_*_samples.jsonl` (ASR vs gold, call đúng/sai, lý do dừng).
- Public submission: `lifemate_runs/.../public/submission-*.jsonl`.
- Checkpoint 1524 đã sao lưu ở `lifemate_runs/.../backup_checkpoint-1524/`.
- Gói tải về: `/workspace/lifemate-handoff-<mode>-<time>.tar.gz` (+ `.sha256`).

Sau khi nộp public và có điểm, ghi điểm cạnh tên file submission để lần sau đối chiếu đúng.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def main() -> None:
    old = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    helpers = source_cell(old, 3, "def bootstrap_github_release")
    prepare = source_cell(old, 12, "prepare_sft.py")
    augment = source_cell(old, 14, "train_schema_augmented.jsonl")
    config_cell = CELL_CONFIG
    overlay_cell = CELL_OVERLAY.replace("__OVERLAY_JSON__", repr(json.dumps(overlay())))
    cells = [
        md(MD_TITLE),
        code(config_cell),
        md("## 1. Tải code + dữ liệu cơ sở và nhập checkpoint 1524"),
        code(helpers),
        code(CELL_BOOTSTRAP_CALL),
        code(CELL_HANDOFF),
        md("## 2. Áp bản code đã sửa\n\nThêm `--warm-start-adapter`, `--max-steps`, `--no-load-best`, log từng mẫu dev, cache ASR. "
           "Prompt và cách suy luận mặc định **không đổi**."),
        code(overlay_cell),
        md("## 3. Đường dẫn và kiểm tra GPU / dữ liệu"),
        code(CELL_PATHS),
        code(CELL_CHECKS),
        md("## 4. Cài thư viện và tái tạo dữ liệu train/dev"),
        code(prepare),
        code(augment),
        md("## 5. Preflight và kiểm hash dữ liệu"),
        code(CELL_PREFLIGHT),
        md("## 6. Train (theo `MODE`)\n\nRun nằm trong thư mục riêng nên 1524 không bị ghi đè. "
           "Nếu instance chết giữa chừng, chạy lại notebook: cell này tự tiếp tục đúng từ checkpoint mới nhất của run đang dở."),
        code(CELL_TRAIN),
        md("## 7. Đo dev (rollout đầy đủ) cho 1524 và adapter mới"),
        code(CELL_DEV),
        code(CELL_COMPARE),
        md("## 8. Public submission cho adapter được chọn"),
        code(CELL_PUBLIC),
        md("## 9. Đóng gói kết quả"),
        code(CELL_PACK),
        md(MD_DONE),
    ]
    notebook = {"cells": cells, "metadata": old["metadata"], "nbformat": 4, "nbformat_minor": 5}
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUTPUT_NOTEBOOK, len(cells), "cells")


if __name__ == "__main__":
    main()
