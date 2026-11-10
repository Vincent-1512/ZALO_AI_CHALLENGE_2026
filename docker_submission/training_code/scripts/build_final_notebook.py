#!/usr/bin/env python3
"""Build notebooks/lifemate_vast_final.ipynb: one idempotent, phase-driven notebook for multi-day Vast use.

Code is taken from a pinned public commit (CODE_COMMIT) so new files (asr_correct.py ...) are available;
bundle/handoff/bootstrap cells are reused verbatim from the resume notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESUME = ROOT / "notebooks/lifemate_vast_resume_1524.ipynb"
OUTPUT = ROOT / "notebooks/lifemate_vast_final.ipynb"


def reuse(nb: dict, index: int, must_contain: str) -> str:
    text = "".join(nb["cells"][index]["source"])
    if must_contain not in text:
        raise SystemExit(f"resume notebook cell {index} changed (missing {must_contain!r})")
    return text


MD_TITLE = """# LifeMate — notebook FINAL (từ checkpoint 1524 đến lúc nộp)

**Cách dùng (mỗi lần thuê máy):** sửa dòng `PHASES` ở mục A → *Run All* → tải gói `lifemate-final-*.tar.gz` về trước khi huỷ máy. Lần sau upload gói đó lên `/workspace` và điền tên vào `RESUME_PACK`. Bước nào đã có kết quả sẽ tự bỏ qua, nên chạy lại an toàn.

| Thứ tự | `PHASES` | Làm gì | Thời gian (RTX 3090) |
|---|---|---|---|
| 1 | `{"oracle"}` | Đo dev với transcript đúng so với ASR thật → biết lỗi nằm ở ASR hay ở mô hình | ~30 phút |
| 2 | `{"asr"}` | So sánh ASR medium/large, có/không sửa lỗi theo ngữ cảnh | ~1–1.5 giờ |
| 3 | `{"train_noise","eval","public"}` | Train 1 epoch từ 1524 với transcript của ASR mới (large_ft + large gốc để có nhiễu), đo dev, tạo submission | ~7–8 giờ |
| 4 | `{"train_final","eval","public"}` | Train trên train+dev+holdout (dev sau đó là số lạc quan) | ~9–10 giờ |
| 5 | `{"latency"}` | Đo độ trễ mỗi mẫu trước khi đóng Docker | ~20 phút |
| tuỳ chọn | `{"zs"}` | Giữ lại 8 app khỏi train để đo khả năng dùng tool chưa thấy | ~10 giờ |

**Cần nhớ**
- Điểm dev in ra (`ĐIỂM_CHÍNH_THỨC`) tính theo công thức BTC: trạng thái cuối phải khớp hash VÀ khớp call (F1). Dev chỉ 154 mẫu nên sai số khoảng ±4 điểm; eval loss không dự đoán điểm.
- Các số là ước tính; thứ tự có thể đổi sau khi có kết quả bước 1.
- Hạn nộp Docker: 16/11 23:59 (GMT+7). Gói Docker làm riêng, theo Docker Guide của BTC.
"""

CELL_CONFIG = r'''from pathlib import Path
import os

# ---------------- nguồn code/dữ liệu (đã dùng ở các notebook trước) ----------------
RELEASE_REPO = "Vincent-1512/ZALO_AI_CHALLENGE_2026"
RELEASE_TAG = "lifemate-resume-508-20261107"
BUNDLE_NAME = "lifemate-vast-resume-clean-508-20261107.tar.gz"
BUNDLE_SHA256 = "b7413e49c002a7afb6fea52889da370b7961a3e9c847201dac0ff0202f764bbf"
WORKSPACE_DIR = Path("/workspace")

HANDOFF_ARCHIVE = Path("/workspace/lifemate-resume-latest.tar.gz")
HANDOFF_NAME = "lifemate-resume-latest.tar.gz"
HANDOFF_SHA256 = "8849e69703fc2a62dd084b15b8ea2f1803b6fd4b8b193d2cf224518f27048b4c"
HANDOFF_RELEASE_TAG = "lifemate-resume-1524-20261109"

# Code mới (asr_correct.py, build_asr_train_data.py ...) lấy từ commit ghim cứng trên GitHub public.
CODE_COMMIT = "__CODE_COMMIT__"

# Gói kết quả của lần chạy trước (file lifemate-final-*.tar.gz bạn đã tải về). Upload vào /workspace rồi điền tên; "" = lần đầu.
RESUME_PACK = ""

# ---------------- chọn việc cần làm ở lần chạy này ----------------
PHASES = {"train_noise", "eval", "public"}   # tập con của: "oracle", "zs", "asr", "train_noise", "train_final", "eval", "public", "latency"

# ASR: các biến thể đem so trên dev (adapter 1524). Tên: (model, asr_correct)
ASR_VARIANTS = [("large_ft", False)]   # chỉ dùng khi PHASES có "asr". Đã đo: medium 30.96, large 33.89, large_ft 54.62 (dev); +correct làm TỤT điểm -> bỏ
ASR_CHOICE = "large_ft"   # ASR dùng cho mọi bước. "auto" = chọn theo dev; hoặc "medium"/"large"/"large_ft" (+ hậu tố "+correct" nếu muốn)
ASR_MIN_GAIN = 0.02       # dev nhỏ (154 mẫu) -> không đổi ASR vì chênh lệch nhỏ hơn ngưỡng này
ASR_LARGE_REPO = "vinai/PhoWhisper-large"   # ~1.55B tham số; tổng hệ thống ~5.6B < 6B, LLM 4.0B < 5B
ASR_FT_REPO = "vincent1512/lifemate-phowhisper-large-ft"   # PhoWhisper-large đã fine-tune trên audio train (dev WER 0.2469 -> 0.0921); repo private -> cần HF token (đọc)
ALT_NOISE_FRACTION = 0.5  # tỉ lệ dòng "có ASR" lấy transcript từ PhoWhisper-large GỐC (nhiễu ~25% WER) thay vì large_ft. Lý do: large_ft đã học thuộc audio train nên transcript train gần như sạch, còn test sẽ nhiễu hơn. 0 = tắt. Đây là giả thuyết, chưa đo.

# Train
NOISE_PROB = 0.5          # xác suất dùng transcript ASR (thay vì text gốc) khi dựng dữ liệu train
STAGE_LR = {"noise": 2e-5, "final": 2e-5}   # 2e-5 là giá trị đã đo ở warm-start (public 25.71 -> 25.99); các giá trị khác là giả thuyết
STAGE_SEED = {"noise": 2028, "final": 2029}
FINAL_INIT = "noise"      # adapter khởi tạo cho train_final: "noise" (nếu đã có) hoặc "baseline" (1524)
EVALS_PER_EPOCH = 2       # eval loss chỉ để theo dõi; early stopping tắt vì eval loss không đo điểm
SMOKE_TEST = True         # chạy thử 2 bước trước mỗi giai đoạn train mới
KEEP_CKPTS = 3            # giữ ít checkpoint để tiết kiệm đĩa; resume dùng cái mới nhất

# Public
PUBLIC_STAGES = ["noise"]            # adapter nào tạo public submission: "baseline", "noise", "final"
SKIP_PUBLIC_IF_WORSE_BY = 0.03       # chặn nếu dev state-match kém baseline hơn 3 điểm (không áp cho "final" vì dev đã lẫn vào train)
LATENCY_STAGE = "noise"
LATENCY_LIMIT_S = 25.0               # biên an toàn so với 30 s/mẫu (GPU của bạn có thể khác A30)

# ---------------- thông số run gốc (KHÔNG đổi) ----------------
RUN_ID = "vast-20261106-124816"
SEED = 2026
MAX_SEQUENCE_LENGTH = 8192
GRAD_ACCUMULATION = 16
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
EVALS_PER_EPOCH_ORIG = 2
SCHEMA_PERMUTATIONS = 1
LR_ORIG = 5e-5
MAX_EPOCHS = 30

# Thí nghiệm zero-shot: giữ lại app khỏi train để đo khả năng dùng tool/app CHƯA THẤY (public test có 11 app lạ)
ZS_HELD_OUT = ["bike_rental", "door_lock", "home_automation", "home_security", "learning", "news", "robot_vacuum", "video_meeting"]
ZS_LR = 1e-4              # train TỪ ĐẦU (không warm-start) nên dùng LR cao hơn warm-start; giả thuyết
ZS_EPOCHS = 2
ZS_SEED = 2030

KNOWN = {"oracle", "asr", "train_noise", "train_final", "eval", "public", "latency", "zs"}
assert PHASES <= KNOWN, PHASES - KNOWN
assert set(PUBLIC_STAGES) <= {"baseline", "noise", "final"}
assert FINAL_INIT in {"noise", "baseline"}
assert 0 < NOISE_PROB <= 1
assert 0 <= ALT_NOISE_FRACTION <= 1
print("PHASES =", sorted(PHASES))
'''

CELL_SYNC = r'''import shutil, subprocess, tempfile

# Lấy code từ commit ghim cứng; chỉ chép thư mục code, KHÔNG đụng data/, models/, lifemate_runs/.
SYNC_DIRS = ("src", "training", "scripts", "configs", "tests")
SYNC_FILES = ("requirements-train.txt", "requirements-inference.txt")
stamp_file = PROJECT_DIR / ".code_commit"
if stamp_file.is_file() and stamp_file.read_text().strip() == CODE_COMMIT:
    print("Code đã đúng commit", CODE_COMMIT[:10])
else:
    tmp = Path(tempfile.mkdtemp(prefix="lifemate-code-"))
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("fetch", "--depth", "1", f"https://github.com/{RELEASE_REPO}.git", CODE_COMMIT)
    git("checkout", "-q", "FETCH_HEAD")
    head = git("rev-parse", "HEAD")
    if head != CODE_COMMIT:
        raise RuntimeError(f"Commit không khớp: {head} != {CODE_COMMIT}")
    for name in SYNC_DIRS:
        dest = PROJECT_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(tmp / name, dest, ignore=shutil.ignore_patterns("__pycache__"))
    for name in SYNC_FILES:
        shutil.copy2(tmp / name, PROJECT_DIR / name)
    shutil.rmtree(tmp)
    stamp_file.write_text(CODE_COMMIT + "\n")
    print("Đã đồng bộ code từ commit", CODE_COMMIT[:10])
assert (PROJECT_DIR / "src/lifemate/asr_correct.py").is_file(), "thiếu asr_correct.py: CODE_COMMIT sai nhánh?"
'''

CELL_RESUME_PACK = r'''import tarfile
from pathlib import PurePosixPath

# Khôi phục kết quả lần trước (adapter các giai đoạn, báo cáo dev, cache ASR, submission).
if RESUME_PACK:
    pack_path = WORKSPACE_DIR / RESUME_PACK
    if not pack_path.is_file():
        raise FileNotFoundError(f"Upload {RESUME_PACK} vào {WORKSPACE_DIR} trước.")
    sidecar = pack_path.with_name(pack_path.name + ".sha256")
    if sidecar.is_file() and file_sha256(pack_path) != sidecar.read_text().split()[0]:
        raise RuntimeError("SHA-256 của gói kết quả không khớp.")
    with tarfile.open(pack_path, "r:gz") as archive:
        members = []
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Mục không an toàn trong gói: {member.name}")
            members.append(member)
        archive.extractall(WORKSPACE_DIR, members=members, filter="data")
    print("Đã khôi phục từ", RESUME_PACK)
else:
    print("RESUME_PACK rỗng: lần chạy đầu (hoặc cùng máy, không mất đĩa).")
'''

CELL_ASR_SETUP = r'''ASR_PATHS = {"medium": PROJECT_DIR / "models/PhoWhisper-medium", "large": PROJECT_DIR / "models/PhoWhisper-large",
             "large_ft": PROJECT_DIR / "models/PhoWhisper-large-ft"}
ASR_CACHES = {"medium": ASR_CACHE, "large": RUN_DIR / "asr_cache_large.json",
              "large_ft": RUN_DIR / "asr_cache_large_ft.json"}   # mỗi model một cache (metadata khác nhau)

def fetch_model(repo, dest):
    if (dest / "config.json").is_file():
        return
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or None
    try:
        snapshot_download(repo_id=repo, local_dir=str(dest), token=token)
    except Exception as exc:
        if token is not None:
            raise
        from getpass import getpass
        print(f"{repo}: tải không được ({type(exc).__name__}); repo private? Nhập token HuggingFace loại READ (đừng dán vào chat):")
        snapshot_download(repo_id=repo, local_dir=str(dest), token=getpass("HF token: "))

def count_params(path):
    import math
    from safetensors import safe_open
    total = 0
    for f in Path(path).glob("*.safetensors"):
        with safe_open(str(f), "pt") as sf:
            total += sum(math.prod(sf.get_slice(k).get_shape()) for k in sf.keys())
    return total

wanted = {m for m, _ in ASR_VARIANTS} if "asr" in PHASES else set()
if ASR_CHOICE != "auto":
    wanted.add(ASR_CHOICE.partition("+")[0])
if PHASES & {"train_noise", "train_final"} and ALT_NOISE_FRACTION > 0:
    wanted.add("large")
for name, repo in (("large_ft", ASR_FT_REPO), ("large", ASR_LARGE_REPO)):
    if name in wanted:
        fetch_model(repo, ASR_PATHS[name])
        print(f"PhoWhisper {name}: {count_params(ASR_PATHS[name]) / 1e9:.3f} B tham số (đếm từ file safetensors).")
'''

CELL_EVAL_FN = r'''import statistics

def dev_eval(tag, adapter, asr="medium", correct=False, oracle=False):
    report = RUN_DIR / f"dev_{tag}.json"
    weights = Path(adapter) / "adapter_model.safetensors"
    if (report.is_file() and report.stat().st_mtime > weights.stat().st_mtime
            and "official_score_100" in json.loads(report.read_text())):
        print(f"Dùng lại báo cáo dev: {report.name}")
    else:
        run([sys.executable, "training/evaluate_dev.py",
             "--data-root", DATA_ROOT, "--adapter", adapter, "--limit-samples", 0,
             "--report", report, "--per-sample-log", RUN_DIR / f"dev_{tag}_samples.jsonl",
             "--asr-model", ASR_PATHS[asr], "--asr-cache", ASR_CACHES[asr],
             *(["--asr-correct"] if correct else []), *(["--oracle-transcript"] if oracle else [])])
    return json.loads(report.read_text())

def latency_summary(tag):
    log = RUN_DIR / f"dev_{tag}_samples.jsonl"
    times = sorted(json.loads(l)["total_seconds"] for l in log.open(encoding="utf-8") if l.strip())
    return {"p50": statistics.median(times), "p95": times[int(0.95 * (len(times) - 1))],
            "max": times[-1], "over_limit": sum(t > LATENCY_LIMIT_S for t in times), "n": len(times)}

def fmt(r):
    official = f"ĐIỂM_CHÍNH_THỨC≈{r['official_score_100']:.2f} hash_đủ={r['full_hash_match_rate']:.3f} " if "official_score_100" in r else ""
    return (f"{official}state_match={r['changed_app_state_match_rate']:.4f} ({r['state_match_count']}/{r['samples']}) "
            f"call_F1_diag={r['tool_call_f1_diagnostic']:.4f} empty_pred={r['empty_prediction_samples']} end={r['end_reason_counts']}")

def variant_name(asr, correct):
    return asr + ("+correct" if correct else "")

STAGE_ADAPTER = {"baseline": BASE_CKPT, "noise": RUN_DIR / "stage_noise/final", "final": RUN_DIR / "stage_final/final"}
'''

CELL_ASR_PHASE = r'''ASR_RESULTS = {}
if "asr" in PHASES:
    for asr, correct in ASR_VARIANTS:
        name = variant_name(asr, correct)
        tag = "ckpt1524" if name == "medium" else f"ckpt1524_{name.replace('+', '_')}"
        ASR_RESULTS[name] = dev_eval(tag, BASE_CKPT, asr, correct)
        r = ASR_RESULTS[name]
        lat = latency_summary(tag)
        print(f"{name:<16} {fmt(r)}\n{'':<16} latency p50={lat['p50']:.1f}s p95={lat['p95']:.1f}s max={lat['max']:.1f}s over{LATENCY_LIMIT_S:.0f}s={lat['over_limit']}/{lat['n']}")
    print("\nCảnh báo: sai số dev ≈ ±4 điểm; chỉ đổi ASR nếu hơn medium thường >= ASR_MIN_GAIN và độ trễ vẫn an toàn.")

def pick_asr():
    if ASR_CHOICE != "auto":
        name = ASR_CHOICE
    else:
        saved = {}
        for asr, correct in ASR_VARIANTS:
            n = variant_name(asr, correct)
            tag = "ckpt1524" if n == "medium" else f"ckpt1524_{n.replace('+', '_')}"
            f = RUN_DIR / f"dev_{tag}.json"
            if f.is_file():
                saved[n] = json.loads(f.read_text())["changed_app_state_match_rate"]
        base = saved.get("medium")
        if base is None:
            name = "medium"
        else:
            best = max(saved, key=saved.get)
            name = best if saved[best] - base >= ASR_MIN_GAIN else "medium"
    asr, _, corr = name.partition("+")
    return asr, bool(corr), name

SEL_ASR, SEL_CORRECT, SEL_NAME = pick_asr()
print("ASR dùng cho các bước sau:", SEL_NAME)
'''

CELL_BASELINE = r'''SEL_TAG_SUFFIX = SEL_NAME.replace("+", "_")
BASE_TAG = "ckpt1524" if SEL_NAME == "medium" else f"ckpt1524_{SEL_TAG_SUFFIX}"
BASE_REPORT = dev_eval(BASE_TAG, BASE_CKPT, SEL_ASR, SEL_CORRECT) if PHASES & {"eval", "public", "train_noise", "train_final", "latency"} else None
print(f"Baseline 1524 + ASR {SEL_NAME}:", fmt(BASE_REPORT) if BASE_REPORT else "không đo")
'''

CELL_TRAIN_FN = r'''def build_stage_data(stage):
    out = RUN_DIR / f"data_{stage}_{SEL_TAG_SUFFIX}_alt{int(ALT_NOISE_FRACTION * 100)}"
    target = out / "train.jsonl"
    if not target.is_file():
        cmd = [sys.executable, "scripts/build_asr_train_data.py", "--data-root", DATA_ROOT, "--output-dir", out,
               "--asr-model", ASR_PATHS[SEL_ASR], "--asr-cache", ASR_CACHES[SEL_ASR],
               "--asr-probability", NOISE_PROB, "--seed", SEED]
        if ALT_NOISE_FRACTION > 0 and SEL_ASR != "large":
            cmd += ["--alt-asr-model", ASR_PATHS["large"], "--alt-asr-cache", ASR_CACHES["large"],
                    "--alt-fraction", ALT_NOISE_FRACTION]
        if stage == "final":
            cmd.append("--include-dev-holdout")
        run(cmd)
    return target

def train_stage(stage, init_adapter):
    out = RUN_DIR / f"stage_{stage}"
    if (out / "final/adapter_config.json").is_file():
        print(f"stage_{stage}/final đã có; bỏ qua train (xoá thư mục này nếu muốn train lại với ASR mới).")
        return out / "final"
    data = build_stage_data(stage)
    rows = sum(1 for _ in data.open(encoding="utf-8"))
    steps = -(-rows // GRAD_ACCUMULATION)
    print(f"stage {stage}: {rows} bản ghi -> {steps} bước (1 epoch), LR={STAGE_LR[stage]}, init={init_adapter}")
    stage_common = list(common)
    stage_common[stage_common.index("--train-jsonl") + 1] = data
    base_cmd = [sys.executable, "training/train_lora.py", *stage_common, "--seed", STAGE_SEED[stage],
                "--max-sequence-length", selected_context, "--learning-rate", STAGE_LR[stage]]
    if SMOKE_TEST and not (RUN_DIR / f".smoke_{stage}_ok").exists():
        smoke = RUN_DIR / f"smoke_{stage}"
        if smoke.exists():
            shutil.rmtree(smoke)
        run([*base_cmd, "--epochs", 1, "--max-steps", 2, "--limit-train-samples", 8, "--limit-dev-samples", 4,
             "--evals-per-epoch", 1, "--disable-early-stopping", "--no-load-best", "--save-total-limit", 1,
             "--output-dir", smoke, "--warm-start-adapter", init_adapter])
        if not (smoke / "final/adapter_config.json").is_file():
            raise RuntimeError(f"Smoke {stage} không tạo adapter: {smoke}")
        shutil.rmtree(smoke)
        (RUN_DIR / f".smoke_{stage}_ok").write_text("ok\n")
        print("Smoke OK.")
    interrupted = latest_checkpoint(out)
    if interrupted is None and out.exists() and list(out.glob("checkpoint-*")):
        raise FileNotFoundError(f"Checkpoint dở ở {out} nhưng không đủ file để resume; giữ nguyên để kiểm tra.")
    start = ["--resume-from-checkpoint", interrupted] if interrupted else ["--warm-start-adapter", init_adapter]
    run([*base_cmd, "--epochs", MAX_EPOCHS, "--max-steps", steps, "--evals-per-epoch", EVALS_PER_EPOCH,
         "--disable-early-stopping", "--no-load-best", "--save-total-limit", KEEP_CKPTS,
         "--output-dir", out, *start])
    if not (out / "final/adapter_config.json").is_file():
        raise FileNotFoundError(f"Train xong nhưng thiếu {out / 'final'}")
    return out / "final"
'''

CELL_TRAIN_NOISE = r'''if "train_noise" in PHASES:
    train_stage("noise", BASE_CKPT)
else:
    print("Bỏ train_noise.")
'''

CELL_TRAIN_FINAL = r'''if "train_final" in PHASES:
    init = STAGE_ADAPTER["noise"] if FINAL_INIT == "noise" else BASE_CKPT
    if not (Path(init) / "adapter_config.json").is_file():
        raise FileNotFoundError(f"FINAL_INIT={FINAL_INIT} nhưng chưa có adapter {init}; chạy train_noise trước hoặc đặt FINAL_INIT='baseline'.")
    train_stage("final", init)
else:
    print("Bỏ train_final.")
'''

CELL_EVAL = r'''STAGE_REPORTS = {}
if "eval" in PHASES:
    for stage in ("noise", "final"):
        adapter = STAGE_ADAPTER[stage]
        if (adapter / "adapter_config.json").is_file():
            STAGE_REPORTS[stage] = dev_eval(f"stage_{stage}_{SEL_TAG_SUFFIX}", adapter, SEL_ASR, SEL_CORRECT)
    print("baseline(1524, medium)", fmt(BASE_REPORT))
    for stage, r in STAGE_REPORTS.items():
        note = "  [dev đã lẫn vào train -> số lạc quan, KHÔNG so với baseline]" if stage == "final" else ""
        print(f"{stage:<22}", fmt(r), note)
    print("\nDev chỉ 154 mẫu (±4 điểm). Quyết định cuối: điểm public.")
else:
    print("Bỏ eval.")
'''

CELL_PUBLIC = r'''PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS = {}
if "public" in PHASES:
    public_input = DATA_ROOT / "public_test/public_test.jsonl"
    expected_ids = [json.loads(l)["id"] for l in public_input.open(encoding="utf-8") if l.strip()]
    for stage in PUBLIC_STAGES:
        adapter = STAGE_ADAPTER[stage]
        if not (adapter / "adapter_config.json").is_file():
            print(f"Bỏ {stage}: chưa có adapter {adapter}")
            continue
        rep = STAGE_REPORTS.get(stage)
        if stage == "noise" and rep and BASE_REPORT and SKIP_PUBLIC_IF_WORSE_BY is not None:
            if rep["changed_app_state_match_rate"] < BASE_REPORT["changed_app_state_match_rate"] - SKIP_PUBLIC_IF_WORSE_BY:
                print(f"CHẶN public {stage}: dev kém baseline hơn {SKIP_PUBLIC_IF_WORSE_BY:.2f}. Đặt SKIP_PUBLIC_IF_WORSE_BY=None nếu vẫn muốn nộp.")
                continue
        asr_name, correct = ("medium", False) if stage == "baseline" else (SEL_ASR, SEL_CORRECT)
        sub = PUBLIC_DIR / f"submission-{stage}-{variant_name(asr_name, correct).replace('+', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        run([sys.executable, "scripts/run_inference.py", "--data", DATA_ROOT, "--split", "public_test", "--output", sub,
             "--adapter", adapter, "--asr-model", ASR_PATHS[asr_name], "--asr-cache", ASR_CACHES[asr_name],
             *(["--asr-correct"] if correct else [])])
        rows = [json.loads(l) for l in sub.open(encoding="utf-8") if l.strip()]
        ids = [r.get("id") for r in rows]
        if len(rows) != len(expected_ids) or set(ids) != set(expected_ids) or len(set(ids)) != len(ids):
            raise RuntimeError("ID/số dòng submission không khớp public test.")
        if any(set(r) != {"id", "tool_calls", "state_hash"} for r in rows):
            raise RuntimeError("Submission sai schema.")
        SUBMISSIONS[stage] = sub
        print(f"Submission {stage}: {sub.name} ({len(rows)} dòng, rỗng={sum(1 for r in rows if not r['tool_calls'])})")
else:
    print("Bỏ public.")
'''

CELL_LATENCY = r'''if "latency" in PHASES:
    adapter = STAGE_ADAPTER[LATENCY_STAGE]
    tag = f"stage_{LATENCY_STAGE}_{SEL_TAG_SUFFIX}"
    dev_eval(tag, adapter, SEL_ASR, SEL_CORRECT)
    lat = latency_summary(tag)
    print(f"GPU: {subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()}")
    print(f"Độ trễ/mẫu (ASR+LLM+tool): p50={lat['p50']:.1f}s p95={lat['p95']:.1f}s max={lat['max']:.1f}s; vượt {LATENCY_LIMIT_S:.0f}s: {lat['over_limit']}/{lat['n']}")
    print("Giới hạn BTC là 30 s/mẫu trên A30 (chậm hơn 3090 một chút). Nếu max gần/vượt 30 s: giảm max_new_tokens, giảm ngữ cảnh hoặc bỏ asr-correct/large.")
else:
    print("Bỏ latency.")
'''

CELL_ORACLE = r'''if "oracle" in PHASES:
    real = dev_eval("ckpt1524", BASE_CKPT, "medium", False)
    oracle = dev_eval("ckpt1524_oracle", BASE_CKPT, "medium", False, oracle=True)
    print("ASR thật   :", fmt(real))
    print("Gold text  :", fmt(oracle))
    d = oracle["official_score_100"] - real["official_score_100"]
    print(f"\\nChênh điểm chính thức (gold text - ASR thật) = {d:+.2f}")
    print("Chênh lớn (>~15): ASR là nút cổ chai -> ưu tiên ASR large/sửa lỗi/train với nhiễu ASR.")
    print("Chênh nhỏ: mô hình/policy (đa bước, điều kiện, tham số) là nút cổ chai -> ưu tiên dữ liệu và cách train.")
'''

CELL_ZS = r'''ZS_DIR = RUN_DIR / "zero_shot"
ZS_DATA = PROJECT_DIR / "data/processed/zero_shot"
ZS_OUT = ZS_DIR / "train_run"
if "zs" in PHASES:
    if not (ZS_DATA / "manifest.json").is_file():
        run([sys.executable, "training/prepare_zero_shot_split.py", "--data-root", DATA_ROOT, "--output-dir", ZS_DATA,
             "--seed", SEED, "--held-out-apps", *ZS_HELD_OUT])
    print(json.dumps(json.loads((ZS_DATA / "manifest.json").read_text())["samples"], ensure_ascii=False))
    ZS_TRAIN_AUG = ZS_DIR / "train_schema_augmented.jsonl"
    ZS_DIR.mkdir(parents=True, exist_ok=True)
    if not ZS_TRAIN_AUG.is_file():
        with (ZS_DATA / "train.jsonl").open(encoding="utf-8") as src, ZS_TRAIN_AUG.open("w", encoding="utf-8") as dst:
            for line in src:
                row = json.loads(line)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                payload = json.loads(row["messages"][1]["content"])
                tools = payload.get("available_tools", [])
                if len(tools) > 1:
                    rng = random.Random(int(hashlib.sha256(f"{SEED}:{row['sample_id']}".encode()).hexdigest()[:16], 16))
                    shuffled = list(tools); rng.shuffle(shuffled)
                    if shuffled == tools:
                        shuffled = shuffled[1:] + shuffled[:1]
                    payload["available_tools"] = shuffled
                    aug = dict(row); aug["messages"] = [dict(m) for m in row["messages"]]
                    aug["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    dst.write(json.dumps(aug, ensure_ascii=False) + "\n")
    zs_cmd = [sys.executable, "training/train_lora.py", "--train-jsonl", ZS_TRAIN_AUG, "--dev-jsonl", ZS_DATA / "dev_id.jsonl",
              "--model-path", MODEL_PATH, "--gradient-accumulation-steps", GRAD_ACCUMULATION,
              "--lora-rank", LORA_RANK, "--lora-alpha", LORA_ALPHA, "--lora-dropout", LORA_DROPOUT,
              "--evals-per-epoch", EVALS_PER_EPOCH, "--seed", ZS_SEED, "--max-sequence-length", selected_context,
              "--learning-rate", ZS_LR, "--epochs", ZS_EPOCHS, "--disable-early-stopping", "--no-load-best",
              "--save-total-limit", KEEP_CKPTS, "--output-dir", ZS_OUT]
    if (ZS_OUT / "final/adapter_config.json").is_file():
        print("zero_shot/train_run/final đã có; bỏ qua train.")
    else:
        interrupted = latest_checkpoint(ZS_OUT) if ZS_OUT.exists() else None
        if interrupted is None and ZS_OUT.exists() and list(ZS_OUT.glob("checkpoint-*")):
            raise FileNotFoundError(f"Checkpoint dở ở {ZS_OUT} nhưng không đủ file để resume; giữ nguyên để kiểm tra.")
        run([*zs_cmd, *(["--resume-from-checkpoint", interrupted] if interrupted else [])])
    ZS_ADAPTER = ZS_OUT / "final"
    ZS_REPORTS = {}
    for name in ("dev_id", "dev_zs"):
        report = RUN_DIR / f"dev_zs_{name}.json"
        if not (report.is_file() and report.stat().st_mtime > (ZS_ADAPTER / "adapter_model.safetensors").stat().st_mtime):
            run([sys.executable, "training/evaluate_dev.py", "--data-root", DATA_ROOT, "--adapter", ZS_ADAPTER,
                 "--sft-dev", ZS_DATA / f"{name}.jsonl", "--limit-samples", 0, "--report", report,
                 "--per-sample-log", RUN_DIR / f"dev_zs_{name}_samples.jsonl",
                 "--asr-model", ASR_PATHS["medium"], "--asr-cache", ASR_CACHES["medium"]])
        ZS_REPORTS[name] = json.loads(report.read_text())
    print("dev_id (app đã thấy)  :", fmt(ZS_REPORTS["dev_id"]))
    print("dev_zs (app CHƯA thấy):", fmt(ZS_REPORTS["dev_zs"]))
    gap = ZS_REPORTS["dev_id"]["changed_app_state_match_rate"] - ZS_REPORTS["dev_zs"]["changed_app_state_match_rate"]
    print(f"\nKhoảng cách tổng quát hoá (state-match): {gap:+.3f}")
    print("Lớn (>~0.15): mô hình nhớ schema -> cần augmentation đổi tên tool/tham số và dữ liệu tổng hợp cho app lạ.")
    print("Nhỏ: nguyên nhân điểm public thấp nằm chỗ khác (ASR, ngày giờ, phân phối yêu cầu).")
else:
    print("Bỏ zs.")
'''

CELL_PACK = r'''import tarfile

stamp = time.strftime("%Y%m%d-%H%M%S")
pack = WORKSPACE_DIR / f"lifemate-final-{stamp}.tar.gz"
include = [p for p in RUN_DIR.glob("dev_*") if p.is_file()]
include += [p for p in RUN_DIR.glob("asr_cache*.json") if p.is_file()]
include += [p for p in PUBLIC_DIR.glob("submission-*.jsonl")]
for stage in ("noise", "final"):
    folder = RUN_DIR / f"stage_{stage}"
    if (folder / "final").exists():
        include.append(folder / "final")
    newest = latest_checkpoint(folder) if folder.exists() else None
    if newest is not None:
        include.append(newest)
    if (folder / "preflight.json").is_file():
        include.append(folder / "preflight.json")
zs_final = RUN_DIR / "zero_shot/train_run/final"
if zs_final.exists():
    include.append(zs_final)
if (PROJECT_DIR / "data/processed/zero_shot/manifest.json").is_file():
    include.append(PROJECT_DIR / "data/processed/zero_shot/manifest.json")
for stage in ("noise", "final"):
    d = RUN_DIR / f"data_{stage}" / "manifest.json"
    if d.is_file():
        include.append(d)
with tarfile.open(pack, "w:gz") as archive:
    for path in include:
        archive.add(path, arcname=str(Path("lifemate") / path.relative_to(PROJECT_DIR)))
digest = hashlib.sha256(pack.read_bytes()).hexdigest()
(pack.parent / (pack.name + ".sha256")).write_text(f"{digest}  {pack.name}\n")
print("Gói kết quả:", pack, f"({pack.stat().st_size / 1024**2:.0f} MiB)")
print("SHA-256:", digest)
print("TẢI file .tar.gz VÀ .sha256 về trước khi huỷ máy; lần sau điền tên vào RESUME_PACK.")
'''

MD_DONE = """## Hoàn tất

Sau mỗi lần chạy: nộp file `submission-*.jsonl` mới lên trang public (5 lượt/ngày), ghi điểm vào bảng so sánh, rồi quyết định bước tiếp theo.

**Trước 16/11 23:59 (GMT+7):** cần Dockerfile (offline, đọc `/data`, ghi `result/submission.jsonl`, chạy tất định, ≤30 s/mẫu trên A30). Gửi tài liệu hướng dẫn Docker của BTC để viết `Dockerfile` song song với việc train. Chỉ lần nộp Docker cuối cùng được tính.
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-commit", required=True, help="40-char commit SHA that contains the code to sync")
    args = parser.parse_args()
    if len(args.code_commit) != 40:
        raise SystemExit("--code-commit must be a full 40-char SHA")
    nb = json.loads(RESUME.read_text(encoding="utf-8"))
    def join(*parts: str) -> str:
        return "\n\n".join(parts)

    cells = [
        md(MD_TITLE),
        md("## A. Cấu hình — chỉ sửa ô này (chủ yếu dòng `PHASES`)"),
        code(CELL_CONFIG.replace("__CODE_COMMIT__", args.code_commit)),
        md("## B. Chuẩn bị môi trường — không cần sửa, chỉ chạy\nTải code + dữ liệu, nhập checkpoint 1524, khôi phục kết quả lần trước, kiểm tra GPU, cài thư viện, tái tạo dữ liệu train/dev."),
        code(reuse(nb, 3, "def bootstrap_github_release")),
        code(join("# --- tải bundle code+dữ liệu ---\n" + reuse(nb, 4, "bootstrap_github_release("),
                  "# --- nhập checkpoint 1524 ---\n" + reuse(nb, 5, "BASE_CKPT = FULL_DIR"),
                  "# --- đồng bộ code mới từ GitHub ---\n" + CELL_SYNC,
                  "# --- khôi phục kết quả lần trước (nếu có) ---\n" + CELL_RESUME_PACK)),
        code(join("# --- đường dẫn ---\n" + reuse(nb, 9, "def latest_checkpoint"),
                  "# --- kiểm tra GPU / đĩa / dữ liệu ---\n" + reuse(nb, 10, "nvidia-smi").replace(
                      "if free_gib < 18:", 'if free_gib < (18 if PHASES & {"train_noise", "train_final", "zs"} else 9):'),
                  "# --- cài thư viện, tải model, tạo dữ liệu SFT ---\n" + reuse(nb, 12, "prepare_sft.py"),
                  "# --- augmentation thứ tự tool ---\n" + reuse(nb, 13, "AUG_TRAIN_JSONL"),
                  "# --- preflight: độ dài ngữ cảnh + kiểm hash dữ liệu ---\n" + reuse(nb, 15, "preflight"))),
        code(join("# --- ASR và hàm đo dev ---\n" + CELL_ASR_SETUP, CELL_EVAL_FN)),
        md("## C. Các giai đoạn — chạy theo `PHASES`, giai đoạn không được chọn sẽ tự bỏ qua"),
        md("### C1. Chẩn đoán `oracle`: ASR thật so với transcript đúng (~30 phút)"),
        code(CELL_ORACLE),
        md("### C2. So sánh ASR `asr`: medium / large, có hoặc không sửa lỗi (~1–1.5 giờ)"),
        code(join(CELL_ASR_PHASE, CELL_BASELINE)),
        md("### C3. Train: `train_noise` (1 epoch có nhiễu ASR) và `train_final` (train+dev+holdout)"),
        code(join(CELL_TRAIN_FN, CELL_TRAIN_NOISE, CELL_TRAIN_FINAL)),
        md("### C4. Đo dev (`eval`) và tạo submission public (`public`)"),
        code(join(CELL_EVAL, CELL_PUBLIC)),
        md("### C5. Đo độ trễ `latency` (trước khi đóng Docker)"),
        code(CELL_LATENCY),
        md("### C6. Thí nghiệm `zs`: giữ lại app khỏi train để đo khả năng dùng tool chưa thấy (~10 giờ, ưu tiên thấp)"),
        code(CELL_ZS),
        md("## D. Đóng gói kết quả — luôn chạy ở cuối"),
        code(CELL_PACK),
        md(MD_DONE),
    ]
    nb["cells"] = cells
    OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUTPUT, len(cells), "cells")


if __name__ == "__main__":
    main()
