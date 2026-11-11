#!/usr/bin/env python3
"""Build notebooks/lifemate_vast_asr_more.ipynb: ONLY the extra PhoWhisper fine-tune (3 epochs from large_ft).

Reuses the proven setup cells and the `asr_more` cell of notebooks/lifemate_vast_final.ipynb, so run
scripts/build_final_notebook.py first if that notebook changes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "notebooks/lifemate_vast_final.ipynb"
OUTPUT = ROOT / "notebooks/lifemate_vast_asr_more.ipynb"

MD_TITLE = """# LifeMate — chỉ train thêm PhoWhisper (3 epoch từ bản fine-tune hiện tại)

**Việc làm:** tiếp tục fine-tune `vincent1512/lifemate-phowhisper-large-ft` thêm 3 epoch (~1 giờ trên RTX 3090), in WER dev trước/sau, rồi tự quyết dùng bản mới hay giữ bản cũ (chỉ dùng bản mới nếu WER dev giảm ≥ `ASR_MORE_MIN_GAIN`). **Không train Qwen**, không tạo submission.

**Cách dùng:** chạy từng cell từ trên xuống (Shift+Enter). Cell cuối đóng gói kết quả, và tuỳ chọn upload lên HuggingFace. Tải gói về trước khi huỷ máy.

**Cần nhớ**
- Cần đĩa từ 80 GB. Các cell chuẩn bị (mục B) vẫn tải cả Qwen, checkpoint 1524 và dựng dữ liệu SFT vì dùng chung với notebook chính; mất thêm vài phút nhưng chạy chắc.
- WER dev đo trên 478 audio dev giữ lại; chênh lệch dưới ~0.005 là nhiễu, nên ngưỡng mặc định là 0.005. Lr 5e-5 và ngưỡng đó là giả thuyết, chưa kiểm chứng.
- Chạy xong, để train Qwen tiếp trên cùng máy: mở notebook chính `lifemate_vast_final.ipynb`, hoặc đặt `ASR_CHOICE`/`ASR_FT2_REPO` đúng như quyết định in ra ở đây.
"""

CELL_FINISH = r'''import tarfile, hashlib, time
from huggingface_hub import HfApi

UPLOAD_TO_HF = False                          # True = upload bản ASR mới lên HuggingFace (hỏi token WRITE bằng ô nhập; ĐỪNG dán token vào chat)
HF_REPO_FT2 = "vincent1512/lifemate-phowhisper-large-ft2"
HF_PRIVATE = False                            # False = public: máy sau tải không cần token

decision = json.loads(ASR_MORE_DECISION.read_text(encoding="utf-8"))
print("QUYẾT ĐỊNH:", json.dumps(decision, indent=2))
print("=> dùng:", decision["choice"], "(large_ft2 = bản vừa train thêm; large_ft = giữ bản cũ)")

if UPLOAD_TO_HF:
    from getpass import getpass
    api = HfApi(token=getpass("HF WRITE token (nhập vào ô này): "))
    api.create_repo(HF_REPO_FT2, repo_type="model", private=HF_PRIVATE, exist_ok=True)
    api.upload_folder(folder_path=str(FT2), repo_id=HF_REPO_FT2, repo_type="model",
                      ignore_patterns=["README.md", "*.tmp"], commit_message="PhoWhisper-large fine-tune +3 epochs")
    print("đã upload:", HF_REPO_FT2)

stamp = time.strftime("%Y%m%d-%H%M%S")
pack = WORKSPACE_DIR / f"lifemate-asr-more-{stamp}.tar.gz"
include = [FT2, ASR_MORE_REPORT, ASR_MORE_DECISION] + [p for p in RUN_DIR.glob("asr_cache*.json") if p.is_file()]
with tarfile.open(pack, "w:gz") as archive:
    for path in include:
        archive.add(path, arcname=str(Path("lifemate") / path.relative_to(PROJECT_DIR)))
digest = hashlib.sha256(pack.read_bytes()).hexdigest()
(pack.parent / (pack.name + ".sha256")).write_text(f"{digest}  {pack.name}\n")
print("Gói kết quả:", pack, f"({pack.stat().st_size / 1024**2:.0f} MiB)")
print("SHA-256:", digest)
print("TẢI file .tar.gz VÀ .sha256 về trước khi huỷ máy.")
'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def text_of(cell: dict) -> str:
    return "".join(cell["source"])


def find(cells: list[dict], marker: str, kind: str = "code") -> dict:
    hits = [c for c in cells if c["cell_type"] == kind and marker in text_of(c)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {kind} cell containing {marker!r}, found {len(hits)}")
    return hits[0]


def main() -> None:
    nb = json.loads(FINAL.read_text(encoding="utf-8"))
    cells = nb["cells"]
    config = text_of(find(cells, "CODE_COMMIT = ", "code"))
    marker = 'PHASES = {"asr_more", "train_noise", "eval", "public"}'
    if marker not in config:
        raise SystemExit("config cell changed: PHASES line not found")
    config = config.replace(marker, 'PHASES = {"asr_more"}')
    env_markers = ["def bootstrap_github_release", "# --- tải bundle code+dữ liệu ---", "# --- đường dẫn ---", "# --- ASR và hàm đo dev ---"]
    out = [
        md(MD_TITLE),
        md("## A. Cấu hình (đã đặt sẵn `PHASES = {\"asr_more\"}`; chỉ cần chạy)"),
        code(config),
        md("## B. Chuẩn bị môi trường — chỉ chạy, không cần sửa"),
        *[find(cells, m) for m in env_markers],
        md("## C. Train thêm ASR (`asr_more`) và tự quyết định"),
        find(cells, "ASR_MORE_REPORT = RUN_DIR", "code"),
        md("## D. Đóng gói kết quả (và tuỳ chọn upload lên HuggingFace)"),
        code(CELL_FINISH),
    ]
    out = [dict(c, outputs=[], execution_count=None) if c["cell_type"] == "code" else c for c in out]
    nb["cells"] = out
    OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUTPUT, len(out), "cells")


if __name__ == "__main__":
    main()
