"""Upload the Qwen LoRA adapter (stage_noise/final) to HuggingFace. Run on your own computer:

    pip install -U huggingface_hub
    python scripts/upload_adapter_hf.py

The WRITE token is asked through getpass (typed into the terminal prompt, never pasted into a chat). Revoke it afterwards.
Only the adapter weights and a small README are uploaded; adapter_config.json is copied with the base model path
rewritten to the public Hub id (the original points to a local Vast folder). Local files are not modified.
"""
import json
import shutil
import tempfile
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi

REPO = "vincent1512/lifemate-qwen-adapter"
PRIVATE = False
BASE = "Qwen/Qwen3-4B-Instruct-2507"
SRC = Path(__file__).resolve().parents[1] / "lifemate/lifemate_runs/vast-20261106-124816/stage_noise/final"

README = f"""---
license: apache-2.0
base_model: {BASE}
library_name: peft
tags:
- lora
- qlora
- tool-calling
- vietnamese
---
# LifeMate Qwen3-4B LoRA adapter (Zalo AI Challenge 2026)

LoRA adapter (r=16, alpha=32) for `{BASE}`, trained for the LifeMate task: a step-by-step agent that outputs one JSON tool call
or a stop signal per turn. Training mixed clean transcripts with ASR-noisy transcripts produced by PhoWhisper-large models.
Used together with `vincent1512/lifemate-phowhisper-large-ft` (speech recognition).

Load with PEFT on top of the base model, or merge into fp16 weights (`scripts/merge_for_submission.py` in the code repository).
"""

def main() -> None:
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (SRC / name).is_file():
            raise SystemExit(f"Thiếu {SRC / name}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copy2(SRC / "adapter_model.safetensors", out / "adapter_model.safetensors")
        config = json.loads((SRC / "adapter_config.json").read_text(encoding="utf-8"))
        config["base_model_name_or_path"] = BASE
        (out / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (out / "README.md").write_text(README, encoding="utf-8")
        api = HfApi(token=getpass("HF WRITE token (nhập vào ô này, KHÔNG dán vào chat): "))
        api.create_repo(REPO, repo_type="model", private=PRIVATE, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=REPO, repo_type="model", commit_message="LoRA adapter (train_noise, 1 epoch)")
    print("Đã upload:", f"https://huggingface.co/{REPO}")
    print("Nhớ thu hồi (revoke) token WRITE sau khi xong.")

if __name__ == "__main__":
    main()
