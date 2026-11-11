"""Upload the TRAIN split (not public_test) and the derived SFT files to a HuggingFace dataset repo.

Run on your own computer, from the repository root:

    pip install "huggingface_hub>=0.34,<1.0"
    python scripts/upload_train_data_hf.py

The WRITE token is asked through getpass (typed into the terminal prompt, never pasted into a chat). Revoke it afterwards.
Uploaded: data/raw/train+public_test/train -> train/   and   data/processed/sft -> processed_sft/
NOT uploaded: public_test, the zip archive, models, anything under runs/ or lifemate_runs/.
Local files are not modified. Needs about 0.9 GB of upload.
"""
import tempfile
from getpass import getpass
from pathlib import Path

from huggingface_hub import HfApi

REPO = "vincent1512/lifemate-train-data"
PRIVATE = False
ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/train+public_test/train"
SFT = ROOT / "data/processed/sft"
IGNORE = ["__pycache__/**", "**/__pycache__/**", ".DS_Store", "**/.DS_Store"]

CARD = """---
language:
- vi
task_categories:
- automatic-speech-recognition
- text-generation
tags:
- zalo-ai-challenge
- lifemate
- tool-calling
---
# LifeMate training data (Zalo AI Challenge 2026)

Training material used by our LifeMate solution, uploaded so that the organizers can reproduce training
(Docker Guide, section "Training Code & Tai lieu mo ta").

- `train/`: the official **train split** of the competition package (release v3.2.0): `train.jsonl`, audio,
  `train_tools_catalog.json`, `train_environment/`, `zaic_tools/`. The competition data belongs to the organizers.
  `public_test` and `private_test` are NOT included.
- `processed_sft/`: files derived from `train/` by `training/prepare_sft.py` (seed 2026; 80/10/10 train/dev/holdout split).

Models: `vincent1512/lifemate-phowhisper-large-ft` (ASR), `vincent1512/lifemate-qwen-adapter` (LLM adapter).
"""


def main() -> None:
    for path in (TRAIN / "train.jsonl", SFT / "train.jsonl"):
        if not path.is_file():
            raise SystemExit(f"Thieu {path}")
    api = HfApi(token=getpass("HF WRITE token (nhap vao o nay, KHONG dan vao chat): "))
    api.create_repo(REPO, repo_type="dataset", private=PRIVATE, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(CARD, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                        commit_message="Dataset card")
    api.upload_folder(folder_path=str(TRAIN), path_in_repo="train", repo_id=REPO, repo_type="dataset",
                      ignore_patterns=IGNORE, commit_message="Train split")
    api.upload_folder(folder_path=str(SFT), path_in_repo="processed_sft", repo_id=REPO, repo_type="dataset",
                      ignore_patterns=IGNORE, commit_message="Derived SFT files")
    print("Da upload:", f"https://huggingface.co/datasets/{REPO}")
    print("Nho thu hoi (revoke) token WRITE sau khi xong.")


if __name__ == "__main__":
    main()
