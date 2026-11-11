"""Upload ONLY a README.md (model card) to each repo. Never changes visibility, never deletes files.

  python scripts/upload_model_cards.py --dry-run        # validate, no token needed
  python scripts/upload_model_cards.py                  # asks for a WRITE token via getpass
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path

import yaml

CARDS = {
    "vincent1512/lifemate-phowhisper-large-ft": Path("docs/model_cards/phowhisper_large_ft.md"),
    "vincent1512/lifemate-backup": Path("docs/model_cards/lifemate_backup.md"),
}


def split_front_matter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, flags=re.DOTALL)
    if not match:
        raise ValueError("README has no YAML front matter")
    return yaml.safe_load(match.group(1)) or {}, match.group(2)


def validate(text: str) -> list[str]:
    problems = []
    meta, _ = split_front_matter(text)
    base = meta.get("base_model")
    for item in ([base] if isinstance(base, str) else list(base or [])):
        if item.startswith(("/", ".", "~")) or item.count("/") != 1:
            problems.append(f"base_model is not a Hub id: {item!r}")
    if not meta.get("license"):
        problems.append("license missing")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    failed = False
    for repo, path in CARDS.items():
        text = path.read_text(encoding="utf-8")
        problems = validate(text)
        meta, _ = split_front_matter(text)
        print(f"{repo}: license={meta.get('license')} base_model={meta.get('base_model')} "
              f"{'OK' if not problems else 'PROBLEMS: ' + '; '.join(problems)}")
        failed |= bool(problems)
    if failed:
        return 1
    if args.dry_run:
        print("dry-run: nothing uploaded")
        return 0
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or getpass.getpass("HF WRITE token: "))
    for repo, path in CARDS.items():
        api.create_commit(repo_id=repo, repo_type="model", commit_message="Add model card and license metadata",
                          operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=path.read_bytes())])
        print("uploaded README.md ->", repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
