from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned, public model revisions for offline use.")
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("asr", "llm"):
        model = config["models"][key]
        destination = Path(model["local_path"])
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            local_dir=str(destination),
        )
        print(f"Downloaded {key}: {model['repo_id']}@{model['revision']} -> {destination}")


if __name__ == "__main__":
    main()
