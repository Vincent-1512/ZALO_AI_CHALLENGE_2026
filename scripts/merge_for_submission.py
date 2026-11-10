"""Build docker_submission/saved_models/{asr,llm} for the final image. Needs a GPU-less machine with ~20 GB RAM.

  python scripts/merge_for_submission.py \
      --llm-base models/Qwen3-4B-Instruct-2507 --adapter <checkpoint dir> \
      --asr-ft models/PhoWhisper-large-ft --output docker_submission/saved_models

The LoRA is merged into a full-precision (fp16) copy of the base model, NOT into the 4-bit one used for
QLoRA training. fp16 matches the dtype used at inference, so the merged model should reproduce the
unmerged scores; verify on dev (BUILD.md step 3) before building the image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


def safetensors_param_count(folder: Path) -> int:
    total = 0
    for path in sorted(folder.glob("*.safetensors")):
        with path.open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in meta["shape"]:
                count *= dim
            total += count
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--asr-ft", required=True)
    parser.add_argument("--output", default="docker_submission/saved_models")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output)
    llm_out, asr_out = out / "llm", out / "asr"
    llm_out.mkdir(parents=True, exist_ok=True)
    asr_out.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    # GPU: gộp thẳng bằng fp16. CPU (vd. Mac): gộp bằng fp32 rồi ép về fp16 (gộp fp16 trên CPU rất chậm).
    model = AutoModelForCausalLM.from_pretrained(args.llm_base, torch_dtype=torch.float16 if use_cuda else torch.float32,
                                                 low_cpu_mem_usage=True, local_files_only=True,
                                                 device_map="cuda" if use_cuda else None)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).merge_and_unload()
    if not use_cuda:
        model = model.to(torch.float16)
    model.save_pretrained(llm_out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.llm_base, local_files_only=True).save_pretrained(llm_out)

    for path in Path(args.asr_ft).iterdir():
        if path.is_file():
            shutil.copy2(path, asr_out / path.name)

    llm_params, asr_params = safetensors_param_count(llm_out), safetensors_param_count(asr_out)
    manifest = {
        "llm_parameters": llm_params, "asr_parameters": asr_params,
        "total_parameters": llm_params + asr_params,
        "limits": {"total_lt": 6_000_000_000, "llm_lt": 5_000_000_000},
        "total_ok": llm_params + asr_params < 6_000_000_000 and llm_params < 5_000_000_000,
        "files": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*.safetensors"))},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    if not manifest["total_ok"]:
        raise SystemExit("Parameter budget exceeded")


if __name__ == "__main__":
    main()
