"""Build the single-file source + official dataset bundle for Vast.ai."""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ".gitignore",
    "README.md",
    "configs/baseline.json",
    "data/README.md",
    "data/archives/train+public_test.zip",
    "docs/analysis/dataset_audit.md",
    "docs/baseline_design.md",
    "docs/competition/lifemate.md",
    "docs/competition/overview.md",
    "docs/competition/regulations/Zalo_AI_Challenge_2026_Regulations_EN.pdf",
    "docs/competition/regulations/Zalo_AI_Challenge_2026_Regulations_EN.txt",
    "docs/competition/regulations/Zalo_AI_Challenge_2026_The_le_VI.pdf",
    "docs/competition/regulations/Zalo_AI_Challenge_2026_The_le_VI.txt",
    "docs/plans/implementation_plan.md",
    "docs/plans/lifemate_strategy.md",
    "docs/vast_runbook.md",
    "requirements-inference.txt",
    "requirements-train.txt",
    "scripts/download_models.py",
    "scripts/run_inference.py",
    "src/lifemate/__init__.py",
    "src/lifemate/data.py",
    "src/lifemate/models.py",
    "src/lifemate/prompts.py",
    "src/lifemate/runner.py",
    "training/attention_backend.py",
    "training/evaluate_dev.py",
    "training/prepare_sft.py",
    "training/response_loss.py",
    "training/train_lora.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_bytes(paths: tuple[tuple[str, Path], ...]) -> bytes:
    lines = [f"{sha256_file(path)}  {name}\n" for name, path in paths]
    return "".join(lines).encode("utf-8")


def add_file(archive: tarfile.TarFile, archive_name: str, path: Path) -> None:
    info = tarfile.TarInfo(archive_name)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def build() -> Path:
    paths = tuple((name, ROOT / name) for name in FILES)
    missing = [name for name, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing bundle inputs: " + ", ".join(missing))

    manifest = manifest_bytes(paths)
    bundle_id = hashlib.sha256(manifest).hexdigest()[:12]
    output_dir = ROOT / "artifacts/vast_upload"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"lifemate-vast-bundle-{bundle_id}.tar.gz"

    with archive_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for name, path in paths:
                    add_file(archive, f"lifemate/{name}", path)
                info = tarfile.TarInfo("lifemate/SHA256SUMS")
                info.size = len(manifest)
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(manifest))

    checksum = sha256_file(archive_path)
    Path(f"{archive_path}.sha256").write_text(
        f"{checksum}  {archive_path.name}\n", encoding="utf-8"
    )
    print(f"Bundle: {archive_path}")
    print(f"Bytes: {archive_path.stat().st_size}")
    print(f"SHA256: {checksum}")
    return archive_path


if __name__ == "__main__":
    build()
