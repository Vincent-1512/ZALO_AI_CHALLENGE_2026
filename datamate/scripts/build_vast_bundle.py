from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
INCLUDED_DIRS = ("configs", "docs", "notebooks", "scripts", "src", "training")
INCLUDED_FILES = (".gitignore", "README.md", "data/README.md", "requirements-train.txt", "requirements-inference.txt")
ARCHIVE_FILES = ("data/archives/train.zip", "data/archives/public_test.zip")
EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, name: str, path: Path) -> None:
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with path.open("rb") as stream:
        archive.addfile(info, stream)


def build() -> Path:
    paths: list[tuple[str, Path]] = [(name, ROOT / name) for name in INCLUDED_FILES + ARCHIVE_FILES]
    for directory in INCLUDED_DIRS:
        paths.extend((path.relative_to(ROOT).as_posix(), path)
                     for path in sorted((ROOT / directory).rglob("*"))
                     if path.is_file() and not (EXCLUDED_PARTS & set(path.parts)))
    missing = [name for name, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Vast bundle inputs: " + ", ".join(missing))
    manifest = "".join(f"{sha256_file(path)}  {name}\n" for name, path in paths).encode("utf-8")
    bundle_id = hashlib.sha256(manifest).hexdigest()[:12]
    output_dir = ROOT / "outputs/handoff"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"datamate-vast-bundle-{bundle_id}.tar.gz"
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name, path in paths:
                    add_file(archive, f"datamate/{name}", path)
                add_file_from_bytes = tarfile.TarInfo("datamate/SHA256SUMS")
                add_file_from_bytes.size = len(manifest)
                add_file_from_bytes.mode = 0o644
                add_file_from_bytes.uid = add_file_from_bytes.gid = 0
                add_file_from_bytes.uname = add_file_from_bytes.gname = ""
                add_file_from_bytes.mtime = 0
                archive.addfile(add_file_from_bytes, io.BytesIO(manifest))
    checksum = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="utf-8"
    )
    print(f"Bundle: {output}")
    print(f"Bytes: {output.stat().st_size}")
    print(f"SHA256: {checksum}")
    return output


if __name__ == "__main__":
    build()
