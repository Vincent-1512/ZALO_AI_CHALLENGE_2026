"""Download a pinned GitHub release bundle for the Vast notebook."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_request(url: str, token: str, binary: bool = False, offset: int = 0):
    headers = {
        "Accept": "application/octet-stream" if binary else "application/vnd.github+json",
        "User-Agent": "lifemate-vast-bootstrap",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    # urllib does not forward unredirected headers to the asset CDN.
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=90)


def get_release_asset(repo: str, tag: str, filename: str, token: str) -> dict:
    base = f"https://api.github.com/repos/{repo}"
    with api_request(f"{base}/releases/tags/{tag}", token) as response:
        release = json.load(response)
    assets = [asset for asset in release.get("assets", []) if asset["name"] == filename]
    if len(assets) != 1 or assets[0].get("state") != "uploaded":
        raise FileNotFoundError(f"Release {tag} chưa có asset hoàn chỉnh: {filename}")
    return assets[0]


def download_once(repo: str, asset: dict, partial: Path, token: str) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    url = f"https://api.github.com/repos/{repo}/releases/assets/{int(asset['id'])}"
    with api_request(url, token, binary=True, offset=offset) as response:
        append = response.status == 206 and offset > 0
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Server trả sai vị trí download tiếp tục.")
        downloaded = offset if append else 0
        last_print = downloaded
        with partial.open("ab" if append else "wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_print >= 64 * 1024**2:
                    print(f"Downloaded {downloaded / 1024**2:.0f}/{asset['size'] / 1024**2:.0f} MiB", flush=True)
                    last_print = downloaded


def download_asset(repo: str, asset: dict, target: Path, digest: str, token: str) -> None:
    if target.is_file() and file_sha256(target) == digest:
        return
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            if not partial.exists() or partial.stat().st_size != asset["size"]:
                download_once(repo, asset, partial, token)
            if partial.stat().st_size != asset["size"] or file_sha256(partial) != digest:
                partial.unlink(missing_ok=True)
                raise RuntimeError("Download thiếu byte hoặc sai SHA-256; sẽ tải lại.")
            partial.replace(target)
            return
        except (OSError, RuntimeError) as error:
            if attempt == 2:
                raise RuntimeError("Download chưa hoàn tất. Chạy lại cell để tiếp tục file .part.") from error
            print(f"Download bị gián đoạn ({type(error).__name__}); thử lại {attempt + 2}/3.", flush=True)
            time.sleep(2)


def extract_bundle(archive: Path, workspace: Path, digest: str) -> Path:
    project = workspace / "lifemate"
    marker = project / ".github_bundle_sha256"
    if marker.is_file() and marker.read_text().strip() == digest:
        return project
    if project.exists():
        raise FileExistsError("Đã có thư mục lifemate khác. Chọn WORKSPACE_DIR mới để giữ checkpoint hiện có.")
    with tempfile.TemporaryDirectory(prefix=".lifemate-extract-", dir=workspace) as temporary:
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                parts = Path(member.name).parts
                if not parts or parts[0] != "lifemate" or not (member.isfile() or member.isdir()):
                    raise ValueError("Bundle chứa đường dẫn hoặc loại file không được phép.")
                source.extract(member, temporary, filter="data")
        staged = Path(temporary) / "lifemate"
        if not (staged / "training/train_lora.py").is_file():
            raise FileNotFoundError("Bundle không có training/train_lora.py.")
        staged.rename(project)
    marker.write_text(digest + "\n")
    return project


def bootstrap_github_release(repo: str, tag: str, filename: str, digest: str, workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    project = workspace / "lifemate"
    marker = project / ".github_bundle_sha256"
    if marker.is_file() and marker.read_text().strip() == digest:
        print("Bundle đã có trên máy; dùng project hiện tại:", project)
        return project
    if project.exists():
        raise FileExistsError("Thư mục lifemate đã tồn tại; sửa WORKSPACE_DIR thành một thư mục mới.")
    free_gib = shutil.disk_usage(workspace).free / 1024**3
    if free_gib < 22:
        raise RuntimeError(f"Chỉ còn {free_gib:.1f} GiB. Cần ít nhất 22 GiB trước khi tải và giải nén bundle.")
    token = os.environ.get("GH_TOKEN", "")
    try:
        asset = get_release_asset(repo, tag, filename, token.strip())
        archive = workspace / filename
        download_asset(repo, asset, archive, digest, token.strip())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GitHub trả HTTP {error.code}; kiểm tra Release đã publish hoặc API đã hết lượt tải.") from None
    finally:
        token = None
    project = extract_bundle(archive, workspace, digest)
    archive.unlink()
    print("Bundle đã xác minh SHA-256 và giải nén:", project)
    return project
