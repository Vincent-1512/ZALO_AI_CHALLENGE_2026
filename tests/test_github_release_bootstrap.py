import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/github_release_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("github_bootstrap", MODULE_PATH)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}


class GithubBootstrapTests(unittest.TestCase):
    def test_interrupted_download_resumes_without_duplicating_bytes(self):
        data = b"release-bundle-content"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bundle.tar.gz"
            partial = target.with_suffix(".gz.part")
            partial.write_bytes(data[:8])
            response = Response(data[8:], 206, {"Content-Range": f"bytes 8-{len(data)-1}/{len(data)}"})
            with patch.object(bootstrap, "api_request", return_value=response) as request:
                bootstrap.download_asset("owner/repo", {"id": 1, "size": len(data)}, target, hashlib.sha256(data).hexdigest(), "")
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(request.call_args.kwargs["offset"], 8)

    def test_server_ignoring_range_restarts_download(self):
        data = b"complete-content"
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "bundle.part"
            partial.write_bytes(b"old-prefix")
            with patch.object(bootstrap, "api_request", return_value=Response(data)):
                bootstrap.download_once("owner/repo", {"id": 1, "size": len(data)}, partial, "")
            self.assertEqual(partial.read_bytes(), data)

    def test_extraction_rejects_paths_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bundle.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                info = tarfile.TarInfo("../outside.txt")
                info.size = 3
                target.addfile(info, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                bootstrap.extract_bundle(archive, root, "digest")
            self.assertFalse((root / "lifemate").exists())

    def test_existing_project_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "lifemate"
            project.mkdir()
            checkpoint = project / "checkpoint.txt"
            checkpoint.write_text("keep")
            with self.assertRaises(FileExistsError):
                bootstrap.extract_bundle(root / "unused.tar.gz", root, "digest")
            self.assertEqual(checkpoint.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
