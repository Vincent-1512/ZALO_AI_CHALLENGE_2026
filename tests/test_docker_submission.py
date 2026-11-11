"""Docker package tests: no GPU, no torch, no model weights (fake infer function)."""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "docker_submission"


def _load(name):
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module        # dataclasses needs the module registered while it executes
    sys.path.insert(0, str(PKG))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before          # do not leak docker_submission/lifemate into other tests
    return module


rt = _load("zac_runtime")


def _runtime(infer_fn, tool_env=None):
    paths = SimpleNamespace(env_root=Path("/data"))
    return rt.Runtime(Path("/data"), "private_test", paths, tool_env, {}, None, None, 8, infer_fn)


def _items(n):
    return [{"id": f"s{i}", "audio": f"audio/s{i}/s{i}.wav"} for i in range(n)]


def test_run_sample_schema_and_time():
    out = rt.run_sample(_items(1)[0], _runtime(lambda *a: ([{"name": "x", "arguments": {}}], "h1")))
    assert set(out) == {"id", "tool_calls", "state_hash", "time"}
    assert out["id"] == "s0" and out["state_hash"] == "h1"
    assert isinstance(out["time"], int) and out["time"] >= 0


def test_failure_does_not_stop_run_and_uses_fallback_hash():
    class Env:
        def __init__(self, split, sid, root=None): pass
        def get_state_hash(self): return "fallback"

    def boom(*a):
        raise RuntimeError("model crashed")

    out = rt.run_sample(_items(1)[0], _runtime(boom, tool_env=Env))
    assert out["tool_calls"] == [] and out["state_hash"] == "fallback"


def test_failure_with_no_env_gives_null_hash():
    def boom(*a):
        raise RuntimeError("x")

    def bad_env(*a, **k):
        raise RuntimeError("env")

    out = rt.run_sample(_items(1)[0], _runtime(boom, tool_env=bad_env))
    assert out["state_hash"] is None and out["tool_calls"] == []


def test_predict_all_writes_one_line_per_sample(tmp_path):
    calls = []

    def infer(item, *a):
        calls.append(item["id"])
        return [], f"h-{item['id']}"

    path = tmp_path / "result" / "submission.jsonl"
    results = rt.predict_all(_runtime(infer), _items(3), path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["s0", "s1", "s2"] == calls
    assert len(results) == 3 and all(isinstance(r["time"], int) for r in rows)


def test_split_detection_and_limit(tmp_path):
    split = tmp_path / "private_test"
    split.mkdir()
    (split / "private_test_tools_catalog.json").write_text("{}")
    (split / "private_test.jsonl").write_text("\n".join(json.dumps(i) for i in _items(5)) + "\n")
    (tmp_path / "other").mkdir()
    assert rt.detect_split(tmp_path) == "private_test"
    assert len(rt.read_items(tmp_path, "private_test")) == 5
    assert len(rt.read_items(tmp_path, "private_test", limit=2)) == 2
    with pytest.raises(FileNotFoundError):
        rt.detect_split(tmp_path / "other")


def test_preprocessing_never_raises_on_empty_data(tmp_path, monkeypatch, capsys):
    pre = _load("preprocessing")
    monkeypatch.setattr(pre.rt, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pre.rt, "RESULT_DIR", tmp_path / "result")
    monkeypatch.setattr(pre.rt, "MODELS_DIR", tmp_path / "saved_models")
    assert pre.main() == 0
    assert (tmp_path / "result").is_dir()


def test_no_network_or_external_paths_in_submission_code():
    forbidden = re.compile(r"requests\.|urllib|snapshot_download|hf_hub_download|from_pretrained\(\s*[\"'][\w.-]+/[\w.-]+[\"']|openai|anthropic")
    for path in list(PKG.glob("*.py")) + list((PKG / "lifemate").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert not forbidden.search(text), f"{path.name} looks like it uses the network"
        assert "/workspace" not in text and "src/lifemate" not in text, path.name


def test_notebook_has_required_cells_and_output_name():
    nb = json.loads((PKG / "predict_notebook.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 3
    joined = "\n".join(code)
    assert "seed_everything" in joined and "/result/jupyter_submission.jsonl" in joined


def test_shell_scripts_match_guide():
    assert "predict.py" in (PKG / "predict.sh").read_text() and "preprocessing.py" in (PKG / "predict.sh").read_text()
    jup = (PKG / "start_jupyter.sh").read_text()
    assert "--port 9777" in jup and "zac2026" in jup


def test_model_card_validation():
    spec = importlib.util.spec_from_file_location("upload_cards", ROOT / "scripts" / "upload_model_cards.py")
    cards = importlib.util.module_from_spec(spec)
    sys.modules["upload_cards"] = cards
    spec.loader.exec_module(cards)
    assert cards.validate("---\nlicense: mit\nbase_model: /workspace/models/Qwen3\n---\nbody\n")
    assert not cards.validate("---\nlicense: mit\nbase_model: Qwen/Qwen3-4B-Instruct-2507\n---\nbody\n")
    for path in cards.CARDS.values():
        assert not cards.validate((ROOT / path).read_text(encoding="utf-8"))
