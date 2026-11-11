from __future__ import annotations

import hashlib
import json
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any


def _torch_dtype(torch, use_gpu: bool):
    return torch.float16 if use_gpu else torch.float32


def _asr_model_fingerprint(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path)
    config_names = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    configs = {}
    for name in config_names:
        path = root / name
        if path.is_file():
            configs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    weight_files = sorted(
        path for path in root.iterdir()
        if path.is_file() and (path.suffix == ".safetensors" or path.suffix == ".bin")
    )
    return {
        "configs": configs,
        "weights": [(path.name, path.stat().st_size) for path in weight_files],
    }


def asr_cache_key(paths: list[str | Path], max_new_tokens: int = 128) -> str:
    """Key the memoized batch by audio bytes, not filenames or timestamps."""
    audio_hashes = []
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        audio_hashes.append(digest.hexdigest())
    payload = json.dumps(
        {"max_new_tokens": max_new_tokens, "audio_sha256": audio_hashes}, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SpeechRecognizer:
    def __init__(self, model_path: str | Path, batch_size: int = 4, sample_rate: int = 16000):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        dtype = _torch_dtype(torch, self.device == "cuda")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            path, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
        ).to(self.device).eval()
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        # Optional memoisation of whole transcribe_many() calls. Off unless LIFEMATE_ASR_CACHE is set,
        # so default scored inference does not hash files or alter model calls.
        self._cache_path = os.environ.get("LIFEMATE_ASR_CACHE") or None
        self._cache_meta = None
        self._cache: dict[str, list[str]] = {}
        if self._cache_path:
            self._cache_meta = {
                "model": Path(path).name,
                "model_fingerprint": _asr_model_fingerprint(path),
                "dtype": str(dtype), "sample_rate": sample_rate,
                "batch_size": batch_size, "language": "vi",
            }
            if Path(self._cache_path).is_file():
                stored = json.loads(Path(self._cache_path).read_text(encoding="utf-8"))
                if stored.get("meta") == self._cache_meta:
                    self._cache = stored.get("items", {})
                else:
                    print("ASR cache metadata differs from this run; ignoring the cache file", flush=True)

    def _cache_key(self, paths: list[Path], max_new_tokens: int) -> str:
        return asr_cache_key(paths, max_new_tokens)

    def _save_cache(self) -> None:
        target = Path(self._cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps({"meta": self._cache_meta, "items": self._cache}, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)

    def _read_audio(self, path: str | Path):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != self.sample_rate:
            ratio = Fraction(self.sample_rate, rate)
            audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid audio data: {path}")
        return audio

    def transcribe_many(self, paths: list[Path], max_new_tokens: int = 128) -> list[str]:
        if not paths:
            return []
        if self._cache_path:
            key = self._cache_key(paths, max_new_tokens)
            if key in self._cache:
                return list(self._cache[key])
            texts = self._transcribe_uncached(paths, max_new_tokens)
            self._cache[key] = texts
            self._save_cache()
            return list(texts)
        return self._transcribe_uncached(paths, max_new_tokens)

    def _transcribe_uncached(self, paths: list[Path], max_new_tokens: int) -> list[str]:
        audios = [self._read_audio(path) for path in paths]
        results: list[str] = []
        for start in range(0, len(audios), self.batch_size):
            batch = audios[start : start + self.batch_size]
            inputs = self.processor(
                batch, sampling_rate=self.sample_rate, padding="longest",
                return_attention_mask=True, return_tensors="pt",
            )
            features = inputs.input_features.to(self.device, dtype=self.model.dtype)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                raise RuntimeError("Whisper processor did not return an attention mask")
            attention_mask = attention_mask.to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    features,
                    attention_mask=attention_mask,
                    language="vi",
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                )
            results.extend(self.processor.batch_decode(generated, skip_special_tokens=True))
        return [text.strip() for text in results]


def parse_action(output: str) -> dict[str, Any] | None:
    text = output.strip()
    if "<tool_call>" in text:
        text = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    if value.get("action") == "stop":
        return {"action": "stop"}
    if value.get("action") == "call":
        name, arguments = value.get("name"), value.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return {"action": "call", "name": name, "arguments": arguments}
    if isinstance(value.get("name"), str) and isinstance(value.get("arguments"), dict):
        return {"action": "call", "name": value["name"], "arguments": value["arguments"]}
    return None


def _load_causal_model(path: str, device: str, dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="auto" if device == "cuda" else None,
    )


class ToolAgent:
    def __init__(
        self,
        model_path: str | Path,
        adapter_path: str | Path | None,
        max_context_tokens: int,
        max_new_tokens: int,
    ):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = _torch_dtype(torch, self.device == "cuda")
        self.model = _load_causal_model(path, self.device, dtype)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path), is_trainable=False)
        self.model.eval()
        self.max_input_tokens = max_context_tokens - max_new_tokens
        self.max_new_tokens = max_new_tokens

    def predict(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        from .prompts import fit_messages

        input_ids = fit_messages(messages, self.tokenizer, self.max_input_tokens)
        input_ids = input_ids.to(self.model.device)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return parse_action(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def predict_candidates(
        self, messages: list[dict[str, str]], num_samples: int, temperature: float | None = None
    ) -> list[dict[str, Any] | None]:
        """Sample multiple JSON actions; callers must schema-validate before execution."""
        if num_samples < 2:
            return [self.predict(messages)]
        if temperature is None or temperature <= 0:
            raise ValueError("temperature must be positive when num_samples > 1")
        from .prompts import fit_messages

        input_ids = fit_messages(messages, self.tokenizer, self.max_input_tokens).to(self.model.device)
        attention_mask = self.torch.ones_like(input_ids)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_samples,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        return [
            parse_action(self.tokenizer.decode(row[input_ids.shape[-1] :], skip_special_tokens=True))
            for row in output_ids
        ]
