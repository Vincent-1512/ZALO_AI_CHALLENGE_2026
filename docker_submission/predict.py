"""Entry point: reads /data/<split>/<split>.jsonl, writes /result/submission.jsonl (with `time` in ms)."""
import argparse
import sys
import time
from pathlib import Path

import zac_runtime as rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(rt.DATA_ROOT))
    parser.add_argument("--result-dir", default=str(rt.RESULT_DIR))
    parser.add_argument("--models-dir", default=str(rt.MODELS_DIR))
    parser.add_argument("--limit", type=int, default=0, help="debug only: first N samples")
    args = parser.parse_args()

    t_start = time.perf_counter()
    rt.seed_everything(rt.SEED)
    runtime = rt.load_models(Path(args.data_root), Path(args.models_dir))
    load_seconds = time.perf_counter() - t_start

    items = rt.read_items(Path(args.data_root), runtime.split, args.limit)
    t_pred = time.perf_counter()
    results = rt.predict_all(runtime, items, Path(args.result_dir) / "submission.jsonl")
    predict_seconds = time.perf_counter() - t_pred

    times = sorted(r["time"] for r in results) or [0]
    print(f"LOAD_SECONDS={load_seconds:.1f}", flush=True)
    print(f"PREDICT_SECONDS={predict_seconds:.1f} samples={len(results)} "
          f"ms_mean={sum(times) / len(times):.0f} ms_max={times[-1]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
