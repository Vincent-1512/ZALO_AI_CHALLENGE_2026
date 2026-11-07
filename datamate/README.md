# DataMate workspace

This is a separate DataMate workspace inside the LifeMate repository. Keep DataMate data, model caches, runs and submissions under this directory; do not copy them into the LifeMate paths.

## Current pipeline

The Vast notebook is [notebooks/datamate_vast_train.ipynb](notebooks/datamate_vast_train.ipynb). It coordinates:

1. dataset integrity checks, file inventory and a fixed holdout by whole business;
2. prompt-only dev evaluation for a baseline score;
3. local Qwen generation of constrained JSON query plans on **train only**;
4. safe execution against each shop's CSV/XLSX/SQLite and retention only when the computed answer exactly matches the train gold;
5. export a 100-plan audit sample; the default `run_all` mode continues automatically, while manual review remains available; then deterministic train-only paraphrase augmentation;
6. QLoRA smoke/full training with per-epoch dev utility early stopping;
7. dev evaluation and public JSONL generation;
8. an optional offline model bundle backup.

The model never receives the gold answer while generating a plan. Gold is used only after execution to verify whether the plan is eligible for training. Failed, malformed and wrong plans are logged and excluded from SFT.

Offline inference entrypoint: [scripts/run_inference.py](scripts/run_inference.py). It accepts a local base-model directory and adapter and sets Hugging Face offline flags before loading either model.

## Important status and limitation

This is an **experimental baseline foundation**, not a finished competition solver. Its generic query-plan executor supports allowlisted table selection, joins, filters, group-by metrics, simple arithmetic and output formatting. It does not yet implement every detailed DataMate rule (for example, full order/refund state semantics, debt aging, inventory lots/unit conversions, industry-specific logic and all relative-period edge cases). A high verified-plan count does not prove broad rule coverage; inspect `plan_generation_report.json`, review representative verified plans, and measure full grouped-dev utility before selecting an adapter. Do not treat a smoke run or successful notebook execution as evidence of competition-ready accuracy.

The split uses six complete train businesses for dev. This avoids question-level leakage, but the holdout is small and cannot guarantee generalization to unseen shops. The DataMate rulebook and Docker guide on the official portal remain authoritative; never infer DataMate requirements from LifeMate documentation.

## Dataset setup

Downloaded competition archives stay local and are ignored by Git:

```text
data/archives/train.zip
data/archives/public_test.zip
data/raw/train/
data/raw/public_test/
```

The notebook first reuses extracted data already on the Vast volume, then searches the project and common `/workspace` locations for BTC ZIP archives and extracts them when required. On a fresh volume, put `train.zip` and `public_test.zip` in `data/archives/` once; the notebook handles extraction and the rest of setup. It never downloads competition data from GitHub or labels/augments public/private questions. Keep raw data and generated run artifacts out of commits and public links.

## Run one notebook on a fresh Vast instance

Download `datamate/notebooks/datamate_vast_train.ipynb` from GitHub, upload just that notebook to Vast, open it, and click **Run All**. Its first code cell downloads the pinned `datamate-fast-v1-20261107` GitHub Release bundle, verifies its SHA256, and extracts source plus train/public archives. Later cells install the Python packages and download the base model if it is not cached.

The standalone notebook still needs internet access for the first bundle/model download. The public release contains the competition archives under the organizer's permission as reported by the team; do not reuse or redistribute it beyond that permission. If an old Vast volume already has the dataset or model cache, the notebook searches for and reuses it.

## Build a Vast upload bundle

From the repository root on the local machine:

```bash
python3 datamate/scripts/build_vast_bundle.py
```

The script writes a compressed bundle and `.sha256` file under `datamate/outputs/handoff/`. It includes the notebook, source, docs, pinned training dependencies and the two dataset archives; it excludes extracted copies, checkpoints and outputs. Upload the `.tar.gz` to a persistent Vast volume, verify its checksum, then extract it so the folder is `/workspace/datamate_job/datamate/`. Open `datamate/notebooks/datamate_vast_train.ipynb` from that project folder.

The local handoff folder keeps only the newest generated bundle. Older bundles are reproducible with the build command above and are removed during cleanup.

Do not let the dependency cell replace the CUDA PyTorch build. If the Vast image has no CUDA-enabled PyTorch, select a PyTorch image or install the matching CUDA wheel using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then restart the notebook kernel.

## Notebook actions

The default `ACTION="run_all"` runs the pipeline through public JSONL in one pass with a fast configuration (16 dev examples, 16 plan-pilot examples, up to 640 labeled train examples and 2 epochs). Set `PLAN_LABEL_LIMIT=0` to use the full train split after the fast run is stable. It writes a 100-plan audit sample and continues without waiting for manual review; inspect that audit after the run. Keep Vast running until the final submission path is printed. For separate or recovery runs, edit the first configuration cell and choose one action:

| Action | Purpose |
|---|---|
| `run_all` | Full pipeline from prepare through public JSONL; default, automatically continues after exporting audit |
| `prepare` | split by business and write file inventory; table loading happens when a business is used |
| `baseline_evaluate` | prompt-only grouped-dev utility before SFT |
| `build_plans` | generate and verify train query plans; `PLAN_LABEL_LIMIT=0` means all train rows |
| `review_plans` | export 100 verified plans for manual correctness review |
| `augment` | create limited deterministic paraphrases from verified train plans |
| `smoke` | short QLoRA + small dev check |
| `train` | full run, early stopping on grouped-dev utility |
| `resume` | resume the same `RUN_ID` from its latest checkpoint |
| `evaluate` | evaluate the selected `best-adapter` on grouped dev |
| `public_predict` | write `result/<RUN_ID>/submission.jsonl` |
| `backup` | package source, base model and selected adapter for offline handoff |

Standalone `train` is gated until `AUDIT_CONFIRMED=True` is set after reviewing the audit file; it also requires at least 128 verified train plans. `run_all` honors the one-pass workflow by exporting the audit and continuing automatically, but still stops if fewer than 128 plans verify. Smoke runs do not bypass the standalone full-run audit gate.

Typical run directory: `outputs/runs/<RUN_ID>/`. Base-model downloads are cached under `models/hf_cache/` on the Vast volume. Preserve that volume until the offline bundle and checksum have been copied out.

## Files

```text
configs/baseline.json
src/datamate/                 readers, split, plan contract, executor, scoring, inference
training/train_lora.py        masked-response QLoRA and utility-based early stop
notebooks/datamate_vast_train.ipynb
docs/dataset_analysis.md
docs/vast_training_plan.md
```
