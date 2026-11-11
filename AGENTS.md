# AGENTS.md — handoff for coding agents (Codex etc.)

Project: Zalo AI Challenge 2026, track **LifeMate** (voice -> agentic tool calls). Owner is a Vietnamese speaker and a **beginner with Docker/Vast**: reply in **Vietnamese**, explain with short everyday analogies, give copy-paste commands, never assume prior knowledge. Cost-sensitive (Vast balance is small). Written 2026-11-10, updated 2026-11-11 (GMT+7).

## 0. Hard rules (do not break)
- Never delete files without explicit user approval. Never push to `main`; work on a branch (`docker-submission` exists on GitHub `Vincent-1512/ZALO_AI_CHALLENGE_2026`). Do not commit/push unless asked.
- Never ask the user to paste secrets into chat. HuggingFace WRITE tokens only via `getpass` in the user's own terminal, then revoke. Never print or repeat a token.
- Competition rules (portal tab Rule): total params < 6B (LLM < 5B, context <= 16,384); no reading `<split>_environment/` or `state.json` (only `zaic_tools` calls: validate, call, get_tool_calls, get_state_hash); no labelling/tuning on public_test/private_test; offline inference; 30 s per sample (timeout = 0 score); deterministic decoding + fixed seeds; open models released before 2026-10-26 with compatible licences.
- Do not change HF repos or Drive file after the Docker deadline. Do not modify the solution after freezing and then re-run private_test: the organisers re-run the submitted Docker to verify results (risk of disqualification).
- Be honest about what is unverified. Prefer measuring over guessing.

## 1. Deadlines and submission mechanics
- Public round until 2026-11-16: result JSONL, 5 submissions/day (Submission tab).
- **2026-11-16 23:59 GMT+7**: Docker package via the portal form "Solution Submission" (Google Drive FILE link, "Anyone with the link", + MD5). Max 2 submissions, only the last counts. Link must be a **file** link (`drive.google.com/file/d/...`); a folder link is rejected ("link is a folder, not a file").
- Tab **Test Solution** = dry run of the Docker image (1 per day, shows only pass/fail, no score).
- **Private round: 2026-11-20 09:00 to 11-23 23:59**: organisers release private_test; **the team runs its frozen solution itself and uploads submission.jsonl** (Submission tab). Organisers re-run the Docker to verify. Top 2 announced Dec 2.
- Scoring hardware: Rule page says 1x NVIDIA A30; Docker Guide says RTX 5090 (driver 595, CUDA 13.2, no network, Xeon, 64 GB). Contradiction, unresolved: ask organisers. Torch in image supports sm_80 and sm_120.

## 2. Pipeline
audio -> ASR (PhoWhisper-large fine-tuned, ~1.54B) -> LLM policy (Qwen3-4B-Instruct-2507 + LoRA r16/a32, merged fp16, 4.02B) -> agent loop (one JSON action per step: call or stop; max 8 steps; repeat guard; budget `LIFEMATE_LOOP_BUDGET_S`=18 s) -> `zaic_tools.ToolEnv`. Total 5.566B params (MANIFEST.json, verified).
Score per sample: 0 if final state hash differs, else F1 of tool calls. Empty gold list scores only if prediction empty.
Data: official package `train+public_test.zip` (v3.2.0): train 1545 samples, public_test 634 (no labels). `training/prepare_sft.py` makes step-wise SFT rows (a sample with K calls -> K+1 rows, ending in `stop`), group-split by normalised request text 80/10/10 (seed 2026) -> 4062/521/503 rows. ASR-noise augmentation: `scripts/build_asr_train_data.py` (asr_probability 0.5, half of noisy rows from PhoWhisper-large base, `--include-dev-holdout` for final).

## 3. Results so far
- Public: 16.90 (ckpt 508), 22.22 (1016), 25.71 (1524, medium ASR), 25.99 (warm start), **50.16** (v1: noise adapter + large-ft ASR; rank ~287 in the SLM column), **50.46** (v2: `train_final`, +dev+holdout; only +0.30, within noise). Top of the leaderboard is ~95-99 (check whether that is the SLM column).
- Dev (154 samples, noise about +-4): 1524+large-ft 54.62; noise adapter (v1) 53.46; merged fp16 53.78.
- Full diagnostics and per-error analysis: `docs/findings_20261111.md` (summary below).
  - Oracle (gold text, ASR bypassed): 67.55 vs 54.62 with real ASR -> ASR costs ~13 points, but the model still loses ~32 points with perfect text. More ASR work cannot exceed ~67.
  - Errors with perfect text (35/154 samples score 0): conditional requests ("if A then X else Y") and tool-result dependence ~11, picking IDs from earlier tool output ~7, `sys.ask_missing_info` under/over-asking ~5, arithmetic/number copying ~5, format/missing args ~3, early empty stop ~2, ambiguous labels ~2.
  - Best-of-k voting (k=5, T=0.7): 53.89 vs 53.46 greedy (2 better, 1 worse, 151 equal) but latency p50 22 s, max 80 s. **Rejected.** Candidates mostly agree: errors are systematic (learned behaviour), not decoding noise.
  - Zipformer `hynt/Zipformer-30M-RNNT-6000h` zero-shot: dev WER 0.189 vs large_ft 0.104; best-of-two ceiling 0.094. Licence cc-by-nc-nd-4.0. **Rejected.**
  - Gold-only vs ASR-noise SFT: no measurable difference on dev; the noise data is ~6% of v1's training steps.
- Latency on 3090: merged model with ASR p50 11.9 s, p95 23.8, max 28.2 (close to the 30 s limit). Greedy dev (unmerged adapter) p50 9.8 s, max 25 s.

## 4. Assets
- HF (public): models `vincent1512/lifemate-phowhisper-large-ft`, `vincent1512/lifemate-qwen-adapter` (v1 adapter = stage_noise), `vincent1512/lifemate-backup`; dataset `vincent1512/lifemate-train-data` (train split + processed_sft, no public_test). Do not overwrite the v1 adapter repo; use a new repo for v2 (e.g. `lifemate-qwen-adapter-final`).
- Local Mac folder (this repo): `data/`, `runs/` (checkpoints/logs of ckpt 1524 and warm start), notebooks in `notebooks/`, Docker kit in `docker_submission/`.
- Docker v1 image: built on the Mac (arm64, Docker Desktop with Rosetta, amd64 build) from `docker_submission/`; exported `zac2026_TeamName.tar.gz` (19,330,062,495 bytes, MD5 `29e1b770c23983015710d5df3dae3fcb`) in `~/Documents/zac_kit/zac_build_kit/`; uploaded to the user's Drive; **Test Solution passed** on 2026-11-10 (8 min). Formal "Solution Submission" of v1 has NOT been done yet.

## 5. Docker image facts
- Base `nvidia/cuda:12.8.1-devel-ubuntu22.04` (Python 3.10), torch 2.9.1 cu128, transformers 4.57.1, peft 0.17.1, accelerate 1.10.1, safetensors 0.6.2, `huggingface_hub>=0.34,<1.0`, scipy pinned 1.15.3 (1.16 needs Python 3.11). Entrypoint is run by organisers as `/bin/bash /code/predict.sh` -> `preprocessing.py` then `predict.py` -> `/result/submission.jsonl` (field `time` in ms). `zac_runtime.detect_split` finds the split dir by `*_tools_catalog.json`, so private_test works unchanged.
- Weights in image: `/code/saved_models/asr/` and `/code/saved_models/llm/` (Docker Guide example shows a single `models.safetensors`; our layout is unconfirmed with organisers). Base Qwen is NOT in the image, only the merged fine-tuned weights.
- Build on Mac: `bash docker_submission/mac_build.sh 2>&1 | tee mac_build.log` (steps preflight, venv, download, merge on CPU fp32->fp16, build, smoke, export; resumable via `.done_mac/`). Env overrides: `ADAPTER_REPO`, `TAG`, `OUT_NAME`. Merge is done by `scripts/merge_for_submission.py` (`merge_and_unload`).
- `training_code/README.md` (inside the image) must list HF URLs: dataset URL and Qwen licence (Apache-2.0, checked 2026-11-10) are filled in the working copy, **but the built image still has the old README**: rebuild before the formal submission. Stage-1 command is reconstructed from `trainer_state.json`.

## 6. Current state and next steps
State (2026-11-11): v1 Docker image built and Test Solution passed (not formally submitted). v2 (`train_final`) trained and scored 50.46; its adapter is in the local pack `lifemate-final-20261111-*.tar.gz` under `stage_final/final` (not uploaded to HF). All Vast instances were cancelled; check the Vast balance (~$2.9 before the oracle/best-of-k runs).
1. **Formal "Solution Submission" of v1 before 2026-11-16** (insurance). Rebuild the image first so `training_code/README.md` inside the image has the dataset URL and Qwen licence (working copy already updated). v2 is optional: +0.30 public is not evidence of a real gain.
2. Improvement ideas (ranked; every training run costs real money, one 3090 step is ~54 s):
   - (a) Code-side constraints without training: restrict IDs (`contact_id`, `flight_id`, ...) to those present in context/tool results, normalise times/amounts deterministically. Cheap, expected few points. Measure on dev with the same dev pipeline.
   - (b) Targeted data from the existing labelled data only (no new labels can be created: do not read `<split>_environment/`): oversample hard categories (conditionals, ID choice, ask_missing_info) x2-3; optionally add short template-generated "reasoning/plan" text before each action (changes the output format, latency grows). Then a short warm-start run (~100-150 steps, ~$0.5-1) and compare on dev against 53.46 (paired per-sample counts, dev is only 154 samples). Ask the organisers about synthetic-data rules first.
   - (c) Nearest-neighbour few-shot examples from the training set in the prompt (needs retraining with the new prompt format; longer prompts are slower).
   - (d) ASR: beam search and context prompt-biasing (`prompt_ids`) are cheap to test on dev audio; expected gain small (ASR ceiling is ~13 points).
   - Not recommended: best-of-k, Zipformer, more of the same SFT (v2 gave +0.3).
3. Ask the organisers in writing: data re-hosting permission, A30 vs 5090, weights layout (`asr/`+`llm/`), licence rules for third-party models. Keep their replies as evidence.
4. Upload model cards (`scripts/upload_model_cards.py`, getpass token, revoke the token after).
5. Private round (2026-11-20 to 11-23): run the frozen Docker on a GPU machine with Docker + NVIDIA toolkit (a Vast "Ubuntu VM" template, not a plain container), mount private_test at `/data`, collect `submission.jsonl`, upload. Do not modify the solution after freezing.

## 7. Gotchas learned
- Vast regular instances are containers (no Docker inside). Some VM hosts have broken CUDA (nvidia_uvm hang) although `nvidia-smi` works; test CUDA with a timeout before building.
- `fetch_model` in the notebook skips download if `config.json` exists (an interrupted download is not resumed). Do not `pip install -U huggingface_hub` (breaks transformers 4.57.1).
- fp16 merge on CPU is extremely slow; merge on CPU in fp32 then cast to fp16 (done in `merge_for_submission.py`). CPU-merged weights were not re-scored; GPU-merged weights scored 53.78 on dev.
- The Test Solution form accepts only a Drive FILE link. Drive shows 18 GB for the 19.33 GB file (GiB vs GB).
- Mac shell: `device_bash` home mapping `$HOME/mnt/zalo_AI_CHALLEGNE`; a stray file `docker_submission/training_code/zac_runtime.py` could not be deleted (no delete permission) and is excluded from the kit.

## 8. Reference docs
`docs/Docker_Guide_ZAC2026_LifeMate.pdf` (official, kept locally, NOT in git), `docs/baseline_design.md`, `docker_submission/BUILD.md`, `docker_submission/HUONG_DAN_DONG_GOI.md` (friend packaging guide, Linux VM path with `vm_build.sh`), `data/README.md` (dataset terms: "do not redistribute", see rules above: the user states the organisers allowed uploading; keep written evidence).

## 9. Repo notes
- Restricted competition data and per-sample logs containing training texts/labels are never committed (`data/`, `reports/*_samples.jsonl`, `lifemate*/`, `*.tar.gz` stay local). Do not add them.
- Notebooks: `notebooks/lifemate_vast_final.ipynb` (phases via `PHASES`), `lifemate_vast_oracle.ipynb` and `lifemate_vast_bestofk.ipynb` (experiments already run), `lifemate_vast_asr_more.ipynb`, `lifemate_vast_merge.ipynb`. Generators: `scripts/build_*_notebook.py`. `scripts/eval_zipformer_dev.py` is a diagnostic only.
- Branch `codex-handoff` holds this handoff state; `main` is untouched and `docker-submission` holds the Docker packaging work.
