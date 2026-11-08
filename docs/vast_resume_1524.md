# Vast resume notebook for checkpoint 1524

## Files to upload

Upload these two files to `/workspace` on Vast:

1. `notebooks/lifemate_vast_resume_1524.ipynb`
2. `artifacts/handoff/lifemate-resume-latest.tar.gz`

The archive SHA-256 must be:

```text
8849e69703fc2a62dd084b15b8ea2f1803b6fd4b8b193d2cf224518f27048b4c
```

The notebook verifies the archive hash before extracting checkpoint 1524. Use the Python 3.12+ kernel with CUDA enabled. The base source/data bundle and model are fetched by the notebook.

## Modes

- `MODE="warmstart"`: default. Loads checkpoint 1524 weights into a new run with fresh optimizer/scheduler. The proposed learning rate `2e-5` and 508-step schedule are hypotheses to measure, not established improvements. A two-step smoke test runs first by default.
- `MODE="continue"`: resumes the original run from the newest complete checkpoint in `full/`. It restores optimizer/scheduler/RNG state and requires the original train/dev hashes.
- `MODE="eval_only"`: skips training and evaluates checkpoint 1524. Its default public choice is the baseline.

## Notebook order

The notebook bootstraps the pinned source/data bundle, verifies and backs up checkpoint 1524, applies the checksum-verified code overlay, checks GPU/disk/data, installs dependencies, rebuilds the exact SFT files and checks their hashes, then evaluates the baseline on dev before any training. In warm-start mode it runs the two-step smoke test, trains the candidate, evaluates that candidate on the same dev set, and prints the comparisons. The optional ASR cache is shared between the baseline and candidate evaluations.

`PUBLIC_CHOICE` is explicit: `candidate` for warm-start/continue and `baseline` for eval-only by default. Change it to `baseline` or `candidate` deliberately before running the public inference cell. If candidate state-match is more than `SKIP_PUBLIC_IF_WORSE_BY` below baseline, public inference is skipped. Set the threshold to `None` to disable this guard. The guard uses local dev state-match only; it does not predict the official score. `RUN_PUBLIC` controls whether inference is run at all.

Dev has 154 samples, so small changes are noisy. The printed call-F1 is a locally computed diagnostic and is not known to match the contest formula. Do not treat it as proof that a technique improves the official score.

## Regenerate the notebook

The Python builder is the source of truth; do not edit the notebook by hand. Keep Git `HEAD` at `main` commit `d3a68438be04e65138ca4ddf3352174284fc5539` so the builder reads the original overlay hashes from the expected baseline, then run:

```bash
python scripts/build_resume_notebook.py
```

The builder embeds patched code and hashes in the notebook. Verify the overlay against the pinned Vast bundle before uploading a regenerated notebook.

## Handoff 1524 qua GitHub Release (không cần upload tay)

Tạo Release một lần từ máy có `gh` (asset phải đúng tên `lifemate-resume-latest.tar.gz`):

```bash
gh release create lifemate-resume-1524-20261109 artifacts/handoff/lifemate-resume-latest.tar.gz \
  --repo Vincent-1512/ZALO_AI_CHALLENGE_2026 --title "LifeMate resume checkpoint-1524" \
  --notes "Handoff checkpoint-1524. SHA-256 8849e69703fc2a62dd084b15b8ea2f1803b6fd4b8b193d2cf224518f27048b4c"
```

Notebook mặc định tải asset này (`HANDOFF_RELEASE_TAG`), kiểm SHA-256 và tự dùng file upload tay nếu Release chưa có.

## Số epoch và early stopping (warm-start)

- Mặc định 1 epoch (508 bước, `WARM_MAX_STEPS`), LR giảm dần về ~0 đúng trong số bước đó. Điểm public tăng qua ba lần nộp (checkpoint 508, 1016, 1524) dù eval loss xấu đi sau bước 762, nên không dùng eval loss để chọn checkpoint.
- Eval dev loss 4 lần/epoch (`WARM_EVALS_PER_EPOCH`). Early stopping (`WARM_EARLY_STOP`, patience `WARM_PATIENCE`, ngưỡng `WARM_THRESHOLD`) chỉ là van an toàn khi loss xấu đi liên tục; với run 1 epoch nó hiếm khi kích hoạt. Adapter cuối là trọng số cuối run (`--no-load-best`).
- Thời gian là ước tính, chưa đo. Dùng GPU nhanh hơn cho bước train nếu có.

## Công tắc tốc độ (mặc định giữ nguyên hành vi, chỉ cho `MODE="warmstart"`)

- `WARM_GRAD_CKPT=False`: tắt gradient checkpointing, nhanh hơn nhưng cần khoảng 80 GB VRAM (A100/H100 80GB). Khi tắt, smoke test chạy các bản ghi dài nhất trước để lộ lỗi hết bộ nhớ sớm. Mức nhanh hơn chưa đo.
- `WARM_USE_AUGMENT=False`: chỉ dùng dữ liệu gốc, không có bản đảo thứ tự tool; 1 epoch còn ~254 bước. Thứ tự tool lúc inference cố định theo bảng chữ cái, nên bản đảo chủ yếu có tác dụng điều hòa; ảnh hưởng đến điểm là giả thuyết chưa đo.
- `STOP_AFTER_BASELINE=True`: dừng sau khi đo dev baseline để xem log lỗi rồi mới quyết định train.
- Smoke test in tên GPU và thời gian smoke 2 bước (gồm nạp model) để ước lượng tốc độ.
