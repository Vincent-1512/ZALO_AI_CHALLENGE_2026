# Resume LifeMate from checkpoint-1524 on Vast

Upload two files to the new Vast instance, open the notebook with a Python 3.12+ kernel, then Run All:

1. `notebooks/lifemate_vast_resume_1524.ipynb`
2. `lifemate-resume-latest.tar.gz` (checkpoint 1524 handoff, kept locally under `artifacts/handoff/`; SHA-256 `8849e69703fc2a62dd084b15b8ea2f1803b6fd4b8b193d2cf224518f27048b4c`) into `/workspace`.

The notebook downloads the pinned base bundle (code, dataset, model downloader) from the existing Release `lifemate-resume-508-20261107`, imports checkpoint 1524, backs it up outside `full/`, and applies the patched code embedded in the notebook (SHA-256 checked).

`MODE` in the first cell:

- `warmstart` (default): a NEW short run initialised from the 1524 adapter weights with a fresh optimizer/scheduler (LR `WARM_LR` decaying to ~0 over `WARM_MAX_STEPS`). Output goes to `lifemate_runs/<run>/warm_from_1524/`. This is not a true resume.
- `continue`: true resume of the original run from 1524 with identical hyperparameters and data hashes. Early stopping still compares against the old best loss (0.03678), so it stops after one or two more evaluations.
- `eval_only`: dev evaluation and optional public submission for 1524 only.

The notebook evaluates 1524 and the new adapter on the 154-sample dev split, selects by dev rollout (not eval loss) and creates the public submission only for the selected adapter. Dev is small, so differences of a few percentage points are within noise.

Regenerate the notebook after changing any embedded file: `python scripts/build_resume_notebook.py`.
