# Resume LifeMate from GitHub on Vast

Download `notebooks/lifemate_vast_github_resume.ipynb` from this repository. Upload only that notebook to the new Vast instance, open it with a Python 3.12+ kernel, then Run All.

The notebook downloads the pinned public Release asset `lifemate-vast-resume-clean-508-20261107.tar.gz` from tag `lifemate-resume-508-20261107`. It verifies SHA-256, safely extracts the project and competition dataset, checks CUDA and the saved training state, reconstructs the original SFT data, and resumes checkpoint 508. The regenerated train/dev hashes must match the original run before training starts.

Requirements:

- NVIDIA GPU with at least 24 GB VRAM.
- At least 22 GiB free before download/extraction; allocate about 50 GB disk or more when renting.
- Internet access for GitHub Release, Python dependencies and pinned Hugging Face model weights.

Default project: `/workspace/lifemate`. If a different project already occupies that folder, choose a new `WORKSPACE_DIR` in the first notebook cell. An extracted bundle with a matching marker is reused, and an interrupted `.part` download can resume by rerunning its cell.

The run keeps the original alphabetical tool order and schema-order augmentation. It saves adapters/checkpoints under `lifemate_runs/vast-20261106-124816/full/` and the full-dev report as `dev_resumed_final_full.json`. Public-test inference is a separate step after reviewing dev results.

The Release is public at the repository owner's explicit request and contains the competition dataset archive and saved LoRA checkpoints. No token is required to download the public asset. A download token should never be written into notebook cells or outputs.

The download/extraction helper has CPU checks for partial downloads, servers ignoring Range, unsafe archive paths and preservation of existing projects. Resume data hashes were reproduced locally. CUDA training has to be run on Vast.
