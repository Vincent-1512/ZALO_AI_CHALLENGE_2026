---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
tags:
- lora
- peft
- backup
- zalo-ai-challenge
---

# lifemate-backup

Backup of one training run for the Zalo AI Challenge 2026 "LifeMate" entry. This is **not** a ready-to-use model.

## Contents
- `qwen_adapter_checkpoint-1524/`: LoRA adapter (QLoRA r=16, alpha=32) for `Qwen/Qwen3-4B-Instruct-2507`, checkpoint at step 1524. Load it on top of the base model:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "vincent1512/lifemate-backup", subfolder="qwen_adapter_checkpoint-1524")
```
- `reports/`: dev evaluation reports and ASR transcript caches produced during the run. The competition audio itself is not included.

No standalone evaluation is claimed for this repository.

## License
The adapter is a derivative of Qwen3-4B-Instruct-2507 (Apache-2.0).
