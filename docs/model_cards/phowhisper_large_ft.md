---
license: bsd-3-clause
base_model: vinai/PhoWhisper-large
language:
- vi
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- vietnamese
- asr
- lora-merged
- zalo-ai-challenge
---

# lifemate-phowhisper-large-ft

[vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) fine-tuned with LoRA on the audio of the Zalo AI Challenge 2026 "LifeMate" task (Vietnamese voice commands for app actions). The adapter is merged into the weights, so this is a standalone Whisper model.

## Training
- Method: LoRA r=32, alpha=64, dropout 0.05 on q/k/v/out_proj/fc1/fc2 (about 57.7 M trainable of 1.60 B parameters), merged after training.
- 4 epochs, 1004 steps, learning rate 1e-4 (cosine, 5% warmup), batch 4 x gradient accumulation 4, bf16 base with fp32 adapter.
- Data: the competition's training audio and its transcripts (current request and past user turns). The data is not redistributed here.

## Results (held-out dev audio, 478 clips not used for training)
| Model | WER |
|---|---|
| vinai/PhoWhisper-large | 0.2469 |
| this model | 0.0921 |

In the full LifeMate pipeline (this ASR + Qwen3-4B tool-calling policy) the official dev score rose from 33.89 to 54.62 (154 dev samples, sampling error about +-4). Other conditions were not evaluated.

## Usage
```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf, torch

processor = AutoProcessor.from_pretrained("vincent1512/lifemate-phowhisper-large-ft")
model = AutoModelForSpeechSeq2Seq.from_pretrained("vincent1512/lifemate-phowhisper-large-ft", torch_dtype=torch.float16).cuda().eval()
audio, rate = sf.read("clip.wav", dtype="float32")          # mono, 16 kHz
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
ids = model.generate(inputs.input_features.cuda().half(), language="vi", task="transcribe")
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## Limitations
- The model memorised its training audio (near-zero WER there); the 9.2% WER above is on held-out dev clips from the same distribution. Apps and speakers not seen in training may give a higher WER; this was not measured.
- Trained for short Vietnamese voice commands; not evaluated on long-form or other-domain speech.

## License and attribution
Derivative of PhoWhisper-large, released under the BSD 3-Clause License (copyright notice and terms of the original apply). PhoWhisper is itself fine-tuned from OpenAI Whisper (MIT).

```bibtex
@inproceedings{PhoWhisper,
  title     = {{PhoWhisper: Automatic Speech Recognition for Vietnamese}},
  author    = {Thanh-Thien Le and Linh The Nguyen and Dat Quoc Nguyen},
  booktitle = {Proceedings of the ICLR 2024 Tiny Papers track},
  year      = {2024}
}
```

*Tóm tắt (vi):* PhoWhisper-large được fine-tune trên audio cuộc thi LifeMate; WER trên dev giữ lại giảm từ 0.2469 xuống 0.0921. Model đã học thuộc audio train, nên app/giọng chưa thấy có thể kém hơn.
