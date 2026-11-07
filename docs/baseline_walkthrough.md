# LifeMate baseline: bản đồ notebook và NLP pipeline

Tài liệu này giải thích notebook Vast, các script nó gọi, dữ liệu đi qua hệ thống thế nào, và những chỗ cần nhớ khi chạy hoặc cải tiến baseline. Notebook là bộ điều khiển; phần xử lý thật nằm trong các module Python.

## 1. Bài toán đang giải

LifeMate nhận một mẫu gồm audio tiếng Việt mới nhất, ngữ cảnh người dùng (thời gian, timezone, vị trí, profile), lịch sử hội thoại và các app/tool hiện cài. Hệ thống phải biến lời nói thành một chuỗi hành động có cấu trúc, gọi tool đúng thứ tự, dùng kết quả tool để quyết định bước sau, rồi để ứng dụng ở trạng thái mong muốn.

Đây không phải chỉ là ASR hay phân loại intent:

```text
audio mới + audio/text history
        ↓ ASR
transcript + context + available tool schemas + history
        ↓ Qwen chọn đúng một hành động
JSON action: call(tool,args) hoặc stop
        ↓ kiểm tra schema → thực thi tool thật
tool result trở thành context bước tiếp
        ↓ lặp đến stop/lỗi/giới hạn bước
tool_calls + state_hash
```

`PhoWhisper-medium` nhận dạng tiếng Việt. `Qwen3-4B-Instruct-2507` quyết định hành động và arguments. Qwen được fine-tune bằng LoRA; PhoWhisper hiện chỉ dùng inference, không được fine-tune trong notebook.

## 2. Dữ liệu và cách tạo ví dụ học

Train có khoảng 1.545 phiên; public test có 634 mẫu theo bản kiểm kê trong dự án. Mỗi phiên train có audio, transcript tham chiếu (`audio_text`), context/history, nhãn `tool_calls`, kết quả các tool và `final_state`. Public test không có nhãn để train.

Một phiên có thể cần nhiều tool. `prepare_sft.py` chuyển mỗi phiên thành nhiều ví dụ supervised fine-tuning (SFT):

- Ở bước đầu, prompt có yêu cầu và context; target là tool call đầu tiên.
- Ở bước tiếp theo, prompt thêm các call/result gold trước đó; target là call kế tiếp.
- Sau call cuối, target là `{"action":"stop"}`.

Do đó một phiên tạo `số calls + 1` bản ghi. Bản kiểm kê ghi nhận 5.086 nhãn bước kế tiếp trên toàn bộ 1.545 phiên; số bản ghi train/dev/holdout được in ra manifest khi chuẩn bị.

`prepare_sft.py` chia phiên thành train/dev/holdout theo tỷ lệ mặc định 80/10/10, seed 2026. Nó gom các yêu cầu có transcript đã chuẩn hóa giống nhau trước khi chia để giảm nguy cơ các bản trùng rơi vào cả train và dev. Sau khi chia mới tạo trajectory records. Holdout được tạo nhưng notebook hiện không tự chạy báo cáo holdout.

Một điểm rất quan trọng: train dùng transcript chuẩn; inference dùng transcript ASR. Vì vậy loss train thấp không đảm bảo ASR sai tên/ngày/số tiền vẫn cho tool arguments đúng. Đây là chênh lệch phân phối đầu vào cần đo.

## 3. Notebook cell theo cell

Các cell được mô tả theo luồng baseline Vast tổng quát. Notebook resume hiện tại là `notebooks/lifemate_vast_resume.ipynb`; notebook này bỏ smoke và chỉ resume checkpoint rồi đánh giá dev.

### Cell 0 — Giới thiệu và luật vận hành

Nói notebook chạy trên Vast, phân biệt smoke, full train, dev và public inference. Chỉ đọc `*_environment` qua API được phép của bộ kit; không mở dữ liệu trạng thái môi trường để dự đoán.

### Cell 1 — Cấu hình và đường dẫn

Đặt `PROJECT_DIR`, `DATA_ROOT`, `RUN_ROOT`, `RUN_ID`, model path và tham số.

- `RUN_ID`: tên thư mục của một thí nghiệm. Thí nghiệm mới phải dùng ID mới; resume phải giữ nguyên ID.
- `LR=5e-5`, `MAX_EPOCHS=30`, context ban đầu 8192, accumulation 16, LoRA r16/alpha32/dropout 0.05: đây là cấu hình thử nghiệm, không phải giá trị tối ưu đã được chứng minh.
- `EVALS_PER_EPOCH=2`, patience 4: nếu mỗi epoch đúng hai lần eval, bốn lần không cải thiện xấp xỉ hai epoch. Early stopping đếm lần đánh giá, không đếm epoch.
- `SCHEMA_PERMUTATIONS=1`: trong code hiện tại đây là bật/tắt (0 hoặc khác 0), không phải số biến thể tùy ý. Mỗi trajectory train được giữ nguyên và thêm một bản có thứ tự `available_tools` đã hoán vị; target và schema không đổi. Bản ghi train gần như tăng gấp đôi nếu có nhiều tool.
- `RUN_SMOKE`, `RUN_FULL_TRAINING`, `RUN_DEV_EVALUATION`, `RUN_PUBLIC_INFERENCE`: bật/tắt từng pha. Cấu hình mặc định hiện bật smoke, full, dev và public; vì thế khi chạy hết notebook nó có thể đi tới sinh submission.

Các biến Python chỉ sống trong kernel. File được ghi dưới `lifemate_runs/<RUN_ID>` vẫn nằm trên ổ đĩa. Khi kernel mất, phải khôi phục đúng RUN_ID và dùng checkpoint để resume.

### Cell 2–3 — Kiểm tra máy, bundle và dữ liệu

Cell code chạy `nvidia-smi`, kiểm tra `torch.cuda`, VRAM và dung lượng; kiểm tra `SHA256SUMS`; nếu cần thì bung `data/archives/train+public_test.zip` vào `data/raw/`.

Máy người dùng hiện là RTX 4090 24 GB. `nvidia-smi` hiện 24564 MiB, tức khoảng 23.5 GiB, nên kiểm tra notebook cần chấp nhận GPU hạng 24 GB với ngưỡng thực tế khoảng 23 GiB. Dataset gốc bung ra khoảng 1.05 GiB theo archive trong workspace. Dung lượng vẫn phải đủ cho model, thư viện, cache và checkpoint.

### Cell 4–5 — Môi trường, model và SFT

Cài `requirements-train.txt`; file này không cài PyTorch CUDA nên torch phải được cài đúng môi trường kernel trước. Sau đó:

1. `scripts/download_models.py` tải revision cố định của PhoWhisper và Qwen vào `models/`.
2. `training/prepare_sft.py` đọc train split, chia train/dev/holdout và tạo JSONL trajectory trong `data/processed/sft/`.
3. `manifest.json` ghi số phiên/bản ghi và hash ID mỗi split.

Cell tải model theo thứ tự ASR rồi LLM; progress `12/13` là số file đã xong, không phải tỷ lệ bytes. Trọng số cuối có thể lớn nên phần trăm theo file có vẻ đứng.

### Cell 6, 8 — Augmentation thứ tự schema

Cell markdown giải thích; cell code giữ bản gốc train và thêm clone có hoán vị thứ tự danh sách tool. Cùng một `sample_id` dùng cùng thứ tự xáo để các bước trajectory nhất quán. Dev/holdout không bị thay.

Đây là augmentation an toàn về nhãn nhưng vẫn là giả thuyết: nó có thể giúp model bớt phụ thuộc vị trí tool trong prompt, cũng có thể không cải thiện. So sánh 0 với 1 ở RUN_ID khác nhau. Không có paraphrase hoặc sinh nhãn mới trong notebook.

### Cell 7 — Ghi chú resume

Nếu train bị gián đoạn, dùng cùng RUN_ID và tham số, tắt smoke để không ghi lại smoke, đặt `RESUME_CHECKPOINT` tới `full/checkpoint-*`. Cần checkpoint còn trên ổ. Final adapter chỉ chứa weights LoRA; resume cần checkpoint Trainer, không thể resume optimizer từ thư mục `final/`.

### Cell 9–10 — Preflight và smoke

Preflight gọi `train_lora.py --preflight-only --auto-context` trên train/dev để tokenize mọi bản ghi và xác định context phù hợp không vượt trần cấu hình. Nó không load model train hay dùng GPU cho backward. Kết quả ở `lifemate_runs/<RUN_ID>/preflight.json`.

Smoke dùng tối đa 64 sample_id train, 16 dev, một epoch; mục tiêu là phát hiện lỗi cài đặt, tokenizer, forward/backward, CUDA, save adapter. Nó không dùng để chọn model cuối. Adapter ở `smoke/final/`.

### Cell 11–12 — Full train / resume

Full dùng toàn bộ train split sau augmentation, tối đa 30 epoch. Trainer tính eval loss nhiều lần mỗi epoch; save và eval cùng bước; theo dõi `eval_loss`, dừng sớm nếu không cải thiện đủ lâu và nạp lại checkpoint tốt nhất trước khi ghi `full/final/`.

Cell train chạy `training/train_lora.py`. Tham số train quan trọng lấy từ `common` ở notebook; script cũng có defaults nếu chạy CLI độc lập. Mỗi lần đổi learning rate/context/augmentation/LoRA nên là RUN_ID mới, nếu mục tiêu là so sánh độc lập.

### Cell 13–14 — Rollout dev

`training/evaluate_dev.py` chạy cả base và LoRA qua pipeline inference thật trên các phiên dev. Báo cáo gồm changed-app state match, tool-call precision/recall/F1 chẩn đoán, OOS exact và latency p50/p95/p99/max. File là `dev_base.json`, `dev_full.json`.

Đây là diagnostics của đội, chưa phải điểm chính thức BTC vì công thức scorer đầy đủ chưa có trong tài liệu đang lưu. Loss chỉ đo khả năng dự đoán token target; rollout dev gần với nhiệm vụ hơn nhưng cũng không chứng minh sẽ thắng public/private. Notebook hiện chưa tự chọn base hay LoRA; public cell mặc định dùng LoRA `full/final` trừ khi sửa `USE_BASE_MODEL`/adapter.

### Cell 15–16 — Public inference và kiểm tra JSONL

` scripts/run_inference.py` nhận `public_test`, không nhãn. Notebook dùng adapter đã chọn, ghi `lifemate_runs/<RUN_ID>/submission.jsonl`, kiểm tra số dòng bằng số mẫu input, ID không trùng và mỗi dòng đúng các khóa `id`, `tool_calls`, `state_hash`.

Cell này chỉ tạo file; không nộp portal. Mỗi mẫu cần dưới 30 giây trên máy chấm A30 theo hướng dẫn LifeMate. Thời gian trên RTX 4090 không thay phép đo A30.

### Cell 17 — Nhắc giới hạn

Nhắc full train, đánh giá, inference là các pha riêng; không dùng public để tune. Notebook không tự đóng Docker hoặc nộp bài.

## 4. Các module Python và vai trò

- `scripts/download_models.py`: tải model theo repo/revision trong `configs/baseline.json`.
- `training/prepare_sft.py`: đọc train labels, chia group split và biến tool trajectories thành prompt/target JSONL.
- `training/train_lora.py`: tokenize/preflight, nạp base Qwen 4-bit, gắn LoRA, tạo Trainer, early stopping/checkpoint và lưu adapter.
- `training/response_loss.py`: chỉ tính loss trên token phản hồi của assistant, bỏ prompt khỏi loss; `-100` là vị trí không tính loss.
- `training/attention_backend.py`: kiểm tra/chọn đường attention phù hợp độ dài context và GPU, tránh cấu hình kernel không hỗ trợ.
- `src/lifemate/prompts.py`: xây system prompt, context, history và danh sách schema tool; cắt history cũ nếu context vượt giới hạn.
- `src/lifemate/models.py`: tải PhoWhisper/Qwen, ASR audio, greedy decode và parse JSON action.
- `src/lifemate/data.py`: tìm split, catalog, audio path và import ToolEnv từ dataset kit.
- `src/lifemate/runner.py`: vòng lặp inference từng mẫu: ASR → prompt → sinh action → validate → call → đưa observation vào bước kế → lấy call list/state hash.
- `training/evaluate_dev.py`: rollout có nhãn dev, tính diagnostics và latency. Chỉ ở diagnostics train mới dùng `env.get_state()` để đối chiếu final state.
- `scripts/run_inference.py`: entry point tạo JSONL test. Không được truy cập gold labels.
- `configs/baseline.json`: model/revision, đường dẫn, inference limits và generation defaults. Notebook CLI override learning rate/epochs/LoRA; đừng nhầm JSON inference config với cell train config.

## 5. Bên trong một optimizer step

Mỗi bản ghi SFT được tokenize bằng chat template của Qwen. Label của prompt là `-100`; label của target action/stop là token ID thật. Batch size vật lý là 1; gradient accumulation 16 gộp gradient của 16 microbatch trước khi optimizer update (effective batch xấp xỉ 16 bản ghi trên một GPU). Padding được mask khỏi attention/loss.

Model nền nạp ở NF4 4-bit với double quantization. Trọng số nền đóng băng; LoRA matrices nhỏ được học trên các projection q/k/v/o/gate/up/down. Gradient checkpointing giảm VRAM bằng cách tính lại activation trong backward; tốn thêm compute. Dtype BF16 nếu GPU hỗ trợ, ngược lại FP16. Optimizer paged AdamW 8-bit và cosine scheduler với warmup 3% là thiết lập tiết kiệm bộ nhớ.

`epoch` là một lượt qua toàn bộ train rows. Vì trajectory tạo nhiều rows trên mỗi phiên và schema augmentation thêm clone, epoch ở đây không bằng lượt qua 1.545 audio một lần; số row được xem trong `manifest.json` và file augmented.

## 6. Cách đọc output và nơi lưu

Với `RUN_ID=vast-...`:

```text
/workspace/lifemate/
  data/archives/train+public_test.zip
  data/raw/train+public_test/                # source data đã bung
  data/processed/sft/{train,dev,holdout}.jsonl
  data/processed/sft/manifest.json
  models/PhoWhisper-medium/
  models/Qwen3-4B-Instruct-2507/
  lifemate_runs/<RUN_ID>/
    preflight.json
    train_schema_augmented.jsonl             # nếu augmentation bật
    smoke/final/                             # smoke adapter
    full/checkpoint-*/                       # resume + optimizer/scheduler state
    full/final/                              # adapter cuối sau nạp best checkpoint
    dev_base.json
    dev_full.json
    submission.jsonl
```

Jupyter output hiện tại hiển thị log từ `subprocess.run`; notebook chưa ghi mọi stdout/stderr thành file log riêng. Checkpoint/model/data được ghi trực tiếp xuống thư mục dự án. Nếu instance bị hủy, khả năng giữ các file phụ thuộc vào ổ Vast; cần bảo đảm `/workspace` là volume còn lại hoặc sao lưu adapter/checkpoint sang nơi được phép.

## 7. Luật và giới hạn ảnh hưởng trực tiếp đến thiết kế

Theo thể lệ tiếng Việt và hướng dẫn LifeMate đã lưu trong dự án: chỉ dùng dữ liệu cho mục đích cuộc thi/nghiên cứu; không chuyển hoặc chia sẻ dataset ra ngoài đội/cho bên thứ ba; không dùng private/test labels để train; inference offline; giữ giới hạn tham số/context/latency; không đọc trực tiếp `*_environment`; giải mã xác định. Nếu nghi ngờ cloud hosting dataset hoặc thay đổi model/data source, xin xác nhận BTC trước.

Mỗi public submission tốn một trong tối đa năm lượt/ngày, nhưng tạo JSONL cục bộ không tốn lượt. Không tune hyperparameter bằng public score nhiều vòng; dùng dev/holdout nội bộ và nộp ứng viên đã chọn.

## 8. Hạn chế cần nhớ khi đánh giá baseline

- Early stopping theo `eval_loss` không chắc chọn rollout tốt nhất.
- Dev rollout là local proxy; không có công thức chính thức đầy đủ nên không gọi là điểm BTC.
- Train transcript sạch nhưng inference transcript nhiễu là khoảng cách lớn chưa giải quyết.
- 30 epoch là trần, không phải khuyến nghị chạy đủ hay cấu hình tối ưu.
- Schema order augmentation chỉ một phép thử; nó không tạo thêm kiến thức mới.
- Model cuối được notebook chọn mặc định là LoRA; người dùng cần nhìn `dev_base.json` và `dev_full.json` trước khi quyết định adapter.
- Context fitting hiện bỏ history cũ; một yêu cầu có đại từ/tham chiếu xa có thể mất thông tin.
- Notebook hiện chỉ kiểm tra schema/đếm public JSONL, không xác nhận đủ điều kiện submission Docker hoặc thời gian A30.

## 9. Bản đồ luồng gọi

```text
notebooks/lifemate_vast_resume.ipynb
 ├─ download_models.py ──> models/ (PhoWhisper, Qwen)
 ├─ prepare_sft.py ──────> data/processed/sft/*.jsonl
 ├─ train_lora.py
 │   ├─ response_loss.py
 │   └─ attention_backend.py
 │   └──────────────────> lifemate_runs/<RUN_ID>/{smoke,full}
 ├─ evaluate_dev.py
 │   └─ src/lifemate/{models,prompts,data,runner}.py
 └─ run_inference.py
     └─ src/lifemate/{models,prompts,data,runner}.py
         └──────────────> lifemate_runs/<RUN_ID>/submission.jsonl
```
