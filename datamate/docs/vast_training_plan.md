# Kế hoạch notebook DataMate trên Vast.ai

**Ngày lập:** 07/11/2026. **Mục tiêu:** tạo baseline DataMate có thể chạy public test, đo bằng dev tách theo doanh nghiệp, và chỉ fine-tune nếu QLoRA cải thiện điểm end-to-end. Kế hoạch này không thay notebook LifeMate.

## Trạng thái triển khai ngày 07/11/2026

- Đã tạo `notebooks/datamate_vast_train.ipynb` với action tách biệt cho prepare, build verified plans, augment, smoke, train/resume, evaluate, public predict và backup.
- Đã dựng reader CSV/XLSX/SQLite, business-group split, plan JSON allowlist, executor, scorer +1/−1/0, plan verification/resume, deterministic paraphrase augmentation, QLoRA trainer với early stop theo utility dev và offline runner.
- Đã chạy bước prepare trên dataset local: ghép đủ 3.000 train IDs, tạo 2.447 train / 553 dev theo sáu shop, xác nhận có 1.000 public questions và không thấy answer fields trong public. Prepare lập inventory file nhanh; reader bảng được nạp lazy theo shop khi build plan.
- Chưa chạy prompt-only GPU baseline, GPU smoke hoặc full training; máy local hiện không có CUDA. Việc đó phải chạy trên Vast sau khi upload bundle và kiểm tra môi trường.
- Đã tạo manual audit gate 100 verified plans trước full train. Cần đọc mẫu đó vì executor có thể thiếu rule đặc thù hoặc plan khớp gold do trùng hợp.
- Executor hiện là nền tảng tổng quát, chưa bao phủ hết các rule nghiệp vụ đặc thù trong rulebook. Cần đọc verified rate, xem log mismatch và đo grouped-dev trước khi dùng adapter làm ứng viên.
- Bundle Vast có thể tạo bằng `python3 datamate/scripts/build_vast_bundle.py`; bundle gồm code/notebook và hai archive dataset, không gồm raw đã bung, model cache hay run output.

## Quyết định thiết kế

### Các cách làm đã cân nhắc

1. **SFT thẳng câu hỏi → đáp án cuối:** triển khai nhanh, nhưng train chỉ có cặp hỏi–đáp, không có lời giải truy vấn; dễ học thuộc số của 30 shop và yếu trên cửa hàng/ngành chưa thấy.
2. **LLM sinh SQL/Python tự do:** linh hoạt, nhưng lỗi cú pháp, cột sai, quy tắc sai hoặc code không an toàn biến thành đáp án sai; khó kiểm soát format chính xác.
3. **LLM → kế hoạch JSON có schema → executor Python/SQL có allowlist:** tách hiểu tiếng Việt khỏi phép tính, kiểm tra từng điều kiện, chạy được trên dữ liệu mới. **Chọn phương án 3.** Notebook có baseline prompt-only và nhánh QLoRA để đo lợi ích, không giả định train chắc chắn sẽ tăng điểm.

### Mô hình khởi đầu

Dùng `Qwen/Qwen3-4B-Instruct-2507` làm ứng viên chính vì đội đã dùng cùng họ model trên Vast, đủ nhỏ để QLoRA trên RTX 3090 Ti 24 GB và nằm dưới ngưỡng SLM 15B trên portal. Model card ghi Apache-2.0; Qwen công bố bản 2507 ngày 06/08/2025, trước cutoff model 26/10/2026. Pin revision đầy đủ, không dùng `main` động. Chỉ thử model lớn hơn nếu 4B chạy end-to-end ổn và có đủ giờ để đo trên cùng dev.

Theo Rule DataMate đội đã cung cấp trước đó: inference phải offline, tổng model trong giới hạn 15B, và private run phải nằm trong ngân sách thời gian/hardware DataMate. Không mang giới hạn A30/30 giây của LifeMate sang nhánh này; đối chiếu lại DataMate Rule/Docker guideline trên portal trước khi đóng gói. Ngưỡng SLM ≤15B và leaderboard SLM-only được nêu trên portal.

Nguồn: [model card Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [thông báo phát hành Qwen3-2507](https://github.com/QwenLM/Qwen3), [portal DataMate](https://challenge.zalo.ai/portal/datamate/).

## Luồng notebook

Notebook mục tiêu: `datamate/notebooks/datamate_vast_train.ipynb`. Notebook điều phối các module nhỏ trong `datamate/src/datamate/`; không nhét parser, executor và scorer thành một cell dài. Trên Vast đặt project ở volume bền vững, ví dụ `/workspace/datamate_job/datamate/`, và lưu mọi artifact trong run ID riêng.

| Cell | Nội dung | Kết quả/điều kiện qua |
|---|---|---|
| 0. Giới thiệu và action | `prepare`, `smoke`, `train`, `resume`, `evaluate`, `public_predict`; hiển thị đường dẫn, run ID | Không chạy nhầm action; resume giữ nguyên run/config |
| 1. Cấu hình | model ID + revision, seed, LoRA, length, batch/accumulation, augmentation, đường dẫn | In và lưu `config.json` |
| 2. Phần cứng | `nvidia-smi`, PyTorch/CUDA, VRAM, disk, RAM, `torch.cuda.is_bf16_supported()` | Dừng sớm nếu GPU/disk không đủ; chọn BF16 hoặc FP16 |
| 3. Môi trường | cài requirements train đã pin; ghi version Python, torch, Transformers, PEFT, bitsandbytes | Có log cài đặt và runtime manifest |
| 4. Dataset | kiểm tra archive, checksum nội bộ, file cần thiết; bung `train.zip` vào `data/raw/train/` nếu chưa bung | Không ghi đè raw; tuyệt đối không trộn train với public |
| 5. Kiểm kê | đọc `questions.jsonl`, `answers.jsonl`, rulebook/calendar, README từng shop; lập catalog file/sheet/bảng/cột | 3.000 train IDs ghép đủ đáp án; public không có labels |
| 6. Dev split | chia theo toàn `business_id` và nhóm cấu trúc; ghi `split_manifest.json` cố định | Không có cùng shop hoặc augmentation cùng gốc ở hai phía |
| 7. Reader/cache | đọc CSV, XLSX nhiều sheet/header, SQLite; giữ provenance; cache bảng chuẩn hóa theo shop | Báo số dòng đã đọc, cột, ngày và lỗi parse; parse mỗi shop một lần |
| 8. Query plan labels | dựng/nhập plan JSON từ train; chỉ giữ plan thực thi hợp lệ và khớp gold theo scorer | Mỗi mẫu có `parent_id`, plan, answer type và nguồn; mẫu sai bị loại |
| 9. Augmentation | paraphrase/slot/composition trên train; kiểm lại bằng executor | Lưu riêng `aug_manifest.json`; validation không augment |
| 10. Tokenization | áp chat template; target chỉ là JSON plan, prompt được mask khỏi loss; đo p50/p95/p99 và truncation | Không cắt target; tự chọn length hợp lý |
| 11. Baseline chưa train | chạy planner prompt-only + executor trên dev cố định | Ghi utility, coverage, thời gian và lỗi theo nhóm |
| 12. Smoke QLoRA | 64–128 mẫu; microbatch/accumulation tạm đặt 1 để có đủ 50–100 optimizer steps | Loss hữu hạn, output parse được, VRAM không OOM, adapter lưu và resume được |
| 13. Full train | QLoRA với checkpoint theo epoch, log ETA, resume an toàn | Có `last` và adapter best, không ghi đè run khác |
| 14. Dev + early stop | sau mỗi epoch chạy end-to-end trên dev group-holdout | Early stop theo utility đáp án, không chỉ theo `eval_loss` |
| 15. So sánh ứng viên | baseline, QLoRA no-aug, QLoRA verified-aug; cùng split/scorer | Chọn ứng viên theo utility và nhóm lỗi, không theo cảm giác |
| 16. Public predict | nạp model đã chọn, chạy public test, output đúng schema | Đủ ID, không trùng, không thiếu, JSONL hợp lệ |
| 17. Lưu và bàn giao | copy adapter/model, config, logs, metrics, data hash, source revision ra volume/Drive | In đường dẫn tải về và checksum; không xóa instance trước khi sao lưu |

Notebook có thể chạy từ đầu hoặc theo `ACTION`. Chạy lại cùng `RUN_ID` chỉ dành cho `resume`; mỗi thử nghiệm khác model, seed, LR hoặc augmentation dùng ID mới.

## Tạo nhãn cho planner: bước không thể bỏ qua

Gold hiện chỉ có `question` và đáp án cuối, không có SQL, phép tính hay chuỗi suy luận. Vì thế không thể lấy answer làm plan train. Trước QLoRA, cần tạo supervision cho IR:

1. Định nghĩa JSON plan nhỏ, ví dụ `intent`, `measure`, `source`, `filters`, `period`, `group_by`, `sort`, `limit`, `answer_type`, `rule_refs`.
2. Tạo plan cho câu train bằng template hoặc teacher model chạy local, chỉ dùng **train split**; không gửi dữ liệu thi lên API ngoài và không đưa public/private questions vào quá trình sinh nhãn train.
3. Chạy plan qua executor trên business tương ứng; chỉ nhận plan khi output theo đúng type/normalization khớp gold. Ghi trace cột/file để review lỗi.
4. Teacher không được tự xác nhận answer của chính nó. Mẫu không kiểm chứng được thì bỏ khỏi SFT hoặc gán nhãn thủ công trên train.
5. Giữ một tập audit 100–200 plan có người xem, ưu tiên công nợ, trả hàng, ngày tương đối, list/table và `unanswerable`.

Nếu executor chưa chạy được, notebook vẫn cho phép `ACTION=prepare` và prompt-only baseline, nhưng **không chạy full SFT câu→đáp án** như thể đó là bằng chứng model biết tính dữ liệu.

## Augmentation có kiểm soát

Làm augmentation sau khi cố định group split; chỉ augment phần train. Không augment bằng cách sửa answer bằng tay.

- **Paraphrase:** tiếng Việt có dấu/không dấu, viết tắt (`dthu`, `sp`), từ địa phương, đảo trật tự câu; giữ nguyên các slot và plan.
- **Slot swap:** thay kỳ/ngày, top-k, mặt hàng/khách bằng giá trị có trong train business; executor tính lại answer.
- **Composition:** ghép điều kiện kỳ + measure + group/sort; thêm top-k, so kỳ, yes/no, list/table theo rulebook.
- **Hard negative/unanswerable:** chỉ tạo khi kiểm tra được sổ/cột/khoảng thời gian thật sự thiếu; không biến kỳ không có giao dịch thành unanswerable nếu dữ liệu cho phép kết luận 0.
- **Dedupe và kiểm soát tỷ lệ:** bắt đầu 1x synthetic verified trên mỗi mẫu gốc; so 0x/1x/2x. Giữ phân bố dev nguyên bản và không để paraphrase của cùng parent lọt qua split.
- **Kiểm tra:** answer type, JSON plan parse, executor result khớp gold, entity/date slot không đổi ngoài ý muốn. Tạo báo cáo số mẫu sinh/loại và lý do.

Không dùng public-test feedback để sửa nhãn/augment. Public submission chỉ dùng để đo theo cơ chế của BTC; nguồn câu hỏi, đáp án suy đoán hoặc kết quả leaderboard không trở thành training examples.

## Cấu hình train khởi điểm trên GPU 24 GB

Các giá trị dưới đây là **điểm xuất phát để benchmark**, không phải khẳng định tối ưu. Notebook phải log cấu hình hiệu lực.

| Tham số | Mặc định | Cách điều chỉnh |
|---|---:|---|
| Base | Qwen3-4B-Instruct-2507, revision pin | Thử 8B chỉ sau 4B; so trên cùng grouped dev |
| Weight quantization | 4-bit NF4 + double quant | Nếu bitsandbytes lỗi, dừng và sửa env; không âm thầm đổi kiểu |
| Compute precision | BF16 nếu GPU báo hỗ trợ | FP16 + GradScaler nếu BF16 không hỗ trợ; không bật đồng thời |
| Attention | PyTorch SDPA trước | Chỉ thử FlashAttention-2 khi cài đúng CUDA và smoke pass |
| LoRA | rank 16, alpha 32, dropout 0.05 | Sweep nhỏ rank 8/16/32; attention + MLP projections |
| Learning rate | `1e-4` | So `5e-5` và `2e-4` trên pilot; không đổi nhiều yếu tố cùng lúc |
| Microbatch | 1 | Tăng 2 nếu smoke còn đủ VRAM |
| Gradient accumulation | 16 | Effective batch 16 trên một GPU; chỉnh sau khi biết train sample count sau split |
| Max sequence | 2048 tokens | Dựa trên p95/p99; tăng 4096 nếu prompt có rule/schema cần thiết và không truncation |
| Epochs | tối đa 5 | Dừng sớm theo utility dev; không mặc định train đủ 5 |
| Scheduler/warmup | cosine / 5% | Gradient clipping 1.0; seed 2026 |
| Loss | chỉ loss trên JSON plan response | Mask prompt; không train suy luận dài hoặc output văn xuôi |
| DataLoader | dynamic padding, length grouping, 2 workers, pin memory | Cache tokenized examples; không nạp XLSX trong mỗi training step |

**AMP và tốc độ:** BF16 vừa là mixed precision vừa tránh nhu cầu loss scaling; trên RTX 3090 Ti phải xác nhận ở preflight. `gradient_checkpointing=True` giảm VRAM đổi lấy thời gian; bật cho QLoRA nếu cần context 2–4K. Bật TF32 khi hỗ trợ. `torch.compile`, sequence packing, DDP và FlashAttention không mặc định bật: chỉ benchmark riêng sau khi notebook đúng và tái lập được. Một GPU không cần DDP.

**Optimizer:** bắt đầu `paged_adamw_8bit` nếu bitsandbytes tương thích; có fallback `adamw_torch_fused`. Ghi rõ nhánh đã dùng. Không xóa cache GPU trong từng batch; dùng `save_total_limit=2`, giữ best + last.

## Early stopping và tiêu chí chọn model

- Dev cố định theo toàn doanh nghiệp, khoảng 20% shop nếu phân bố cho phép; không dùng public labels.
- Chỉ số chính: utility trung bình chính thức (đúng +1, sai −1, abstain 0) theo đúng evaluator/format; kèm exact-match, coverage, unanswerable recall, abstain rate và latency.
- Đánh giá full dev sau mỗi epoch. Early stop sau **2 lần đánh giá liên tiếp** không tăng ít nhất `0.01` utility; `MAX_EPOCHS=5`. Lưu best theo utility; `eval_loss` chỉ là chẩn đoán phụ.
- Do dev chỉ có vài business holdout, chọn model cũng phải xem từng business/schema/type; không nâng model chỉ vì một nhóm cải thiện trong khi các nhóm khác sụt mạnh.
- Sai bị −1 và abstain được 0: ngưỡng trả lời phải được hiệu chỉnh trên train-dev, không suy từ public test. `unanswerable` là nhãn có thể trả lời đúng, không đồng nghĩa với `abstain`.

## Thử nghiệm gọn, không sweep tràn lan

1. **A0:** prompt-only 4B, không train; chốt lỗi của parser/executor trước.
2. **A1:** QLoRA rank 16, LR `1e-4`, không augmentation, smoke rồi tối đa 5 epoch.
3. **A2:** giữ nguyên split, thêm 1x verified paraphrase/slot augmentation.
4. **A3:** chỉ nếu A1/A2 ổn: pilot LR `5e-5` hoặc `2e-4`, hoặc rank 8/32; thay một yếu tố một lần.
5. So score end-to-end và thời gian bằng cùng split/seed. Nếu LoRA không thắng prompt-only ngoài dao động giữa các shop, giữ baseline không train.

## Ước lượng thời gian

Giả định cùng máy Vast đã dùng trước đây: **RTX 3090 Ti 24 GB**, 1 GPU, còn ít nhất 20–25 GB disk trống. Tốc độ thực phụ thuộc p95 token length, driver, bitsandbytes và cache model; đây là khoảng dự trù, không phải cam kết:

| Công việc | Ước lượng |
|---|---:|
| Tải/cài model 4B khi cache lạnh | 15–45 phút, phụ thuộc mạng |
| Kiểm kê/parse và cache bộ train | 15–60 phút; cần đo trên Vast |
| Smoke | 10–30 phút |
| QLoRA full 3–5 epoch trên ~2.4K câu gốc và verified augmentation | khoảng 2–6 giờ GPU |
| Dev end-to-end theo epoch + chọn checkpoint | khoảng 0.5–2 giờ, tùy độ dài output và số dev shop |
| Một run từ cache lạnh tới metrics chọn được | **khoảng 4–8 giờ** sau khi code, plan labels và executor đã sẵn |

Nếu max sequence phải lên 4096, augmentation tăng nhiều hoặc dùng 8B, dự trù **6–12 giờ/run**. Công việc tạo/verifying plan labels và sửa parser/executor không nằm trong giờ train; có thể mất thêm 1–3 ngày nhóm làm việc và sẽ là phần khó đoán hơn GPU.

Notebook sẽ đo 100 optimizer steps đầu và một lát dev, rồi in ETA thực:

```text
ETA = thời gian tải/khởi tạo
    + số optimizer steps còn lại × median step time
    + số lần dev evaluation × thời gian dev vừa đo
    + thời gian save/checkpoint
```

Không chạy sweep toàn bộ grid trước phép đo này. Nếu smoke cho biết hơn 8 giờ/run, giảm độ dài prompt bằng retrieval rule/schema, giảm số epoch/pilot, không xóa bước kiểm chứng đáp án.

## Lịch khả thi tới hạn 16/11

| Ngày | Mốc cần có |
|---|---|
| 07–08/11 | Chốt contract JSON plan/scorer; catalog schema và split theo business |
| 09–10/11 | Reader/cache + executor cho doanh thu/thời gian; prompt-only dev baseline |
| 10–11/11 | Verified plan set, augmentation 1x, notebook preflight và smoke |
| 11–13/11 | Một full run QLoRA; early stop; tối đa một ablation hữu ích |
| 13–14/11 | Chạy public predict, sửa lỗi format/runtime; đóng băng model ứng viên |
| 14–15/11 | Kiểm tra Docker/offline, checksum, hướng dẫn và phục hồi artifact |
| 16/11 | Chừa buffer trước hạn BTC 23:59 GMT+7; không đợi phút cuối để upload |

Hạn cuối này theo [trang chủ challenge](https://challenge.zalo.ai/); kiểm tra portal trước lúc nộp vì BTC có thể cập nhật hướng dẫn.

## File dự kiến và cổng kiểm tra

- `datamate/notebooks/datamate_vast_train.ipynb`: UI điều khiển run, preflight, smoke/full/resume/eval/public.
- `datamate/src/datamate/`: module `readers`, `schema_catalog`, `query_plan`, `executor`, `augment`, `scoring`, `runner`.
- `datamate/requirements-train.txt`: bản pin dependencies train; inference dependency tách riêng.
- `datamate/outputs/runs/<RUN_ID>/`: logs, manifests, split IDs, token stats, metrics, configs và checkpoint tạm.
- `datamate/models/<RUN_ID>/best-adapter/`: adapter đã chọn; model base cache tách khỏi adapter.
- `datamate/result/submission.jsonl`: kết quả public; không đưa public test thành train data.

Gate trước khi full train: dataset/hash đúng; không có leakage business; plan labels được executor xác minh; `answer` được serialize đúng; GPU BF16/FP16 smoke qua; resume và backup hoạt động. Gate trước khi dừng Vast: tải best adapter, config, logs, reports và checksum về storage của đội; không xem `/workspace` là bản duy nhất.
