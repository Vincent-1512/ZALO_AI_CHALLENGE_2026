# LifeMate: kế hoạch nghiên cứu và nâng baseline toàn diện

Ngày: 06/11/2026. Trạng thái: kế hoạch, chưa triển khai và chưa có bằng chứng tăng điểm.

## 1. Mục tiêu và nguyên tắc lựa chọn
Tối đa hóa chất lượng agent dưới giới hạn BTC, chọn bằng thực nghiệm dev/holdout và kiểm tra A30. Không coi số epoch, số augmentation hay số kỹ thuật là bằng chứng chất lượng. Mọi tham số dưới đây là đề xuất thử nghiệm, không phải thông số tối ưu đã biết.

Giữ baseline hoạt động hiện tại làm đối chứng. Phát hành notebook Vast riêng; không thay notebook Colab thành notebook phụ thuộc Vast. Kết quả cuối gồm checkpoint/adapter được chọn, cấu hình inference, JSONL public, báo cáo dev/holdout, manifest phiên bản và bundle offline.

## 2. Luật và điều kiện phát hành
Nguồn tại workspace: docs/competition/lifemate.md, bản PDF thể lệ tiếng Việt và bản text trích xuất. Portal https://challenge.zalo.ai/portal/lifemate được mở ngày 06/11 nhưng không trả nội dung đọc được cho công cụ; không tuyên bố đã xác minh trực tiếp các cập nhật mới trên portal.

- Theo hướng dẫn LifeMate lưu trong dự án: hệ thống <6B tham số, LLM <5B; context <=16384 token; inference offline và xác định; tối đa 30 giây/mẫu trên một A30. Đếm cả thành phần model bổ sung/adapter một cách bảo thủ; lượng tử hóa không làm giảm số tham số.
- Mốc phát hành tài nguyên public trước 26/10/2026 và giấy phép phải kiểm tra theo revision/model cụ thể. Không tự thêm model mới chỉ vì nó mạnh hơn.
- Không đọc thư mục *_environment trực tiếp. Inference chỉ validate/call/get_tool_calls/get_state_hash. Không lấy get_state làm tín hiệu sửa prediction. Chỉ diagnostics train có thể dùng get_state theo hướng dẫn lưu tại dự án.
- History đã thực thi; không replay. Mỗi mẫu đúng một ToolEnv; giữ cả call thất bại trong output lấy từ môi trường.
- Không dùng nhãn hay dự đoán thủ công trên public/private để train/tune. Public inference chỉ được thực hiện sau khi chọn cấu hình bằng train-dev; không phân tích public để tạo hard case hoặc augmentation.
- Thể lệ 4.3 hạn chế truyền dữ liệu cuộc thi ra ngoài đội. Không gửi dữ liệu BTC sang API teacher/LLM bên ngoài theo mặc định. Nếu synthetic cần teacher, ưu tiên local, kiểm tra quyền sử dụng và yêu cầu BTC trước khi dùng dịch vụ ngoài. Model teacher không triển khai vẫn cần kiểm tra điều kiện tài nguyên huấn luyện, không mặc định được miễn.
- Tối đa 5 bài/ngày; giới hạn gói giải pháp và yêu cầu đóng băng theo Submission Guidance riêng. Bản tóm tắt lưu ghi hạn solution 16/11 23:59 GMT+7, Docker offline và tối đa 2 lần nộp solution. Xác minh lại hướng dẫn Docker và lịch trên portal trước phát hành.
- Chưa có đủ công thức/scorer BTC trong tài liệu đội lưu: diagnostics local không được gắn nhãn official score.

## 3. Những điểm yếu đã thấy trong code
- Notebook hiện chỉ train Qwen từ hai JSON train; chưa gồm ASR, rollout dev và public inference.
- SFT dùng transcript chuẩn, inference dùng ASR: có độ lệch phân phối đầu vào chưa đo.
- Training có eval_loss mỗi epoch nhưng không early stopping/load-best. Save theo steps và eval theo epoch chưa thích hợp cho load-best.
- Chưa có báo cáo lỗi theo loại: ASR, tên tool, args, missing info, stop, lỗi trạng thái, timeout.
- Context fitting hiện xóa history cũ; có thể mất tham chiếu quan trọng. Output max_new_tokens=192 chưa được kiểm kê độ dài target.
- Catalog public có tool mới (bản kiểm kê: 34 tool public không có trong train). Cần mô phỏng tool chưa thấy bằng dữ liệu train, không học thuộc tên.
- Parse lỗi có thể dẫn đến stop; cần phân biệt JSON hỏng với hành động stop hợp lệ.
- Chưa có bộ nhớ đệm ASR theo nội dung audio, giới hạn thời gian thực toàn mẫu, hay benchmark A30 end-to-end đầy đủ.

## 4. Dữ liệu và đánh giá trước khi train lớn — ưu tiên P0
1. Audit train JSON, schema, target length và transcript history; không mở environment. Giữ group split trước trajectory/augmentation, kiểm tra duplicate gần giống và nhóm yêu cầu cùng thực thể nếu có.
2. Giữ split 80/10/10 và seed=2026 làm đối chứng, ghi ID/hash. Báo cáo phân bố tool/steps/OOS/missing-info/history giữa split. Không thay split để làm đẹp điểm.
3. Thêm stress evaluation theo tool/app chưa gặp chỉ dùng train: fit trên phần không có nhóm tool được giữ lại và đánh giá riêng, tránh tuyên bố nó thay thế dev chính. Nếu nhóm hiếm quá nhỏ, báo cỡ mẫu thay vì ép split giả ổn định.
4. Chạy base ở hai chế độ dev: gold transcript và ASR transcript; chênh lệch giúp phân bổ ngân sách ASR/LLM. Gold transcript chỉ dành cho diagnostics, không dùng làm kết quả inference thực tế.
5. Metrics: state-match, call precision/recall/F1 diagnostics, OOS, ask_missing_info, JSON/schema failure, thiếu/thừa call, latency p50/p95/p99/max, timeout và errors. Báo theo nhóm độ dài/history/call count/unseen tool. Sai state là rủi ro lớn theo luật; chưa suy diễn trọng số scorer chính thức.
6. Khoảng tin cậy bootstrap theo sample cho ứng viên sát nhau. Holdout chỉ mở khi chốt; nếu đã dùng chọn model thì ghi nhận nó thành validation, không tiếp tục gọi là kiểm chứng độc lập.

## 5. Training: cấu hình và ma trận thử nghiệm — P1
Giữ Qwen3-4B-Instruct-2507 và PhoWhisper-medium hiện tại trước. Kiểm tra revision, license, tham số thực bằng config/weights; tổng cấu hình hiện ~4.02B +0.769B trước adapter, còn phải tính adapter/thành phần mới.

Cấu hình đối chứng cải tiến đề xuất:
- QLoRA NF4 + double quant; BF16 khi GPU hỗ trợ, FP16 có smoke kiểm tra NaN ở GPU khác.
- LoRA r=16, alpha=32, dropout=0.05; q/k/v/o/gate/up/down projections như hiện tại.
- LR=5e-5 cho ứng viên khởi đầu thận trọng, đối chứng với 1e-4 hiện tại. Đây là giả thuyết cần đo.
- Effective batch=16 (microbatch theo VRAM x accumulation); giữ batch hiệu dụng khi benchmark tốc độ.
- AdamW paged 8-bit, warmup_ratio=0.03, cosine, max_grad_norm=1.0, weight_decay=0.01 là ứng viên; ghi rõ khác biệt với default cũ.
- Max epochs=30 chỉ là trần an toàn, early stopping patience=8 lần eval không cải thiện, absolute min_delta=1e-3 trên eval_loss (cấu hình được). Không bảo đảm sẽ chạy gần 30; ưu tiên thời gian thử LR hơn đẩy tất cả đến trần.
- Eval và save cùng chu kỳ: khoảng một phần tư epoch, tính theo optimizer update steps thực tế, tối thiểu 1; giữ best checkpoint và các checkpoint phục hồi. Patience=8 ở lịch này tương đương khoảng hai epoch; patience được tính bằng số lần đánh giá, không phải số epoch. Có tùy chọn eval mỗi epoch nếu eval quá tốn; notebook luôn in lịch và ý nghĩa patience.
- Dùng response-only loss; kiểm chứng chuẩn hóa loss với accumulation và độ dài target khác nhau, không mặc định custom loss đã đúng do train không crash. Token-weighted vs record-weighted phải được định nghĩa nhất quán.
- Bật load_best_model_at_end theo eval_loss, nhưng giữ vài checkpoint ứng viên để chọn cuối bằng rollout dev. Không mặc định checkpoint cuối tốt nhất. Giới hạn dung lượng checkpoint và bảo vệ best khỏi cleanup.
- Context: chọn từ thống kê sau prompt compression, trần 16384 gồm prompt + generated output. 8192 đối chứng; thử 12288/16384 khi có lỗi thiếu context và A30 còn thời gian. Không cắt gold target, request, schema bắt buộc hoặc current steps để che lỗi độ dài.
- Output token budget suy ra từ p99/max target và đo runtime; thử 192/256/384 chỉ khi độ dài target/parse lỗi cho thấy cần.

Thứ tự chạy tuần tự, không Cartesian grid:
A0: base, không SFT; A1: cấu hình cũ r16/LR1e-4 làm đối chứng.
A2: r16/LR5e-5 + checkpoint/early stopping mới; cùng split và dữ liệu.
A3: LR2e-5 hoặc r32/alpha64 chỉ khi lỗi/learning curves cho thấy còn cần; thay một nhóm yếu tố mỗi lần.
A4: dữ liệu ASR-mixed trên cấu hình thắng.
A5: schema permutation trên cấu hình thắng.
A6: kết hợp chỉ các cải tiến có bằng chứng; xác nhận hai seed nếu ngân sách cho phép.
Thử rsLoRA r32 là nhánh sau cùng, không thay cùng lúc với LR/data. DoRA, DPO/GRPO, distillation, multi-model ensemble không bật mặc định vì chưa có reward/pair dữ liệu đủ tin cậy và có chi phí/giới hạn inference.

## 6. Data augmentation có kiểm soát — P1/P2
Tất cả augmentation chỉ sau split, trên train; mỗi record có parent_id, transform, seed, kiểm tra schema và nhãn. Dev/holdout luôn nguyên bản.

### Ưu tiên 1: ASR-mixed input
- Chạy PhoWhisper trên audio train và history train, cache theo hash audio + model revision + cấu hình. Tạo song song transcript chuẩn và transcript ASR cho agent, giữ label gốc.
- Đối chứng clean-only với mixed 75% clean/25% ASR, rồi 50/50 nếu cần. Lấy mẫu theo original sample để số bản sao không âm thầm đổi effective epoch.
- ASR mất/sai thực thể có thể khiến gold không còn suy ra được. Đánh dấu tên/số/ngày/tiền bị thay, kiểm tra thủ công chỉ trên train hoặc tiêu chí tự động bảo thủ; không ép mọi transcript sai nặng phải sinh gold không có căn cứ. Giữ noisy cases chỉ khi nhãn vẫn hợp lý hoặc như một nhánh thử nghiệm có báo cáo riêng.

### Ưu tiên 2: schema-order robustness
- Hoán vị thứ tự available_tools, thứ tự key object nếu tương đương; không xáo array có thứ tự ngữ nghĩa, enum hay danh sách kết quả. Không đổi tool/arg names.
- Tối đa một biến thể mỗi mẫu ở thí nghiệm đầu, sampling weight giữ tổng cân bằng.

### Ưu tiên 3: paraphrase có ràng buộc
- Chỉ train, bảo toàn entities, phủ định, số lượng, đơn vị, quan hệ thời gian, timezone, references và intent. Kiểm tra consistency; mẫu không chắc thì bỏ.
- Khởi đầu <=0.5 biến thể/mẫu trung bình; không nhân dữ liệu 10 lần và xem là thêm thông tin.
- Chưa tạo synthetic tool-result/ID/final-state không qua simulator hợp lệ. Không thay giá trị entity mà giữ nguyên nhãn cũ. Không xóa args rồi tự gán ask_missing_info nếu không chắc dữ liệu còn có thể tra cứu.

### Nhánh sau: hard cases và âm thanh
- Phân loại lỗi trên train out-of-fold hoặc tập chẩn đoán riêng; không biến lỗi dev/holdout thành mẫu train rồi tiếp tục báo cùng dev đó.
- Oversample stop/OOS/missing-info nhẹ (tối đa 2x ban đầu), đánh giá over-stop/over-ask; không cân bằng lớp mù quáng.
- Audio speed/noise chỉ dùng cho nhánh ASR fine-tune hoặc tạo transcript train qua ASR, không phải augmentation trực tiếp của text LLM. Chỉ làm nếu oracle-vs-ASR cho thấy ASR là nút thắt; nguồn noise phải có quyền sử dụng.

## 7. Prompt và chính sách tool — P1
- Đồng nhất serialization train/inference và dự toán token thật. Nén JSON/schema không loại bỏ mô tả hoặc constraints cần thiết.
- Giữ current request, context thời gian/timezone, history tham chiếu và các kết quả tool liên quan. Thử history selection dựa luật bảo thủ với dev đối chứng; không dùng summary sinh tùy ý có thể mất entity. Thêm test history đại từ, phủ định và sửa yêu cầu.
- Giữ đầy đủ tools được cài ở baseline. Retrieval/top-k tools chỉ là nhánh sau: đo recall gold tool trên dev trước, luôn có fallback đầy đủ; không thêm embedding model chưa tính vào giới hạn tham số.
- Phân biệt stop/OOS và ask_missing_info; yêu cầu lookup ID qua tool thay vì đoán. Chỉ dùng observation từ tool thật trong lượt hiện tại.
- Schema validate trước thực thi và một lần sửa có giới hạn thời gian; JSON hỏng có cơ chế retry rõ ràng, không tự coi là stop. Schema repair không được thực thi call để thử trạng thái.
- Constrained decoding: thử giới hạn JSON/action/tool-name theo schema động nếu backend hỗ trợ ổn định; kiểm tra giá trị unicode, enum, optional/nullable và latency. Chỉ bật khi giảm lỗi trên dev, không chỉ vì output luôn parse được.
- Giới hạn số bước 8 hiện tại là cấu hình baseline, cần kiểm tra Guidance có quy định riêng không; chọn theo train trajectory và latency. Không cắt cứng vì gold train tối đa 6: private có thể dài hơn.

## 8. Inference và tốc độ — P1
- ASR small vs medium: so WER/CER diagnostics cùng tác động task, không chọn chỉ bằng WER. Batch audio theo sample/độ dài và cache history theo nội dung; cache phải đúng revision và reset khi cấu hình đổi. Không cache đáp án theo ID hay dùng dữ liệu chấm trước.
- Qwen greedy deterministic; adapter merge offline thành model riêng nếu kiểm thử output đạt yêu cầu và nhanh hơn. Không tính quantization là giảm tham số. Merge LoRA sau QLoRA có thể đổi output; phải đánh giá đúng artifact cuối.
- Đo BF16/FP16 vs 4bit inference trên A30; không mặc định 4bit nhanh hơn. Một model LLM và ASR cùng resident nếu VRAM đủ; kiểm tra memory spike theo history dài.
- KV/prefix cache chỉ khi cache-key và backend thực sự đúng; không mang KV khác sample. Không bật backend mới trước khi có smoke correctness.
- Timeout toàn mẫu bao gồm ASR+policy+tool+repair; log theo stage và các trường hợp vượt 30s. Timeout thực thi cứng/worker cần thiết kế tránh làm hỏng ToolEnv. Mục tiêu nội bộ headroom p95 <24s chỉ là mục tiêu, vẫn kiểm tra từng mẫu và max.
- Vast GPU nhanh không thay bằng chứng A30. Nếu chưa có A30, đánh dấu chưa xác nhận tính hợp lệ runtime.

## 9. Notebook Vast: cấu trúc cụ thể
0. Hướng dẫn, giới hạn, run state và chi phí ước tính từ smoke; không tự thuê GPU hay thanh toán.
1. Một cell cấu hình: PROJECT_ROOT, DATASET_ZIP/ROOT, RUN_ROOT/RUN_ID, ACTION, seed, model revisions, precision, LoRA/LR/batch/context/output budget, augmentation, eval/save/early stop, candidate.
2. Môi trường: GPU, CUDA, kernel NF4 forward/backward/optimizer, storage; giữ PyTorch CUDA có sẵn, pin dependencies và ghi phiên bản.
3. Source/dataset checksum, unzip an toàn và kiểm tra file; tách source, dataset, weights, generated SFT, run artifacts.
4. Audit/split + thống kê; invariant train/dev/holdout không overlap.
5. Download model đúng revision và offline-load verification.
6. ASR diagnostics/cache tùy chọn; tạo SFT theo preset augmentation và ghi fingerprints.
7. CPU token preflight mọi split cần dùng; đo max/p95 prompt/target, không bỏ lỗi âm thầm.
8. Smoke longest-first + tập smoke đa dạng, gồm save/reload/resume ngắn.
9. Full/resume có guard cấu hình/data; tiến trình chạy bằng log file và cơ chế tách khỏi terminal phù hợp môi trường. Không phụ thuộc biến còn trong kernel.
10. Learning curves, checkpoint và thông báo lý do stop; khôi phục best.
11. Rollout dev base/ứng viên; error breakdown; bảng so sánh latency/chất lượng. ACTION riêng để không vô tình train lại.
12. Chọn ứng viên có policy rõ ràng, xác nhận holdout một lần; không tự gắn bất kỳ adapter cuối nào vào public.
13. Public inference từ artifact đã chọn; append/resume chỉ khi fingerprints khớp, unique ID, journal/atomic output tránh duplicate/truncated last line. Không đổi trọng số trong vòng public inference.
14. Validate submission IDs/order nếu Guidance yêu cầu/schema/UTF-8/count từ input thực tế, state_hash lấy môi trường, đủ calls kể cả thất bại. Không hardcode 634 thay kiểm tra input, nhưng báo khi khác dataset đã audit.
15. Export JSONL + training/inference/config/model manifests/licenses/checksums và hướng dẫn Docker. Công bố riêng log không chứa dữ liệu nhạy cảm; không tự công khai dataset Drive.

## 10. Kỷ luật chọn thí nghiệm và ngân sách
- Notebook mặc định một run, không tự train hàng chục cấu hình. Sweep chỉ chạy khi người dùng bật với MAX_RUNS/MAX_GPU_HOURS; ước lượng trước từ thời gian smoke/step, eval và số token.
- Theo dõi bảng experiment: config/data/code hash, best step, eval loss, dev diagnostics, latency, GPU hours, lý do giữ/bỏ.
- Giữ cải tiến nếu nâng chỉ tiêu task hoặc giảm lỗi quan trọng mà không làm timeout/OOS xấu đi; ghi tradeoff khi metrics xung đột. Không dựng công thức official chưa biết; chốt rule chọn ứng viên trước khi xem kết quả.
- Local experiments không tiêu lượt submission. Public dùng xác nhận ứng viên đã khóa, không dùng 5 lượt/ngày như bộ tối ưu hyperparameter.
- Sau khi chọn recipe, có thể cân nhắc train+dev refit với epoch/steps cố định đã chọn và giữ holdout để kiểm chứng. Không dùng holdout làm early stopping rồi coi nó là test; không mặc định refit chắc tốt hơn.

## 11. Kiểm chứng triển khai và bàn giao
- Test loss so với full-logits reference, prompt mask, batch padding, accumulation và tiny save/reload/resume.
- Test split/augmentation/invariant labels, schema/context budget, parse/stop/missing-info, JSONL interrupted-write/resume.
- Test early-stop theo eval count, best preservation, config mismatch, optimizer/scheduler resume.
- Test notebook syntax/paths, output guard, action idempotence và bundle checksum; CPU tests local không thay CUDA integration.
- CUDA smoke và A30 end-to-end cần thực chạy mới kết luận. Không tuyên bố điểm cải thiện trước đo.
- Bàn giao notebook Vast, runbook, patch source nhỏ cho gói đã upload, archive đầy đủ mới và SHA256. Không upload hay nộp BTC thay người dùng nếu chưa được yêu cầu.

## 12. Nguồn nghiên cứu và cách sử dụng
- Luật đã lưu: docs/competition/lifemate.md và docs/competition/regulations/Zalo_AI_Challenge_2026_The_le_VI.pdf. Portal https://challenge.zalo.ai/portal/lifemate không trả nội dung đọc được trong lần kiểm tra này.
- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 — model card/giấy phép; không suy ra điểm LifeMate.
- https://huggingface.co/vinai/PhoWhisper-medium — mô hình ASR tiếng Việt và giấy phép.
- https://arxiv.org/abs/2305.14314 — QLoRA/NF4 tiết kiệm VRAM; không chứng minh LR/rank nào tối ưu cho LifeMate.
- https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/callback — EarlyStoppingCallback và cơ chế callback; dùng đúng phiên bản đang pin.
- https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer — TrainingArguments/checkpoint; kiểm tra logic đồng bộ save/eval/load-best khi triển khai.
- https://huggingface.co/docs/peft/v0.17.1/developer_guides/lora — kiểm tra kỹ thuật LoRA theo bản pin trước dùng; không mang API mới từ docs latest vào môi trường cũ.
- https://arxiv.org/abs/2307.16789 — nghiên cứu tool-use làm cơ sở xem xét dữ liệu/schema; không mặc định được dùng dataset/teacher của paper trong cuộc thi.

Đề xuất augmentation, tham số, lịch thí nghiệm và ngân sách trong tài liệu là thiết kế riêng cho dự án dựa trên code/dataset đã kiểm kê, cần ablation để xác nhận. Không có nguồn nào ở trên bảo đảm leaderboard cao nhất.
