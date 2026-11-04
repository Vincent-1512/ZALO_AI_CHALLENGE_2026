# Kế hoạch LifeMate — Zalo AI Challenge 2026

Ngày lập: **04/11/2026, giờ Việt Nam (GMT+7)**. Đội: **5 người**, có thể thuê GPU. Đây là kế hoạch đề xuất để triển khai; chưa phải báo cáo đã chạy thử hay đã huấn luyện mô hình.

**Cập nhật sau khi nhận dataset và các hướng dẫn đầy đủ:** đọc [chiến lược nâng chất lượng LifeMate](lifemate_strategy.md) trước khi triển khai. Tài liệu bổ sung đánh giá app-held-out, thiết kế thí nghiệm và tiêu chí chọn mô hình theo dữ liệu thực.

## 1. Kết quả cần đạt và điểm xuất phát

Mục tiêu là có hệ thống LifeMate hợp lệ sớm, sau đó tăng điểm bằng các thí nghiệm đo được, và nộp giải pháp tái lập được trước hạn. Không có kế hoạch nào bảo đảm điểm tuyệt đối hoặc thứ hạng; các quyết định mô hình cuối cùng phải dựa trên dữ liệu và thời gian chạy thực tế.

Workspace có README, thể lệ VI/EN và gói `data/archives/train+public_test.zip` bản **v3.2.0**. Gói đã được giải nén vào `data/raw/train+public_test/`: **1.545 mẫu train**, **634 public test**, khoảng **1,1 GB** sau giải nén (zip tải về khoảng **836 MB**). Đếm được 5.036 train WAV và 1.933 public WAV vì audio của các lượt lịch sử cũng được tính. Chưa có mã agent hoặc checkpoint. Không đọc trực tiếp bất cứ file nào trong `*_environment/`.

Đầu vào của hệ thống: audio tiếng Việt, ngữ cảnh người dùng, lịch sử hội thoại, danh sách app/công cụ, môi trường thực thi BTC. Đầu ra: `submission.jsonl`, mỗi dòng gồm `id`, `tool_calls`, `state_hash` theo schema BTC.

Ba mốc nội bộ:

- **Trong 48 giờ từ lúc nhận đủ kit + GPU:** một bản chạy toàn bộ public test, đúng định dạng, có bài gửi portal được tiếp nhận; điểm đầu tiên là mốc đo, chưa đặt điểm tùy ý.
- **14/11:** gói Docker ứng viên đã chạy offline trên máy sạch, link và checksum sẵn sàng.
- **16/11 trước 12:00:** mục tiêu nộp chính thức lần đầu nếu form đã mở; dành buổi chiều cho xác nhận hoặc thay thế có kiểm soát.

## 2. Luật đã xác minh trực tiếp từ portal

Nguồn: [LifeMate](https://challenge.zalo.ai/portal/lifemate), các tab Description, Evaluation, Rule, Submission Guidance; kiểm tra ngày 04/11/2026. [Thể lệ VI](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf) bổ sung yêu cầu tài liệu và tài khoản.

| Nội dung | Yêu cầu chính thức | Hệ quả cho kế hoạch |
|---|---|---|
| Kích thước | Toàn hệ thống **<6B** tham số; LLM **<5B** | Đếm cả ASR, LLM và thành phần mô hình khác; lượng tử hóa không làm giảm số tham số |
| Context | Tối đa **16.384 token** | Giới hạn tại runtime, tính cả prompt, lịch sử, schema, kết quả công cụ và phần sinh |
| Máy chấm | 1 NVIDIA A30, CPU Xeon Gold 6442Y 16 core, RAM 64 GB | Đo lại trên A30; GPU thuê mạnh hơn không chứng minh đạt hạn trên máy chấm |
| Thời gian | **30 giây/mẫu**; timeout hoặc lỗi thực thi được 0 | Đo từ đầu đến cuối; làm rõ cách tính cold start với BTC |
| Inference | Hoàn toàn offline; không dịch vụ/API bên ngoài | Gói sẵn trọng số, tokenizer, dependency; không tải tài nguyên khi chạy |
| Tài nguyên | Dữ liệu công khai và mô hình nguồn mở phát hành **trước 26/10/2026**, giấy phép phù hợp | Lưu nguồn, revision, ngày phát hành và giấy phép cho từng tài nguyên |
| Dữ liệu tổng hợp | Có thể dùng dịch vụ/mô hình khác tạo dữ liệu trong phát triển; không dùng chúng lúc inference | Chỉ dùng nguồn được phép; không gửi dữ liệu cuộc thi sang dịch vụ ngoài khi chưa có quyền |
| Test data | Cấm gán nhãn public/private test để train hoặc tune, bằng người hay dịch vụ/mô hình ngoài | Phát triển trên train và validation tách từ train; leaderboard dùng để đối chiếu hạn chế |
| Môi trường | Không đọc/copy/parse `<split>_environment/`, kể cả `state.json` | Chỉ tương tác qua `zaic_tools`; không dò dữ liệu trạng thái qua filesystem |
| ToolEnv | Đúng **một ToolEnv/mẫu**; mọi lần gọi lên nó phải xuất hiện trong bài nộp | Không clone, reset hay thử nhiều nhánh môi trường để chọn kết quả |
| API cho inference | Chỉ `validate`, `call`, `get_tool_calls`, `get_state_hash` | Không `get_state`, `states`, `Executor`, `Simulator`; `get_state` chỉ được dùng cho train |
| Tái lập | Seed cố định; ASR/LLM giải mã xác định | Greedy/temperature 0 theo backend; đo tính ổn định khi chạy lại toàn pipeline |
| Chấm điểm | Hash cuối sai → mẫu 0; tool-call match cũng được chấm. Mỗi call chỉ ghép một lần; fuzzy cần token-F1 >0.6 sau lowercase và Unicode NFC | Tối ưu state correctness trước, giảm call dư/lỗi schema sau; dùng evaluator BTC làm điểm chính |
| Ngoài phạm vi | Gold tool list rỗng chỉ đúng khi output rỗng | Phải xử lý out-of-scope, không ép mọi câu thành hành động |
| Thiếu thông tin | `sys.ask_missing_info` được chấm như công cụ khác | Không tự bịa đối số; có thể hỏi sau các bước hợp lệ trước đó |
| Match thứ tự | Thứ tự không dùng để ghép tool-call, nhưng vẫn ảnh hưởng state khi replay | Thực thi theo dependency; không xáo call trong môi trường |
| Tái chấm/hoà điểm | BTC replay `tool_calls` bằng `zaic_tools`; nếu khác, kết quả BTC là cuối. Hoà điểm: tính lại F1 call, bỏ kiểm tra hash; nếu vẫn hoà, latency trung bình thấp hơn thắng | Khóa image, kit, seed và runtime; giữ trace để tái lập |

A30 có 24 GB VRAM theo [NVIDIA](https://www.nvidia.com/en-gb/data-center/products/a30-gpu/). Cần chừa bộ nhớ cho ASR, KV cache và runtime; không xem 24 GB là dung lượng chỉ dành cho LLM.

### Lịch và cách nộp đã xác minh

| Mốc | Việc bắt buộc |
|---|---|
| Đến 16/11 | Chạy public test và nộp JSONL qua **Submission**; tối đa 5 lần/ngày, quota reset cuối ngày |
| **16/11, 23:59** | Nộp Docker package hoàn chỉnh gồm training code, inference code và data qua **Solution Submission**; gửi **checksum + link Google Drive công khai** để BTC tải |
| Gói giải pháp | Tối đa **2 lần nộp**; **chỉ lần cuối được BTC đánh giá** |
| **20/11, 09:00** | Theo Submission Guidance trong portal đã đăng nhập, BTC phát hành private test; chạy giải pháp đã nộp và đóng băng |
| **23/11, 23:59** | Nộp kết quả JSONL private qua Submission |
| 02/12 | Công bố top 2 mỗi nhánh |
| 12/12 | Công bố kết quả chung cuộc |

Trang chủ công khai vẫn ghi 11:00 cho private test, nhưng Submission Guidance trong portal đã đăng nhập ghi **09:00**. P1 theo dõi portal; cả đội sẵn sàng từ 08:30.

Thể lệ còn yêu cầu mô tả phương pháp và hướng dẫn chạy. Sau khi upload gói lên Google Drive, P2 tải lại và kiểm checksum trước khi nộp.

## 3. Chiến lược kỹ thuật đề xuất

| Hướng | Ưu điểm | Chi phí/rủi ro | Quyết định đề xuất |
|---|---|---|---|
| ASR có sẵn → LLM 4B có sẵn → tool loop | Có bản đầu nhanh, tách được lỗi nghe và lỗi hành động | Cần tối ưu schema, lịch sử, vòng gọi tool | **Làm đầu tiên** |
| Cùng pipeline, fine-tune LLM và ASR có chọn lọc | Học đúng kiểu dữ liệu cuộc thi; kiểm soát ngân sách | Tốn GPU, dễ overfit nếu chia dữ liệu sai | **Nhánh nâng điểm chính** sau baseline |
| Mô hình audio end-to-end hoặc nhiều agent/ensemble | Có thể có tiềm năng | Khó kiểm soát tổng tham số, latency, container và train trong thời gian ngắn | Chỉ khảo sát nếu hai hướng trên đã ổn và còn thời gian; không đặt trên đường găng |

Đề xuất mặc định: **PhoWhisper-medium + Qwen3-4B-Instruct-2507**, rồi so PhoWhisper-large nếu ASR là nút thắt. Đây là ứng viên cần benchmark, chưa phải mô hình thắng được chứng minh.

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507): model card ghi 4.0B, Apache-2.0, non-thinking. Dù model hỗ trợ context lớn hơn, bài thi vẫn phải khóa ≤16.384 token.
- [PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium) và [PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large): ứng viên ASR tiếng Việt, BSD-3-Clause. Cần đếm tham số chính xác từ revision tải về và cộng với toàn hệ thống trước khi chốt.
- Không tải model theo `latest` cho gói nộp. Lưu revision cụ thể, hash, tokenizer/processor và license; xác nhận revision đủ điều kiện thời gian.
- Bắt đầu bằng inference đơn giản với backend hỗ trợ model. Chỉ chuyển sang backend tăng tốc sau khi có số đo; không thay đồng thời model, backend và dữ liệu rồi suy diễn nguyên nhân cải thiện.

### Luồng xử lý một mẫu

1. Đọc input theo schema BTC; dùng đúng context thời gian/múi giờ của mẫu, không dùng ngày giờ máy chủ thay thế.
2. Nhận dạng audio hiện tại và lịch sử audio nếu schema thực tế yêu cầu. Giữ ngữ nghĩa số tiền, tên người, địa chỉ, phủ định và thời gian.
3. Dựng prompt chứa yêu cầu, context, lịch sử và schema của app được cài. Lịch sử không được tự động replay thành hành động mới; kiểm tra quy ước BTC.
4. Tạo đúng một ToolEnv cho mẫu theo kit; model chọn một bước hành động hoặc dừng/hỏi thêm.
5. Kiểm tra tool và arguments theo schema. Gọi tool hợp lệ qua API cho phép; đưa kết quả quan sát vào lượt sau. Không bịa `device_id`, `contact_id`, số dư hay kết quả công cụ.
6. Dùng giới hạn thời gian, token và số bước đã đo trên validation. Không gọi lại một thao tác thay đổi trạng thái chỉ vì lỗi parse ở bước kế tiếp. Chính sách retry phải xét tính lặp lại an toàn của từng tool.
7. Kết thúc bằng tool-call log từ môi trường và `get_state_hash()`; giữ đúng thứ tự thực thi, kể cả các lần gọi lỗi theo contract kit. Không tự lọc call thừa ra khỏi output sau khi đã thực thi.
8. Ghi đúng một dòng JSON cho mẫu. Log chẩn đoán để tệp riêng; không lẫn vào JSONL hoặc stdout dùng làm output.

Các ngưỡng đề xuất để đo: context vận hành 8K nếu đủ, tối đa 16K; timeout mềm khoảng 25 giây và dư thời gian xuất kết quả; số bước khởi điểm 8–12. **Không hard-code các ngưỡng này trước khi thống kê số bước, audio và schema của train.** Dừng sớm chỉ là bảo vệ vận hành, không bảo đảm mẫu đúng điểm.

## 4. Phân công 5 người

Gán tên thật vào P1–P5 trong ngày 04/11. Mỗi đầu việc có một người chịu trách nhiệm và một người kiểm tra.

| Người | Trách nhiệm chính | Đầu ra | Người kiểm tra |
|---|---|---|---|
| **P1 — tích hợp/đội trưởng** | Chốt luật, contract, nhánh release, tài khoản và nộp bài; giải quyết việc chặn đường găng | Hợp đồng dữ liệu, bản release, receipt nộp bài, nhật ký quyết định | P5 |
| **P2 — dữ liệu/đánh giá** | Kiểm kê train, chia validation, evaluator BTC, error analysis | Split cố định, baseline report, error buckets, bảng thí nghiệm | P1 |
| **P3 — ASR** | Baseline ASR, đo lỗi thực thể, fine-tune nếu cần | Module ASR, transcript train/dev, benchmark accuracy/latency | P2 |
| **P4 — LLM/agent learning** | Prompt/schema, chuẩn bị trajectory train, LoRA, chọn checkpoint | Model agent và ablation report | P2 |
| **P5 — runtime/Docker** | Tool adapter, offline runner, latency, container, reproducibility | Runner, validator, Docker archive, manifest/checksum | P1 |

P4 sở hữu chính sách quyết định của agent; P5 sở hữu thực thi và ghi log môi trường. P1 cố định interface giữa chúng để tránh sửa chồng cùng file. P2 đánh giá độc lập, không để người train tự chọn metric có lợi.

Nhịp làm việc đề xuất: đầu ngày 15 phút chốt đầu ra, giữa ngày báo blocker, cuối ngày 20 phút xem metric và chọn thí nghiệm hôm sau. Giờ cụ thể điều chỉnh theo lịch 5 người; chưa giả định ai làm full-time. Mỗi người tối đa một việc chính đang làm. Sau 60 phút bị chặn phải báo P1 để đổi hướng hoặc hỗ trợ.

## 5. Đường găng và lịch từng ngày

Đường găng: **quyền dữ liệu → đọc kit → contract + evaluator → pipeline hoàn chỉnh → public hợp lệ → tối ưu dựa trên validation → Docker offline → nộp và xác nhận**. Train không được chặn việc tạo baseline/container.

| Ngày | P1 | P2 | P3 | P4 | P5 | Điều kiện kết thúc ngày |
|---|---|---|---|---|---|---|
| **04/11** | Chốt đội, quyền truy cập, nguồn luật, contract v0 | Audit kit, đếm mẫu, chia train/dev | Chạy ASR vài mẫu train | Chạy Qwen trên ví dụ train theo schema | Dựng môi trường GPU, ToolEnv mẫu, Docker tối thiểu | Có kit hợp lệ; một mẫu đi hết audio → tool → JSONL; nếu chưa có kit, ghi blocker cụ thể |
| **05/11** | Tích hợp baseline | Chạy evaluator và báo cáo lỗi đầu | So medium/large trên cùng dev | Prompt tool loop, thiếu thông tin/OOS | Chạy full public, kiểm tra id/schema/offline | JSONL đủ mẫu, không id trùng; bài public đầu được portal tiếp nhận trong mục tiêu 48h |
| **06/11** | Khóa contract v1, chốt nhánh train | Đo nhóm lỗi, audit leakage | Lỗi số/tên/giờ, latency | Tạo dữ liệu next-action từ train; LoRA smoke run | Chạy container đầu trên máy sạch | Có score baseline chính thức và số đo local; biết 3 nhóm lỗi lớn nhất |
| **07/11** | Ưu tiên theo điểm mất | Kiểm tra split/counterexamples | Quyết định có train ASR không | LoRA đợt 1 trên train subset | Profile A30/24GB, bộ nhớ và tool loop | Checkpoint đầu được đánh giá end-to-end; có phương án chạy phù hợp A30 |
| **08/11** | Review ứng viên | Ablation cùng dev split | ASR train thử nếu có transcript hợp lệ | Train cấu hình tốt nhất có giới hạn | Giảm overhead/schema/context | Chọn baseline hoặc fine-tuned theo score + latency, không theo loss đơn lẻ |
| **09/11** | Tích hợp bản tốt | Kiểm tra lỗi đa bước và lịch sử | ASR cải thiện thực thể | Dữ liệu nhiễu ASR từ train; sửa policy lỗi nổi bật | Kiểm tra ghi call/hash và timeout | Candidate v1 vượt baseline trên dev hoặc giải thích được vì sao giữ baseline |
| **10/11** | Thu hẹp nhánh thử nghiệm | Đánh giá grouped dev đã cố định | Chốt ASR tạm | So checkpoint và prompt cuối | Chạy full container offline | Có 1 ứng viên chính + 1 dự phòng tương thích cùng runner |
| **11/11** | Review luật/model ledger | Kiểm tra OOS, missing info, phủ định | Kiểm tra phương ngữ nếu có nhãn | Sửa tối đa 1–2 lỗi lớn | Chạy lại toàn pipeline, đo độ lệch | Báo cáo ổn định, peak VRAM, p50/p95/p99/max, số timeout |
| **12/11** | Chốt phạm vi release | P2 mở holdout lần đầu cho ứng viên đã chốt trên dev | Bàn giao trọng số/revision | Chốt recipe train; nếu refit phải còn cửa sổ kiểm tra | Chạy máy sạch, không cache/mạng | Có đánh giá holdout lần đầu; mọi dependency và weights có trong image, không còn đường dẫn máy cá nhân |
| **13/11** | Freeze ứng viên nội bộ | Audit metric/coverage cuối | Tài liệu ASR | Tài liệu agent/train | Build release candidate | Không thay kiến trúc/model mới sau mốc này; thay đổi sửa lỗi phải đánh giá lại |
| **14/11** | Chuẩn bị form, link, checksum | Kiểm tra từ archive đã tải lại | Hỗ trợ kiểm tra | Method report hoàn tất | Save/load archive; xác nhận máy sạch | Gói sẵn sàng nộp; nếu form cho phép nộp sớm thì nộp, nếu chưa mở thì giữ artifact bất biến |
| **15/11** | Tổng duyệt và kiểm tra quyền link | Review độc lập manifest/receipt nháp | Trực sửa lỗi | Trực sửa lỗi | Đo toàn bộ public bằng đúng image | Có ứng viên dự phòng, log và checksum trùng bản lưu |
| **16/11** | Nộp lần 1 mục tiêu trước 12h; xác nhận | Đối chiếu hồ sơ | Trực | Trực | Hỗ trợ tải/chạy nếu BTC yêu cầu | Có bằng chứng tiếp nhận; chỉ dùng lần 2 nếu thật sự cần, mục tiêu trước 18h; hạn cứng 23:59 |
| **17–19/11** | Theo dõi BTC, giữ nguyên frozen image | Chuẩn bị kiểm tra cấu trúc private | Không thay model | Không thay model | Dựng lệnh chạy chính image đã nộp | GPU, dung lượng, script vận hành và người trực sẵn sàng |
| **20/11** | Nhận private lúc 09:00 theo Submission Guidance | Kiểm tra schema/id, không tạo nhãn | Hỗ trợ vận hành | Hỗ trợ vận hành | Chạy frozen image, lưu log/checksum | Có JSONL private lần chạy đầu; báo BTC nếu lỗi tương thích thay vì tự sửa mô hình |
| **21–22/11** | Nộp kết quả sớm, theo dõi tiếp nhận | Kiểm tra coverage/format | Trực | Trực | Kiểm tra tái lập bằng cùng image nếu thời gian cho phép | Có kết quả được portal tiếp nhận, khớp artifact đóng băng; private leaderboard xếp theo giờ nộp |
| **23/11** | Xác nhận lần cuối trước 12h | So receipt với đúng file | Trực | Trực | Hỗ trợ lỗi hạ tầng | Hoàn tất trước 23:59; lưu toàn bộ bằng chứng |

Nếu đến hết 06/11 chưa có baseline hoàn chỉnh: dừng thử mô hình mới, P1/P4/P5 cùng sửa đường găng; P3 giữ ASR có sẵn, P2 thu nhỏ bộ train-dev để tái hiện lỗi. Nếu 10/11 chưa có checkpoint vượt baseline: giữ baseline và tập trung correctness, thời gian, Docker.

## 6. Danh sách việc triển khai có thể giao ngay

Các đường dẫn sau là **đầu ra dự kiến ban đầu**. Cập nhật thực thi bên dưới ghi rõ phần nào đã có file, phần nào còn cần GPU hoặc kit BTC. Tận dụng module sẵn có của BTC; không viết lại evaluator hay ToolEnv.

| ID | Việc và file dự kiến | Chủ trì | Phụ thuộc | Bằng chứng hoàn thành |
|---|---|---|---|---|
| T01 | `docs/rules.md`, `docs/resources.md`: luật, model/data ledger, câu hỏi BTC | P1 | Portal | Có nguồn và ngày kiểm tra; mọi điểm chưa rõ được ghi riêng |
| T02 | `data/manifests/`, `configs/split.json`: inventory/split, chỉ metadata được phép | P2 | Dataset | Đếm mẫu, đường dẫn, hash, group overlap = 0; dữ liệu gốc ở vùng riêng |
| T03 | `docs/io-contract.md`: sample, history, tools, kết quả, output schema, version kit | P1/P5 | T02 | P3/P4/P5 thống nhất interface bằng ví dụ train thực |
| T04 | `src/asr.py`, `configs/baseline.yaml`: ASR có sẵn, xử lý audio theo kit | P3 | T03 | Transcript dev + latency; không mất phủ định/số do chuẩn hóa tùy tiện |
| T05 | `src/environment.py`: adapter mỏng cho API BTC | P5 | T03 | Một env/mẫu, đúng tool whitelist, không truy cập state cấm, call log khớp thực thi |
| T06 | `src/agent.py`, `prompts/`: vòng quyết định → tool → observation | P4 | T03/T05 | Train-dev case đơn bước/đa bước/OOS/thiếu thông tin/lịch sử chạy đúng kỳ vọng |
| T07 | `src/run.py`, `src/submission.py`: CLI, JSONL, lỗi/timeout, resume giữa mẫu | P5 | T04–T06 | Đủ id, đúng schema, hash do env sinh, không trùng; resume không tái dùng env đang dở |
| T08 | `evaluation/`, `reports/baseline.md`: gọi evaluator BTC | P2 | T07 | Điểm end-to-end và lỗi phân nhóm; bản evaluator và dataset được ghi version |
| T09 | `training/prepare.py`, `training/train_agent.py`, cấu hình train | P4 | T02/T08 | Dữ liệu chỉ từ train hợp lệ; smoke run, recipe, adapter và dev report |
| T10 | `training/train_asr.py` nếu cần | P3 | T08 + transcript train | Chỉ thực hiện nếu lỗi ASR đóng góp đáng kể; report trước/sau cùng tập |
| T11 | `Dockerfile`, `run.sh`, dependency lock, model manifest | P5 | T07 | Máy Linux sạch load archive, tắt mạng, mount test tại `/data`, tạo `result/submission.jsonl` theo BTC |
| T12 | `docs/method.md`, `docs/reproduce.md`, `release/manifest.json` | P1 + cả đội | T08–T11 | Người khác chạy được từ archive, không cần hỏi tác giả; checksum/version/model/data đủ |
| T13 | Hồ sơ portal và receipt lưu riêng | P1 | T12 | Link đúng archive, download được, checksum khớp, form xác nhận lần nộp |

### Tiến độ thực tế — cập nhật 04/11/2026

| Task | Trạng thái | Artifact / phần còn thiếu |
|---|---|---|
| T01 | Hoàn tất tài liệu nền | `docs/competition/`, kế hoạch và strategy; kiểm kê kit v3.2.0 |
| T02 | Hoàn tất kiểm kê cấu trúc | `docs/analysis/dataset_audit.md`; split SFT 80/10/10 đã tạo local, seed 2026, grouped theo transcript yêu cầu chuẩn hóa; data cache không theo dõi bởi Git |
| T03 | Hoàn tất contract v1 | `docs/baseline_design.md`, mẫu JSONL output và quy tắc history/tool loop |
| T04 | Code đã chuẩn bị, chưa benchmark | `src/lifemate/models.py`; cần tải PhoWhisper và đo ASR/latency trên A30 |
| T05–T07 | Runner đã chuẩn bị, chưa chạy trên GPU | `src/lifemate/runner.py`, `prompts.py`, `data.py`, `scripts/run_inference.py`; cần xác nhận layout private mount và chạy end-to-end |
| T08 | Evaluator chẩn đoán đã chuẩn bị, chưa chạy | `training/evaluate_dev.py`; cần CUDA. Chưa có công thức tổng đầy đủ dạng text/scorer BTC |
| T09 | Data prep hoàn tất; LoRA script chưa chạy | `training/prepare_sft.py`, `training/train_lora.py`; kế tiếp là tải model, smoke run rồi train đủ trên Vax.ai |
| T10 | Chưa bắt đầu | Chỉ mở nhánh ASR fine-tune nếu rollout chứng minh ASR là nguồn mất điểm |
| T11–T13 | Chưa bắt đầu | Docker, artifact checksum, upload Drive/form làm sau khi khóa model và có Docker guide BTC |

**Blocker hiện tại:** máy local không có CUDA; chưa xác nhận GPU/VRAM và storage của instance Vax.ai. Chưa tải weights, chưa có điểm baseline, chưa train và chưa tạo bài nộp. Bước cần đội hỗ trợ tiếp theo là đưa repo cùng dataset local lên Vax.ai sau khi xác nhận instance có CUDA và ít nhất 24 GB VRAM.

Mỗi task chia nhánh/file sở hữu rõ, tích hợp ít nhất mỗi ngày; nhánh release chỉ nhận thay đổi đã có kết quả kiểm tra. Không bắt buộc dùng dịch vụ quản lý công việc mới: một bảng TODO/DOING/DONE trong repo là đủ.

## 7. Dữ liệu và đánh giá để tránh chọn nhầm mô hình

1. Sau khi nhận kit, đọc README, version, schema, evaluator trước khi train. Thống kê số mẫu, thời lượng audio, số lượt hội thoại, công cụ/app, độ dài trajectory và nhãn đặc biệt. Không đoán cấu trúc từ ví dụ portal.
2. Chia dev theo **hội thoại/scenario/entity hoặc nhóm nguồn tương ứng mà dữ liệu cho phép**; các câu cùng hội thoại/biến thể gần trùng không được nằm ở cả train và dev. Nếu có speaker/session metadata thì xem xét giữ tách để đo ASR thực chất.
3. Khởi điểm đề xuất: 80% train, 10% dev chọn cấu hình, 10% holdout cuối; chỉ dùng tỷ lệ này khi đủ mẫu/nhóm. Tập nhỏ thì ưu tiên grouped train/dev + holdout phù hợp, ghi sai số và số mẫu; không xem tỷ lệ là luật cứng.
4. P2 giữ riêng nhãn/kết quả holdout; P3/P4 chỉ dùng train và dev đến hết 11/11. Ngày 12/11, sau khi chốt ứng viên chính trên dev, P2 mở holdout để đánh giá lần đầu và ghi thời điểm mở. Nếu sau đó dùng kết quả/lỗi holdout để chọn checkpoint, sửa prompt hoặc refit, phải ghi rõ tập đã tham gia quyết định; các lần chạy lại là kiểm tra hồi quy, không còn được gọi là đánh giá độc lập. Chốt split bằng manifest trước thí nghiệm, ghi seed và phương pháp gộp gần trùng.
5. Dùng evaluator chính thức để tính điểm chính. Không tự suy diễn công thức từ ảnh trên portal. Tự bổ sung số liệu chẩn đoán không thay thế điểm BTC.
6. Mỗi thí nghiệm lưu: commit, config, split version, model revision, seed, data provenance, train time, end-to-end score, nhóm lỗi, latency, VRAM và GPU thực tế.
7. Chọn thay đổi bằng so sánh ghép cặp trên cùng mẫu; báo cả số mẫu được sửa và bị làm hỏng. Nếu chênh nhỏ, ưu tiên bản ổn định/nhanh và dùng bootstrap theo nhóm nếu cần ước lượng bất định.
8. Giữ một bộ ca hồi quy nhỏ từ train-dev: thiếu app, yêu cầu ngoài phạm vi, thiếu người nhận, nhiều liên hệ trùng tên, thời gian tương đối, phủ định, nhiều bước phụ thuộc, lỗi tool, lịch sử đã thực thi, chuỗi Unicode và audio khó.

Không gán nhãn public/private test để cải thiện model hoặc prompt. Không dùng test ids, đáp án đoán hoặc hash đích làm lookup. Leaderboard là tín hiệu kiểm tra tổng quát, không làm vòng chọn model mỗi vài phút. Có thể nộp ít hơn quota 5/ngày; mỗi lần cần có mục đích và bản local report.

### Thứ tự đọc lỗi

| Nhóm lỗi | Chẩn đoán bằng train-dev | Can thiệp ưu tiên |
|---|---|---|
| Nghe sai thực thể | So transcript, WER/CER và độ đúng số/tên/giờ | Chọn ASR, xử lý audio đúng, fine-tune ASR có mục tiêu |
| Hiểu sai ý định | So policy trên transcript chuẩn với transcript ASR của train-dev | Prompt hoặc SFT agent; bảo toàn phủ định và lịch sử |
| Tool/arguments sai | So schema/tool match, id tra cứu, đối số thời gian | Ràng buộc output, buộc tra cứu bằng tool, thêm train case hợp lệ |
| Thiếu bước phụ thuộc | Quan sát trace đã gọi và tool observations | Train trajectory nhiều bước, không sinh toàn bộ plan mà bỏ qua observation |
| Hash sai | So hành vi quan sát được và trace qua API hợp lệ | Sửa execution/order/tool arguments; tuyệt đối không đọc state test |
| Gọi dư | Đếm unmatched calls dù trạng thái cuối đúng | Điều kiện dừng rõ, tránh query/validation/execution lặp không cần |
| Timeout/OOM | Profile từng pha trên GPU gần máy chấm | Giảm context, số bước/token, backend overhead; cân nhắc ASR nhỏ hơn |
| Không ổn định | Chạy lại toàn bộ invocation từ đầu cùng image/input | Khóa seed/backend, greedy, số worker, dependency và thứ tự xử lý |

Theo Evaluation, debug theo hai lớp: (1) state hash chính xác — sai thì mẫu 0; (2) độ khớp tool-call theo precision/recall/F1, gồm call dư và call bị schema từ chối. Fuzzy argument phải vượt token-F1 0.6 sau lowercase và Unicode NFC. Mỗi gold/submit call chỉ match tối đa một lần. Out-of-scope chỉ được điểm khi cả gold lẫn dự đoán đều không gọi tool. Dùng evaluator BTC cho công thức tổng thể; không suy trọng số từ ảnh công thức bị thiếu trong bản văn bản.

## 8. Kế hoạch huấn luyện và thí nghiệm

**Thứ tự:** pretrained baseline → SFT/LoRA agent → train với transcript ASR nhiễu từ train → ASR fine-tune nếu lỗi nghe còn là nút thắt → tối ưu runtime. Không cần huấn luyện từ đầu.

Cho agent, biến trajectory train thành từng bước: thông tin mà agent thực sự thấy tại bước đó → tool call tiếp theo hoặc tín hiệu kết thúc. Observation huấn luyện phải đến từ trace/kit hợp lệ; không đưa gold tương lai hay trạng thái ẩn vào prompt mà inference không có.

Nếu train có đủ supervision phù hợp, thử LoRA một cấu hình nhỏ trước. Cấu hình khởi điểm để khảo sát: rank 16 hoặc 32, learning rate khoảng 1e-4, 1–3 epoch, context 4K/8K rồi tăng nếu cần, effective batch 16–32 tùy GPU và độ dài mẫu. Đây là **điểm bắt đầu thí nghiệm**, không phải thông số tối ưu đã biết. Target modules, loss mask, packing và precision chỉ chốt sau khi đọc model/trajectory thực tế.

Giới hạn tối đa 2 cấu hình train chính cùng lúc. Kiểm tra 50–100 mẫu trước khi chạy dài; nếu dataset nhỏ thì dùng mẫu đại diện tương ứng. Chọn checkpoint theo rollout end-to-end trên dev, không chỉ teacher-forced accuracy/loss. Với adapter, kiểm tra artifact merge/unmerged thực tế trước khi đóng gói; tính tham số cả hệ thống theo cách BTC yêu cầu.

ASR: so medium/large trước trên cùng tập. Chỉ fine-tune khi có transcript train hợp lệ và lợi ích kỳ vọng rõ. Khi đánh giá khả năng chịu lỗi ASR của agent, dùng transcript ASR sinh từ train cho phần train và giữ dev tách biệt. Không lấy public test đi nhờ mô hình ngoài phiên âm/gán nhãn để train.

Mỗi ablation đổi một yếu tố: ASR, prompt, training data, checkpoint, context hoặc backend. Bản tốt phải qua cả accuracy, latency, memory và quy định. Khi refit trên toàn bộ train sau khi chọn recipe, vẫn cần kiểm tra hồi quy/offline/tính ổn định; không giả định refit luôn tốt hơn checkpoint đã kiểm chứng.

## 9. GPU, hạ tầng và kiểm soát chi phí

Bạn nói đội thuê “vax.ai”; kế hoạch dùng giả định **dịch vụ thuê GPU mà đội đang có**, chưa xác nhận có phải Vast.ai. Không cần chốt thương hiệu để chia việc, nhưng phải xác nhận trước khi thuê hoặc cung cấp thông tin tài khoản. Chưa có ngân sách trần, nên không đặt giá thuê giả hoặc tự thuê máy.

| Tài nguyên | Dùng cho | Cách dùng đề xuất |
|---|---|---|
| 1 GPU 24 GB, ưu tiên A30 để kiểm chứng | Baseline, inference, profile, thử QLoRA nếu cần | Giữ cấu hình giống máy chấm nhất có thể; card khác chỉ cho số đo tham khảo |
| 1 GPU 48–80 GB theo nhu cầu | LoRA LLM hoặc ASR train | Thuê theo đợt sau khi dữ liệu/config và smoke run đã sẵn sàng |
| CPU và storage bền vững | Chuẩn bị dữ liệu, evaluator, checkpoints và Docker archive | Không để bản duy nhất trên ổ tạm của instance |

Không khởi động 5 GPU chỉ vì có 5 người. Ban đầu một máy GPU cho baseline là đủ; bật máy train thứ hai khi T09 sẵn sàng. Có thể dùng cùng GPU theo lịch nếu ngân sách nhỏ. Không cam kết model sẽ vừa RAM/VRAM trước khi đo.

Trước mỗi job: kiểm tra model GPU/VRAM thực cấp, driver, CUDA/framework, Docker + GPU passthrough, dung lượng và quyền lưu checkpoint. Không yêu cầu CUDA mới nhất nếu stack đang tương thích. Chốt môi trường sau smoke run.

Chi phí dự toán = Σ(đơn giá GPU-giờ × số giờ) + storage + phí truyền dữ liệu. Đo một job ngắn để ngoại suy thời gian trước khi duyệt full run; ghi chi phí vào experiment log. Ưu tiên giữ ngân sách cho kiểm tra A30, lưu artifact và chạy private; mức chia đề xuất ban đầu 50% train/thử nghiệm, 30% inference/kiểm tra, 20% dự phòng, điều chỉnh khi có trần tiền thật.

Checkpoint định kỳ theo thời gian, giữ last + best + config. Có kế hoạch resume sau mất instance. Sao lưu weights cần thiết và Docker archive sang chỗ đội kiểm soát; kiểm tra hash sau truyền. Không đưa khóa truy cập vào image/log/tệp public.

## 10. Các cổng kiểm tra trước khi chuyển bước

| Cổng | Điều kiện đạt | Nếu không đạt |
|---|---|---|
| G0 — đủ đầu vào | Có quyền dataset, kit/version, evaluator và luật; biết cấu trúc input | P1 giải quyết quyền/BTC; không train trên schema tự đoán |
| G1 — baseline | Toàn bộ id đúng một lần, JSONL parse được, chạy evaluator, portal tiếp nhận | Tạm dừng train/tối ưu để sửa runner |
| G2 — đáng giữ | Cải thiện trên dev có phân nhóm, không phá ca quan trọng; đáp ứng luật | Giữ baseline hoặc thu nhỏ thay đổi |
| G3 — chạy trên máy chấm | Không OOM; đo full workload; 30 giây/mẫu; toàn bộ token ≤16K | Giảm tải hoặc đổi ứng viên, không lấy p95 để che mẫu quá hạn |
| G4 — Docker | Archive load được trên Linux sạch, offline, không dùng cache ngoài, output đúng đường dẫn | Chưa nộp; dùng artifact đã qua kiểm tra |
| G5 — hồ sơ | Code/train recipe/model/data manifest/docs/license/checksum/link đúng; đã xử lý điểm mơ hồ của form | P1/P5 đối chiếu hướng dẫn BTC |
| G6 — nộp xong | Có receipt hoặc trạng thái xác nhận, id/lần nộp/thời gian/file hash khớp | Theo dõi và xử lý trước hạn, không coi click Submit là hoàn tất |

Mục tiêu nội bộ runtime là có khoảng dư đáng kể dưới 30 giây, chẳng hạn p95 ≤20 giây và không mẫu nào vượt 30 giây trên bộ kiểm tra đầy đủ. Đây là ngưỡng kế hoạch, phải thay bằng số đo và bao gồm định nghĩa thời gian của BTC. Kiểm tra cold/warm start, audio dài nhất, chuỗi tool nhiều bước và peak RAM/VRAM.

## 11. Đóng gói và nộp giải pháp ngày 16/11

P5 chuẩn bị image Linux đúng kiến trúc BTC và GPU stack tương thích. Trọng số, tokenizer/processor, adapter hoặc model merge, schema/runtime cần thiết, dependency và entrypoint phải có sẵn. Khóa model/config/prompt; không để inference tải Hugging Face hay pip.

Theo Rule, BTC mount private folder tại `/data`, chạy inference entrypoint và lấy `result/submission.jsonl`; container phải chạy hoàn toàn offline, không có network. **Entrypoint, working directory, vị trí result là relative/absolute, quyền ghi, cấu trúc archive và phiên bản kit cần đối chiếu tài liệu Docker riêng của BTC trước khi chốt.** Không tự xem CLI đề xuất của đội là contract BTC.

Checklist phát hành:

- [ ] Đúng image/platform và có tag/version bất biến.
- [ ] Load từ archive, không chỉ chạy image có sẵn trên máy người build. Tham khảo [Docker image save](https://docs.docker.com/reference/cli/docker/image/save/); chọn định dạng nén đúng BTC chấp nhận.
- [ ] Tắt mạng khi chạy; không mount model cache từ máy ngoài.
- [ ] Dữ liệu input mới, đúng kit; output đủ id, không NaN/JSON lỗi/duplicate.
- [ ] Chạy lại cùng image/input, ghi mức độ khớp của tool_calls/hash; không chỉ kiểm temperature 0.
- [ ] Có mã train, inference, cấu hình, seed, provenance, license, mô tả phương pháp và hướng dẫn tái lập.
- [ ] Có bảng đếm tham số cho mọi model và giới hạn context tại runtime.
- [ ] Có manifest gồm commit, image tag/digest, archive size, checksum (đề xuất SHA-256 nếu form không quy định loại khác), model revisions và kit version.
- [ ] Google Drive link đúng tệp archive bất biến, mở tải được khi không dùng tài khoản đội; P2 tải lại và kiểm checksum.
- [ ] Bao gồm thành phần data cần thiết mà Submission Guidance yêu cầu trong Docker package. Link công khai chỉ trỏ tới đúng gói gửi BTC; loại bỏ private test, dữ liệu ngoài yêu cầu và credentials.
- [ ] P1 và P5 cùng kiểm trước lần nộp; lưu receipt, giờ GMT+7, link, checksum và bản form.

Gói có thể lớn: đo kích thước và tốc độ upload thực tế từ sớm. Thời gian tối thiểu ≈ dung lượng byte × 8 / băng thông bit/giây, rồi cộng khoảng dư cho retry, Drive processing và tải xác minh. Đến 14/11 phải có bản đã upload/kiểm tra, không đợi chiều 16/11 mới chuyển weights.

Không ghi đè file sau khi checksum hoặc link đã nộp. Có thay đổi phải tạo artifact mới và ghi rõ lần nộp. Chỉ dùng lần 2 khi bản mới đã qua toàn bộ kiểm tra; vì BTC chỉ đánh giá lần cuối nên nộp lại bản chưa kiểm chứng có thể làm mất bản tốt.

## 12. Vận hành private test

Trước 20/11 lưu frozen image, code/model/prompt/config, archive checksum, manifest và recipe; không đổi các thành phần này để thích nghi private. Nếu BTC phát hành kit mới hoặc có lỗi contract, hỏi kênh chính thức về phần được thay đổi, giữ log chấp thuận.

Ngày phát hành: nhận dữ liệu đúng quyền; kiểm tra checksum/version và schema thuần cấu trúc. Chạy đúng image đã nộp, không đọc `<split>_environment/` ngoài API cho phép. Không gán nhãn, fine-tune hay sửa prompt dựa trên private.

Đo tiến độ bằng số mẫu hoàn tất và tốc độ thực; ước lượng thời gian còn lại sau vài mẫu đại diện. Khi resume chỉ tiếp tục mẫu chưa hoàn tất trong invocation hợp lệ; không ghép call/hash từ môi trường bị gián đoạn. Tránh chạy nhiều phiên ghi vào cùng JSONL.

Mục tiêu có kết quả và nộp sớm trong 20–21/11; 22/11 dành kiểm tra và xử lý lỗi vận hành. Nếu cùng image cho kết quả khác nhau, lưu cả log và đối chiếu chính sách tái lập; không chọn riêng đầu ra có vẻ tốt của từng mẫu. Hạn cứng 23/11, 23:59.

## 13. Rủi ro và cách rút gọn khi chậm

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Chưa lấy được kit | Không đủ quyền/login hoặc file tải lỗi | P1 xử lý ngay; tận dụng thời gian đọc luật/container, không tự tạo dữ liệu giả thay tài liệu BTC |
| Train tăng loss đẹp nhưng score không tăng | Tool/hash score không tốt hơn baseline | Kiểm tra trajectory/observation leakage; chọn theo rollout; giữ baseline |
| ASR nghe sai số/tên | Entity accuracy thấp dù WER tổng ổn | Ưu tiên lỗi thực thể và downstream score; thử ASR lớn hơn trong budget |
| Tool hallucination | Tool/id không tồn tại, lặp write | Whitelist schema, dùng lookup tool, điều kiện dừng và kiểm trace |
| GPU thuê nhanh nhưng A30 chậm | p99/max vượt hạn, OOM khi ghép ASR/LLM | Dành máy A30 đo sớm; giảm context/model/runtime overhead |
| Mất instance | Checkpoint/weights chỉ nằm trên ổ tạm | Backup có hash, resume recipe, chỉ thuê lại sau kiểm tra bản lưu |
| Docker thiếu tài nguyên | Chạy được khi online nhưng fail offline | Máy sạch + network disabled + không external cache từ tuần đầu |
| Public score gây chọn sai | Local dev không cải thiện tương ứng | Không chase leaderboard; giữ holdout từ train và nguồn dữ liệu rõ |
| Hết thời gian | Sau 10/11 chưa có model tốt | Dừng search model/ASR train mới; giữ baseline, sửa correctness và packaging |
| Hồ sơ cuối không tải được | Drive permission/link/checksum sai | P2 tải không đăng nhập, checksum khớp; lưu receipt, không thay file sau nộp |

Ưu tiên khi phải cắt việc: giữ baseline + evaluator + tool correctness + Docker + nộp đúng; cắt model thứ ba, UI/demo, framework mới, sweep lớn, dữ liệu tổng hợp chưa kiểm và tối ưu nhỏ chưa chứng minh lợi ích.

## 14. Việc đội làm ngay trong buổi đầu

1. P1 điền tên 5 người, xác nhận lịch làm và một mức ngân sách trần; kiểm tra mỗi người đã đăng ký và vào đúng một đội.
2. P1 lấy quyền Data/kit; cả đội đọc README và Rule/Submission Guidance mới nhất. Lưu version và ngày kiểm tra.
3. P2 kiểm kê dataset, xác định grouping cho split, đưa ra 10–20 mẫu train đại diện để thống nhất contract.
4. P3 và P4 chuẩn bị hai ứng viên pretrained nêu trên với revision/license rõ; chưa chạy job train dài.
5. P5 kiểm tra một máy GPU, một mẫu ToolEnv và container tối thiểu; P1 tích hợp đường chạy xuyên suốt.
6. Cuối buổi lưu `status.md`: việc xong, metric đã đo, blocker, người xử lý và deadline; không ghi “done” khi chưa có artifact hoặc bằng chứng.

## 15. Thông tin còn cần chốt, không ngăn việc chuẩn bị baseline

- Giờ làm thực tế mỗi người, kỹ năng mạnh/yếu và tên thật tương ứng P1–P5.
- Ngân sách GPU trần, GPU/VRAM của instance Vax.ai, dung lượng lưu trữ và quyền dùng Docker.
- Hướng dẫn Docker chi tiết: entrypoint, output, cold start, version/environment compatibility và kiểu checksum.
- Phần “data” BTC muốn nhận trong gói Docker; cách chuyển tài nguyên hạn chế khi form yêu cầu link công khai.
- Công thức Evaluation đầy đủ hoặc scorer chính thức để đối chiếu metric cục bộ.
- Giờ phát hành private và thời gian form Solution Submission mở nếu portal có cập nhật mới.

Các thông tin này được P1 xác minh qua portal/diễn đàn hoặc kênh BTC. Kế hoạch chưa gửi bất kỳ email, thuê GPU, chấp nhận điều khoản, công khai dữ liệu hay nộp bài thay đội.

## 16. Nguồn và giới hạn của kế hoạch

- [Portal LifeMate](https://challenge.zalo.ai/portal/lifemate): Description, Evaluation, Rule, Submission Guidance, đọc trực tiếp 04/11/2026.
- [Trang chủ/lịch](https://challenge.zalo.ai/), [Thể lệ VI](https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf).
- [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [PhoWhisper medium](https://huggingface.co/vinai/PhoWhisper-medium), [PhoWhisper large](https://huggingface.co/vinai/PhoWhisper-large).
- [NVIDIA A30](https://www.nvidia.com/en-gb/data-center/products/a30-gpu/), [Docker image save](https://docs.docker.com/reference/cli/docker/image/save/).

Luật và lịch có thể được BTC cập nhật; P1 kiểm tra hằng ngày và ghi thay đổi tác động đến release. Kiến trúc, phân công, lịch nội bộ và hyperparameter ở trên là đề xuất của kế hoạch, chưa phải quyết định được đo hoặc kết quả đã đạt.
