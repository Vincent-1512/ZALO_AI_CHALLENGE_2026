# Chiến lược LifeMate: chất lượng, khả năng tổng quát và nộp bài đúng hạn

Ngày 04/11/2026. Dùng cùng [kế hoạch triển khai theo ngày](implementation_plan.md). Tài liệu này bổ sung ưu tiên nghiên cứu và tiêu chí chọn giải pháp sau khi đã có dataset v3.2.0 cùng Data, Evaluation, Rule và Submission Guidance do đội cung cấp. Các lựa chọn kỹ thuật là đề xuất cần đo; chưa có điểm baseline hay kết quả train.

## 1. Định nghĩa thành công

Đội cần tối đa hóa điểm private của một giải pháp hợp lệ, chạy tái lập được trên A30 trong 30 giây/mẫu. Điểm public hỗ trợ kiểm tra tích hợp; không phải mục tiêu chọn mô hình chính. Không thể bảo đảm giải hoặc điểm tuyệt đối; mọi thay đổi được giữ lại phải có bằng chứng cải thiện trên dữ liệu train tách riêng để đánh giá.

Thứ tự ưu tiên: **đúng trạng thái → đúng và đủ tool-call → xử lý app/schema mới → ổn định thời gian và bộ nhớ → đóng gói, kiểm chứng và nộp**. Docker được làm từ baseline, không chờ đến cuối.

## 2. Điều dữ liệu thật đã cho biết

Thống kê chỉ đọc JSONL, catalog và metadata audio; không đọc file trong `*_environment/`.

| Quan sát | Kết quả | Quyết định |
|---|---|---|
| Train / public | 1.545 / 634 mẫu | Dataset nhỏ: ưu tiên pretrained mạnh + thích nghi có kiểm soát |
| Catalog train | 46 app, 147 tool | Không viết agent chỉ hỗ trợ một số app quen thuộc |
| Catalog public | 36 app, 122 tool | Catalog phải được nạp theo split, không hard-code catalog train |
| Giao hai catalog | 25 app chung; 11 app public không có trong train | Khả năng dùng schema mới là một trục đánh giá bắt buộc |
| Train có history | 1.018 mẫu, tối đa 24 lượt | Phải xử lý audio trong history và tham chiếu hội thoại; không chỉ nghe câu cuối |
| Train cần hỏi thiếu | 217 mẫu có `sys.ask_missing_info` | Dành nhóm đánh giá và supervision riêng |
| Train không gọi tool | 111 mẫu có gold calls rỗng | Đánh giá chính xác hành vi dừng/OOS, không ép tạo hành động |
| Số gold call | 0:111; 1:369; 2:428; 3:359; 4:161; 5:107; 6:10 | Phần lớn bài cần nhiều bước; giới hạn bước không đặt thấp tùy ý |
| Audio | 5.036 train WAV, 1.933 public WAV | Thêm audio history vào dự toán ASR và ngân sách 30 giây |
| Dung lượng | Zip khoảng 836 MB; giải nén khoảng 1,1 GB trên đĩa | Chừa riêng storage cho weights, checkpoint và Docker lớn hơn dataset |

11 app mới là khác biệt ở mức **catalog**, không phải tuyên bố mỗi app đều xuất hiện trong các mẫu public. Không gán nhãn hay nghiên cứu lỗi theo từng mẫu public để huấn luyện/tune. So sánh metadata catalog nhằm xác định nhu cầu tổng quát; không tạo bộ train đặc thù từ câu hỏi public.

## 3. Hướng giải pháp được đề xuất

Giữ kiến trúc **ASR → văn bản + context/history → LLM nhỏ đọc schema → vòng tool-observation → JSONL**. Hướng này cho phép đo riêng lỗi nghe, lỗi quyết định và lỗi thực thi, phù hợp 5 người làm song song.

Ứng viên đầu: PhoWhisper-medium + Qwen3-4B-Instruct-2507 như kế hoạch hiện có. So PhoWhisper-large nếu lợi ích downstream đủ bù latency. Chốt revision/license trước ngày giới hạn của BTC và đếm chính xác toàn bộ tham số; không suy ra hợp lệ chỉ từ tên model. Không dùng ensemble nhiều model trong bản nộp nếu chưa chứng minh thỏa tổng <6B, LLM <5B và thời gian.

Agent đọc schema của app được cài trong từng mẫu, dùng tool đọc để lấy ID/thông tin, rồi mới gọi tool ghi. Hỗ trợ `required`, enum, kiểu dữ liệu, `x-one-of`, `x-ref`, `x-match`; không xem mọi trường có trong schema là bắt buộc.

Mô hình sinh từng bước và nhận observation thật trước bước sau. Một tool call lỗi vẫn nằm trong log bài nộp; không chạy thử write call để dò rồi xóa dấu vết. Dùng `validate` cho lỗi schema trước thực thi; không coi validate là kiểm tra đủ tính đúng nghiệp vụ.

History đã được áp dụng vào môi trường. Chỉ thực hiện hành động cho yêu cầu cuối; dùng history để hiểu tham chiếu. Chuẩn hóa thời gian theo `context.current_time` và timezone của mẫu. Không replay hành động cũ, không dùng ngày giờ của máy chủ thay context.

## 4. Sáu thí nghiệm quan trọng nhất

| ID | Câu hỏi | Cách đo | Tiêu chí giữ |
|---|---|---|---|
| E0 | Agent có thực sự chạy đúng vòng tool không? | Baseline trên transcript chuẩn của train-dev, không đưa label vào prompt | Có output hợp lệ, state/call diagnostics và trace; không phải bài nộp public vì public không có transcript chuẩn |
| E1 | Lỗi ASR làm mất bao nhiêu điểm? | Cùng agent/split, thay transcript chuẩn bằng ASR cho tất cả user turn cần thiết | Đo chênh điểm và lỗi thực thể; xác định ưu tiên ASR hay policy |
| E2 | ASR medium hay large đáng dùng hơn? | So end-to-end cùng dev, latency/RAM/VRAM trên A30 | Tăng điểm với thời gian đủ dư; WER giảm một mình không đủ |
| E3 | Schema/prompt có đủ dùng app mới không? | App-held-out từ train; cùng policy, schema nạp động | Giảm khoảng cách giữa app quen và app-held-out; không hard-code tool theo ID |
| E4 | LoRA có cải thiện so với pretrained không? | SFT trajectory train, rollout trên cùng dev; so cả app-held-out | Tăng state/call score, không làm suy giảm rõ khả năng dùng schema mới |
| E5 | Huấn luyện với transcript ASR nhiễu có giúp không? | Transcript ASR chỉ sinh từ train; giữ dev nguyên | Tăng điểm audio end-to-end, không dựa vào loss hoặc kết quả teacher-forced |

Chỉ sau E0–E5 mới xét ASR fine-tune, retrieval ví dụ từ train hoặc dữ liệu tổng hợp bổ sung. Mỗi nhánh mới phải chỉ ra nhóm lỗi và số điểm kỳ vọng lấy lại. Không triển khai RL, framework đa agent hoặc mô hình audio end-to-end trên đường găng hiện tại.

## 5. Thiết kế đánh giá có khả năng phát hiện overfit

P2 lập manifest cố định trước huấn luyện:

1. Gom các mẫu cùng hội thoại/scenario, transcript gần trùng hoặc nhóm liên quan bằng metadata thực có. Kiểm tra overlap trước khi chia; không mặc định `id` là nhóm độc lập.
2. Có tập dev đại diện cho app quen, độ dài history, OOS, missing-info và số bước. Tỷ lệ 80/10/10 trong kế hoạch cũ chỉ là điểm bắt đầu, điều chỉnh theo số nhóm thực.
3. Bổ sung đánh giá **app-held-out từ chính train**: chọn một số app làm app mới đối với phần train của thí nghiệm; loại mọi mẫu có liên quan các app này khỏi phần fit. Schema vẫn được cung cấp lúc đánh giá, giống khả năng cần có khi inference. Với mẫu đa app, xét tập app liên quan để tránh lọt qua split.
4. Do dữ liệu nhỏ, tập app-held-out có thể ít và khác phân bố; báo số mẫu, app và độ dài trajectory. Không coi một điểm số nhỏ lẻ là bằng chứng tổng quát tuyệt đối.
5. Giữ holdout cuối theo lịch 12/11 như kế hoạch; P2 chưa tiết lộ lỗi/nhãn trước khi chốt ứng viên. Nếu dùng nó để sửa model, công khai trong report rằng holdout đã tham gia quyết định.

Không sử dụng public/private test làm validation train hoặc tạo nhãn bổ sung. App-held-out chỉ lấy từ train; không dựng bài tổng hợp mô phỏng riêng các câu hỏi test.

### Công thức và công cụ đánh giá

Bản Evaluation đội gửi thiếu hình công thức. Kit hiện đã xác định có CLI **kiểm tra schema**, nhưng kiểm tra schema không đồng nghĩa chấm đầy đủ state-gated tool-call F1. Trước khi gọi một con số là “điểm đúng BTC”, P2 phải lấy đủ công thức/tài liệu hoặc scorer chính thức.

Nếu BTC không cung cấp scorer, triển khai local scorer theo công thức được xác nhận, có ca tính tay và đối chiếu. Trước đó chỉ báo các chỉ số chẩn đoán được định nghĩa rõ, không gắn nhãn official score.

Đối chiếu state trên train bằng công cụ BTC và nhãn train được phép. Nếu cần tạo gold hash reference bằng gold calls, thực hiện trong tiến trình đánh giá tham chiếu train riêng, một ToolEnv/mẫu/lần chạy; không tạo môi trường thứ hai, clone hoặc đọc state file trong agent inference. Không áp dụng gold-reference từ test. Chốt phương pháp này với luật nếu BTC có giải thích bổ sung.

Report mỗi ứng viên gồm: state-match rate, tool-call match, lỗi schema, OOS accuracy, missing-info match, nhóm history dài/đa bước, audio end-to-end so với transcript chuẩn, và số mẫu được sửa/bị làm hỏng so với baseline. Kèm p50/p95/p99/max latency, số timeout, peak VRAM/RAM, thông tin GPU và version code/model/data.

## 6. Cách huấn luyện để học hành vi thay vì nhớ đáp án

Train đã có `audio_text`, `label.tool_calls`, `tool_results`, `final_state`: đủ để tạo supervision từng bước mà không nhờ mô hình ngoài gán nhãn.

- Input mỗi bước chỉ gồm dữ liệu agent sẽ thấy tại thời điểm đó: user/context/history, schema, các call trước và observation trước. Target là quyết định kế tiếp; không đưa `final_state`, gold tương lai hoặc nhãn vào prompt inference.
- Chỉ tính loss cho phần assistant cần học; giữ nguyên ngữ nghĩa tool result, kiểu JSON và ID. Có tín hiệu kết thúc rõ ràng.
- Dữ liệu train không chỉ có trường hợp thành công: giữ OOS, missing-info và history. Nếu lấy mẫu cân bằng các nhóm hiếm, đo lại trên dev có phân bố tự nhiên để phát hiện hành vi hỏi quá nhiều.
- Sinh transcript ASR của **train** rồi dùng làm biến thể nhiễu; không sinh lỗi chính tả ngẫu nhiên không giống lỗi nghe thực.
- Prompt/schema ở train và inference dùng cùng contract. Nếu rút gọn schema để tăng tốc, so ablation trước khi giữ; không cắt mô tả bắt buộc hoặc liên kết `x-ref` cần dùng.
- LoRA ưu tiên hai cấu hình nhỏ đã lên kế hoạch; không sweep hàng chục learning rate. Kiểm tra loss-mask/trajectory bằng một batch trước job dài; chọn checkpoint theo rollout.
- Dữ liệu tổng hợp chỉ bổ sung nhóm lỗi đã đo, từ nguồn train/nguồn công khai được phép; ghi provenance, license và kiểm tra chất lượng. Không gửi dataset hạn chế sang dịch vụ ngoài khi chưa được phép.

Không mặc định refit toàn bộ train sẽ tốt hơn. Nếu refit sau khi chọn recipe, phải còn thời gian chạy hồi quy, offline, latency và giữ checkpoint dự phòng.

## 7. Kế hoạch 48 giờ đầu từ khi GPU sẵn sàng

| Khoảng thời gian | Người chính | Việc | Bằng chứng |
|---|---|---|---|
| 0–6 giờ | P1 + P2 | Chốt schema/kit version, split, scoring contract; thống kê còn thiếu | Manifest, bảng nhóm mẫu, công thức/scorer hoặc danh sách diagnostic rõ |
| 0–6 giờ | P3 + P4 + P5 | Tải revision hợp lệ, nạp ASR/LLM và ToolEnv theo kit | Một train sample hoàn tất, không lấy gold làm output của agent |
| 6–18 giờ | P4 + P2 | E0: agent trên transcript train-dev | Trace và báo cáo state/tool; lỗi được phân loại |
| 6–18 giờ | P3 + P5 | ASR tất cả audio turn cần dùng, ghép history, JSONL runner | Audio → output hoàn chỉnh, thời gian từng pha |
| 18–30 giờ | P1 + P5 | Tích hợp và Docker offline phiên bản đầu | Chạy từ archive trên máy sạch, nạp đúng split `zaic_tools` |
| 18–30 giờ | P2 + P3 + P4 | E1/E2 bản nhỏ, ưu tiên lỗi lớn | Chênh transcript/audio và model ASR, không chọn theo cảm giác |
| 30–48 giờ | P1 + P5 | Chạy public đầy đủ, kiểm output, nộp JSONL đầu | Đủ 634 id, có receipt, ghi model/config/artifact checksum |
| 30–48 giờ | P2 + P4 | Chuẩn bị SFT từ train và E3 app-held-out | Dữ liệu không leakage, job train ngắn sẵn sàng |

Nếu 48 giờ không đạt, ưu tiên gỡ lỗi integration; không khởi động thêm model mới. Trên Mac hiện tại không có `nvidia-smi`; chuẩn bị/đọc dữ liệu có thể làm local, còn số đo GPU phải lấy từ máy thuê. Chưa có quyền truy cập máy thuê hoặc ngân sách trần trong workspace.

## 8. Phân bổ 5 người và tài nguyên

Giữ vai trò P1–P5 của kế hoạch gốc. P2 sở hữu split, metric và report, P4 không tự sửa tập dev để tăng điểm. P3 sở hữu ASR và audio history. P5 sở hữu ToolEnv adapter, giới hạn tài nguyên và Docker. P1 tích hợp, giải quyết blocker và quản lý nộp bài.

Một GPU cho tích hợp/benchmark và thêm một GPU cho train khi cần là cấu hình đề xuất. Dành một phiên A30 thật từ sớm; 4090/A100 nhanh không chứng minh đạt 30 giây trên A30. Không thuê hoặc chi tiền trước khi có cấu hình instance, giá và trần tiền được đội xác nhận.

Mỗi ngày mỗi người báo đúng bốn mục: artifact hoàn tất, kết quả đo, blocker, việc tiếp theo. Review thí nghiệm bằng bảng so sánh cùng split; dừng nhánh nếu không giải quyết nguồn mất điểm lớn hoặc chiếm thời gian kiểm chứng trước hạn.

## 9. Các mốc quyết định để tránh nghiên cứu vô hạn

| Mốc | Quyết định |
|---|---|
| 05–06/11 | Có baseline public hợp lệ và metric local được định nghĩa rõ; nếu chưa có, ưu tiên runner/scorer |
| 07–09/11 | Chạy E3/E4/E5 theo nhóm lỗi; tối đa hai job train chính đồng thời |
| 10/11 | Chọn ứng viên chính + dự phòng dựa trên dev và app-held-out từ train |
| 11–12/11 | Chốt recipe, mở holdout lần đầu, đo full workload A30 và tính tái lập |
| 13/11 | Ngừng thay kiến trúc; sửa lỗi có bằng chứng và đánh giá lại |
| 14–15/11 | Docker archive, tài liệu train/inference/data, checksum, tải lại từ Drive và kiểm tra |
| 16/11 trước 12h | Mục tiêu nộp lần 1; dùng lần 2 chỉ với bản đã qua kiểm chứng; hạn cứng 23:59 |
| 20/11 lúc 09h | Nhận private và chạy đúng giải pháp đã đóng băng |
| 21–22/11 | Mục tiêu nộp kết quả private sớm; xác nhận receipt |
| 23/11 lúc 23:59 | Hạn cuối kết quả private |

Public JSONL tối đa 5/ngày; final Docker tối đa 2 lần, BTC chỉ xét lần cuối. Không dùng lượt thay thế cuối cùng cho một bản có điểm tốt nhưng chưa kiểm tra artifact, latency và offline.

## 10. Checklist để gọi giải pháp là sẵn sàng

- [ ] Công thức metric đầy đủ hoặc scorer BTC đã được xác nhận; report phân biệt diagnostic và official-equivalent score.
- [ ] Baseline và bản tốt nhất cùng được đánh giá trên split cố định, có khả năng dùng app-held-out từ train.
- [ ] Không dùng label/final_state/audio_text của test; không truy cập trực tiếp file môi trường; đúng một ToolEnv mỗi mẫu trong inference.
- [ ] Log output lấy từ env, gồm mọi call thành công/thất bại; hash do env sinh.
- [ ] ASR xử lý audio history cần thiết; không replay các hành động history.
- [ ] Tổng model <6B, LLM <5B, context ≤16.384, license/revision hợp lệ và có bảng đếm.
- [ ] Full workload A30 không OOM; đo mọi mẫu, không dùng riêng p95 để che timeout ở mẫu khó.
- [ ] Offline hoàn toàn, weights/dependencies có sẵn; cùng frozen image tái lập được.
- [ ] Docker đọc `/data`, tạo `result/submission.jsonl` theo hướng dẫn riêng của BTC; cả data/code/train recipe cần nộp đều sẵn.
- [ ] Link/checksum/receipt khớp đúng archive; raw contest data không lên GitHub.

Phần còn thiếu quan trọng nhất hiện nay: hình công thức đầy đủ hoặc scorer chính thức; cấu hình/quyền truy cập GPU thuê, lịch làm thực tế và ngân sách. Các mục này không ngăn việc chuẩn bị code, nhưng ngăn việc khẳng định một ứng viên đã đạt điểm và tốc độ mục tiêu.
