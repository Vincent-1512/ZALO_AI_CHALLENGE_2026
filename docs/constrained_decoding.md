# Ghi chú nghiên cứu: constrained decoding cho action JSON

## Phạm vi

Chỉ ghi nhận hướng thử nghiệm; chưa có mã constrained decoding và chưa đo latency/điểm trên dev.
Repository ghim `transformers==4.57.1`. Tài liệu phiên bản này có `prefix_allowed_tokens_fn` và `logits_processor` trên `generate()`; mô tả API nói prefix callback giới hạn beam search, còn `LogitsProcessor` cho phép biến đổi score ở mỗi bước sinh. Xem [Generate API 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation) và [generation utilities 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils).

## Cách áp dụng cho LifeMate

Mỗi bước chỉ được trả một trong hai dạng: `{"action":"stop"}` hoặc một tool call có `name` và `arguments`. Callback/processor sẽ theo dõi token prefix hiện tại rồi chỉ cho phép token còn có thể hoàn thành JSON hợp lệ theo schema của các tool đã cài. Danh sách tool và enum thay đổi theo sample, nên grammar/automaton cần dựng từ `available_tools` của prompt ở từng bước.

- `prefix_allowed_tokens_fn`: tra trie/automaton theo prefix và trả tập token kế tiếp hợp lệ. API được tài liệu mô tả trong ngữ cảnh beam search; cần kiểm thử riêng với chế độ greedy hiện tại (`num_beams=1`) trước khi cân nhắc tích hợp.
- `LogitsProcessor`: đặt score thành `-inf` cho token không hợp lệ theo parser trạng thái. Có thể áp dụng ở từng bước generation, nhưng processor phải đồng bộ chính xác với tokenization, dấu ngoặc/escape trong JSON, kiểu số/boolean/null, enum và các trường `required`/`x-one-of`.

## Chi phí và rủi ro

Processor chạy mỗi token sinh. Cách thử mọi token trong vocabulary bằng cách nối vào prefix rồi parse có chi phí xấp xỉ `O(T × V × P)`, với `T` là số token output, `V` là vocabulary và `P` là chi phí kiểm tra prefix. Trie hoặc DFA giảm số phép thử nhưng làm parser và trạng thái schema phức tạp hơn. Sai grammar có thể khóa mọi token hợp lệ hoặc loại nhầm tool/argument đúng; JSON hợp lệ vẫn có thể mang ngữ nghĩa sai.

Constrained decoding có thể giảm lỗi cú pháp và lỗi schema, nhưng không tự sửa ASR, chọn nhầm tool hợp lệ, suy luận sai arguments, thứ tự gọi tool hay state cuối. `ToolEnv.validate` hiện từ chối schema chứ không đánh giá ngữ nghĩa của ID/tham số tồn tại.

## Khuyến nghị hiện tại

Chưa nên đưa vào baseline public. Trước tiên dùng log dev để đếm `PARSE_FAIL` và `SCHEMA_REJECTED`. Chỉ làm prototype nếu hai nhóm này chiếm tỷ lệ đáng kể; so sánh với cùng checkpoint, cùng dev và cùng prompt, đo lại state-match, call diagnostics và latency. Chưa có số đo nào chứng minh điểm tăng hay tốc độ chấp nhận được.
