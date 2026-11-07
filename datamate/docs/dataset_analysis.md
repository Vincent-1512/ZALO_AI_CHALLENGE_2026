# DataMate: phân tích đề và dữ liệu

**Kiểm kê local:** 07/11/2026. Phạm vi: hai archive BTC đã tải, train labels, README/rulebook từng split và cấu trúc các tệp. Không dùng public test để tạo nhãn hay huấn luyện/tune.

## Tóm tắt

DataMate là bài hỏi đáp tiếng Việt trên dữ liệu bảng của từng doanh nghiệp. Hệ thống phải tìm đúng dữ liệu trong CSV/XLSX/SQLite, hiểu câu hỏi và `as_of_date`, áp đúng rulebook của split, tính kết quả chính xác rồi xuất JSONL đúng kiểu. Bài này không giải tốt bằng cách đưa nguyên mọi file cho LLM và yêu cầu “tính giúp”: phần lớn đầu ra là số cần khớp chính xác, trong khi bảng lớn và nghiệp vụ thay đổi theo shop.

Hướng baseline nên là **LLM phân tích câu hỏi thành một kế hoạch có cấu trúc; Python/SQL đọc và tính trên dữ liệu; bộ kiểm tra quyết định trả lời, unanswerable hay abstain**. LLM không tự bịa con số cuối.

## Bộ dữ liệu trong máy

| Split | Câu hỏi | Doanh nghiệp | File bảng trong thư mục doanh nghiệp | Excel sheets |
|---|---:|---:|---:|---:|
| train | 3.000 | 30 | 128: 107 XLSX, 20 CSV, 1 SQLite | 302 |
| public_test | 1.000 | 14 | 80: 69 XLSX, 10 CSV, 1 SQLite | 136 |

Train có đúng 3.000 nhãn, ID không trùng, không thiếu đáp án và không có nhãn thừa. `as_of_date` của train nằm từ 2025-01-15 đến 2026-09-20. Độ dài câu hỏi: trung vị 59 ký tự, khoảng 18–200.

`train.zip` có 128.118.095 byte và khoảng 185 MiB nội dung giải nén; `public_test.zip` có 64.149.714 byte và khoảng 104 MiB giải nén. Train có thêm một SQLite 17 bảng/268.911 dòng; các workbook chứa nhiều sheet và hàng dữ liệu. Không nên nạp lại toàn bộ XLSX cho từng câu hỏi: đọc/cache mỗi doanh nghiệp một lần.

Các đường dẫn hiện tại:

```text
datamate/data/archives/train.zip
datamate/data/archives/public_test.zip
datamate/data/raw/train/{questions.jsonl, answers.jsonl, rulebook.md, calendar.csv, data/}
datamate/data/raw/public_test/{questions.jsonl, rulebook.md, calendar.csv, data/}
```

Public test không có `answers.jsonl`. Chỉ dùng public test để tạo bài nộp và đánh giá theo portal; không dùng nội dung câu hỏi để tạo nhãn hoặc tune theo từng câu.

## Phân bố nhãn train

| Kiểu đáp án | Số | Tỷ lệ |
|---|---:|---:|
| number | 1.884 | 62,8% |
| percent | 578 | 19,3% |
| list | 193 | 6,4% |
| date | 145 | 4,8% |
| string | 88 | 2,9% |
| unanswerable | 70 | 2,3% |
| boolean | 23 | 0,8% |
| table | 19 | 0,6% |

Number và percent chiếm **82,1%**; lỗi dấu phân cách, làm tròn, đơn vị hoặc điều kiện lọc đều có thể biến đáp án thành sai. Có 167 record mang `alias` (42 string, 106 list, 19 table); cần xem đúng semantics của evaluator trước khi viết scorer cục bộ, không tự coi alias là fuzzy matching.

Phân bố câu hỏi theo doanh nghiệp lệch: 23 shop có 118–119 câu, 7 shop chỉ có 40 câu. Chia ngẫu nhiên câu hỏi khiến câu cùng doanh nghiệp lọt cả train và dev, tạo điểm dev lạc quan. Dev phải giữ trọn doanh nghiệp ra ngoài.

Một lượt dò từ khóa cho thấy thời gian xuất hiện rất thường xuyên (khoảng 2.053/3.000 câu có từ chỉ ngày/tuần/tháng/quý/năm hoặc ca). Phân nhóm từ khóa có thể chồng lấn và chỉ để định hướng kiểm kê, không phải nhãn chủ đề chính thức.

## Độ biến thiên cấu trúc

Mỗi doanh nghiệp có README riêng, một hoặc nhiều tệp bảng. Train thực tế có các dạng:

- CSV phẳng, CSV nhiều bảng nối bằng mã.
- Excel xuất từ POS với hóa đơn, chi tiết hóa đơn, sản phẩm, khách, thanh toán, kho và nhập hàng ở nhiều file hoặc sheet.
- Sổ bán hàng thủ công: mỗi tháng một sheet, có viết tắt, tiêu đề nhiều dòng, dòng hàng và dòng ghi chú.
- SQLite POS nhiều bảng; ví dụ train có bảng hàng hóa, đơn hàng, chi tiết, thanh toán, lô thuốc, tồn kho và trả hàng.
- Dữ liệu theo lô/hạn dùng, quy đổi đơn vị, nhiều đợt giao; tệp có thể phân theo tháng/năm.

Vì vậy, cần đọc README và khám phá schema cho từng business; không giả định header nằm dòng đầu, tên cột chuẩn hóa, một sheet duy nhất, hay mọi dòng đều là một đơn hàng. Với sổ ghi tay, ô trống có thể kế thừa thông tin từ dòng đầu nhóm. Với hóa đơn nhiều dòng, trạng thái/khách có thể lặp hoặc để trống.

Public input cho thấy ít nhất ba dạng cửa hàng không có trong README train: sửa xe (`dev_012`), mẹ và bé (`dev_013`), cửa hàng điện thoại (`dev_014`), tổng cộng 170 câu hỏi public. Điều này làm lộ điểm yếu của cách ghi luật riêng theo `business_id`: hệ thống cần xử lý schema mới theo README và luật ngành, không nhận dạng shop bằng ID. Rulebook public cũng mở rộng/điều chỉnh so với train; hash hai file khác nhau.

## Luật nghiệp vụ dễ làm sai

Các quy tắc chính nằm trong `data/raw/<split>/rulebook.md`; README từng shop bổ sung cách hiểu tên file/cột và liên kết bảng.

1. **Doanh thu không phải tiền thu:** theo ngày đặt và giá trị dòng hàng sau giảm giá dòng; không trừ voucher cấp đơn, không cộng VAT/phí giao. Đơn đang giao vẫn tính; đơn hủy hoặc có trả hàng (kể cả trả một phần) bị loại khỏi doanh thu.
2. **Mốc thời gian là `as_of_date` của câu hỏi**, không phải ngày chạy agent. “Tuần này”, “tuần rồi”, “tháng này”, kỳ so sánh và Tết có quy ước riêng; tra Tết từ `calendar.csv`. Ngày phát sinh và số dư “đến hết ngày” là hai loại truy vấn khác nhau.
3. **Công nợ là một luồng riêng:** đơn trả hàng vẫn có thể còn nợ; COD/sàn không phải nợ; tiền trả cùng ngày mua không phải trả nợ sau ngày mua. Cần nối lịch sử thanh toán theo đúng đơn và ngày.
4. **Số lượng phải quy đổi theo mặt hàng:** xếp hạng bán chạy theo đơn vị cơ bản, không theo số dòng hay tiền; tỷ lệ quy đổi không đồng nhất giữa shop/sản phẩm.
5. **Tồn kho và giá vốn:** tồn cuối ngày, lô thuốc hết hạn và giá trị tồn có cách tính khác nhau. Khi rulebook nói dùng giá vốn ghi trên dòng bán thì không tự dựng lại từ giá nhập.
6. **Dòng hàng và đơn hàng không đồng nhất:** phải group theo đúng mã đơn; bỏ bản ghi xuất trùng trong cùng đơn theo rulebook, nhưng không xóa hai dòng giống nhau thuộc hai đơn khác nhau.
7. **Không có dữ liệu không có nghĩa là 0.** Nếu kỳ nằm ngoài thời gian có dữ liệu hoặc thiếu loại sổ bắt buộc, đáp án là `unanswerable`; nếu hệ thống không đủ tự tin thì `abstain`.
8. **Ngành có luật riêng.** Public rulebook thêm các quy tắc cho mẹ và bé, sửa xe, điện thoại, ca tiền mặt, khách trung thành, giờ cao điểm và xếp hạng. Luôn dùng rulebook đúng split thay vì chỉ dùng luật train.

## Chấm điểm và đầu ra

Theo rulebook train, mỗi câu đúng được **+1**, sai **−1**, bỏ trống/`abstain` **0**. Vì sai bị phạt, hệ thống nên abstain khi xác suất đáp án đúng không vượt ngưỡng hòa vốn 50%; ngưỡng triển khai thực tế cần hiệu chỉnh bằng dev theo doanh nghiệp. Nếu dữ liệu rõ ràng không hỗ trợ câu hỏi, trả `unanswerable` thay vì abstain.

Format phải đúng kiểu: number/percent là JSON number; tiền là số nguyên VND; percent ghi 12.4 cho 12,4%; date dùng ISO; boolean là `true/false`; list luôn là array; table là `{"columns": [...], "rows": [...]}`. Tiền/số đếm so khớp chính xác; percent đúng hai chữ số thập phân; tên phải lấy từ dữ liệu và giữ đúng dấu. Tie-break theo mã, hoặc tên khi không có mã.

## Baseline đề xuất

```text
questions.jsonl + as_of_date
          │
          ▼
Business loader: README + rulebook + CSV/XLSX/SQLite, cache theo doanh nghiệp
          │
          ▼
Question planner: tiếng Việt → JSON plan có measure, filters, period,
group-by, sort/top-k, output type, bảng/cột và rule clauses
          │
          ▼
Allowlisted deterministic executor: joins + dates + business aggregations
          │
          ▼
Answerability/confidence → exact type/rounding → schema validation → JSONL
```

- LLM dùng để hiểu cách hỏi/ánh xạ vào plan, không tính số bằng văn bản tự do. Plan nên là DSL/JSON có schema và các phép toán cho phép; không chạy Python tùy ý do model sinh.
- Tạo adapter đọc CSV/XLSX/SQLite và lưu nguồn gốc từng giá trị (file, sheet, dòng, cột) để truy vết lỗi. Parse README/rulebook ở inference; chuẩn hóa kiểu dữ liệu nhưng giữ nguyên mã/tên gốc.
- Bắt đầu bằng câu hỏi phổ biến: lọc kỳ, doanh thu/số đơn, so sánh kỳ, top-k, công nợ, tồn kho. Mỗi phép tính phải có định nghĩa testable, tie-break và làm tròn riêng.
- Dev split theo **business_id và nhóm schema/ngành**, không chia ngẫu nhiên từng câu. Báo cáo utility tổng, từng answer type, từng business, từng dạng file và tỷ lệ abstain/unanswerable. Có thể dùng GroupKFold trên train để thấy độ dao động.
- Chỉ cân nhắc fine-tune sau khi pipeline đọc bảng và executor chạy ổn; 3.000 cặp hỏi–đáp không có SQL/trace giải thích nên SFT trực tiếp câu→đáp án dễ học thuộc, không chỉ cách truy vấn.
- Pin model/dependencies, xác minh license và mọi giới hạn của Rule/Docker DataMate hiện hành; inference offline, không gọi API ngoài.

## Việc nên làm tiếp theo

1. Lập catalog schema từ 30 README train: loại shop, danh sách file/sheet/bảng, cột, khóa nối, date range, quy ước dòng.
2. Dựng parser chung và một bộ normalized view có provenance; ưu tiên đọc stream/cache thay vì lặp đi lặp lại XLSX lớn.
3. Tạo dev group-holdout trên train; đánh giá baseline an toàn trước khi chọn model.
4. Cài planner/executor theo nhóm phép tính, bắt đầu từ doanh thu và thời gian; thêm công nợ, kho, quy đổi, ngành theo từng lát.
5. Khóa scorer cục bộ và validator JSONL dựa theo rulebook/evaluation chính thức; dùng public test cho inference/submission đúng quy định.

## Nguồn trong workspace

- Dataset train: `data/raw/train/`; nhãn: `data/raw/train/answers.jsonl`.
- Luật train/public: `data/raw/train/rulebook.md`, `data/raw/public_test/rulebook.md`.
- README cấu trúc: `data/raw/<split>/data/<business_id>/README.md`.
- Calendar: `data/raw/<split>/calendar.csv`.
- Mô tả bài DataMate và portal: [challenge.zalo.ai/portal/datamate](https://challenge.zalo.ai/portal/datamate/).
