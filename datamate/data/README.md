# DataMate data: tải, lưu và giải nén

## Bản dữ liệu hiện có

| Gói | Kích thước archive | Đã giải nén tại |
|---|---:|---|
| `train.zip` | 128.118.095 byte | `datamate/data/raw/train/` |
| `public_test.zip` | 64.149.714 byte | `datamate/data/raw/public_test/` |

Train chứa `questions.jsonl` và `answers.jsonl`; public test chỉ chứa câu hỏi cùng gói dữ liệu, không có nhãn đáp án. Thống kê chi tiết ở [dataset analysis](../docs/dataset_analysis.md).

## Tải dữ liệu

Đăng nhập [portal DataMate](https://challenge.zalo.ai/portal/datamate/) bằng tài khoản đội, tải các gói được BTC cấp ở mục Data, rồi lưu **nguyên file tải về** vào:

```text
datamate/data/archives/
```

Không đặt chúng vào `data/archives/` ở gốc repo: vị trí đó dành cho kit LifeMate. Tên file và cấu trúc archive có thể theo bản phát hành của BTC; giữ tên gốc để dễ đối chiếu checksum/README.

## Kiểm tra archive trước khi giải nén

Từ thư mục gốc repo, liệt kê file đã tải:

```sh
find datamate/data/archives -maxdepth 1 -type f -print
```

Với mỗi file ZIP, xem danh sách bên trong trước (thay `FILE.zip` bằng tên thật):

```sh
unzip -l "datamate/data/archives/FILE.zip" | head -60
```

Với hai archive hiện có, từ thư mục gốc repo có thể giải nén đúng cấu trúc top-level `train/` và `public_test/` như sau:

```sh
unzip -n -q datamate/data/archives/train.zip -d datamate/data/raw
unzip -n -q datamate/data/archives/public_test.zip -d datamate/data/raw
```

Nếu tải bản kit mới, trước khi giải nén kiểm tra `unzip -l` và README: ZIP hiện tại đã có thư mục gốc `train/` hoặc `public_test/`, nên đích giải nén là `datamate/data/raw/` để không tạo đường dẫn lồng hai tầng.

## Layout đích mong muốn

```text
datamate/data/
  archives/             # ZIP gốc BTC; giữ nguyên
  raw/
    train/              # train sau giải nén
    public_test/         # public test sau giải nén
    private_test/        # chỉ tạo khi BTC phát hành; không dùng để train/tune
  processed/            # dữ liệu dẫn xuất; không ghi đè raw
```

Tên file và thư mục bên trong từng split phải theo README/schema của kit DataMate, không suy đoán từ cấu trúc LifeMate. DataMate có thể chứa nhiều doanh nghiệp và tệp bảng; parser cần khám phá CSV/XLSX/SQLite theo metadata/README thay vì giả định một file duy nhất.

## Quy tắc bảo quản

- Các thư mục `archives/`, `raw/` và `processed/` được Git bỏ qua; không commit dữ liệu thi lên GitHub.
- Giữ archive gốc để có thể kiểm tra lại và bung lại sạch.
- Không chỉnh sửa dữ liệu raw; ghi mọi bước làm sạch/chuyển đổi vào pipeline và ghi kết quả ở `processed/`.
- Không đọc hoặc gán nhãn public/private test để huấn luyện hay tune. Dùng train để phát triển; tuân thủ chính xác quyền truy cập được BTC nêu cho mỗi split.
- Dataset cuộc thi chỉ dùng theo giấy phép và quy định của BTC; không chia sẻ ra ngoài đội.
