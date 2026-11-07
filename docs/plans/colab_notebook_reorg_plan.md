# Kế hoạch tổ chức lại baseline Colab

## Mục tiêu

Tạo một quy trình Colab có thể chạy lại từ đầu sau khi runtime mất, lấy dataset từ một đường dẫn Drive cố định, tách rõ chuẩn bị/smoke/full/resume và lưu mọi log/checkpoint/adapter vào đúng thư mục run.

## Quy ước Drive

```text
MyDrive/AI-ZALO/
  dataset/train+public_test/train/{train.jsonl,train_tools_catalog.json}
  bundles/lifemate-colab-bundle-v3-*.tar.gz
  lifemate_runs/<RUN_ID>/
```

Notebook chỉ dùng hai file train được phép (`train.jsonl` và `train_tools_catalog.json`) được đặt trong Drive. Bundle chứa mã nguồn/config, không nhúng dataset.

## Thiết kế notebook

- Một cell cấu hình duy nhất: `ACTION`, `RUN_ID`, đường dẫn Drive, seed, epochs, learning rate, gradient accumulation, context ceiling, smoke limits và checkpoint cadence.
- `ACTION` có bốn giá trị rõ ràng: `prepare`, `smoke`, `full`, `resume`. Full mới không tự resume; resume chỉ chọn checkpoint đầy đủ.
- Chạy lại từ đầu sau reset bằng `Run all`; cùng `RUN_ID` khôi phục đúng thư mục. Smoke/full không chạy lại trừ khi được chọn trong cấu hình.
- Smoke adapter, full adapter, checkpoints, báo cáo và log nằm trên Drive. Smoke output ghi trực tiếp vào Drive để tránh mất adapter giữa lúc train xong và lúc sao chép.
- Notebook kiểm tra dữ liệu, bundle, thư mục run và adapter trước khi chạy; lỗi báo rõ đường dẫn cần sửa.

## Tệp dự kiến

- `notebooks/lifemate_colab_train.ipynb`: notebook v3.
- `scripts/build_colab_bundle.py`: đóng gói source không kèm dataset, tạo artifact v3 và checksum.
- `docs/colab_runbook.md`, `artifacts/colab/UPLOAD_README.md`, `README.md`: thống nhất cách upload/chạy/resume.
- `artifacts/colab/`: notebook và bundle v3 phát hành.

## Kiểm tra

- Kiểm tra cấu trúc notebook và cú pháp các cell bằng môi trường local.
- Tạo bundle v3 và xác nhận bundle không chứa dataset.
- Không thể xác nhận GPU/T4 hoặc chạy full training trong môi trường local; smoke mới trên Colab vẫn là bước kiểm chứng thực tế.
