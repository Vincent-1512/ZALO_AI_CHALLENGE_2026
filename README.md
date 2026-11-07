# Zalo AI Challenge 2026 — LifeMate

Workspace của đội 5 người cho bài toán nghe yêu cầu tiếng Việt và thực hiện hành động qua công cụ BTC.

## Bắt đầu nhanh

- **LifeMate trên Vast:** upload [notebook GitHub resume](notebooks/lifemate_vast_github_resume.ipynb) rồi bấm Run All. Notebook tải bundle đã ghim phiên bản từ GitHub Release. Xem [hướng dẫn Vast/GitHub](docs/vast_github_resume.md).
- **LifeMate trên Colab:** bản dự phòng ở [notebook Colab](notebooks/lifemate_colab_train.ipynb); không dùng cho luồng Vast hiện tại.
- **DataMate:** workspace riêng trong [`datamate/`](datamate/README.md), có notebook và dữ liệu đầu ra độc lập.

## Tài liệu tham khảo

1. [Chiến lược chất lượng](docs/plans/lifemate_strategy.md): ưu tiên thí nghiệm, khả năng dùng app mới, đánh giá và chọn mô hình.
2. [Kế hoạch triển khai](docs/plans/implementation_plan.md): phân công, lịch từng ngày, GPU, Docker và checklist nộp bài.
3. [Yêu cầu LifeMate](docs/competition/lifemate.md): dữ liệu, luật kỹ thuật, chấm điểm và cách nộp.
4. [Runbook Vast.ai](docs/vast_runbook.md): quy trình Vast thủ công và các bước train.
5. [Runbook Colab](docs/colab_runbook.md): quy trình Colab dự phòng.
6. [Thông tin cuộc thi](docs/competition/overview.md) và [thể lệ nguồn](docs/competition/regulations/).

## Trạng thái ngày 07/11/2026

- Đã nhận và giải nén kit v3.2.0: 1.545 train, 634 public test.
- Đã kiểm kê audio/history/catalog và chốt thiết kế baseline ASR → schema-aware agent → ToolEnv; chi tiết ở [kiểm kê dữ liệu](docs/analysis/dataset_audit.md) và [thiết kế baseline](docs/baseline_design.md).
- Đã chuẩn bị runner offline, pipeline tạo trajectory SFT và script QLoRA. Run Vast `vast-20261106-124816` có checkpoint 254 và 508; checkpoint 508 đã được đóng gói để resume. Public inference đã tạo `lifemate_runs/vast-20261106-124816/public/submission.jsonl` (634 dòng).
- Bộ kiểm tra schema trong kit chưa được xác nhận là scorer đầy đủ. Hình công thức Evaluation còn cần bổ sung.
- Notebook Colab v3 tách dữ liệu train khỏi code bundle. Hai file `train.jsonl` và `train_tools_catalog.json` nằm ở `MyDrive/AI-ZALO/dataset/train+public_test/train/`; bundle ở `bundles/` và output ở `lifemate_runs/<RUN_ID>/`. Action/siêu tham số nằm ở cell đầu; full và resume là hai action riêng.
- Thư mục `datamate/` là workspace tách riêng cho bài DataMate; không dùng notebook hoặc dữ liệu trong đó cho LifeMate.

## Cấu trúc

```text
README.md
.gitignore
docs/
  analysis/
    dataset_audit.md
  baseline_design.md
  competition/
    lifemate.md
    overview.md
    regulations/             # PDF và bản text thể lệ công khai
  plans/
    implementation_plan.md
    lifemate_strategy.md
src/lifemate/                 # runner, prompts, model adapters
scripts/                      # inference và model download
training/                     # tạo SFT trajectory và QLoRA
configs/baseline.json         # model revisions, giới hạn runtime và train
requirements-*.txt            # môi trường inference / train
data/
  README.md                  # Hướng dẫn đặt dữ liệu, được commit
  archives/                  # Gói tải BTC, chỉ local
  raw/train+public_test/     # Kit giải nén, chỉ local
artifacts/
  vast_upload/               # Bundle Vast dự phòng để upload thủ công; Git bỏ qua
  colab/                     # Bundle Colab dự phòng; Git bỏ qua
lifemate_runs/               # Checkpoint, backup và submission LifeMate; Git bỏ qua
datamate/                    # Workspace riêng cho bài DataMate
```

Clone repo không tải kèm dataset. Thành viên đã đăng ký lấy kit từ portal BTC và đặt theo [hướng dẫn dữ liệu](data/README.md). Không commit dữ liệu thi, trọng số, secrets hoặc kết quả chứa nội dung dữ liệu.

Lệnh chuẩn bị SFT, train LoRA và chạy public được ghi ở [thiết kế baseline](docs/baseline_design.md). Các model phải tải theo revision pin rồi đóng gói offline; inference không gọi dịch vụ ngoài.

## Các mốc nộp bài

Giờ Việt Nam (GMT+7), theo Submission Guidance đội cung cấp; theo dõi portal nếu BTC cập nhật.

| Mốc | Việc |
|---|---|
| 16/11/2026, 23:59 | Docker solution: training code, inference code, data theo yêu cầu BTC; checksum và link Drive qua Solution Submission; tối đa 2 lần, chỉ xét lần cuối |
| 20/11/2026, 09:00 | Phát hành private test sau khi đóng băng giải pháp |
| 23/11/2026, 23:59 | Nộp JSONL private test |

Public JSONL được nộp qua Submission, tối đa 5 lần/ngày. [Portal LifeMate](https://challenge.zalo.ai/portal/lifemate) là nguồn hướng dẫn cập nhật; repo không thay thế thể lệ chính thức.
