# Zalo AI Challenge 2026 — LifeMate

Workspace của đội 5 người cho bài toán nghe yêu cầu tiếng Việt và thực hiện hành động qua công cụ BTC.

## Đọc theo thứ tự

1. [Chiến lược chất lượng](docs/plans/lifemate_strategy.md): ưu tiên thí nghiệm, khả năng dùng app mới, đánh giá và chọn mô hình.
2. [Kế hoạch triển khai](docs/plans/implementation_plan.md): phân công, lịch từng ngày, GPU, Docker và checklist nộp bài.
3. [Yêu cầu LifeMate](docs/competition/lifemate.md): dữ liệu, luật kỹ thuật, chấm điểm và cách nộp.
4. [Thông tin cuộc thi](docs/competition/overview.md) và [thể lệ nguồn](docs/competition/regulations/).

## Trạng thái ngày 04/11/2026

- Đã nhận và giải nén kit v3.2.0: 1.545 train, 634 public test.
- Đã kiểm kê metadata và lập chiến lược; chưa có mã agent, checkpoint, điểm baseline hay kết quả huấn luyện.
- Bộ kiểm tra schema trong kit chưa được xác nhận là scorer đầy đủ. Hình công thức Evaluation còn cần bổ sung.
- Cần cấu hình/quyền truy cập GPU thuê, lịch làm mỗi người và ngân sách để chốt kế hoạch chạy.

## Cấu trúc

```text
README.md
.gitignore
docs/
  competition/
    lifemate.md
    overview.md
    regulations/             # PDF và bản text thể lệ công khai
  plans/
    implementation_plan.md
    lifemate_strategy.md
data/
  README.md                  # Hướng dẫn đặt dữ liệu, được commit
  archives/                  # Gói tải BTC, chỉ local
  raw/train+public_test/     # Kit giải nén, chỉ local
```

Clone repo không tải kèm dataset. Thành viên đã đăng ký lấy kit từ portal BTC và đặt theo [hướng dẫn dữ liệu](data/README.md). Không commit dữ liệu thi, trọng số, secrets hoặc kết quả chứa nội dung dữ liệu.

## Các mốc nộp bài

Giờ Việt Nam (GMT+7), theo Submission Guidance đội cung cấp; theo dõi portal nếu BTC cập nhật.

| Mốc | Việc |
|---|---|
| 16/11/2026, 23:59 | Docker solution: training code, inference code, data theo yêu cầu BTC; checksum và link Drive qua Solution Submission; tối đa 2 lần, chỉ xét lần cuối |
| 20/11/2026, 09:00 | Phát hành private test sau khi đóng băng giải pháp |
| 23/11/2026, 23:59 | Nộp JSONL private test |

Public JSONL được nộp qua Submission, tối đa 5 lần/ngày. [Portal LifeMate](https://challenge.zalo.ai/portal/lifemate) là nguồn hướng dẫn cập nhật; repo không thay thế thể lệ chính thức.
