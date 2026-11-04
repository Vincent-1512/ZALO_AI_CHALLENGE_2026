# Dữ liệu local

Dataset chỉ dành cho đội đã đăng ký; README trong kit yêu cầu không phát tán/phân phối. Thư mục raw và archives được Git bỏ qua. Không đưa nội dung kit lên GitHub.

## Vị trí

```text
data/archives/train+public_test.zip
data/raw/train+public_test/
  README.md
  VERSION
  train/
  public_test/
```

Thành viên lấy gói từ mục Data của [portal LifeMate](https://challenge.zalo.ai/portal/lifemate) bằng tài khoản có quyền. Từ gốc repo, sau khi đặt zip vào archives:

```sh
mkdir -p data/raw
unzip -n -q 'data/archives/train+public_test.zip' 'train+public_test/*' -d data/raw
```

Dataset root cho runner tương lai là `data/raw/train+public_test`, tức thư mục chứa `train/` và `public_test/`. Dùng đúng thư viện `zaic_tools` của split đang chạy. Không sửa kit gốc.

## Kiểm kê ngày 04/11/2026

| Thuộc tính | Giá trị |
|---|---|
| Release | v3.2.0 |
| Build trong VERSION | 2026-10-24T23:35:25+0700 |
| Zip | 876.430.231 byte, khoảng 836 MiB |
| Trên đĩa sau giải nén | khoảng 1,1 GB theo `du -sh` |
| Train / public | 1.545 / 634 mẫu |
| Train / public WAV | 5.036 / 1.933, gồm audio history |

Không đọc/copy/parse file trong `<split>_environment/`, kể cả `state.json`. Chỉ truy cập dữ liệu ứng dụng qua API `zaic_tools` được phép; đúng một ToolEnv/mẫu trong inference. Train có transcript/labels; public không có chúng. Không gán nhãn test để huấn luyện hoặc tune.

Private test sẽ được BTC phát hành sau khi đóng băng giải pháp. Khi nhận, giữ bên dưới `data/` để tiếp tục được Git bỏ qua. Đường mount trong Docker chấm thi là `/data` theo Rule; đây là contract runtime, không phải yêu cầu đặt dataset local ở thư mục tuyệt đối đó.
