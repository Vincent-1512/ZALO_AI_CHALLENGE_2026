# Chạy LifeMate trên Google Colab

## Mục đích

Notebook `notebooks/lifemate_colab_train.ipynb` tải model nền, tạo SFT từ nhãn train, chạy smoke train QLoRA và sao lưu adapter lên Google Drive. Bundle chưa có checkpoint đã train. Full train mặc định tắt để tránh chạy job dài ngoài ý muốn.

Bundle Colab chứa source repo và đúng hai tệp dữ liệu train cần cho SFT: `train.jsonl` và `train_tools_catalog.json`. Nó không bao gồm audio, `public_test`, `private_test`, hay bất kỳ thư mục `*_environment/` nào. Không commit dữ liệu thi hoặc model weights vào GitHub.

## Cách chạy

1. Tải `lifemate-colab-bundle-<revision>.tar.gz` lên thư mục gốc `MyDrive`.
2. Mở notebook trong Colab, bật GPU, rồi chạy lần lượt từ trên xuống.
3. Kiểm tra GPU, dung lượng đĩa và checksum dữ liệu trước khi tải model.
4. Chạy smoke train 64 mẫu; adapter được sao lưu lên Drive ngay sau train. Nếu smoke chạy ổn và đội còn thời gian GPU, bật `RUN_FULL_TRAINING = True`; full train ghi checkpoints vào Drive. Chạy lại cell metadata sau khi hoàn tất.
6. Tải adapter từ Drive và đánh giá ở môi trường có `zaic_tools` theo quy định BTC. Notebook không thể chạy rollout ToolEnv vì bundle không chứa environment data.

## Model và tài nguyên

Qwen3-4B-Instruct-2507 không nằm trong bundle; notebook chỉ tải đúng revision LLM từ `configs/baseline.json`, vì PhoWhisper không cần trong lúc train adapter. Khi chạy inference/đánh giá end-to-end, tải thêm PhoWhisper bằng `scripts/download_models.py --models asr`. Colab cần internet trong giai đoạn chuẩn bị/tải model. Inference của sản phẩm nộp BTC vẫn phải chạy offline.

Smoke train mặc định chọn sequence length 4096 nếu GPU có ít nhất 22 GiB VRAM, 2048 nếu có ít nhất 14 GiB, và 1024 nếu thấp hơn. Nếu CUDA OOM, giảm `MAX_SEQUENCE_LENGTH` về 1024. Full train có thể dài hơn một phiên Colab. Checkpoints được giữ trên Drive nhưng notebook chưa tự resume nếu phiên bị ngắt.

## Lưu ý tuân thủ

Chỉ dùng split train có nhãn để chuẩn bị SFT. Không đưa `public_test`, `private_test` hoặc thư mục môi trường lên Colab qua bundle này. Không đọc trực tiếp state; khi đánh giá tool/state, chỉ dùng `zaic_tools` theo hướng dẫn BTC. Kiểm tra điều khoản BTC về dịch vụ bên thứ ba trước khi tải dữ liệu lên Colab.
