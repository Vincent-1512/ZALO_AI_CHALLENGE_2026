# TÀI LIỆU KIẾN TRÚC & HƯỚNG DẪN TRIỂN KHAI LIFEMATE AGENT
## ZALO AI CHALLENGE 2026 — TRACK: VOICE ASSISTANT AGENT

---

## 1. TỔNG QUAN BÀI TOÁN & THÁCH THỨC CỐT LÕI

### 1.1. Yêu cầu cuộc thi
- **Đầu vào:** File âm thanh giọng nói tiếng Việt (`.wav`), kèm ngữ cảnh thời gian (`current_time`, `timezone`, `location`), hồ sơ người dùng (`user_profile`), ứng dụng đã cài (`installed_apps`), và lịch sử hội thoại (`history`).
- **Nhiệm vụ:** Agent phải tương tác đa bước (Multi-step ReAct Loop) với môi trường giả lập ứng dụng (`ToolEnv`), bốc đúng công cụ (`tool_calls`), truyền đúng tham số (`arguments`), và đạt trạng thái khớp tuyệt đối (`state_hash`).
- **Đầu ra:** File `submission.jsonl` chứa `id`, `tool_calls` và `state_hash`.

### 1.2. Thách thức lớn nhất
1. **Lỗi ASR (Nhận dạng giọng nói):** Tiếng ồn môi trường, từ ngữ địa phương, phát âm lóng/ngắn ("bắn hai trẹo rưỡi", "báo thức năm rưỡi sáng mai") làm ASR dịch sai chính tả hoặc sai lệch ý định.
2. **Không gian công cụ khổng lồ:** 36 ứng dụng, 122 công cụ khác nhau (bao gồm **11 ứng dụng hoàn toàn mới** ở tập Public Test — bài toán Zero-Shot). Nếu nhồi 122 tool vào prompt thì LLM sẽ bị quá tải context, suy luận chậm và dễ ảo giác (hallucination).
3. **Bài toán số học & thời gian qua đêm:** Nếu người dùng yêu cầu hẹn giờ 05:30 sáng vào lúc 16:10 chiều, hệ thống phải tự động tính ra ngày mai (+1 day) chuẩn ISO-8601 kèm múi giờ.
4. **Giới hạn phần cứng & quy chế thi cực kỳ nghiêm ngặt:**
   - LLM: **< 5.0 Tỷ tham số (< 5B)**.
   - Tổng toàn bộ hệ thống: **< 6.0 Tỷ tham số (< 6B)**.
   - Chạy offline 100% trong Docker trên **1 GPU NVIDIA A30 (24GB VRAM)**, thời gian $\le 30\text{ giây / mẫu}$.

---

## 2. KIẾN TRÚC ĐỘT PHÁ: "DUAL-LLM + TOOLNET GRAPH ENGINE"

Để giải quyết triệt để các thách thức trên mà không vi phạm bất kỳ quy chế nào, hệ thống sử dụng kiến trúc **3 Tầng phối hợp (Multi-Stage Pipeline)**:

```mermaid
flowchart TD
    Audio["1. File âm thanh (.wav)"] --> ASR["PhoWhisper-medium (~0.77B)<br/>Nhận dạng ra transcript thô"]
    
    ASR --"Transcript thô"--> Rewriter["2. QUÂN SƯ SUY LUẬN (Reasoning Rewriter)<br/>DeepSeek-R1-Distill-Qwen-1.5B (~1.54B)<br/>- Suy nghĩ logic trong thẻ <think>...</think><br/>- Sửa lỗi chính tả ASR, phương ngữ, tiếng lóng<br/>- Tính toán mốc ngày giờ ISO chuẩn xác<br/>- Xuất câu lệnh tiếng Việt chuẩn mực (Canonical)"]

    Rewriter --"Câu lệnh chuẩn mực"--> ToolGraph["3. TOOLNET GRAPH ENGINE (Python In-Memory, <= 0.5ms)<br/>- BipartiteToolGraph: Lọc 122 tools -> Top 10-15 tools bằng wRRF<br/>- ToolNet Bottom-up Adaptation: Dìm điểm tool lỗi, đẩy tool thay thế"]

    ToolGraph --"Prompt tinh gọn + Top 10 Tools"--> ActionLLM["4. THỐNG SOÁI HÀNH ĐỘNG (Action Policy)<br/>Qwen2.5-3B-Instruct (~3.09B + LoRA)<br/>- Đọc prompt đã dọn sẵn<br/>- Chỉ tập trung 1 việc duy nhất: Nhả đúng JSON Tool Call"]

    ActionLLM --'{"action": "call", ...}'--> ToolEnv["5. Thực thi ToolEnv<br/>(Lưu kết quả -> Gọi lại LLM)"]
    ActionLLM --'{"action": "stop"}'--> Finish(["KẾT THÚC MẪU TEST"])
    ToolEnv --> ToolGraph
```

---

## 3. BẢNG CÂN ĐỐI THAM SỐ & VRAM (COMPLIANCE BUDGET)

Hệ thống được thiết kế nằm gọn trong **"Vùng an toàn tuyệt đối"** của Ban tổ chức:

| Thành phần | Model ID / Công nghệ | Số tham số thực tế | Giới hạn quy chế | Đánh giá |
| :--- | :--- | :--- | :--- | :---: |
| **Mô hình ASR** | `vinai/PhoWhisper-medium` | **769 Triệu** ($0.769\text{B}$) | — |  Hợp lệ |
| **Reasoning Model** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | **1.54 Tỷ** ($1.540\text{B}$) | — |  Hợp lệ |
| **Action LLM** | `Qwen/Qwen2.5-3B-Instruct` + LoRA | **3.10 Tỷ** ($3.104\text{B}$) | — |  Hợp lệ |
| **ToolNet Graph** | Code Python In-Memory | **0 tham số** (Thuật toán đại số) | — |  Hợp lệ |
| **TỔNG LLM** | **Reasoning (1.54B) + Action (3.10B)** | **4.64 Tỷ ($4.644\text{B}$)** | **< 5.00B** |  **DƯ 0.356B** |
| **TỔNG HỆ THỐNG** | **ASR (0.77B) + Cả 2 LLMs (4.64B)** | **5.41 Tỷ ($5.413\text{B}$)** | **< 6.00B** |  **DƯ 0.587B** |
| **VRAM chạy trên A30** | Load toàn bộ ở định dạng bfloat16/float16 | **~12.5 GB / 24 GB** | **24 GB** |  **DƯ ~11.5 GB** |
| **Thời gian xử lý / mẫu** | ASR (1s) + Rewriter (1.5s) + Action (2s) | **~4.5 – 6.0 giây / mẫu** | **< 30 giây** |  **DƯ ~24 giây** |

---

## 4. CẤU TRÚC CODEBASE & TRÁCH NHIỆM TỪNG FILE

Người triển khai cần chú ý các module chính trong thư mục `src/lifemate/` và `training/`:

```
ZALO_AI_CHALLENGE_2026/
├── configs/
│   └── baseline.json                 # Cấu hình chuẩn cho ASR, Rewriter, Action LLM và siêu tham số
├── src/lifemate/
│   ├── reasoning_rewriter.py         # [TẦNG 1] Module bọc DeepSeek-R1-1.5B sửa lỗi ASR & chuẩn hóa câu
│   ├── graph_retrieval.py            # [TẦNG 2] ToolNet: Đồ thị Bipartite Tool Graph lọc công cụ bằng wRRF & thích ứng lỗi
│   ├── prompts.py                    # Điểm hội tụ build_messages(): ráp nối ToolNet vào prompt cho LLM
│   ├── models.py                     # Quản lý load PhoWhisper, Qwen2.5-3B và adapter LoRA
│   ├── middleware.py                 # Bộ lọc an toàn: vá lỗi định dạng JSON, ép kiểu tham số, Reflexion feedback
│   └── runner.py                     # Vòng lặp ReAct trực tuyến lúc chạy test public / docker
├── training/
│   ├── prepare_sft.py                # Sinh dữ liệu huấn luyện đa bước (kèm Hammer Masking)
│   └── train_lora.py                 # Huấn luyện QLoRA 4-bit / bfloat16 cho Qwen2.5-3B
└── notebooks/
    ├── eval_asr_nbest_rewriter.ipynb # Đánh giá chất lượng ASR N-best hypotheses & LLM 1.5B Rewriter
    └── lifemate_colab_train.ipynb    # Notebook chạy thử nghiệm huấn luyện trên Colab / A100
```

---

## 5. HƯỚNG DẪN 5 BƯỚC TRIỂN KHAI CHO KỸ SƯ TIẾP NHẬN

### Bước 1: Tải các mô hình nền tảng (Weights)
Chạy script tự động tải 3 mô hình về thư mục `models/` (đã cấu hình sẵn trong `configs/baseline.json`):
```bash
python scripts/download_models.py --models asr rewriter llm
```
- ASR: `models/PhoWhisper-medium`
- Rewriter: `models/DeepSeek-R1-Distill-Qwen-1.5B`
- LLM: `models/Qwen2.5-3B-Instruct`

### Bước 2: Thiết lập ToolNet Graph (`graph_retrieval.py`) & Nhúng vào `prompts.py`
1. Đảm bảo `BipartiteToolGraph` trong `graph_retrieval.py` thực hiện xếp hạng wRRF và thích ứng đồ thị động (Bottom-up Graph Adaptation: dìm điểm tool bị lỗi, đẩy tool thay thế).
2. Trong `src/lifemate/prompts.py`, hàm `build_messages()` gọi `available_tool_specs()` nhận cả `prior_calls` và `prior_results` để đồ thị cập nhật thích ứng theo thời gian thực.

### Bước 3: Tạo tập dữ liệu huấn luyện SFT (Multi-turn Trajectory)
Mỗi mẫu trong tập train phải được rã thành các bước kế tiếp (`Step 1` $\rightarrow$ `Step 2` $\rightarrow$ `Stop`):
```bash
python training/prepare_sft.py --data-root data/raw/train+public_test --output-dir data/processed/sft
```
*Kết quả sinh ra 3 file: `train.jsonl`, `dev.jsonl`, `holdout.jsonl`.*

### Bước 4: Huấn luyện LoRA cho Qwen2.5-3B trên A100 / GPU
Chạy script huấn luyện với các siêu tham số tối ưu:
```bash
python training/train_lora.py \
    --train-jsonl data/processed/sft/train.jsonl \
    --dev-jsonl data/processed/sft/dev.jsonl \
    --model-path models/Qwen2.5-3B-Instruct \
    --output-dir models/lifemate-qwen2.5-3b-lora \
    --epochs 3 \
    --learning-rate 2e-4 \
    --max-sequence-length 4096 \
    --gradient-accumulation-steps 4
```
*Trên A100, bật `bf16=True` và `flash_attention_2` để hoàn thành quá trình train trong ~8 phút.*

### Bước 5: Chạy suy luận toàn diện trên Public Test
Chạy kiểm thử toàn bộ luồng tích hợp:
```bash
python -m lifemate.runner \
    --data-path data/raw/train+public_test \
    --split public_test \
    --output submission/public_test_predictions.jsonl \
    --config configs/baseline.json \
    --adapter-path models/lifemate-qwen2.5-3b-lora/final
```

---

## 6. DANH SÁCH KIỂM TRA (CHECKLIST) TRƯỚC KHI ĐÓNG GÓI DOCKER

- [ ] Tổng số tham số của 3 mô hình không vượt quá 6.0B (kiểm tra bằng script `sum(p.numel() for p in model.parameters())`).
- [ ] Phần LLM không vượt quá 5.0B.
- [ ] Môi trường Docker chạy hoàn toàn OFFLINE (`HF_HUB_OFFLINE=1`, không gọi API ra ngoài).
- [ ] VRAM chạy ổn định dưới 16GB, không phát sinh lỗi `CUDA OOM`.
- [ ] Thời gian xử lý trung bình mỗi sample $\le 8$ giây.
- [ ] Trạng thái kết thúc của từng mẫu phải có `{"action": "stop"}` hoặc chạm ngưỡng an toàn `max_tool_steps = 6`.

