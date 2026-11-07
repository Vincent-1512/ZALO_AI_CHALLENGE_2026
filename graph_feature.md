# Tích hợp Agent-as-a-Graph: Bipartite Knowledge Graph & wRRF cho LifeMate

Tài liệu này mô tả chi tiết việc triển khai kiến trúc **[Agent-as-a-Graph: Tool and Agent Retrieval for LLM Multi-Agent Systems](Agent-as-a-GraphTool%20and%20Ag%20Retrieval%20for%20LLM%20Multi-Agent%20Systems.md)** vào bộ truy xuất và xếp hạng công cụ (`src/lifemate/graph_retrieval.py` & `src/lifemate/prompts.py`) của dự án **LifeMate — Zalo AI Challenge 2026**.

---

## 1. Bối cảnh & Vấn đề cần giải quyết trong LifeMate

Trong mỗi mẫu của LifeMate, người dùng có thể cài đặt tới **18 ứng dụng (62 công cụ)**. Ngoài ra, tập Public Test có **11 ứng dụng mới (34 công cụ mới)** hoàn toàn không xuất hiện trong tập Train.
- **Nếu chỉ tìm theo Ứng dụng (App-only):** Mô tả tổng quát của `App` có thể không chứa từ khóa chi tiết của một hành động hoặc tham số cụ thể nằm sâu bên trong `Tool`, dẫn tới xếp hạng sai ứng dụng cần dùng.
- **Nếu chỉ tìm theo Công cụ rời rạc (Tool-only):** Mô hình chỉ thấy công cụ thao tác `write` khớp từ khóa (ví dụ `orders.return_item`) mà đánh mất ngữ cảnh của ứng dụng chủ quản và **bỏ sót công cụ tiền đề `read` đi kèm** (như `orders.get_order_details`), làm đứt gãy chuỗi suy luận đa bước.

---

## 2. Ba trụ cột của Agent-as-a-Graph trong `src/lifemate/graph_retrieval.py`

### 2.1. Đồ thị Tri thức Hai phần (Bipartite Knowledge Graph) $\mathcal{G} = (\mathcal{A}, \mathcal{T}, \mathcal{E})$
Hàm `BipartiteToolGraph.from_sample_and_catalog()` mô hình hóa toàn bộ danh mục công cụ được cài đặt trong mẫu thành đồ thị hai phần:
- **Tập nút Agent/App ($\mathcal{A}$ — `AppNode`):** Mỗi ứng dụng được cài đặt ($a \in \mathcal{A}$, ví dụ `clock`, `calendar`, `orders`, `ecommerce`...) là một nút đại diện cho cụm ngữ cảnh nghiệp vụ, chứa tên ứng dụng, `description` cấp ứng dụng và vector tần suất từ khóa tổng hợp (unigram + bigram tiếng Việt chuẩn hóa Unicode NFC).
- **Tập nút Tool ($\mathcal{T}$ — `ToolNode`):** Mỗi công cụ cụ thể ($t \in \mathcal{T}$) là một nút chứa `tool_id`, `kind` (`read`/`write`), schema tham số đã nén (`_compact_schema`), và vector từ khóa trích xuất từ tên hàm + mô tả hàm + tên và mô tả của toàn bộ các trường trong `parameters.properties`.
- **Tập cạnh sở hữu ($\mathcal{E}$):** Mỗi công cụ $t \in \mathcal{T}$ liên kết hai chiều với ứng dụng chủ quản $a = \text{owner}(t) \in \mathcal{A}$.

---

### 2.2. Thuật toán Xếp hạng Hợp nhất Có Trọng Số Loại (wRRF — Type-Specific Weighted Reciprocal Rank Fusion)
Với mỗi bước suy luận, hệ thống tạo truy vấn có trọng số từ yêu cầu hiện tại (`request`, hệ số $\times 3$) và 4 lượt hội thoại gần nhất trong `history` (hệ số $\times 1$), cộng điểm liên tục ngữ cảnh cho các ứng dụng vừa được gọi trong `prior_calls` / `history`.

1. **Xếp hạng độc lập trên hai kho $\mathcal{C}_A$ và $\mathcal{C}_T$:**
   - Tính độ tương đồng Cosine từ vựng (unigram + bigram) trên tập nút App $\mathcal{C}_A \rightarrow$ thu được thứ hạng $r_A(a) \in \{1, \dots, |\mathcal{A}|\}$.
   - Tính độ tương đồng Cosine từ vựng trên tập nút Tool $\mathcal{C}_T \rightarrow$ thu được thứ hạng $r_T(t) \in \{1, \dots, |\mathcal{T}|\}$.
2. **Tính điểm wRRF theo đúng công thức của bài báo:**
   $$s_{\text{wRRF}}(e) = \begin{cases} \dfrac{\alpha_T}{k + r_T(e)}, & e \in \mathcal{C}_T \\[8pt] \dfrac{\alpha_A}{k + r_A(e)}, & e \in \mathcal{C}_A \end{cases}$$
   với bộ siêu tham số chuẩn từ bài báo: **$\alpha_A = 1.5$** (ưu tiên độ ổn định định tuyến cấp Agent/App), **$\alpha_T = 1.0$** (bám sát chi tiết cấp Tool), và hằng số làm mượt $k = 10$.

---

### 2.3. Duyệt đồ thị ngược từ Tool về App (Tool $\rightarrow$ App Graph Traversal)
Sau khi có điểm $s_{\text{wRRF}}(t)$ của từng công cụ và $s_{\text{wRRF}}(a)$ của từng ứng dụng:
- Hệ thống duyệt ngược dọc theo cạnh sở hữu $\mathcal{E}$ ($t \rightarrow \text{owner}(t) = a$) để truyền tín hiệu từ các công cụ khớp mạnh nhất về ứng dụng chủ quản:
  $$S_{\text{Graph}}(a) = s_{\text{wRRF}}(a) + \sum_{t \in \text{Top-2}(\text{tools}(a))} s_{\text{wRRF}}(t)$$
- **Kết quả thu được:**
  1. Các cụm ứng dụng $\mathcal{A}^\star$ được sắp xếp theo thứ tự giảm dần của $S_{\text{Graph}}(a)$, đưa đúng cụm ứng dụng cần thiết lên đầu danh sách `available_tools`.
  2. Khi một công cụ thao tác `write` khớp từ khóa với câu nói của người dùng, phép duyệt đồ thị $t \rightarrow a$ kéo toàn bộ ứng dụng $a$ lên đầu, và bên trong ứng dụng $a$, các công cụ **`kind="read"` được đặt ngay trước `kind="write"`** (theo thứ tự $-s_{\text{wRRF}}(t)$), đảm bảo mô hình luôn nhìn thấy công cụ tra cứu ID trước khi gọi công cụ thao tác!

---

## 3. Cắt tỉa An toàn theo Ngân sách Token (wRRF Budget-Aware Pruning)

Trong [`fit_messages()`](src/lifemate/prompts.py) và [`trim_history()`](training/train_lora.py):
- Khi chạy trên các mẫu quá dài hoặc trên GPU giới hạn bộ nhớ (như Google Colab T4 với `max_length = 2048`), nếu sau khi đã cắt hết các lượt `history` cũ mà tổng số token của prompt vẫn vượt ngân sách `max_input_tokens`:
- Hệ thống tự động cắt tỉa từng công cụ ở **đáy bảng xếp hạng wRRF** (`_pop_lowest_wrrf_tool`), tuyệt đối giữ lại các cụm ứng dụng có điểm $S_{\text{Graph}}(a)$ cao nhất ở đầu danh sách và giữ nguyên công cụ hệ thống `sys.ask_missing_info`.
- Nhờ đó, quá trình huấn luyện QLoRA trên Colab và quá trình suy luận trên A30 không bao giờ bị văng lỗi `ValueError: Prompt requires ... tokens; budget is ...` mà vẫn bảo toàn độ phủ công cụ (Recall) cao nhất.

---

## 4. Danh sách file liên quan

| File | Mô tả |
|---|---|
| [`src/lifemate/graph_retrieval.py`](src/lifemate/graph_retrieval.py) | Cài đặt `AppNode`, `ToolNode`, `BipartiteToolGraph`, thuật toán xếp hạng `rank_tools_wrrf(alpha_a=1.5, alpha_t=1.0)` và duyệt đồ thị `Tool -> App`. |
| [`src/lifemate/prompts.py`](src/lifemate/prompts.py) | Kết nối `BipartiteToolGraph` vào `available_tool_specs()` và bổ sung cơ chế cắt tỉa đuôi wRRF `_pop_lowest_wrrf_tool()` trong `fit_messages()`. |
| [`training/train_lora.py`](training/train_lora.py) | Đồng bộ cơ chế cắt tỉa đuôi wRRF trong `trim_history()` để chống tràn bộ nhớ VRAM khi train QLoRA. |

