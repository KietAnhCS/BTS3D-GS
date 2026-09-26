# BRIEF CHUNG CHO 15 BOT — Bổ sung nội dung SADGS vào slide

## Gốc
- ROOT = C:/Users/kelly/OneDrive/Desktop/digital-twin-learn
- CODE  = $ROOT/SADGS            (source thật: scene/gaussian_model.py, utils/freq_utils.py, utils/loss_utils.py, utils/gaussian_sampling.py, train.py, arguments/__init__.py, gaussian_renderer/__init__.py, submodules/diff-gaussian-rasterization_structgs)
- DECK  = $ROOT/DOCS/Slides67    (main.tex, part*.tex, figures/)
- DECK45= $ROOT/DOCS/Slides67/Slides45min (main45.tex, part45_*.tex)

## Nhiệm vụ mỗi bot
1. **SCAN THẬT** code trong CODE cho chủ đề được giao. Mọi con số, tên hàm, tên tham số, ngưỡng, công thức PHẢI trích từ code thật (ghi kèm `file.py:dòng`). TUYỆT ĐỐI không bịa.
2. Viết file slide MỚI cho bản đầy đủ: `$DECK/part_sadgsx_NN.tex` (NN = số bot, 2 chữ số) — **6–9 frame**.
3. Viết file slide MỚI cho bản 45 phút: `$DECK45/part45_sadgsx_NN.tex` — **đúng 1 frame** cô đọng nhất (tóm tắt chủ đề, 1 hình).
4. **Mỗi công thức toán trong slide phải có 1 hình matplotlib tương ứng.** Script sinh hình: `$DECK/figures/sadgsx/scripts/botNN_figs.py`, output PNG vào `$DECK/figures/sadgsx/NN_<ten>.png`. CHẠY script cho ra PNG thật (python có sẵn, matplotlib 3.11.2, numpy 2.5.2).

## Quy ước LaTeX (BẮT BUỘC — khớp deck hiện có)
- Tiếng Việt, beamer, `\documentclass[aspectratio=169,9pt]`, theme Madrid/seahorse. KHÔNG viết preamble, chỉ viết các `\begin{frame}...\end{frame}`.
- Mở đầu file bằng comment `% ===== BOT NN: <chủ đề> =====`.
- Dùng `\footnotesize` hoặc `\scriptsize` trong frame cho vừa 1 trang. Nội dung đặc, không để tràn slide.
- Hình: `\includegraphics[width=0.6\linewidth]{sadgsx/NN_ten.png}` — dùng ĐÚNG đường dẫn tương đối này cho cả 2 deck (graphicspath đã trỏ tới figures/ và ../figures/).
- Chú thích hình dùng `\captionof{figure}{\footnotesize ...}`.
- Escape ký tự đặc biệt: `\_` trong tên hàm (hoặc dùng `\texttt{ten\_ham}`), `\%`, `\&`.
- Sản phẩm gọi tên cơ chế là **SADGS** (không gọi StructGS/FastGS trừ khi so sánh lịch sử).

## Quy ước matplotlib
- `import matplotlib; matplotlib.use("Agg")` ở đầu.
- `plt.rcParams["font.family"]="DejaVu Sans"` (hỗ trợ dấu tiếng Việt).
- dpi=150, `bbox_inches="tight"`, nền trắng, figsize gọn (~6x4 hoặc 8x4 cho 2 panel).
- Hình phải MINH HOẠ ĐÚNG công thức: vẽ đường cong/vùng quyết định/heatmap/sơ đồ thật từ công thức, không phải hình trang trí.
- Nhãn trục + legend rõ ràng, tiếng Việt được.

## Ranh giới file (TRÁNH XUNG ĐỘT)
- CHỈ ĐƯỢC tạo/sửa 4 file của riêng bot: part_sadgsx_NN.tex, part45_sadgsx_NN.tex, figures/sadgsx/scripts/botNN_figs.py, và các PNG có tiền tố `NN_`.
- **KHÔNG sửa** main.tex, main45.tex, các part*.tex cũ, hay file của bot khác. Orchestrator sẽ nối vào main.

## Báo cáo cuối
Trả về: danh sách frame (tiêu đề), danh sách PNG đã sinh, và các phát hiện quan trọng từ code (kèm file:dòng).
