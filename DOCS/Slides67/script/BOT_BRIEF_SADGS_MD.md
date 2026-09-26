# BRIEF — 10 BOT: viết chương MD chuyên sâu SADGS + script thuyết trình 45 phút

## Gốc
- ROOT   = C:/Users/kelly/OneDrive/Desktop/digital-twin-learn
- CODE   = $ROOT/SADGS  (source THẬT: scene/gaussian_model.py, utils/freq_utils.py, utils/loss_utils.py,
           utils/gaussian_sampling.py, train.py, arguments/__init__.py, gaussian_renderer/__init__.py,
           submodules/diff-gaussian-rasterization_structgs)
- DECK   = $ROOT/DOCS/Slides67           (part_sadgsx_NN.tex — 8–9 frame/bot, ĐÃ XONG, chỉ đọc)
- DECK45 = $ROOT/DOCS/Slides67/Slides45min (part45_sadgsx_NN.tex — 1 frame/bot, ĐÃ XONG, chỉ đọc)
- FIGS   = $ROOT/DOCS/Slides67/figures/sadgsx  (83 PNG matplotlib + scripts/botNN_figs.py — ĐÃ XONG, chỉ đọc)
- OUT_MD = $ROOT/DOCS/BOOK/18-sadgs-co-che-chuyen-sau/
- OUT_SCRIPT = $ROOT/DOCS/Slides67/script/script45/

## Bối cảnh
15 bot trước đã quét source SADGS và sinh ra slide + hình matplotlib. Việc còn lại là **viết phần
văn bản (.md) đi kèm**: một chương sách chuyên sâu bằng tiếng Việt, **nhúng TOÀN BỘ hình matplotlib**
đã sinh, và **script nói** cho bản 45 phút.

## Nhiệm vụ mỗi bot (2 sản phẩm)

### 1) Chương MD chuyên sâu — `$OUT_MD/<file được giao>.md`
- Tiếng Việt, văn phong sách kỹ thuật (giống `$ROOT/DOCS/BOOK/17-sadgs-structure-aware-densification.md`
  và `$ROOT/DOCS/BOOK/12-adaptive-density-control.md` — ĐỌC 2 file này trước để khớp giọng văn, cách
  đánh số mục, cách trích dẫn `file.py:dòng`, cách dùng bảng).
- Mở đầu bằng dòng điều hướng: `[← Mục lục chương 18](00-muc-luc.md) · Chương 18.N`
- Có khối `> Nguồn:` liệt kê các file/dòng code thật đã dựa vào.
- **BẮT BUỘC nhúng HẾT các PNG thuộc chủ đề của mình** (tiền tố số bot, xem phân công) bằng
  đường dẫn tương đối: `![mô tả](../../Slides67/figures/sadgsx/NN_ten.png)`
  kèm caption in nghiêng ngay dưới: `*Hình 18.N.k — giải thích hình nói lên điều gì.*`
- **Mỗi hình phải gắn với công thức toán tương ứng** (LaTeX `$...$` / `$$...$$`) và được giải thích:
  công thức là gì, biến là gì, hình minh hoạ đoạn nào của công thức, kết luận rút ra.
- Đọc `scripts/botNN_figs.py` để biết CHÍNH XÁC hình vẽ gì (tham số, trục, dải giá trị) — mô tả
  đúng nội dung hình, KHÔNG suy đoán.
- **Mọi con số / tên hàm / ngưỡng / hyperparameter PHẢI trích từ code thật kèm `file.py:dòng`.
  TUYỆT ĐỐI KHÔNG BỊA.** Nếu một cơ chế trong code đang bị tắt / không được gọi, phải nói rõ điều đó.
- Độ dài: 250–450 dòng. Đặc, không lan man, không lặp lại phần bot khác.
- Kết bằng mục "Tóm tắt" (bảng hoặc bullet) + dòng điều hướng tới chương kế.

### 2) Script nói 45 phút — `$OUT_SCRIPT/sadgsx_botNN_45min.md`
- Kịch bản người trình bày **nói ra miệng** cho đúng (các) slide `part45_sadgsx_NN.tex` của mình.
- Định dạng: tiêu đề slide → **Thời lượng: ~45 giây** → lời thoại liền mạch (không bullet khô khan,
  viết như nói) → 1 dòng "Chuyển tiếp:" dẫn sang slide sau.
- Kèm mục **"Nếu bị hỏi"**: 2–3 câu hỏi khả dĩ + câu trả lời ngắn có số liệu/code ref.
- Tham chiếu format ở `$OUT_SCRIPT/FULL_45min_script.md` (đọc vài mục đầu để khớp giọng).

## Ranh giới file (TRÁNH XUNG ĐỘT — tuyệt đối tuân thủ)
CHỈ tạo/sửa đúng 2 file của riêng mình (nêu trong phân công). KHÔNG sửa main.tex, main45.tex,
part*.tex, các PNG, file của bot khác, hay 00-muc-luc.md (orchestrator lo).

## Báo cáo cuối
Trả về: đường dẫn 2 file đã viết, số dòng, danh sách PNG đã nhúng (phải khớp đủ), và 3–5 phát hiện
quan trọng nhất từ code (kèm file:dòng).
