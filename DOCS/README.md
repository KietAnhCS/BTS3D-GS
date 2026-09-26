# DOCS/ — Bản đồ tài liệu SADGS (Structure-Aware Densification cho 3D Gaussian Splatting)

README gốc ở thư mục cha trỏ vào vài đường dẫn (`DOCS/fastgs-acceleration-method.md`,
`DOCS/colab-t4-guide.md`, `DOCS/history-train.md`, `DOCS/DIGITAL-TWIN-GS-PIPELINE-{1,2,3}.md`)
không còn tồn tại dưới tên đó — toàn bộ nội dung đã được **hợp nhất vào `DOCS/BOOK/`**
(17 chương, một mạch đọc liền, không lược bỏ gì). Trang này là bản đồ điều hướng: đi từ
tên file cũ (hoặc từ nhu cầu đọc) tới đúng chương trong `BOOK/`, cộng thêm mục lục
`Report/` và `Slides67/`.

Lưu ý về tên dự án: mã nguồn thật (`SADGS/`) là bản triển khai của paper "Faster 3D
Gaussian Splatting Convergence via Structure-Aware Densification" (SIGGRAPH 2026, Lyu
et al., MPI Informatik). SADGS kế thừa 3DGS + FastGS nhưng **thay thế cơ chế densification
của FastGS bằng structure-aware densification**: so khớp screen-space extent của mỗi
Gaussian với cấu trúc texture đa tỉ lệ (multiscale image structure) của ảnh, tách dị hướng
(anisotropic split) thay vì clone/split đẳng hướng, và có bước multiview-consistency pruning
để loại các Gaussian không nhất quán giữa nhiều góc nhìn. Các hàm cốt lõi nằm ở
`SADGS/scene/gaussian_model.py`: `densify_and_split_structgs`, `densify_and_clone_structgs`,
`densify_and_prune_structgs`, `final_prune_structgs`. Nhiều tài liệu trong `DOCS/` (đặc
biệt các chương viết trước khi SADGS được tích hợp) vẫn còn nhắc "FastGS-lite"/"FasterGS"
như tên dự án cũ — coi các tên đó là bí danh lịch sử của cùng một pipeline, đang được cập
nhật dần sang tên SADGS.

## Kết quả thực nghiệm mới nhất

**Cập nhật: đã có số liệu SADGS thật** — chạy thật trên Colab T4 (không mô phỏng), scene
`HCM0539` (dữ liệu cuộc thi `VAI_NVS_DATA_ROUND2`, 240 ảnh train / 60 ảnh test), 30 000
iterations, dùng chính notebook `bts_digital_twin.ipynb` và cơ chế densify/prune structure-aware
thật của SADGS (`densify_and_split_structgs` / `densify_and_clone_structgs` /
`densify_and_prune_structgs` / `final_prune_structgs`, không còn phải nhánh 3DGS gốc):

| Score | PSNR | SSIM | LPIPS | Gaussians cuối | Thời gian train | VRAM đỉnh |
|---|---|---|---|---|---|---|
| **0.8681** | 23.778 dB | 0.8897 | 0.0915 | 2 968 520 | 4234.8 s (~1h10p) | 7.21 GB |

So với mốc tham khảo cũ của bản FastGS-lite (score 0.8579, PSNR 25.27dB, SSIM 0.8659, LPIPS
0.1364, 344 484 Gaussian, 1776s, VRAM đỉnh 1.92GB — chạy **trước khi** tích hợp densify/prune
SADGS thật): score/SSIM/LPIPS của SADGS đều tốt hơn (LPIPS giảm ~33%), nhưng PSNR thấp hơn
~1.5dB và số Gaussian/thời gian train/VRAM đỉnh tăng lần lượt ~8.6×/2.4×/3.8× — một đánh đổi
cần cân nhắc, không phải chiến thắng toàn diện. Phân tích đầy đủ (9 phần, 7 biểu đồ matplotlib
từ log thật, ảnh submission thật, bảng `cfg_args` — phát hiện một số tham số densify/prune
"chết" trong code khiến cơ chế prune gần như bất hoạt, giải thích vì sao số Gaussian phình to)
nằm ở [**Chương 19 — Kết quả thực nghiệm SADGS thật**](BOOK/19-ket-qua-sadgs-that/19-ket-qua-sadgs-that.md).

Nhật ký cấu hình/triển khai Colab T4 (không gắn với lần chạy SADGS thật ở trên) vẫn còn ở
[14-trien-khai-colab-nhat-ky-train.md](BOOK/14-trien-khai-colab-nhat-ky-train.md), và phần
so sánh lý thuyết 3DGS gốc / FastGS / SADGS nằm ở
[17-sadgs-structure-aware-densification.md](BOOK/17-sadgs-structure-aware-densification.md).

## Từ tên file cũ tới chương mới

| Tên cũ (không còn tồn tại) | Nằm ở đâu trong `BOOK/` bây giờ |
|---|---|
| `DOCS/fastgs-acceleration-method.md` | [13-tong-hop-chi-phi-fastgs.md](BOOK/13-tong-hop-chi-phi-fastgs.md) — nguyên văn Phần I–IX: nền toán 3DGS → mô hình chi phí per-iteration → ba đòn bẩy tăng tốc → so sánh FastGS-lite vs 3DGS |
| `DOCS/colab-t4-guide.md` | [14-trien-khai-colab-nhat-ky-train.md](BOOK/14-trien-khai-colab-nhat-ky-train.md) — preset T4, quy tắc chống OOM, nhật ký các phiên train thật |
| `DOCS/history-train.md` | cũng trong [14-trien-khai-colab-nhat-ky-train.md](BOOK/14-trien-khai-colab-nhat-ky-train.md) (phần nhật ký) |
| `DOCS/DIGITAL-TWIN-GS-PIPELINE-1.md` | [02-kien-truc-pipeline-phan-1.md](BOOK/02-kien-truc-pipeline-phan-1.md) |
| `DOCS/DIGITAL-TWIN-GS-PIPELINE-2.md` | [03-vong-lap-huan-luyen-phan-2.md](BOOK/03-vong-lap-huan-luyen-phan-2.md) |
| `DOCS/DIGITAL-TWIN-GS-PIPELINE-3.md` | [04-luu-render-cham-diem-phan-3.md](BOOK/04-luu-render-cham-diem-phan-3.md) |

## Mục lục `BOOK/` (19 chương)

Xem đầy đủ tại [BOOK/00-muc-luc.md](BOOK/00-muc-luc.md); tóm tắt nhanh:

1. [Giới thiệu & Tổng quan dự án](BOOK/01-gioi-thieu-tong-quan.md)
2. [Kiến trúc Pipeline — Phần 1](BOOK/02-kien-truc-pipeline-phan-1.md) — CLI/notebook, `Scene`, `GaussianModel`
3. [Vòng lặp huấn luyện — Phần 2](BOOK/03-vong-lap-huan-luyen-phan-2.md) — render, rasterizer, loss, backward, densify/prune
4. [Lưu, Render, Chấm điểm & Vận hành — Phần 3](BOOK/04-luu-render-cham-diem-phan-3.md) — `.ply`, `submission.zip`, chẩn đoán sự cố
5. [Ký hiệu & Nền tảng toán học chung](BOOK/05-ky-hieu-nen-tang-toan-hoc.md)
6. [Initialization](BOOK/06-initialization.md) — SfM points → Gaussian khởi tạo
7. [Biểu diễn 3D Gaussians](BOOK/07-3d-gaussians.md)
8. [Projection & Compact Box](BOOK/08-projection-compact-box.md) — đòn bẩy giảm K
9. [Differentiable Tile Rasterizer](BOOK/09-differentiable-tile-rasterizer.md)
10. [Từ Ảnh đến Loss & Metrics](BOOK/10-anh-den-loss-metrics.md)
11. [Gradient Flow & Backpropagation](BOOK/11-gradient-flow-backprop.md)
12. [Adaptive Density Control](BOOK/12-adaptive-density-control.md) — đòn bẩy giảm N; nền tảng ADC mà SADGS thay thế bằng structure-aware densification (đóng góp chính của SADGS)
13. [Tổng hợp Mô hình Chi phí & Phương pháp Tăng tốc FastGS](BOOK/13-tong-hop-chi-phi-fastgs.md)
14. [Triển khai Thực tế: Colab T4 & Nhật ký Huấn luyện](BOOK/14-trien-khai-colab-nhat-ky-train.md)
15. [Phụ lục: Kiểm định số chéo, Dữ liệu & Tài nguyên](BOOK/15-phu-luc-kiem-dinh-tai-nguyen.md)
16. [Bài toán lớn: Từ COLMAP đến `.ply` render 3D](BOOK/16-bai-toan-lon-de-bai.md) + [lời giải 8 phần](BOOK/16-loi-giai/00-muc-luc-loi-giai.md)
17. [So sánh chất lượng: 3DGS gốc vs FastGS vs SADGS (structure-aware densification)](BOOK/17-sadgs-structure-aware-densification.md)
18. [SADGS: Cơ chế chuyên sâu (10 mục, 82 hình matplotlib)](BOOK/18-sadgs-co-che-chuyen-sau/00-muc-luc.md)
19. [Kết quả thực nghiệm SADGS thật — HCM0539](BOOK/19-ket-qua-sadgs-that/19-ket-qua-sadgs-that.md) — lần train thật đầu tiên với densify/prune SADGS thật, 7 biểu đồ + ảnh submission thật

## `Report/`

Báo cáo minh hoạ bằng hình (`Report/assets/`, `Report/test/figures/` — covariance
ellipsoid, SH sphere, EWA projection, alpha blending, Adam trajectory, cost model, v.v.).
Hiện thư mục này **chỉ có ảnh, chưa có file `.md`/`.tex` thuyết minh riêng** — narrative
tương ứng với từng ảnh đã được viết trực tiếp trong các chương `BOOK/06`–`BOOK/13` (mỗi
chương đó dẫn ảnh từ đúng thư mục `Report/assets/chN_*.png` này).

## `Slides67/`

Hai bộ slide LaTeX (beamer, XeLaTeX), cùng dùng chung `figures/`:

| Bộ | File gốc | Kết quả | Dùng khi nào |
|---|---|---|---|
| **Đầy đủ** | `main.tex` | `main.pdf` — **359 trang** | Tra cứu, bảo vệ dài, trả lời câu hỏi sâu |
| **45 phút** | `Slides45min/main45.tex` | `main45.pdf` — **60 trang** | Buổi báo cáo chính |

Cấu trúc bộ đầy đủ: `part1`–`part7` bám chương 1–13 của `BOOK/`; `partADC_*` đi sâu chương 12
(Adaptive Density Control — nền tảng mà SADGS thay thế); `partSH_*` đi sâu Spherical Harmonics;
`part_sadgs_01..08` giới thiệu SADGS; và **`part_sadgsx_01..15` (127 frame) là phần SADGS chuyên sâu**
— 15 chủ đề quét trực tiếp từ source `SADGS/`, mỗi công thức kèm một hình matplotlib sinh từ chính
công thức đó (82 hình ở `figures/sadgsx/`, script sinh hình ở `figures/sadgsx/scripts/botNN_figs.py`).
Bản 45 phút lấy đúng 1 frame cô đọng cho mỗi chủ đề (`part45_sadgsx_01..15`).

Toàn bộ nội dung 15 chủ đề này được viết lại dạng sách, **nhúng đủ 82 hình**, ở
[BOOK/18-sadgs-co-che-chuyen-sau/](BOOK/18-sadgs-co-che-chuyen-sau/00-muc-luc.md).

### Kịch bản nói

`Slides67/script/` chứa script thuyết trình. Bản hoàn chỉnh cho buổi 45 phút là
[`script/script45/FULL_45min_script.md`](Slides67/script/script45/FULL_45min_script.md) —
60 slide, có bảng phân bổ thời gian và mục "Nếu bị hỏi" cho từng slide SADGS.

### Build

```bash
cd DOCS/Slides67          && xelatex main.tex   && xelatex main.tex
cd DOCS/Slides67/Slides45min && xelatex main45.tex && xelatex main45.tex
```

Lưu ý: frame nào chứa `verbatim`/`semiverbatim` **phải** khai báo `egin{frame}[fragile]`,
nếu không XeLaTeX sẽ nuốt nội dung và sinh hàng trăm lỗi `Undefined control sequence`.
