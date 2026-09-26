[← Mục lục sách](../00-muc-luc.md) · [← Chương 17](../17-sadgs-structure-aware-densification.md)

# Chương 18 — SADGS: Cơ chế chuyên sâu (đọc từ source)

Chương 17 trả lời câu hỏi **"SADGS là gì và khác 3DGS gốc ở đâu"**. Chương 18 đi thêm một tầng:
mở từng hàm trong `SADGS/` ra, dựng lại công thức từ code, và **mỗi công thức có một hình
matplotlib được sinh trực tiếp từ chính công thức đó** (không phải hình trang trí).

> **Quy tắc của chương này**
> - Mọi con số, tên hàm, ngưỡng, hyperparameter đều trích từ source thật kèm `file.py:dòng`.
> - Cơ chế nào có code nhưng **không được gọi** trong `train.py` đều được ghi rõ là *đang tắt* —
>   đây là điểm dễ nhầm nhất khi đọc repo SADGS.
> - Toàn bộ 82 hình nằm ở `DOCS/Slides67/figures/sadgsx/`, script sinh hình ở
>   `DOCS/Slides67/figures/sadgsx/scripts/botNN_figs.py` (matplotlib, chạy lại được).

## Bản đồ 11 mục

> **Bắt đầu ở đâu:** nếu chưa hình dung được mạch chạy, đọc [18.11 — Bản đồ toàn tuyến](11-ban-do-tensor-den-adc.md) trước, rồi quay lại 18.1.

| Mục | Nội dung | Hàm/file chính trong `SADGS/` |
|---|---|---|
| [18.1](01-structure-tensor-da-ti-le.md) | Structure tensor & cấu trúc ảnh đa tỉ lệ | `utils/loss_utils.py` — `get_structure_tensor_torch`, `get_multiscale_structure_tensor_v1/v2` |
| [18.2](02-chieu-truc-va-ti-so-eta.md) | Chiếu trục Gaussian lên màn hình & tỉ số $\eta$ | `utils/freq_utils.py` — `compute_projected_axes_subset` |
| [18.3](03-thong-ke-eta-da-view.md) | Thống kê $\eta$ online đa view & tỉ lệ nhất quán | `utils/freq_utils.py` — `update_freq_stats_online`, `sampling_cameras` |
| [18.4](04-anisotropic-split.md) | Split **dị hướng** theo từng trục | `scene/gaussian_model.py` — `densify_and_split_structgs` |
| [18.5](05-clone-va-expand.md) | Clone, ghép tensor vào optimizer, `expand_undersized_gs` | `scene/gaussian_model.py` — `densify_and_clone_structgs`, `expand_undersized_gs` |
| [18.6](06-densify-prune-va-final-prune.md) | Vòng densify+prune hợp nhất & prune cuối | `scene/gaussian_model.py` — `densify_and_prune_structgs`, `final_prune_structgs` |
| [18.7](07-optimizer-lr-opacity-reset-va-3d-filter.md) | Optimizer, LR schedule, opacity reset, bộ lọc 3D chống alias | `scene/gaussian_model.py`, `gaussian_renderer/__init__.py` |
| [18.8](08-loss-va-lay-mau.md) | Hàm mất mát (L1/SSIM/tone-curve/frequency) & lấy mẫu camera + Gaussian | `utils/loss_utils.py`, `utils/gaussian_sampling.py` |
| [18.9](09-sieu-tham-so-va-rasterizer.md) | Toàn bộ siêu tham số, lịch huấn luyện & rasterizer riêng | `arguments/__init__.py`, `submodules/diff-gaussian-rasterization_structgs` |
| [18.10](10-tong-hop-dau-cuoi.md) | Tổng hợp đầu-cuối: pipeline, so sánh 3 phương pháp, chi phí, rủi ro | `train.py`, `run_train.sh`, `full_eval.py` |
| [18.11](11-ban-do-tensor-den-adc.md) | **Bản đồ toàn tuyến: structure tensor → ADC** — một hình nối cả tuyến, không có công thức mới | tổng hợp `loss_utils.py` + `freq_utils.py` + `train.py` + `gaussian_model.py` |

## Ba cơ chế cốt lõi — đọc theo thứ tự này nếu vội

1. **Đo cấu trúc ảnh** (18.1) → biết mỗi pixel có texture "mịn" tới mức nào.
2. **So Gaussian với cấu trúc đó** (18.2, 18.3) → ra đại lượng $\eta$ và tỉ lệ nhất quán đa view.
3. **Hành động theo $\eta$** (18.4, 18.5, 18.6) → split dị hướng / clone / prune.

Ba mục còn lại (18.7–18.9) là hạ tầng đỡ cho ba cơ chế trên; 18.10 ráp lại thành một vòng train,
còn [18.11](11-ban-do-tensor-den-adc.md) vẽ toàn tuyến trên một hình duy nhất — **nếu chưa hình dung được mạch, đọc 18.11 trước**.

## Liên kết với bộ slide

| Tài liệu | Vị trí |
|---|---|
| Deck đầy đủ (370 trang, 6 `\part`) | `DOCS/Slides67/main.tex` → `main.pdf`, phần SADGS chuyên sâu ở section bản đồ `part_sadgsx_16` (B00) + 15 section `part_sadgsx_01..15` |
| Deck 45 phút (62 slide, 6 `\section`) | `DOCS/Slides67/Slides45min/main45.tex` → `main45.pdf`, 2 slide bản đồ `part45_sadgsx_00` + 15 slide `part45_sadgsx_01..15` |
| Script nói 45 phút | `DOCS/Slides67/script/script45/FULL_45min_script.md` + `sadgsx_botNN_45min.md` |

---

**Tiếp theo:** [18.1 — Structure tensor & cấu trúc ảnh đa tỉ lệ](01-structure-tensor-da-ti-le.md)
