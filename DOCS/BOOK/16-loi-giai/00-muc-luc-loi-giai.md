[← Đề bài](../16-bai-toan-lon-de-bai.md) · [Mục lục sách](../00-muc-luc.md)

# Lời giải — Bài toán lớn: Từ COLMAP đến `.ply` render được 3D

> Lời giải chi tiết cho [đề bài ở Chương 16](../16-bai-toan-lon-de-bai.md), chia làm **8 phần** tương ứng
> chương 6–13 của sách, mỗi phần tính tay theo đúng công thức của chương gốc (formula → thay số → kết
> quả), trên cùng "cảnh đồ chơi dùng chung" (4 điểm SfM, 3 camera, ảnh $48\times32$) đã định nghĩa ở
> [§6.2](../06-initialization.md). Tổng độ dài toàn bộ lời giải ~30 000 dòng markdown.

| Phần | Chương gốc | Nội dung | File |
|---|---|---|---|
| 1 | 6 — Initialization | Input COLMAP → 4 Gaussian × 59 tham số | [01-init.md](01-init.md) |
| 2 | 7 — 3D Gaussians | $\Sigma=RSS^\top R^\top$, $G(x)$, giải mã màu SH | [02-gaussians.md](02-gaussians.md) |
| 3 | 8 — Projection & Compact Box | EWA splatting, $\Sigma'$, compact box, tile list | [03-projection.md](03-projection.md) |
| 4 | 9 — Tile Rasterizer | Sort theo độ sâu, alpha-blend → ảnh render | [04-rasterizer.md](04-rasterizer.md) |
| 5 | 10 — Loss & Metrics | $L_1$, SSIM, PSNR, Score, Loss huấn luyện | [05-loss-metrics.md](05-loss-metrics.md) |
| 6 | 11 — Gradient & Adam | Backprop 236 tham số, cập nhật Adam tại $t=5000$ | [06-gradient-adam.md](06-gradient-adam.md) |
| 7 | 12 — Structure-Aware Densification (SADGS) | $\eta_{3ch}$ đa view, `densify_and_clone/split/prune_structgs` → $\mathcal G_1$ | [07-density-control.md](07-density-control.md) |
| 8 | 13 — Xuất `.ply` | `point_cloud.ply` 63 cột (SADGS, gồm `filter_3D`) + render 3D | [08-cost-model-ply.md](08-cost-model-ply.md) |

File nhị phân minh hoạ (`.ply` thật, mở được trong trình xem 3D) nằm ở
[`assets/point_cloud_iter5000.ply`](assets/point_cloud_iter5000.ply) và bản suy biến
[`assets/point_cloud_iter5000_simple_pointcloud.ply`](assets/point_cloud_iter5000_simple_pointcloud.ply)
(point cloud RGB đơn giản, mở được ngay bằng MeshLab/CloudCompare/Blender). **Lưu ý:** hai file này được
sinh cho bản nháp trước (layout 62 cột, quần thể 4 Gaussian) và không được ghi đè khi Phần 7–8 được viết
lại theo đúng cơ chế SADGS thật (63 cột, quần thể 18 Gaussian sau densify) — xem §8.B.4 của
[08-cost-model-ply.md](08-cost-model-ply.md) để biết phạm vi/giới hạn của hai file mẫu này.

---

[← Đề bài](../16-bai-toan-lon-de-bai.md) | [Mục lục sách](../00-muc-luc.md) | [Bắt đầu Phần 1 →](01-init.md)
