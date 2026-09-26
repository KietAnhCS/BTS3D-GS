[← Mục lục](00-muc-luc.md) · Chương 15/17

# Chương 15 — Phụ lục: Tài nguyên & trạng thái kiểm định của `SADGS/`

> Nguồn: `SADGS/README.md`, `SADGS/full_eval.py`, `SADGS/static/`, `SADGS/submodules/`, `SADGS/LICENSE`,
> `SADGS/LICENSE_ORIGINAL.md`.
>
> **Đổi so với bản trước.** Chương 15 cũ tổng hợp 8 bài kiểm định số (`DOCS/Report/test/`), hai file dữ liệu
> `history.csv`/`leaderboard.csv`, công cụ tương tác `test.html`, và danh mục 16 script `chNN_test.py`/`chNN_plot.py`
> — toàn bộ thuộc bộ tài liệu tiếng Việt viết riêng cho pipeline 3DGS gốc bản tăng tốc trước đây (`DOCS/Report/test/`), **không tồn
> tại trong `SADGS/`**. `SADGS/` không có thư mục `Report/`, không có `test.html`, không có script kiểm định số
> nào tương đương. Các bài kiểm định số chéo cho phần toán học 3DGS chung (Chương 6–12, cảnh đồ chơi 4 điểm SfM)
> vẫn còn giá trị **cho phần lý thuyết 3DGS gốc** — chương đó không đổi, tự nó không nói gì về SAD-GS.
> Chương này viết lại theo đúng tài nguyên thật có trong `SADGS/`.

## 15.1 — Không có bộ kiểm định số riêng cho SAD-GS

Không có script nào trong `SADGS/` chạy lại các công thức ở Chương 17 ($\eta$ wavelength/projection, structure
tensor Di Zenzo, anisotropic split) trên một cảnh đồ chơi nhỏ để in ra số trung gian, kiểu các file
`ch0N_test.py` đã có cho Chương 6–13 cũ. Việc kiểm định công thức ở Chương 17 hiện dừng ở mức **đối chiếu
tĩnh** giữa văn bản chương và số dòng code cụ thể (đã ghi rõ trong từng mục của Chương 17), không có phép đo
số bằng chương trình độc lập. Muốn tự kiểm định, cách khả thi nhất là viết một script NumPy nhỏ tái tạo đúng
`get_structure_tensor_torch` và `compute_projected_axes_subset` trên một ảnh tổng hợp đơn giản (ví dụ ảnh có
một cạnh sắc nét) rồi so tay với công thức đã trích ở mục 17.1–17.3 — chưa có ai làm việc này trong repo.

## 15.2 — Trạng thái kiểm định 5 cơ chế theo Chương 17

Nhắc lại bảng đã có ở cuối Chương 17 (mục 17.6–17.7), để phần phụ lục này đứng độc lập được:

| Cơ chế | Review tĩnh (đọc code, đối chiếu công thức) | Chạy thử trên GPU thật | Benchmark định lượng (PSNR/SSIM/LPIPS/thời gian) |
|---|---|---|---|
| Structure tensor đơn tỉ lệ (Di Zenzo) | Đã làm (17.1) | Chưa | Chưa |
| Multiscale structure tensor v1/v2 | Đã làm (17.2) | Chưa | Chưa |
| $\eta$ wavelength / projection | Đã làm (17.3) | Chưa | Chưa |
| Multiview high/low ratio | Đã làm (17.4) | Chưa | Chưa |
| Anisotropic split $(k_x,k_y,k_z)$ | Đã làm (17.5) | Chưa | Chưa |

Toàn bộ cột "Đã làm" là công việc đọc mã nguồn trong một môi trường **không có GPU/CUDA toolchain** — không
build được `diff-gaussian-rasterization_structgs`, không chạy được `run_train.sh`. Hai cột sau chỉ có thể
hoàn thành trên máy có GPU CUDA thật, theo đúng quy trình đã nêu ở Chương 13 (mục 13.5) và Chương 14.

## 15.3 — Tài nguyên tĩnh có sẵn trong `SADGS/`

| Đường dẫn | Nội dung |
|---|---|
| `SADGS/static/teaser.png` | ảnh teaser minh hoạ phương pháp, dùng trong `README.md` |
| `SADGS/LICENSE`, `SADGS/LICENSE_ORIGINAL.md` | giấy phép của repo SAD-GS và giấy phép gốc kế thừa từ 3DGS/Inria — đọc trước khi dùng lại mã nguồn |
| `SADGS/submodules/diff-gaussian-rasterization_structgs/` | CUDA rasterizer riêng của SAD-GS (fork của rasterizer 3DGS gốc) |
| `SADGS/submodules/simple-knn/`, `SADGS/submodules/fused-ssim/` | hai submodule phụ trợ giống hệt bản 3DGS gốc (KNN cho khởi tạo scale, SSIM tăng tốc) |
| `SADGS/full_eval.py` | script eval kế thừa 3DGS gốc — **không chạy được với `train.py` hiện tại** (Chương 13, mục 13.3); vẫn hữu ích để tra bảng `big_budgets`/`budget_multipliers` (đã chép lại ở Chương 13) |
| `SADGS/run_train.sh` | quy trình huấn luyện + eval thật duy nhất chạy được (Chương 13, mục 13.2; Chương 14) |

Không có dữ liệu số thô dạng CSV nào (`history.csv`, `leaderboard.csv`) đi kèm repo — khác `DOCS/assets/`
của bản tài liệu 3DGS gốc cũ. Không có công cụ minh hoạ tương tác dạng HTML/JS nào tương đương `test.html`.

## 15.4 — Đối chiếu công thức: chỗ nào tài liệu chương 6–12 vẫn dùng được nguyên vẹn cho SAD-GS

Vì `SADGS/` không thay chuỗi render/tối ưu lõi (Chương 13, mục 13.1), phần lớn Chương 5–12 của cuốn sách này
(nền tảng toán học 3DGS, projection, rasterizer, loss, gradient, Adam, khung ADC bảy bước) áp dụng **nguyên
vẹn** cho SAD-GS — đọc mã nguồn tương ứng trong `SADGS/scene/`, `SADGS/gaussian_renderer/`,
`SADGS/utils/loss_utils.py` xác nhận cùng công thức (chiếu EWA, alpha-blend, $\mathcal L=(1-\lambda)\mathcal
L_1+\lambda\mathcal L_{\text{D-SSIM}}$, Adam). Phần **không** dùng được nguyên vẹn là đúng những gì Chương 17
đã khoanh vùng: cơ chế chọn Gaussian để split/clone/prune (thay Importance/Pruning score bằng $\eta$ +
multiview ratio) và cách sinh Gaussian con khi split (lưới tất định $k_x k_y k_z$ thay Monte-Carlo 2 bản).

Bảng tương ứng nhanh (chương nào của sách này còn áp dụng trực tiếp, chương nào cần đọc kèm Chương 17):

| Chương | Áp dụng cho SAD-GS |
|---|---|
| 5 — Ký hiệu & nền tảng | Có, nguyên vẹn |
| 6 — Initialization | Có, nguyên vẹn (SAD-GS không đổi `create_from_pcd`) |
| 7 — 3D Gaussians | Có, nguyên vẹn |
| 8 — Projection & compact box | Có, nguyên vẹn (cùng công thức EWA/compact box; `mult` chỉ đổi giá trị dùng, không đổi công thức) |
| 9 — Rasterizer | Có, nguyên vẹn (rasterizer `_structgs` là fork không đổi thuật toán blend) |
| 10 — Loss & metrics | Có, nguyên vẹn |
| 11 — Gradient & Adam | Có, nguyên vẹn |
| 12 — Adaptive Density Control (3DGS gốc) | Đọc **cùng với** Chương 17 — SAD-GS thay tiêu chí split/prune, không thay khung 7 bước |
| 13 — Quy trình & chi phí | Viết riêng cho SAD-GS (chương này đã đổi) |
| 17 — SAD-GS densification | Nội dung riêng, không có ở bản 3DGS gốc |

---

[← Chương 14](14-trien-khai-colab-nhat-ky-train.md) | [Mục lục](00-muc-luc.md) | [Chương 16 →](16-bai-toan-lon-de-bai.md)
