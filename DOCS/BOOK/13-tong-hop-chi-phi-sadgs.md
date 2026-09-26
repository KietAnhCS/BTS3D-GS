[← Mục lục](00-muc-luc.md) · Chương 13/17

# Chương 13 — Quy trình huấn luyện & mô hình chi phí SAD-GS

> Nguồn: `SADGS/run_train.sh`, `SADGS/full_eval.py`, `SADGS/README.md`, `SADGS/environment.yml`, `SADGS/arguments/__init__.py`, `SADGS/scene/gaussian_model.py`.
>
> **Đổi so với bản trước của chương này.** Chương 13 trước đây (giữ trong lịch sử dưới một tên file khác)
> tổng hợp một **mô hình chi phí lý thuyết** ($T_{\text{iter}}=aN+bNK+cN\cdot\mathbb1+F$)
> và một bộ số đo thật (`history.csv`/`leaderboard.csv`) — cả hai đều thuộc về một pipeline khác hẳn
> (`pipeline/`, `demos/`, notebook Colab, định dạng nộp bài `submission.zip`) **không tồn tại trong `SADGS/`**.
> Repo `SADGS/` là bản triển khai của paper SIGGRAPH 2026 *"Faster 3D Gaussian Splatting Convergence via
> Structure-Aware Densification"* — không có thư mục `pipeline/`, không có `demos/`, không có notebook, không
> có file `history.csv`/`leaderboard.csv`, và không có script mô phỏng chi phí nào (script mô phỏng chi phí
> của pipeline cũ không tồn tại trong `SADGS/`). Vì vậy chương này viết lại **từ đầu**, bám đúng những gì thật sự có trong
> `SADGS/`: quy trình huấn luyện thật (`run_train.sh`), script eval kế thừa từ 3DGS gốc (`full_eval.py`), và
> các tham số quyết định chi phí ($N$, tần suất densify) đọc trực tiếp từ `arguments/__init__.py` và
> `scene/gaussian_model.py`. **Không có số liệu benchmark thật (PSNR/SSIM/LPIPS/VRAM/thời gian) trong repo
> này** — không bịa số, chỉ mô tả đúng công thức và quy trình.

## 13.1 — SAD-GS không đổi cấu trúc chi phí của 3DGS gốc

Về mặt chi phí tính toán, SAD-GS **không đổi** chuỗi render/tối ưu cốt lõi đã trình bày ở các chương 6–12:
chiếu Gaussian, rasterize theo tile, alpha-blend, backward, bước Adam. CUDA extension riêng của SAD-GS
(`submodules/diff-gaussian-rasterization_structgs/`) là một fork giữ nguyên bộ khung rasterizer của 3DGS gốc,
không đổi thuật toán rasterize — chỉ đổi **cơ chế quyết định Gaussian nào bị split/clone/prune và theo hướng
nào** (Chương 17). Nói cách khác: SAD-GS can thiệp đúng vào đòn bẩy "giảm $N$" (Chương 12) của mô hình chi phí
ba đòn bẩy cũ, theo một tiêu chí khác (cấu trúc tần số ảnh thay vì sai số màu render↔GT), cộng thêm một chiều
mới — **dị hướng theo trục** (chia $k_x,k_y,k_z$ độc lập, mục 17.5) — chứ không tạo ra một số hạng chi phí mới
nào trong $T_{\text{iter}}=aN+bNK+cN\cdot\mathbb1[\text{Adam}]+F$. Không có script nào trong `SADGS/` đo lại các
hệ số $a,b,c,F$ này; phần lý thuyết đầy đủ của công thức vẫn giữ nguyên giá trị tham khảo ở các chương 6–12
(áp dụng cho 3DGS gốc, không riêng SAD-GS).

Điểm khác biệt về **chi phí runtime thêm ra** của SAD-GS so với 3DGS gốc nằm ở hai chỗ mới (đối chiếu
Chương 17, mục 17.4 và 17.6):

| Cơ chế mới | Tần suất | Chi phí thêm |
|---|---|---|
| Precompute structure tensor GT (`get_multiscale_structure_tensor_v1/v2`) | 1 lần đầu train, cho mọi ảnh train (`train.py:160-200`) | tuyến tính theo số ảnh, không lặp lại trong vòng lặp chính |
| `update_freq_stats_online` (tính $\eta$, cập nhật high/low ratio) | mỗi 10 vòng (`SADGS/train.py:275`) | `grid_sample` + phân rã Cholesky cho mọi Gaussian visible mỗi lần gọi — không cần render (không qua rasterizer), rẻ hơn hàm tính điểm densify kiểu cũ (`compute_gaussian_score_fastgs`, vốn phải render $2\times10$ ảnh phụ mỗi lần densify, xem Chương 12) |

Đây là mô tả **định tính** — không có phép đo thời gian thật nào cho hai dòng trên trong repo.

## 13.2 — Quy trình huấn luyện thật: `run_train.sh`

`run_train.sh` là script chạy được duy nhất trong `SADGS/` tạo ra toàn bộ số liệu công bố trong paper
(README: *"All numbers reported in the paper are produced by this script"*). Nó không dùng `full_eval.py`
(mục 13.3) mà gọi trực tiếp `train.py` → `render.py` → `metrics.py` cho từng scene, với **bốn cấu hình siêu
tham số khác nhau** theo loại dataset — không có một bộ tham số "một cỡ vừa tất cả" như pipeline cũ từng có
(`Config` mặc định trong pipeline cũ).

| Switch | Dataset | Scenes | Đặc điểm cấu hình |
|---|---|---|---|
| 1 | Tanks & Temples | `train`, `truck` | `iterations=7000`, `ks_scale_power 1.2`, `highfeature_lr 0.01`, `split_ratio_threshold 0.4`, `mult 0.7` |
| 2 | Mip-NeRF 360 (trong nhà) | `bonsai`, `counter`, `kitchen`, `room` | `iterations=3000`, `scale_rotation_scheduler`, `adam_eps_order 10`, `batch_size 2`, `mult 0.25` |
| 3 | Mip-NeRF 360 (ngoài trời, trừ garden) | `bicycle`, `stump`, `flowers`, `treehill` | `iterations=3000`, `split_ratio_threshold 0.4`, `mult 0.25` |
| — | Mip-NeRF 360 — `garden` | `garden` | giống config 3 + `adam_eps_order 10`, chạy tách riêng (không nằm trong hàm `run_dataset_pipeline`) |
| 4 | Deep Blending | `drjohnson`, `playroom` | `iterations=3000`, `ks_scale_power 1.2`, `opacity_reset_decay 0.01`, `freq_opacity_threshold 0.2`, `batch_size 2`, `mult 0.25` |

Cờ chung cho cả bốn config: `--eval --optimizer_type hybrid --sample_bbox_faces --warmup_densification`, cùng
cặp `--densification_interval 500` và `--save_iterations` bằng đúng `--iterations`. Sau train, mỗi scene được
render (`render.py -m output/<scene> --skip_train --iteration <n> --mult <cùng mult đã dùng khi train>`) rồi
chấm điểm (`metrics.py -m output/<scene>`) — **không có** bước đóng gói `submission.zip` hay tính điểm tổng
hợp `Score` như pipeline cũ (`metrics.py` ở đây là bản gốc 3DGS, chỉ in PSNR/SSIM/LPIPS ra
`results.json`, không có công thức Score $0.4(1-\text{LPIPS})+0.3\,\text{SSIM}+0.3\,\widehat{\text{PSNR}}$).

**Vì sao iterations khác nhau theo dataset.** Ngân sách vòng lặp không đồng nhất — Tanks & Temples dùng 7000,
ba cấu hình còn lại chỉ 3000. Đây là quyết định thiết kế thật của tác giả paper (đọc trực tiếp từ
`run_train.sh`), không phải giá trị mặc định của `arguments/__init__.py` (nơi `iterations=30_000`, kế thừa
nguyên vẹn từ 3DGS gốc) — nghĩa là **mọi cấu hình thật sự dùng trong `run_train.sh` đều rút ngắn đáng kể** so
với ngân sách 30k mặc định của `OptimizationParams`, khác hẳn cách pipeline cũ khuyến nghị giữ 30000 vòng
để hưởng trọn lịch Adam thưa dần (Chương 12).

**`--mult`** (hệ số compact box, Chương 8) cũng khác theo dataset: `0.7` cho Tanks & Temples (cảnh có nền
phức tạp, cần hộp bao rộng hơn), `0.25` cho ba cấu hình còn lại — thấp hơn hẳn giá trị mặc định `0.7` trong
`arguments/__init__.py:114`. Không có giải thích bằng công thức nào trong code cho hai con số này; đây là
kết quả tinh chỉnh thực nghiệm của tác giả, không suy ra được từ lý thuyết compact box một mình.

## 13.3 — `full_eval.py`: kế thừa từ 3DGS gốc, KHÔNG khớp `train.py` hiện tại

`SADGS/full_eval.py` giữ nguyên header bản quyền Inria và cấu trúc của script `full_eval.py` gốc trong repo
3DGS chính thức (Kerbl et al. 2023) — có hai chế độ `--mode big` (ngân sách `final_count` cố định theo từng
scene, bảng `big_budgets`) và `--mode budget` (hệ số nhân theo `budget_multipliers`), gọi `train.py` với cờ
`--budget <n> --mode final_count` hoặc `--budget <k> --mode multiplier`.

**Vấn đề:** `SADGS/train.py` (đọc trực tiếp từ `ArgumentParser`/`ModelParams`/`OptimizationParams` ở mục
517-529 và `arguments/__init__.py`) **không định nghĩa cờ `--budget` hay `--mode`** ở bất kỳ đâu. Hai cờ này
chỉ tồn tại trong phiên bản 3DGS-MCMC/Taming-3DGS gốc mà `full_eval.py` được copy từ đó — chạy
`full_eval.py` trên `SADGS/train.py` hiện tại **sẽ báo lỗi tham số không hợp lệ** ngay từ scene đầu tiên.
Đây là một điểm mã nguồn không nhất quán, ghi nhận đúng theo tinh thần "chỗ code/tài liệu không khớp, không
suy diễn thêm ý đồ tác giả" đã dùng ở các chương trước — `full_eval.py` gần như chắc chắn là tàn dư
(leftover) chưa được dọn khi fork từ 3DGS gốc, **không phải** quy trình eval thật của SAD-GS. Quy trình eval
thật, chạy được, là `run_train.sh` (mục 13.2).

Hai bảng dữ liệu `big_budgets` và `budget_multipliers` trong `full_eval.py` (dòng 21–52) là số thật, hữu ích
để tham khảo ngân sách Gaussian cuối cùng mà 3DGS gốc dùng làm chuẩn so sánh — không phải số của SAD-GS:

| Scene | `big_budgets` (final_count) | `budget_multipliers` |
|---|---|---|
| bicycle | 5,987,095 | 15 |
| flowers | 3,618,411 | 15 |
| garden | 5,728,191 | 15 |
| stump | 4,867,429 | 15 |
| treehill | 3,770,257 | 15 |
| room | 1,548,960 | 2 |
| counter | 1,190,919 | 2 |
| kitchen | 1,803,735 | 2 |
| bonsai | 1,252,367 | 2 |
| truck | 2,584,171 | 2 |
| train | 1,085,480 | 2 |
| playroom | 2,326,100 | 5 |
| drjohnson | 3,273,600 | 5 |

## 13.4 — Tham số nào quyết định chi phí ($N$, $K$) trong SAD-GS

Đọc trực tiếp `arguments/__init__.py:107-156` (đối chiếu Chương 12 và Chương 17), các tham số ảnh hưởng số
Gaussian $N$ cuối cùng và số tile $K$ mỗi Gaussian:

| Tham số | Mặc định | Vai trò trong chi phí |
|---|---|---|
| `mult` | 0.7 | hệ số compact box — điều khiển $K$ trực tiếp (Chương 8); `run_train.sh` hạ xuống 0.25 cho hầu hết scene |
| `densification_interval` | 100 | mặc định gốc, nhưng `run_train.sh` luôn ép `500` — giảm tần suất tính $\eta$/split-mask (mục 13.1) |
| `densify_grad_threshold`, `densify_grad_abs_threshold` | 0.0002 / 0.0004 | ngưỡng gradient gốc 3DGS, vẫn được giữ làm một trong hai điều kiện OR với `structure_boost_mask` (Chương 17, mục 17.6) |
| `split_ratio_threshold`, `prune_ratio_threshold` | 0.8 / 0.8 | ngưỡng tỉ lệ high/low-ratio quyết định split/prune theo $\eta$ (Chương 17, mục 17.4) — `run_train.sh` hạ `split_ratio_threshold` xuống `0.4` cho config 1 và 3, tức **dễ split hơn** (chỉ cần 40% số lần quan sát bị flag, thay vì 80%) |
| `max_clones_per_axis` | 8 | trần số bản sao mỗi trục khi `adaptive_clone=True` — không có trần tương đương cho `densify_and_split_structgs` (Chương 17, mục 17.5 đã ghi nhận thiếu clamp max cho $k_{\text{axis}}$) |
| `ks_scale_power` | 1.0 | số mũ chia scale khi split; `run_train.sh` dùng `1.2` cho config 1 và 4 — co Gaussian con mạnh hơn tuyến tính |
| `prune_from_iter`, `prune_until_iter`, `prune_interval` | 6000 / 30000 / 3000 | lịch prune multinomial kế thừa từ cơ chế ADC gốc (Chương 12); với `iterations=3000` hoặc `7000` của `run_train.sh`, `prune_from_iter=6000` **không bao giờ kích hoạt** — nghịch lý giữa giá trị mặc định (thiết kế cho ngân sách 30k) và ngân sách thật đang dùng, tương tự cảnh báo `opacity_reset_interval` không co giãn đã ghi ở Chương 14 cũ |
| `batch_size` | 1 | số camera tích luỹ gradient trước một bước Adam; config 2 và 4 dùng `2` — giảm số bước Adam/N cho cùng số ảnh nhìn thấy, ảnh hưởng trực tiếp số hạng $cN\cdot\mathbb1[\text{Adam}]$ |

**Cảnh báo `prune_from_iter=6000` với ngân sách 3000 vòng.** Ba trong bốn config của `run_train.sh` (2, 3, 4)
chạy đúng `--iterations 3000`. Vì `prune_from_iter` mặc định là `6000` và không được `run_train.sh` ghi đè ở
các switch này, nhánh prune multinomial theo lịch (`prune_from_iter`→`prune_until_iter`, bước `prune_interval`)
**không bao giờ chạy** trong các cấu hình đó — toàn bộ việc giảm $N$ ở ba config này chỉ tới từ
`densify_and_prune_structgs`, hàm chạy trong khối densify chính (dùng ngưỡng `low_ratio`/`prune_ratio_threshold`
của Chương 17), không phải từ prune theo lịch cố định. Điều này không được nói rõ ở đâu trong README hay
comment của `run_train.sh` — suy ra trực tiếp từ đối chiếu số vòng lặp với giá trị mặc định tham số.

## 13.5 — Không có mô hình chi phí đo được, và vì sao

Khác với pipeline cũ (từng có `demos/fastgs_cost_model.py` chạy được không cần GPU, đo ba tỉ số
$R_{\text{tile}}$, $R_{\text{gauss}}$, $R_{\text{adam}}$ và suy ra tốc độ tổng), `SADGS/` **không có** script mô
phỏng hay script benchmark nào. Không có `output/` với log thật, không có `history.csv`, không có leaderboard.
Muốn có một mô hình chi phí định lượng tương tự cho SAD-GS so với 3DGS gốc, cần tự:

1. Chạy `run_train.sh` (hoặc từng lệnh `train.py` riêng lẻ trong đó) trên một máy có CUDA, với hai nhánh —
   một dùng `densify_and_prune_structgs` (mặc định, nhánh chính `train.py:323-386`), một ép về nhánh warmup
   gọi thẳng `densify_and_prune` gốc 3DGS (`train.py:390` — chỉ chạy khi cấu hình bật `warmup_densification`
   ở giai đoạn đầu, xem Chương 17 mục 17.0) để có đường so sánh.
2. Ghi lại số Gaussian theo vòng lặp ($N(t)$), thời gian mỗi vòng, VRAM đỉnh — không có hook sẵn trong
   `train.py` để tự động ghi CSV (khác `pipeline/trainer.py` của pipeline cũ); phải tự thêm logging.
3. So `results.json` do `metrics.py` xuất ra (PSNR/SSIM/LPIPS chuẩn 3DGS, không có Score tổng hợp) giữa hai
   nhánh, trên cùng scene, cùng `--iterations`, cùng `--mult`.

Không có bước nào trong ba bước trên đã được thực hiện trong repo tại thời điểm viết chương này. Mọi phát
biểu về "SAD-GS hội tụ nhanh hơn" trong README là tuyên bố của paper gốc (đối chiếu bên ngoài repo:
[trang dự án](https://vcai.mpi-inf.mpg.de/projects/SAD-GS/)), không phải số đo có sẵn trong mã nguồn này.

---

[← Chương 12](12-adaptive-density-control.md) | [Mục lục](00-muc-luc.md) | [Chương 14 →](14-trien-khai-colab-nhat-ky-train.md)
