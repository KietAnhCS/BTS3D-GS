[← Mục lục](00-muc-luc.md) · Chương 3/15

> Nguồn: `DOCS/DIGITAL-TWIN-GS-PIPELINE-2.md` (toàn văn, §19–§33)

# KIẾN TRÚC SADGS (2/3) — Một vòng lặp train: render, loss, backward, densify

> ⚠ **Ghi chú phạm vi.** Repo này (`SADGS/`) chỉ có một điểm vào huấn luyện:
> `SADGS/train.py`, không có thư mục `pipeline/`. Chương này đối chiếu vòng lặp
> train thật của `train.py` với `pipeline/trainer.py::train_scene()` của
> **SADGS-lite** — một fork Colab tách biệt, ngoài repo này (xem
> [Chương 1 §1.2](01-gioi-thieu-tong-quan.md)) — để cho thấy SADGS-lite chỉ đóng
> gói lại chứ không đổi thuật toán. Mọi cột/hàng nói tới `pipeline/trainer.py`
> dưới đây là mã của fork đó, không phải file trong `SADGS/`.

Tài liệu này mổ xẻ đúng một vòng lặp `for iteration in range(...)` của `SADGS/train.py` (repo này): từ lúc bốc camera, gọi rasterizer CUDA, tính loss, `loss.backward()`, đến các bước điều khiển mật độ Gaussian (densify/prune/reset) và bước `optimizer.step()`. Nội dung được đối chiếu trực tiếp với `train.py` (đường dòng lệnh, repo này) và, để so sánh, với `pipeline/trainer.py::train_scene()` (SADGS-lite, ngoài repo này) — không mang lại bất kỳ khái niệm nào từ bản tài liệu cũ (mask SAM2, DUSt3R, `pose_optimizer`, `compute_combined_loss`... — những thứ đó **không tồn tại** trong mã nguồn hiện tại).

## MỤC LỤC

**PHẦN IV — MỘT VÒNG LẶP TRAIN**
- §19. Khung một vòng lặp trong repo này, đối chiếu SADGS-lite
- §20. `update_learning_rate` và `oneupSHdegree`
- §21. Bốc camera khỏi `viewpoint_stack`
- §22. `render_structgs` — cấu hình rasterizer và compact box `--mult`
- §23. Gọi rasterizer CUDA
- §24. `l1_loss` và `fused_ssim`
- §25. Loss tổng và `lambda_dssim`

**PHẦN V — BACKWARD VÀ KIỂM SOÁT MẬT ĐỘ** *(agent khác viết, chỉ liệt kê ở đây)*
- §26. `loss.backward()` — gradient chảy đi đâu
- §27. `add_densification_stats`
- §28. `sampling_cameras` + `update_freq_stats_online` (structure tensor đa tỉ lệ, nhất quán đa góc nhìn)
- §29. `densify_and_prune_structgs` — điều kiện kép, clone vs split
- §30. Ba tầng pruning + `final_prune_structgs`
- §31. Phẫu thuật trạng thái Adam
- §32. `reset_opacity`
- §33. `optimizer_step`, lịch learning rate và lưu `.ply`

---

# PHẦN IV — MỘT VÒNG LẶP TRAIN

## 19. Khung một vòng lặp trong repo này, đối chiếu với SADGS-lite

`SADGS/train.py::training()` (repo này, chạy từ CLI) là cài đặt thật duy nhất
trong repo này. Fork **SADGS-lite** (ngoài repo, §1.2 Chương 1) có một bản
tương đương, `pipeline/trainer.py::train_scene()`, dùng trong notebook Colab,
gọi lại đúng các hàm SADGS của `SADGS/` như thư viện. `train_scene` **gọi cùng
các hàm lõi** (`render_structgs`, `l1_loss`, `fast_ssim`,
`add_densification_stats`, `update_freq_stats_online`,
`densify_and_prune_structgs`, `final_prune_structgs`, `optimizer_step`) theo
đúng thứ tự và đúng ngưỡng số như `train.py`, chỉ khác phần vỏ bọc theo
dõi/lưu — bảng dưới đây minh hoạ điều đó bằng cách đối chiếu song song.

Bảng đối chiếu từng bước trong một iteration (file:line là vị trí bắt đầu của khối lệnh tương ứng):

| Bước trong 1 iteration | `train.py::training()` (repo này) | `pipeline/trainer.py::train_scene()` (SADGS-lite, ngoài repo) |
|---|---|---|
| Cầu nối viewer qua websocket (nếu bật `--websockets`) | `train.py:67-73` | **không có** |
| `update_learning_rate(iteration)` | `train.py:78` | `pipeline/trainer.py:91` |
| `oneupSHdegree()` mỗi 1000 iter | `train.py:81-82` | `pipeline/trainer.py:92-93` |
| Bốc camera khỏi `viewpoint_stack`/`viewpoint_indices` | `train.py:86-91` | `pipeline/trainer.py:95-100` |
| Bật `pipe.debug` tại `--debug_from` | `train.py:94-95` | **không có** (không nhận `debug_from`) |
| `render_structgs(...)` | `train.py:97-98` | `pipeline/trainer.py:104` |
| `l1_loss` + `fast_ssim` + loss tổng | `train.py:101-103` | `pipeline/trainer.py:108-110` |
| `loss.backward()` | `train.py:104` | `pipeline/trainer.py:111` |
| Cập nhật thanh tiến trình (EMA loss) | `train.py:110-114` (tqdm, mỗi 10 iter) | `pipeline/trainer.py:114`, `146-149` (tqdm + hậu tố Score/PSNR mỗi 10 iter) |
| Lưu checkpoint theo lịch | `train.py:120-122` (`saving_iterations`, gọi `scene.save`) | `pipeline/trainer.py:154-156` (`cfg.save_every`, gọi `_save_checkpoint` — lưu `.ply` rồi xoá checkpoint cũ) |
| Densify: `max_radii2D` + `add_densification_stats` | `train.py:129-131` | `pipeline/trainer.py:118-120` |
| `densify_and_prune_structgs` mỗi `densification_interval` | `train.py:133-142` | `pipeline/trainer.py:121-128` |
| `reset_opacity` mỗi `opacity_reset_interval` | `train.py:144-145` | `pipeline/trainer.py:129-130` |
| Pruning đa góc nhìn 3 tầng (`final_prune_structgs`, 15k–30k, mỗi 3000 iter) | `train.py:148-156` | `pipeline/trainer.py:132-135` |
| `optimizer_step` | `train.py:159-160` | `pipeline/trainer.py:137-141` |
| Đo thời gian bằng CUDA event | `train.py:53-54, 108, 165-169` | **không có** — dùng `time.time()` tổng thể (`pipeline/trainer.py:80, 165`) |

Những gì **`train_scene` thêm** so với `train.py`:
- Chấm điểm trực tuyến định kỳ mỗi `score_every` iteration qua `evaluate_cameras` (đọc PSNR/SSIM/LPIPS/Score chuẩn hoá trên tập `holdout`), có in delta so với lần chấm trước (`d_score`, `d_psnr`, …) — `pipeline/trainer.py:146-168`.
- Hậu tố tiến trình tqdm hiển thị % hoàn thành, số Gaussian, loss, Score, PSNR ngay trên thanh (`pipeline/trainer.py:146-149`).
- Checkpoint định kỳ theo `cfg.save_every`, tự xoá checkpoint giữa chừng trước đó để tiết kiệm đĩa Colab (`_save_checkpoint`, `pipeline/trainer.py:41-47`).
- Giới hạn RAM mềm: nếu `usage["ram_used_gb"] > cfg.ram_soft_limit_gb` thì gọi `gc.collect()` + `torch.cuda.empty_cache()` (`pipeline/trainer.py:169-171`).
- Gọi `safe_state(True)` — tức luôn chạy silent/quiet (`pipeline/trainer.py:59`), khác `train.py` truyền `args.quiet` từ CLI (`train.py:266`).
- Dọn bộ nhớ tường minh mỗi iteration (`del pkg, image, gt, viewspace, visibility, radii, loss, ll1, ssim_value` — `pipeline/trainer.py:172`) và dọn toàn bộ model sau khi train nếu `keep_model=False` (`pipeline/trainer.py:182-188`).

Những gì **`train_scene` bỏ** so với `train.py` (đã kiểm tra từng cái trong `train.py`):
- Cầu nối viewer qua websocket: `train.py` import `network_gui_ws` và có khối `if websockets: ...` gửi ảnh render trực tiếp cho client xem trực tiếp (`train.py:16, 67-73, 261-263, 273`). `train_scene` không import `gaussian_renderer.network_gui_ws`, không có tham số nào tương đương — xác nhận đã bỏ.
- Tensorboard: `train.py` có `TENSORBOARD_FOUND`, `SummaryWriter`, hàm `training_report()` ghi scalar/ảnh lên TB (`train.py:28-31, 200, 209-243`) — nhưng lưu ý dòng gọi thực tế `training_report(...)` trong vòng lặp chính đã bị **comment out** (`train.py:118`), nên ngay trong `train.py` nó cũng không chạy khi train thường. `train_scene` không có bất kỳ đoạn nào liên quan TensorBoard.
- `--checkpoint_iterations` / `checkpoint` / `--start_checkpoint`: `train.py` nhận `checkpoint_iterations` và `checkpoint` làm tham số của `training()` nhưng bên trong hàm chỉ dùng `checkpoint` để `torch.load` model phục hồi (`train.py:36-39`) — **không có đoạn nào trong `training()` lưu theo `checkpoint_iterations`** (chỉ có `saving_iterations` được dùng để `scene.save`). `train_scene` không nhận và không dùng khái niệm `checkpoint_iterations`/`start_checkpoint` nào cả — nó tự quản lý qua `cfg.save_every`.
- `--debug_from`: `train.py` bật `pipe.debug = True` khi `(iteration - 1) == debug_from` (`train.py:94-95`), dùng để dump snapshot lỗi CUDA. `train_scene` không nhận `debug_from`, `pipe.debug` giữ nguyên giá trị mặc định `False` suốt quá trình train qua notebook.
- `torch.autograd.set_detect_anomaly(args.detect_anomaly)`: chỉ có ở khối `if __name__ == "__main__"` của `train.py:269`, không có trong `pipeline/trainer.py`.

## 20. `update_learning_rate` và `oneupSHdegree`

`GaussianModel.training_setup(training_args)` (`scene/gaussian_model.py:162-180`) tạo 5 param-group cho optimizer chính (`self.optimizer`), mỗi nhóm có `lr` riêng cố định trừ `xyz`:

| Param group | `lr` khởi tạo | Có được `update_learning_rate` cập nhật mỗi iteration không |
|---|---|---|
| `xyz` | `position_lr_init * spatial_lr_scale` | **Có** — theo lịch exponential (`get_expon_lr_func`) |
| `f_dc` | `lowfeature_lr` | Không — cố định suốt training |
| `opacity` | `opacity_lr` | Không |
| `scaling` | `scaling_lr` | Không |
| `rotation` | `rotation_lr` | Không |
| `f_rest` (trong `shoptimizer` riêng) | `highfeature_lr / 20.0` | Không |

`self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)` truyền `lr=0.0` làm giá trị mặc định của `Adam`, nhưng từng phần tử trong `l` đã tự mang khoá `'lr'` riêng nên override giá trị này — `lr=0.0` thực chất không bao giờ được dùng.

`update_learning_rate(self, iteration)` (`scene/gaussian_model.py:182-188`):
```python
for param_group in self.optimizer.param_groups:
    if param_group["name"] == "xyz":
        lr = self.xyz_scheduler_args(iteration)
        param_group['lr'] = lr
        return lr
```
Nó chỉ tìm đúng group tên `"xyz"`, gán `lr` mới rồi `return lr` ngay lập tức (bỏ qua các group còn lại trong vòng `for` — không cần `break` vì đã `return`). Giá trị trả về không được `train.py`/`train_scene` sử dụng (lời gọi `gaussians.update_learning_rate(iteration)` không gán biến nào) — hàm được gọi vì tác dụng phụ (side-effect) lên `param_group['lr']`.

`xyz_scheduler_args` được tạo bởi `get_expon_lr_func(lr_init=position_lr_init*spatial_lr_scale, lr_final=position_lr_final*spatial_lr_scale, max_steps=position_lr_max_steps)` (`scene/gaussian_model.py:178-180`) — lịch suy giảm mũ (log-linear interpolation giữa `lr_init` và `lr_final` theo `iteration/max_steps`, có delay mult ở giai đoạn đầu).

`oneupSHdegree()` (`scene/gaussian_model.py:133-135`):
```python
def oneupSHdegree(self):
    if self.active_sh_degree < self.max_sh_degree:
        self.active_sh_degree += 1
```
Cả hai vòng lặp gọi hàm này mỗi 1000 iteration (`if iteration % 1000 == 0`) — mỗi lần tăng bậc SH (spherical harmonics) lên 1, cho tới khi chạm `max_sh_degree` (= `dataset.sh_degree`, mặc định 3). `active_sh_degree` là bậc SH thực sự được rasterizer dùng (`raster_settings.sh_degree=pc.active_sh_degree`, §22) — tăng dần độ chi tiết màu theo góc nhìn khi training tiến triển, tránh học nhiễu bậc cao ngay từ đầu khi hình học còn chưa ổn định.

## 21. Bốc camera khỏi `viewpoint_stack`

Cả hai bản đều dùng cùng một mẫu lấy-không-hoàn-lại (sampling without replacement) mỗi iteration, thay vì đơn giản `random.choice`:

```python
if not viewpoint_stack:
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
rand_idx = randint(0, len(viewpoint_indices) - 1)
viewpoint_cam = viewpoint_stack.pop(rand_idx)
_ = viewpoint_indices.pop(rand_idx)
```
(`train.py:85-91`; tương đương ở `pipeline/trainer.py:95-100` với tên biến `stack`/`indices`/`pick`/`cam`).

`viewpoint_stack` là danh sách camera train, bị `pop` dần mỗi iteration cho tới khi rỗng thì nạp lại toàn bộ (`scene.getTrainCameras().copy()`) — đảm bảo mỗi camera được thấy đúng một lần mỗi "epoch" (một vòng qua hết N camera) trước khi lặp lại, thay vì có thể bốc trùng liên tục.

`viewpoint_indices` là một danh sách chỉ số song song (`0..N-1`), bị `pop` cùng vị trí `rand_idx` với `viewpoint_stack`. Bản thân giá trị `_ = viewpoint_indices.pop(rand_idx)` không được dùng ở đâu khác trong vòng lặp — nó tồn tại chỉ để giữ độ dài `len(viewpoint_indices)` luôn đồng bộ với `len(viewpoint_stack)`, làm biên trên cho `randint(0, len(viewpoint_indices) - 1)`. Về mặt logic, `randint(0, len(viewpoint_stack) - 1)` sẽ cho kết quả y hệt — việc giữ danh sách song song là di sản viết code (có thể để tiện debug chỉ số gốc), không mang thêm thông tin nào khác trong bản hiện tại.

## 22. `render_structgs` — cấu hình rasterizer và compact box `--mult`

Đọc `gaussian_renderer/__init__.py:18-110` theo đúng thứ tự:

**Chữ ký hàm** (`gaussian_renderer/__init__.py`):
```python
def render_structgs(viewpoint_camera, pc: GaussianModel, pipe, bg_color, mult,
                     scaling_modifier=1.0, override_color=None, get_flag=None,
                     metric_map=None, compute_extra=False)
```
Vòng lặp train chính gọi `render_structgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)` — không truyền `get_flag`/`metric_map`/`compute_extra` (dùng giá trị mặc định). `compute_extra` là cờ riêng của SADGS điều khiển việc rasterizer có tính thêm các bản đồ phụ trợ (`depth_map`, `opacity_map`, `normal_map`, xem bên dưới) hay để chúng rỗng để tiết kiệm compute khi không cần.

**`screenspace_points`** (dòng 27):
```python
screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
```
Đây là tensor "giả" toàn số 0, shape `(N, 4)` (chú thích ở dòng 26 còn ghi shape cũ `(N,3)` — dấu vết code cũ, không khớp dòng thực thi ngay dưới nó dùng shape `(N,4)`). Vì `requires_grad=True` và được truyền làm `means2D` vào rasterizer, PyTorch autograd sẽ gán vào `.grad` của nó đúng gradient của loss theo toạ độ màn hình 2D (screen-space) của từng Gaussian — cách "mượn" gradient này (thay vì `pc.get_xyz` không có gradient màn-hình trực tiếp) là kỹ thuật chuẩn của 3DGS để lấy `viewspace_points.grad` phục vụ `add_densification_stats` (§27). Vì `screenspace_points` không phải leaf-tensor được optimizer theo dõi (không nằm trong `training_setup`), phải gọi `retain_grad()` (dòng 29, bọc trong `try/except` phòng trường hợp không cần gradient — ví dụ khi `torch.no_grad()`).

**`tanfovx`/`tanfovy`** (dòng 34-35): `tan(FoVx/2)`, `tan(FoVy/2)` — nửa góc nhìn ngang/dọc của camera, dùng để rasterizer chuyển toạ độ camera-space sang screen-space (tham số chuẩn của phép chiếu perspective).

**`metric_map`**: nếu không truyền, khởi tạo tensor 0 kiểu `int` dài `H*W` trên CUDA — buffer đếm cũ còn lại từ kiến trúc rasterizer, không được đường densify chính của SADGS sử dụng nữa (thay bằng `cov2D`/`update_freq_stats_online`, §28); trong render bình thường của vòng lặp train nó chỉ là buffer rỗng không dùng tới.

**`GaussianRasterizationSettings`** (dòng 40-56) — mọi field và nguồn gốc:

| Field | Giá trị | Ý nghĩa |
|---|---|---|
| `image_height`, `image_width` | từ `viewpoint_camera` | kích thước ảnh cần render |
| `tanfovx`, `tanfovy` | tính ở trên | góc nhìn |
| `bg` | `bg_color` (tham số hàm) | màu nền (đen hoặc trắng, hoặc ngẫu nhiên nếu `random_background`) |
| `scale_modifier` | `scaling_modifier` (mặc định 1.0) | hệ số nhân thêm lên scale Gaussian |
| `viewmatrix` | `viewpoint_camera.world_view_transform` | ma trận world→camera |
| `projmatrix` | `viewpoint_camera.full_proj_transform` | ma trận chiếu đầy đủ |
| `sh_degree` | `pc.active_sh_degree` | bậc SH hiện tại (xem §20) |
| `campos` | `viewpoint_camera.camera_center` | vị trí camera trong world-space |
| `mult` | tham số hàm `mult` (= `opt.mult`) | hệ số compact-box, xem dưới |
| `prefiltered` | `False` | luôn tắt (không lọc trước Gaussian ngoài frustum ở phía Python) |
| `debug` | `pipe.debug` | bật dump `snapshot_fw.dump`/`snapshot_bw.dump` khi lỗi CUDA (xem §19, `--debug_from`) |
| `get_flag` | tham số hàm (mặc định `None`) | cờ bật thu thập `metric_map` trong kernel CUDA — di sản kiến trúc cũ, không dùng ở đường densify chính |
| `metric_map` | tính ở trên hoặc truyền vào | buffer đếm di sản kiến trúc cũ |
| `compute_extra` | tham số hàm (mặc định `False`) | **riêng SADGS** — bật tính thêm `depth_map`/`opacity_map`/`normal_map` |

**Compact box — `mult`**: giá trị này đi thẳng vào `GaussianRasterizationSettings.mult` rồi xuống kernel CUDA `duplicateToTilesTouched` của rasterizer `diff-gaussian-rasterization_structgs`. Kernel tính ngưỡng cắt hộp bao (bounding box) mỗi splat theo kiểu SNUGBOX: `t = 2*log(opacity*255)`, sau đó `t = mult * t`. `t` càng nhỏ (mult càng nhỏ) → hộp bao ellipse càng hẹp → splat chạm ít tile 16×16 hơn → ít công việc rasterize hơn (nhanh hơn) nhưng có nguy cơ cắt mất phần đuôi mờ của Gaussian nếu `mult` quá nhỏ. Giá trị mặc định SADGS `opt.mult = 0.7` (`arguments/__init__.py`, tăng so với giá trị mặc định `0.5` trước đó). **`--mult` phải khớp giữa `train.py` và `render.py`** khi render lại sau train vì nó ảnh hưởng trực tiếp đến hình dạng splat được rasterize.

**`compute_cov3D_python` / `convert_SHs_python`** (cả hai mặc định `False` trong `PipelineParams`), giống hệt cơ chế 3DGS gốc:
- Nếu `pipe.compute_cov3D_python=True`: `cov3D_precomp = pc.get_covariance(scaling_modifier)` — hiệp phương sai 3D tính sẵn ở phía Python.
- Nếu `False` (mặc định): `scales = pc.get_scaling_with_3D_filter`, `rotations = pc.get_rotation` truyền thẳng, rasterizer CUDA tự dựng ma trận hiệp phương sai. **Khác 3DGS gốc**: SADGS đọc `get_scaling_with_3D_filter` (một property riêng áp low-pass filter kiểu Mip-Splatting nếu `opt.compute_3d_filter=True`, mặc định `False` nên tương đương `get_scaling` thường) chứ không phải `get_scaling` trực tiếp — tương tự `opacity = pc.get_opacity_with_3D_filter` thay vì `get_opacity`.
- Nếu `pipe.convert_SHs_python=True`: SH eval thành RGB ngay ở Python; ngược lại kernel CUDA tự eval SH→RGB, tách riêng `dc`/`shs` vì SADGS dùng hai learning-rate riêng cho hai nhóm SH (`lowfeature_lr`/`highfeature_lr`).

## 23. Gọi rasterizer CUDA — bảy giá trị trả về, không phải ba

`rasterizer(means3D=..., means2D=..., dc=..., shs=..., colors_precomp=..., opacities=..., scales=..., rotations=..., cov3D_precomp=...)` trả về **7 giá trị** — nhiều hơn hẳn 3DGS gốc (`color, radii`):
```python
rendered_image, radii, accum_metric_counts, cov2D, depth_map, opacity_map, normal_map = rasterizer(...)
```
- `accum_metric_counts` — di sản buffer đếm của kiến trúc rasterizer cũ, vẫn được trả ra nhưng không dùng ở đường densify chính.
- `cov2D` — tensor `[N, 7]` **mới của SADGS**: `[cov_xx, cov_xy, cov_yy, mean_x, mean_y, depth, max_transmittance]` cho mỗi Gaussian tại camera hiện tại — đây chính là dữ liệu `update_freq_stats_online` dùng để tính hệ số vi phạm Nyquist $\eta$ mà không cần render thêm lượt phụ (§28).
- `depth_map`, `opacity_map`, `normal_map` — các bản đồ phụ trợ theo pixel (độ sâu, opacity tích luỹ, pháp tuyến ước lượng), chỉ được tính đầy đủ khi `compute_extra=True`; vòng lặp train chính không bật cờ này nên các bản đồ này thường rỗng/không dùng.

`render_structgs` build dict trả về:
```python
return {"render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "accum_metric_counts": accum_metric_counts,
        "means2D": means2D,
        "cov2D": cov2D,
        "depth_map": depth_map,
        "opacity_map": opacity_map,
        "normal_map": normal_map}
```
`visibility_filter` vẫn là `(radii > 0).nonzero()` — tensor chỉ số, không phải boolean mask thuần, giống hệt 3DGS gốc (nhưng `train.py` gọi thêm `.squeeze(1)` ngay sau khi nhận về để dùng làm boolean/1-D index trong `update_freq_stats_online`, §28).

`radii > 0` là điều kiện một Gaussian có "bán kính màn hình" dương — tức nó chạm ít nhất một pixel sau khi chiếu và cắt bởi compact box (§22); Gaussian bị frustum-cull hoặc quá nhỏ/mờ sẽ có `radii = 0` và bị loại khỏi các thống kê densify/prune.

**Forward pass tile-based (tóm tắt cơ chế CUDA, không phải mã Python)**: rasterizer chia ảnh thành các tile `16×16` pixel (`#define BLOCK_X 16`, `#define BLOCK_Y 16`, `cuda_rasterizer/config.h:16-17`). Với mỗi tile, các Gaussian chạm tile đó (xác định bởi `duplicateToTilesTouched`, §22) được sắp xếp theo độ sâu (depth, front-to-back) rồi blend tuần tự theo công thức alpha-blending chuẩn (`cuda_rasterizer/forward.cu`, vòng lặp quanh dòng 333-413):
```
alpha = min(0.99, opacity * exp(power))      # power: hàm mũ Gaussian 2D tại pixel
C    += color * alpha * T                     # cộng dồn màu, trọng số bởi độ trong suốt còn lại T
T    *= (1 - alpha)                           # cập nhật độ trong suốt còn lại cho lớp sau
```
Vòng lặp dừng sớm khi `T < 0.0001` (tile coi như đã bão hoà, các Gaussian xa hơn không còn đóng góp đáng kể) — đây là early-termination chuẩn của 3DGS, không phải cơ chế riêng của SADGS.

## 24. `l1_loss` và `fused_ssim`

`utils/loss_utils.py:20-21`:
```python
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()
```
Trả về **một scalar** — trung bình trị tuyệt đối sai khác trên toàn bộ tensor (mọi pixel, mọi kênh màu), không phải per-pixel hay per-channel.

File này còn định nghĩa `ssim(img1, img2, window_size=11, size_average=True)` (dòng 36-44) — SSIM cổ điển cài bằng `conv2d` với cửa sổ Gaussian 11×11 tự viết bằng PyTorch thuần (không CUDA). Với `size_average=True` (mặc định) nó trả về **một scalar** (`ssim_map.mean()`); nếu `False` trả về vector theo batch.

Vòng lặp train (cả `train.py` và `train_scene`) **không dùng** `ssim` này — nó import `fused_ssim as fast_ssim` từ package `fused_ssim` (một CUDA submodule ngoài, biên dịch riêng), gọi `fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))`. Lý do: `fused_ssim` chạy trên GPU bằng kernel CUDA fused (gộp nhiều phép convolution/elementwise vào một kernel) nên nhanh hơn đáng kể so với `ssim` cài bằng `F.conv2d` tuần tự — quan trọng vì SSIM được tính lại mỗi iteration training.

Ngược lại, `metrics.py` (script đánh giá sau khi train xong, chạy một lần trên toàn bộ test set chứ không phải mỗi iteration) dùng `from utils.loss_utils import ssim` (`metrics.py:17`, gọi ở dòng 72) — bản Python thuần, không cần biên dịch CUDA riêng, đơn giản và dễ tái lập cho việc đánh giá cuối cùng, tốc độ không phải ưu tiên ở đây.

Tóm lại:

| Nơi dùng | Hàm | Cài đặt |
|---|---|---|
| Vòng lặp train (`train.py`, `pipeline/trainer.py`) | `fast_ssim` (alias của `fused_ssim`) | CUDA fused kernel (submodule ngoài) |
| Đánh giá cuối (`metrics.py`) | `ssim` | PyTorch thuần, `utils/loss_utils.py` |

## 25. Loss tổng và `lambda_dssim`

Công thức loss, giống hệt ở cả hai vòng lặp (`train.py:103`, `pipeline/trainer.py:110`):
```python
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
```
tức `(1 - λ)·L1 + λ·(1 - SSIM)`, với `λ = opt.lambda_dssim`.

Giá trị mặc định: `self.lambda_dssim = 0.2` (`arguments/__init__.py:82`, trong `OptimizationParams`). `README.md:173` xác nhận cùng giá trị mặc định `0.2`.

Preset notebook ghi đè: `pipeline/config.py:44` truyền `"--lambda_dssim", "0.25"` vào `build_args` — tức đường train qua notebook luôn train với `λ = 0.25`, coi trọng SSIM (cấu trúc ảnh) hơn một chút so với mặc định CLI `0.2`. Đây là ví dụ minh hoạ số cụ thể của cấu hình mặc định notebook, không phải quy tắc cố định — người dùng CLI hoàn toàn có thể tự truyền `--lambda_dssim` khác.

**Không có mask loss, không có pose loss trong codebase hiện tại.** Toàn bộ `loss` chỉ gồm hai số hạng L1 và DSSIM ở trên — không có `compute_combined_loss`, không có `compute_instance_losses`, không có nhánh cộng thêm theo mask SAM2 hay theo sai số pose như tài liệu cũ mô tả; những hàm/khái niệm đó không tồn tại trong `train.py`, `pipeline/trainer.py`, `utils/loss_utils.py`, hay `utils/fast_utils.py` của codebase hiện tại (đã kiểm tra: hàm `get_loss`/`compute_photometric_loss` trong `utils/fast_utils.py` chỉ được dùng nội bộ trong luồng chấm điểm SADGS-lite ở PHẦN V, không nằm trong công thức loss chính của vòng lặp train).

---

# PHẦN V — BACKWARD VÀ KIỂM SOÁT MẬT ĐỘ

## 26. `loss.backward()` — gradient chảy đi đâu

Cả hai vòng lặp gọi đúng một dòng: `loss.backward()` (`train.py:104`, `pipeline/trainer.py:114`). `loss` là scalar `(1-λ)·L1 + λ·(1-SSIM)` dựng trên `image` — ảnh render trả về từ `render_structgs`. Autograd đi ngược qua `_RasterizeGaussians.apply(...)` (`submodules/diff-gaussian-rasterization_fastgs/diff_gaussian_rasterization_fastgs/__init__.py:33-44`), tức là qua đúng một `torch.autograd.Function` mà `forward`/`backward` được cài bằng CUDA (`_C.rasterize_gaussians` / `_C.rasterize_gaussians_backward`). `_RasterizeGaussians.backward` (dòng 112-172) nhận `grad_out_color` (dL/d ảnh) và trả về gradient cho đúng 9 input đã truyền vào forward, theo thứ tự khai báo:

```
grads = (grad_means3D, grad_means2D, grad_dc, grad_sh, grad_colors_precomp,
          grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, None)
```

Trong `render_structgs` (`gaussian_renderer/__init__.py:60-102`) các input này được gán từ property của `GaussianModel`:

| Input rasterizer | Nguồn (`GaussianModel`) | Activation nằm giữa gradient và tham số thô |
|---|---|---|
| `means3D` | `pc.get_xyz` → `self._xyz` | không có activation, `get_xyz` trả thẳng `_xyz` (`scene/gaussian_model.py:109-111`) |
| `opacities` | `pc.get_opacity` → `sigmoid(self._opacity)` | `torch.sigmoid` (`scene/gaussian_model.py:37, 157-158`) |
| `scales` | `pc.get_scaling` → `exp(self._scaling)` | `torch.exp` (`scene/gaussian_model.py:33`) |
| `rotations` | `pc.get_rotation` → `normalize(self._rotation)` | `torch.nn.functional.normalize` (`scene/gaussian_model.py:40, 135-137`) |
| `dc`, `shs` | `pc.get_features_dc`, `pc.get_features_rest` | không có activation (chỉ transpose) |
| `means2D` | `screenspace_points` (tensor phụ, xem bên dưới) | không có activation |

Vì `get_xyz`, `get_opacity`, `get_scaling`, `get_rotation` đều là `@property` áp activation lên `nn.Parameter` gốc, gradient của rasterizer đối với *giá trị đã activate* sẽ tự động được nhân với đạo hàm của activation đó (chain rule của autograd) khi lan tới `self._scaling`, `self._opacity`, `self._rotation`. Đây là lý do các tham số thô có thể mang giá trị âm/không giới hạn (`_scaling` là log-scale, `_opacity` là logit) trong khi giá trị dùng để render luôn hợp lệ (dương, trong [0,1], đơn vị norm).

**Trick `screenspace_points` / `viewspace_point_tensor`.** `means2D` không phải toạ độ 2D thật của Gaussian trên màn hình — nó là một tensor phụ khởi tạo bằng 0, không liên quan gì tới phép chiếu:

```python
screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype,
                                  requires_grad=True, device="cuda") + 0
screenspace_points.retain_grad()
```
(`gaussian_renderer/__init__.py:26-31`)

`means3D` (toạ độ 3D thật) mới là thứ được rasterizer chiếu sang màn hình để tính `dL/d(mean2D)` nội bộ. Nhưng CUDA kernel *cộng dồn* gradient màn hình đó vào input `means2D` mà nó nhận được — vì `means2D = screenspace_points` (một tensor zero, không đóng góp giá trị gì vào forward, nên phần "gradient của loss theo giá trị của means2D" chính là gradient màn hình thô, không bị pha trộn với gradient của `means3D`). Nói cách khác, `screenspace_points` được dùng như một "cổng dò" (probe) — trị số của nó luôn là 0 nên nó không ảnh hưởng đến ảnh render, nhưng backward vẫn phải tính `dL/d(means2D)` để lan ra `means3D` theo chuỗi phép chiếu, và giá trị trung gian đó được ghi thẳng vào `.grad` của `screenspace_points` nhờ `retain_grad()` (nếu không gọi `retain_grad()`, PyTorch sẽ giải phóng `.grad` của mọi tensor không phải lá sau `backward()`). Do đó sau `loss.backward()`, `render_pkg["viewspace_points"].grad` (đặt tên `viewspace_point_tensor` ở `train.py:97` / `viewspace` ở `pipeline/trainer.py:107`) chính là gradient 2D-màn hình mà `add_densification_stats` cần, không phải suy ra lại từ `means3D.grad`.

Điểm khác lạ so với 3DGS gốc: `screenspace_points` ở đây có **4 cột**, không phải 3 (dòng 27, dòng comment 26 còn giữ lại bản cũ 3 cột đã bị comment-out). Xem §27 để biết ý nghĩa 2 cột sau.

## 27. `add_densification_stats` — điều gì được cộng dồn

```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter]     += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
    self.denom[update_filter] += 1
```
(`scene/gaussian_model.py:493-496`)

- `update_filter` chính là `visibility_filter` — trong cả hai vòng lặp nó được tính là `(radii > 0).nonzero()` (`gaussian_renderer/__init__.py:108`), tức **chỉ số** (không phải mask bool) của các Gaussian có bán kính màn hình > 0 ở lượt render đó. Do đó chỉ Gaussian *nhìn thấy được ở camera vừa render* mới được cộng dồn — Gaussian bị frustum-cull hoặc có `radii == 0` không được cập nhật ở bước này.
- Cột `[:, :2]` của `viewspace_point_tensor.grad` là gradient 2D màn hình "ký hiệu" thông thường — norm Euclid của nó được cộng vào `xyz_gradient_accum`.
- Cột `[:, 2:]` (cột 2 và 3) là gradient **giá trị tuyệt đối** tích luỹ trên GPU: kernel CUDA cộng dồn `fabs(dL/dx)` và `fabs(dL/dy)` theo từng pixel thay vì cộng có dấu (`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/backward.cu:592-596,607-610` — biến `Register_dL_dmean2D_z/_w` dùng `fabs(tmp_x)`, `fabs(tmp_y)`). Đây là kỹ thuật kiểu AbsGS/Pixel-GS: với một Gaussian lớn phủ nhiều pixel có gradient trái dấu (một phần ảnh cần nó dịch trái, phần khác cần dịch phải), gradient có-dấu bị triệt tiêu gần 0 dù Gaussian đó thực sự "kém", còn gradient trị tuyệt đối thì không bị triệt tiêu. Norm của cặp này được cộng vào `xyz_gradient_accum_abs`.
- `denom[update_filter] += 1` đếm số lần Gaussian đó được nhìn thấy (qua các lượt render khác nhau, tức các iteration khác nhau) — dùng làm mẫu số khi lấy trung bình ở `densify_and_prune_structgs` (§29): `grad_vars = xyz_gradient_accum / denom`.

`max_radii2D` **không** được cập nhật trong hàm này — nó được duy trì ngay tại lời gọi, ở caller:
```python
gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
```
(`train.py:129`, tương đương `pipeline/trainer.py:121-122`) — chạy **trước** `add_densification_stats` trong cả hai vòng lặp, dùng `torch.max` theo từng phần tử để giữ lại bán kính lớn nhất từng quan sát được của mỗi Gaussian (dùng làm điều kiện prune "quá to trên màn hình" ở §30).

Tất cả các bộ đếm này (`xyz_gradient_accum`, `xyz_gradient_accum_abs`, `denom`, `max_radii2D`) bị **reset về 0** mỗi khi có Gaussian mới sinh ra hoặc bị xoá, trong `densification_postfix` (dòng 426-429) và `prune_points` (dòng 375-379) — chúng chỉ tích luỹ *giữa hai lần densify liên tiếp*, không phải trong suốt quá trình train.

## 28. Structure tensor đa tỉ lệ + `update_freq_stats_online` — trái tim của SADGS

3DGS gốc không có bước chấm điểm Gaussian riêng theo tần số/texture nào cả — nó chỉ densify theo gradient tích luỹ. Một cách tiếp cận trung gian trước đây từng chấm điểm Gaussian bằng cách render lại 10 camera mẫu mỗi lần densify và đếm số pixel-lỗi photometric (hàm `compute_gaussian_score_fastgs`, đã xoá khỏi codebase). SADGS thay thế **hoàn toàn** hướng đó bằng một tiêu chí **cấu trúc ảnh (image structure)**: so sánh kích thước hình chiếu (screen-space extent) của mỗi Gaussian với **bước sóng cục bộ của texture** đo được từ một structure tensor đa tỉ lệ tính sẵn cho từng camera, rồi tích luỹ bằng chứng đó **trực tuyến, mỗi vài iteration, qua nhiều góc nhìn khác nhau** thay vì render lại một loạt camera riêng tại thời điểm densify.

**Bước 0 — tiền tính structure tensor một lần cho mỗi camera** (`train.py`, trước vòng lặp chính). Ảnh train được nhóm theo độ phân giải rồi xử lý theo batch; với mỗi ảnh, `get_multiscale_structure_tensor_v1` hoặc `_v2` (`utils/loss_utils.py`, chọn qua `opt.st_mode`) tính ma trận cấu trúc $2\times2$ $S=\begin{pmatrix}S_{xx}&S_{xy}\\S_{xy}&S_{yy}\end{pmatrix}$ tại mỗi pixel, gộp qua `opt.st_levels` mức đa tỉ lệ (mặc định 4) — về bản chất là tích của gradient ảnh theo hai hướng, làm mượt ở nhiều tỉ lệ Gaussian-blur khác nhau để nắm được cả cạnh mảnh lẫn texture thô. Kết quả được cache vào `structure_tensor_cache[image_name]`, dùng lại suốt quá trình train (không tính lại mỗi iteration).

**Bước 1 — lấy mẫu camera cho mỗi iteration.** `sampling_cameras(my_viewpoint_stack, mode, num_cams, weights)` (`utils/freq_utils.py`) hỗ trợ hai chế độ: `"random"` (bốc ngẫu nhiên, `pop` khỏi list để không trùng) và `"fps"` — **farthest point sampling** theo vị trí camera trong world-space (suy ra từ `world_view_transform`, chọn lần lượt camera xa nhất với tập đã chọn) để trải đều góc nhìn thay vì bốc ngẫu nhiên thuần. Chế độ chọn qua `opt.camera_sampling` (mặc định `"random"`). Không giống cách tiếp cận trước đây (luôn render đúng 10 camera cố định tại thời điểm densify), SADGS dùng `sampling_cameras` để **thay mới toàn bộ `viewpoint_stack`** mỗi khi rỗng — tức nó là cơ chế lấy mẫu camera cho *toàn bộ vòng lặp train*, không phải một lượt riêng chỉ chạy khi densify.

**Bước 2 — tích luỹ bằng chứng tần số trực tuyến, mỗi 10 iteration.** Trong `training()` (`train.py`), sau mỗi lượt `render_structgs` và `loss.backward()`, nếu `iteration < opt.densify_until_iter và iteration % 10 == 0`, gọi:
```python
update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter,
                          structure_tensor_cache, viewspace_point_tensor=viewspace_point_tensor,
                          grad_threshold=opt.freq_grad_threshold,
                          transmittance_threshold=opt.freq_transmittance_threshold,
                          opacity_threshold=opt.freq_opacity_threshold,
                          eta_compute_mode=opt.eta_compute_mode)
```
`cov2D` (một tensor `[N, 7]` mới — `[cov_xx, cov_xy, cov_yy, mean_x, mean_y, depth, max_transmittance]`) được rasterizer CUDA trả kèm theo ảnh render (`render_pkg["cov2D"]`, xem §22) — đây là dữ liệu hình học 2D của mỗi Gaussian tại camera hiện tại, không cần render thêm lượt thứ hai để tính điểm số như cách tiếp cận trước đây.

`update_freq_stats_online` (`utils/freq_utils.py`), với mỗi camera đi qua, thực hiện:

1. **Lọc Gaussian "đang hoạt động"**: chỉ giữ Gaussian có `max_transmittance > freq_transmittance_threshold` (còn đóng góp thật vào ảnh, không bị Gaussian khác che hết), `opacity > freq_opacity_threshold`, và (nếu có `viewspace_point_tensor`) gradient màn hình `> freq_grad_threshold` — loại bỏ Gaussian đã hội tụ/ổn định khỏi việc tính toán tốn kém.
2. **Lấy mẫu jitter ngẫu nhiên trong ellipse 1-sigma** của mỗi Gaussian còn lại (qua phân rã Cholesky của hiệp phương sai 2D `[cov_xx, cov_xy, cov_yy]`) thay vì chỉ lấy đúng tâm — tránh thiên lệch khi biên Gaussian rơi đúng vào cạnh texture.
3. **Tính hệ số vi phạm Nyquist $\eta$ theo 3 trục $(x,y,z)$ của Gaussian**, tuỳ `eta_compute_mode`:
   - `"wavelength"` (mặc định): quy đổi năng lượng tần số cao nhất của structure tensor tại điểm lấy mẫu (trị riêng lớn nhất $\lambda_1$ của $S$) thành **bước sóng texture cục bộ** $w_{\min}=1/(\sqrt{\lambda_1}+\varepsilon)$ (đơn vị pixel — cạnh/texture càng mảnh thì $w_{\min}$ càng nhỏ), rồi so với **độ dài hình chiếu của từng trục Gaussian trên màn hình** $\ell_k$ (từ ma trận hiệp phương sai 2D):
     $$\eta_k=\frac{\ell_k}{w_{\min}},\qquad k\in\{x,y,z\}.$$
     $\eta_k>1$ nghĩa là Gaussian **to hơn** đặc trưng texture nó cần tái hiện theo trục đó → alias (mờ/blocky) → cần split theo đúng trục đó.
   - `"projection"`: chiếu trực tiếp structure tensor $S$ lên từng hướng trục Gaussian bằng dạng toàn phương $\eta_k=\sqrt{S_{xx}u_k^2+2S_{xy}u_kv_k+S_{yy}v_k^2}$ với $(u_k,v_k)$ là hình chiếu 2D của trục $k$ — cho kết quả có hướng (định hướng cạnh khác nhau cho $\eta$ khác nhau theo từng trục), hữu ích khi texture có cấu trúc dị hướng mạnh (ví dụ vân gỗ, đường kẻ).
4. **Trọng số theo transmittance** (`eta_3ch *= weights_valid`) rồi cộng dồn vào các bộ đếm mức Gaussian: `accum_eta`, `accum_view_count` (tổng có trọng số / số lần quan sát), `max_eta_3ch` (giá trị $\eta$ lớn nhất từng thấy theo từng trục, dùng để định hình split — §29).
5. **Phân loại nhất quán đa góc nhìn** — đây là phần lõi thay thế hoàn toàn `compute_gaussian_score_fastgs`: lấy $\eta_{\max}=\max_k \eta_k$ của lượt quan sát này rồi phân vào ba lớp bằng hai ngưỡng cố định trong mã nguồn, $\tau_{\text{high}}=1.0$ và $\tau_{\text{low}}=0.1$:
   ```python
   is_high = eta_max_scalar > 1.0     # vi phạm Nyquist rõ rệt ở view này
   is_low  = eta_max_scalar <= 0.1    # quá mịn so với texture ở view này
   is_mid  = ~is_high & ~is_low
   gaussians.eta_high_count[...] += 1.0
   gaussians.eta_mid_count[...]  += 1.0
   gaussians.eta_low_count[...]  += 1.0
   ```
   Ba bộ đếm `eta_high_count`/`eta_mid_count`/`eta_low_count` cộng dồn **qua nhiều camera khác nhau, nhiều iteration khác nhau** (không reset mỗi iteration, chỉ reset khi densify chạy — §29) — đây chính là cơ chế "nhất quán đa góc nhìn" (multiview consistency) của SADGS, thay cho việc render 10 camera cùng lúc rồi cộng dồn tức thời như cách tiếp cận trước đây.

### Bằng chứng tích luỹ dần thay vì đo tức thời

| | Cách tiếp cận trước đây (`compute_gaussian_score_fastgs`, đã xoá) | SADGS (`update_freq_stats_online`) |
|---|---|---|
| Nguồn bằng chứng | Lỗi ảnh (photometric L1) so với ground-truth | Cấu trúc/texture ảnh gốc (structure tensor đa tỉ lệ), so với kích thước hình chiếu Gaussian |
| Khi nào đo | Render lại 10 camera **một lần** ngay trước mỗi lần densify | Tích luỹ mỗi 10 iteration trong suốt cửa sổ giữa hai lần densify, dùng đúng camera đang được bốc để train (không render thêm) |
| Đơn vị tích luỹ | Số pixel bị đánh dấu lỗi | Số **lượt quan sát** được phân loại high/mid/low theo $\eta$ |
| Tiêu chí "nhất quán" | `floor(tổng pixel lỗi / 10)` — trung bình cộng | `eta_high_count / accum_view_count` — **tỉ lệ phần trăm số lượt quan sát** đồng ý là vi phạm tần số |
| Ứng dụng | AND với gradient để chọn clone/split | OR với gradient để mở rộng `split_mask`/`prune_mask` (§29) |

**Vai trò của tỉ lệ (ratio) thay cho phép `floor`.** Một Gaussian chỉ bị đánh giá "vi phạm tần số nhất quán" khi *tỉ lệ* số lượt quan sát thuộc lớp `high` trên tổng số lượt quan sát (`high_ratio = eta_high_count/accum_view_count`) vượt `opt.split_ratio_threshold` (mặc định `0.8`). Nghĩa là: muốn được chọn split, Gaussian phải bị đánh giá "quá to so với texture" ở **ít nhất 80% số lần nó được quan sát**, không phải chỉ một vài view lẻ tẻ — cùng triết lý "multi-view consistent" như cách tiếp cận trước đây, chỉ khác nguồn bằng chứng (cấu trúc ảnh thay vì lỗi ảnh) và công thức gộp (tỉ lệ thay vì trung bình pixel).

### Ví dụ số minh hoạ tỉ lệ nhất quán đa góc nhìn (dùng $V=3$ cho gọn; ngưỡng thật `split_ratio_threshold=0.8`)

Giả sử Gaussian $G_A$ được quan sát ở 3 lượt (đủ điều kiện "hoạt động") với $\eta_{\max}$ đo được mỗi lượt là $1.4,\,1.2,\,1.6$ (đều $>\tau_{\text{high}}=1.0$) → `eta_high_count=3`, `accum_view_count=3` → `high_ratio=1.0>0.8` → **ứng viên split** (nếu đồng thời gradient tuyệt đối cũng vượt `grad_abs_thresh`, §29).

Gaussian $G_B$ có $\eta_{\max}$ đo được là $1.5,\,0.3,\,0.4$ — chỉ 1/3 lượt vượt $\tau_{\text{high}}$ → `high_ratio=0.33<0.8` → **không split**, dù lượt đầu vi phạm rất mạnh (1.5). Một Gaussian chỉ "trông sai" từ một góc nhìn (specular, phản chiếu, hoặc trục Gaussian đang xoay ngang với hướng nhìn đó) không nên bị tách chỉ vì một lượt quan sát ngẫu nhiên.

Gaussian $G_C$ có $\eta_{\max}$ dưới $\tau_{\text{low}}=0.1$ ở cả 3 lượt → `low_ratio=1.0>prune_ratio_threshold(0.8)` → **ứng viên prune** (đã nhỏ hơn nhiều so với mức texture cần tái hiện — thừa, không giúp thêm chi tiết).

> **`split_mask`/`prune_mask` được hợp nhất bằng `OR` với điều kiện gradient truyền thống, không thay thế nó.** `densify_and_prune_structgs` (§29) OR hai luồng bằng chứng (gradient tích luỹ liên tục kiểu 3DGS gốc, và tỉ lệ nhất quán đa góc nhìn theo cấu trúc-tần số của SADGS) trước khi AND với điều kiện kích thước (`split_qualifiers`/`clone_qualifiers`) — hai nguồn bằng chứng bổ sung cho nhau, một Gaussian đủ điều kiện split nếu **một trong hai** đồng ý, miễn kích thước của nó thuộc đúng nhóm.

## 29. `densify_and_prune_structgs` — hợp nhất gradient, kích thước, và nhất quán đa góc nhìn

Chữ ký thật (`scene/gaussian_model.py`):
```python
def densify_and_prune_structgs(self, max_screen_size, min_opacity, extent, radii, args,
                                importance_score=None, pruning_score=None,
                                custom_split_mask=None, custom_prune_mask=None,
                                viewspace_points_indices=None, max_eta_3ch=None):
```

**Bước A — tính hai luồng bằng chứng bên ngoài hàm, ngay trong `train.py`, trước khi gọi.** Khi `is_normal_densification` đúng (`iteration > densify_from_iter và iteration % densification_interval == 0`, trong cửa sổ `iteration < densify_until_iter`):

```python
grads = gaussians.xyz_gradient_accum / gaussians.denom;  grads[isnan] = 0.0
is_grad_high = torch.norm(grads, dim=-1) >= 1e-5                 # ngưỡng cứng, KHÔNG đọc args.grad_thresh

valid_mask = gaussians.accum_view_count > 0
high_ratio = eta_high_count / accum_view_count   (0 nếu invalid)
low_ratio  = eta_low_count  / accum_view_count   (0 nếu invalid)

split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high   # §28 — nhất quán tần số CAO, có gradient
prune_mask = (low_ratio  > opt.prune_ratio_threshold) & valid_mask     # §28 — nhất quán tần số THẤP

avg_high_eta_3ch[Gaussian có eta_high_count>0] = eta_high_sum_3ch / eta_high_count.unsqueeze(1)
max_high_eta = gaussians.max_eta_3ch   # giá trị eta lớn nhất từng thấy theo 3 trục, dùng định hình split
```
`split_mask`/`prune_mask` này được truyền vào `densify_and_prune_structgs` qua `custom_split_mask`/`custom_prune_mask`; `max_high_eta` qua `max_eta_3ch`; `importance_score=gaussians.accum_view_count` (số lượt Gaussian được quan sát/tích luỹ thống kê tần số trong cửa sổ densify vừa qua — **không phải** số pixel lỗi như cách tiếp cận trước đây).

**Bước B — bên trong hàm, hợp nhất với gradient/kích thước truyền thống:**
```python
grad_vars = xyz_gradient_accum / denom;      grad_vars[isnan] = 0
grads_abs = xyz_gradient_accum_abs / denom;  grads_abs[isnan] = 0
grad_qualifiers     = norm(grad_vars, dim=-1) >= args.grad_thresh       # 0.0002
grad_qualifiers_abs = norm(grads_abs, dim=-1) >= args.grad_abs_thresh   # 0.0002 (SADGS)

full_split_mask[...] = custom_split_mask    # đặt đúng theo viewspace_points_indices, hoặc thẳng nếu None
full_prune_mask[...] = custom_prune_mask

clone_qualifiers = max(get_scaling, dim=1) <= args.dense * extent
split_qualifiers = max(get_scaling, dim=1)  > args.dense * extent

final_split_mask = (full_split_mask OR grad_qualifiers_abs) AND split_qualifiers
final_clone_mask = (full_split_mask OR grad_qualifiers)     AND clone_qualifiers   # ⚠ dùng full_split_mask cho cả hai nhánh

metric_mask = importance_score > args.importance_score_threshold   # mặc định 0.5
densify_and_clone_structgs(metric_mask, final_clone_mask)
combined_split_mask = metric_mask AND final_split_mask
densify_and_split_structgs(combined_split_mask, max_eta_3ch=max_eta_3ch, scale_power=args.ks_scale_power)
```
⚠ Đọc đúng mã nguồn: nhánh clone dùng lại **`full_split_mask`** (bằng chứng nhất quán đa góc nhìn "tần số cao", không phải một mask riêng cho clone) OR với `grad_qualifiers` — tức trong SADGS hiện tại, bằng chứng cấu trúc-tần số chỉ được đưa vào qua đúng một kênh (`custom_split_mask`), dùng chung cho cả quyết định clone lẫn split, và chỉ khác nhau ở điều kiện kích thước AND sau đó (`clone_qualifiers` cho Gaussian nhỏ, `split_qualifiers` cho Gaussian lớn).

`extent = scene.cameras_extent` (bán kính cảnh, §14) nên `args.dense * extent` vẫn là ngưỡng kích thước tương đối theo scale thực của scene.

**Nhánh clone** (`densify_and_clone_structgs`) — sao chép y hệt mọi thuộc tính (bao gồm cả các bộ đếm mới: `densify_count` kế thừa nguyên giá trị cha, không tăng) sang bản mới, nối vào cuối qua `densification_postfix`, không dịch chuyển vị trí — giống hệt cơ chế clone của 3DGS gốc.

**Nhánh split — `densify_and_split_structgs`: split dị hướng phân tích (analytic anisotropic split), khác hẳn "chia đôi ngẫu nhiên N=2" của 3DGS gốc.** Với mỗi Gaussian được chọn:

1. Nếu có `max_eta_3ch` (đường densify chính luôn có): số Gaussian con trên mỗi trục $k\in\{x,y,z\}$ được tính **riêng biệt theo lý thuyết lấy mẫu (Nyquist)**:
   $$k_{\text{axis}}=\Big\lceil \sqrt{\eta_{\text{axis}}}\Big\rceil,\qquad \eta_{\text{axis}}=\max\text{-eta từng thấy trên trục đó (đã tích luỹ qua nhiều view, §28)},$$
   kẹp `min=1` (không giới hạn trên trong code hiện tại, dù `max_clones_per_axis` tồn tại như tham số dự phòng). Tổng số Gaussian con của một Gaussian cha là $N=k_x\cdot k_y\cdot k_z$ — có thể lớn hơn 2 rất nhiều nếu cha vi phạm Nyquist nặng trên nhiều trục cùng lúc (ví dụ $k_x=k_y=k_z=2\Rightarrow N=8$).
   Nếu không có `max_eta_3ch` (đường dự phòng `warmup_densification`): $k=2$ trên đúng trục có scale lớn nhất, các trục còn lại $k=1$ — đây mới là hành vi giống 3DGS gốc ("chia đôi Gaussian dài nhất").
2. Scale con: $\text{scale}_{\text{con}}=\text{scale}_{\text{cha}}/k^{\,\texttt{ks\_scale\_power}}$ theo **từng trục riêng** (mặc định `ks_scale_power=1.0`, tức chia tuyến tính; SADGS không dùng công thức `0.8*N` cố định của 3DGS gốc nữa).
3. Vị trí con: dựng lưới toạ độ $(i_x,i_y,i_z)$ đã căn giữa quanh $0$ cho từng trục, nhân với độ giãn cách $\text{stds}_{\text{con}}\cdot\sqrt{12}$ rồi xoay theo `rotation` của cha và cộng vào tâm cha — một lưới đều đặn theo hình hộp chữ nhật $k_x\times k_y\times k_z$ trong hệ toạ độ cục bộ của Gaussian, khác hẳn cách lấy mẫu ngẫu nhiên Gaussian (`torch.normal`) của 3DGS gốc.
4. Gaussian cha luôn bị xoá sau khi tạo đủ $N$ con (`prune_points`), giống 3DGS gốc — split luôn "thay thế", không "giữ lại cha".

**Prune trong cùng lượt gọi:**
```python
prune_mask = (opacity < min_opacity)
if max_screen_size:
    prune_mask |= (max_radii2D > max_screen_size) | (scale.max(dim=1) > 0.1*extent)
prune_mask |= full_prune_mask    # bằng chứng nhất quán đa góc nhìn "tần số thấp" (§28)

if pruning_score is not None:
    ... lấy mẫu ngân sách 50% theo trọng số 1/(1-pruning_score) (§30) ...
else:
    self.prune_points(prune_mask)   # đường thật: train.py gọi với pruning_score=None
```
Đường gọi thật trong `train.py` luôn truyền `pruning_score=None`, nên **nhánh lấy-mẫu-ngân-sách không chạy trong cấu hình mặc định** — `prune_points(prune_mask)` xoá thẳng toàn bộ Gaussian bị đánh dấu, không giới hạn 50%.

Cuối hàm, opacity được **kẹp trần ở 0.8** (không phải reset opacity đầy đủ): `opacities_new = inverse_sigmoid(min(get_opacity, 0.8))`, ghi lại qua `replace_tensor_to_optimizer` — khác với `reset_opacity()` (§32, kẹp về `opacity_reset_decay`, gọi riêng theo chu kỳ `opacity_reset_interval`).

**Reset bộ đếm sau mỗi lần densify** (`train.py`, ngay sau lời gọi): `accum_eta`, `accum_view_count`, `max_eta_3ch`, `accum_weights_valid`, `eta_high_count`, `eta_high_sum_3ch`, `eta_mid_count`, `eta_mid_sum_3ch`, `eta_low_count` đều bị `zero_()` — cửa sổ tích luỹ nhất quán đa góc nhìn (§28) bắt đầu lại từ 0 cho chu kỳ densify tiếp theo. `xyz_gradient_accum`/`xyz_gradient_accum_abs`/`denom`/`max_radii2D` cũng bị reset (bên trong `densification_postfix`/`prune_points`, như 3DGS gốc).

**Lịch gọi.** `min_opacity=0.1` (không phải `0.005` như giá trị mặc định trước đây — tham số truyền cứng trong `train.py`, có comment "0.005 is a good default" nhưng giá trị thật dùng là `0.1`); `max_screen_size = 20` khi `iteration > opt.opacity_reset_interval` (mặc định `3000`, hằng số `OptimizationParams`), ngược lại `None`. Nhánh dự phòng `opt.warmup_densification` (mặc định `False`) gọi `densify_and_prune` (hàm 3DGS thuần, không dùng structure tensor) mỗi 100 iteration khi **không** trùng chu kỳ densify chính — nhánh này tồn tại trong code nhưng tắt theo mặc định.

## 30. Ba lớp prune trong SADGS: trong-vòng-densify, lịch cứng theo iteration, và `final_prune_structgs` (chưa được gọi mặc định)

| Lớp | Khi nào chạy | Điều kiện xoá | Nguồn |
|---|---|---|---|
| 1. Prune trong mỗi lần densify | Mỗi `densification_interval` iteration, trong `(densify_from_iter, densify_until_iter)` | `opacity < 0.1` HOẶC (nếu `max_screen_size`) `max_radii2D > 20` HOẶC `scale.max > 0.1·extent` HOẶC `low_ratio > prune_ratio_threshold` (nhất quán đa góc nhìn "tần số thấp", §28-29) | `densify_and_prune_structgs`, cuối hàm |
| 2. Prune cứng theo lịch iteration | Tại các mốc trong `--prune_iterations` (mặc định `[4000, 8000]`, độc lập với `densification_interval`) | `opacity < 0.1` | `train.py`: `prune_mask = (gaussians.get_opacity < 0.1).squeeze(); gaussians.prune_points(prune_mask)` |
| 3. `final_prune_structgs` (hậu kỳ, dựa trên `pruning_score` đa góc nhìn) | Có sẵn trong `GaussianModel`, **nhưng lời gọi tại các mốc `prune_iterations` hiện bị comment trong `train.py`** | `opacity < min_opacity` HOẶC `pruning_score > 0.9` | `scene/gaussian_model.py::final_prune_structgs` |

So với việc chạy theo mặc định trong một cách tiếp cận trước đây: SADGS **không còn** cơ chế lấy-mẫu-ngân-sách 50% (`torch.multinomial`) chạy theo mặc định — nhánh đó vẫn tồn tại nguyên vẹn trong `densify_and_prune_structgs` (kích hoạt khi truyền `pruning_score` khác `None`) nhưng đường gọi thật trong `train.py` luôn truyền `pruning_score=None`, nên lớp 1 xoá thẳng toàn bộ Gaussian bị đánh dấu. Lớp 3 (`final_prune_structgs`) là hàm đã viết sẵn — dùng `pruning_score` (điểm nhất quán đa góc nhìn dạng liên tục, không phải `high_ratio`/`low_ratio` nhị phân) để dọn "những gì đã thử densify mà không giúp được" — nhưng lời gọi nó tại các mốc `prune_iterations` đang bị **comment** trong `train.py` hiện tại (thay vào đó chỉ còn một prune thô theo opacity, lớp 2 ở trên). Đây là điểm cần lưu ý khi đọc code: tài liệu paper mô tả cơ chế prune theo "multi-view reconstruction consistency" đầy đủ, nhưng bản `train.py` đang chạy chỉ kích hoạt một phần của nó (lớp 1, qua `low_ratio`) — lớp 3 nằm sẵn trong `GaussianModel` chờ được bật lại.

## 31. 🔒 Phẫu thuật trạng thái Adam

Vì số Gaussian thay đổi theo từng lần densify/prune, không thể để nguyên các tensor tham số của Adam (`nn.Parameter` có kích thước cố định) — mọi thao tác thêm/bớt điểm đều phải đồng bộ ba thứ cùng lúc: tensor tham số, `optimizer.state[param]["exp_avg"]`, `optimizer.state[param]["exp_avg_sq"]`.

**Xoá điểm — `_prune_optimizer(mask)`** (`scene/gaussian_model.py:307-327`), với `mask` ở đây là **valid mask** (`~mask` xoá, gọi từ `prune_points`, dòng 364-366):
```python
for opt in [self.optimizer, self.shoptimizer nếu có]:
    for group in opt.param_groups:
        stored_state = opt.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"]    = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
            del opt.state[group['params'][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            opt.state[group['params'][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
```
Chìa khoá: `nn.Parameter` mới được tạo ra (không sửa in-place tensor cũ), nên khoá cũ trong `opt.state` (dict khoá theo `id(tensor)`) sẽ trỏ tới tham số đã bị thay — bắt buộc phải `del opt.state[group['params'][0]]` (khoá **cũ**, đọc trước khi gán) rồi gán lại `opt.state[group['params'][0]]` (khoá **mới**) = `stored_state` đã được index theo cùng `mask`. Nếu quên bước `del`/gán lại này, Adam sẽ dùng state cũ (kích thước không khớp tensor mới) → lỗi shape, hoặc tệ hơn là state "ma" không bao giờ bị garbage-collect (rò rỉ bộ nhớ GPU) vì `opt.state` vẫn giữ tham chiếu tới `nn.Parameter` cũ. Kết quả: các điểm **giữ lại** mang đúng `exp_avg`/`exp_avg_sq` cũ của chúng (đã tích luỹ qua các bước Adam trước đó, chỉ lọc theo mask), các điểm bị xoá mất theo.

**Thêm điểm (clone/split) — `cat_tensors_to_optimizer(tensors_dict)`**, gọi từ `densification_postfix`:
```python
stored_state["exp_avg"]    = cat([stored_state["exp_avg"],    zeros_like(extension_tensor)], dim=0)
stored_state["exp_avg_sq"] = cat([stored_state["exp_avg_sq"], zeros_like(extension_tensor)], dim=0)
del opt.state[group['params'][0]]
group["params"][0] = nn.Parameter(cat([group["params"][0], extension_tensor], dim=0).requires_grad_(True))
opt.state[group['params'][0]] = stored_state
```
Cả điểm **clone** lẫn điểm **split** đều nhận `exp_avg`/`exp_avg_sq` **khởi tạo lại bằng 0** — không kế thừa moment bậc 1/2 từ Gaussian cha. Cùng lúc, `densification_postfix` cũng nối thêm phần tử `0` cho **toàn bộ** bộ đếm phụ trợ mới của SADGS: `accum_eta`, `accum_view_count`, `max_eta_3ch`, `accum_weights_valid`, `eta_high_count`, `eta_high_sum_3ch`, `eta_mid_count`, `eta_mid_sum_3ch`, `eta_low_count` — Gaussian mới sinh luôn bắt đầu ở trạng thái "chưa được quan sát lần nào" đối với nhất quán đa góc nhìn, đúng như `xyz_gradient_accum`/`denom`/`max_radii2D`. Riêng `densify_count` (đếm số lần một Gaussian từng bị split, dùng để giới hạn over-densification — hiện đang bị comment tắt trong `update_freq_stats_online`, §28) là ngoại lệ: hàm gọi `densification_postfix` xong sẽ **ghi đè** đoạn `0` vừa nối bằng giá trị kế thừa thật (Gaussian split nhận `densify_count` cha `+1`; Gaussian clone nhận nguyên `densify_count` cha, không tăng).

**Điều gì hỏng nếu làm sai:** nếu index-theo-mask không nhất quán giữa param và state (ví dụ prune tensor tham số nhưng quên prune state, hoặc dùng nhầm `mask` thay vì `~mask`), Adam sẽ áp `exp_avg` của điểm A lên điểm B → gradient/momentum bị gán nhầm chủ. Việc `del` + gán lại key theo tensor mới (thay vì sửa item tại chỗ) là bắt buộc vì `torch.optim.Optimizer.state` là `defaultdict` khoá theo **object identity** (`id()`) của `nn.Parameter`, và `group["params"][0] = nn.Parameter(...)` tạo object mới mỗi lần.

**Hai optimizer, ba chế độ chọn qua `opt.optimizer_type`.** `training_setup` dựng `self.optimizer`/`self.shoptimizer` khác nhau tuỳ `optimizer_type` (`adam_eps = 10^-adam_eps_order`, mặc định `adam_eps_order=8`):

| `optimizer_type` | `self.optimizer` (nhóm `xyz, f_dc, opacity, scaling, rotation`) | `self.shoptimizer` (nhóm `f_rest`) |
|---|---|---|
| `"default"` | `torch.optim.Adam(l, betas=(0.9,0.999))` | `torch.optim.Adam(sh_l, betas=(0.9,0.999))` |
| `"sparse_adam"` | `SparseGaussianAdam(l + sh_l, ...)` (gộp cả hai nhóm) | không dùng riêng |
| `"hybrid"` **(mặc định `OptimizationParams.optimizer_type = "hybrid"`)** | `torch.optim.Adam(l, betas=(0.9,0.999), amsgrad=True)` | `SparseGaussianAdam(sh_l, ...)` |

Khác với một fork trước đây (nơi `sparse_adam` là một nhánh "chết" — import từ package vanilla `diff_gaussian_rasterization` không tồn tại), `SparseGaussianAdam` trong SADGS được import từ đúng extension CUDA của repo (`from diff_gaussian_rasterization_structgs import SparseGaussianAdam`) — **nó thật sự hoạt động**: `step(visible_mask, N)` chỉ cập nhật tham số cho Gaussian nằm trong `visible_mask` (thường là `radii > 0`), bỏ qua hoàn toàn phần compute cho Gaussian đang không được camera hiện tại nhìn thấy — đây là cách SADGS đạt hiệu quả tương tự "sparse Adam" của Taming-3DGS bằng một kernel CUDA riêng, thay vì bằng lịch step thưa dần theo iteration.

## 32. `reset_opacity`

```python
def reset_opacity(self, decay=0.01):
    opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*decay))
    optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    self._opacity = optimizable_tensors["opacity"]
```
Khác với 3DGS gốc (trần cố định `0.01`), SADGS truyền `decay` như một **tham số** — `opt.opacity_reset_decay`, mặc định `0.1` (`arguments/__init__.py`). Với mỗi Gaussian, opacity sau activation bị **kẹp trần ở `opacity_reset_decay`**: Gaussian nào đang có opacity ≤ ngưỡng đó giữ nguyên, còn lại bị ép xuống — chỉ *hạ trần*, không đặt cứng toàn bộ về cùng một giá trị.

`replace_tensor_to_optimizer` thực hiện đúng kiểu phẫu thuật Adam như §31 nhưng cho trường hợp "thay tensor mà không đổi số lượng điểm": state Adam của riêng nhóm `"opacity"` bị **reset về 0**, các nhóm tham số khác không bị đụng tới.

**Lịch gọi** (`train.py`):
```python
if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
    gaussians.reset_opacity(opt.opacity_reset_decay)
```
`opacity_reset_interval = 3000` (default `OptimizationParams`, hằng số cố định — không có lớp `pipeline/`/`Config` nào co giãn nó theo `%` tổng số iteration trong SADGS thuần). Trường hợp đặc biệt nền trắng: nếu `dataset.white_background` bật, còn có một lần reset sớm đúng tại `iteration == densify_from_iter` (500).

**Lý do cần bước này:** kỹ thuật chuẩn của 3DGS — Gaussian tích luỹ opacity cao chỉ vì "ăn theo" nền hoặc bị Gaussian khác che khuất phần render sai của chúng sẽ không bao giờ chạm điều kiện `opacity < min_opacity` (§30). Định kỳ ép trần opacity buộc mọi Gaussian "chứng minh lại" độ cần thiết — Gaussian không đóng góp thật sẽ tụt trở lại dưới `min_opacity` và bị lớp pruning kế tiếp dọn đi.

## 33. Áp dụng bước Adam và lưu `.ply`

**Không còn lịch "thưa dần theo iteration" (16/32/64) của cách tiếp cận trước đây.** Hàm `optimizer_step(iteration)` (kiểu lịch cũ) vẫn tồn tại trong `GaussianModel` nhưng **chỉ được gọi khi `opt.optimizer_type == "default"`** — không phải cấu hình mặc định. Với mặc định `optimizer_type = "hybrid"` (§31), vòng lặp chính trong `train.py` áp dụng Adam **mỗi iteration**, không thưa dần:
```python
elif opt.optimizer_type == "hybrid":
    visible = radii > 0
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    gaussians.shoptimizer.step(visible, radii.shape[0])
    gaussians.shoptimizer.zero_grad(set_to_none=True)
```
`self.optimizer` (Adam thường, nhóm `xyz/f_dc/opacity/scaling/rotation`) cập nhật toàn bộ Gaussian mỗi iteration; `self.shoptimizer` (`SparseGaussianAdam`, nhóm `f_rest`) chỉ cập nhật đúng những Gaussian có `radii > 0` ở lượt render vừa rồi — tiết kiệm compute cho SH bậc cao mà **không** cần giảm tần suất step theo lịch cố định như cách tiếp cận trước đây. Đây là điểm khác biệt cốt lõi: cách tiếp cận trước đây đánh đổi *tần suất* để tiết kiệm; SADGS đánh đổi *phạm vi* (chỉ Gaussian visible) mỗi iteration.

**Lịch learning rate ghim theo `position_lr_max_steps`.** `training_setup`:
```python
self.xyz_scheduler_args = get_expon_lr_func(
    lr_init=training_args.position_lr_init * spatial_lr_scale,
    lr_final=training_args.position_lr_final * spatial_lr_scale,
    max_steps=training_args.position_lr_max_steps)
```
`position_lr_max_steps` mặc định **30 000** (`arguments/__init__.py:78`). Trường này không tự đọc `--iterations`; đường notebook bù lại bằng cách cho `build_args` truyền `--position_lr_max_steps` bằng đúng số vòng train, còn đường CLI thuần thì người dùng phải tự truyền. `update_learning_rate(iteration)` (dòng 182-188) chỉ chỉnh lr của nhóm `"xyz"` theo hàm suy giảm mũ này, tính theo `iteration` tuyệt đối truyền vào — **không** tính theo tỉ lệ % tiến trình so với tổng số iteration thực tế của lần train đó.

Hệ quả khi train ít hơn 30 000 iteration (ví dụ mặc định notebook `iterations=7000`, `pipeline/config.py:31`): lr của `xyz` mới suy giảm được một đoạn nhỏ đầu của lịch trình mũ 30k bước, dừng lại ở một giá trị còn khá cao so với `position_lr_final` — nó **không bao giờ đi hết lịch trình** để chạm tới `lr_final = 0.0000016`. Đường notebook đã đóng khoảng cách này: `pipeline/trainer.py::build_args` truyền `--position_lr_max_steps` bằng đúng số vòng train, nên lịch mũ co lại vừa khít và lr vẫn chạm `lr_final` khi train kết thúc. Đường CLI thuần (`train.py`, `train_base.sh`, `train_big.sh`) **không** làm việc đó — chạy ngắn hơn 30k mà không tự truyền cờ này thì vẫn kết thúc với lr `xyz` cao hơn thiết kế gốc.

**Lưu `.ply` — hai cơ chế khác nhau ở hai vòng lặp:**
- `train.py`: `--save_iterations` (mặc định `[30_000]`, dòng 255) cộng thêm `args.iterations` vào cuối danh sách (dòng 262: `args.save_iterations.append(args.iterations)`), rồi mỗi khi `iteration in saving_iterations` gọi `scene.save(iteration)` (dòng 120-122). Không có xoá checkpoint cũ — mỗi mốc lưu là một thư mục `point_cloud/iteration_<n>/` riêng, giữ lại toàn bộ.
- `pipeline/trainer.py`: `_save_checkpoint(scene_obj, iteration, previous, drop_previous)` (dòng 39-47) gọi `scene_obj.save(iteration)` rồi, nếu `drop_previous=True` và có `previous`, xoá thư mục `point_cloud/iteration_<previous>/` bằng `shutil.rmtree`. Vòng lặp gọi hàm này mỗi `cfg.save_every` iteration (mặc định **2000**, `pipeline/config.py:34`) khi `iteration < iterations` (dòng 186-189), truyền `cfg.keep_last_checkpoint` (mặc định **True**, dòng 35) làm `drop_previous` — nghĩa là mặc định chỉ giữ **một** checkpoint trung gian tại một thời điểm (checkpoint mới ghi đè bằng cách xoá cái cũ ngay sau khi ghi cái mới), để tiết kiệm dung lượng đĩa Colab. Sau khi vòng lặp kết thúc, còn một lần lưu cuối cùng bắt buộc: `_save_checkpoint(scene_obj, iterations, saved_at, cfg.keep_last_checkpoint)` (dòng 194) — đảm bảo luôn có checkpoint tại đúng iteration cuối cùng dù `save_every` có chia hết cho `iterations` hay không.

**`.ply` không phải checkpoint có thể resume.** `save_ply` (dòng 260-277) chỉ ghi `xyz, f_dc, f_rest, opacity, scale, rotation` dưới dạng thuộc tính PLY thuần — không có `optimizer.state_dict()`, không có `xyz_gradient_accum`/`xyz_gradient_accum_abs`/`denom`/`max_radii2D`, không có `active_sh_degree` hay `spatial_lr_scale` (những thứ mà `capture()`/`restore()` mới lưu đủ, dòng 73-131, dùng cho checkpoint `.pth`). `train.py` vẫn có **nửa** cơ chế này: nếu truyền `--start_checkpoint`, nó `torch.load(checkpoint)` rồi `gaussians.restore(model_params, opt)` (dòng 43-45) để khôi phục đầy đủ trạng thái Adam/densify từ một file `.pth`. Nhưng **nửa còn lại — ghi file đó — không tồn tại**: tham số `--checkpoint_iterations` được khai báo (`train.py:252`, mặc định `[30_000]`) và truyền vào `training(...)` nhưng không hề được dùng bên trong hàm để gọi `torch.save((gaussians.capture(...), iteration), ...)` ở bất kỳ đâu — đây là phần vestigial còn sót lại từ mã 3DGS gốc, không hoạt động trong repo hiện tại. `pipeline/trainer.py` không có cơ chế `--start_checkpoint`/`.pth` nào cả, chỉ dùng `.ply` qua `_save_checkpoint`. Kết luận chung: trong pipeline hiện tại (cả `train.py` chạy mặc định lẫn `pipeline/trainer.py`), việc dừng giữa chừng rồi "tiếp tục" chỉ có thể khôi phục lại đám mây điểm từ `.ply`, không khôi phục được trạng thái Adam hay các bộ đếm densify — train tiếp từ một `.ply` tương đương khởi động lại optimizer/densify từ đầu trên một point cloud đã qua huấn luyện, chứ không phải "tiếp tục đúng như đang dở".

## Bài tập (Exercise)

**Bài tập 3.1.** Bảng ở §19 đối chiếu từng bước của `train.py::training()` với `pipeline/trainer.py::train_scene()`. Hãy liệt kê ba thứ mà `train_scene` **bỏ** so với `train.py` (websocket viewer, TensorBoard, `--debug_from`) và với mỗi thứ, giải thích bằng một câu tại sao việc bỏ nó hợp lý trong bối cảnh chạy trên notebook Colab thay vì CLI cục bộ.

**Bài tập 3.2.** Đọc lại đoạn code `update_learning_rate` ở §20:
```python
for param_group in self.optimizer.param_groups:
    if param_group["name"] == "xyz":
        lr = self.xyz_scheduler_args(iteration)
        param_group['lr'] = lr
        return lr
```
Giải thích vì sao hàm `return lr` ngay trong vòng `for` không cần `break`, và vì sao giá trị trả về không được `train.py`/`train_scene` sử dụng. Nhóm tham số nào (trong 5 nhóm ở bảng §20) *không* được hàm này cập nhật, và learning rate của chúng thay đổi thế nào trong suốt quá trình train?

**Bài tập 3.3.** Tại §22, kernel CUDA tính ngưỡng cắt hộp bao theo công thức $t = 2\log(\text{opacity}\times 255)$, sau đó $t = \text{mult}\times t$. Với một Gaussian có `opacity = 0.5` (sau sigmoid), tính $t$ trước khi nhân `mult`, rồi tính $t$ sau khi nhân với hai giá trị `mult` khác nhau: `mult = 0.5` (giá trị mặc định trước đây) và `mult = 0.7` (mặc định `opt.mult` của SADGS). `mult` nào cho hộp bao hẹp hơn, và hệ quả về tốc độ/độ chính xác rasterize là gì? Vì sao việc SADGS tăng mặc định lên `0.7` hợp lý khi rasterizer giờ còn phải tính thêm `cov2D` chính xác cho mỗi Gaussian (§23, dùng cho `update_freq_stats_online`)?

**Bài tập 3.4.** So sánh hai cách "mượn gradient" được mô tả ở §26: tại sao `screenspace_points` phải được khởi tạo bằng 0 và gọi `retain_grad()`, thay vì lấy trực tiếp `pc.get_xyz.grad`? Trong câu trả lời, hãy chỉ rõ `means3D` và `means2D` đóng vai trò gì khác nhau trong lời gọi rasterizer.

**Bài tập 3.5.** Dùng đúng ví dụ số ở §28 (rút gọn $V=3$ lượt quan sát, ba Gaussian $G_A$/$G_B$/$G_C$ với chuỗi $\eta_{\max}$ đo được lần lượt là $(1.4,1.2,1.6)$, $(1.5,0.3,0.4)$, và luôn dưới $0.1$):
- (a) Tính lại `eta_high_count`, `eta_low_count`, `accum_view_count`, `high_ratio` và `low_ratio` cho cả ba Gaussian.
- (b) Giải thích bằng lời tại sao $G_B$ (có một lượt vi phạm rất mạnh, $\eta_{\max}=1.5$) vẫn **không** được chọn split, trong khi $G_A$ (không lượt nào vượt quá $1.6$, thấp hơn đỉnh của $G_B$) lại được chọn.
- (c) Với `split_ratio_threshold=0.8` và `prune_ratio_threshold=0.8`, Gaussian nào trong ba Gaussian trên rơi vào `split_mask`, Gaussian nào rơi vào `prune_mask`?

**Bài tập 3.6.** So sánh công thức gộp bằng chứng đa góc nhìn của cách tiếp cận trước đây (`importance = floor(tổng pixel lỗi / V)`, trung bình cộng rồi làm tròn) với công thức của SADGS (`high_ratio = eta_high_count / accum_view_count`, tỉ lệ phần trăm). Cả hai đều nhằm lọc "nhất quán qua nhiều góc nhìn" — hãy chỉ ra một tình huống cụ thể mà hai công thức cho **kết quả khác nhau** về việc một Gaussian có được densify hay không (gợi ý: xét trường hợp một Gaussian chỉ được quan sát ở rất ít lượt, ví dụ `accum_view_count=1`, so với một Gaussian được quan sát ở rất nhiều lượt).

**Bài tập 3.7.** §31 mô tả "phẫu thuật" trạng thái Adam khi số Gaussian thay đổi: sau khi tạo `nn.Parameter` mới, code phải `del opt.state[group['params'][0]]` (khoá cũ) rồi gán lại `opt.state[group['params'][0]] = stored_state` (khoá mới). Giải thích bằng cơ chế `id()`/object identity của `torch.optim.Optimizer.state` tại sao bỏ qua bước `del`/gán lại này sẽ gây lỗi shape hoặc rò rỉ bộ nhớ GPU. Việc này khác gì giữa trường hợp **xoá điểm** (`_prune_optimizer`) và trường hợp **thêm điểm** (`cat_tensors_to_optimizer`) về giá trị khởi tạo của `exp_avg`/`exp_avg_sq`?

---

## Ghi chú: Morton reordering không có trong SADGS

Một bản trước của tài liệu này (viết cho một fork khác, tích hợp thêm các cơ chế lấy ý tưởng từ Faster-GS) mô tả một cơ chế "Morton/Z-order reordering định kỳ" (`morton_reorder_interval`, hàm `apply_morton_ordering`/`_reorder_optimizer`) nhằm cải thiện tính cục bộ bộ nhớ khi rasterize. Đã kiểm tra lại: **`SADGS/scene/gaussian_model.py` và `SADGS/arguments/__init__.py` hiện tại không có bất kỳ hàm hay tham số nào liên quan đến Morton/Z-order reordering.** Đây không phải một phần của mã nguồn SADGS (repo gốc của Lyu et al.) — mục này được loại bỏ khỏi tài liệu để tránh mô tả sai một cơ chế không tồn tại. Nếu một fork cụ thể nào đó của SADGS có tích hợp lại cơ chế này như một lớp tối ưu bộ nhớ độc lập với densify, nó cần được tài liệu hoá riêng dựa trên mã nguồn thật của fork đó, không suy diễn từ tài liệu của các fork trước đây.

---

[← Chương 2](02-kien-truc-pipeline-phan-1.md) | [Mục lục](00-muc-luc.md) | [Chương 4 →](04-luu-render-cham-diem-phan-3.md)
