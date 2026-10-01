# Công thức toán học trong `arguments/__init__.py`

## Nhận định chung

`arguments/__init__.py` không chứa phép tính toán học nào trực tiếp (ngoài `extract()` xử lý Namespace và `get_combined_args` merge config). Giá trị của file nằm ở việc **định nghĩa các siêu tham số (hyperparameter)** — mỗi tham số đóng vai trò một hằng số/ngưỡng/hệ số trong các công thức đã nêu ở `gaussian_model.md` và `train.md`. Tài liệu này liệt kê từng nhóm tham số và chỉ rõ công thức mà nó tham gia, không chép lại toàn bộ công thức.

---

## 0. Trích dẫn nguyên văn ba nhóm tham số (nguồn của mọi hằng số dùng bên dưới)

### `ModelParams` (`arguments/__init__.py`, dòng 47–62)

```python
class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g
```

### `PipelineParams` (dòng 64–71)

```python
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")
```

### `OptimizationParams` (dòng 73–160)

```python
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.opacity_lr = 0.05 # 0.025 
        self.scaling_lr = 0.01 # 0.005
        self.rotation_lr = 0.002 # 0.001
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016 #0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        
        self.opacity_reset_interval = 3000
        self.opacity_reset_decay = 0.1
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.densify_grad_abs_threshold = 0.0004

        self.prune_until_iter = 25000
        self.min_weight = 0.7

        self.prune_from_iter = 6000
        self.prune_until_iter = 30_000
        self.prune_interval = 3000
        self.densify_prune_ratio = 0.45
        self.after_densify_prune_ratio = 0.01
        
        

        # fastgs parameters
        self.loss_thresh = 0.02
        self.grad_abs_thresh = 0.0002  
        self.highfeature_lr = 0.005 # 0.005
        self.lowfeature_lr = 0.0025 # 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.7      # multiplier for the compact box to control the tile number of each splat

        # frequency parameters
        self.lambda_l2 = 2.0
        self.lambda_tone=0.
        self.lambda_freq = 0. # Frequency-based loss weight
        self.st_levels = 4
        self.st_mode = "v1" # "v1" or "v2"
        
        self.freq_grad_threshold = 0.00002
        self.importance_score_threshold = 0.5
        self.min_contribution_threshold = 0.1
        self.importance_error_threshold = 0.06
        self.random_background = False
        self.optimizer_type = "hybrid"
        self.sample_bbox_faces = False
        self.warmup_densification = False
        self.camera_sampling = "random"
        self.compute_3d_filter = False
        
        # expansion parameters
        self.tau_expand = 1.0
        self.adaptive_clone = False # [NEW] Enable frequency-aware scale expansion before cloning
        self.expansion_speed = 0.1
        self.ks_scale_power = 1.0  # Exponent for scale division when splitting (new_scale = old_scale / k^scale_power)
        
        # initialization parameters
        self.sample_far_plane = False
        self.far_plane_dist = 10.0
        self.far_plane_res = 32
        self.densification_window_width = 200  # Number of iterations to run densification per window
        self.freq_opacity_threshold = 0.05
        self.freq_transmittance_threshold = 0.0
        self.batch_size = 1  # Number of cameras to accumulate gradients from before optimizer step
        self.split_ratio_threshold = 0.8
        self.prune_ratio_threshold = 0.8
        self.eta_compute_mode = "wavelength" # "wavelength" or "projection"
        
        # Multi-clone parameters
        self.clone_target_eta = 1.0  # Target eta for low-frequency areas (lower = more clones)
        self.scale_rotation_scheduler = False
        self.max_clones_per_axis = 8 # Maximum number of clones per axis
        self.adam_eps_order = 8



        super().__init__(parser, "Optimization Parameters")
```

**Phát hiện khi đối chiếu với code thật:** `self.prune_until_iter` được gán **hai lần** — lần đầu $=25000$ (dòng 96, đi cùng `self.min_weight = 0.7` ở dòng 97, nhóm tham số này không thấy được dùng trực tiếp trong `train.py`/`gaussian_model.py` đã đọc ở các file MATH khác), rồi bị **ghi đè** thành $=30000$ ở dòng 100 ngay bên dưới `self.prune_from_iter = 6000` (dòng 99). Giá trị hiệu lực duy nhất khi chương trình chạy là $\texttt{prune\_until\_iter}=30000$ (dòng 100); giá trị $25000$ ở dòng 96 là **dead code** (gán rồi bị ghi đè ngay, không ảnh hưởng hành vi). Đây là một điểm cần lưu ý khi đọc code, không phải lỗi toán học nhưng dễ gây hiểu nhầm nếu chỉ đọc dòng 96.

---

## 1. `ModelParams` — tham số mô hình/scene

| Tham số | Giá trị mặc định | Vai trò toán học |
|---|---|---|
| `sh_degree` | 3 | Bậc cực đại $\ell_{max}$ của Spherical Harmonics — xác định chiều $\mathbb{R}^{3\times(\ell_{max}+1)^2}$ của hệ số màu $c_i$ (xem `gaussian_model.md` §1) |
| `source_path`, `model_path`, `images`, `data_device` | đường dẫn/chuỗi | không mang ý nghĩa toán học |
| `resolution` | -1 (auto) | hệ số scale ảnh đầu vào, ảnh hưởng gián tiếp đến phép chiếu camera $u,v$ trong `compute_3D_filter` (`gaussian_model.md` §2.1) |
| `white_background` | False | chọn màu nền $(1,1,1)$ hay $(0,0,0)$ dùng trong alpha blending của renderer, và điều kiện reset opacity đặc biệt trong `train.py` (`iteration == densify_from_iter`) |
| `eval` | False | chia tập train/test, không ảnh hưởng công thức |

## 2. `PipelineParams` — tham số pipeline render

| Tham số | Vai trò toán học |
|---|---|
| `separate_sh` | chọn đường tính riêng SH bậc 0 (DC) và bậc cao trong rasterizer |
| `convert_SHs_python`, `compute_cov3D_python` | chọn cài đặt Python hay CUDA cho việc tính màu từ SH ($c = \sum_\ell Y_\ell(\mathbf{d})\,k_\ell$) và ma trận hiệp phương sai $\Sigma_i = R_iS_iS_i^\top R_i^\top$ (`gaussian_model.md` §1.3) — không đổi công thức, chỉ đổi nơi tính |
| `antialiasing` | bật/tắt hiệu chỉnh kiểu Mip-Splatting liên quan đến bộ lọc 3D (`compute_3D_filter`, `gaussian_model.md` §2) |
| `debug` | cờ gỡ lỗi, không toán học |

## 3. `OptimizationParams` — siêu tham số huấn luyện/densify

### 3.1. Learning rate (lịch mũ `get_expon_lr_func`, `gaussian_model.md` §4)

| Tham số | Vai trò |
|---|---|
| `position_lr_init`, `position_lr_final`, `position_lr_delay_mult`, `position_lr_max_steps` | tham số $\mathrm{lr}_{init}, \mathrm{lr}_{final}, \delta_{mult}, t_{delay}$ trong công thức $\mathrm{lr}(t)=\delta(t)\exp(\log(\mathrm{lr}_{init})(1-t)+\log(\mathrm{lr}_{final})t)$ áp dụng cho `xyz` |
| `scaling_lr`, `rotation_lr` | learning rate cơ sở $\mathrm{lr}_0$ cho `scaling`/`rotation`; khi `scale_rotation_scheduler=True` được nhân thành $\mathrm{lr}_{init}=3\,\mathrm{lr}_0$, $\mathrm{lr}_{final}=0.5\,\mathrm{lr}_0$ |
| `opacity_lr`, `feature_lr`, `shfeature_lr`, `highfeature_lr`, `lowfeature_lr` | hằng số learning rate (Adam) cho các nhóm tham số tương ứng, không thay đổi theo lịch mũ |
| `scale_rotation_scheduler` | cờ bật lịch mũ cho `scaling`/`rotation` thay vì hằng số |
| `adam_eps_order` | bậc $\varepsilon = 10^{-\text{order}}$ dùng ổn định số học trong Adam (ảnh hưởng `optimizer_step`, mục 7 `gaussian_model.md`) |

### 3.2. Hàm mất mát (`train.md` §1)

| Tham số | Giá trị | Vai trò |
|---|---|---|
| `lambda_dssim` | 0.2 | trọng số $\lambda_{dssim}$ trong $\mathcal{L}_{rgb}=(1-\lambda_{dssim})\mathcal{L}_1+\lambda_{dssim}(1-\mathrm{SSIM})+\lambda_{l2}\mathcal{L}_2$ |
| `lambda_l2` | 2.0 | trọng số $\lambda_{l2}$ trong cùng công thức |
| `lambda_tone`, `lambda_freq` | 0, 0 | trọng số cho `tone_curve_loss`, `frequency_loss` — định nghĩa trong `utils/loss_utils.py` nhưng **không được cộng vào loss trong `train.py`** ở phiên bản hiện tại (mặc định bằng 0 và không dùng) |
| `batch_size` | 1 | số camera gộp gradient trước một bước optimizer (`train.md` §1, §6) |

### 3.3. Densification chuẩn (`gaussian_model.md` §5)

| Tham số | Vai trò |
|---|---|
| `percent_dense` | ngưỡng $p_{dense}$ phân biệt clone ($\max_k s_{i,k}\le p_{dense}E$) và split ($> p_{dense}E$) trong `densify_and_clone`/`densify_and_split` |
| `densify_grad_threshold`, `densify_grad_abs_threshold` | ngưỡng $\tau_{grad}, \tau_{abs}$ cho gradient view-space trong điều kiện clone/split (dùng trong nhánh `warmup_densification` gọi `densify_and_prune`) |
| `densify_from_iter`, `densify_until_iter`, `densification_interval` | lịch kích hoạt vòng densify trong `train.py` (`train.md` §5) |
| `opacity_reset_interval`, `opacity_reset_decay` | chu kỳ và hệ số $\gamma$ trong `reset_opacity`: $\alpha^{new}=\alpha\cdot\gamma$ (`gaussian_model.md` §2.4) |
| `warmup_densification` | cờ bật nhánh densify kiểu 3DGS gốc (`densify_and_prune`) song song với `densify_and_prune_structgs` |
| `prune_until_iter` | **lưu ý**: gán $25000$ ở dòng 96 rồi bị ghi đè thành $30000$ ở dòng 100 (xem mục 0) — giá trị hiệu lực là $30000$ |
| `min_weight` | $=0.7$ (dòng 97), ngưỡng trọng số tối thiểu — không thấy được tham chiếu trực tiếp trong `train.py`/`gaussian_model.py` đã đọc |

### 3.4. Tham số dành riêng StructGS / FastGS (`gaussian_model.md` §6)

| Tham số | Vai trò |
|---|---|
| `grad_thresh`, `grad_abs_thresh` | ngưỡng $\tau_{grad}, \tau_{abs}$ phiên bản StructGS, dùng trong `densify_and_prune_structgs` (điều kiện `split_i`, `clone_i`) |
| `dense` | ngưỡng $\rho$ phân biệt `clone_qual`/`split_qual` theo tỉ lệ ($\max_k s_{i,k}$ so với $\rho\cdot E$) trong `densify_and_prune_structgs` |
| `mult` | hệ số nhân kích thước "compact box" kiểm soát số tile mỗi splat, truyền vào `render_structgs` (ảnh hưởng hiệu năng rasterize, không phải công thức Gaussian) |
| `loss_thresh` | ngưỡng loss (khai báo, không thấy được tham chiếu trong `train.py`/`gaussian_model.py` đã đọc) |
| `importance_score_threshold` | ngưỡng $\tau_{imp}$ cho `metric_i = (\text{importance\_score}_i > \tau_{imp})` trong `densify_and_prune_structgs` |
| `tau_expand` | ngưỡng $\tau_{expand}$ để kích hoạt mở rộng Gaussian nhỏ: $\text{mask}=(0<\eta<\tau_{expand})$ trong `expand_undersized_gs` |
| `ks_scale_power` | số mũ $p$ trong $s^{child}=s^{parent}/k^p$ khi split dị hướng (`densify_and_split_structgs`/`densify_and_prune_structgs`) |
| `adaptive_clone`, `expansion_speed`, `clone_target_eta`, `max_clones_per_axis` | tham số điều khiển mở rộng/nhân bản thích ứng theo $\eta$ (dùng ở các biến thể mở rộng của densify; vai trò toán học cụ thể nằm trong các hàm `*_structgs` liên quan đến $\eta$ và số lần clone tối đa theo trục) |
| `split_ratio_threshold`, `prune_ratio_threshold` | ngưỡng $\tau_{split}, \tau_{prune}$ cho tỉ lệ view có $\eta$ cao/thấp, dùng trực tiếp trong `train.py` để tạo `split_mask`/`prune_mask` (`train.md` §4) |
| `eta_compute_mode` | chọn công thức tính $\eta$ ("wavelength" hay "projection" — định nghĩa cụ thể nằm ngoài các file đã đọc, trong `utils/freq_utils.py`) |

### 3.5. Tham số lấy mẫu tần số / structure tensor

| Tham số | Vai trò |
|---|---|
| `st_levels`, `st_mode` | số mức (octave) và phiên bản (`v1`/`v2`) của `get_multiscale_structure_tensor_*` dùng để tính $S_{xx}, S_{xy}, S_{yy}$ (structure tensor), là nguồn tính $\eta$ |
| `freq_grad_threshold`, `freq_transmittance_threshold`, `freq_opacity_threshold` | ngưỡng truyền vào `update_freq_stats_online` (ngoài phạm vi các file đã đọc) để lọc điểm hợp lệ khi tích luỹ thống kê $\eta$ |

### 3.6. Prune

| Tham số | Vai trò |
|---|---|
| `prune_from_iter`, `prune_until_iter`, `prune_interval` | lịch prune (khai báo nhưng vòng lặp `train.py` dùng `prune_iterations` truyền từ CLI thay vì các tham số này) |
| `densify_prune_ratio`, `after_densify_prune_ratio` | tỉ lệ prune dự kiến (không thấy được tham chiếu trực tiếp trong `train.py`/`gaussian_model.py` đã đọc) |
| `min_weight` | ngưỡng trọng số đóng góp tối thiểu (liên quan prune theo transmittance, không thấy dùng trực tiếp trong các file đã đọc) |

### 3.7. Khởi tạo / lấy mẫu bổ sung

| Tham số | Vai trò |
|---|---|
| `sample_bbox_faces`, `far_plane_*`, `sample_far_plane` | điều khiển khởi tạo thêm điểm Gaussian trên biên bounding box / mặt phẳng xa (xem khối code trong `train.py` phần "Sample new GS to cover the 6 faces") — các phép tính ở đây là nội suy lưới tuyến tính `torch.linspace`/`meshgrid`, không liên quan trực tiếp đến công thức Gaussian trong `gaussian_model.py` |
| `densification_window_width` | độ rộng cửa sổ densify (khai báo, không thấy dùng trực tiếp trong `train.py` đã đọc) |
| `optimizer_type` | chọn nhánh `default`/`sparse_adam`/`hybrid` trong bước optimizer (`train.md` §5, `gaussian_model.md` §4) |
| `camera_sampling` | chọn chiến lược lấy camera (`fps` hay random) — ảnh hưởng thứ tự duyệt, không phải công thức Gaussian |
| `compute_3d_filter` | bật/tắt tính $r_i^{filter}$ trong `compute_3D_filter` (`gaussian_model.md` §2.1) |
| `random_background` | nền ngẫu nhiên $bg\sim\mathcal{U}([0,1]^3)$ thay vì màu cố định khi render train |

---

## Bảng hằng số/ngưỡng (giá trị mặc định)

| Tham số | Giá trị mặc định |
|---|---|
| `iterations` | 30000 |
| `position_lr_init` | 0.00016 |
| `position_lr_final` | 0.0000016 |
| `position_lr_delay_mult` | 0.01 |
| `position_lr_max_steps` | 30000 |
| `opacity_lr` | 0.05 |
| `scaling_lr` | 0.01 |
| `rotation_lr` | 0.002 |
| `feature_lr` | 0.0025 |
| `shfeature_lr` | 0.005 |
| `percent_dense` | 0.001 |
| `lambda_dssim` | 0.2 |
| `lambda_l2` | 2.0 |
| `densification_interval` | 100 |
| `opacity_reset_interval` | 3000 |
| `opacity_reset_decay` | 0.1 |
| `densify_from_iter` | 500 |
| `densify_until_iter` | 15000 |
| `densify_grad_threshold` | 0.0002 |
| `densify_grad_abs_threshold` | 0.0004 |
| `grad_thresh` | 0.0002 |
| `grad_abs_thresh` | 0.0002 |
| `dense` | 0.001 |
| `mult` | 0.7 |
| `tau_expand` | 1.0 |
| `ks_scale_power` | 1.0 |
| `importance_score_threshold` | 0.5 |
| `split_ratio_threshold` | 0.8 |
| `prune_ratio_threshold` | 0.8 |
| `batch_size` | 1 |
| `adam_eps_order` | 8 |
| `sh_degree` | 3 |

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức / vai trò |
|---|---|
| `self.sh_degree = 3` | $\ell_{max}=3$, chiều hệ số SH $c_i\in\mathbb{R}^{3\times(\ell_{max}+1)^2}$ (`gaussian_model.md` §1) |
| `self.position_lr_init = 0.00016` | $\mathrm{lr}_{init}$ trong $\mathrm{lr}(t)=\delta(t)\exp(\log(\mathrm{lr}_{init})(1-t)+\log(\mathrm{lr}_{final})t)$ |
| `self.position_lr_final = 0.0000016` | $\mathrm{lr}_{final}$ trong công thức trên |
| `self.position_lr_delay_mult = 0.01` | $\delta_{mult}$ trong $\delta(t)=\delta_{mult}+(1-\delta_{mult})\sin(\cdots)$ |
| `self.position_lr_max_steps = 30_000` | mẫu số chuẩn hoá $t=\text{iter}/\text{max\_steps}$ |
| `self.percent_dense = 0.001` | $p_{dense}$ trong $\max_k s_{i,k}\lessgtr p_{dense}\cdot E$ |
| `self.lambda_dssim = 0.2` | $\lambda_{dssim}$ trong $\mathcal{L}_{rgb}$ |
| `self.lambda_l2 = 2.0` | $\lambda_{l2}$ trong $\mathcal{L}_{rgb}$ |
| `self.densification_interval = 100` | chu kỳ `iteration % densification_interval == 0` |
| `self.opacity_reset_interval = 3000` | chu kỳ gọi `reset_opacity` |
| `self.opacity_reset_decay = 0.1` | $\gamma$ trong $\alpha^{new}=\alpha\cdot\gamma$ |
| `self.densify_grad_threshold = 0.0002` | $\tau_{grad}$ (nhánh `densify_and_prune` cổ điển) |
| `self.densify_grad_abs_threshold = 0.0004` | $\tau_{abs}$ (nhánh `densify_and_prune` cổ điển) |
| `self.grad_thresh = 0.0002` | $\tau_{grad}$ (nhánh StructGS, `densify_and_prune_structgs`) |
| `self.grad_abs_thresh = 0.0002` | $\tau_{abs}$ (nhánh StructGS) |
| `self.dense = 0.001` | $\rho$ trong `clone_qual_i`/`split_qual_i` |
| `self.mult = 0.7` | hệ số nhân box tile truyền vào `render_structgs` |
| `self.tau_expand = 1.0` | $\tau_{expand}$ trong mask $(0<\eta<\tau_{expand})$ của `expand_undersized_gs` |
| `self.ks_scale_power = 1.0` | $p$ trong $s^{child}=s^{parent}/k^p$ |
| `self.importance_score_threshold = 0.5` | $\tau_{imp}$ trong $\text{metric}_i=(\text{importance\_score}_i>\tau_{imp})$ |
| `self.split_ratio_threshold = 0.8` | $\tau_{split}$ trong `split_mask` (`train.py`) |
| `self.prune_ratio_threshold = 0.8` | $\tau_{prune}$ trong `prune_mask` (`train.py`) |
| `self.batch_size = 1` | $B$ trong $\overline{\mathcal{L}}^{ema}_t=0.4\cdot(\sum_b\mathcal{L}^{(b)}_{rgb}/B)+0.6\,\overline{\mathcal{L}}^{ema}_{t-1}$ |
| `self.adam_eps_order = 8` | bậc $\varepsilon=10^{-8}$ ổn định số học Adam |
| `g.source_path = os.path.abspath(g.source_path)` | không phải công thức toán — chuẩn hoá đường dẫn tuyệt đối |
