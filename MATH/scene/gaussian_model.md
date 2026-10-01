# Công thức toán học trong `scene/gaussian_model.py`

Tài liệu tổng hợp toàn bộ cơ sở toán học đứng sau các hàm trong `GaussianModel`, bám sát thứ tự xuất hiện trong code. Mỗi công thức có trích dẫn nguyên văn code (kèm số dòng thật của `scene/gaussian_model.py`, hoặc `utils/general_utils.py`/`utils/sh_utils.py` khi hàm phụ trợ nằm ở đó) ngay trước phần diễn giải.

---

## 1. Biểu diễn một Gaussian 3D

Mỗi điểm Gaussian $i$ được tham số hoá bởi:

- Tâm (mean): $\mu_i \in \mathbb{R}^3$ — tensor `_xyz`
- Tỉ lệ (scale, dạng log): $s_i^{raw} \in \mathbb{R}^3$ — tensor `_scaling`
- Quaternion xoay (dạng chưa chuẩn hoá): $q_i^{raw} \in \mathbb{R}^4$ — tensor `_rotation`
- Độ mờ (opacity, dạng logit): $o_i^{raw} \in \mathbb{R}$ — tensor `_opacity`
- Hệ số cầu điều hoà (Spherical Harmonics) bậc $\le \ell_{max}$: $c_i \in \mathbb{R}^{3\times(\ell_{max}+1)^2}$ — `_features_dc`, `_features_rest`

### 1.1. Kích hoạt tham số (`setup_functions`)

Trích nguyên văn (`scene/gaussian_model.py`, dòng 39–46):

```python
self.scaling_activation = torch.exp
self.scaling_inverse_activation = torch.log

self.covariance_activation = build_covariance_from_scaling_rotation
self.opacity_activation = torch.sigmoid
self.inverse_opacity_activation = inverse_sigmoid

self.rotation_activation = torch.nn.functional.normalize
```

$$
s_i = \exp(s_i^{raw}), \qquad s_i^{raw} = \log(s_i)
$$

$$
\alpha_i = \sigma(o_i^{raw}) = \frac{1}{1+e^{-o_i^{raw}}}, \qquad o_i^{raw} = \sigma^{-1}(\alpha_i) = \log\frac{\alpha_i}{1-\alpha_i}
$$

(hàm `inverse_sigmoid`, trích nguyên văn `utils/general_utils.py` dòng 21–22: `return torch.log(x/(1-x))`.)

$$
q_i = \frac{q_i^{raw}}{\lVert q_i^{raw} \rVert_2}
$$

(`torch.nn.functional.normalize` mặc định chuẩn hoá $\ell_2$ trên chiều cuối.)

### 1.2. Chế độ opacity thay thế (`modify_functions`)

Trích nguyên văn (dòng 48–52):

```python
def modify_functions(self):
    old_opacities = self.get_opacity.clone()
    self.opacity_activation = torch.abs
    self.inverse_opacity_activation = identity_gate
    self._opacity = self.opacity_activation(old_opacities)
```

Thay vì sigmoid, cho phép opacity tuyến tính không âm:

$$
\alpha_i = |o_i^{raw}|, \qquad o_i^{raw} = \alpha_i \ \text{(identity gate, `utils/general_utils.py` dòng 18–19: `def identity_gate(x): return x`)}
$$

### 1.3. Ma trận hiệp phương sai (`get_covariance`, `build_covariance_from_scaling_rotation`)

Trích nguyên văn (dòng 33–37, 203–204):

```python
def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm
```

```python
def get_covariance(self, scaling_modifier = 1):
    return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
```

Với ma trận xoay $R_i = R(q_i)$ và ma trận tỉ lệ $S_i = \mathrm{diag}(m \cdot s_i)$ ($m$ = `scaling_modifier`, xây dựng trong `utils/general_utils.py::build_scaling_rotation` dòng 146–155, dùng `build_rotation` dòng 81–102):

$$
L_i = R_i S_i, \qquad \Sigma_i = L_i L_i^\top = R_i S_i S_i^\top R_i^\top
$$

Chỉ phần tam giác trên (6 giá trị độc lập) được lưu do $\Sigma_i$ đối xứng (`strip_symmetric` → `strip_lowerdiag`, `utils/general_utils.py` dòng 67–79).

---

## 2. Bộ lọc không gian 3D (3D Smoothing Filter) — Mip-Splatting

### 2.1. Tính bán kính lọc (`compute_3D_filter`)

Trích nguyên văn (dòng 219–254):

```python
R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
 # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
xyz_cam = xyz @ R + T[None, :]

xyz_to_cam = torch.norm(xyz_cam, dim=1)

# project to screen space
valid_depth = xyz_cam[:, 2] > 0.2


x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
z = torch.clamp(z, min=0.001)

x = x / z * camera.focal_x + camera.image_width / 2.0
y = y / z * camera.focal_y + camera.image_height / 2.0

# use similar tangent space filtering as in the paper
in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))


valid = torch.logical_and(valid_depth, in_screen)

distance[valid] = torch.min(distance[valid], z[valid])
valid_points = torch.logical_or(valid_points, valid)
if focal_length < camera.focal_x:
    focal_length = camera.focal_x

distance[~valid_points] = distance[valid_points].max()

filter_3D = distance / focal_length * (0.2 ** 0.5)
self.filter_3D = filter_3D[..., None]
```

Với mỗi camera $c$, điểm $\mu_i$ được chiếu sang hệ camera:

$$
\mathbf{x}^{cam}_i = R_c^\top \mu_i + t_c
$$

(code dùng $R$ đã transpose sẵn do quy ước `glm`, comment dòng 221 xác nhận điều này), chiếu phối cảnh:

$$
u_i = \frac{x^{cam}_{i,x}}{z^{cam}_i} f_x + \frac{W}{2}, \qquad
v_i = \frac{x^{cam}_{i,y}}{z^{cam}_i} f_y + \frac{H}{2}
$$

Một điểm hợp lệ nếu $z^{cam}_i > 0.2$ và nằm trong vùng mở rộng màn hình $[-0.15W, 1.15W]\times[-0.15H,1.15H]$.

Khoảng cách gần nhất tới mọi camera hợp lệ (chú ý: biến lưu tên `distance` nhưng thực chất là giá trị $z$ — độ sâu theo trục quang học — chứ không phải khoảng cách Euclid `xyz_to_cam`, vì dòng `distance[valid] = torch.min(distance[valid], z[valid])` dùng `z`, không dùng `xyz_to_cam`; biến `xyz_to_cam` được tính nhưng **không dùng** ở bước này):

$$
d_i = \min_{c \ \text{hợp lệ}} z^{cam}_{i,c}
$$

Bán kính lọc 3D (xấp xỉ kích thước 1 pixel chiếu ngược vào không gian thế giới):

$$
r_i^{filter} = \frac{d_i}{f_{max}} \sqrt{0.2}, \qquad f_{max} = \max_c f_{x,c}
$$

### 2.2. Tỉ lệ hiệu dụng có lọc (`get_scaling_with_3D_filter`)

Trích nguyên văn (dòng 155–161):

```python
@property
def get_scaling_with_3D_filter(self):
    scales = self.get_scaling

    scales = torch.square(scales) + torch.square(self.filter_3D)
    scales = torch.sqrt(scales)
    return scales
```

$$
\tilde{s}_i = \sqrt{s_i^2 + (r_i^{filter})^2}
$$

### 2.3. Opacity hiệu dụng có lọc (`get_opacity_with_3D_filter`)

Trích nguyên văn (dòng 189–201):

```python
@property
def get_opacity_with_3D_filter(self):
    opacity = self.opacity_activation(self._opacity)
    # apply 3D filter
    scales = self.get_scaling

    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)

    scales_after_square = scales_square + torch.square(self.filter_3D)
    det2 = scales_after_square.prod(dim=1)
    coef = torch.sqrt(det1 / det2)
    return opacity * coef[..., None]
```

Coi $\Sigma$ là chéo hoá với $\det \Sigma_1 = \prod_k s_{i,k}^2$ (trước lọc) và $\det \Sigma_2 = \prod_k (s_{i,k}^2 + r_i^{filter\,2})$ (sau lọc). Hệ số hiệu chỉnh năng lượng tích phân Gaussian giữ nguyên:

$$
\kappa_i = \sqrt{\frac{\det \Sigma_1}{\det \Sigma_2}}, \qquad \alpha_i^{filter} = \alpha_i \cdot \kappa_i
$$

### 2.4. Reset opacity có bù lọc (`reset_opacity`)

Trích nguyên văn (dòng 415–433):

```python
def reset_opacity(self, decay_factor=0.1):
    # reset opacity to by considering 3D filter
    current_opacity_with_filter = self.get_opacity_with_3D_filter
    opacities_new = current_opacity_with_filter * decay_factor
    # apply 3D filter
    scales = self.get_scaling

    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)

    scales_after_square = scales_square + torch.square(self.filter_3D)
    det2 = scales_after_square.prod(dim=1)
    coef = torch.sqrt(det1 / det2)
    opacities_new = opacities_new / coef[..., None]
    opacities_new = self.inverse_opacity_activation(opacities_new)

    optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    self._opacity = optimizable_tensors["opacity"]
```

$$
\alpha_i^{new,filter} = \alpha_i^{filter} \cdot \gamma \quad (\gamma = \text{decay\_factor})
$$

Bù ngược hệ số $\kappa_i$ để lưu lại opacity "gốc" (chưa lọc):

$$
\alpha_i^{new} = \frac{\alpha_i^{new,filter}}{\kappa_i}, \qquad o_i^{raw,new} = \sigma^{-1}(\alpha_i^{new})
$$

---

## 3. Khởi tạo từ Point Cloud (`create_from_pcd`)

Trích nguyên văn (dòng 261–283):

```python
def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
    self.spatial_lr_scale = spatial_lr_scale
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
```

Màu RGB chuyển sang hệ số DC của SH:

$$
c_{i,0} = \mathrm{RGB2SH}(\text{color}_i)
$$

(`utils/sh_utils.py` dòng 114–115: `def RGB2SH(rgb): return (rgb - 0.5) / C0`, với $C_0$ là hằng số SH bậc 0; các bậc cao hơn khởi tạo 0 — dòng `features[:, 3:, 1:] = 0.0`.)

Tỉ lệ ban đầu dựa trên khoảng cách tới 3 láng giềng gần nhất (`distCUDA2`):

$$
d^2_i = \max(\mathrm{dist2}(\mu_i), 10^{-7}), \qquad s_i^{raw} = \log\sqrt{d_i^2} = \tfrac12 \log d_i^2
$$

(lặp lại cho cả 3 trục, dòng `.repeat(1, 3)`)

Quaternion khởi tạo là xoay đơn vị: $q_i = (1,0,0,0)$ (dòng `rots[:, 0] = 1`).

Opacity khởi tạo:

$$
\alpha_i = 0.1, \qquad o_i^{raw} = \sigma^{-1}(0.1)
$$

---

## 4. Lịch học (`training_setup`, `update_learning_rate`)

Trích nguyên văn khởi tạo scheduler (dòng 325–339):

```python
self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                            max_steps=training_args.position_lr_max_steps)

self.scale_rotation_scheduler = training_args.scale_rotation_scheduler
if self.scale_rotation_scheduler:
    self.scaling_scheduler_args = get_expon_lr_func(lr_init=training_args.scaling_lr*3.0,
                                                    lr_final=training_args.scaling_lr * 0.5,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr*3.0,
                                                     lr_final=training_args.rotation_lr * 0.5,
                                                     lr_delay_mult=training_args.position_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)
```

Định nghĩa `get_expon_lr_func` (`utils/general_utils.py` dòng 50–63):

```python
def helper(step):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp
```

Với $t \in [0,1] = \text{iter}/\text{max\_steps}$ và có trễ khởi động:

$$
\delta(t) = \delta_{mult} + (1-\delta_{mult})\sin\!\left(\frac{\pi}{2}\,\mathrm{clip}\!\left(\frac{t_{\text{delay}}}{t_{delay,steps}},0,1\right)\right)
$$

$$
\mathrm{lr}(t) = \delta(t)\cdot \exp\!\big(\log(\mathrm{lr}_{init})(1-t) + \log(\mathrm{lr}_{final})\,t\big)
$$

> **Phát hiện quan trọng (không có ở bản cũ):** cả 3 lời gọi `get_expon_lr_func` trong `training_setup` (dòng 325–339) chỉ truyền `lr_delay_mult`, **không truyền `lr_delay_steps`** → tham số này nhận giá trị mặc định `lr_delay_steps=0` (chữ ký hàm dòng 33). Do đó nhánh `if lr_delay_steps > 0:` ở dòng 54 **luôn luôn `False`** trong SADGS, và `delay_rate = 1.0` **cố định** (nhánh `else` dòng 60) — công thức $\delta(t)$ có dạng sin ở trên **không bao giờ thực sự kích hoạt** trong pipeline này, dù hàm vẫn định nghĩa nó. Thực tế lịch học đang dùng chỉ còn:
>
> $$
> \mathrm{lr}(t) = \exp\!\big(\log(\mathrm{lr}_{init})(1-t) + \log(\mathrm{lr}_{final})\,t\big), \qquad t=\mathrm{clip}(\text{iter}/\text{max\_steps},0,1)
> $$
>
> Đây là **log-linear interpolation (suy giảm mũ) thuần tuý**, không có giai đoạn "warm-up" làm chậm learning rate ban đầu như mô tả ở bản cũ.

Áp dụng cho `xyz` (luôn), và khi `scale_rotation_scheduler=True` cũng áp dụng cho `scaling`, `rotation` với $\mathrm{lr}_{init}=3\,\mathrm{lr}_0$, $\mathrm{lr}_{final}=0.5\,\mathrm{lr}_0$ (dòng 332–339).

Trích nguyên văn `update_learning_rate` (dòng 341–355):

```python
def update_learning_rate(self, iteration):
    ''' Learning rate scheduling per step '''
    xyz_lr = None
    for param_group in self.optimizer.param_groups:
        if param_group["name"] == "xyz":
            lr = self.xyz_scheduler_args(iteration)
            param_group['lr'] = lr
            xyz_lr = lr
        elif self.scale_rotation_scheduler and param_group["name"] == "scaling":
            lr = self.scaling_scheduler_args(iteration)
            param_group['lr'] = lr
        elif self.scale_rotation_scheduler and param_group["name"] == "rotation":
            lr = self.rotation_scheduler_args(iteration)
            param_group['lr'] = lr
    return xyz_lr
```

### Lịch cập nhật Adam thưa (`optimizer_step`)

Trích nguyên văn (dòng 357–377):

```python
def optimizer_step(self, iteration):
    ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
    if iteration <= 15000:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.shoptimizer.step()
        self.shoptimizer.zero_grad(set_to_none = True)
    elif iteration <= 20000:
        if iteration % 32 ==0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none = True)
    else:
        if iteration % 64 ==0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            self.shoptimizer.step()
            self.shoptimizer.zero_grad(set_to_none = True)
```

$$
\text{step} \iff
\begin{cases}
\text{mỗi iter} & \text{iter} \le 15000\\
\text{iter} \bmod 32 = 0 & 15000 < \text{iter} \le 20000\\
\text{iter} \bmod 64 = 0 & \text{iter} > 20000
\end{cases}
$$

---

## 5. Densification chuẩn (3DGS gốc)

### 5.1. Thống kê gradient (`add_densification_stats`)

Trích nguyên văn (dòng 1054–1057):

```python
def add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
    self.denom[update_filter] += 1
```

Với $v_i$ là gradient của điểm chiếu lên màn hình (viewspace), 2 thành phần đầu là gradient "view-space" thường dùng để clone, 2 thành phần sau (kênh phụ, "abs" — sinh ra từ `Register_dL_dmean2D_z/w` trong `cuda/backward.md` mục 1.1) dùng để split:

$$
G_i \mathrel{+}= \lVert v_{i,0:2} \rVert_2, \qquad G_i^{abs} \mathrel{+}= \lVert v_{i,2:4} \rVert_2, \qquad N_i \mathrel{+}= 1
$$

Gradient trung bình (tính tại `densify_and_prune`, dòng 914–918): $\bar{G}_i = G_i / N_i$, $\bar{G}_i^{abs} = G_i^{abs}/N_i$.

### 5.2. Clone (`densify_and_clone`)

Trích nguyên văn (dòng 893–909):

```python
def densify_and_clone(self, grads, grad_threshold, scene_extent):
    # Extract points that satisfy the gradient condition
    selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
                                          torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

    new_xyz = self._xyz[selected_pts_mask]
    new_features_dc = self._features_dc[selected_pts_mask]
    new_features_rest = self._features_rest[selected_pts_mask]
    new_opacities = self._opacity[selected_pts_mask]
    new_scaling = self._scaling[selected_pts_mask]
    new_rotation = self._rotation[selected_pts_mask]
```

Nhân đôi các Gaussian **nhỏ** nhưng có gradient lớn (under-reconstruction):

$$
\text{mask}_i = \big(\lVert \bar{G}_i \rVert \ge \tau_{grad}\big) \ \wedge \ \big(\max_k s_{i,k} \le p_{dense}\cdot E\big)
$$

($E$ = scene extent). Điểm mới là bản sao y hệt điểm cha (cùng $\mu,s,q,\alpha$, SH) — không có phép biến đổi nào, chỉ indexing bằng mask.

### 5.3. Split (`densify_and_split`, $N=2$)

Trích nguyên văn (dòng 866–891):

```python
def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    n_init_points = self.get_xyz.shape[0]
    padded_grad = torch.zeros((n_init_points), device="cuda")
    padded_grad[:grads.shape[0]] = grads.squeeze()
    selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
                                          torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

    stds = self.get_scaling[selected_pts_mask].repeat(N,1)
    means =torch.zeros((stds.size(0), 3),device="cuda")
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
```

Chọn Gaussian **lớn** có gradient lớn (over-reconstruction):

$$
\text{mask}_i = \big(\lVert \bar{G}^{abs}_i \rVert \ge \tau_{grad}\big) \ \wedge \ \big(\max_k s_{i,k} > p_{dense}\cdot E\big)
$$

Sinh $N$ điểm con bằng lấy mẫu Gaussian cục bộ rồi xoay vào hệ thế giới:

$$
\epsilon \sim \mathcal{N}(0, \mathrm{diag}(s_i)), \qquad \mu_i^{child} = \mu_i + R_i\,\epsilon
$$

Tỉ lệ con co lại:

$$
s_i^{child} = \frac{s_i}{0.8\,N}
$$

Các điểm cha bị xoá (`prune_points`, dòng 890–891) sau khi thêm $N$ điểm con.

### 5.4. Prune (`densify_and_prune`)

Trích nguyên văn (dòng 911–928):

```python
def densify_and_prune(self, max_grad,max_grad_abs, min_opacity, extent, max_screen_size, radii):
    self.tmp_radii = radii

    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0

    grads_abs = self.xyz_gradient_accum_abs / self.denom
    grads_abs[grads_abs.isnan()] = 0.0

    self.densify_and_clone(grads, max_grad, extent)
    self.densify_and_split(grads_abs, max_grad_abs, extent)

    prune_mask = (self.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    self.prune_points(prune_mask)
```

$$
\text{prune}_i = (\alpha_i < \alpha_{min}) \ \vee \ (r_i^{2D} > r_{max}) \ \vee \ \big(\max_k s_{i,k} > 0.1\,E\big)
$$

> **Lưu ý:** `densify_and_clone` dùng `grads` (gradient thường, không phải abs) còn `densify_and_split` dùng `grads_abs` — khớp đúng với vai trò "clone dùng gradient chuẩn, split dùng gradient abs" đã nêu ở mục 5.1–5.3; không có nhầm lẫn giữa hai biến này trong code.

---

## 6. Densification dị hướng theo tần số (StructGS, các hàm `*_structgs`)

Ý tưởng cốt lõi: dùng một chỉ số năng lượng tần số $\eta$ (alias/undersampling theo lý thuyết lấy mẫu Nyquist) để quyết định **tách bất đẳng hướng theo từng trục** thay vì tách đều.

### 6.1. Định nghĩa $\eta$

$$
\eta = (\sigma \cdot \omega_{max})^2
$$

trong đó $\sigma$ là độ lệch chuẩn (tỉ lệ) của Gaussian theo một trục, $\omega_{max}$ là tần số không gian lớn nhất cần tái tạo (tích luỹ trong `accum_eta`, `max_eta_3ch` — khởi tạo ở `training_setup` dòng 293–304, cập nhật ở bước render, **ngoài phạm vi file này**). $\eta > 1$ nghĩa là Gaussian quá lớn so với tần số tín hiệu (vi phạm Nyquist, gây alias); $\eta < \tau_{expand}$ nghĩa là Gaussian quá nhỏ (dư thừa).

### 6.2. Mở rộng Gaussian quá nhỏ (`expand_undersized_gs`)

Trích nguyên văn (dòng 833–864):

```python
def expand_undersized_gs(self, tau_expand, max_eta_3ch):
    """
    Theory:
    - eta = (sigma * omega)^2 where sigma is scale, omega is max spatial frequency
    - To satisfy Nyquist (eta = 1): sigma_target = sigma_old / sqrt(eta)
    - In log-space: log(sigma_target) = log(sigma_old) - 0.5 * log(eta)
    """
    if max_eta_3ch is None:
        return

    # Identify undersized axes (eta < tau_expand and eta > 0)
    undersized_mask = (max_eta_3ch < tau_expand) & (max_eta_3ch > 0)

    if not undersized_mask.any():
        return

    with torch.no_grad():
        eta_vals = torch.clamp(max_eta_3ch[undersized_mask], min=1e-6)
        delta_log_scale = -0.5 * torch.log(eta_vals)  # Direct correction

        delta = torch.zeros_like(self._scaling)
        delta[undersized_mask] = delta_log_scale

        new_scaling = self._scaling + delta
```

Điều kiện kích hoạt theo từng kênh/trục:

$$
\text{mask} = (0 < \eta < \tau_{expand})
$$

Mục tiêu đưa $\eta_{new}=1$ (đạt chuẩn Nyquist) bằng nghiệm giải tích trong không gian log:

$$
\sigma_{target} = \frac{\sigma_{old}}{\sqrt{\eta}}
\;\Longrightarrow\;
\log \sigma_{target} = \log \sigma_{old} - \tfrac12\log\eta
$$

$$
s^{raw,new} = s^{raw,old} - \tfrac12 \log(\eta), \qquad \eta \ge 10^{-6}\ (\text{clamp, dòng } 854)
$$

### 6.3. Tách dị hướng giải tích theo 3 trục (`densify_and_split_structgs`)

**Số lần tách mỗi trục** — trích nguyên văn (dòng 687–709):

```python
if max_eta_3ch is not None:
    # Frequency-based splitting
    etavals = max_eta_3ch[split_indices] # [K, 3]

    # Sampling Theory:
    # eta is the frequency energy ratio (~ (sigma * omega_max)^2 ).
    # To resolve the aliasing, we need to reduce sigma by factor k such that sigma_new * omega_max <= 1.
    # sigma_new = sigma / k => (sigma/k)^2 * omega^2 <= 1 => eta / k^2 <= 1 => k >= sqrt(eta).

    ks = torch.sqrt(torch.clamp(etavals, min=1.0)).ceil().int()
    ks = torch.clamp(ks, min=1)

else:
    # Gradient-based split fallback (Standard 3DGS behavior)
    max_scale_vals, max_scale_indices = torch.max(current_scales, dim=1)
    ks = torch.ones((current_scales.shape[0], 3), dtype=torch.int, device="cuda")
    ks.scatter_(1, max_scale_indices.unsqueeze(1), 2)
```

$k=(k_x,k_y,k_z)$, suy từ $\eta$ theo từng kênh màu/trục (lý thuyết lấy mẫu: cần tăng mật độ mẫu theo hệ số $\sqrt{\eta}$ để $\eta_{new}\le 1$):

$$
k = \left\lceil \sqrt{\max(\eta, 1)} \right\rceil, \qquad k \ge 1
$$

> **Lưu ý khớp code chính xác:** `torch.clamp(etavals, min=1.0)` nghĩa là $\max(\eta,1)$ áp dụng **trước** khi lấy căn bậc hai, đúng công thức trên (không phải $\max(\sqrt\eta,1)$) — code và công thức khớp nhau.

(nếu không có $\eta$: fallback kiểu 3DGS gốc — $k=2$ chỉ trên trục có tỉ lệ lớn nhất, còn lại $k=1$, đúng dòng `ks.scatter_(1, max_scale_indices.unsqueeze(1), 2)`)

Tổng số điểm con của 1 điểm cha — trích nguyên văn (dòng 714):

```python
N_per_point = ks.prod(dim=1) # [M]
```

$$
N = k_x k_y k_z
$$

**Tỉ lệ con** — trích nguyên văn (dòng 722, có tham số $p$ = `scale_power`):

```python
new_scaling = self.scaling_inverse_activation(current_scales / ks.float()**scale_power).repeat_interleave(repeats, dim=0)
```

$$
s^{child} = \frac{s^{parent}}{k^{\,p}}
$$

(mặc định `scale_power=1.0` — tuyến tính).

**Lưới toạ độ cục bộ** — trích nguyên văn (dòng 757–765):

```python
iz = child_indices % kz_r
iy = (child_indices // kz_r) % ky_r
ix = child_indices // (ky_r * kz_r)

grid_x = ix.float() - (kx_r.float() - 1.0) / 2.0
grid_y = iy.float() - (ky_r.float() - 1.0) / 2.0
grid_z = iz.float() - (kz_r.float() - 1.0) / 2.0
```

Với chỉ số nguyên $(i_x,i_y,i_z)$, $0\le i_x<k_x,\dots$, decode từ chỉ số phẳng theo thứ tự `meshgrid(ij)` (trục $z$ biến thiên nhanh nhất):

$$
i_z = n \bmod k_z,\quad i_y = \left\lfloor n/k_z \right\rfloor \bmod k_y,\quad i_x = \left\lfloor n/(k_y k_z)\right\rfloor
$$

Toạ độ lưới căn giữa (để điểm con phân bố đối xứng quanh tâm cha):

$$
g_x = i_x - \frac{k_x-1}{2}, \qquad (\text{tương tự cho } g_y, g_z)
$$

**Khoảng cách phân tách** (separation) — trích nguyên văn (dòng 774–778):

```python
scales_repeated = current_scales.repeat_interleave(repeats, dim=0)
stds_new_repeated = scales_repeated / ks_repeated.float()
separations = stds_new_repeated * (12**0.5)

local_offsets = separations * grid_flat # [Sum(N), 3]
```

suy từ phương sai mới và giả định phân phối đều rời rạc có phương sai bằng Gaussian liên tục ($\mathrm{Var}(\mathcal{U}) = L^2/12 \Rightarrow L = \sigma\sqrt{12}$):

$$
\sigma^{child} = \frac{\sigma^{parent}}{k}, \qquad \Delta = \sigma^{child}\sqrt{12}
$$

> **Lưu ý:** `stds_new_repeated = scales_repeated / ks_repeated` dùng $k$ (không phải $k^p$ như tỉ lệ `new_scaling` ở trên) — tức khoảng cách phân tách và tỉ lệ hiển thị của con dùng **công thức chia khác nhau** khi $p\ne1$: separation luôn chia cho $k^1$, còn scale hiển thị chia cho $k^p$. Đây là điểm cần lưu ý khi `scale_power`$\ne1$ (không phải lỗi, nhưng tách biệt rõ hai vai trò: separation đảm bảo đúng phương sai thống kê của lưới, còn scale hiển thị có thể co lại mạnh/yếu hơn tuỳ `scale_power`).

**Độ lệch cục bộ** và **phép xoay vào hệ thế giới** — trích nguyên văn (dòng 782–788):

```python
R_sub = build_rotation(new_rotation)
world_offsets = torch.bmm(R_sub, local_offsets.unsqueeze(-1)).squeeze(-1)
new_xyz = current_xyz.repeat_interleave(repeats, dim=0) + world_offsets
```

$$
o^{local} = \Delta \odot g, \qquad o^{world} = R_i\, o^{local}, \qquad \mu^{child} = \mu^{parent} + o^{world}
$$

Các thuộc tính khác (SH, opacity, rotation, filter_3D, radii) được nhân bản (`repeat_interleave`, dòng 723–728) từ cha sang mọi con. Bộ đếm `densify_count` của con $=$ `densify_count` của cha $+\,1$ (dòng 729). Sau khi sinh $N$ con, điểm cha gốc bị xoá (dòng 807–831, `prune_points(full_prune_filter)`).

### 6.4. Clone cho StructGS (`densify_and_clone_structgs`)

Trích nguyên văn (dòng 934–954):

```python
def densify_and_clone_structgs(self, metric_mask, filter):
    selected_pts_mask = torch.logical_and(metric_mask, filter)
    ...
    # [NEW] Clone inherits parent's densify_count (no increment)
    parent_counts = self.densify_count[selected_pts_mask]
    n_old = self.get_xyz.shape[0]

    self.densification_postfix(...)

    n_new = new_xyz.shape[0]
    self.densify_count[n_old:n_old+n_new] = parent_counts
```

Giống clone chuẩn nhưng mask $= \text{metric\_mask} \wedge \text{filter}$ (không ràng buộc theo tỉ lệ/gradient cố định mà nhận mask tuỳ ý từ bên ngoài — ví dụ từ điểm số quan trọng). `densify_count` được **kế thừa nguyên vẹn** (không tăng) vì đây là nhân bản, không phải chia nhỏ.

### 6.5. Hợp nhất điều kiện split/clone (`densify_and_prune_structgs`)

Trích nguyên văn điều kiện chất lượng và merge (dòng 990–1002):

```python
clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

final_split_mask = torch.logical_and(
    torch.logical_or(full_split_mask, grad_qualifiers_abs),
    split_qualifiers
)

final_clone_mask = torch.logical_and(
    torch.logical_or(full_split_mask, grad_qualifiers),
    clone_qualifiers
)
```

$$
\text{clone\_qual}_i = \max_k s_{i,k} \le \rho \cdot E, \qquad
\text{split\_qual}_i = \max_k s_{i,k} > \rho \cdot E \quad (\rho = \texttt{args.dense})
$$

$$
\text{split}_i = \Big(\text{custom\_split}_i \ \vee\ \lVert \bar{G}^{abs}_i\rVert \ge \tau_{abs}\Big) \wedge \text{split\_qual}_i
$$

$$
\text{clone}_i = \Big(\text{custom\_split}_i \ \vee\ \lVert \bar{G}_i\rVert \ge \tau_{grad}\Big) \wedge \text{clone\_qual}_i
$$

> **Lưu ý quan trọng:** cả `final_split_mask` lẫn `final_clone_mask` đều dùng `full_split_mask` (từ `custom_split_mask`) làm điều kiện OR — code **không có** `full_clone_mask`/`custom_clone_mask` riêng (chỉ có `custom_split_mask`, `custom_prune_mask` trong chữ ký hàm dòng 957–962). Tức mọi điểm được gắn cờ "split tuỳ chỉnh" từ module multiview-consistency có thể kích hoạt **cả hai nhánh** clone và split (tuỳ thêm điều kiện `clone_qual`/`split_qual` loại trừ lẫn nhau theo kích thước) — không phải lỗi, nhưng không có kênh "ép clone tuỳ chỉnh" độc lập với "ép split tuỳ chỉnh".

Áp thêm ngưỡng điểm quan trọng (nếu có) — trích nguyên văn (dòng 1005):

```python
metric_mask = importance_score > args.importance_score_threshold if importance_score is not None else torch.ones_like(final_clone_mask)
```

$$
\text{metric}_i = \big(\text{importance\_score}_i > \tau_{imp}\big)
$$

Số điểm tách theo trục dùng $k=\lceil\sqrt{\max(\eta,1)}\rceil$ như mục 6.3, với $\eta=$ `max_eta_3ch`, lũy thừa tỉ lệ $p=\texttt{args.ks\_scale\_power}$ (dòng 1011: `self.densify_and_split_structgs(combined_split_mask, max_eta_3ch=max_eta_3ch, scale_power=args.ks_scale_power)`).

**Prune** kết hợp điều kiện chuẩn + mask tuỳ chỉnh — trích nguyên văn (dòng 1014–1027):

```python
prune_mask = (self.get_opacity < min_opacity).squeeze()
if max_screen_size:
    big_points_vs = self.max_radii2D > max_screen_size
    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

n_new = prune_mask.shape[0]
n_old = full_prune_mask.shape[0]
if n_new > n_old:
    padding = torch.zeros(n_new - n_old, dtype=torch.bool, device="cuda")
    full_prune_mask = torch.cat([full_prune_mask, padding])

prune_mask = torch.logical_or(prune_mask, full_prune_mask)
```

$$
\text{prune}_i = (\alpha_i<\alpha_{min}) \vee (r_i^{2D}>r_{max}) \vee (\max_k s_{i,k}>0.1E) \vee \text{custom\_prune}_i
$$

**Prune ngẫu nhiên có trọng số** — trích nguyên văn (dòng 1029–1042):

```python
if pruning_score is not None:
    scores = 1 - pruning_score
    to_remove = torch.sum(prune_mask)
    remove_budget = int(0.5 * to_remove)

    if remove_budget:
        n_init_points = self.get_xyz.shape[0]
        padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
        padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
        sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
        selected_pts_mask[sampled_indices] = True
        final_prune = torch.logical_and(prune_mask, selected_pts_mask)
        self.prune_points(final_prune)
else:
    self.prune_points(prune_mask)
```

(nếu có `pruning_score`, lấy mẫu không hoàn lại theo phân phối tỉ lệ nghịch với điểm số, chỉ xoá một nửa ngân sách dự kiến):

$$
w_i = \frac{1}{10^{-6} + (1-\text{pruning\_score}_i)}, \qquad
B = \left\lfloor 0.5 \cdot \sum_i \mathbb{1}[\text{prune}_i] \right\rfloor
$$

Lấy mẫu $B$ chỉ số theo trọng số $w_i$ (multinomial, không lặp) trong tập thoả `prune`, chỉ các chỉ số được chọn mới thực sự bị xoá.

> **Lưu ý:** nếu `remove_budget == 0` (tức `to_remove` quá nhỏ để `0.5*to_remove` làm tròn xuống 0), nhánh `if remove_budget:` ở dòng 1034 là `False` → **không gọi `self.prune_points` nào cả** trong trường hợp này (không rơi vào nhánh `else` vì `else` chỉ gắn với `if pruning_score is not None`, không gắn với `if remove_budget`) — tức khi `pruning_score` được cung cấp nhưng ngân sách xoá bằng 0, **không có Gaussian nào bị xoá ở bước này dù `prune_mask` có thể khác rỗng**. Đây là một nhánh biên cần lưu ý (không hẳn là bug vì chủ đích "xoá một nửa ngân sách", nhưng khi ngân sách làm tròn về 0 thì hoàn toàn không xoá, không phải "xoá 0 trong số ít nhất 1").

**Giới hạn opacity trên** sau mỗi vòng densify — trích nguyên văn (dòng 1046–1048):

```python
opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
self._opacity = optimizable_tensors["opacity"]
```

$$
\alpha_i^{new} = \min(\alpha_i,\, 0.8), \qquad o_i^{raw,new} = \sigma^{-1}(\alpha_i^{new})
$$

### 6.6. Prune cuối cùng (`final_prune_structgs`)

Trích nguyên văn (dòng 1059–1066):

```python
def final_prune_structgs(self, min_opacity, pruning_score = None):
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    scores_mask = pruning_score > 0.9
    final_prune = torch.logical_or(prune_mask, scores_mask)
    self.prune_points(final_prune)
```

$$
\text{prune}_i = (\alpha_i < \alpha_{min}) \ \vee \ (\text{pruning\_score}_i > 0.9)
$$

---

## 7. Cập nhật trạng thái Adam khi thay đổi số điểm

Trích nguyên văn `replace_tensor_to_optimizer` (dòng 485–501):

```python
def replace_tensor_to_optimizer(self, tensor, name):
    optimizable_tensors = {}
    for group in self.optimizer.param_groups:
        if group["name"] == name:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            if "max_exp_avg_sq" in stored_state:
                stored_state["max_exp_avg_sq"] = torch.zeros_like(tensor)
            ...
```

- **Thay thế tensor** (`replace_tensor_to_optimizer`): reset mô-men bậc 1 và bậc 2 của Adam về 0 cho tham số bị ghi đè — $m \leftarrow 0,\ v \leftarrow 0$ (và $v_{max}\leftarrow 0$ nếu dùng amsgrad).

Trích nguyên văn `_prune_optimizer` (dòng 503–526, phần cốt lõi dòng 512–516):

```python
stored_state["exp_avg"] = stored_state["exp_avg"][mask]
stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
if "max_exp_avg_sq" in stored_state:
    stored_state["max_exp_avg_sq"] = stored_state["max_exp_avg_sq"][mask]
```

- **Cắt tỉa** (`_prune_optimizer`): $m, v, v_{max} \leftarrow m[\text{mask}], v[\text{mask}], v_{max}[\text{mask}]$.

Trích nguyên văn `cat_tensors_to_optimizer` (dòng 566–593, phần cốt lõi dòng 578–582):

```python
stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
if "max_exp_avg_sq" in stored_state:
    stored_state["max_exp_avg_sq"] = torch.cat((stored_state["max_exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
```

- **Nối thêm** (`cat_tensors_to_optimizer`): $m \leftarrow [m; 0], \ v \leftarrow [v; 0]$ cho các điểm mới (không mang theo động lượng cũ).

Đây là lý do mọi thao tác densify/prune đều phải đồng bộ lại optimizer, tránh lệch hình trạng Adam so với số lượng điểm Gaussian. Lưu ý: `_prune_optimizer` và `cat_tensors_to_optimizer` (dòng 505–506, 568–569) lặp qua **cả `self.optimizer` lẫn `self.shoptimizer`** (nếu tồn tại) để đồng bộ cả hai optimizer khi dùng chế độ `hybrid`/`default` có SH tách riêng.

---

## 8. Tóm tắt các hằng số/ngưỡng xuất hiện

| Ký hiệu | Ý nghĩa | Vị trí |
|---|---|---|
| $p_{dense}$ | ngưỡng tỉ lệ phân biệt clone/split (percent_dense) | `densify_and_clone/split` |
| $\rho$ (`args.dense`) | ngưỡng tỉ lệ cho bản StructGS | `densify_and_prune_structgs` |
| $\tau_{grad}, \tau_{abs}$ | ngưỡng gradient (thường & abs) | `args.grad_thresh`, `args.grad_abs_thresh` |
| $\tau_{imp}$ | ngưỡng điểm quan trọng | `args.importance_score_threshold` |
| $\tau_{expand}$ | ngưỡng $\eta$ để mở rộng Gaussian nhỏ | `expand_undersized_gs` |
| $\alpha_{min}$ | opacity tối thiểu trước khi bị xoá | `min_opacity` |
| $0.1E,\ 0.1\cdot\text{world size}$ | ngưỡng kích thước thế giới tối đa | `densify_and_prune*` |
| $0.8$ | trần opacity sau densify | `densify_and_prune_structgs` dòng 1046 |
| $0.8N$ | hệ số co tỉ lệ khi split chuẩn | `densify_and_split` dòng 880 |
| $\sqrt{0.2}$ | hệ số hộp→Gaussian cho `filter_3D` | `compute_3D_filter` dòng 254 |
| $0.2$ | ngưỡng depth tối thiểu hợp lệ trong `compute_3D_filter` | dòng 227 |
| $[-0.15,1.15]$ | vùng mở rộng màn hình hợp lệ (tangent-space filtering) | `compute_3D_filter` dòng 239 |

### Tổng kết các điểm đã sửa/bổ sung so với bản cũ của chính file này

1. **Bổ sung mọi code trích dẫn nguyên văn kèm số dòng** trước từng công thức (yêu cầu định dạng mới), đối chiếu trực tiếp với `scene/gaussian_model.py`, `utils/general_utils.py`, `utils/sh_utils.py`.
2. **Phát hiện mới quan trọng (mục 4)**: `lr_delay_steps` không được truyền ở bất kỳ lời gọi `get_expon_lr_func` nào trong `training_setup`, nên nhánh "delay" dạng sin **không bao giờ kích hoạt** trong SADGS — lịch học thực tế chỉ là log-linear interpolation thuần, không có giai đoạn warm-up như công thức tổng quát gợi ý.
3. **Phát hiện mới (mục 2.1)**: biến `distance` trong `compute_3D_filter` thực chất lưu toạ độ $z$ (độ sâu dọc trục quang học), không phải khoảng cách Euclid `xyz_to_cam` (biến này được tính nhưng không dùng).
4. **Phát hiện mới (mục 6.3)**: separation (`stds_new_repeated`) luôn chia cho $k^1$, trong khi `new_scaling` hiển thị chia cho $k^p$ — hai công thức tách biệt khi `scale_power`$\ne1$.
5. **Phát hiện mới (mục 6.5)**: không có `custom_clone_mask` riêng — `full_split_mask` (từ `custom_split_mask`) được dùng chung cho cả điều kiện OR của split lẫn clone; và khi `remove_budget==0` trong nhánh pruning ngẫu nhiên có trọng số, không có Gaussian nào bị xoá ở bước đó dù `prune_mask` khác rỗng.
6. Giữ nguyên mọi nội dung toán học đã đúng từ bản cũ (định nghĩa $\eta$, công thức split/clone/prune, Adam state resize).
