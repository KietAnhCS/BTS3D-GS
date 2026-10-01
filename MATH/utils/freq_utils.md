# Công thức toán học trong `utils/freq_utils.py`

File này là nơi **thực sự định nghĩa và tích luỹ** chỉ số tần số $\eta$ (`accum_eta`, `max_eta_3ch`, `eta_high/mid/low_*`) được tham chiếu nhưng không định nghĩa trong `scene/gaussian_model.py` (mục 6.1 của `gaussian_model.md`). Tài liệu bám sát thứ tự xuất hiện trong hàm chính `update_freq_stats_online`, cùng hai hàm phụ trợ `sampling_cameras`, `compute_projected_axes_subset`. Toàn bộ trích dẫn dưới đây lấy nguyên văn từ `utils/freq_utils.py` (419 dòng), kèm số dòng chính xác.

---

## 1. Lấy mẫu camera cho mỗi vòng thống kê (`sampling_cameras`, dòng 12–87)

### 1.1. Chế độ `random` (dòng 26–32)

```python
if mode == "random":
    # Original random sampling
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    return camlist
```

Chọn ngẫu nhiên không hoàn lại $N_{cam}$ camera từ tập $\{1,\dots,M\}$ (pop ngẫu nhiên khỏi stack).

### 1.2. Chế độ `fps` (Farthest Point Sampling, dòng 34–83)

Vị trí camera trong hệ thế giới (dòng 42–48):

```python
for cam in my_viewpoint_stack:
    # world_view_transform is [4, 4], extract rotation and translation
    w2c = cam.world_view_transform.transpose(0, 1)  # [4, 4]
    R = w2c[:3, :3]  # [3, 3]
    t = w2c[:3, 3]   # [3]
    # Camera position in world coordinates: -R^T @ t
    cam_pos = -R.T @ t
    camera_positions.append(cam_pos)
```

suy từ ma trận world-to-camera $W2C=[R\,|\,t]$ (lấy từ `world_view_transform`, đã transpose):

$$
\mathbf{p}_{cam} = -R^\top t
$$

Thuật toán lấy mẫu xa nhất (dòng 55–72):

```python
distances = torch.full((N,), float('inf'), device=camera_positions.device)

# Start with a random camera
current_idx = random.randint(0, N - 1)
selected_indices.append(current_idx)

for _ in range(num_cams - 1):
    # Update distances to the nearest selected point
    current_pos = camera_positions[current_idx]
    dists_to_current = torch.norm(camera_positions - current_pos, dim=1)
    distances = torch.minimum(distances, dists_to_current)
    
    # Exclude already selected points
    distances[selected_indices] = -1
    
    # Select the farthest point
    current_idx = torch.argmax(distances).item()
    selected_indices.append(current_idx)
```

khởi tạo $d_i=+\infty\ \forall i$, chọn ngẫu nhiên điểm đầu $c_0$, sau đó lặp:

$$
d_i \leftarrow \min\big(d_i,\ \lVert \mathbf{p}_i - \mathbf{p}_{c_{k}}\rVert_2\big), \qquad c_{k+1} = \arg\max_i d_i
$$

(loại các chỉ số đã chọn bằng cách gán $d_i=-1$, dòng 68) cho tới khi đủ $N_{cam}$ camera.

---

## 2. Hàm mất mát tham chiếu (`get_loss`, `compute_photometric_loss`, dòng 92–102) — không phải trọng tâm $\eta$ nhưng dùng chung module

```python
def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss
```

$$
L_1(x,y) = \mathrm{mean}_{\text{kênh}}\lvert x-y\rvert, \qquad
\widehat{L_1} = \frac{L_1-\min L_1}{\max L_1-\min L_1}
$$

$$
\mathcal{L}_{photo} = 0.8\,\lVert I-\hat I\rVert_1 + 0.2\,(1-\mathrm{SSIM}(I,\hat I))
$$

(hệ số $0.8$ viết trong code dưới dạng `(1.0 - 0.2)`, dòng 101 — về giá trị số học khớp chính xác $0.8$.)

Hàm `normalize` (dòng 104–114): chuẩn hoá một tensor giá trị dương theo trung vị của chính nó:

```python
def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value
```

$$
v_i^{norm} = \begin{cases} c\cdot \dfrac{v_i}{\mathrm{median}(v_{j>0})} & v_i>0\\ 0 & \text{khác}\end{cases}
$$

với $c=$ `config_value` (`multiplier`, dòng 105), và `NaN` được thay bằng 0 trước khi lọc (dòng 106).

---

## 3. Chiếu trục chính của Gaussian 3D lên mặt phẳng ảnh (`compute_projected_axes_subset`, dòng 116–178)

### 3.1. Suy ngược toạ độ camera-space (dòng 121–135)

```python
fx = viewpoint_camera.focal_x
fy = viewpoint_camera.focal_y

# 1. Back-project means2D to Camera Space (x, y)
# u = (x/z)*fx + cx  =>  x = (u - cx) * z / fx
# We need this to build the Jacobian at the correct location
W = viewpoint_camera.image_width
H = viewpoint_camera.image_height

vec_x = (means2D[:, 0] - W * 0.5) * (depths / fx)
vec_y = (means2D[:, 1] - H * 0.5) * (depths / fy)
```

Với điểm đã có toạ độ màn hình $\mathbf{u}_i=(u_i,v_i)$ (`means2D`) và độ sâu $z_i$ (`depths`), xấp xỉ pinhole $c_x=W/2,c_y=H/2$:

$$
x_i^{cam} = (u_i - \tfrac{W}{2})\cdot\frac{z_i}{f_x}, \qquad y_i^{cam} = (v_i - \tfrac{H}{2})\cdot\frac{z_i}{f_y}
$$

### 3.2. Jacobian phép chiếu phối cảnh (dòng 137–147)

```python
# 2. Build Jacobian J per point
# J = [ fx/z   0     -fx*x/z^2 ]
#     [ 0      fy/z  -fy*y/z^2 ]

inv_z = 1.0 / (depths + 1e-7)
inv_z2 = inv_z * inv_z

J_00 = fx * inv_z
J_02 = -fx * vec_x * inv_z2
J_11 = fy * inv_z
J_12 = -fy * vec_y * inv_z2
```

Jacobian tại điểm đó (tuyến tính hoá $\partial(u,v)/\partial(x,y,z)$, giống EWA splatting):

$$
J_i = \begin{bmatrix} \dfrac{f_x}{z_i} & 0 & -\dfrac{f_x x_i^{cam}}{z_i^2} \\[4pt] 0 & \dfrac{f_y}{z_i} & -\dfrac{f_y y_i^{cam}}{z_i^2} \end{bmatrix}
$$

(trong code, $z_i$ được cộng $\epsilon=10^{-7}$ trước khi nghịch đảo để tránh chia 0, dòng 141 — một chi tiết số học không có trong công thức giải tích ở trên.)

### 3.3. Ma trận xoay tổng hợp và 3 trục chính (dòng 149–165)

```python
# Get World->View rotation
W_view = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
R_view = W_view[:3, :3] # [3, 3]

# Convert quaternions to rotation matrices [N, 3, 3]
R_local = build_rotation(rotations) 

# R_total = R_view @ R_local
# Expand R_view to match batch size
R_view_batch = R_view.unsqueeze(0).expand(scales.shape[0], -1, -1)
R_total = torch.bmm(R_view_batch, R_local) # [N, 3, 3]

# Scale axes: Axis_vectors = R_total * Scales
# scales is [N, 3]. We broadcast multiply columns.
# This gives us the 3 axes (columns) in Camera coordinates
axes_cam = R_total * scales.unsqueeze(1) # [N, 3, 3]
```

Với $R_{view}$ là khối quay của `world_view_transform` và $R_i=R(q_i)$ (`build_rotation`, xem `general_utils.md`):

$$
R_i^{total} = R_{view}\, R_i
$$

**3 trục chính trong hệ camera**, co giãn theo tỉ lệ $s_i=(s_{i,x},s_{i,y},s_{i,z})$:

$$
A_i^{cam} = R_i^{total}\cdot \mathrm{diag}(s_i) \quad\in\mathbb{R}^{3\times3}\ (\text{mỗi cột là 1 trục đã co giãn})
$$

### 3.4. Chiếu mỗi trục xuống 2D (dòng 167–178)

```python
# 4. Project Axes to 2D
ax_x = axes_cam[:, 0, :] # [N, 3] (x-component of axis 0, 1, 2)
ax_y = axes_cam[:, 1, :]
ax_z = axes_cam[:, 2, :]

# u = J00*x + J02*z
# v = J11*y + J12*z
u_vec = J_00.unsqueeze(1) * ax_x + J_02.unsqueeze(1) * ax_z
v_vec = J_11.unsqueeze(1) * ax_y + J_12.unsqueeze(1) * ax_z

# [N, 3, 2]
return torch.stack([u_vec, v_vec], dim=2)
```

$$
\begin{pmatrix}u_i^{(k)}\\v_i^{(k)}\end{pmatrix} = J_i\, A_i^{cam}[{:},k] \;=\;
\begin{pmatrix} J_{00}\,a_x^{(k)} + J_{02}\,a_z^{(k)} \\ J_{11}\,a_y^{(k)} + J_{12}\,a_z^{(k)} \end{pmatrix}
$$

Kết quả `axes_2d` $\in\mathbb{R}^{N\times3\times2}$: với mỗi Gaussian, 3 vector 2D biểu diễn hình chiếu của 3 trục chính lên màn hình (tính bằng pixel).

---

## 4. Cấu trúc tensor cục bộ của ảnh và lấy mẫu dao động (jitter) (`update_freq_stats_online`, dòng 181–416)

### 4.1. Mặt nạ điểm "đang hoạt động" (active & visible) (dòng 205–229)

```python
# 3. Transmittance AND Opacity-based Active Mask
current_opacities = gaussians.get_opacity[global_indices].squeeze(-1)
is_active_vis = (max_transmittance > transmittance_threshold) & (current_opacities > opacity_threshold)

# [NEW] Gradient-Based Masking
if viewspace_point_tensor is not None and grad_threshold is not None:
    if viewspace_point_tensor.grad is not None:
        # viewspace_point_tensor is [N_total, 3]
        # Extract grads for visible points
        # Standard 3DGS accumulates norm of first 2 dimensions (x, y)
        current_grads = viewspace_point_tensor.grad[global_indices] # [N_vis, 3]
        grad_norms = torch.norm(current_grads[:, :2], dim=-1)

        is_grad_high = grad_norms > grad_threshold
        is_active_vis = is_active_vis & is_grad_high

# [NEW] Densify Count Filtering - Skip GS that have been densified too many times
# This prevents over-densification in already densified regions
max_densify_count = 3  
# if hasattr(gaussians, 'densify_count') and gaussians.densify_count.numel() > 0:
#     current_densify_counts = gaussians.densify_count[global_indices]
#     is_not_overdensified = current_densify_counts <= max_densify_count
#     is_active_vis = is_active_vis & is_not_overdensified

# Filter to get "Active & Visible" indices in the global array
valid_indices_global = global_indices[is_active_vis] # [N_valid]
```

$$
\text{active}_i = (\tau_i^{trans} > \tau_{trans}) \wedge (\alpha_i > \tau_{\alpha})\ \big[\wedge\ (\lVert \nabla v_i\rVert_2 > \tau_{grad})\big]
$$

trong đó $\tau_i^{trans}$ = `max_transmittance`, $\alpha_i$ = opacity hiệu dụng (`current_opacities`), điều kiện gradient là tuỳ chọn (`grad_threshold`, chỉ áp dụng khi `viewspace_point_tensor.grad is not None`). Khối "**[NEW] Densify Count Filtering**" (dòng 221–227) khai báo `max_densify_count=3` nhưng toàn bộ logic lọc theo `densify_count` **đang bị comment-out** — hiện tại không có điều kiện nào dựa trên số lần densify được áp dụng vào `is_active_vis`, đây là tính năng đã viết nhưng chưa bật.

### 4.2. Phân rã Cholesky để lấy mẫu dao động trong hiệp phương sai 2D (dòng 251–278)

```python
# 1. Extract 2D Covariance Components
cov_xx = current_cov2D[is_active_vis, 0]
cov_xy = current_cov2D[is_active_vis, 1]
cov_yy = current_cov2D[is_active_vis, 2]

# 2. Generate Random Jitter using Cholesky Decomposition
eps1 = torch.randn_like(cov_xx)
eps2 = torch.randn_like(cov_xx)

# L11 = sqrt(cov_xx)
L11 = torch.sqrt(torch.clamp(cov_xx, min=1e-6))
# L21 = cov_xy / L11
L21 = cov_xy / L11
# L22 = sqrt(cov_yy - L21^2)
L22 = torch.sqrt(torch.clamp(cov_yy - L21**2, min=1e-6))

jitter_x = L11 * eps1
jitter_y = L21 * eps1 + L22 * eps2

# 4. Apply Jitter to Sampling Coordinates
sample_x = means2D_valid[:, 0] + jitter_x
sample_y = means2D_valid[:, 1] + jitter_y
```

Hiệp phương sai màn hình $\Sigma_i^{2D}=\begin{pmatrix}\sigma_{xx}&\sigma_{xy}\\\sigma_{xy}&\sigma_{yy}\end{pmatrix}$ được phân rã $\Sigma_i^{2D}=LL^\top$:

$$
L_{11}=\sqrt{\sigma_{xx}}, \qquad L_{21}=\frac{\sigma_{xy}}{L_{11}}, \qquad L_{22}=\sqrt{\sigma_{yy}-L_{21}^2}
$$

(trong code, $\sigma_{xx}$ và $\sigma_{yy}-L_{21}^2$ đều được `clamp(min=1e-6)` trước khi lấy căn, dòng 267 và 271 — tránh `NaN` khi hiệp phương sai gần suy biến, chi tiết không có trong công thức giải tích gốc.)

Lấy mẫu dao động ngẫu nhiên $\epsilon_1,\epsilon_2\sim\mathcal N(0,1)$:

$$
\Delta u = L_{11}\epsilon_1, \qquad \Delta v = L_{21}\epsilon_1 + L_{22}\epsilon_2
$$

Vị trí lấy mẫu trong ảnh (thay vì chỉ lấy tâm Gaussian, lấy một điểm ngẫu nhiên trong ellipsoid $1\sigma$):

$$
(u_i^{sample}, v_i^{sample}) = (u_i+\Delta u,\ v_i+\Delta v)
$$

Chuẩn hoá về $[-1,1]$ để dùng `grid_sample` (dòng 290–292):

```python
norm_x = (sample_x / (w - 1)) * 2 - 1
norm_y = (sample_y / (h - 1)) * 2 - 1
grid_valid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)
```

$$
\hat u = \frac{u^{sample}}{W-1}\cdot2-1, \qquad \hat v = \frac{v^{sample}}{H-1}\cdot2-1
$$

Lấy mẫu song tuyến tính bản đồ cấu trúc tensor $S^{map}$ (3 kênh $S_{xx},S_{xy},S_{yy}$, định nghĩa trong `loss_utils.get_structure_tensor_torch`, xem mục 6) tại $(\hat u,\hat v)$ (dòng 296: `F.grid_sample(st_map, grid_valid, align_corners=True).squeeze()`).

---

## 5. Định nghĩa $\eta$ — chỉ số vi phạm Nyquist (hai chế độ `eta_compute_mode`)

### 5.1. Chế độ `"wavelength"` (mặc định dùng để tích luỹ vào `gaussian_model`, dòng 313–348)

```python
raw_Sxx = sampled_S[:, 0]
raw_Sxy = sampled_S[:, 1]
raw_Syy = sampled_S[:, 2]

trace = raw_Sxx + raw_Syy
det = raw_Sxx * raw_Syy - raw_Sxy**2
delta = torch.sqrt(torch.clamp((trace/2)**2 - det, min=0.0))

lambda1 = (trace / 2) + delta  # Max Eigenvalue (High Frequency Energy)

wavelength_min = 1.0 / (torch.sqrt(lambda1) + 1e-5) 

u_vec = axes_2d[:, :, 0] 
v_vec = axes_2d[:, :, 1] 

axis_lengths = torch.sqrt(u_vec**2 + v_vec**2 + 1e-8) # [N, 3]

eta_3ch = axis_lengths / (wavelength_min.unsqueeze(1))
```

Từ ma trận cấu trúc tensor đã lấy mẫu $S=\begin{pmatrix}S_{xx}&S_{xy}\\S_{xy}&S_{yy}\end{pmatrix}$, tính trị riêng lớn nhất $\lambda_1$ (năng lượng tần số cao nhất tại điểm đó):

$$
\mathrm{tr} = S_{xx}+S_{yy}, \qquad \det = S_{xx}S_{yy}-S_{xy}^2
$$

$$
\lambda_1 = \frac{\mathrm{tr}}{2} + \sqrt{\max\!\left(\left(\frac{\mathrm{tr}}{2}\right)^2-\det,\ 0\right)}
$$

Chuyển năng lượng gradient sang **bước sóng không gian nhỏ nhất** (độ dài chu kỳ hoạ tiết tính bằng pixel), dùng xấp xỉ $f\sim\sqrt{\lambda}$, $w\sim 1/f$:

$$
\lambda_{min}^{wave} = \frac{1}{\sqrt{\lambda_1}+10^{-5}}
$$

Độ dài hình chiếu mỗi trục Gaussian trên màn hình (từ `axes_2d`, mục 3):

$$
\ell_k = \sqrt{(u^{(k)})^2+(v^{(k)})^2+10^{-8}}
$$

**Chỉ số $\eta$ cho từng trục $k\in\{1,2,3\}$**:

$$
\boxed{\eta_k = \frac{\ell_k}{\lambda_{min}^{wave}}}
$$

Diễn giải: $\eta_k$ là tỉ số giữa kích thước Gaussian trên màn hình và bước sóng hoạ tiết cục bộ. $\eta_k>1$ nghĩa là Gaussian lớn hơn chi tiết ảnh cần tái tạo → alias (vi phạm Nyquist) → cần split; $\eta_k$ nhỏ → Gaussian quá mịn so với hoạ tiết → có thể dư thừa.

### 5.2. Chế độ `"projection"` (thay thế, chiếu trực tiếp structure tensor lên trục Gaussian, dòng 350–373)

```python
Sxx = sampled_S[:, 0].unsqueeze(1)  # [N, 1]
Sxy = sampled_S[:, 1].unsqueeze(1)  # [N, 1]
Syy = sampled_S[:, 2].unsqueeze(1)  # [N, 1]

u = axes_2d[:, :, 0]  # [N, 3] - X component of each axis
v = axes_2d[:, :, 1]  # [N, 3] - Y component of each axis

# Projection Frequency Violation:
# eta = axis^T @ S @ axis = Sxx*u² + 2*Sxy*u*v + Syy*v²
eta_3ch = Sxx * u**2 + 2 * Sxy * u * v + Syy * v**2  # [N, 3]

eta_3ch = torch.sqrt(eta_3ch)
```

$$
\eta_k = \sqrt{S_{xx}u_k^2 + 2S_{xy}u_k v_k + S_{yy}v_k^2}
$$

(dạng toàn phương $\sqrt{\mathbf{a}_k^\top S\, \mathbf{a}_k}$ với $\mathbf{a}_k=(u_k,v_k)$ là trục chiếu thứ $k$).

### 5.3. Trọng số theo độ truyền qua (transmittance weighting) (dòng 236–238, 378–380)

```python
weights_valid = torch.ones_like(max_transmittance)[is_active_vis] # [N_valid]
```

```python
# [NEW] Weight by Transmittance (Importance Sampling)
# If the Gaussian is transparent or occluded, we care less about its aliasing.
eta_3ch = eta_3ch * weights_valid.unsqueeze(1)
```

$$
\eta_k \leftarrow \eta_k \cdot w_i, \qquad w_i = 1\ (\text{mặc định, tất cả trọng số bằng 1 trong code hiện tại, xem dòng 238:}\ \texttt{torch.ones\_like(...)})
$$

---

## 6. Tích luỹ $\eta$ qua nhiều view (các buffer trong `GaussianModel`, dòng 382–415)

```python
eta_total = eta_3ch.sum(dim=1)

gaussians.accum_eta[valid_indices_global] += eta_total
gaussians.accum_view_count[valid_indices_global] += 1.0
gaussians.accum_weights_valid[valid_indices_global] += weights_valid

current_max = gaussians.max_eta_3ch[valid_indices_global]
gaussians.max_eta_3ch[valid_indices_global] = torch.max(current_max, eta_3ch)
```

Với mỗi Gaussian $i$ được quan sát ở view hiện tại (chỉ các điểm "active & visible"):

$$
\eta_i^{total} = \sum_{k=1}^{3}\eta_{i,k}
$$

**Tích luỹ tổng/đếm (để tính trung bình về sau, "leaky average")**:

$$
\mathrm{accum\_eta}_i \mathrel{+}= \eta_i^{total}, \qquad
\mathrm{accum\_view\_count}_i \mathrel{+}= 1, \qquad
\mathrm{accum\_weights\_valid}_i \mathrel{+}= w_i
$$

**Giá trị lớn nhất theo từng kênh/trục qua mọi view đã thấy** (dùng trực tiếp trong `densify_and_split_structgs`, `densify_and_prune_structgs` của `gaussian_model.py` dưới tên `max_eta_3ch`):

$$
\mathrm{max\_eta\_3ch}_i \leftarrow \max\big(\mathrm{max\_eta\_3ch}_i,\ \boldsymbol{\eta}_i\big) \quad (\text{theo từng trong 3 kênh})
$$

### 6.1. Ngưỡng phân loại high/mid/low (multiview consistency, dòng 394–415)

```python
TAU_HIGH = 1.0  # Frequency violation threshold
TAU_LOW = 0.1   # Background/smooth threshold

# Get max eta across 3 axes for classification
eta_max_scalar = eta_3ch.max(dim=1).values  # [N_valid]

# Classify each observation
is_high = eta_max_scalar > TAU_HIGH
is_low = eta_max_scalar <= TAU_LOW
is_mid = ~is_high & ~is_low

# Update counts
gaussians.eta_high_count[valid_indices_global[is_high]] += 1.0
gaussians.eta_mid_count[valid_indices_global[is_mid]] += 1.0
gaussians.eta_low_count[valid_indices_global[is_low]] += 1.0

# Accumulate sums for high and mid (for computing average later)
if is_high.any():
    gaussians.eta_high_sum_3ch[valid_indices_global[is_high]] += eta_3ch[is_high]
if is_mid.any():
    gaussians.eta_mid_sum_3ch[valid_indices_global[is_mid]] += eta_3ch[is_mid]
```

$$
\tau_{high}=1.0, \qquad \tau_{low}=0.1
$$

Lấy giá trị $\eta$ lớn nhất trong 3 trục của view hiện tại:

$$
\eta_i^{max} = \max_{k}\eta_{i,k}
$$

Phân loại:

$$
\text{is\_high}_i = \eta_i^{max} > \tau_{high}, \qquad
\text{is\_low}_i = \eta_i^{max} \le \tau_{low}, \qquad
\text{is\_mid}_i = \neg\text{is\_high}_i \wedge \neg\text{is\_low}_i
$$

Cập nhật bộ đếm số lần quan sát thuộc mỗi lớp:

$$
\mathrm{eta\_high\_count}_i \mathrel{+}=\mathbb 1[\text{is\_high}_i], \quad
\mathrm{eta\_mid\_count}_i \mathrel{+}=\mathbb 1[\text{is\_mid}_i], \quad
\mathrm{eta\_low\_count}_i \mathrel{+}=\mathbb 1[\text{is\_low}_i]
$$

Và tích luỹ tổng $\eta$ (theo 3 kênh) cho các lớp high/mid để tính trung bình sau này:

$$
\mathrm{eta\_high\_sum\_3ch}_i \mathrel{+}= \boldsymbol{\eta}_i\ [\text{nếu is\_high}_i], \qquad
\mathrm{eta\_mid\_sum\_3ch}_i \mathrel{+}= \boldsymbol{\eta}_i\ [\text{nếu is\_mid}_i]
$$

(không tích luỹ sum riêng cho lớp low — xác nhận trực tiếp từ code: không có dòng `eta_low_sum_3ch += ...` tương ứng ở khối `if`/`elif` cuối hàm, chỉ `eta_low_count` được cập nhật ở dòng 409.)

---

## 7. Liên hệ với `gaussian_model.py`

- $\eta$ tính trong mục 5 (`eta_3ch`, luỹ kế vào `max_eta_3ch`) chính là $\eta$ dùng trong công thức $k=\lceil\sqrt{\max(\eta,1)}\rceil$ ở mục 6.3 của `gaussian_model.md` (`densify_and_split_structgs`) và mục 6.5 (`densify_and_prune_structgs`).
- Công thức $\eta=(\sigma\omega_{max})^2$ nêu trong `gaussian_model.md` mục 6.1 là **diễn giải lý thuyết tổng quát** (lấy mẫu Nyquist: $\sigma$ = độ lệch chuẩn Gaussian, $\omega_{max}$ = tần số không gian cần tái tạo); công thức triển khai thực tế trong `freq_utils.py` xấp xỉ $\sigma\omega_{max}$ trực tiếp bằng tỉ số $\eta_k=\ell_k/\lambda_{min}^{wave}$ (độ dài trục chiếu / bước sóng tối thiểu) — về bản chất cùng là tỉ số "kích thước Gaussian" chia "kích thước chi tiết ảnh nhỏ nhất cần phân giải", không khai triển dạng bình phương $(\cdot)^2$ tường minh trong code hiện tại.
- Ngưỡng `expand_undersized_gs` ($\tau_{expand}$, mục 6.2 của `gaussian_model.md`) dùng cùng giá trị $\eta$ này nhưng theo chiều ngược lại (giá trị nhỏ ⇒ mở rộng).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Dòng | Công thức LaTeX |
|---|---|---|
| `cam_pos = -R.T @ t` | 47 | $\mathbf{p}_{cam} = -R^\top t$ |
| `distances = torch.minimum(distances, dists_to_current)` | 65 | $d_i \leftarrow \min(d_i, \lVert \mathbf{p}_i-\mathbf{p}_{c_k}\rVert_2)$ |
| `current_idx = torch.argmax(distances).item()` | 71 | $c_{k+1} = \arg\max_i d_i$ |
| `l1_loss = torch.mean(torch.abs(...), 0)` | 93 | $L_1 = \mathrm{mean}_{\text{kênh}}\lvert x-y\rvert$ |
| `(l1_loss - min)/(max-min)` | 94 | $\widehat{L_1} = (L_1-\min L_1)/(\max L_1-\min L_1)$ |
| `(1.0-0.2)*Ll1 + 0.2*(1-ssim)` | 101 | $\mathcal L_{photo}=0.8\lVert I-\hat I\rVert_1+0.2(1-\mathrm{SSIM})$ |
| `multiplier * (valid_value / torch.median(valid_value))` | 112 | $v^{norm}=c\cdot v/\mathrm{median}(v)$ |
| `vec_x = (means2D[:,0]-W*0.5)*(depths/fx)` | 134 | $x^{cam}=(u-W/2)\cdot z/f_x$ |
| `J_00 = fx*inv_z`, `J_02 = -fx*vec_x*inv_z2` | 144–145 | $J_{00}=f_x/z,\ J_{02}=-f_x x^{cam}/z^2$ |
| `R_total = R_view_batch @ R_local` (`torch.bmm`) | 160 | $R^{total}=R_{view}R(q)$ |
| `axes_cam = R_total * scales.unsqueeze(1)` | 165 | $A^{cam}=R^{total}\mathrm{diag}(s)$ |
| `u_vec = J_00*ax_x + J_02*ax_z` | 174 | $u^{(k)}=J_{00}a_x^{(k)}+J_{02}a_z^{(k)}$ |
| `L11 = sqrt(clamp(cov_xx, min=1e-6))` | 267 | $L_{11}=\sqrt{\sigma_{xx}}$ |
| `L21 = cov_xy / L11` | 269 | $L_{21}=\sigma_{xy}/L_{11}$ |
| `L22 = sqrt(clamp(cov_yy - L21**2, min=1e-6))` | 271 | $L_{22}=\sqrt{\sigma_{yy}-L_{21}^2}$ |
| `jitter_x = L11*eps1`, `jitter_y = L21*eps1+L22*eps2` | 273–274 | $\Delta u=L_{11}\epsilon_1,\ \Delta v=L_{21}\epsilon_1+L_{22}\epsilon_2$ |
| `norm_x = (sample_x/(w-1))*2-1` | 290 | $\hat u = u^{sample}/(W-1)\cdot2-1$ |
| `trace = raw_Sxx+raw_Syy` | 325 | $\mathrm{tr}=S_{xx}+S_{yy}$ |
| `det = raw_Sxx*raw_Syy - raw_Sxy**2` | 326 | $\det=S_{xx}S_{yy}-S_{xy}^2$ |
| `lambda1 = trace/2 + delta` | 327, 329 | $\lambda_1=\mathrm{tr}/2+\sqrt{\max((\mathrm{tr}/2)^2-\det,0)}$ |
| `wavelength_min = 1.0/(sqrt(lambda1)+1e-5)` | 334 | $\lambda_{min}^{wave}=1/(\sqrt{\lambda_1}+10^{-5})$ |
| `axis_lengths = sqrt(u_vec**2+v_vec**2+1e-8)` | 342 | $\ell_k=\sqrt{u_k^2+v_k^2+10^{-8}}$ |
| `eta_3ch = axis_lengths/wavelength_min.unsqueeze(1)` | 348 | $\eta_k = \ell_k/\lambda_{min}^{wave}$ |
| `eta_3ch = Sxx*u**2+2*Sxy*u*v+Syy*v**2` (mode projection) | 371 | $\eta_k^2 = S_{xx}u_k^2+2S_{xy}u_kv_k+S_{yy}v_k^2$ |
| `eta_3ch = torch.sqrt(eta_3ch)` | 373 | $\eta_k=\sqrt{\cdot}$ |
| `eta_3ch = eta_3ch * weights_valid.unsqueeze(1)` | 380 | $\eta_k\leftarrow\eta_k\cdot w_i$ |
| `eta_total = eta_3ch.sum(dim=1)` | 382 | $\eta_i^{total}=\sum_k\eta_{i,k}$ |
| `gaussians.accum_eta[...] += eta_total` | 384 | $\mathrm{accum\_eta}_i \mathrel{+}=\eta_i^{total}$ |
| `gaussians.accum_view_count[...] += 1.0` | 385 | $\mathrm{accum\_view\_count}_i\mathrel{+}=1$ |
| `gaussians.max_eta_3ch[...] = torch.max(current_max, eta_3ch)` | 392 | $\mathrm{max\_eta\_3ch}_i\leftarrow\max(\cdot,\boldsymbol\eta_i)$ |
| `TAU_HIGH = 1.0`, `TAU_LOW = 0.1` | 395–396 | $\tau_{high}=1.0,\ \tau_{low}=0.1$ |
| `eta_max_scalar = eta_3ch.max(dim=1).values` | 399 | $\eta_i^{max}=\max_k\eta_{i,k}$ |
| `is_high = eta_max_scalar > TAU_HIGH` | 402 | $\text{is\_high}_i=\eta_i^{max}>\tau_{high}$ |
| `is_low = eta_max_scalar <= TAU_LOW` | 403 | $\text{is\_low}_i=\eta_i^{max}\le\tau_{low}$ |
| `is_mid = ~is_high & ~is_low` | 404 | $\text{is\_mid}_i=\neg\text{high}\wedge\neg\text{low}$ |
| `gaussians.eta_high_count[...] += 1.0` | 407 | $\mathrm{eta\_high\_count}_i\mathrel{+}=\mathbb1[\text{is\_high}_i]$ |
| `gaussians.eta_high_sum_3ch[...] += eta_3ch[is_high]` | 413 | $\mathrm{eta\_high\_sum\_3ch}_i\mathrel{+}=\boldsymbol\eta_i$ |

---

## Kiểm chứng tính đúng sai

Đã đọc lại thật toàn bộ `utils/freq_utils.py` (419 dòng) và đối chiếu từng công thức đã có trong bản cũ với code thật:

- Tất cả công thức ($\mathbf p_{cam}$, FPS, $L_1$/SSIM photometric loss, `normalize`, Jacobian phối cảnh, ma trận xoay tổng hợp, phân rã Cholesky, $\lambda_1$, $\eta_k$ ở hai chế độ, tích luỹ accum/max, phân loại high/mid/low) đều khớp chính xác với code thật — không phát hiện sai sót toán học nào.
- Bổ sung so với bản cũ: nêu rõ các hằng số ổn định số học không có trong công thức giải tích gốc nhưng có trong code thật — $\epsilon=10^{-7}$ khi nghịch đảo depth (dòng 141), `clamp(min=1e-6)` trước khi lấy căn trong phân rã Cholesky (dòng 267, 271), và xác nhận rằng lớp "low" không có sum riêng (chỉ có count, dòng 409) — khớp với chú thích cũ "(không tích luỹ sum riêng cho lớp low...)" nhưng nay có trích dẫn số dòng cụ thể để kiểm chứng thay vì chỉ nêu bằng lời.
- Hệ số $0.8$ trong $\mathcal L_{photo}$ viết trong code dưới dạng biểu thức `(1.0 - 0.2)` (dòng 101) chứ không phải hằng số `0.8` trực tiếp — về giá trị số học tương đương, đã ghi chú rõ.
