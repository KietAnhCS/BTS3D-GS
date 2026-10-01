# Công thức toán học trong `gaussian_renderer/__init__.py`

Tài liệu mô tả cơ sở toán học của hàm `render_structgs`, hàm duy nhất trong file này. Hàm này **không tự thực hiện** phép chiếu phối cảnh hay alpha compositing — nó chỉ **chuẩn bị tham số đầu vào** rồi gọi rasterizer CUDA (`diff_gaussian_rasterization_structgs.GaussianRasterizer`), nơi thực sự thực hiện rasterization vi phân (projection, sorting theo depth, alpha blending). Do đó tài liệu tập trung vào các phép biến đổi/chuẩn bị tham số diễn ra trong Python, và chỉ nêu công thức rasterization ở mức tham chiếu (không được thực thi trong file này).

---

## 1. Tensor giữ gradient không gian màn hình (`screenspace_points`)

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 27):

```python
screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
```

Đây là một tensor "giả" (zero) có kích thước $N\times 4$ (N = số điểm Gaussian), dùng như điểm neo để lấy gradient của phép chiếu 2D ngược về. Về mặt toán học, đây là biến trung gian $y_i \in \mathbb{R}^4$ với:

$$
y_i = \mathbf{0}, \qquad \frac{\partial \mathcal{L}}{\partial \mu_i^{2D}} \ \text{được rasterizer cộng dồn vào } y_i \ \text{qua } y_i.\text{grad}
$$

nhờ cơ chế autograd của PyTorch: $y_i$ được cộng "ảo" vào toạ độ màn hình bên trong rasterizer CUDA, nên đạo hàm ngược (backward) của toạ độ màn hình sẽ chảy vào `screenspace_points.grad`.

---

## 2. Chuyển FOV sang $\tan(\text{fov}/2)$

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 34–35):

```python
tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
```

$$
\tan_x = \tan\!\left(\frac{\mathrm{FoV}_x}{2}\right), \qquad \tan_y = \tan\!\left(\frac{\mathrm{FoV}_y}{2}\right)
$$

Hai giá trị này được truyền thẳng vào `GaussianRasterizationSettings` và dùng bên trong CUDA kernel để dựng ma trận chiếu phối cảnh (projection matrix) dạng OpenGL chuẩn — phần dựng ma trận này **không nằm trong file Python**, chỉ có bước quy đổi góc nhìn sang hệ số tan ở đây.

---

## 3. Bản đồ chỉ số (`metric_map`) mặc định

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 37–38) — lưu ý code thật kiểm tra `metric_map==None` và dùng `image_height`/`image_width` của chính camera, không phải biến `H, W` tổng quát:

```python
if metric_map==None:
    metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')
```

Nếu không truyền `metric_map` từ ngoài, khởi tạo một vector không có độ dài $H\times W$ (mỗi pixel một chỉ số mặc định $= 0$), với $H=$ `image_height`, $W=$ `image_width`:

$$
\text{metric\_map}_{(u,v)} = 0, \qquad \forall (u,v) \in [0,W)\times[0,H)
$$

---

## 4. Tập tham số rasterization (`GaussianRasterizationSettings`)

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 40–57):

```python
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map,
        compute_extra=compute_extra
    )
```

Toàn bộ các tham số hình học của cảnh được đóng gói, không có phép biến đổi số học nào xảy ra ở bước này ngoài việc truyền trực tiếp:

- $V$ = `viewmatrix` $=$ `world_view_transform` (ma trận view $4\times4$, do `scene.cameras` tạo sẵn)
- $P$ = `projmatrix` $=$ `full_proj_transform` (ma trận kết hợp view–projection $4\times4$, cũng tạo sẵn ngoài file này)
- $\mathbf{p}_{cam}$ = `campos` $=$ `camera_center`
- $m$ = `scale_modifier` $=$ `scaling_modifier` (hệ số nhân tỉ lệ Gaussian, mặc định $1.0$)
- $\ell$ = `sh_degree` $=$ `pc.active_sh_degree` (bậc Spherical Harmonics hiện dùng — tăng dần theo lịch huấn luyện, được quản lý bởi `GaussianModel`, không phải file này)

Những tham số này sẽ được rasterizer CUDA dùng để tính:

$$
\mathbf{x}^{clip}_i = P\, \big(V\, [\mu_i, 1]^\top\big)
$$

(phép chiếu đồng nhất phối cảnh) — **bước này chạy trong CUDA**, file Python chỉ cấp đầu vào $V, P$.

---

## 5. Opacity hiệu dụng có lọc 3D

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 63):

```python
opacity = pc.get_opacity_with_3D_filter
```

$$
\alpha_i^{filter} = \alpha_i \cdot \kappa_i
$$

(công thức $\kappa_i$ được định nghĩa và tính trong `scene/gaussian_model.py`, mục "Opacity hiệu dụng có lọc"; file này chỉ gọi thuộc tính đã tính sẵn).

---

## 6. Covariance: tiền tính Python hoặc để rasterizer tính

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 71–75):

```python
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation
```

Hai nhánh loại trừ nhau:

- **Nhánh Python**: tiền tính trực tiếp $\Sigma_i = R_i S_i S_i^\top R_i^\top$ (với $m=$ `scaling_modifier` nhân vào $S_i$ — xem `gaussian_model.md` mục 1.3), truyền thẳng ma trận hiệp phương sai $6$ giá trị tam giác trên vào rasterizer.
- **Nhánh CUDA**: chỉ truyền $\tilde{s}_i$ (tỉ lệ có lọc 3D) và $q_i$ (quaternion đã chuẩn hoá), rasterizer tự dựng $\Sigma_i$ bên trong kernel bằng đúng công thức trên, nhân thêm $m$ = `scale_modifier` lúc dựng $S_i$.

$$
\tilde{s}_i = \sqrt{s_i^2 + (r_i^{filter})^2} \quad (\text{xem mục 2.2 của } \texttt{gaussian\_model.md})
$$

---

## 7. Màu sắc: SH tiền tính trong Python hoặc để rasterizer tính

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 82–89):

```python
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc, shs = pc.get_features_dc, pc.get_features_rest
```

**Hướng nhìn chuẩn hoá** (view direction) từ mỗi Gaussian đến camera (dòng 84–85):

$$
\mathbf{d}_i = \mu_i - \mathbf{p}_{cam}, \qquad \hat{\mathbf{d}}_i = \frac{\mathbf{d}_i}{\lVert \mathbf{d}_i \rVert_2}
$$

**Đánh giá hàm cầu điều hoà** (Spherical Harmonics) tới bậc $\ell_{max}=$ `active_sh_degree` (dòng 86):

$$
\mathrm{RGB}_i^{raw} = \sum_{\ell=0}^{\ell_{max}} \sum_{m=-\ell}^{\ell} c_{i,\ell m}\, Y_\ell^m(\hat{\mathbf{d}}_i)
$$

(hàm cơ sở $Y_\ell^m$ và việc đánh giá cụ thể nằm trong `utils.sh_utils.eval_sh`, không thuộc file này).

**Dịch về khoảng màu hợp lệ** (SH được huấn luyện quanh gốc $0$, DC offset $0.5$ tương ứng màu xám trung bình, dòng 87):

$$
\mathrm{RGB}_i = \max\big(\mathrm{RGB}_i^{raw} + 0.5,\ 0\big)
$$

Nếu `convert_SHs_python = False`, hệ số SH thô (`dc`, `shs`) được truyền thẳng vào rasterizer để CUDA tự thực hiện đúng hai công thức trên.

---

## 8. Alpha compositing (thực hiện trong CUDA, không trong file này)

Rasterizer trả về ảnh render bằng công thức tích hợp thể tích (volume rendering) chuẩn của 3D Gaussian Splatting — các Gaussian được sắp theo độ sâu và hoà trộn theo thứ tự front-to-back:

$$
C(u,v) = \sum_{i=1}^{N} c_i\, \alpha_i^{2D}(u,v) \prod_{j=1}^{i-1}\big(1-\alpha_j^{2D}(u,v)\big)
$$

trong đó $\alpha_i^{2D}(u,v)$ là opacity 2D của Gaussian tại pixel $(u,v)$, suy từ $\alpha_i^{filter}$ (mục 5) và hình chiếu 2D của $\Sigma_i$ (mục 6). **File `gaussian_renderer/__init__.py` chỉ cung cấp $\mu_i$, $\Sigma_i$ (hoặc $s_i,q_i$), $\alpha_i^{filter}$, $c_i$ (hoặc hệ số SH) làm đầu vào; công thức này được thực thi hoàn toàn bên trong rasterizer CUDA.**

---

## 9. Lọc Gaussian hiển thị (`visibility_filter`)

Trích nguyên văn (`gaussian_renderer/__init__.py`, dòng 120):

```python
            "visibility_filter" : (radii > 0).nonzero(),
```

$$
\text{visible}_i \iff r_i^{2D} > 0
$$

($r_i^{2D}$ = bán kính chiếu lên màn hình do rasterizer trả về — Gaussian bị frustum-cull hoặc chiếu ra bán kính 0 bị loại khỏi thống kê gradient densification).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `screenspace_points = torch.zeros((N,4), requires_grad=True) + 0` | $y_i = \mathbf{0},\ \dfrac{\partial \mathcal{L}}{\partial \mu_i^{2D}} \to y_i.\text{grad}$ |
| `tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)` | $\tan_x = \tan(\mathrm{FoV}_x/2)$ |
| `tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)` | $\tan_y = \tan(\mathrm{FoV}_y/2)$ |
| `metric_map = torch.zeros(H*W, dtype=torch.int)` | $\text{metric\_map}_{(u,v)} = 0$ |
| `viewmatrix=viewpoint_camera.world_view_transform` | $V$ (ma trận view $4\times4$) |
| `projmatrix=viewpoint_camera.full_proj_transform` | $P$ (ma trận view–projection $4\times4$) |
| `campos=viewpoint_camera.camera_center` | $\mathbf{p}_{cam}$ |
| `scale_modifier=scaling_modifier` | $m$ |
| `sh_degree=pc.active_sh_degree` | $\ell_{max}$ |
| `opacity = pc.get_opacity_with_3D_filter` | $\alpha_i^{filter} = \alpha_i \cdot \kappa_i$ |
| `cov3D_precomp = pc.get_covariance(scaling_modifier)` | $\Sigma_i = R_i S_i S_i^\top R_i^\top,\ S_i=\mathrm{diag}(m\cdot s_i)$ |
| `scales = pc.get_scaling_with_3D_filter` | $\tilde{s}_i = \sqrt{s_i^2+(r_i^{filter})^2}$ |
| `rotations = pc.get_rotation` | $q_i = q_i^{raw}/\lVert q_i^{raw}\rVert_2$ |
| `dir_pp = pc.get_xyz - camera_center` | $\mathbf{d}_i = \mu_i - \mathbf{p}_{cam}$ |
| `dir_pp_normalized = dir_pp/dir_pp.norm(dim=1)` | $\hat{\mathbf{d}}_i = \mathbf{d}_i/\lVert \mathbf{d}_i\rVert_2$ |
| `sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)` | $\mathrm{RGB}_i^{raw} = \sum_{\ell,m} c_{i,\ell m} Y_\ell^m(\hat{\mathbf{d}}_i)$ |
| `colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)` | $\mathrm{RGB}_i = \max(\mathrm{RGB}_i^{raw}+0.5,\ 0)$ |
| `rendered_image, radii, ... = rasterizer(...)` (CUDA) | $C(u,v) = \sum_i c_i \alpha_i^{2D} \prod_{j<i}(1-\alpha_j^{2D})$ |
| `"visibility_filter": (radii > 0).nonzero()` | $\text{visible}_i \iff r_i^{2D} > 0$ |
