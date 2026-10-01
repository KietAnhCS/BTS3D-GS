# Công thức toán học của `diff_gaussian_rasterization_structgs/__init__.py`

File này là lớp wrapper Python giữa PyTorch autograd và backend CUDA (`_C`, biên dịch từ `rasterize_points.cu`). Bản thân file gần như không có công thức toán học *mới* — nhiệm vụ của nó là (1) đóng gói tham số hình học camera thành `GaussianRasterizationSettings`, (2) định nghĩa `_RasterizeGaussians` — một `torch.autograd.Function` tuỳ biến nối forward/backward CUDA vào đồ thị tính gradient của PyTorch, và (3) cung cấp `markVisible` — wrapper frustum culling. Tài liệu này diễn giải chính xác các công thức ẩn trong cách truyền/nhận tham số, và **kiểm tra tính nhất quán** giữa `forward` và `backward` (số lượng/thứ tự input–output), vì đây là nơi rất dễ phát sinh lỗi runtime khi sửa code.

Tham khảo thêm (không trích lại công thức): `MATH/cuda/rasterize_points.md` (glue C++↔CUDA), `MATH/cuda/forward.md` (toàn bộ toán của lượt forward CUDA), `MATH/cuda/backward.md` (toán của lượt backward CUDA).

---

## Ký hiệu

- $P$: số lượng Gaussian (points). $M$: số hệ số SH bậc $>0$ mỗi kênh màu (`sh.size(1)`, $=0$ nếu dùng `colors_precomp`).
- $H,W$: chiều cao/rộng ảnh (pixel). $f_x,f_y$: tiêu cự theo pixel.
- $\mathrm{fov}_x,\mathrm{fov}_y$: trường nhìn ngang/dọc (radian).
- $W_{view}\in\mathbb R^{4\times4}$: ma trận view (world→camera), $P_{full}=P_{proj}W_{view}\in\mathbb R^{4\times4}$: ma trận chiếu đầy đủ (projection $\circ$ view), tương ứng `viewmatrix`, `projmatrix` trong code.
- $\mathbf o_{cam}\in\mathbb R^3$: tâm camera trong world-space (`campos`).
- $\mu_i\in\mathbb R^3$: tâm Gaussian thứ $i$ (`means3D`). $\hat\mu_i\in\mathbb R^2$: tâm chiếu lên ảnh — về mặt toán là output, nhưng Python truyền vào một tensor **placeholder** cùng tên `means2D` (xem Bước 3).
- $s_i\in\mathbb R^3$: scale (`scales`), $q_i\in\mathbb R^4$: quaternion xoay (`rotations`), $\Sigma_{3D,i}\in\mathbb R^{3\times3}$: hiệp phương sai 3D (`cov3D_precomp`, tuỳ chọn).
- $\mathrm{sh}_i$: hệ số spherical harmonics bậc $\ge1$ (`sh`), $\mathrm{sh}_{dc,i}$: hệ số DC (`dc`, tách riêng), $\alpha_i$: opacity.
- $\mathcal L$: hàm mất mát cuối cùng (ảnh render so với ground-truth). $\nabla_x\mathcal L$ viết tắt `grad_x`.

---

## Bước 1: `rasterize_gaussians` (dòng 21–44) — hàm cổng vào

```python
def rasterize_gaussians(
    means3D, means2D, dc, sh, colors_precomp, opacities,
    scales, rotations, cov3Ds_precomp, raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D, means2D, dc, sh, colors_precomp, opacities,
        scales, rotations, cov3Ds_precomp, raster_settings,
    )
```

Đây chỉ là forwarding tới `torch.autograd.Function.apply`. Không có công thức; điểm cần ghi nhận là **thứ tự 10 tham số** — thứ tự này được dùng lại nguyên vẹn làm chữ ký của `_RasterizeGaussians.forward` (Bước 3) và do đó định nghĩa luôn thứ tự bắt buộc của gradient trả về ở `backward` (Bước 4).

---

## Bước 2: `GaussianRasterizationSettings` (dòng 177–193) — tham số camera

```python
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    mult : float
    prefiltered : bool
    debug : bool
    get_flag : bool
    metric_map : torch.Tensor
    compute_extra : bool = False
```

Đây là một `NamedTuple` nên bản thân nó không chứa phép tính — nhưng mỗi trường mang một quan hệ hình học camera cụ thể, được tính ở phía gọi (`gaussian_renderer/__init__.py`, không thuộc file này) rồi mới truyền vào:

$$
\tan\!\Big(\frac{\mathrm{fov}_x}{2}\Big)=\texttt{tanfovx},\qquad
\tan\!\Big(\frac{\mathrm{fov}_y}{2}\Big)=\texttt{tanfovy}
$$

Đây đúng là quan hệ $\tan(\mathrm{fov}/2)=W/(2f)$ (tiêu cự $f$ theo pixel, $W$ là nửa kích thước ảnh theo trục tương ứng), vì $f_x=\dfrac{W}{2\tan(\mathrm{fov}_x/2)}$. `tanfovx,tanfovy` là tham số **duy nhất** mang thông tin tiêu cự truyền sang CUDA — $f_x,f_y$ thực tế được backend suy ngược lại từ $W,H,\texttt{tanfovx},\texttt{tanfovy}$ (xem `forward.cu`, không lặp lại ở đây).

`viewmatrix` $=W_{view}$ (world→camera, hàng cuối $(0,0,0,1)$), `projmatrix` $=P_{full}=P_{proj}W_{view}$ (đã nhân sẵn phép chiếu phối cảnh vào phép đổi hệ quy chiếu — đây chính là ma trận dùng trong frustum culling, Bước 5). `campos` $=\mathbf o_{cam}$, tâm camera trong world-space, dùng để tính hướng nhìn $\mathbf d=(\mu_i-\mathbf o_{cam})/\lVert\mu_i-\mathbf o_{cam}\rVert$ cho SH (xem `forward.md` mục 2). `bg`: màu nền $\mathbf{bg}\in\mathbb R^3$ dùng ở alpha-compositing cuối. `scale_modifier` $=m$: hệ số nhân vào scale, $S=\mathrm{diag}(m s_x,m s_y,m s_z)$. `mult`: hệ số mở rộng ngưỡng số tile chạm tới (xem `forward.md` mục 7). `sh_degree`: bậc SH dùng thực tế $\deg\le 3$. `get_flag`, `debug`, `prefiltered`, `compute_extra`: cờ điều khiển không mang công thức.

---

## Bước 3: `_RasterizeGaussians.forward` (dòng 46–113)

```python
@staticmethod
def forward(ctx, means3D, means2D, dc, sh, colors_precomp, opacities,
            scales, rotations, cov3Ds_precomp, raster_settings):
    get_flag = raster_settings.get_flag
    if get_flag == None:
        get_flag = False

    args = (
        raster_settings.bg, means3D, colors_precomp, opacities, scales, rotations,
        raster_settings.scale_modifier, cov3Ds_precomp, raster_settings.metric_map,
        raster_settings.viewmatrix, raster_settings.projmatrix,
        raster_settings.tanfovx, raster_settings.tanfovy,
        raster_settings.image_height, raster_settings.image_width,
        dc, sh, raster_settings.sh_degree, raster_settings.campos,
        raster_settings.mult, raster_settings.prefiltered, raster_settings.debug,
        get_flag, raster_settings.compute_extra
    )

    num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer, \
        sampleBuffer, accum_metric_counts, cov2D, depth_map, opacity_map, normal_map \
        = _C.rasterize_gaussians(*args)

    ctx.raster_settings = raster_settings
    ctx.num_rendered = num_rendered
    ctx.num_buckets = num_buckets
    ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
                           radii, dc, sh, geomBuffer, binningBuffer, imgBuffer,
                           sampleBuffer, cov2D)
    return color, radii, accum_metric_counts, cov2D, depth_map, opacity_map, normal_map
```

**Thứ tự 10 input** (không tính `ctx`), đánh số để đối chiếu với `backward`:

$$
(1)\ \mu\ (2)\ \hat\mu_{\text{ph}}\ (3)\ \mathrm{sh}_{dc}\ (4)\ \mathrm{sh}\ (5)\ c_{\text{precomp}}\ (6)\ \alpha\ (7)\ s\ (8)\ q\ (9)\ \Sigma_{3D,\text{precomp}}\ (10)\ \texttt{raster\_settings}
$$

**Ý nghĩa `means2D` (input thứ 2).** Nó *không* được truyền vào `args` gửi cho `_C.rasterize_gaussians` — CUDA tự tính tâm chiếu $\hat\mu_i$ từ $\mu_i$ nội bộ. `means2D` ở đây chỉ là một tensor "placeholder" cùng shape $\hat\mu$ (do phía gọi tạo bằng `torch.zeros(..., requires_grad=True)`, xem `gaussian_renderer/__init__.py` dòng 27), với mục đích **neo node đồ thị autograd**: vì `_RasterizeGaussians` là `Function` tuỳ biến, gradient $\partial\mathcal L/\partial \hat\mu$ (độ nhạy của loss theo vị trí 2D trên ảnh — đại lượng dùng để quyết định densify) chỉ có thể "thoát ra ngoài" qua `backward` nếu có một input tương ứng nhận gradient đó; `means2D` chính là input đóng vai trò nhận `grad_means2D$ dù giá trị của nó không dùng ở forward.

**Gọi CUDA.** $(\,\text{color},\text{radii},\ldots)=\_C.\text{rasterize\_gaussians}(\text{bg},\mu,c_{\text{precomp}},\alpha,s,q,m,\Sigma_{3D,\text{precomp}},\text{metric\_map},W_{view},P_{full},\tan\tfrac{\mathrm{fov}_x}2,\tan\tfrac{\mathrm{fov}_y}2,H,W,\mathrm{sh}_{dc},\mathrm{sh},\deg,\mathbf o_{cam},\text{mult},\text{prefiltered},\text{debug},\text{get\_flag},\text{compute\_extra})$ — công thức bên trong đã mô tả đầy đủ ở `MATH/cuda/forward.md`.

**Lưu cho backward.** `ctx.save_for_backward` lưu đúng 13 tensor: $(c_{\text{precomp}},\mu,s,q,\Sigma_{3D,\text{precomp}},\text{radii},\mathrm{sh}_{dc},\mathrm{sh},\text{geomBuffer},\text{binningBuffer},\text{imgBuffer},\text{sampleBuffer},\text{cov2D})$; cộng thêm `ctx.raster_settings`, `ctx.num_rendered`, `ctx.num_buckets` lưu trực tiếp làm thuộc tính (không qua cơ chế `save_for_backward` vì không phải tensor cần theo dõi gradient).

**7 output** theo đúng thứ tự: $(\text{color},\text{radii},\text{accum\_metric\_counts},\text{cov2D},\text{depth\_map},\text{opacity\_map},\text{normal\_map})$.

---

## Bước 4: `_RasterizeGaussians.backward` (dòng 115–175)

```python
@staticmethod
def backward(ctx, grad_out_color, _, g_metric, _cov2D, _depth_map, _opacity_map, _normal_map):
    num_rendered = ctx.num_rendered
    num_buckets = ctx.num_buckets
    raster_settings = ctx.raster_settings
    colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, \
        geomBuffer, binningBuffer, imgBuffer, sampleBuffer, cov2D = ctx.saved_tensors

    args = (raster_settings.bg, means3D, radii, colors_precomp, scales, rotations,
            raster_settings.scale_modifier, cov3Ds_precomp,
            raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            grad_out_color, dc, sh, raster_settings.sh_degree, raster_settings.campos,
            geomBuffer, num_rendered, binningBuffer, imgBuffer, num_buckets,
            sampleBuffer, raster_settings.debug)

    grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, \
        grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

    grads = (
        grad_means3D, grad_means2D, grad_dc, grad_sh, grad_colors_precomp,
        grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, None,
    )
    return grads
```

**Chữ ký nhận gradient — 7 tham số** (khớp 7 output của forward, theo đúng thứ tự):

$$
\underbrace{\texttt{grad\_out\_color}}_{\nabla_{\text{color}}\mathcal L}\ \ \underbrace{\_}_{\nabla_{\text{radii}}\ (\text{bỏ})}\ \ \underbrace{\texttt{g\_metric}}_{\nabla_{\text{accum\_metric\_counts}}\ (\text{bỏ})}\ \ \underbrace{\texttt{\_cov2D}}_{(\text{bỏ})}\ \ \underbrace{\texttt{\_depth\_map}}_{(\text{bỏ})}\ \ \underbrace{\texttt{\_opacity\_map}}_{(\text{bỏ})}\ \ \underbrace{\texttt{\_normal\_map}}_{(\text{bỏ})}
$$

Chỉ `grad_out_color` $=\partial\mathcal L/\partial\,\text{color}$ được dùng tiếp — tất cả các gradient khác (radii, accum_metric_counts, cov2D, depth_map, opacity_map, normal_map) đều bị **vứt bỏ** (gán cho `_` hoặc biến có tiền tố `_` không dùng lại). Về mặt toán học, điều này có nghĩa: nếu một đoạn code khác lan truyền loss qua `depth_map` hay `normal_map` (ví dụ loss giám sát độ sâu), gradient đó **không bao giờ chảy ngược được** vào $\mu,s,q,\ldots$ qua `_RasterizeGaussians` này — nó bị cắt đứt tại đây, bất kể `compute_extra=True` ở forward. Đây là một giới hạn/khả năng-bug quan trọng của wrapper, không phải lỗi cú pháp, nên nêu rõ ở mục kiểm chứng.

**Gọi CUDA backward**, nhận về đúng 9 gradient tensor, theo thứ tự **riêng của C++** (không trùng thứ tự input của forward):

$$
(\text{grad\_means2D},\ \text{grad\_colors\_precomp},\ \text{grad\_opacities},\ \text{grad\_means3D},\ \text{grad\_cov3Ds\_precomp},\ \text{grad\_dc},\ \text{grad\_sh},\ \text{grad\_scales},\ \text{grad\_rotations})
$$

**Sắp lại (`grads`) theo đúng thứ tự 10 input của `forward`** (xem Bước 3) để trả cho PyTorch autograd — PyTorch yêu cầu `backward` trả về đúng số lượng gradient, đúng vị trí, khớp 1-1 với danh sách input của `forward`:

$$
\begin{array}{c|c}
\text{Input forward (thứ tự)} & \text{Output backward (thứ tự)}\\\hline
(1)\ \mu & \text{grad\_means3D}\\
(2)\ \hat\mu_{\text{ph}} & \text{grad\_means2D}\\
(3)\ \mathrm{sh}_{dc} & \text{grad\_dc}\\
(4)\ \mathrm{sh} & \text{grad\_sh}\\
(5)\ c_{\text{precomp}} & \text{grad\_colors\_precomp}\\
(6)\ \alpha & \text{grad\_opacities}\\
(7)\ s & \text{grad\_scales}\\
(8)\ q & \text{grad\_rotations}\\
(9)\ \Sigma_{3D,\text{precomp}} & \text{grad\_cov3Ds\_precomp}\\
(10)\ \texttt{raster\_settings} & \texttt{None}
\end{array}
$$

`raster_settings` không phải tensor nên không có gradient — `None` ở vị trí thứ 10 là bắt buộc để PyTorch không báo lỗi "số gradient không khớp số input".

---

## Bước 5: `markVisible` (dòng 200–209) — frustum culling

```python
def markVisible(self, positions):
    with torch.no_grad():
        raster_settings = self.raster_settings
        visible = _C.mark_visible(
            positions,
            raster_settings.viewmatrix,
            raster_settings.projmatrix)
    return visible
```

Hàm Python chỉ chuyển tiếp `positions` ($\mu\in\mathbb R^{P\times3}$), `viewmatrix` ($W_{view}$), `projmatrix` ($P_{full}$) cho `_C.mark_visible`, bọc trong `torch.no_grad()` (không có gradient — culling là phép rời rạc). Backend là `CudaRasterizer::Rasterizer::markVisible` → kernel `checkFrustum` (`rasterizer_impl.cu:105`) → hàm `in_frustum` (`auxiliary.h:152`), gọi với `prefiltered=false`. Công thức thật sự thực thi cho mỗi điểm $i$:

$$
p_{\text{hom}} = P_{full}\,\tilde\mu_i \in\mathbb R^4,\qquad \tilde\mu_i=(\mu_{i,x},\mu_{i,y},\mu_{i,z},1)^\top
$$

$$
p_{\text{proj}} = \Big(\frac{p_{\text{hom},x}}{p_{\text{hom},w}+\varepsilon},\ \frac{p_{\text{hom},y}}{p_{\text{hom},w}+\varepsilon},\ \frac{p_{\text{hom},z}}{p_{\text{hom},w}+\varepsilon}\Big),\qquad \varepsilon=10^{-7}
$$

$$
p_{\text{view}} = W_{view}\,\tilde\mu_i\ \ (\text{3 thành phần affine, bỏ hàng } w)
$$

$$
\boxed{\ \text{visible}_i = \big[\,p_{\text{view},z}>0.2\,\big]\ }
$$

**Điểm cần nêu rõ — dòng code bị comment:**

```cpp
if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
```

Điều kiện đầy đủ đáng lẽ phải là near-plane culling **và** culling theo biên clip-space mở rộng $1.3\times$ (cùng hệ số $1.3$ dùng ở `computeCov2D`, `forward.md` mục 4):

$$
\text{visible}_i \overset{?}{=} \big[p_{\text{view},z}>0.2\big]\ \wedge\ \big[-1.3\le p_{\text{proj},x}\le1.3\big]\ \wedge\ \big[-1.3\le p_{\text{proj},y}\le1.3\big]
$$

nhưng vế sau **đã bị comment ra**, nên điều kiện thực thi chỉ còn lại $p_{\text{view},z}>0.2$ — tức **chỉ near-plane culling** (loại điểm ở sau hoặc quá gần camera), **không** loại điểm nằm ngoài trường nhìn ngang/dọc dù `projmatrix` vẫn được tính và truyền vào. Xem mục "Kiểm chứng" bên dưới.

---

## Kiến thức toán nền tảng

**1. Custom `torch.autograd.Function`.** PyTorch cho phép định nghĩa một node trong đồ thị tính gradient bằng cặp `forward`/`backward` tĩnh thay vì ghép từ các phép toán nguyên thuỷ đã có sẵn autograd. Hợp đồng bắt buộc:
- `forward(ctx, *inputs) -> outputs` (có thể trả tuple nhiều tensor).
- `backward(ctx, *grad_outputs) -> grad_inputs`, trong đó `grad_outputs[j] = ∂L/∂outputs[j]` do PyTorch tự truyền vào (được tính bởi node phía sau trong đồ thị), và hàm phải trả `grad_inputs[k] = ∂L/∂inputs[k]` — **đúng số lượng, đúng thứ tự** với danh sách `inputs` của `forward` (không tính `ctx`). Input nào không cần gradient (hằng số, cấu hình như `raster_settings`) thì trả `None` ở đúng vị trí.
- `ctx.save_for_backward(*tensors)` lưu các tensor cần cho backward mà không bị PyTorch theo dõi như một phần đồ thị (tránh giữ graph không cần thiết); `ctx.saved_tensors` đọc lại đúng thứ tự đã lưu. Thuộc tính thường (`ctx.raster_settings`, `ctx.num_rendered`) dùng cho dữ liệu không phải tensor.
- Về bản chất đây là áp dụng quy tắc chuỗi (chain rule) "thủ công": $\dfrac{\partial\mathcal L}{\partial \text{input}_k}=\sum_j \dfrac{\partial\mathcal L}{\partial\text{output}_j}\cdot\dfrac{\partial\,\text{output}_j}{\partial\,\text{input}_k}$, với vế phải được CUDA backend tính giải tích (`backward.cu`) thay vì autograd tự động vi phân từng phép toán.

**2. Hình học camera — clip space & FOV.** Một điểm world $\mu$ được đưa về clip space qua $p_{\text{hom}}=P_{full}\,\tilde\mu=(P_{proj}W_{view})\tilde\mu$; chia phối cảnh $p_{\text{ndc}}=p_{\text{hom}}/p_{\text{hom},w}$ đưa về $[-1,1]^3$ (NDC) nếu điểm nằm trong frustum. Trường nhìn $\mathrm{fov}$ liên hệ với tiêu cự pixel qua $\tan(\mathrm{fov}/2)=\dfrac{\text{kích thước nửa sensor (pixel)}}{f}$ — đây là lý do `tanfovx`,`tanfovy` đủ để backend suy ra $f_x,f_y$ khi biết $W,H$.

**3. Frustum culling.** Mục đích là loại sớm các điểm chắc chắn không xuất hiện trên ảnh (phía sau camera, sau far plane, hoặc lệch quá xa khỏi góc nhìn ngang/dọc) trước khi tốn chi phí rasterize. Dạng đầy đủ thường kiểm tra điểm có nằm trong *lục diện frustum* hay không bằng 6 mặt phẳng (near/far/left/right/top/bottom), ở đây rút gọn/gần đúng bằng kiểm tra trên $p_{\text{view},z}$ (near plane, không far plane) và $p_{\text{proj},x},p_{\text{proj},y}$ (left/right/top/bottom, mở rộng biên $1.3\times$ để dung sai Jacobian affine ở rìa ảnh — xem `forward.md` mục 4).

---

## Kiểm chứng tính đúng sai

**1. Số lượng & thứ tự gradient trả về của `backward` so với input của `forward`.**

`forward` nhận 10 tham số (không kể `ctx`): `means3D, means2D, dc, sh, colors_precomp, opacities, scales, rotations, cov3Ds_precomp, raster_settings`.

`backward` trả `grads = (grad_means3D, grad_means2D, grad_dc, grad_sh, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp, None)` — đếm được **đúng 10 phần tử**, và theo bảng đối chiếu ở Bước 4, **từng vị trí khớp đúng tên biến tương ứng** của `forward` (vị trí 1↔1 `means3D`/`grad_means3D`, …, vị trí 10↔10 `raster_settings`/`None`). $\Rightarrow$ **ĐÚNG**, không lệch thứ tự — đây là điểm dễ sai nhất khi sửa code (ví dụ hoán đổi `scales`/`rotations`) nhưng ở bản hiện tại nó nhất quán.

Lưu ý: `_C.rasterize_gaussians_backward` (hàm CUDA thô) trả 9 giá trị theo một **thứ tự khác** (`grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations`) — Python phải **sắp xếp lại thủ công** thành `grads` ở trên. Đếm: 9 giá trị CUDA trả về $+$ 1 `None` (cho `raster_settings`) $=10=$ số input của `forward`. Khớp.

**2. `ctx.save_for_backward` có đủ cho backward không?**

`backward` cần, từ `ctx.saved_tensors`: `colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, dc, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, cov2D` — đúng 13 tensor, đúng tên, đúng thứ tự với những gì `forward` đã lưu ở dòng 112. $\Rightarrow$ **ĐỦ**.

Tuy vậy, `cov2D` được lưu (`save_for_backward`) nhưng **không** xuất hiện trong `args` truyền cho `_C.rasterize_gaussians_backward` (dòng 125–148) — bị lưu nhưng không dùng. Đây không gây sai kết quả (chỉ tốn một chút bộ nhớ context), nhưng là chi tiết thừa đáng ghi chú.

Ngược lại, `grad_out_color` (gradient output) được truyền trực tiếp từ tham số `backward` nhận vào — **không** qua `save_for_backward` (đúng quy ước: *gradient output* không phải thứ cần lưu từ forward, nó do node sau cung cấp).

**3. Gradient của `depth_map, opacity_map, normal_map, accum_metric_counts` có lan truyền không?**

Forward trả về 7 output nhưng `backward` chỉ dùng `grad_out_color`; 6 gradient còn lại ($\_$, `g_metric`, `_cov2D`, `_depth_map`, `_opacity_map`, `_normal_map`) bị **nhận vào rồi bỏ qua hoàn toàn** — không có dòng nào cộng chúng vào `args` gửi cho CUDA backward, và CUDA backward (`rasterize_gaussians_backward`) cũng không có tham số nhận các gradient này (xem `rasterize_points.md`). $\Rightarrow$ Nếu pipeline train thật sự tính loss dựa trên `depth_map`/`normal_map`/`opacity_map` (ví dụ depth-supervision), gradient từ loss đó **không chảy ngược được** vào $\mu,s,q,\mathrm{sh},\ldots$ qua đường này — về mặt toán, đạo hàm bị implicit set $=0$ tại đây dù đạo hàm giải tích thật khác 0. Đây **không phải lỗi cú pháp** (code chạy không crash) nhưng là **giới hạn chức năng quan trọng**, nên nêu rõ nếu có ý định thêm depth/normal loss.

**4. Điều kiện frustum culling của `markVisible` có hợp lý về mặt chiếu phối cảnh không?**

Theo code thật thi hành (Bước 5), điều kiện được dùng chỉ là $p_{\text{view},z}>0.2$ — đây **là một điều kiện hình học hợp lệ, đúng công thức near-plane culling** chuẩn trong pinhole camera (loại điểm ở sau hoặc quá sát camera, nơi phép chia phối cảnh $x/z$ không ổn định/không xác định khi $z\to0$). Về mặt số, nó đúng.

Tuy nhiên điều kiện **không đầy đủ**: phần kiểm tra $p_{\text{proj},x},p_{\text{proj},y}\in[-1.3,1.3]$ (left/right/top/bottom culling) đã bị **comment** (`auxiliary.h:167`), nên về mặt chức năng, `markVisible` hiện tại **không loại các điểm nằm ngoài góc nhìn ngang/dọc** — chỉ cần $z_{\text{view}}>0.2$ là được đánh dấu visible, kể cả khi điểm ở rất xa hai bên, phía trên/dưới khung hình. Hệ quả toán học: `projmatrix` vẫn được truyền vào `_C.mark_visible` và dùng để tính `p_hom`/`p_proj` (dòng 162–164 `auxiliary.h`), nhưng **kết quả `p_proj` không được dùng ở bất kỳ đâu khác trong hàm** (dead computation) — về chức năng, `markVisible` ở bản này **tương đương với việc chỉ cần `viewmatrix`**, không thực sự cần `projmatrix`. $\Rightarrow$ **Kết luận: công thức đang dùng (near-plane only) tự nó không sai, nhưng không khớp với ý định đầy đủ của frustum culling (4–6 mặt phẳng), do một dòng điều kiện bị vô hiệu hoá bằng comment.** Đây là điểm cần lưu ý nếu `markVisible` được dùng làm bước lọc sớm trước densify/prune — các Gaussian ở ngoài rìa khung hình theo trục $x,y$ vẫn bị tính là "visible".

---

## Ví dụ số

### A. `markVisible` — kiểm chứng tay điều kiện near-plane

Lấy $P=4$ điểm world, camera đặt tại gốc toạ độ nhìn theo $+z$, `viewmatrix` $=I_4$ (nên $p_{\text{view}}=\mu$ không đổi):

| Điểm | $\mu=(x,y,z)$ | $p_{\text{view},z}$ | $p_{\text{view},z}>0.2$? | `visible` |
|---|---|---|---|---|
| G0 | $(0,0,1.0)$ | $1.0$ | ✓ | **True** |
| G1 | $(0,0,0.1)$ | $0.1$ | ✗ | **False** |
| G2 | $(0,0,-1.0)$ | $-1.0$ | ✗ | **False** |
| G3 | $(5.0,0,1.0)$ | $1.0$ | ✓ | **True** |

G3 minh hoạ chính xác lỗ hổng ở mục Kiểm chứng (4): $\mu_x=5.0$ rất lệch trục quang — nếu $f_x\approx W/2$ (fov $\approx90°$) thì chiếu phối cảnh ở $z=1$ cho toạ độ NDC $p_{\text{proj},x}=f_x\cdot 5/1 \gg 1.3$, tức **hoàn toàn ngoài khung hình** — nhưng vì điều kiện $x/y$ đã bị comment, `markVisible` vẫn trả **True** cho G3 do $p_{\text{view},z}=1>0.2$. Nếu dòng comment được bật lại, G3 sẽ bị loại (`False`).

### B. `forward` → `backward` — luồng shape với $N=3$ Gaussian

Giả sử không dùng SH (`colors_precomp` cho sẵn), không dùng `cov3D_precomp` (dùng `scales`+`rotations`), ảnh $H=W=8$, $C=3$ kênh màu.

**Input `forward`** (thứ tự đúng như Bước 3):

| # | Tên | Shape | Ghi chú |
|---|---|---|---|
| 1 | `means3D` ($\mu$) | $(3,3)$ | 3 Gaussian, world-space |
| 2 | `means2D` (placeholder) | $(3,4)$ | `torch.zeros(..., requires_grad=True)`, giá trị không dùng ở forward |
| 3 | `dc` | $(0,)$ | rỗng (không dùng SH) |
| 4 | `sh` | $(0,)$ | rỗng |
| 5 | `colors_precomp` | $(3,3)$ | màu RGB cho sẵn mỗi Gaussian |
| 6 | `opacities` | $(3,1)$ | $\alpha_i\in(0,1)$ |
| 7 | `scales` | $(3,3)$ | $s_i$ |
| 8 | `rotations` | $(3,4)$ | quaternion $q_i$ |
| 9 | `cov3Ds_precomp` | $(0,)$ | rỗng |
| 10 | `raster_settings` | — | NamedTuple, $H=W=8$ |

**Output `forward`** (7 tensor, thứ tự đúng như dòng 113):

| Tên | Shape |
|---|---|
| `color` | $(3,8,8)$ |
| `radii` | $(3,)$ |
| `accum_metric_counts` | phụ thuộc `metric_map`, ví dụ $(8\cdot8,)$ |
| `cov2D` | $(3,7)$ |
| `depth_map` | $(1,8,8)$ hoặc $(8,8)$ tuỳ build |
| `opacity_map` | $(1,8,8)$ |
| `normal_map` | $(3,8,8)$ |

Giả sử sau forward, `radii = [5, 0, 3]` — nghĩa là **Gaussian #1 có bán kính màn hình $=0$** (không chạm pixel nào, ví dụ bị culling hoặc ở ngoài khung hình); quy ước chuẩn 3DGS coi đây là **không visible** (`visibility_filter = (radii > 0)`, xem `gaussian_renderer/__init__.py` dòng 120).

**Input `backward`** (7 gradient, khớp 7 output của forward):

| # | Tên tham số nhận | Shape | Dùng tiếp? |
|---|---|---|---|
| 1 | `grad_out_color` | $(3,8,8)$ | ✓ dùng |
| 2 | `_` (grad của `radii`) | $(3,)$ | ✗ bỏ |
| 3 | `g_metric` | $(64,)$ | ✗ bỏ |
| 4 | `_cov2D` | $(3,7)$ | ✗ bỏ |
| 5 | `_depth_map` | $(1,8,8)$ | ✗ bỏ |
| 6 | `_opacity_map` | $(1,8,8)$ | ✗ bỏ |
| 7 | `_normal_map` | $(3,8,8)$ | ✗ bỏ |

**Output `backward` (`grads`)**, đối chiếu lại đúng 10 vị trí của input `forward`:

| # | Tên | Shape | Khớp input `forward` vị trí |
|---|---|---|---|
| 1 | `grad_means3D` | $(3,3)$ | 1 (`means3D`) ✓ |
| 2 | `grad_means2D` | $(3,4)$ | 2 (`means2D`) ✓ |
| 3 | `grad_dc` | $(0,)$ | 3 (`dc`) ✓ |
| 4 | `grad_sh` | $(0,)$ | 4 (`sh`) ✓ |
| 5 | `grad_colors_precomp` | $(3,3)$ | 5 (`colors_precomp`) ✓ |
| 6 | `grad_opacities` | $(3,1)$ | 6 (`opacities`) ✓ |
| 7 | `grad_scales` | $(3,3)$ | 7 (`scales`) ✓ |
| 8 | `grad_rotations` | $(3,4)$ | 8 (`rotations`) ✓ |
| 9 | `grad_cov3Ds_precomp` | $(0,)$ | 9 (`cov3Ds_precomp`) ✓ |
| 10 | `None` | — | 10 (`raster_settings`) ✓ |

10/10 vị trí khớp shape và khớp "vai trò" — xác nhận lại bằng số liệu cụ thể kết luận ở mục Kiểm chứng (1): **không có hoán vị sai giữa input của `forward` và gradient trả về của `backward`**.

Vì Gaussian #1 có `radii=0` (không được rasterize), gradient tương ứng với hàng thứ 2 trong mọi tensor gradient ở trên (`grad_means3D[1]`, `grad_scales[1]`, …) sẽ bằng $\mathbf 0$ — đây là hành vi đúng kỳ vọng: Gaussian không đóng góp vào ảnh thì không nhận gradient đẩy nó đi/co giãn, nhất quán với $C=\sum_i c_i\alpha_iT_{i-1}$ (`forward.md` mục 9) — nếu Gaussian không nằm trong danh sách tile nào thì không xuất hiện trong tổng, nên $\partial\mathcal L/\partial(\cdot)_1=0$.

---

## Tóm tắt luồng

$$
\underbrace{\texttt{markVisible}}_{p_{view,z}>0.2}\ \to\
\underbrace{\text{forward}}_{10\ \text{input}\to 7\ \text{output},\ \text{CUDA}}\ \to\
\underbrace{\text{ctx.save\_for\_backward}}_{13\ \text{tensor}}\ \to\
\underbrace{\text{backward}}_{7\ \text{grad\_output}\to 10\ \text{grad\_input (đúng thứ tự forward)}}
$$

## Nhận xét

- Toàn bộ "công thức" nằm ở phía CUDA (`forward.cu`, `backward.cu`); file Python chỉ định tuyến tham số đúng vị trí, đúng shape — nhưng chính vì vậy một lỗi sắp xếp thứ tự ở đây sẽ **âm thầm gán sai gradient cho sai tensor** (không crash, không NaN ngay) thay vì báo lỗi rõ ràng — cần kiểm tra thủ công như mục Kiểm chứng mỗi khi sửa chữ ký `forward`/`backward`.
- `markVisible` chỉ thực sự dùng `viewmatrix`; việc truyền `projmatrix` vào là "để dành" cho điều kiện frustum đầy đủ hiện đang bị tắt.
- Gradient của các output phụ (`depth_map`, `opacity_map`, `normal_map`, `accum_metric_counts`) bị cắt hoàn toàn ở `backward` — các nhánh loss dùng những output này (nếu có ở nơi khác trong codebase) sẽ không lan truyền được gradient vào Gaussian qua rasterizer này.
