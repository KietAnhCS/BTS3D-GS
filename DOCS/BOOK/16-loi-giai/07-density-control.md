[← Phần 6](06-gradient-adam.md) · [Mục lục lời giải](00-muc-luc-loi-giai.md) · Bài toán lớn — Phần 7/8

# Phần 7 — Structure-Aware Densification (SADGS) tại $t=5000$

> **Đầu vào nhận từ Phần 6:** 4 Gaussian × 59 tham số đã trải qua một bước Adam tại $t=5000$, cộng bộ
> đếm gradient tích luỹ `xyz_gradient_accum`, `xyz_gradient_accum_abs`, `denom` — xem [Phần 6](06-gradient-adam.md).

> **Đính chính so với bản nháp trước:** bản trước của Phần 7 mô tả một cơ chế densify/prune **giả định**
> (split bằng lấy mẫu Gauss ngẫu nhiên, prune bằng multinomial theo `pruning_score`) không khớp với
> codebase thật của đề bài. Codebase thật là **SADGS**
> (`SADGS/scene/gaussian_model.py`, `SADGS/train.py`, `SADGS/utils/freq_utils.py`). Cơ chế densify/prune
> thật của SADGS là **structure-aware**: quyết định split/clone/
> prune một Gaussian dựa trên tỉ số $\eta$ giữa kích thước hình chiếu của Gaussian và bước sóng cục bộ
> của kết cấu ảnh (texture), đo bằng structure tensor đa tỉ lệ, tích luỹ **online** qua nhiều view liên
> tiếp trong huấn luyện — không phải bằng một vòng "chấm điểm" riêng tại đúng vòng densify như bản nháp
> trước giả định. Toàn bộ Phần 7 dưới đây được viết lại theo đúng mã nguồn thật.

---

## Mục lục Phần 7

- [7.1 — Lịch densify/prune thật trong `train.py`](#71)
- [7.2 — Thống kê tần số online: $\eta_{3ch}$, `update_freq_stats_online`](#72)
- [7.3 — `densify_and_prune_structgs`: tập hợp các mặt nạ](#73)
- [7.4 — `densify_and_clone_structgs`](#74)
- [7.5 — `densify_and_split_structgs`: split dị hướng giải tích](#75)
- [7.6 — Pruning và ép opacity](#76)
- [7.7 — `final_prune_structgs` và `expand_undersized_gs`](#77)
- [7.8 — Ví dụ số minh hoạ trên cảnh đồ chơi 4 Gaussian](#78)
- [7.9 — Đầu ra chuyển cho Phần 8](#79)
- [7.10 — Ghi chú đối chiếu với code](#710-ghi-chu)
- [7.11 — Bài tập](#711)

---

<a id="71"></a>
## 7.1 — Lịch densify/prune thật trong `train.py`

Trích `SADGS/train.py` (khối chính, dòng ~315–419):

```python
if iteration < opt.densify_until_iter:                      # 15_000
    gaussians.max_radii2D[visibility_filter] = torch.max(...)
    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    is_normal_densification = iteration > opt.densify_from_iter and \
                               iteration % opt.densification_interval == 0
    if is_normal_densification:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        grads = gaussians.xyz_gradient_accum / gaussians.denom
        is_grad_high = torch.norm(grads, dim=-1) >= 1e-5              # ngưỡng cứng, không phải CLI

        valid_mask = gaussians.accum_view_count > 0
        high_ratio[valid_mask] = gaussians.eta_high_count[valid_mask] / gaussians.accum_view_count[valid_mask]
        low_ratio[valid_mask]  = gaussians.eta_low_count[valid_mask]  / gaussians.accum_view_count[valid_mask]

        split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high   # 0.8
        prune_mask = (low_ratio  > opt.prune_ratio_threshold) & valid_mask     # 0.8
        max_high_eta = gaussians.max_eta_3ch

        gaussians.densify_and_prune_structgs(
            max_screen_size=size_threshold, min_opacity=0.1,
            extent=scene.cameras_extent, radii=radii, args=opt,
            importance_score=gaussians.accum_view_count, pruning_score=None,
            custom_split_mask=split_mask, custom_prune_mask=prune_mask,
            viewspace_points_indices=None, max_eta_3ch=max_high_eta)

        # reset toàn bộ bộ tích luỹ tần số cho cửa sổ kế tiếp
        gaussians.accum_eta.zero_(); gaussians.accum_view_count.zero_()
        gaussians.max_eta_3ch.zero_(); gaussians.accum_weights_valid.zero_()
        gaussians.eta_high_count.zero_(); gaussians.eta_high_sum_3ch.zero_()
        gaussians.eta_mid_count.zero_();  gaussians.eta_mid_sum_3ch.zero_()
        gaussians.eta_low_count.zero_()

    if iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter):
        gaussians.reset_opacity(opt.opacity_reset_decay)
```

| Việc | Điều kiện thật | Tại $t=5000$ | Kết luận |
|---|---|---|---|
| Tích luỹ $\eta$ online (`update_freq_stats_online`) | `iteration < densify_until_iter` **và** `iteration % 10 == 0` | $5000<15000$, $5000\bmod10=0$ | ✅ chạy — nhưng chỉ mỗi **10** vòng, không mỗi vòng |
| Densify + prune (`densify_and_prune_structgs`) | `densify_from_iter(500) < t`, `t % densification_interval(100) == 0` | $500<5000$, $5000\bmod100=0$ | ✅ **CHẠY** |
| `size_threshold` (bật điều kiện big_points_vs) | `t > opacity_reset_interval(3000)` | $5000>3000$ | ✅ bật (=20 px) |
| Opacity reset | `t % 3000 == 0` | $5000\bmod3000=2000\ne0$ | ❌ không chạy ở $t=5000$ |
| `final_prune_structgs` | gọi tại `iteration in prune_iterations` (giai đoạn cuối, dùng `pruning_score` từ multiview) — **lời gọi thật trong `train.py:418-419` đang bị comment**; chỉ opacity-prune đơn giản (`get_opacity<0.1`) chạy tại các mốc đó | $5000$ không phải mốc `prune_iterations` | ❌ chưa tới — minh hoạ công thức ở [§7.7](#77) |

Khác biệt quan trọng so với bản nháp trước:
- `densification_interval` thật $=100$ (không phải $500$); `densify_grad_threshold`/`densify_grad_abs_threshold` trong `arguments/__init__.py` (`0.0002`/`0.0004`) là tham số cho nhánh `densify_and_prune` (3DGS gốc, dùng khi `warmup_densification=True`), **không** phải nhánh SADGS chính — nhánh chính dùng ngưỡng cứng `1e-5` cho `is_grad_high` và `args.grad_thresh=0.0002`/`args.grad_abs_thresh=0.0002` **bên trong** `densify_and_prune_structgs`.
- Không có bước "chấm điểm 3 camera" chạy đồng bộ tại đúng vòng densify (như bản nháp trước giả định). Thống kê tần số được **tích luỹ dần** mỗi 10 vòng lặp bình thường (cửa sổ 100 vòng giữa hai lần densify ⇒ khoảng $10$ lần cập nhật `update_freq_stats_online`, mỗi lần dùng đúng camera đang render batch đó, không lấy mẫu lại).

---

<a id="72"></a>
## 7.2 — Thống kê tần số online: $\eta_{3ch}$, `update_freq_stats_online`

### 7.2.0 Ý tưởng

Khác với cách chấm điểm bằng ảnh lỗi màu tổng hợp mà bản nháp trước giả định, SADGS so khớp **kích thước hình chiếu** của mỗi Gaussian với
**bước sóng cục bộ** của kết cấu ảnh GT tại đúng vị trí Gaussian đó, đo bằng structure tensor 2D
(`get_structure_tensor_torch`, `utils/loss_utils.py`) tính đa tỉ lệ (`st_levels=4`) trên ảnh GT, cache
theo tên ảnh (`structure_tensor_cache`).

### 7.2.1 Công thức (chế độ mặc định `eta_compute_mode="wavelength"`, `utils/freq_utils.py:313-348`)

Với mỗi Gaussian nhìn thấy trong view $v$:

$$
\text{trace}=S_{xx}+S_{yy},\quad \det=S_{xx}S_{yy}-S_{xy}^2,\quad
\lambda_1=\frac{\text{trace}}2+\sqrt{\max\Bigl(\bigl(\tfrac{\text{trace}}2\bigr)^2-\det,\,0\Bigr)}
$$

$$
\boxed{\ \lambda_{\min}=\frac1{\sqrt{\lambda_1}+10^{-5}}\ }\qquad\text{(bước sóng cục bộ, tính bằng pixel)}
$$

$$
\boxed{\ \eta_{3ch}=\frac{\lVert\text{trục chiếu 2D thứ }k\rVert}{\lambda_{\min}},\quad k\in\{1,2,3\}\ }\qquad
\eta_{3ch}\leftarrow \eta_{3ch}\cdot w_{\text{trans}}
$$

$w_{\text{trans}}$ là trọng số transmittance/opacity của Gaussian tại pixel đó (`weights_valid`, cắt bởi
`freq_transmittance_threshold`, `freq_opacity_threshold`). $\eta>1$ nghĩa là trục Gaussian **lớn hơn**
đặc trưng texture nó phủ lên (Gaussian quá to, gây alias/mờ) ⇒ cần **split**. $\eta$ nhỏ nghĩa là Gaussian
đã đủ nhỏ so với texture (vùng phẳng) ⇒ ứng viên **prune**.

(Có một chế độ thứ hai `eta_compute_mode="projection"` dùng dạng toàn phương
$\eta=S_{xx}u^2+2S_{xy}uv+S_{yy}v^2$ thay vì tỉ số độ dài/bước sóng — không phải mặc định, không dùng
trong ví dụ số §7.8.)

### 7.2.2 Tích luỹ và phân loại đa view (`update_freq_stats_online:384-415`)

Mỗi lần một Gaussian được nhìn thấy ở một view (mỗi 10 vòng lặp):

```python
gaussians.accum_eta[idx]        += eta_3ch.sum(dim=1)
gaussians.accum_view_count[idx] += 1.0
gaussians.max_eta_3ch[idx] = torch.max(gaussians.max_eta_3ch[idx], eta_3ch)   # max chạy tích luỹ

eta_max_scalar = eta_3ch.max(dim=1).values
TAU_HIGH, TAU_LOW = 1.0, 0.1
is_high = eta_max_scalar > TAU_HIGH
is_low  = eta_max_scalar <= TAU_LOW
is_mid  = ~is_high & ~is_low
gaussians.eta_high_count[idx[is_high]] += 1.0
gaussians.eta_mid_count[idx[is_mid]]   += 1.0
gaussians.eta_low_count[idx[is_low]]   += 1.0
```

Sau cửa sổ densify (từ vòng vừa densify trước tới $t=5000$), mỗi Gaussian có bộ đếm
$(\text{accum\_view\_count}_i,\ \text{eta\_high\_count}_i,\ \text{eta\_mid\_count}_i,\ \text{eta\_low\_count}_i,\ \max_v\eta_{3ch,i}^{(v)})$
— đây chính là **"đa số phiếu qua nhiều view"** (multiview consistency) mà `train.py` dùng để quyết định
split/prune, thay cho ảnh lỗi màu tổng hợp mà bản nháp trước giả định.

---

<a id="73"></a>
## 7.3 — `densify_and_prune_structgs`: tập hợp các mặt nạ

### 7.3.0 Trích code (`scene/gaussian_model.py:957-1053`, rút gọn)

```python
grad_vars  = self.xyz_gradient_accum     / self.denom
grads_abs  = self.xyz_gradient_accum_abs / self.denom
grad_qualifiers     = torch.norm(grad_vars, dim=-1) >= args.grad_thresh       # 0.0002
grad_qualifiers_abs = torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh   # 0.0002

full_split_mask = custom_split_mask   # (viewspace_points_indices=None ⇒ gán thẳng, không scatter)
full_prune_mask = custom_prune_mask

clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense * extent   # dense=0.001
split_qualifiers = torch.max(self.get_scaling, dim=1).values >  args.dense * extent

final_split_mask = (full_split_mask | grad_qualifiers_abs) & split_qualifiers
final_clone_mask = (full_split_mask | grad_qualifiers)     & clone_qualifiers

metric_mask = importance_score > args.importance_score_threshold if importance_score is not None \
              else torch.ones_like(final_clone_mask)         # importance_score = accum_view_count, ngưỡng 0.5

self.densify_and_clone_structgs(metric_mask, final_clone_mask)
combined_split_mask = metric_mask & final_split_mask
self.densify_and_split_structgs(combined_split_mask, max_eta_3ch=max_eta_3ch, scale_power=args.ks_scale_power)

prune_mask = (self.get_opacity < min_opacity).squeeze()
if max_screen_size:
    prune_mask |= (self.max_radii2D > max_screen_size) | (self.get_scaling.max(dim=1).values > 0.1*extent)
prune_mask = prune_mask | full_prune_mask       # gộp thêm mặt nạ "low-eta đa số phiếu" từ train.py

if pruning_score is not None:
    ...multinomial trên một nửa số ứng viên...    # KHÔNG chạy trong train.py thật (pruning_score=None)
else:
    self.prune_points(prune_mask)

opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
self._opacity = self.replace_tensor_to_optimizer(opacities_new, "opacity")["opacity"]
```

Ba điểm khác biệt cốt lõi so với 3DGS gốc:

1. **`full_split_mask` (từ `custom_split_mask` = `split_mask` của `train.py`) tham gia vào CẢ hai** nhánh
   `final_split_mask` **và** `final_clone_mask` (phép OR) — một Gaussian có tỉ lệ high-eta cao qua nhiều
   view **luôn** được đưa vào ứng viên densify, bất kể gradient vị trí; gradient (`grad_qualifiers`,
   `grad_qualifiers_abs`) chỉ là điều kiện OR thứ hai, không phải điều kiện bắt buộc.
2. **Cổng "importance"** không còn là số pixel lỗi màu (như bản nháp trước giả định) mà là **số view đã quan sát được Gaussian**
   (`accum_view_count`), ngưỡng $0.5$ ⇒ thực chất chỉ loại các Gaussian **chưa từng được nhìn thấy** trong
   cửa sổ vừa qua (`accum_view_count=0`).
3. **`full_prune_mask`** (mặt nạ `prune_mask` tính trong `train.py` từ `low_ratio`) được **OR** thẳng vào
   `prune_mask` cuối — một Gaussian nằm trong vùng phẳng (thấy low-eta ở $>80\%$ số view) bị prune ngay,
   độc lập với opacity/kích thước.
4. **Không có `densify_and_split(N=2)` ngẫu nhiên và không có multinomial removal** ở nhánh chính (đường
   `pruning_score is not None` không chạy vì `train.py` luôn gọi với `pruning_score=None`).

---

<a id="74"></a>
## 7.4 — `densify_and_clone_structgs`

```python
def densify_and_clone_structgs(self, metric_mask, filter):
    selected_pts_mask = metric_mask & filter
    new_xyz = self._xyz[selected_pts_mask]; ... (sao chép nguyên vẹn mọi thuộc tính)
    parent_counts = self.densify_count[selected_pts_mask]     # KHÔNG +1 khi clone
    self.densification_postfix(...)
    self.densify_count[n_old:n_old+n_new] = parent_counts
```

Giống 3DGS gốc: nhân đôi Gaussian tại đúng vị trí/scale/màu/opacity của cha, không xê dịch, không thu
nhỏ. Điểm khác duy nhất so với `densify_and_clone` gốc: `densify_count` (bộ đếm "đã bị split bao nhiêu
lần") được **kế thừa nguyên vẹn**, không tăng — clone không được tính là một "thế hệ split" mới.

---

<a id="75"></a>
## 7.5 — `densify_and_split_structgs`: split dị hướng giải tích

### 7.5.0 Ý tưởng lý thuyết (Nyquist per-axis)

$\eta=(\sigma\cdot\omega_{\max})^2$ đo mức vi phạm Nyquist theo mỗi trục. Muốn $\eta_{\text{new}}\le1$ cần
thu nhỏ $\sigma$ theo hệ số $k\ge\sqrt\eta$. SADGS tính $k$ **riêng cho từng trục** (không chỉ trục dài
nhất như 3DGS gốc):

$$
\boxed{\ k_{\text{axis}}=\Bigl\lceil\sqrt{\max(\eta_{\text{axis}},\,1)}\Bigr\rceil,\quad k_{\text{axis}}\ge1\ }
$$

Nếu `max_eta_3ch=None` (không có tín hiệu tần số — fallback), code quay lại hành vi 3DGS gốc: chỉ trục
scale lớn nhất được $k=2$, hai trục còn lại $k=1$.

### 7.5.1 Số con và lưới lấy mẫu (`gaussian_model.py:638-832`)

Một Gaussian cha sinh ra $N=k_x\cdot k_y\cdot k_z$ con (không cố định $N=2$ như 3DGS gốc). Mỗi con
$j=(i_x,i_y,i_z)$, $0\le i_a<k_a$, nằm tại toạ độ lưới **đều đặn, tất định** (không lấy mẫu Gauss ngẫu
nhiên như bản nháp trước giả định):

$$
g_a=i_a-\frac{k_a-1}2,\qquad
\text{offset}_{\text{local}}=\bigl(g_x,g_y,g_z\bigr)\odot\Bigl(\frac{s}{k}\Bigr)\sqrt{12}
$$

$$
\boxed{\ \mu^{(j)}=\mu_i+R(q_i)\cdot\text{offset}_{\text{local}}^{(j)}\ },\qquad
\boxed{\ \tilde s^{(j)}=\log\!\Bigl(\frac{s_i}{k^{\,p}}\Bigr)\ },\ p=\texttt{ks\_scale\_power}\ (=1.0\text{ mặc định})
$$

$q,\alpha,f_{\text{dc}},f_{\text{rest}},$ `filter_3D` được **kế thừa nguyên vẹn** (`repeat_interleave`).
`densify_count` con $=$ `densify_count` cha $+1$. Bản gốc bị xoá ngay sau khi con được thêm vào
(`prune_points` trên `total_prune_mask`).

Vì $\sqrt{12}\,\sigma/k$ chính là **độ rộng ô lưới đều** chia $[-3\sigma,3\sigma]$ thành $k$ phần, cách
đặt này rải các con **phủ khít, không chồng lấn ngẫu nhiên** — khác hẳn `densify_and_split` 3DGS gốc
(lấy mẫu $\epsilon\sim\mathcal N(0,\Sigma)$, có thể trùng nhau hoặc để trống góc).

### 7.5.2 `expand_undersized_gs` (định nghĩa trong code, lời gọi trong `train.py` đang bị comment)

```python
undersized_mask = (max_eta_3ch < tau_expand) & (max_eta_3ch > 0)     # tau_expand = 1.0
delta_log_scale = -0.5 * torch.log(torch.clamp(max_eta_3ch[undersized_mask], min=1e-6))
self._scaling[undersized_mask] += delta_log_scale        # ⇒ eta_new := 1.0 chính xác (giải tích)
```

Đây là chiều **ngược** của split: nếu $\eta<\tau_{\text{expand}}=1$ (Gaussian nhỏ hơn mức cần thiết so
với texture — dưới-lấy-mẫu ngược), scale được **phóng to giải tích** sao cho $\eta_{\text{new}}=1$ đúng
bằng công thức $\log s_{\text{new}}=\log s_{\text{old}}-\tfrac12\log\eta$ — không cần lặp lại nhiều bước
gradient descent để "lớn dần". Hàm tồn tại và đã kiểm định trong `gaussian_model.py:833-865`, nhưng lời
gọi ở `train.py:356-359` đang bị comment ở phiên bản huấn luyện chính (không chạy trong vòng lặp thật
hiện tại) — Phần 7 nêu công thức để đầy đủ với mã nguồn, không tính nó vào ví dụ số §7.8.

---

<a id="76"></a>
## 7.6 — Pruning và ép opacity

Ba nguồn ứng viên prune được **OR** lại (đã liệt kê ở §7.3):

$$
\mathcal P = \underbrace{[\alpha_i<0.1]}_{\text{opacity thấp}} \ \lor\
\underbrace{[r^{2D}_i>20\text{px}]}_{\text{chỉ bật khi }t>3000} \ \lor\
\underbrace{[\max s_i>0.1\cdot\text{extent}]}_{\text{chỉ bật khi }t>3000} \ \lor\
\underbrace{\bigl[\tfrac{\text{eta\_low\_count}_i}{\text{accum\_view\_count}_i}>0.8\bigr]}_{\text{multiview: vùng phẳng đa số phiếu}}
$$

Không có bước `torch.multinomial` trong đường chạy thật (`pruning_score=None`) — mọi Gaussian khớp
$\mathcal P$ bị xoá **toàn bộ**, không lấy mẫu một nửa (nhánh `pruning_score is not None`, có tồn tại
trong `densify_and_prune_structgs` như một tuỳ chọn, chỉ minh hoạ chứ không chạy trong `train.py` thật).

Sau khi prune, opacity bị **kẹp trên** $0.8$ (không phải giảm — khác `reset_opacity` của 3DGS gốc, vốn
**nhân** $\alpha$ với `opacity_reset_decay=0.1` và chỉ chạy định kỳ mỗi `opacity_reset_interval`):

$$
\alpha_i \leftarrow \min(\alpha_i,\ 0.8)\quad\text{(áp dụng vô điều kiện, mọi vòng có densify)}
$$

---

<a id="77"></a>
## 7.7 — `final_prune_structgs` và ý nghĩa giai đoạn cuối

```python
def final_prune_structgs(self, min_opacity, pruning_score=None):
    prune_mask  = (self.get_opacity < min_opacity).squeeze()
    scores_mask = pruning_score > 0.9
    self.prune_points(prune_mask | scores_mask)
```

Dùng cho giai đoạn dọn dẹp cuối (sau khi ngừng densify, $t>$ `densify_until_iter`$=15000$), xoá thêm các
Gaussian có "điểm không nhất quán đa view" (`pruning_score`) vượt $0.9$. Trong `train.py` thật, lời gọi
tại các mốc `prune_iterations` (dòng 418-419) **đang bị comment**; nhánh đang chạy tại các mốc đó chỉ là
một prune đơn giản theo opacity:

```python
if iteration in prune_iterations:
    prune_mask = (gaussians.get_opacity < 0.1).squeeze()
    gaussians.prune_points(prune_mask)
```

$t=5000$ chưa tới giai đoạn này (`densify_until_iter=15000`) — hàm `final_prune_structgs` tồn tại và
đúng theo mã nguồn, nhưng không được tính vào kết quả số của Phần 7/8 (không có lời gọi thật nào tại
$t=5000$, và lời gọi ở giai đoạn cuối thật ra đang tắt).

---

<a id="78"></a>
## 7.8 — Ví dụ số minh hoạ trên cảnh đồ chơi 4 Gaussian

### 7.8.0 Vì sao không thể "thay số" $\eta$ chính xác bằng công thức đóng như với ảnh lỗi màu

$\eta_{3ch}$ phụ thuộc **structure tensor của ảnh GT thật** (đạo hàm ảnh theo $x,y$, làm mượt Gauss,
lấy mẫu song tuyến tại đúng toạ độ hình chiếu) — đây là phép tính trên **ảnh pixel thật**, không có
công thức closed-form gọn để tính tay như $L_1$/SSIM. Ví dụ số dưới đây do đó dùng **giá trị $\eta$ minh
hoạ** (không phải chạy trên GPU thật), được chọn để đi qua đủ các nhánh của thuật toán (split đẳng
hướng, split "trơ" khi $\eta<1$, giữ nguyên, không prune) và vẫn dùng đúng $\mu,s,\alpha,q$ của 4
Gaussian gốc (bảng §6.2, giống các phần trước). Mọi ngưỡng, công thức, thứ tự lệnh gọi **là chính xác**
theo mã nguồn; chỉ các số $\eta$ và bộ đếm view là số minh hoạ, không phải kết xuất từ `ch07_test.py`.

### 7.8.1 Đầu vào

| $i$ | $\mu_i$ | $s_i$ | $\alpha_i$ | accum\_view\_count | (eta\_high, eta\_mid, eta\_low)\_count (giả định, tổng = accum\_view\_count) | $\max_v\eta_{3ch,i}$ (giả định, 3 trục bằng nhau vì đẳng hướng) | $\bar g_i$ (từ Phần 6) |
|---|---|---|---|---|---|---|---|
| $G_1$ | $(0,0,0)$ | 0.8505 | 0.1 | 50 | (45, 5, 0) | 2.89 | $7.08\times10^{-3}$ |
| $G_2$ | $(0.5,0.3,0.5)$ | 0.9434 | 0.1 | 50 | (2, 8, 40) | 0.35 | $5.93\times10^{-3}$ |
| $G_3$ | $(-0.4,-0.2,1.0)$ | 1.1150 | 0.1 | 50 | (30, 20, 0) | 1.44 | $5.98\times10^{-3}$ |
| $G_4$ | $(0.3,-0.5,0.2)$ | 0.8888 | 0.1 | 0 | (0,0,0) | 0.00 | $6.04\times10^{-3}$ |

$\text{extent}=1.690250$; $\texttt{dense}\cdot\text{extent}=0.001\times1.690250=0.0016903$;
$0.1\cdot\text{extent}=0.169025$. $q_i=(1,0,0,0)$ (không xoay) cho cả 4 Gaussian.

### 7.8.2 Các mặt nạ trung gian (từ `train.py`)

| $i$ | high\_ratio | low\_ratio | is\_grad\_high ($\ge10^{-5}$) | split\_mask (high>0.8 ∧ grad\_high) | prune\_mask (low>0.8) |
|---|---|---|---|---|---|
| $G_1$ | $45/50=0.90$ | $0/50=0.00$ | ✅ | **True** | False |
| $G_2$ | $2/50=0.04$ | $40/50=0.80$ | ✅ | False | False ($0.80\not>0.80$, biên) |
| $G_3$ | $30/50=0.60$ | $0/50=0.00$ | ✅ | False | False |
| $G_4$ | — (`valid_mask=False`, accum\_view\_count=0) | — | — | False | False |

### 7.8.3 `densify_and_prune_structgs` — các mặt nạ nội bộ

$\max s_i=(0.8505,0.9434,1.1150,0.8888)$, tất cả $\gg 0.0016903$ ⇒ `split_qualifiers`$=[T,T,T,T]$,
`clone_qualifiers`$=[F,F,F,F]$ (như đã lập luận ở các Phần trước — 4 điểm SfM ban đầu rất thưa so với
`percent_dense`).

$\bar g_i\ge$ `grad_thresh`$=2\times10^{-4}$ cho cả 4 (Phần 6) ⇒ `grad_qualifiers`$=[T,T,T,T]$; giả định
tương tự `grad_qualifiers_abs`$=[T,T,T,T]$ (cùng bậc độ lớn, xem Phần 6).

$$
\text{final\_split}=(\text{split\_mask}\lor\text{grad\_qualifiers\_abs})\land\text{split\_qualifiers}
=([T,F,F,F]\lor[T,T,T,T])\land[T,T,T,T]=[T,T,T,T]
$$
$$
\text{final\_clone}=(\text{split\_mask}\lor\text{grad\_qualifiers})\land\text{clone\_qualifiers}
=[T,T,F,F]\land[F,F,F,F]=[F,F,F,F]
$$

`importance_score = accum_view_count = (50,50,50,0)`, ngưỡng $0.5$ ⇒ `metric_mask`$=[T,T,T,F]$
($G_4$ chưa từng được nhìn thấy trong cửa sổ này ⇒ bị loại khỏi **cả densify lẫn** vòng này, dù
gradient/scale của nó đủ điều kiện).

- `densify_and_clone_structgs`: `metric_mask ∧ final_clone` $=[F,F,F,F]$ ⇒ **không clone ai**.
- `combined_split_mask = metric_mask ∧ final_split` $=[T,T,T,F]$ ⇒ **split $G_1,G_2,G_3$**, $G_4$ giữ
  nguyên (không đủ "importance", dù về mặt kích thước nó cũng đủ điều kiện split).

### 7.8.4 Split $G_1$ — minh hoạ đầy đủ ($\eta_{3ch,1}=(2.89,2.89,2.89)$, đẳng hướng)

$$
k=\lceil\sqrt{\max(2.89,1)}\rceil=\lceil1.7000\rceil=2\quad\text{(cả 3 trục, vì đẳng hướng)}
$$

$N=k_x k_y k_z=2^3=8$ con (khác hẳn $N=2$ của 3DGS gốc — vì cả ba trục cùng vi phạm Nyquist).
Lưới $(i_x,i_y,i_z)\in\{0,1\}^3$, $g_a=i_a-0.5\in\{-0.5,+0.5\}$:

$$
\tilde s^{(j)}=\log\Bigl(\frac{0.8505}{2^{1.0}}\Bigr)=\log(0.42525)=-0.855321,\qquad s^{(j)}=0.42525\ \ (\text{cả 8 con})
$$
$$
\text{offset}_{\text{local}}=(g_x,g_y,g_z)\cdot\frac{s}{k}\sqrt{12}=(g_x,g_y,g_z)\cdot\frac{0.8505}{2}\times3.4641=(g_x,g_y,g_z)\times1.47316
$$

Với $R_1=I$: $\mu^{(j)}=(0,0,0)+1.47316\cdot(g_x,g_y,g_z)$. Ví dụ con $j=(0,0,0)$: $g=(-0.5,-0.5,-0.5)$
⇒ $\mu^{(000)}=(-0.7366,-0.7366,-0.7366)$; con $j=(1,1,1)$: $g=(0.5,0.5,0.5)$ ⇒
$\mu^{(111)}=(0.7366,0.7366,0.7366)$; sáu con còn lại hoán vị dấu tương tự — 8 con nằm đúng tại 8 đỉnh
của một hình lập phương cạnh $2\times1.47316=2.9463$ tâm tại $\mu_1$, mỗi con kế thừa $q=(1,0,0,0)$,
$\alpha=0.1$, $k_{00}=(1.063,-1.063,-1.063)$, `densify_count`$=0+1=1$.

$G_2$ ($\eta_{3ch,2}=(0.35,0.35,0.35)<1$): $k=\lceil\sqrt{\max(0.35,1)}\rceil=\lceil1\rceil=1$ trên cả 3
trục ⇒ $N=1$ — **"split" không sinh con mới về mặt hình học** (chỉ 1 bản duy nhất, scale không đổi vì
$k^{p}=1^1=1$), vì $\eta<1$ nghĩa là Gaussian **đã đủ nhỏ** dù nó vẫn lọt qua cổng gradient/scale. Đây là
điểm khác biệt cốt lõi so với 3DGS gốc (nơi gradient cao luôn ép chia đôi bất kể $\eta$): SADGS **tự bảo
vệ** trước over-splitting khi tín hiệu tần số nói không cần.

$G_3$ ($\eta_{3ch,3}=(1.44,1.44,1.44)$): $k=\lceil\sqrt{1.44}\rceil=\lceil1.2\rceil=2$ mỗi trục ⇒ $N=8$
con, $\tilde s^{(j)}=\log(1.1150/2)=\log(0.5575)=-0.584249$, cùng cách rải lưới lập phương như $G_1$
(bán kính lưới $=\tfrac{1.1150}{2}\sqrt{12}=1.9313$).

### 7.8.5 Quần thể sau split, trước prune

$$
N: 4\ \xrightarrow{\text{split }G_1(\times8),\,G_2(\times1),\,G_3(\times8)}\ (8+1+8)+1_{G_4}=18
$$

($G_2$ "split" thành đúng 1 bản = chính nó bị `densification_postfix` thêm vào rồi bản gốc bị xoá, ròng
không đổi số lượng nhưng `densify_count` tăng 1; $G_4$ không đụng tới.)

### 7.8.6 Prune

`prune_mask` (opacity/size) $=[F,\ldots,F]$ cho toàn bộ 19 điểm ở bước này (tất cả $\alpha=0.1$; chú ý
$\alpha=0.1$ **không** nhỏ hơn `min_opacity=0.1` — điều kiện $\alpha<0.1$ là so sánh chặt nên **sai** cho
mọi điểm, không ai bị xoá vì opacity). `max_radii2D` của các con mới $=0$ (chưa render lần nào) nên
không kích hoạt `big_points_vs`. `full_prune_mask` (đa số phiếu low-eta, tính TRƯỚC densify trên quần
thể cũ 4 điểm, KHÔNG áp dụng lại cho các con mới vì `densification_postfix` chèn số 0 vào các bộ đếm —
hệ quả: các con mới sinh ra trong **chính** vòng densify này không thể bị `full_prune_mask` chặn ngay,
dù cha của chúng từng có `low_ratio` cao) $=[F,F,F,F]$ cho cả 4 gốc ở ví dụ này (không ai có
`low_ratio>0.8`, kể cả $G_2$ đứng sát biên $0.80$). ⇒ **không ai bị prune** ở vòng $t=5000$ này.

Opacity bị kẹp: $\alpha=0.1<0.8$ với mọi điểm ⇒ $\min(\alpha,0.8)=0.1$ không đổi gì (kẹp trên vô hại ở
cảnh đồ chơi vì opacity ban đầu rất thấp).

### 7.8.7 Quần thể chuyển giao cho Phần 8

$$
\mathcal G_1=\{\,8\text{ con của }G_1,\ 1\text{ bản của }G_2,\ 8\text{ con của }G_3,\ G_4\text{ (không đổi)}\,\},\qquad N=18
$$

Vì $\mu$ và $\tilde s$ của mỗi con là kết quả của một công thức giải tích tất định (không lấy mẫu ngẫu
nhiên), toàn bộ $\mathcal G_1$ ở trên là **tái lập được 100%** chỉ từ bảng đầu vào §7.8.1 — không phụ
thuộc seed như bản nháp trước (vốn giả định dùng `np.random.normal`).

---

<a id="79"></a>
## 7.9 — Đầu ra chuyển cho Phần 8

$\mathcal G_1$ ở trên (18 Gaussian, mỗi Gaussian đủ 14 tham số cốt lõi $(\mu,\tilde s,\tilde\alpha,q,k_{00})$
+ 45 hệ số $f_{\text{rest}}=0$ + `filter_3D=0` — xem [Phần 8](08-cost-model-ply.md) để xuất `.ply` với
đúng 63 cột theo `construct_list_of_attributes` **thật** của `SADGS/scene/gaussian_model.py`).

---

<a id="710-ghi-chu"></a>
## 7.10 — Ghi chú đối chiếu với code

| Mục | Vị trí code | Ghi chú |
|---|---|---|
| Lịch gọi trong vòng lặp huấn luyện | `train.py:275-276, 315-403, 412-419` | `update_freq_stats_online` mỗi 10 vòng; densify/prune mỗi `densification_interval=100` trong $(500,15000)$; opacity reset mỗi 3000; `final_prune_structgs` (comment) tại `prune_iterations` |
| $\eta_{3ch}$, phân loại high/mid/low | `utils/freq_utils.py:181-415` (`update_freq_stats_online`) | `TAU_HIGH=1.0`, `TAU_LOW=0.1` hard-code trong hàm, không đọc từ CLI |
| `densify_and_prune_structgs` | `scene/gaussian_model.py:957-1053` | tham số `args.grad_thresh`, `args.grad_abs_thresh`, `args.dense`, `args.importance_score_threshold`, `args.ks_scale_power` |
| `densify_and_clone_structgs` | `scene/gaussian_model.py:934-956` | `densify_count` kế thừa, không tăng |
| `densify_and_split_structgs` | `scene/gaussian_model.py:638-832` | $k_{\text{axis}}=\lceil\sqrt{\max(\eta,1)}\rceil$, lưới toạ độ tất định, `densify_count`$\mathrel{+}=1$ |
| `expand_undersized_gs` | `scene/gaussian_model.py:833-865` | tồn tại trong code, lời gọi ở `train.py:356-359` đang bị comment |
| `final_prune_structgs` | `scene/gaussian_model.py:1059-1066` | `min_opacity`, `pruning_score>0.9`; lời gọi thật ở `train.py:418-419` đang bị comment |
| Tham số CLI liên quan | `arguments/__init__.py:85-156` | `grad_thresh=0.0002`, `grad_abs_thresh=0.0002`, `dense=0.001`, `importance_score_threshold=0.5`, `split_ratio_threshold=0.8`, `prune_ratio_threshold=0.8`, `tau_expand=1.0`, `ks_scale_power=1.0`, `densification_interval=100`, `densify_from_iter=500`, `densify_until_iter=15000`, `opacity_reset_interval=3000` |

---

<a id="711"></a>
## 7.11 — Bài tập

1. Tại sao `densify_and_prune_structgs` cho phép một Gaussian vào `final_split_mask` chỉ nhờ
   `custom_split_mask` (đa số phiếu high-eta) **mà không cần** `grad_qualifiers_abs`? Nêu một tình huống
   cụ thể (Gaussian tĩnh, gradient vị trí gần 0 nhưng phủ lên cạnh sắc nét) nơi điều này quan trọng.
2. Vì sao `expand_undersized_gs` tồn tại trong `gaussian_model.py` nhưng lời gọi trong `train.py` lại bị
   comment? Nếu bật lại, nó sẽ tương tác thế nào với `densify_and_split_structgs` trong cùng một vòng lặp
   (thứ tự nào hợp lý hơn: expand trước split, hay ngược lại)?
3. Chứng minh bằng đại số rằng khi $\eta_x=\eta_y=\eta_z=\eta$ (đẳng hướng) và $\eta\in(1,4]$, công thức
   $k=\lceil\sqrt\eta\rceil$ luôn cho $k=2$, do đó $N=k^3=8$ — không bao giờ ra $N=2,3,\dots,7$ ở chế độ
   đẳng hướng. Với $\eta$ dị hướng thế nào thì $N=2$ là có thể (chỉ một trục có $k=2$, hai trục còn lại
   $k=1$)?
4. So sánh chi phí bộ nhớ đỉnh giữa cơ chế 3DGS gốc ($N\to 2N$ mỗi split) và SADGS đẳng hướng nặng
   ($N\to8N$ khi $\eta>1$ trên cả 3 trục) — với cùng một Gaussian cha, sau bao nhiêu vòng densify liên
   tiếp (mỗi vòng đều split hết) thì SADGS vượt 3DGS về số điểm, nếu ban đầu cả hai đều có 1 điểm?
5. Mục 7.8.2 cho $G_2$ nằm sát biên `low_ratio`$=0.80$ (đúng bằng ngưỡng `prune_ratio_threshold`, so
   sánh chặt `>`) nên không bị prune. Nếu một view khác quan sát thêm khiến `eta_low_count`$_2$ tăng lên
   $41/51\approx0.804$, hãy truy vết lại toàn bộ §7.8.3–7.8.6 xem kết luận nào thay đổi.
6. Giải thích tại sao `full_prune_mask` (tính từ `low_ratio` của quần thể **trước** densify) không được
   áp dụng lại cho các Gaussian con vừa sinh ra trong cùng vòng lặp (§7.8.6) — đây có phải một lỗ hổng
   thiết kế hay là chủ đích ("cho con một cơ hội trước khi đánh giá lại ở cửa sổ densify tiếp theo")?

---

## Đầu ra chuyển cho Phần 8

**Đầu ra chuyển cho Phần 8: $\mathcal G_1$ = quần thể Gaussian sau `densify_and_prune_structgs` tại
$t=5000$ (18 Gaussian, bảng đầy đủ ở §7.8.4–7.8.7).** Mỗi Gaussian mang đủ $59+1=60$ trường nội bộ
($\mu$: 3, $\tilde q$: 4, $\tilde s$: 3, $\tilde\alpha$: 1, $f_{\text{dc}}$: 3, $f_{\text{rest}}$: 45,
`filter_3D`: 1) — Phần 8 dùng $\mathcal G_1$ này để: (a) so sánh mô hình chi phí $T_{\text{iter}}$
trước/sau bước densify, và (b) ghi ra `point_cloud.ply` theo đúng layout **63 cột** thật của
`construct_list_of_attributes`/`save_ply` trong `SADGS/scene/gaussian_model.py` (khác 62 cột của bản
nháp trước — SADGS có thêm cột `filter_3D`).

[Phần 8 →](08-cost-model-ply.md)
