# Công thức toán học trong `train.py`

Tài liệu tổng hợp cơ sở toán học trong vòng lặp huấn luyện `training()`, bám sát đúng thứ tự và hệ số xuất hiện trong code.

---

## 1. Hàm mất mát tổng (RGB loss)

Trích nguyên văn (dòng 255–259):

```python
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            Ll2 = l2_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)+ opt.lambda_l2 * Ll2
```

Mỗi batch render một camera, tính trên ảnh dự đoán $I$ và ảnh ground-truth $I^{gt}$:

$$
\mathcal{L}_{1} = \frac{1}{N}\sum |I - I^{gt}| \qquad (\texttt{l1\_loss})
$$

$$
\mathcal{L}_{2} = \frac{1}{N}\sum (I - I^{gt})^2 \qquad (\texttt{l2\_loss})
$$

$$
\mathrm{SSIM}(I, I^{gt}) \qquad (\texttt{fast\_ssim}, \text{thuật toán } fused\_ssim)
$$

**Loss RGB tổng** (dòng 259):

$$
\mathcal{L}_{rgb} = (1-\lambda_{dssim})\,\mathcal{L}_1 \;+\; \lambda_{dssim}\,(1-\mathrm{SSIM}) \;+\; \lambda_{l2}\,\mathcal{L}_2
$$

Với giá trị mặc định trong `arguments/__init__.py`: $\lambda_{dssim}=0.2$, $\lambda_{l2}=2.0$.

Trích nguyên văn (dòng 261–267):

```python
            # Combine with main loss, scale by batch size for proper gradient averaging
            loss = rgb_loss #/ opt.batch_size
            loss.backward()
            
            # Accumulate for logging
            batch_loss += loss.item() #* opt.batch_size
            batch_rgb_loss += rgb_loss.item()
```

Loss cuối dùng để backward chính là $\mathcal{L}_{rgb}$ (biến `loss = rgb_loss`, dòng 262), không chia cho `batch_size` (phép chia `#/ opt.batch_size` bị comment out), nên khi `batch_size > 1`, gradient được **cộng dồn trực tiếp** qua `loss.backward()` cho từng camera trong vòng lặp batch, không có trung bình hoá tường minh ở bước backward.

Ghi chú: `tone_curve_loss` và `frequency_loss` tồn tại trong `utils/loss_utils.py` (trọng số `lambda_tone`, `lambda_freq`) nhưng **không được gọi** trong `train.py` ở phiên bản hiện tại (dòng 17 chỉ import `l1_loss, l2_loss, get_multiscale_structure_tensor_v1, get_multiscale_structure_tensor_v2`) — chỉ các hệ số đã nêu ở trên thực sự tham gia vào `loss`. Biến `batch_rgb_loss` (dòng 267) được cộng dồn nhưng không thấy đọc lại trong phần còn lại của vòng lặp đã đọc được.

---

## 2. EMA (Exponential Moving Average) cho loss hiển thị

Trích nguyên văn (dòng 70–71, khởi tạo, và dòng 286–288, cập nhật):

```python
    ema_loss_for_log = 0.0
    ema_rgb_loss_for_log = 0.0
```

```python
        with torch.no_grad():
            # Progress bar (use averaged batch loss)
            ema_loss_for_log = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss_for_log
```

Dùng để làm mượt log tiến trình (progress bar), không ảnh hưởng gradient:

$$
\overline{\mathcal{L}}^{ema}_{t} = 0.4 \cdot \left(\frac{\sum_{b} \mathcal{L}_{rgb}^{(b)}}{\texttt{batch\_size}}\right) + 0.6 \cdot \overline{\mathcal{L}}^{ema}_{t-1}
$$

tức hệ số làm mượt $\beta = 0.6$, trọng số mẫu mới $1-\beta = 0.4$. (biến `ema_loss_for_log`, dòng 288; biến `ema_rgb_loss_for_log` được khai báo ở dòng 71 nhưng không dùng tiếp trong vòng lặp).

---

## 3. Gradient trung bình dùng cho densify

Trích nguyên văn (dòng 326–333):

```python
                    # 1. Calculate Gradients (Standard 3DGS metric)
                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    
                    # 2. Create Gradient Mask 
                    # [CRITICAL] We MUST use this to prevent 7M points. 
                    # Only split if the geometry is struggling (high error).
                    is_grad_high = torch.norm(grads, dim=-1) >= 1e-5
```

$$
\bar{G}_i = \frac{\texttt{xyz\_gradient\_accum}_i}{\texttt{denom}_i}, \qquad \bar{G}_i \leftarrow 0 \text{ nếu NaN}
\qquad (\text{dòng 327–328})
$$

Điều kiện gradient cao (ngưỡng cố định $10^{-5}$, không phải `densify_grad_threshold`, dòng 333):

$$
\text{is\_grad\_high}_i = \lVert \bar{G}_i \rVert_2 \ge 10^{-5}
$$

---

## 4. Tiêu chí đa-góc-nhìn (Multiview Consistency) cho split/prune

Trích nguyên văn (dòng 335–353):

```python
                    # 3. [NEW] Multiview Consistency Criterion
                    # Compute ratios of high/low eta counts
                    valid_mask = gaussians.accum_view_count > 0
                    
                    high_ratio = torch.zeros_like(gaussians.accum_view_count)
                    low_ratio = torch.zeros_like(gaussians.accum_view_count)
                    high_ratio[valid_mask] = gaussians.eta_high_count[valid_mask] / gaussians.accum_view_count[valid_mask]
                    low_ratio[valid_mask] = gaussians.eta_low_count[valid_mask] / gaussians.accum_view_count[valid_mask]
                    
                    # 4. Split if consistently high eta across views 
                    split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high
                    
                    # 5. Compute average high eta 3ch for densification guidance
                    avg_high_eta_3ch = torch.zeros_like(gaussians.max_eta_3ch)
                    has_high = gaussians.eta_high_count > 0
                    avg_high_eta_3ch[has_high] = gaussians.eta_high_sum_3ch[has_high] / gaussians.eta_high_count[has_high].unsqueeze(1)
                    max_high_eta = gaussians.max_eta_3ch
                    # 6. Prune if consistently low eta across views 
                    prune_mask = (low_ratio > opt.prune_ratio_threshold) & valid_mask
```

Với bộ đếm tích luỹ qua nhiều view: `eta_high_count`, `eta_low_count`, `accum_view_count` (được cập nhật online trong `update_freq_stats_online`, nằm ngoài file này nhưng điều khiển từ `train.py`, dòng 276):

$$
\text{high\_ratio}_i = \frac{\text{eta\_high\_count}_i}{\text{accum\_view\_count}_i}, \qquad
\text{low\_ratio}_i = \frac{\text{eta\_low\_count}_i}{\text{accum\_view\_count}_i}
\quad (\text{chỉ tính khi accum\_view\_count}_i>0,\ \text{dòng 341–342})
$$

Mặt nạ split theo điều kiện kết hợp (dòng 345):

$$
\text{split\_mask}_i = \big(\text{high\_ratio}_i > \tau_{split}\big) \wedge \text{is\_grad\_high}_i
\qquad (\tau_{split} = \texttt{opt.split\_ratio\_threshold})
$$

Mặt nạ prune theo tỉ lệ thấp (dòng 353):

$$
\text{prune\_mask}_i = \big(\text{low\_ratio}_i > \tau_{prune}\big) \wedge (\text{accum\_view\_count}_i>0)
\qquad (\tau_{prune} = \texttt{opt.prune\_ratio\_threshold})
$$

Trung bình $\eta$ cao dùng làm "hướng dẫn" số lượng tách theo trục (dòng 348–350):

$$
\overline{\eta}^{high}_i = \frac{\texttt{eta\_high\_sum\_3ch}_i}{\texttt{eta\_high\_count}_i} \quad (\text{chỉ nơi } \texttt{eta\_high\_count}_i>0)
$$

Nhưng thực tế lời gọi hàm dưới đây (dòng 362–374, trích đoạn liên quan) truyền `max_eta_3ch=max_high_eta` với `max_high_eta = gaussians.max_eta_3ch` (dòng 351, 373):

```python
                    gaussians.densify_and_prune_structgs(
                        max_screen_size=size_threshold,
                        min_opacity=0.1, # 0.005 is a good default
                        extent=scene.cameras_extent,
                        radii=radii,
                        args=opt,
                        importance_score=gaussians.accum_view_count,
                        pruning_score=None,
                        custom_split_mask=split_mask,
                        custom_prune_mask=prune_mask,
                        viewspace_points_indices=None,
                        max_eta_3ch=max_high_eta  # Use max high eta for shaping the split
                    )
```

tức giá trị $\eta$ cao nhất tích luỹ được (`gaussians.max_eta_3ch`), không phải trung bình $\overline{\eta}^{high}$ — biến `avg_high_eta_3ch` được tính ở dòng 348–350 nhưng **không được dùng** để gọi hàm.

Trích nguyên văn phần reset (dòng 376–386):

```python
                    # Reset accumulators
                    gaussians.accum_eta.zero_()
                    gaussians.accum_view_count.zero_()
                    gaussians.max_eta_3ch.zero_()
                    gaussians.accum_weights_valid.zero_()
                    # Reset multiview consistency accumulators
                    gaussians.eta_high_count.zero_()
                    gaussians.eta_high_sum_3ch.zero_()
                    gaussians.eta_mid_count.zero_()
                    gaussians.eta_mid_sum_3ch.zero_()
                    gaussians.eta_low_count.zero_()
```

Sau mỗi vòng densify, các bộ tích luỹ eta/đếm được reset về 0 (`accum_eta`, `accum_view_count`, `max_eta_3ch`, `accum_weights_valid`, `eta_high_count`, `eta_high_sum_3ch`, `eta_mid_count`, `eta_mid_sum_3ch`, `eta_low_count`).

---

## 5. Lịch trình / ngưỡng vòng lặp huấn luyện

Trích nguyên văn các điều kiện liên quan (dòng 216–217, 316, 321–324, 388–390, 402–403, 405–409, 412–417, 422–434):

```python
        if iteration % 1 == 0:
            gaussians.oneupSHdegree()
```

```python
            if iteration < opt.densify_until_iter:
```

```python
                is_normal_densification = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
                is_warmup_densification = opt.warmup_densification and iteration > opt.densify_from_iter and iteration % 100 == 0 and not is_normal_densification
                if is_normal_densification:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
```

```python
                elif is_warmup_densification:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
```

```python
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(opt.opacity_reset_decay)
```

```python
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    if opt.compute_3d_filter:
                        gaussians.compute_3D_filter(cameras=scene.getTrainCameras())
```

```python
            if iteration in prune_iterations:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack)

                prune_mask = (gaussians.get_opacity < 0.1).squeeze()
                gaussians.prune_points(prune_mask)
```

```python
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                elif opt.optimizer_type == "hybrid":
                    visible = radii > 0
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    gaussians.shoptimizer.step(visible, radii.shape[0])
                    gaussians.shoptimizer.zero_grad(set_to_none = True)
```

| Điều kiện (code) | Ý nghĩa |
|---|---|
| `iteration % 1 == 0` → `oneupSHdegree()` | Tăng bậc SH mỗi iteration (thực chất là mỗi bước, do điều kiện luôn đúng) |
| `iteration < opt.densify_until_iter` | Cửa sổ thời gian cho phép densify/accumulate thống kê |
| `iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0` | Điều kiện densify "chuẩn" |
| `opt.warmup_densification and iteration > opt.densify_from_iter and iteration % 100 == 0 and not is_normal_densification` | Điều kiện densify "warm-up" (3DGS gốc, tần suất 100 bước) |
| `size_threshold = 20 if iteration > opt.opacity_reset_interval else None` | Ngưỡng bán kính màn hình tối đa để prune (đơn vị pixel) chỉ áp dụng sau lần reset opacity đầu tiên |
| `iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter)` | Reset opacity định kỳ (hoặc ngay từ đầu nếu nền trắng) |
| `iteration % 100 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations - 100` | Cập nhật lại bộ lọc 3D (`compute_3D_filter`) định kỳ sau khi densify kết thúc, trừ 100 bước cuối |
| `iteration in prune_iterations` (mặc định `[4000, 8000]`) | Prune cứng theo opacity: $\text{prune}_i = (\alpha_i < 0.1)$ |
| `iteration < opt.iterations` | Điều kiện thực hiện bước optimizer (3 chế độ: `default`, `sparse_adam`, `hybrid`) |

Tham số truyền cứng (hard-coded, không lấy từ `opt`) trong lời gọi `densify_and_prune_structgs` (dòng 364, trích trong khối ở §4):

```python
                        min_opacity=0.1, # 0.005 is a good default
```

$$
\alpha_{min} = 0.1 \quad (\text{comment trong code: "0.005 is a good default"})
$$

---

## 6. Chuẩn hoá thời gian và bộ nhớ (không phải công thức mô hình, chỉ là phép tính đo đạc)

Trích nguyên văn (dòng 290–292, 304, 436–440):

```python
            gb_unit = 1024 ** 3
            alloc = torch.cuda.memory_allocated() / gb_unit
            rsrv = torch.cuda.memory_reserved() / gb_unit
```

```python
            iter_time = iter_start.elapsed_time(iter_end)
```

```python
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3
```

$$
t_{iter} = \text{iter\_start.elapsed\_time(iter\_end)}, \qquad
t_{optim} = \text{optim\_start.elapsed\_time(optim\_end)}
$$

$$
T_{total} \mathrel{+}= \frac{t_{iter}+t_{optim}}{1000}\ \text{(ms} \to \text{s)}
$$

$$
\text{GPU}_{alloc} = \frac{\texttt{torch.cuda.memory\_allocated()}}{1024^3}\ \text{GB}, \qquad
\text{GPU}_{rsrv} = \frac{\texttt{torch.cuda.memory\_reserved()}}{1024^3}\ \text{GB}
$$

---

## 7. Đánh giá định kỳ (`training_report`)

Trích nguyên văn (dòng 484–498, hàm `training_report`):

```python
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
```

Với mỗi view trong tập test/train mẫu:

$$
I = \mathrm{clip}(\text{render}(v),0,1), \qquad I^{gt} = \mathrm{clip}(\text{original\_image}(v),0,1)
$$

$$
\overline{L1} = \frac{1}{|V|}\sum_v \mathcal{L}_1(I,I^{gt}), \quad
\overline{\mathrm{PSNR}} = \frac{1}{|V|}\sum_v \mathrm{PSNR}(I,I^{gt}), \quad
\overline{\mathrm{SSIM}} = \frac{1}{|V|}\sum_v \mathrm{SSIM}(I,I^{gt}), \quad
\overline{\mathrm{LPIPS}} = \frac{1}{|V|}\sum_v \mathrm{LPIPS}_{vgg}(I,I^{gt})
$$

(hàm `psnr` lấy từ `utils/image_utils.py`, `fast_ssim` là `fused_ssim`, `lpips` dùng mạng VGG — `net_type='vgg'`.) Lưu ý: `training_report` được định nghĩa trong file nhưng lời gọi thật của nó ở dòng 306 đang bị **comment out** trong vòng lặp chính (`# training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_structgs, (pipe, background, opt.mult))`), nên trong phiên bản hiện tại của `train.py` hàm này **không được gọi** trong quá trình huấn luyện thực tế — chỉ còn tồn tại như code chết/dự phòng.

---

## 8. Bảng hằng số/ngưỡng xuất hiện trong `train.py`

| Ký hiệu / biến | Giá trị / nguồn | Vai trò |
|---|---|---|
| $\lambda_{dssim}$ | `opt.lambda_dssim` (mặc định 0.2) | Trọng số SSIM trong loss RGB |
| $\lambda_{l2}$ | `opt.lambda_l2` (mặc định 2.0) | Trọng số L2 trong loss RGB |
| $\beta_{ema}$ | $0.6$ (hard-code) | Hệ số làm mượt EMA loss hiển thị |
| ngưỡng gradient split (multiview) | $10^{-5}$ (hard-code, biến `is_grad_high`) | Ngưỡng norm gradient view-space để cho phép split |
| $\tau_{split}$ | `opt.split_ratio_threshold` (0.8) | Ngưỡng tỉ lệ view có $\eta$ cao để split |
| $\tau_{prune}$ | `opt.prune_ratio_threshold` (0.8) | Ngưỡng tỉ lệ view có $\eta$ thấp để prune |
| `densify_from_iter` | 500 | Bắt đầu densify |
| `densify_until_iter` | 15000 | Kết thúc cửa sổ densify |
| `densification_interval` | 100 | Chu kỳ densify chuẩn |
| `opacity_reset_interval` | 3000 | Chu kỳ reset opacity |
| size_threshold | 20 (pixel) | Ngưỡng bán kính 2D tối đa để prune, chỉ sau reset đầu tiên |
| $\alpha_{min}$ (trong train.py) | 0.1 | Ngưỡng opacity tối thiểu, truyền cứng vào `densify_and_prune_structgs` |
| `prune_iterations` | mặc định $[4000, 8000]$ | Các bước prune cứng theo opacity $<0.1$ |
| `highresolution_index` | ngưỡng `image_width >= 800` | Đánh dấu camera độ phân giải cao (biến thu thập nhưng không thấy dùng tiếp trong đoạn code đọc được) |
| chu kỳ cập nhật thống kê tần số | `iteration % 10 == 0` | Gọi `update_freq_stats_online` |
| Adam eps / optimizer step | xem `gaussian_model.py` mục 4 | Lịch sparsify bước Adam theo `optimizer_step` |

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức |
|---|---|
| `Ll1 = l1_loss(image, gt_image)` | $\mathcal{L}_1 = \frac{1}{N}\sum\lvert I-I^{gt}\rvert$ |
| `Ll2 = l2_loss(image, gt_image)` | $\mathcal{L}_2 = \frac{1}{N}\sum (I-I^{gt})^2$ |
| `ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))` | $\mathrm{SSIM}(I,I^{gt})$ |
| `rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + opt.lambda_l2 * Ll2` | $\mathcal{L}_{rgb} = (1-\lambda_{dssim})\mathcal{L}_1 + \lambda_{dssim}(1-\mathrm{SSIM}) + \lambda_{l2}\mathcal{L}_2$ |
| `loss = rgb_loss` | $\mathcal{L} = \mathcal{L}_{rgb}$ (không chia batch_size) |
| `batch_loss += loss.item()` | $\sum_b \mathcal{L}_{rgb}^{(b)}$ |
| `ema_loss_for_log = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss_for_log` | $\overline{\mathcal{L}}^{ema}_t = 0.4\cdot\frac{\sum_b \mathcal{L}^{(b)}_{rgb}}{B}+0.6\,\overline{\mathcal{L}}^{ema}_{t-1}$ |
| `alloc = torch.cuda.memory_allocated() / gb_unit` | $\text{GPU}_{alloc}=\texttt{memory\_allocated}()/1024^3$ |
| `rsrv = torch.cuda.memory_reserved() / gb_unit` | $\text{GPU}_{rsrv}=\texttt{memory\_reserved}()/1024^3$ |
| `grads = gaussians.xyz_gradient_accum / gaussians.denom` | $\bar{G}_i = \texttt{xyz\_gradient\_accum}_i/\texttt{denom}_i$ |
| `grads[grads.isnan()] = 0.0` | $\bar{G}_i \leftarrow 0$ nếu $\bar{G}_i = \text{NaN}$ |
| `is_grad_high = torch.norm(grads, dim=-1) >= 1e-5` | $\lVert \bar{G}_i\rVert_2 \ge 10^{-5}$ |
| `high_ratio[valid_mask] = gaussians.eta_high_count[valid_mask] / gaussians.accum_view_count[valid_mask]` | $\text{high\_ratio}_i = \texttt{eta\_high\_count}_i/\texttt{accum\_view\_count}_i$ |
| `low_ratio[valid_mask] = gaussians.eta_low_count[valid_mask] / gaussians.accum_view_count[valid_mask]` | $\text{low\_ratio}_i = \texttt{eta\_low\_count}_i/\texttt{accum\_view\_count}_i$ |
| `split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high` | $\text{split}_i=(\text{high\_ratio}_i>\tau_{split})\wedge\text{is\_grad\_high}_i$ |
| `prune_mask = (low_ratio > opt.prune_ratio_threshold) & valid_mask` | $\text{prune}_i=(\text{low\_ratio}_i>\tau_{prune})\wedge(\texttt{accum\_view\_count}_i>0)$ |
| `avg_high_eta_3ch[has_high] = gaussians.eta_high_sum_3ch[has_high] / gaussians.eta_high_count[has_high].unsqueeze(1)` | $\overline{\eta}^{high}_i=\texttt{eta\_high\_sum\_3ch}_i/\texttt{eta\_high\_count}_i$ |
| `size_threshold = 20 if iteration > opt.opacity_reset_interval else None` | $r_{max}=20$ px, chỉ áp dụng khi $\text{iter} > \text{opacity\_reset\_interval}$ |
| `gaussians.reset_opacity(opt.opacity_reset_decay)` | $\alpha^{new}=\alpha\cdot\gamma$, $\gamma=\texttt{opacity\_reset\_decay}$ (xem `gaussian_model.md` §2.4) |
| `prune_mask = (gaussians.get_opacity < 0.1).squeeze()` | $\text{prune}_i = (\alpha_i < 0.1)$ |
| `iter_time = iter_start.elapsed_time(iter_end)` | $t_{iter}$ (ms, CUDA event) |
| `optim_time = optim_start.elapsed_time(optim_end)` | $t_{optim}$ (ms, CUDA event) |
| `total_time += (iter_time + optim_time) / 1e3` | $T_{total}\mathrel{+}=(t_{iter}+t_{optim})/1000$ |
| `psnr_test += psnr(image, gt_image).mean().double()` rồi `/= len(config['cameras'])` | $\overline{\mathrm{PSNR}}=\frac1{|V|}\sum_v \mathrm{PSNR}(I,I^{gt})$ |
| `ssim_test += fast_ssim(...)`, chia `len(config['cameras'])` | $\overline{\mathrm{SSIM}}=\frac1{|V|}\sum_v \mathrm{SSIM}(I,I^{gt})$ |
| `lpips_test += lpips(image, gt_image, net_type='vgg')`, chia `len(...)` | $\overline{\mathrm{LPIPS}}_{vgg}=\frac1{|V|}\sum_v \mathrm{LPIPS}_{vgg}(I,I^{gt})$ |
