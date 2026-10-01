# Công thức toán học trong `pipeline/trainer.py`

File này bọc quanh `train.py` gốc của repo. Phần dưới đây **chỉ** trích các công thức/ngưỡng **riêng của `trainer.py`** (co giãn lịch theo `cfg.iterations`, EMA loss, tỉ lệ split/prune theo tần số) — loss cơ bản ($L1$, $D$-SSIM, $L2$) và cơ chế densify/prune gốc đã có trong `scene/gaussian_model.md`, không lặp lại chi tiết ở đây.

---

## 1. Co giãn lịch densify theo số vòng lặp (`build_args`)

Lịch gốc 3DGS được thiết kế cho 30k vòng; `trainer.py` co giãn theo tỉ lệ để dùng được với số vòng lặp tuỳ ý $n_{iter}$.

Trích nguyên văn (`pipeline/trainer.py`, dòng 29 và 33, trong hàm `build_args`):

```python
densify_until = max(1, int(round(getattr(cfg, "densify_until_frac", 0.5) * n_iter)))
```

```python
opacity_reset = max(1, int(round(getattr(cfg, "opacity_reset_frac", 0.2) * n_iter)))
```

$$
\text{densify\_until\_iter} = \max\big(1,\ \mathrm{round}(f_{d}\cdot n_{iter})\big), \qquad f_d = \texttt{cfg.densify\_until\_frac} \ (\text{mặc định } 0.5)
\tag{dòng 29}
$$

$$
\text{opacity\_reset\_interval} = \max\big(1,\ \mathrm{round}(f_{o}\cdot n_{iter})\big), \qquad f_o = \texttt{cfg.opacity\_reset\_frac}\ (\text{mặc định } 0.2)
\tag{dòng 33}
$$

## 2. Mốc prune theo tỉ lệ cố định (`train_scene`)

3DGS gốc prune cứng tại vòng 4000 và 8000 trên lịch 30k ($4000/30000\approx0.1\overline{3}$ không khớp — thực chất tỉ lệ gốc là so với `densify_until_iter=15000$: $4000/15000\approx0.267$, $8000/15000\approx0.533$). `trainer.py` giữ đúng hai tỉ lệ này và áp lên `densify_until_iter` đã co giãn ở mục 1.

Trích nguyên văn (`pipeline/trainer.py`, dòng 134–136, trong `train_scene`):

```python
# 3DGS gốc dùng 4000/8000 trên lịch 30k -> co theo cùng tỷ lệ với densify_until_iter.
prune_iterations = {max(1, round(0.267 * opt.densify_until_iter)),
                    max(1, round(0.533 * opt.densify_until_iter))}
```

$$
\text{prune\_iterations} = \Big\{\max(1,\mathrm{round}(0.267\cdot \text{densify\_until\_iter})),\ \max(1,\mathrm{round}(0.533\cdot \text{densify\_until\_iter}))\Big\}
\tag{dòng 135–136}
$$

Tại các vòng này (dòng 248–250), mọi Gaussian có $\alpha_i < 0.1$ bị xoá trực tiếp (không qua `densify_and_prune_structgs`):

```python
if iteration in prune_iterations:
    prune_mask = (gaussians.get_opacity < 0.1).squeeze()
    gaussians.prune_points(prune_mask)
```

## 3. Loss huấn luyện mỗi camera (vòng lặp batch)

Trích nguyên văn (`pipeline/trainer.py`, dòng 174–177, trong `train_scene`):

```python
ll1 = l1_loss(image, gt)
ll2 = l2_loss(image, gt)
ssim_value = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value) + opt.lambda_l2 * ll2
```

$$
\mathcal{L} = (1-\lambda_{dssim})\,L_1 + \lambda_{dssim}\,(1-\mathrm{SSIM}) + \lambda_{L2}\,L_2
\tag{dòng 177}
$$

(đây là công thức loss gốc của `train.py`/`utils/loss_utils.py`, được gọi lại nguyên văn trong `trainer.py`, không định nghĩa thêm — nêu lại để thấy vị trí dùng trong vòng lặp batch).

## 4. EMA của loss để hiển thị (`train_scene`)

Trung bình động mũ (Exponential Moving Average) của loss trung bình một batch.

Trích nguyên văn (`pipeline/trainer.py`, dòng 179 và 192):

```python
batch_loss += loss.item()
```

```python
ema_loss = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss
```

$$
\overline{\mathcal{L}}_{batch} = \frac{1}{B}\sum_{k=1}^{B}\mathcal{L}_k \qquad (B=\texttt{opt.batch\_size})
\tag{tích luỹ dòng 179, chia dòng 192}
$$

$$
\mathrm{ema}_{t} = 0.4\cdot \overline{\mathcal{L}}_{batch,t} + 0.6\cdot \mathrm{ema}_{t-1}
\tag{dòng 192}
$$

## 5. Tỉ lệ tần số cao/thấp để quyết định split/prune (`train_scene`)

Với mỗi Gaussian $i$ có $\texttt{accum\_view\_count}_i>0$ (tích luỹ số lần được tính $\eta$ qua các view, xem `utils/freq_utils.py` — không nằm trong file này).

Trích nguyên văn (`pipeline/trainer.py`, dòng 205–216):

```python
grads = gaussians.xyz_gradient_accum / gaussians.denom
grads[grads.isnan()] = 0.0
is_grad_high = torch.norm(grads, dim=-1) >= 1e-5

valid_mask = gaussians.accum_view_count > 0
high_ratio = torch.zeros_like(gaussians.accum_view_count)
low_ratio = torch.zeros_like(gaussians.accum_view_count)
high_ratio[valid_mask] = gaussians.eta_high_count[valid_mask] / gaussians.accum_view_count[valid_mask]
low_ratio[valid_mask] = gaussians.eta_low_count[valid_mask] / gaussians.accum_view_count[valid_mask]

split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high
prune_mask = (low_ratio > opt.prune_ratio_threshold) & valid_mask
```

$$
\text{is\_grad\_high}_i = \left\lVert \frac{\texttt{xyz\_gradient\_accum}_i}{\texttt{denom}_i}\right\rVert_2 \ge 10^{-5}
\tag{dòng 205–207}
$$

(NaN trong gradient trung bình được gán về 0 trước khi so sánh — dòng 206.)

$$
\text{high\_ratio}_i = \frac{\texttt{eta\_high\_count}_i}{\texttt{accum\_view\_count}_i}, \qquad
\text{low\_ratio}_i = \frac{\texttt{eta\_low\_count}_i}{\texttt{accum\_view\_count}_i}
\tag{dòng 209–213, chỉ tính trên valid\_mask, phần còn lại giữ 0}
$$

Điều kiện kích hoạt split/prune tuỳ chỉnh truyền vào `densify_and_prune_structgs`:

$$
\text{split\_mask}_i = (\text{high\_ratio}_i > \tau_{split}) \wedge \text{is\_grad\_high}_i, \qquad
\tau_{split}=\texttt{opt.split\_ratio\_threshold}
\tag{dòng 215}
$$

$$
\text{prune\_mask}_i = (\text{low\_ratio}_i > \tau_{prune}) \wedge (\texttt{accum\_view\_count}_i>0), \qquad
\tau_{prune}=\texttt{opt.prune\_ratio\_threshold}
\tag{dòng 216}
$$

## 6. Kết quả cuối cùng (`train_scene`)

Thời gian huấn luyện và VRAM đỉnh chỉ là thống kê trực tiếp từ `time.time()` và `torch.cuda.max_memory_allocated()`, không phải công thức dẫn xuất — không trích riêng.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `round(cfg.densify_until_frac * n_iter)` | $\text{densify\_until\_iter}=\max(1,\mathrm{round}(f_d\, n_{iter}))$ |
| `round(cfg.opacity_reset_frac * n_iter)` | $\text{opacity\_reset\_interval}=\max(1,\mathrm{round}(f_o\, n_{iter}))$ |
| `{round(0.267*densify_until), round(0.533*densify_until)}` | $\{\mathrm{round}(0.267\cdot D),\ \mathrm{round}(0.533\cdot D)\}$ |
| `(gaussians.get_opacity < 0.1)` tại `prune_iterations` | prune nếu $\alpha_i<0.1$ |
| `(1-lambda_dssim)*ll1 + lambda_dssim*(1-ssim) + lambda_l2*ll2` | $\mathcal{L}=(1-\lambda_{dssim})L_1+\lambda_{dssim}(1-\mathrm{SSIM})+\lambda_{L2}L_2$ |
| `batch_loss / opt.batch_size` | $\overline{\mathcal{L}}_{batch}=\frac1B\sum_k \mathcal{L}_k$ |
| `0.4*(batch_loss/opt.batch_size) + 0.6*ema_loss` | $\mathrm{ema}_t = 0.4\,\overline{\mathcal{L}}_{batch,t}+0.6\,\mathrm{ema}_{t-1}$ |
| `eta_high_count/accum_view_count` | $\text{high\_ratio}_i=\texttt{eta\_high\_count}_i/\texttt{accum\_view\_count}_i$ |
| `eta_low_count/accum_view_count` | $\text{low\_ratio}_i=\texttt{eta\_low\_count}_i/\texttt{accum\_view\_count}_i$ |
| `torch.norm(grads, dim=-1) >= 1e-5` | $\lVert \texttt{xyz\_gradient\_accum}_i/\texttt{denom}_i\rVert_2 \ge 10^{-5}$ |
| `(high_ratio > split_ratio_threshold) & is_grad_high` | $\text{split\_mask}_i=(\text{high\_ratio}_i>\tau_{split})\wedge\text{is\_grad\_high}_i$ |
| `(low_ratio > prune_ratio_threshold) & valid_mask` | $\text{prune\_mask}_i=(\text{low\_ratio}_i>\tau_{prune})\wedge(\texttt{accum\_view\_count}_i>0)$ |
