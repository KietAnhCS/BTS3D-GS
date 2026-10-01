# Công thức toán học trong `pipeline/score.py`

File này định nghĩa **công thức điểm tổng hợp (composite score)** của cuộc thi và hàm đánh giá trung bình trên một tập camera. Đây là file có nhiều công thức nhất trong 6 file của `pipeline/`.

---

Khai báo trọng số ở đầu file (dòng 9):

```python
W_LPIPS, W_SSIM, W_PSNR = 0.4, 0.3, 0.3
```

## 1. Chuẩn hoá PSNR (`composite_score`)

Trích nguyên văn (dòng 12–14):

```python
def composite_score(psnr_val, ssim_val, lpips_val, psnr_max=30.0):
    """Trả về (score, psnr_norm) đúng công thức ban tổ chức."""
    psnr_norm = torch.clamp(torch.as_tensor(float(psnr_val)) / float(psnr_max), 0.0, 1.0)
```

PSNR không bị chặn trên, cần quy về $[0,1]$ để cộng cùng thang với SSIM/LPIPS:

$$
\mathrm{PSNR}_{norm} = \mathrm{clip}\!\left(\frac{\mathrm{PSNR}}{\mathrm{PSNR}_{max}},\ 0,\ 1\right)
$$

với $\mathrm{PSNR}_{max}=30\,\mathrm{dB}$ mặc định (tham số `psnr_max`, dòng 12).

## 2. Công thức điểm tổng hợp (`composite_score`)

Trích nguyên văn (dòng 15–18):

```python
    ssim_t = torch.as_tensor(float(ssim_val))
    lpips_t = torch.as_tensor(float(lpips_val))
    score = W_LPIPS * (1.0 - lpips_t) + W_SSIM * ssim_t + W_PSNR * psnr_norm
    return float(score), float(psnr_norm)
```

$$
\mathrm{Score} = w_{LPIPS}\,(1-\mathrm{LPIPS}) + w_{SSIM}\,\mathrm{SSIM} + w_{PSNR}\,\mathrm{PSNR}_{norm}
$$

với trọng số cố định khai báo ở đầu file (dòng 9):

$$
w_{LPIPS}=0.4,\qquad w_{SSIM}=0.3,\qquad w_{PSNR}=0.3
$$

LPIPS càng nhỏ càng tốt nên được đảo dấu thành $(1-\mathrm{LPIPS})$ trước khi nhân trọng số; SSIM và $\mathrm{PSNR}_{norm}$ càng lớn càng tốt nên giữ nguyên chiều.

## 3. Trung bình trên tập camera (`evaluate_cameras`)

Trích nguyên văn (dòng 30–46):

```python
    cams = list(cams)
    if max_views:
        cams = cams[:max_views]
    if not cams:
        return None

    psnr_sum = ssim_sum = lpips_sum = 0.0
    for cam in cams:
        rendered = torch.clamp(render_structgs(cam, gaussians, pipe, background, mult)["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        psnr_sum += psnr_fn(rendered, gt).mean().item()
        ssim_sum += fast_ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).item()
        lpips_sum += lpips_fn(rendered, gt, net_type=lpips_net).mean().item()
        del rendered, gt

    n = len(cams)
    psnr_val, ssim_val, lpips_val = psnr_sum / n, ssim_sum / n, lpips_sum / n
```

Với $N$ camera có ảnh ground-truth, mỗi camera $j$ cho ra $(\mathrm{PSNR}_j, \mathrm{SSIM}_j, \mathrm{LPIPS}_j)$ (ảnh render và GT được clamp về $[0,1]$ trước khi đưa vào các hàm metric, dòng 38–39):

$$
\overline{\mathrm{PSNR}} = \frac{1}{N}\sum_{j=1}^{N}\mathrm{PSNR}_j,\qquad
\overline{\mathrm{SSIM}} = \frac{1}{N}\sum_{j=1}^{N}\mathrm{SSIM}_j,\qquad
\overline{\mathrm{LPIPS}} = \frac{1}{N}\sum_{j=1}^{N}\mathrm{LPIPS}_j
$$

(dòng 40–42 cộng dồn tổng, dòng 46 chia cho $N=$ `n = len(cams)`.) Tiếp theo, dòng 47 gọi lại `composite_score`:

```python
    score, psnr_norm = composite_score(psnr_val, ssim_val, lpips_val, psnr_max)
```

Ba giá trị trung bình này được đưa trực tiếp vào `composite_score` ở mục 2 để ra $\mathrm{Score}$ và $\mathrm{PSNR}_{norm}$ cuối cùng cho cả tập camera (không phải trung bình của từng Score riêng lẻ — Score được tính **sau khi** đã lấy trung bình ba metric thô).

Nếu `max_views` được truyền vào (dòng 31–32, `cams = cams[:max_views]`), chỉ $N=\min(|\text{cams}|, \text{max\_views})$ camera đầu tiên được dùng (không lấy mẫu ngẫu nhiên).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `W_LPIPS, W_SSIM, W_PSNR = 0.4, 0.3, 0.3` | $w_{LPIPS}=0.4,\ w_{SSIM}=0.3,\ w_{PSNR}=0.3$ |
| `torch.clamp(psnr_val/psnr_max, 0.0, 1.0)` | $\mathrm{PSNR}_{norm} = \mathrm{clip}(\mathrm{PSNR}/\mathrm{PSNR}_{max},0,1)$ |
| `W_LPIPS*(1-lpips_t) + W_SSIM*ssim_t + W_PSNR*psnr_norm` | $\mathrm{Score}=w_{LPIPS}(1-\mathrm{LPIPS})+w_{SSIM}\,\mathrm{SSIM}+w_{PSNR}\,\mathrm{PSNR}_{norm}$ |
| `psnr_sum += psnr_fn(...).mean().item()` (lặp qua `cams`, chia `n`) | $\overline{\mathrm{PSNR}}=\frac1N\sum_j \mathrm{PSNR}_j$ |
| `ssim_sum += fast_ssim(...).item()` (chia `n`) | $\overline{\mathrm{SSIM}}=\frac1N\sum_j \mathrm{SSIM}_j$ |
| `lpips_sum += lpips_fn(...).mean().item()` (chia `n`) | $\overline{\mathrm{LPIPS}}=\frac1N\sum_j \mathrm{LPIPS}_j$ |
| `cams = cams[:max_views]` | $N=\min(|\text{cams}|,\text{max\_views})$ |
