# Công thức toán học trong `metrics.py`

File `metrics.py` (gốc tại `C:\Users\kelly\OneDrive\Desktop\BTS_SADGS\metrics.py`) tính các chỉ số đánh giá chất lượng ảnh (SSIM, PSNR, LPIPS) giữa ảnh dựng (`renders`) và ảnh thật (`gt`) cho từng scene/method, rồi lưu trung bình và theo từng ảnh ra JSON. Bản thân `metrics.py` chỉ **gọi** các hàm `ssim()` (`utils/loss_utils.py`), `psnr()` (`utils/image_utils.py`) và `lpips()` (`lpipsPyTorch`); công thức toán học thật nằm trong các hàm bị gọi đó, nên các trích dẫn code dưới đây lấy từ đúng những file đó (ghi rõ nguồn).

Lệnh gọi thật trong vòng lặp đánh giá (`metrics.py`, dòng 73–76):

```python
for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
    ssims.append(ssim(renders[idx], gts[idx]))
    psnrs.append(psnr(renders[idx], gts[idx]))
    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
```

---

## 1. SSIM (Structural Similarity) — `ssim()` trong `utils/loss_utils.py`

### 1.1. Cửa sổ Gaussian 2D (`utils/loss_utils.py`, dòng 36–44)

```python
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
```

Cửa sổ Gaussian 1D $g_i \propto \exp\!\big(-\frac{(i-\lfloor K/2\rfloor)^2}{2\sigma^2}\big)$ được chuẩn hoá tổng $=1$ (chia cho `gauss.sum()`), với $K=11$ (`window_size`), $\sigma=1.5$. Cửa sổ 2D $w = g\,g^\top$ (ngoại tích), áp dụng đồng nhất cho mỗi kênh màu (`expand(channel, 1, window_size, window_size)`, dùng `groups=channel` khi convolution).

### 1.2. Thống kê cục bộ $\mu,\sigma^2,\sigma_{12}$ (`utils/loss_utils.py`, dòng 57–66)

```python
mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

mu1_sq = mu1.pow(2)
mu2_sq = mu2.pow(2)
mu1_mu2 = mu1 * mu2

sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
```

$$
\mu_1 = w * I_1, \qquad \mu_2 = w * I_2
$$

$$
\sigma_1^2 = w*(I_1^2) - \mu_1^2, \qquad \sigma_2^2 = w*(I_2^2) - \mu_2^2, \qquad \sigma_{12} = w*(I_1 I_2) - \mu_1\mu_2
$$

### 1.3. Công thức SSIM cuối cùng (`utils/loss_utils.py`, dòng 68–76)

```python
C1 = 0.01 ** 2
C2 = 0.03 ** 2

ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

if size_average:
    return ssim_map.mean()
else:
    return ssim_map.mean(1).mean(1).mean(1)
```

$$
\mathrm{SSIM}(I_1,I_2) = \frac{(2\mu_1\mu_2 + C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}
$$

với $C_1 = 0.01^2 = 10^{-4}$, $C_2 = 0.03^2 = 9\times10^{-4}$. Giá trị cuối gọi từ `metrics.py` là trung bình toàn bộ bản đồ SSIM, vì `ssim()` (`utils/loss_utils.py` dòng 46–54) gọi `_ssim(..., size_average=True)` theo mặc định — nhánh `ssim_map.mean()` được dùng, không phải nhánh `size_average=False`.

Lưu ý: đây là cài đặt SSIM "chuẩn" (sliding-window, thuần PyTorch) trong `utils/loss_utils.py`, **khác** với `fast_ssim` (thư viện `fused_ssim`) được dùng trong `train.py` — `metrics.py` không dùng `fused_ssim`.

## 2. PSNR (Peak Signal-to-Noise Ratio) — `psnr()` trong `utils/image_utils.py`

Trích nguyên văn toàn bộ hàm (`utils/image_utils.py`, dòng 17–19):

```python
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

$$
\mathrm{MSE}(I_1,I_2) = \frac{1}{N}\sum_{p} (I_{1,p} - I_{2,p})^2
$$

(tính trung bình trên toàn bộ pixel/kênh đã được "flatten" của mỗi ảnh trong batch — dòng `.view(img1.shape[0], -1).mean(1, keepdim=True)`, dòng 18)

$$
\mathrm{PSNR}(I_1,I_2) = 20\log_{10}\!\left(\frac{1}{\sqrt{\mathrm{MSE}}}\right) = -10\log_{10}(\mathrm{MSE})
$$

(tương đương dạng chuẩn $10\log_{10}(\mathrm{MAX}^2/\mathrm{MSE})$ với $\mathrm{MAX}=1$, vì ảnh được chuẩn hoá $[0,1]$; code viết dưới dạng $20\log_{10}(1/\sqrt{\mathrm{MSE}})$, về mặt đại số tương đương $10\log_{10}(1/\mathrm{MSE})$).

Hàm liền kề `mse()` (`utils/image_utils.py`, dòng 14–15) định nghĩa y hệt công thức MSE dùng nội bộ trong `psnr()` (không được `metrics.py` gọi trực tiếp, chỉ `psnr()` được gọi):

```python
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
```

## 3. LPIPS

Gọi trực tiếp hàm `lpips(renders[idx], gts[idx], net_type='vgg')` (`metrics.py`, dòng 76, trích ở đầu tài liệu) từ thư viện `lpipsPyTorch` — một mạng perceptual loss học sẵn (VGG), không có công thức toán tường minh trong file `metrics.py` (black-box qua mạng neural, bản thân `metrics.py` chỉ gọi và tổng hợp kết quả).

## 4. Tổng hợp thống kê (per scene/method)

Trích nguyên văn (`metrics.py`, dòng 78–88):

```python
print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
print("")

full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                        "LPIPS": torch.tensor(lpipss).mean().item()})
per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
```

Với tập ảnh $V$ của một method:

$$
\overline{\mathrm{SSIM}} = \frac{1}{|V|}\sum_{v\in V}\mathrm{SSIM}_v, \qquad
\overline{\mathrm{PSNR}} = \frac{1}{|V|}\sum_{v\in V}\mathrm{PSNR}_v, \qquad
\overline{\mathrm{LPIPS}} = \frac{1}{|V|}\sum_{v\in V}\mathrm{LPIPS}_v
$$

Tất cả tính bằng `torch.tensor([...]).mean()` — trung bình số học đơn giản (không trọng số), rồi lưu cả giá trị trung bình (`full_dict`) lẫn từng ảnh (`per_view_dict`, không tính trung bình — giữ nguyên giá trị từng ảnh gắn với tên file, qua `zip(..., image_names)`).

### Lưu file kết quả (`metrics.py`, dòng 90–93)

```python
with open(scene_dir + "/results.json", 'w') as fp:
    json.dump(full_dict[scene_dir], fp, indent=True)
with open(scene_dir + "/per_view.json", 'w') as fp:
    json.dump(per_view_dict[scene_dir], fp, indent=True)
```

Không phải công thức toán, nhưng xác nhận hai output: `results.json` (trung bình) và `per_view.json` (theo từng ảnh).

---

## Bảng hằng số/ngưỡng

| Ký hiệu | Giá trị | Vị trí |
|---|---|---|
| $C_1$ | $0.01^2 = 10^{-4}$ | `utils/loss_utils.py` dòng 68 (hằng số ổn định SSIM) |
| $C_2$ | $0.03^2 = 9\times10^{-4}$ | `utils/loss_utils.py` dòng 69 (hằng số ổn định SSIM) |
| window_size (SSIM) | $11$ | `ssim()` mặc định, `utils/loss_utils.py` dòng 46 |
| $\sigma$ (SSIM window) | $1.5$ | `create_window()` gọi `gaussian(window_size, 1.5)`, `utils/loss_utils.py` dòng 41 |
| $\mathrm{MAX}$ (PSNR) | $1.0$ (ảnh đã chuẩn hoá $[0,1]$, ngầm định) | `psnr()`, `utils/image_utils.py` |
| `net_type` (LPIPS) | `'vgg'` | `metrics.py` dòng 76 |

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Nguồn | Công thức |
|---|---|---|
| `mu1 = F.conv2d(img1, window, ...)` | `utils/loss_utils.py:57` | $\mu_1 = w*I_1$ |
| `mu2 = F.conv2d(img2, window, ...)` | `utils/loss_utils.py:58` | $\mu_2 = w*I_2$ |
| `sigma1_sq = F.conv2d(img1*img1, window, ...) - mu1_sq` | `utils/loss_utils.py:64` | $\sigma_1^2 = w*(I_1^2)-\mu_1^2$ |
| `sigma2_sq = F.conv2d(img2*img2, window, ...) - mu2_sq` | `utils/loss_utils.py:65` | $\sigma_2^2 = w*(I_2^2)-\mu_2^2$ |
| `sigma12 = F.conv2d(img1*img2, window, ...) - mu1_mu2` | `utils/loss_utils.py:66` | $\sigma_{12} = w*(I_1I_2)-\mu_1\mu_2$ |
| `ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))` | `utils/loss_utils.py:71` | $\mathrm{SSIM}=\dfrac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}$ |
| `mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)` | `utils/image_utils.py:18` | $\mathrm{MSE} = \frac1N\sum (I_1-I_2)^2$ |
| `return 20 * torch.log10(1.0 / torch.sqrt(mse))` | `utils/image_utils.py:19` | $\mathrm{PSNR} = 20\log_{10}(1/\sqrt{\mathrm{MSE}}) = -10\log_{10}(\mathrm{MSE})$ |
| `lpips(renders[idx], gts[idx], net_type='vgg')` | `metrics.py:76` | $\mathrm{LPIPS}_{vgg}(I_1,I_2)$ (mạng học sẵn, không có công thức giải tích trong file) |
| `torch.tensor(ssims).mean()` | `metrics.py:83` | $\overline{\mathrm{SSIM}} = \frac1{|V|}\sum_v \mathrm{SSIM}_v$ |
| `torch.tensor(psnrs).mean()` | `metrics.py:84` | $\overline{\mathrm{PSNR}} = \frac1{|V|}\sum_v \mathrm{PSNR}_v$ |
| `torch.tensor(lpipss).mean()` | `metrics.py:85` | $\overline{\mathrm{LPIPS}} = \frac1{|V|}\sum_v \mathrm{LPIPS}_v$ |

---

## Kiểm chứng tính đúng sai

So với bản cũ: toàn bộ công thức toán học đã đúng (SSIM, MSE, PSNR, trung bình). Đã đọc lại thật `utils/loss_utils.py` (dòng 36–76) và `utils/image_utils.py` (dòng 14–19) để xác nhận số dòng và nội dung khớp 100% với trích dẫn. Một điểm làm rõ thêm so với bản cũ: `metrics.py` tự nó **không chứa** định nghĩa `ssim`/`psnr` — các công thức này nằm ở hai file `utils/*.py` khác, nên bảng trích dẫn ở trên ghi rõ nguồn từng dòng để tránh hiểu nhầm là code nằm trong `metrics.py`. Không phát hiện sai sót toán học nào.
