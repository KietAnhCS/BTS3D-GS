# Công thức toán học trong `utils/loss_utils.py`

File này định nghĩa các hàm mất mát (loss) dùng để huấn luyện 3D Gaussian Splatting: L1, L2, SSIM dạng cửa sổ trượt (window-based, qua convolution Gaussian), loss "tone curve" có stop-gradient, và các hàm phụ trợ cho loss tần số không gian (structure-tensor, dùng ở các biến thể StructGS). Công thức bám sát đúng thứ tự code.

---

## 1. L1 loss (`l1_loss`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 22–23):

```python
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()
```

$$
\mathcal{L}_1(\hat{I}, I) = \frac{1}{N}\sum_{i} \big|\hat{I}_i - I_i\big|
$$

với $\hat{I}=$ `network_output` (ảnh render), $I=$ `gt` (ảnh ground-truth), trung bình trên **mọi** phần tử (toàn bộ $C\times H\times W$, không giữ chiều batch).

## 2. L2 loss (`l2_loss`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 25–26):

```python
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()
```

$$
\mathcal{L}_2(\hat{I}, I) = \frac{1}{N}\sum_{i} \big(\hat{I}_i - I_i\big)^2
$$

## 3. Tone-curve loss (`tone_curve_loss`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 28–34):

```python
def tone_curve_loss(network_output, gt, epsilon=1e-6,norm=2):
    """
    Implements the "gradient supervision" tone curve loss.
    L = mean( ((pred - gt) / (sg(pred) + epsilon))**2 )
    where sg indicates a stop-gradient.
    """
    return (((network_output - gt) / (network_output.detach() + epsilon)) ** norm).mean()
```

Loss "gradient supervision" với stop-gradient (sg) trên mẫu số, $\epsilon=10^{-6}$ mặc định, bậc chuẩn hoá `norm` (mặc định $=2$):

$$
\mathcal{L}_{tone} = \frac{1}{N}\sum_i \left( \frac{\hat{I}_i - I_i}{\mathrm{sg}(\hat{I}_i) + \epsilon} \right)^{\text{norm}}
$$

trong đó $\mathrm{sg}(\hat{I}_i)$ là `network_output.detach()` — giá trị không truyền gradient ngược qua mẫu số.

---

## 4. SSIM dạng cửa sổ trượt (Structural Similarity, window-based)

### 4.1. Cửa sổ Gaussian 1D (`gaussian`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 36–38):

```python
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
```

Với kích thước cửa sổ `window_size` và $\sigma=$ `sigma`, tâm cửa sổ tại $w_0 = \lfloor \text{window\_size}/2 \rfloor$ (chia nguyên), giá trị chưa chuẩn hoá tại vị trí nguyên $x\in\{0,\dots,\text{window\_size}-1\}$:

$$
g(x) = \exp\!\left(-\frac{(x-w_0)^2}{2\sigma^2}\right)
$$

Chuẩn hoá để tổng bằng 1:

$$
G_{1D}(x) = \frac{g(x)}{\sum_{x'} g(x')}
$$

(trong code, `create_window` luôn gọi `gaussian(window_size, 1.5)` — dòng 41 — tức $\sigma=1.5$ cố định).

### 4.2. Cửa sổ Gaussian 2D (`create_window`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 40–44):

```python
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
```

Lấy tích ngoài (outer product) của cửa sổ 1D với chính nó, tương ứng công thức Gaussian 2D tách biến (separable) với $\sigma_x=\sigma_y=\sigma=1.5$:

$$
G_{2D}(x,y) = G_{1D}(x)\cdot G_{1D}(y) \;\propto\; \exp\!\left(-\frac{(x-w_0)^2+(y-w_0)^2}{2\sigma^2}\right)
$$

tức đúng dạng chuẩn hoá của $G(x,y)=\exp(-(x^2+y^2)/2\sigma^2)$ (sau khi dịch tâm về $w_0$ và chuẩn hoá tổng $=1$). Cửa sổ 2D này được lặp lại (`expand`) cho từng kênh màu, dùng làm `groups=channel` trong convolution (áp dụng độc lập theo từng kênh, không trộn kênh).

### 4.3. Các thống kê cục bộ qua convolution (`_ssim`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 57–66):

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

Với $w=$ `window_size`, convolution $G_{2D}$ với padding $=\lfloor w/2 \rfloor$, `groups=channel` (depthwise), ký hiệu $*$ là phép tích chập 2D:

$$
\mu_1 = G_{2D} * \hat{I}, \qquad \mu_2 = G_{2D} * I
$$

$$
\mu_1^2 = \mu_1\odot\mu_1, \qquad \mu_2^2=\mu_2\odot\mu_2, \qquad \mu_1\mu_2 = \mu_1\odot\mu_2
$$

Phương sai và hiệp phương sai cục bộ (ước lượng qua $E[X^2]-E[X]^2$ dùng convolution làm toán tử kỳ vọng cục bộ):

$$
\sigma_1^2 = \big(G_{2D} * \hat{I}^2\big) - \mu_1^2
$$

$$
\sigma_2^2 = \big(G_{2D} * I^2\big) - \mu_2^2
$$

$$
\sigma_{12} = \big(G_{2D} * (\hat{I}\odot I)\big) - \mu_1\mu_2
$$

### 4.4. Hằng số ổn định $C_1, C_2$

Trích nguyên văn (`utils/loss_utils.py`, dòng 19–20 ở phạm vi module, và lặp lại y hệt ở dòng 68–69 bên trong `_ssim`):

```python
C1 = 0.01 ** 2
C2 = 0.03 ** 2
```

$$
C_1 = 0.01^2 = 10^{-4}, \qquad C_2 = 0.03^2 = 9\times10^{-4}
$$

### 4.5. Bản đồ SSIM (SSIM map) và giá trị cuối

Trích nguyên văn (`utils/loss_utils.py`, dòng 71–76):

```python
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
```

$$
\mathrm{SSIM}(\hat{I},I) = \frac{(2\mu_1\mu_2 + C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}
$$

tính theo từng pixel, từng kênh (bản đồ $\in\mathbb{R}^{B\times C\times H\times W}$). Giá trị SSIM cuối cùng:

- Nếu `size_average=True` (mặc định):

$$
\mathrm{SSIM}_{final} = \frac{1}{B\,C\,H\,W}\sum_{b,c,h,w} \mathrm{SSIM}(\hat{I},I)_{b,c,h,w}
$$

- Nếu `size_average=False`: trung bình theo $C,H,W$ nhưng giữ chiều batch:

$$
\mathrm{SSIM}_{final,b} = \frac{1}{C\,H\,W}\sum_{c,h,w}\mathrm{SSIM}(\hat{I},I)_{b,c,h,w}
$$

---

## 5. Loss tổng hợp dùng trong `train.py`

File `loss_utils.py` chỉ cung cấp các thành phần; **loss tổng hợp thực tế** được ghép ở `train.py` (không nằm trong `loss_utils.py`), dùng $\mathcal{L}_1$ từ hàm `l1_loss`, $\mathcal{L}_2$ từ `l2_loss`, và SSIM từ thư viện ngoài `fused_ssim` (CUDA, cùng công thức toán học SSIM window-based như mục 4, chỉ khác cài đặt tăng tốc):

$$
\mathcal{L}_{rgb} = (1-\lambda_{dssim})\,\mathcal{L}_1 + \lambda_{dssim}\,(1-\mathrm{SSIM}) + \lambda_{l2}\,\mathcal{L}_2
$$

với $\lambda_{dssim}=$ `opt.lambda_dssim`, $\lambda_{l2}=$ `opt.lambda_l2` là các siêu tham số huấn luyện. Loss cuối dùng để `backward()`: $\mathcal{L}_{total} = \mathcal{L}_{rgb}$.

---

## 6. Loss tần số không gian — frequency-domain (phụ trợ cho StructGS)

### 6.1. `frequency_loss`

Trích nguyên văn (`utils/loss_utils.py`, dòng 80–110):

```python
def frequency_loss(means2D, cov2D, st_map, H, W):
    # Normalize coordinates to [-1,1] for grid_sample
    nx = (means2D[:,0] / (W - 1)) * 2 - 1
    ny = (means2D[:,1] / (H - 1)) * 2 - 1
    grid = torch.stack([nx, ny], dim=-1).view(1,1,-1,2)

    # Sample structure tensor
    Sxx, Sxy, Syy = F.grid_sample(st_map, grid, align_corners=True, padding_mode='border').view(3, -1)

    # Compute directional energy eta = u^T J u for the Gaussian principal axis directions
    a, b, c = cov2D[:,0], cov2D[:,1], cov2D[:,2]  # ellipse parameters

    # 1. Compute Determinant (Area^2)
    det = a * c - b * b

    # 2. Compute Linear Scale (Radius approx)
    # sqrt(det) = Area (s^2). sqrt(sqrt(det)) = Radius (s).
    sigma_area = torch.sqrt(det + 1e-6)
    sigma_linear = torch.sqrt(sigma_area + 1e-6)

    # 3. Frequency Threshold (Wavelength in pixels)
    # Ensure st_map is normalized so freq_target is in meaningful units (e.g., 0.0 to 1.0)
    freq_target = torch.sqrt(Sxx + Syy + 1e-6)

    wavelength_limit = 1.0 / (freq_target + 1e-6)

    # 4. Loss: Penalize if Radius > Wavelength
    # Using 'sigma_linear' ensures we compare pixels to pixels
    loss = torch.relu(sigma_linear - wavelength_limit).mean()

    return loss
```

Lấy mẫu structure tensor $\{S_{xx},S_{xy},S_{yy}\}$ tại vị trí chiếu 2D $\mathbf{m}_i=(m_{i,x},m_{i,y})$ của từng Gaussian bằng nội suy song tuyến (`grid_sample`), toạ độ chuẩn hoá về $[-1,1]$ (dòng 82–83):

$$
n_x = \frac{m_{i,x}}{W-1}\cdot 2 - 1, \qquad n_y = \frac{m_{i,y}}{H-1}\cdot 2 - 1
$$

Với $\mathrm{cov2D}_i = (a_i,b_i,c_i)$ là 3 tham số ellipse hiệp phương sai 2D của Gaussian (dòng 90), định thức (diện tích bình phương, dòng 93):

$$
\det_i = a_i c_i - b_i^2
$$

Bán kính xấp xỉ (độ lệch chuẩn tuyến tính), qua 2 lần căn bậc hai với epsilon $=10^{-6}$ chống âm (dòng 97–98):

$$
\sigma^{area}_i = \sqrt{\det_i + 10^{-6}}, \qquad \sigma^{linear}_i = \sqrt{\sigma^{area}_i + 10^{-6}}
$$

Tần số mục tiêu và bước sóng giới hạn (dòng 102, 104):

$$
f_{target,i} = \sqrt{S_{xx,i}+S_{yy,i}+10^{-6}}, \qquad \lambda_{limit,i} = \frac{1}{f_{target,i}+10^{-6}}
$$

Loss phạt khi bán kính Gaussian vượt quá bước sóng cho phép (ReLU — chỉ phạt phần dương, dòng 108):

$$
\mathcal{L}_{freq} = \frac{1}{N}\sum_i \mathrm{ReLU}\big(\sigma^{linear}_i - \lambda_{limit,i}\big)
$$

### 6.2. `frequency_loss_simple`

Trích nguyên văn (`utils/loss_utils.py`, dòng 389–392):

```python
def frequency_loss_simple(rendered_image, st_map):
    pred_st = get_structure_tensor_torch(rendered_image.unsqueeze(0))
    loss= l1_loss(pred_st, st_map)
    return loss
```

So sánh trực tiếp structure tensor của ảnh render với structure tensor mục tiêu bằng L1:

$$
\mathcal{L}_{freq,simple} = \mathcal{L}_1\big(\mathrm{ST}(\hat{I}),\, \mathrm{ST}_{target}\big)
$$

với $\mathrm{ST}(\cdot)$ là `get_structure_tensor_torch`.

### 6.3. Structure tensor Di Zenzo đa kênh (`get_structure_tensor_torch`)

Trích nguyên văn phần Sobel + tổng năng lượng theo kênh (`utils/loss_utils.py`, dòng 187–208):

```python
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device=image_tensor.device).view(1,1,3,3)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, device=image_tensor.device).view(1,1,3,3)

    # Repeat kernel for C channels (e.g., 3 for RGB)
    weight_x = sobel_x_kernel.repeat(C, 1, 1, 1)
    weight_y = sobel_y_kernel.repeat(C, 1, 1, 1)

    # Ix, Iy will be shape (B, C, H, W)
    Ix = F.conv2d(img_smooth, weight_x, padding=1, groups=C)
    Iy = F.conv2d(img_smooth, weight_y, padding=1, groups=C)

    # 3. Compute Products per Channel
    Ixx_c = Ix**2
    Ixy_c = Ix*Iy
    Iyy_c = Iy**2

    # 4. SUM ACROSS CHANNELS (Di Zenzo's Method)
    Ixx = Ixx_c.sum(dim=1, keepdim=True)
    Ixy = Ixy_c.sum(dim=1, keepdim=True)
    Iyy = Iyy_c.sum(dim=1, keepdim=True)
```

Làm mờ Gaussian ảnh gốc với $\sigma$ (tiền xử lý, dòng 183: `img_smooth = fast_gaussian_blur(image_tensor, k_size, sigma)`), rồi tính đạo hàm Sobel $I_x, I_y$ trên từng kênh màu độc lập (`groups=C`). Tensor cấu trúc theo Di Zenzo (tổng năng lượng trên mọi kênh màu để không bỏ sót biên đẳng độ sáng — iso-luminant edges):

$$
I_{xx} = \sum_{channel} I_x^2, \qquad I_{xy} = \sum_{channel} I_x I_y, \qquad I_{yy} = \sum_{channel} I_y^2
$$

Làm mờ (tích hợp cửa sổ) các thành phần trên với $\rho$ và chuẩn hoá theo giá trị lớn nhất của trace trong batch (dòng 215–227):

```python
    Sxx = fast_gaussian_blur(Ixx, k_size_rho, rho)
    Sxy = fast_gaussian_blur(Ixy, k_size_rho, rho)
    Syy = fast_gaussian_blur(Iyy, k_size_rho, rho)

    magnitude = Sxx + Syy
    max_val = torch.amax(magnitude, dim=(1, 2, 3), keepdim=True) + 1e-6 # Use batch-wise maximum

    Sxx = Sxx / max_val
    Sxy = Sxy / max_val
    Syy = Syy / max_val
```

$$
S_{xx} = G_\rho * I_{xx}, \qquad S_{xy} = G_\rho * I_{xy}, \qquad S_{yy} = G_\rho * I_{yy}
$$

$$
M = \max_{h,w}(S_{xx}+S_{yy}) + 10^{-6}, \qquad (S_{xx},S_{xy},S_{yy}) \leftarrow (S_{xx},S_{xy},S_{yy})/M
$$

### 6.4. Structure tensor đa tỉ lệ (`get_multiscale_structure_tensor_v1`, `v2`)

Trích nguyên văn vòng lặp theo octave (`utils/loss_utils.py`, dòng 262–311, hàm `v1`; cấu trúc tương tự ở `v2` dòng 335–377):

```python
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)

        sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))
        next_smooth = fast_gaussian_blur(current_smooth, None, sigma_inc)

        # Band Response (Difference of Gaussians)
        band_response = (current_smooth - next_smooth).pow(2).sum(dim=1, keepdim=True).sqrt()
        ...
        # Normalize to orientation-only
        trace_i = Sxx_i + Syy_i + 1e-6
        Sxx_norm_i = Sxx_i / trace_i
        Sxy_norm_i = Sxy_i / trace_i
        Syy_norm_i = Syy_i / trace_i

        # Weight by band energy
        weight_i = band_response.pow(power_factor)

        target_wavelength_sq_inv = band_freq**2

        accum_Sxx += Sxx_norm_i * weight_i * target_wavelength_sq_inv
        accum_Sxy += Sxy_norm_i * weight_i * target_wavelength_sq_inv
        accum_Syy += Syy_norm_i * weight_i * target_wavelength_sq_inv
        accum_weight += weight_i

        current_smooth = next_smooth
        sigma_accum = target_sigma

    # Final normalization
    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)
```

Với mức tỉ lệ (octave) $i=0,\dots,L-1$, hệ số bước octave $r=$ `octave_step`, $\sigma_{base}=$ `base_sigma`:

$$
\sigma_i = \sigma_{base}\cdot r^{\,i}, \qquad f_i = \frac{1}{r^{\,i}} \ \text{(tần số băng)}
$$

Làm mờ tăng dần bằng độ lệch chuẩn gia tăng (cộng phương sai độc lập — tính chất Gaussian chập Gaussian):

$$
\sigma_{inc} = \sqrt{\max(10^{-6},\ \sigma_i^2 - \sigma_{i-1}^2)}
$$

Đáp ứng băng tần (band response, kiểu Difference-of-Gaussians trên toàn kênh màu):

$$
R_i = \sqrt{\sum_{channel}\big(I_{smooth,i-1} - I_{smooth,i}\big)^2}
$$

Trọng số theo năng lượng băng với luỹ thừa `power_factor` ($p$):

$$
w_i = R_i^{\,p}
$$

Chuẩn hoá thành phần định hướng (orientation-only, chia cho trace) và tích luỹ có trọng số theo nghịch đảo bình phương bước sóng $f_i^2$:

$$
\hat{S}_{xx,i} = \frac{S_{xx,i}}{S_{xx,i}+S_{yy,i}+10^{-6}}, \quad\text{(tương tự cho } \hat S_{xy,i}, \hat S_{yy,i}\text{)}
$$

$$
\mathrm{Acc}_{xx} \mathrel{+}= \hat{S}_{xx,i}\, w_i\, f_i^2, \qquad \mathrm{Acc}_w \mathrel{+}= w_i
$$

Kết quả cuối cùng (chuẩn hoá theo tổng trọng số):

$$
S_{xx}^{final} = \frac{\mathrm{Acc}_{xx}}{\mathrm{Acc}_w + 10^{-6}} \quad \text{(tương tự cho } S_{xy}^{final}, S_{yy}^{final}\text{)}
$$

Khác biệt giữa v1 và v2: v1 tính structure tensor **một lần** ở scale gốc (dòng 241: `base_st = get_structure_tensor_torch(image_tensor, sigma=base_sigma)`) rồi chỉ làm mờ lại (blur) $S_{xx,base},S_{xy,base},S_{yy,base}$ theo từng tỉ lệ (dòng 282–284); v2 tính **lại structure tensor Di Zenzo đầy đủ** (mục 6.3, qua lệnh gọi `get_structure_tensor_torch(next_smooth, ...)` ở dòng 357) tại ảnh đã làm mờ của từng tỉ lệ — không ảnh hưởng công thức tích luỹ cuối, chỉ khác cách sinh $S_{xx,i},S_{xy,i},S_{yy,i}$ đầu vào.

### 6.5. Ước lượng số Gaussian cần thiết (`estimate_required_gaussians`)

Trích nguyên văn (`utils/loss_utils.py`, dòng 406–435):

```python
    B, C, H, W = image.shape
    total_pixels = H * W

    st_map = get_multiscale_structure_tensor(image)

    Sxx = st_map[:, 0, :, :]
    Syy = st_map[:, 2, :, :]

    local_energy = Sxx + Syy

    total_structure_energy = local_energy.sum().item()

    count_coverage = total_pixels * base_density

    count_detail = total_structure_energy * detail_sensitivity

    estimated_count = int(count_coverage + count_detail)
```

Năng lượng cục bộ (trace của structure tensor đa tỉ lệ):

$$
E_{local} = S_{xx} + S_{yy}
$$

Tổng năng lượng cấu trúc trên toàn ảnh:

$$
E_{total} = \sum_{h,w} E_{local}(h,w)
$$

Ước lượng số Gaussian cần thiết, kết hợp một số lượng nền tối thiểu (coverage) theo diện tích ảnh và một lượng bổ sung theo năng lượng cấu trúc (chi tiết/texture), với $\rho_{base}=$ `base_density`, $k=$ `detail_sensitivity`:

$$
N_{coverage} = H\cdot W \cdot \rho_{base}, \qquad N_{detail} = E_{total}\cdot k
$$

$$
N_{est} = \lfloor N_{coverage} + N_{detail} \rfloor
$$

**Lưu ý phát hiện khi đọc lại code (không phải lỗi công thức, nhưng cần ghi chú):** hàm `estimate_required_gaussians` (dòng 394–437) gọi `get_multiscale_structure_tensor(image)` ở dòng 411, nhưng file `utils/loss_utils.py` chỉ định nghĩa `get_multiscale_structure_tensor_v1` và `get_multiscale_structure_tensor_v2` (không có hàm tên `get_multiscale_structure_tensor` không hậu tố) — nếu hàm này được gọi trong thực tế (không chỉ đọc code tĩnh) sẽ ném `NameError`, trừ khi một alias `get_multiscale_structure_tensor = get_multiscale_structure_tensor_v1` (hoặc `v2`) được định nghĩa ở nơi khác (ví dụ nơi import hàm này) mà không nằm trong phạm vi file `loss_utils.py`.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `torch.abs((network_output - gt)).mean()` | $\mathcal{L}_1 = \frac{1}{N}\sum_i \lvert \hat{I}_i - I_i\rvert$ |
| `((network_output - gt) ** 2).mean()` | $\mathcal{L}_2 = \frac{1}{N}\sum_i (\hat{I}_i - I_i)^2$ |
| `((network_output - gt) / (network_output.detach() + epsilon)) ** norm).mean()` | $\mathcal{L}_{tone}=\frac{1}{N}\sum_i\left(\dfrac{\hat{I}_i-I_i}{\mathrm{sg}(\hat{I}_i)+\epsilon}\right)^{norm}$ |
| `exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))` | $g(x)=\exp\!\big(-(x-w_0)^2/2\sigma^2\big)$ |
| `gauss / gauss.sum()` | $G_{1D}(x) = g(x)/\sum_{x'} g(x')$ |
| `_1D_window.mm(_1D_window.t())` | $G_{2D}(x,y) = G_{1D}(x) G_{1D}(y)$ |
| `gaussian(window_size, 1.5)` | $\sigma = 1.5$ |
| `F.conv2d(img1, window, ...)` | $\mu_1 = G_{2D} * \hat{I}$ |
| `F.conv2d(img2, window, ...)` | $\mu_2 = G_{2D} * I$ |
| `mu1.pow(2)` | $\mu_1^2$ |
| `mu2.pow(2)` | $\mu_2^2$ |
| `mu1 * mu2` | $\mu_1\mu_2$ |
| `F.conv2d(img1*img1, window,...) - mu1_sq` | $\sigma_1^2 = (G_{2D}*\hat{I}^2) - \mu_1^2$ |
| `F.conv2d(img2*img2, window,...) - mu2_sq` | $\sigma_2^2 = (G_{2D}*I^2) - \mu_2^2$ |
| `F.conv2d(img1*img2, window,...) - mu1_mu2` | $\sigma_{12} = (G_{2D}*(\hat{I}\odot I)) - \mu_1\mu_2$ |
| `C1 = 0.01 ** 2` | $C_1 = 10^{-4}$ |
| `C2 = 0.03 ** 2` | $C_2 = 9\times10^{-4}$ |
| `((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))` | $\mathrm{SSIM}=\dfrac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}$ |
| `ssim_map.mean()` | $\mathrm{SSIM}_{final}=\frac{1}{BCHW}\sum \mathrm{SSIM}$ |
| `ssim_map.mean(1).mean(1).mean(1)` | $\mathrm{SSIM}_{final,b}=\frac{1}{CHW}\sum_{c,h,w}\mathrm{SSIM}_{b,c,h,w}$ |
| `(1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) + opt.lambda_l2 * Ll2` (train.py) | $\mathcal{L}_{rgb}=(1-\lambda_{dssim})\mathcal{L}_1+\lambda_{dssim}(1-\mathrm{SSIM})+\lambda_{l2}\mathcal{L}_2$ |
| `det = a * c - b * b` | $\det_i = a_i c_i - b_i^2$ |
| `sigma_area = torch.sqrt(det + 1e-6)` | $\sigma^{area}_i=\sqrt{\det_i+10^{-6}}$ |
| `sigma_linear = torch.sqrt(sigma_area + 1e-6)` | $\sigma^{linear}_i=\sqrt{\sigma^{area}_i+10^{-6}}$ |
| `freq_target = torch.sqrt(Sxx + Syy + 1e-6)` | $f_{target,i}=\sqrt{S_{xx,i}+S_{yy,i}+10^{-6}}$ |
| `wavelength_limit = 1.0 / (freq_target + 1e-6)` | $\lambda_{limit,i}=1/(f_{target,i}+10^{-6})$ |
| `torch.relu(sigma_linear - wavelength_limit).mean()` | $\mathcal{L}_{freq}=\frac{1}{N}\sum_i \mathrm{ReLU}(\sigma^{linear}_i-\lambda_{limit,i})$ |
| `Ixx_c = Ix**2`, sum theo channel | $I_{xx}=\sum_{channel} I_x^2$ |
| `Sxx = fast_gaussian_blur(Ixx, ...)` | $S_{xx} = G_\rho * I_{xx}$ |
| `Sxx = Sxx / max_val` | chuẩn hoá theo $M=\max(S_{xx}+S_{yy})+10^{-6}$ |
| `sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))` | $\sigma_{inc}=\sqrt{\max(10^{-6},\sigma_i^2-\sigma_{i-1}^2)}$ |
| `band_response = (current_smooth - next_smooth).pow(2).sum(dim=1,...).sqrt()` | $R_i=\sqrt{\sum_{channel}(I_{i-1}-I_i)^2}$ |
| `weight_i = band_response.pow(power_factor)` | $w_i = R_i^{\,p}$ |
| `Sxx_norm_i = Sxx_i / trace_i` | $\hat{S}_{xx,i}=S_{xx,i}/(S_{xx,i}+S_{yy,i}+10^{-6})$ |
| `accum_Sxx += Sxx_norm_i * weight_i * target_wavelength_sq_inv` | $\mathrm{Acc}_{xx}\mathrel{+}=\hat{S}_{xx,i}w_i f_i^2$ |
| `final_Sxx = accum_Sxx / (accum_weight + 1e-6)` | $S_{xx}^{final}=\mathrm{Acc}_{xx}/(\mathrm{Acc}_w+10^{-6})$ |
| `local_energy = Sxx + Syy` | $E_{local}=S_{xx}+S_{yy}$ |
| `total_structure_energy = local_energy.sum().item()` | $E_{total}=\sum_{h,w}E_{local}$ |
| `count_coverage = total_pixels * base_density` | $N_{coverage}=HW\rho_{base}$ |
| `count_detail = total_structure_energy * detail_sensitivity` | $N_{detail}=E_{total}k$ |
| `estimated_count = int(count_coverage + count_detail)` | $N_{est}=\lfloor N_{coverage}+N_{detail}\rfloor$ |
