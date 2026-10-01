# Công thức toán học trong `utils/image_utils.py`

File này chứa hai hàm đo lường chất lượng ảnh cơ bản: MSE và PSNR, dùng để đánh giá sai khác giữa ảnh render và ảnh ground-truth.

---

## 1. Sai số bình phương trung bình — MSE (`mse`)

Trích nguyên văn (`utils/image_utils.py`, dòng 14–15):

```python
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
```

Với hai ảnh (batch) $I_1, I_2 \in \mathbb{R}^{B\times C\times H\times W}$, hàm làm phẳng mọi chiều trừ batch rồi lấy trung bình:

$$
\mathrm{MSE}(I_1,I_2)_b = \frac{1}{C\cdot H\cdot W}\sum_{c,h,w}\big(I_{1}[b,c,h,w]-I_{2}[b,c,h,w]\big)^2
$$

Kết quả giữ chiều batch (`keepdim=True`): $\mathrm{MSE}\in\mathbb{R}^{B\times 1}$.

---

## 2. Tỉ số tín hiệu trên nhiễu đỉnh — PSNR (`psnr`)

Trích nguyên văn (`utils/image_utils.py`, dòng 17–19):

```python
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

PSNR được tính lại MSE nội bộ (giống hệt định nghĩa ở mục 1, không tái sử dụng hàm `mse`), rồi áp dụng công thức PSNR chuẩn với giá trị đỉnh tín hiệu $MAX_I = 1.0$ (ảnh chuẩn hoá $[0,1]$):

$$
\mathrm{PSNR}(I_1,I_2)_b = 20\log_{10}\!\left(\frac{MAX_I}{\sqrt{\mathrm{MSE}_b}}\right), \qquad MAX_I = 1.0
$$

Tương đương dạng thường gặp trong tài liệu ($MAX_I=1$):

$$
\mathrm{PSNR} = 20\log_{10}(1) - 10\log_{10}(\mathrm{MSE}) = -10\log_{10}(\mathrm{MSE})
$$

(hai cách viết toán học tương đương; code dùng trực tiếp dạng $20\log_{10}(1/\sqrt{\mathrm{MSE}})$).

**Ghi chú tham chiếu**: file `metrics.py` (ở thư mục gốc repo) **không** định nghĩa lại PSNR — nó `import` trực tiếp hàm `psnr` từ `utils/image_utils.py` (`from utils.image_utils import psnr`) và dùng để tính PSNR trung bình trên tập ảnh test/train. Do đó không có công thức PSNR thứ hai nào khác trong codebase; `metrics.py` chỉ là nơi gọi và tổng hợp (lấy trung bình, lưu JSON).

Không có hàm `mse2psnr` riêng biệt trong file này.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `(((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)` (trong `mse`) | $\mathrm{MSE}_b = \dfrac{1}{CHW}\sum_{c,h,w}(I_{1}-I_{2})^2$ |
| `mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)` (trong `psnr`) | $\mathrm{MSE}_b$ (tính lại nội bộ, cùng công thức trên) |
| `20 * torch.log10(1.0 / torch.sqrt(mse))` | $\mathrm{PSNR}_b = 20\log_{10}\!\left(\dfrac{1}{\sqrt{\mathrm{MSE}_b}}\right)$ |
