# Công thức toán học trong `lpipsPyTorch/modules/networks.py`

Tài liệu mô tả cơ sở toán học trong các mạng backbone trích đặc trưng dùng cho LPIPS, bám sát thứ tự xuất hiện trong code.

---

## 1. Hàm chọn mạng (`get_network`)

Trích nguyên văn (dòng 12–20):

```python
def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')
```

Thuần boilerplate — rẽ nhánh theo chuỗi `net_type` để khởi tạo một trong ba lớp `AlexNet`, `SqueezeNet`, `VGG16`. Không có nội dung toán học.

---

## 2. Lớp tuyến tính theo kênh (`LinLayers`)

Trích nguyên văn (dòng 23–33):

```python
class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False
```

Với mỗi layer đích $l$ có số kênh $C_l$ (lấy từ `n_channels_list`), định nghĩa một ánh xạ tuyến tính không bias bằng tích chập $1\times1$ (dòng 28, `nn.Conv2d(nc, 1, 1, 1, 0, bias=False)`):

$$
\mathrm{conv}_l(z)_{hw} = \sum_{c=1}^{C_l} w_{l,c}\, z_{c,hw}, \qquad w_l \in \mathbb{R}^{1\times C_l}
$$

Trọng số $w_l$ được đóng băng (dòng 32–33, `param.requires_grad = False`) vì đã được nạp sẵn từ checkpoint LPIPS đã huấn luyện (xem `lpips.py`, `utils.py`). Công thức sử dụng của lớp này được trình bày chi tiết trong `lpips.md`.

---

## 3. Mạng cơ sở (`BaseNet`)

### 3.1. Hằng số chuẩn hoá đầu vào (`mean`, `std`)

Trích nguyên văn (dòng 36–44):

```python
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])
```

$$
m = (-0.030,\ -0.088,\ -0.188), \qquad \sigma = (0.458,\ 0.448,\ 0.450)
$$

Đây là các hằng số dịch/chuẩn hoá kênh RGB đặc thù của LPIPS (không phải mean/std ImageNet chuẩn), dùng để đưa ảnh đầu vào (đã ở khoảng $[-1,1]$, xem `utils.md`/bước tiền xử lý phía ngoài) về đúng miền giá trị mà mạng backbone (vốn huấn luyện trên ImageNet) kỳ vọng.

### 3.2. Chuẩn hoá z-score theo kênh (`z_score`)

Trích nguyên văn (dòng 50–51):

```python
    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std
```

$$
\hat{x}_{c,hw} = \frac{x_{c,hw} - m_c}{\sigma_c}, \qquad c \in \{R,G,B\}
$$

($m_c, \sigma_c$ broadcast theo kênh nhờ shape `[None, :, None, None]`.)

### 3.3. Lan truyền xuôi và trích đặc trưng tại các layer đích (`forward`)

Trích nguyên văn (dòng 53–63):

```python
    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output
```

$$
x_0 = \hat{x} = \mathrm{z\_score}(x) \qquad (\text{dòng 54})
$$

$$
x_i = \mathrm{layer}_i(x_{i-1}), \qquad i = 1,\dots \qquad (\text{dòng 57–58})
$$

Tại mỗi chỉ số layer $i \in \texttt{target\_layers}$, đặc trưng được chuẩn hoá đơn vị theo kênh (dòng 59–60, gọi `normalize_activation(x)` — công thức $\mathrm{normalize\_activation}$, xem `utils.md`):

$$
\hat{y}^{(l)} = \frac{x_i}{\lVert x_i\rVert_2 + \epsilon} \quad (\text{tính theo chiều kênh})
$$

và được thu thập vào danh sách output; vòng lặp dừng sớm khi đã đủ số layer đích (dòng 61–62, `len(output) == len(target_layers)`).

### 3.4. Khoá gradient (`set_requires_grad`)

Trích nguyên văn (dòng 46–48):

```python
    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state
```

Thuần boilerplate: đặt `requires_grad = state` cho mọi tham số và buffer — không có công thức toán học, chỉ phục vụ việc đóng băng mạng backbone trong quá trình huấn luyện.

---

## 4. Các mạng cụ thể (`SqueezeNet`, `AlexNet`, `VGG16`)

Trích nguyên văn ba lớp con (dòng 66–96):

```python
class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
```

Mỗi lớp chỉ khai báo:

- `self.layers`: backbone pretrained lấy từ `torchvision.models` (boilerplate, kiến trúc mạng chuẩn, không định nghĩa thêm công thức mới).
- `self.target_layers`: danh sách chỉ số layer $i$ dùng để trích đặc trưng (tham số cấu trúc, không phải công thức toán).
- `self.n_channels_list`: số kênh $C_l$ tương ứng tại mỗi layer đích, dùng để khởi tạo `LinLayers` ở mục 2.

Không có phép biến đổi toán học riêng biệt nào khác ngoài những gì đã nêu ở `BaseNet` (mục 3) — các lớp con chỉ khác nhau về kiến trúc backbone và vị trí trích đặc trưng.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `get_network(net_type)` | rẽ nhánh chọn mạng (boilerplate, không có công thức) |
| `nn.Conv2d(nc, 1, 1, 1, 0, bias=False)` trong `LinLayers` | $\mathrm{conv}_l(z)_{hw} = \sum_{c=1}^{C_l} w_{l,c} z_{c,hw}$ |
| `self.register_buffer('mean', ...)` | $m = (-0.030,-0.088,-0.188)$ |
| `self.register_buffer('std', ...)` | $\sigma = (0.458,0.448,0.450)$ |
| `z_score(x) = (x - self.mean) / self.std` | $\hat{x}_{c,hw} = \dfrac{x_{c,hw}-m_c}{\sigma_c}$ |
| `x = self.z_score(x)` trong `forward` | $x_0 = \mathrm{z\_score}(x)$ |
| `x = layer(x)` (lặp qua `self.layers`) | $x_i = \mathrm{layer}_i(x_{i-1})$ |
| `output.append(normalize_activation(x))` khi `i in target_layers` | $\hat{y}^{(l)} = x_i/(\lVert x_i\rVert_2+\epsilon)$ |
| `set_requires_grad(state)` | boilerplate, không có công thức |
| `self.target_layers`, `self.n_channels_list` | tham số cấu trúc (không phải công thức) |
