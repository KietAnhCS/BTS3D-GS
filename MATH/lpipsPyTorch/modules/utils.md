# Công thức toán học trong `lpipsPyTorch/modules/utils.py`

Tài liệu mô tả cơ sở toán học trong các hàm tiện ích dùng cho LPIPS, bám sát thứ tự xuất hiện trong code.

---

## 1. Chuẩn hoá đơn vị theo kênh (`normalize_activation`, `lpipsPyTorch/modules/utils.py` dòng 6–8)

Trích nguyên văn (dòng 6–8):

```python
def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)
```

Với $x \in \mathbb{R}^{B\times C\times H\times W}$, chuẩn $\ell_2$ được tính theo **chiều kênh** (`dim=1`) tại mỗi vị trí không gian $(h,w)$:

$$
\lVert x_{hw}\rVert_2 = \sqrt{\sum_{c=1}^{C} x_{c,hw}^2}
$$

Đặc trưng được chuẩn hoá về vector đơn vị theo kênh:

$$
\hat{x}_{hw} = \frac{x_{hw}}{\lVert x_{hw}\rVert_2 + \epsilon}, \qquad \epsilon = 10^{-10}
$$

Đây chính xác là công thức chuẩn hoá đặc trưng theo kênh (unit-normalize) $\hat{y}_{hw} = y_{hw}/(\lVert y_{hw}\rVert_2+\epsilon)$ được dùng trong LPIPS, áp dụng tại mỗi layer đích của mạng backbone (xem `networks.md`).

> **Lưu ý:** file này không chứa bước chuẩn hoá ảnh đầu vào về $[-1,1]$ hay trừ mean/std ImageNet — phép z-score theo kênh (với hằng số LPIPS riêng, không phải ImageNet) nằm ở `BaseNet.z_score` trong `networks.py` (xem `networks.md`, mục 3.1–3.2). `utils.py` chỉ chịu trách nhiệm chuẩn hoá **đặc trưng** (feature activation), không chuẩn hoá **ảnh**.

---

## 2. Tải và đổi tên trọng số đã huấn luyện (`get_state_dict`, `lpipsPyTorch/modules/utils.py` dòng 11–30)

Trích nguyên văn (dòng 11–30):

```python
def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
```

Hàm này thuần là **boilerplate I/O**: xây dựng URL, tải checkpoint `.pth` đã huấn luyện sẵn (chứa đúng các trọng số $w_{l,c}$ dùng trong `LinLayers`, xem `networks.md`/`lpips.md`), rồi đổi tên khoá (`key`) trong `state_dict` cho khớp với tên module trong code hiện tại (bỏ tiền tố `lin`, `model.`). Không có phép biến đổi toán học nào xảy ra ở đây — chỉ là ánh xạ tên chuỗi (string mapping) giữa hai cách đặt tên tham số.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))` | $\lVert x_{hw}\rVert_2 = \sqrt{\sum_{c=1}^{C} x_{c,hw}^2}$ |
| `return x / (norm_factor + eps)` | $\hat{x}_{hw} = \dfrac{x_{hw}}{\lVert x_{hw}\rVert_2+\epsilon}$ |
| `eps=1e-10` | $\epsilon = 10^{-10}$ |
| `get_state_dict(net_type, version)` | boilerplate tải trọng số $\{w_{l,c}\}$ (không có công thức) |
| `new_key.replace('lin', '').replace('model.', '')` | ánh xạ tên tham số (string mapping, không phải công thức toán) |
