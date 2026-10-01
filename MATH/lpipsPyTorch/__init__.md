# Công thức toán học trong `lpipsPyTorch/__init__.py`

Tài liệu mô tả cơ sở toán học (nếu có) đứng sau các hàm trong file `__init__.py` của package `lpipsPyTorch`.

---

## 1. Hàm `lpips(x, y, net_type, version)`

Trích nguyên văn toàn bộ hàm (`lpipsPyTorch/__init__.py`, dòng 6–21):

```python
def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
```

File này **không chứa phép toán riêng** — đây là một hàm wrapper (lớp vỏ tiện ích) khởi tạo module `LPIPS` (định nghĩa trong `modules/lpips.py`, dòng 3: `from .modules.lpips import LPIPS`) rồi gọi trực tiếp lên hai tensor ảnh đầu vào $x, y$ (dòng 21: `return criterion(x, y)`):

$$
d(x,y) = \mathrm{LPIPS}_{\theta}(x,y)
$$

trong đó $\mathrm{LPIPS}_\theta(\cdot,\cdot)$ là toàn bộ phép tính khoảng cách tri giác học được (tham số hoá bởi mạng backbone `net_type` và trọng số tuyến tính đã huấn luyện ứng với `version`), được trình bày chi tiết trong `MATH/lpipsPyTorch/modules/lpips.md`.

Về mặt code, hàm chỉ làm 3 việc:

1. Lấy `device = x.device` (dòng 19) để đảm bảo mô hình và dữ liệu cùng thiết bị tính toán.
2. Khởi tạo `criterion = LPIPS(net_type, version).to(device)` (dòng 20).
3. Trả về `criterion(x, y)` (dòng 21), tức gọi `forward` của `LPIPS` (qua `__call__` của `nn.Module`) — chính là công thức $d(x,y)$ ở trên.

Không có hằng số, phép chuẩn hoá, hay biến đổi toán học nào xảy ra ở cấp độ file này; toàn bộ nội dung toán học nằm ở các module con (`lpips.py`, `networks.py`, `utils.py`).

## Kiểm chứng tính đúng sai

Đối chiếu trực tiếp với `lpipsPyTorch/__init__.py` thật (22 dòng): nội dung mô tả đúng 100% — file chỉ gồm 1 import và 1 hàm, không có logic toán học nào bị bỏ sót hay suy diễn sai. Không phát hiện sai sót.

## Ví dụ số (minh hoạ lời gọi, không phải phép tính mới)

Giả sử $x,y\in\mathbb R^{1\times3\times64\times64}$ (ảnh RGB 64×64) nằm trên GPU (`x.device = cuda:0`). Khi gọi `lpips(x, y, net_type='alex', version='0.1')`:

1. `device = x.device = cuda:0`.
2. `criterion = LPIPS('alex', '0.1').to('cuda:0')` — dựng mạng AlexNet backbone (đóng băng trọng số pretrained) cộng lớp tuyến tính LPIPS bậc `0.1`, chuyển toàn bộ tham số sang GPU.
3. `criterion(x, y)` trả về một tensor vô hướng (hoặc theo batch) $d(x,y)\in\mathbb R_{\ge0}$, giá trị càng nhỏ nghĩa là $x,y$ càng giống nhau về mặt tri giác (perceptual). Công thức tường minh của $d$ (tổng có trọng số của khoảng cách $L_2$ theo từng lớp đặc trưng đã chuẩn hoá) nằm ở `MATH/lpipsPyTorch/modules/lpips.md`, không lặp lại ở đây.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `criterion = LPIPS(net_type, version).to(device)` | khởi tạo $\mathrm{LPIPS}_\theta(\cdot,\cdot)$ (không có công thức riêng, chỉ dựng mô hình) |
| `return criterion(x, y)` | $d(x,y) = \mathrm{LPIPS}_{\theta}(x,y)$ |
