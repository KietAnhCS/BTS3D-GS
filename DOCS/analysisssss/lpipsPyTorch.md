# `lpipsPyTorch/` — Giải thích chi tiết (LPIPS perceptual loss)

LPIPS (Learned Perceptual Image Patch Similarity) đo độ khác biệt tri giác giữa 2 ảnh
bằng cách so khớp đặc trưng ở nhiều lớp của một mạng CNN đã huấn luyện trước (AlexNet/
SqueezeNet/VGG16), có trọng số tuyến tính học riêng cho LPIPS.

Tổng cộng **8 hàm/method** trên 4 file.

---

## `lpipsPyTorch/__init__.py` — 1 hàm

### `lpips(x, y, net_type='alex', version='0.1')` (dòng 6–21)

Hàm cổng vào: tạo `LPIPS(net_type, version)` rồi gọi `criterion(x, y)`. Không công thức
riêng — xem công thức tổng ở `LPIPS.forward` bên dưới.

---

## `lpipsPyTorch/modules/lpips.py` — 2 method

### 1. `LPIPS.__init__(self, net_type='alex', version='0.1')` (dòng 17–28)

Khởi tạo mạng backbone (`get_network`) và lớp tuyến tính 1×1 trên mỗi tầng đặc trưng
(`LinLayers`), nạp trọng số đã huấn luyện sẵn (tải từ GitHub của tác giả LPIPS gốc).
Không công thức.

### 2. `LPIPS.forward(self, x, y)` (dòng 30–36)

Công thức LPIPS đầy đủ. Với $L$ tầng đặc trưng đã chuẩn hoá $\hat f_x^{(l)}, \hat f_y^{(l)}$
(chuẩn hoá theo kênh — xem `normalize_activation` bên dưới), và trọng số học $w^{(l)}$
(tích chập $1\times1$ không bias trong `LinLayers`):

$$
d(x,y) = \sum_{l=1}^{L} \dfrac{1}{H_lW_l}\sum_{h,w} w^{(l)} \odot \big(\hat f_x^{(l)}_{h,w} - \hat f_y^{(l)}_{h,w}\big)^2
$$

Trong code: `diff = (fx - fy)**2`, sau đó mỗi tầng được đưa qua lớp tuyến tính
`l(d)` (tương đương nhân trọng số kênh $w^{(l)}$ rồi cộng theo kênh) và lấy trung bình
không gian `.mean((2,3))`, cuối cùng cộng tổng qua các tầng (`torch.sum(..., 0)`).

---

## `lpipsPyTorch/modules/networks.py` — 5 hàm/class

### 1. `get_network(net_type)` (dòng 12–20)

Factory chọn backbone (`AlexNet`/`SqueezeNet`/`VGG16`). Không công thức.

### 2. `LinLayers.__init__(self, n_channels_list)` (dòng 24–33)

Tạo 1 `Conv2d(nc, 1, kernel=1, bias=False)` cho mỗi tầng đặc trưng — chính là trọng số
$w^{(l)}$ trong công thức LPIPS ở trên, đóng băng gradient (`requires_grad=False`).

### 3. `BaseNet.z_score(self, x)` (dòng 50–51)

Chuẩn hoá ảnh đầu vào theo thống kê kênh cố định (ước lượng từ ImageNet, lệch tâm quanh 0
thay vì 0.5 vì LPIPS huấn luyện trên ảnh đã chuẩn hoá khác chuẩn ImageNet gốc):

$$
\hat x = \dfrac{x-\mu}{\sigma},\qquad
\mu=(-0.030,-0.088,-0.188),\quad \sigma=(0.458,0.448,0.450)
$$

### 4. `BaseNet.forward(self, x)` (dòng 53–63)

Chạy tuần tự qua các layer CNN của backbone, trích đặc trưng tại các `target_layers`
được chọn sẵn, mỗi đặc trưng trích ra được **chuẩn hoá theo chuẩn L2 kênh** ngay
(gọi `normalize_activation`, công thức ở file `utils.py` bên dưới).

### 5. `AlexNet.__init__` / `SqueezeNet.__init__` / `VGG16.__init__` (dòng 66–96)

Tải backbone CNN có pretrained weights (ImageNet) từ `torchvision.models`, cố định
danh sách tầng trích đặc trưng (`target_layers`) và số kênh tương ứng
(`n_channels_list`). Không công thức riêng (chỉ cấu hình).

---

## `lpipsPyTorch/modules/utils.py` — 2 hàm

### 1. `normalize_activation(x, eps=1e-10)` (dòng 6–8)

Chuẩn hoá đơn vị theo chuẩn L2 dọc theo trục kênh (unit-normalize feature vector tại
mỗi vị trí không gian) — bước quan trọng để LPIPS không bị chi phối bởi độ lớn kích
hoạt tuyệt đối mà chỉ so khớp *hướng* đặc trưng:

$$
\hat f_{c,h,w} = \dfrac{f_{c,h,w}}{\sqrt{\sum_{c'} f_{c',h,w}^2} + \epsilon}
$$

### 2. `get_state_dict(net_type='alex', version='0.1')` (dòng 11–30)

Tải trọng số tuyến tính LPIPS pretrained từ URL GitHub chính thức, đổi tên key cho
khớp với cấu trúc `LinLayers` ở trên. Thuần I/O, không công thức.

---

## Tổng kết số lượng hàm

| File | Số hàm/method |
|---|---|
| `__init__.py` | 1 |
| `modules/lpips.py` | 2 |
| `modules/networks.py` | 5 (`get_network`, `LinLayers.__init__`, `BaseNet.z_score`, `BaseNet.forward`, 3 subclass `__init__` tính gộp 1 mục) |
| `modules/utils.py` | 2 |

**Ghi chú**: LPIPS chỉ được dùng làm **chỉ số đánh giá** (metric, ví dụ trong script
đo PSNR/SSIM/LPIPS sau khi train), không phải một phần gradient trong vòng lặp tối ưu
chính của 3DGS/StructGS (loss chính nằm ở `utils/loss_utils.py`).
