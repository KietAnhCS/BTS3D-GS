# `submodules/` — Giải thích chi tiết (wrapper Python cho CUDA)

Thư mục này chứa 3 submodule CUDA (biên dịch native), mỗi submodule chỉ có phần
**Python wrapper** (autograd Function, setup.py build) là đọc được trực tiếp; phần
công thức toán lõi (CUDA kernel) không nằm trong file `.py`. Tài liệu này giải thích
công thức toán ở mức wrapper lộ ra (những gì truyền vào/nhận lại từ CUDA) và dẫn chiếu
công thức toán học đằng sau mỗi thuật toán.

---

# A. `diff-gaussian-rasterization_structgs/` — rasterizer chính

## `diff_gaussian_rasterization_structgs/__init__.py` — 7 hàm/class

### 1. `cpu_deep_copy_tuple(input_tuple)` (dòng 17–19)

Copy tensor sang CPU để lưu snapshot debug khi CUDA kernel crash. Không công thức.

### 2. `rasterize_gaussians(...)` (dòng 21–44)

Hàm tiện ích gọi `_RasterizeGaussians.apply(...)` — kích hoạt `torch.autograd.Function`
tuỳ chỉnh để forward/backward qua CUDA kernel. Không công thức Python.

### 3. `_RasterizeGaussians.forward(ctx, ...)` (dòng 48–113)

Gọi `_C.rasterize_gaussians(*args)` — đây là nơi **splatting thật sự** diễn ra trong
CUDA, thực hiện (công thức tham khảo từ bài báo 3D Gaussian Splatting, Kerbl et al. 2023):

- **Chiếu Gaussian 3D → 2D** bằng xấp xỉ tuyến tính hoá (Jacobian $J$ của phép chiếu
  phối cảnh) và ma trận thế giới→camera $W$:

$$
\Sigma' = J\,W\,\Sigma\,W^{T}J^{T}
$$

  (chỉ lấy khối 2×2 góc trên trái làm hiệp phương sai 2D trên màn hình).

- **Alpha compositing** theo thứ tự độ sâu (front-to-back) cho mỗi pixel, với $N$
  Gaussian phủ lên pixel đó, màu $c_i$ và độ mờ $\alpha_i$ (từ opacity Gaussian nhân
  với giá trị hàm mật độ Gaussian 2D tại pixel):

$$
C = \sum_{i=1}^{N} c_i\,\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

- **Độ mờ từng Gaussian tại 1 pixel** $\mathbf{x}$ (hàm mật độ Gaussian 2D chưa chuẩn hoá,
  nhân với opacity $o_i$):

$$
\alpha_i(\mathbf x) = o_i \cdot \exp\!\left(-\tfrac12 (\mathbf x-\boldsymbol\mu_i)^T \Sigma_i'^{-1} (\mathbf x-\boldsymbol\mu_i)\right)
$$

- **Màu từ SH** (nếu không truyền `colors_precomp`): $c_i = \text{eval\_sh}(\text{deg}, sh_i, \mathbf d_i)$
  (công thức SH ở `utils/sh_utils.py`, xem `DOCS/BOOK/utils_full.md`).

Hàm cũng trả thêm `accum_metric_counts`, `cov2D` (Σ' đã tính, dùng lại ở `freq_utils.py`
để tính η), `depth_map`, `opacity_map`, `normal_map` — các kênh phụ đặc thù StructGS.

### 4. `_RasterizeGaussians.backward(ctx, ...)` (dòng 115–175)

Gọi `_C.rasterize_gaussians_backward(*args)` — lan truyền ngược gradient loss qua
công thức alpha-compositing và chiếu Gaussian ở trên, trả về gradient cho
`means3D, means2D, colors, opacities, scales, rotations, cov3D, dc, sh`. Toàn bộ vi phân
($\partial C/\partial \alpha_i$, $\partial C/\partial c_i$, $\partial \Sigma'/\partial \Sigma$...)
được cài đặt trong CUDA, không hiện diện ở Python.

### 5. `GaussianRasterizationSettings` (NamedTuple, dòng 177–193)

Chỉ là cấu trúc dữ liệu chứa tham số camera/pipeline. Không công thức.

### 6. `GaussianRasterizer` — `markVisible` + `forward` (dòng 195–247)

- `markVisible`: gọi `_C.mark_visible` — frustum culling (kiểm tra điểm có nằm trong
  khung nhìn camera hay không), thuần hình học nhưng cài trong CUDA.
- `forward`: kiểm tra input hợp lệ rồi gọi `rasterize_gaussians` (#2).

### 7. `SparseGaussianAdam.step(self, visibility, N)` (dòng 249–277)

Biến thể Adam **chỉ cập nhật các Gaussian hiển thị** (visibility mask) thay vì toàn bộ
tensor — tiết kiệm tính toán. Công thức Adam chuẩn vẫn áp dụng nhưng chỉ trên tập con:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \qquad (\text{chỉ với } i \in \text{visible})
$$
$$
\theta_t[i] = \theta_{t-1}[i] - \text{lr}\cdot\dfrac{m_t[i]}{\sqrt{v_t[i]}+\epsilon}
$$

($\beta_1=0.9,\ \beta_2=0.999$ cố định trong lệnh gọi `_C.adamUpdate`). Phép cập nhật
thực thi trong CUDA (`_C.adamUpdate`), Python chỉ chuẩn bị state và gọi kernel.

### `setup.py` (35 dòng) — build script `pip install`, không công thức, không phải hàm runtime.

---

# B. `fused-ssim/` — SSIM loss hợp nhất kernel (nhanh hơn bản PyTorch thuần)

## `fused_ssim/__init__.py` — 4 hàm/class

### 1. `FusedSSIMMap.forward(ctx, C1, C2, img1, img2, padding, train)` (dòng 9–21)

Gọi kernel CUDA `fusedssim(...)` tính **SSIM map** theo công thức chuẩn (Wang et al. 2004),
dùng cửa sổ Gaussian 11×11, $\sigma=1.5$ (xem công thức đầy đủ ở mục `ssim`/`_ssim` trong
`utils/loss_utils.py`, `DOCS/BOOK/utils_full.md`):

$$
\text{SSIM}(x,y)=\dfrac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
$$

với $C_1=0.01^2,\ C_2=0.03^2$. Kernel CUDA trả thêm đạo hàm riêng phần
$\partial m/\partial \mu_1$, $\partial m/\partial \sigma_1^2$, $\partial m/\partial \sigma_{12}$
để dùng trong backward (tránh phải lưu toàn bộ đồ thị tính toán conv2D).

Nếu `padding="valid"`, cắt viền 5 pixel mỗi phía (ứng với bán kính cửa sổ $\lfloor 11/2\rfloor$).

### 2. `FusedSSIMMap.backward(ctx, opt_grad)` (dòng 23–32)

Dùng chain rule thủ công với các đạo hàm riêng đã lưu, gọi kernel
`fusedssim_backward(...)`:

$$
\dfrac{\partial L}{\partial x} =
\dfrac{\partial L}{\partial m}\cdot\dfrac{\partial m}{\partial \mu_1}
+\dfrac{\partial L}{\partial m}\cdot\dfrac{\partial m}{\partial \sigma_1^2}
+\dfrac{\partial L}{\partial m}\cdot\dfrac{\partial m}{\partial \sigma_{12}}
$$

(áp dụng qua kernel CUDA, công thức chain rule đầy đủ nằm trong CUDA code).

### 3. `fused_ssim(img1, img2, padding="same", train=True)` (dòng 34–41)

$$
\text{loss} = \dfrac1{HW}\sum_{x,y}\text{SSIM}_{map}(x,y)
$$

### 4. `fused_ssim_(img1, img2, padding="same", train=True)` (dòng 43–50)

Biến thể chỉ lấy trung bình theo kênh đầu (`squeeze(0).mean(0)`) thay vì trung bình
toàn bộ batch — dùng khi cần SSIM theo từng mẫu riêng.

## `setup.py` (17 dòng) — build script, không công thức.

## `tests/test.py` — bản SSIM tham chiếu thuần PyTorch để đối chiếu số học + benchmark tốc độ

- `gaussian(window_size, sigma)`: tạo kernel Gaussian 1D chuẩn hoá tổng = 1:

$$
g_i = \dfrac{\exp\!\left(-\dfrac{(i-\lfloor w/2\rfloor)^2}{2\sigma^2}\right)}{\sum_j \exp\!\left(-\dfrac{(j-\lfloor w/2\rfloor)^2}{2\sigma^2}\right)}
$$

- `create_window(window_size, channel)`: cửa sổ 2D = tích ngoài của cửa sổ 1D:
  $W = g\,g^{T}$, lặp lại cho mỗi channel (depthwise conv).
- `ssim`/`_ssim`: triển khai công thức SSIM đầy đủ bằng `conv2d` làm bộ lọc trung bình
  trượt cục bộ — giống hệt công thức ở mục B.1 và `utils/loss_utils.py`.
- Phần `__main__`: so khớp số học + đo thời gian forward/backward giữa 3 cách cài đặt
  (tham chiếu PyTorch, `pytorch_msssim`, `fused_ssim`). Không công thức mới.

## `tests/genplot.py` (100 dòng) và `tests/train_image.py` (29 dòng)

Script demo/benchmark: `train_image.py` tối ưu trực tiếp một ảnh ngẫu nhiên để khớp
ảnh đích bằng cách **tối thiểu hoá $1-\text{SSIM}$** qua Adam:

$$
\min_{I_{pred}}\ \big(1 - \text{SSIM}(I_{pred}, I_{gt})\big)
$$

`genplot.py` dùng để vẽ biểu đồ so sánh hiệu năng (không có công thức toán học mới,
chỉ matplotlib). Cả hai không phải phần của pipeline train chính thức.

---

# C. `simple-knn/` — chỉ có `setup.py` (35 dòng), không có mã Python runtime

Submodule này cung cấp hàm `distCUDA2` (dùng trong `gaussian_model.py` mục
`create_from_pcd`) nhưng toàn bộ cài đặt nằm trong C++/CUDA (`simple_knn.cu`), không có
file `.py` nào định nghĩa hàm — `setup.py` chỉ là script build/biên dịch extension.

Công thức mà `distCUDA2` tính (theo tên và cách dùng trong `gaussian_model.py`): với mỗi
điểm, tìm khoảng cách Euclid bình phương trung bình tới $k$ láng giềng gần nhất
(thường $k=3$ trong 3DGS gốc):

$$
d_i^2 = \dfrac1k\sum_{j\in \text{kNN}(i)} \lVert \mathbf x_i - \mathbf x_j\rVert_2^2
$$

dùng làm scale khởi tạo ban đầu cho mỗi Gaussian (xem `DOCS/BOOK/gaussian_model.md`
mục 18).

---

## Tổng kết số lượng hàm/class (chỉ tính phần `.py` đọc được)

| File | Số hàm/class |
|---|---|
| `diff-gaussian-rasterization_structgs/__init__.py` | 7 (2 hàm top-level, `_RasterizeGaussians` với forward/backward, `GaussianRasterizationSettings`, `GaussianRasterizer` với 2 method, `SparseGaussianAdam` với 1 method) |
| `fused-ssim/__init__.py` | 4 (`FusedSSIMMap` forward/backward, `fused_ssim`, `fused_ssim_`) |
| `fused-ssim/tests/*.py` | 3 file demo/benchmark, không phải API chính thức |
| `simple-knn/setup.py` | 0 hàm Python (chỉ build script; logic nằm trong CUDA) |
