# Các file binding/khai báo không chứa công thức toán học riêng

Các file dưới đây đã được đọc trực tiếp. Chúng chủ yếu là khai báo struct, chữ ký hàm, pybind11 binding hoặc build script — **không trích công thức toán học bịa đặt** cho các file này; công thức thật sự nằm ở `forward.cu`, `backward.cu`, `rasterizer_impl.cu`, `adam.cu` (xem `MATH/cuda/adam.md`, `MATH/cuda/rasterizer_impl.md`).

> Đồng bộ: nội dung dưới đây đã được đối chiếu lại với bản kiểm chứng kỹ hơn tại `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md` (mục "2.4. Tóm tắt các hàm binding" và "ext.cpp"), đọc lại source thật. Khi có mâu thuẫn, bản `submodules/...` được ưu tiên vì đọc gần đây hơn.

Nguồn: `submodules/diff-gaussian-rasterization_structgs/rasterize_points.cu`, `rasterize_points.h`, `ext.cpp`, `cuda_rasterizer/config.h`, `cuda_rasterizer/rasterizer.h`, và 3 file `setup.py` của các submodule CUDA.

## `cuda_rasterizer/config.h`

Chỉ định nghĩa hằng số biên dịch: `NUM_CHAFFELS = 3` (số kênh màu RGB), `BLOCK_X = 16`, `BLOCK_Y = 16` (kích thước tile). Không có công thức.

## `cuda_rasterizer/rasterizer.h`

Khai báo interface `class CudaRasterizer::Rasterizer` với 3 hàm tĩnh: `markVisible`, `forward`, `backward` — chỉ là chữ ký hàm (danh sách tham số con trỏ/kiểu dữ liệu), cài đặt thực tế ở `rasterizer_impl.cu`. Không có công thức trong file này.

## `rasterize_points.h` / `rasterize_points.cu`

File này là lớp **glue** C++/CUDA ↔ PyTorch, không chứa công thức toán mới — chỉ kiểm tra shape, cấp phát/resize tensor đệm, và chuyển con trỏ dữ liệu thô vào các hàm CUDA đã có công thức đầy đủ ở nơi khác (`forward.cu`, `backward.cu`, `rasterizer_impl.cu`, `adam.cu`).

| Hàm C++ (`rasterize_points.h`) | Vai trò | Tham số chính | Gọi vào |
|---|---|---|---|
| `RasterizeGaussiansCUDA` | Forward rasterization: render ảnh RGB + (tuỳ chọn) depth/opacity/normal map từ $N$ Gaussian 3D | `means3D, colors, opacity, scales, rotations, scale_modifier, cov3D_precomp, metric_map, viewmatrix, projmatrix, tan_fovx/fovy, H, W, dc, sh, degree, campos, mult, prefiltered, debug, get_flag, compute_extra` | `CudaRasterizer::Rasterizer::forward` |
| `RasterizeGaussiansBackwardCUDA` | Backward: lan truyền $\partial\mathcal L/\partial(\text{out\_color})$ ngược về gradient của mọi tham số Gaussian | `dL_dout_color, geomBuffer, binningBuffer, imageBuffer, sampleBuffer, R, B, ...` (các tensor forward lưu lại) | `CudaRasterizer::Rasterizer::backward` |
| `markVisible` | Frustum-culling: đánh dấu Gaussian nào nằm trong view frustum của camera | `means3D, viewmatrix, projmatrix` → trả `present` (bool mỗi Gaussian) | `CudaRasterizer::Rasterizer::markVisible` |
| `adamUpdate` | Wrapper gọi kernel Adam (xem `MATH/cuda/adam.md` và `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md` §2.1–2.2) | `param, param_grad, exp_avg, exp_avg_sq, visible, lr, b1, b2, eps, N, M` | `ADAM::adamUpdate` |

**Kích thước buffer cấp phát** (không phải công thức số học, nhưng là ràng buộc shape để đọc đúng ý nghĩa tensor, suy từ `rasterize_points.cu` dòng 90–131, 223–232):

$$
\text{out\_color}\in\mathbb R^{C\times H\times W},\ C=\texttt{NUM\_CHAFFELS}\ (=3),\qquad
\text{radii}\in\mathbb Z^{P},\qquad
\text{cov2D}\in\mathbb R^{P\times7}
$$

$$
\text{depth\_map}\in\mathbb R^{H\times W},\quad \text{opacity\_map}\in\mathbb R^{H\times W},\quad \text{normal\_map}\in\mathbb R^{3\times H\times W}\quad(\text{chỉ khi } \texttt{compute\_extra}=\text{true})
$$

$$
\text{metricCount}\in\mathbb Z^{P}\ (\text{chỉ khi }\texttt{get\_flag}=\text{true, khởi tạo }=0)
$$

$$
\text{dL\_dmeans3D}\in\mathbb R^{P\times3},\quad
\text{dL\_dmeans2D}\in\mathbb R^{P\times4}\ (\text{2 chiều gradient chuẩn}+\text{2 chiều "abs-gradient" cho densify})
$$

$$
\text{dL\_dcolors}\in\mathbb R^{P\times C},\quad
\text{dL\_dconic}\in\mathbb R^{P\times2\times2},\quad
\text{dL\_dopacity}\in\mathbb R^{P\times1},\quad
\text{dL\_dcov3D}\in\mathbb R^{P\times6}
$$

$$
\text{dL\_ddc}\in\mathbb R^{P\times1\times3},\quad
\text{dL\_dsh}\in\mathbb R^{P\times M_{\text{sh}}\times3},\quad
\text{dL\_dscales}\in\mathbb R^{P\times3},\quad
\text{dL\_drotations}\in\mathbb R^{P\times4}
$$

với $M_\text{sh}=\texttt{sh.size(1)}$ nếu `sh` khác rỗng (suy ra động từ tensor đầu vào), ngược lại $M_\text{sh}=0$ (trường hợp dùng `colors_precomp`). `geomBuffer`, `binningBuffer`, `imgBuffer`, `sampleBuffer` là buffer byte thô (`torch::kByte`), được các hàm `resizeFunctional`/`resizeIntFunctional`/`resizeFloatFunctional` (dòng 28–50) cấp phát động theo số byte mà `CudaRasterizer::Rasterizer::forward` yêu cầu nội bộ — kích thước cụ thể không được tính trong file này mà do `rasterizer_impl.cu` quyết định qua con trỏ hàm `std::function<char*(size_t)>`.

## `ext.cpp`

Macro `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` (dòng 15–20) đăng ký 4 hàm Python-callable, chỉ là ánh xạ tên thuần tuý, không có logic toán học:

| Tên gọi từ Python | Hàm C++ tương ứng |
|---|---|
| `_C.rasterize_gaussians` | `RasterizeGaussiansCUDA` |
| `_C.rasterize_gaussians_backward` | `RasterizeGaussiansBackwardCUDA` |
| `_C.mark_visible` | `markVisible` |
| `_C.adamUpdate` | `adamUpdate` (wrapper trong `rasterize_points.cu` dòng 295–320, gọi `ADAM::adamUpdate`) |

(Lưu ý: `rasterize_points.h` còn khai báo `conv2DForward`, nhưng hàm này **không** được `ext.cpp` đăng ký qua `m.def`, nên không export sang Python từ file `ext.cpp` này.)

## `diff_gaussian_rasterization_structgs/__init__.py`

Là lớp Python wrapper (`torch.autograd.Function` tên `_RasterizeGaussians`, class `GaussianRasterizer`, `GaussianRasterizationSettings`, `SparseGaussianAdam`). Đã kiểm tra kỹ: **không có công thức chuẩn bị tham số/covariance nào ở đây** — hàm `forward`/`backward` chỉ đóng gói tensor thành `args` tuple rồi gọi thẳng `_C.rasterize_gaussians(*args)` / `_C.rasterize_gaussians_backward(*args)`; việc tính toán 2D covariance, conic, v.v. xảy ra hoàn toàn bên trong CUDA (`forward.cu`). Điểm đáng chú ý duy nhất có tính "công thức" là trong `SparseGaussianAdam.step`:

```python
M = param.numel() // N
_C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)
```

tức $M = \text{numel}(\theta)/N$ (số thành phần mỗi Gaussian) và $\beta_1=0.9,\beta_2=0.999$ hard-code — nội dung này đã được trích đầy đủ trong `MATH/cuda/adam.md` và đối chiếu chi tiết hơn trong `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md` §2.3, không lặp lại ở đây để tránh trùng lặp.

## `setup.py` — build script của 3 submodule

- **`submodules/diff-gaussian-rasterization_structgs/setup.py`**: `CUDAExtension` tên `diff_gaussian_rasterization_structgs._C`, biên dịch `rasterizer_impl.cu`, `forward.cu`, `backward.cu`, `adam.cu`, `rasterize_points.cu`, `ext.cpp`, có include thêm `third_party/glm/`. Chỉ là cấu hình build, không có công thức.
- **`submodules/fused-ssim/setup.py`**: `CUDAExtension` tên `fused_ssim_cuda`, biên dịch `ssim.cu`, `ext.cpp`. Chỉ là cấu hình build, không có công thức.
- **`submodules/simple-knn/setup.py`**: `CUDAExtension` tên `simple_knn._C`, biên dịch `spatial.cu`, `simple_knn.cu`, `ext.cpp`, có cờ `/wd4624` cho MSVC trên Windows. Chỉ là cấu hình build, không có công thức.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `#define BLOCK_X 16` / `#define BLOCK_Y 16` (`config.h`) | kích thước tile $16\times16$ — hằng số, không phải công thức |
| `m.def("rasterize_gaussians", &RasterizeGaussiansCUDA)` (`ext.cpp`) | ánh xạ binding thuần tuý, không có công thức |
| `m.def("adamUpdate", &adamUpdate)` (`ext.cpp`) | ánh xạ binding thuần tuý → `ADAM::adamUpdate` (công thức ở `adam.md`) |
| `M = param.numel() // N` (`__init__.py`, `SparseGaussianAdam.step`) | $M = \dfrac{\text{numel}(\theta)}{N}$ |
| `_C.adamUpdate(..., lr, 0.9, 0.999, eps, N, M)` (`__init__.py`) | $\beta_1=0.9,\ \beta_2=0.999$ (tham chiếu chi tiết: `MATH/cuda/adam.md`, `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md`) |
| `CUDAExtension(..., sources=[...])` (3×`setup.py`) | không có công thức — chỉ liệt kê file nguồn để biên dịch |
