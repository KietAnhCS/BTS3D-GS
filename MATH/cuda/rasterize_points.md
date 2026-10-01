# Công thức toán học trong `rasterize_points.cu` / `.h`

> Đã đồng bộ với bản đối chiếu kỹ hơn tại `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md` (mục "2.4. Tóm tắt các hàm binding"). Tài liệu này giữ cấu trúc mirror thư mục `cuda/` nhưng nội dung đã được kiểm chứng lại bằng cách đọc trực tiếp `submodules/diff-gaussian-rasterization_structgs/rasterize_points.cu` và `.h` thật (không chỉ phần Adam).

File này là lớp **glue** giữa PyTorch (`torch::Tensor`) và các hàm CUDA thuần tuý đã khai báo ở `CudaRasterizer::Rasterizer` (`forward`, `backward`, `markVisible`) và `ADAM::adamUpdate`. Bản thân file **không chứa công thức toán học mới** — nó chỉ:

- Kiểm tra kích thước tensor đầu vào (`means3D` phải có shape `(P,3)`).
- Cấp phát/resize các tensor đệm (`geomBuffer`, `binningBuffer`, `imgBuffer`, `sampleBuffer`, `metricCount`, `depth_map`, `opacity_map`, `normal_map`) và các tensor gradient đầu ra (`dL_dmeans3D`, `dL_dmeans2D`, v.v.) với kích thước cố định suy từ $P$ (số Gaussian), $M$ (số hệ số SH mỗi Gaussian = `sh.size(1)`), $H,W$ (kích thước ảnh).
- Chuyển con trỏ dữ liệu thô (`data_ptr<float>()`, v.v.) sang các hàm CUDA đã mô tả công thức đầy đủ ở các file khác:
  - `RasterizeGaussiansCUDA` → gọi `CudaRasterizer::Rasterizer::forward` (xem `MATH/cuda/forward.md` và `MATH/cuda/rasterizer_impl.md`).
  - `RasterizeGaussiansBackwardCUDA` → gọi `CudaRasterizer::Rasterizer::backward` (xem `MATH/cuda/backward.md`).
  - `markVisible` → gọi `CudaRasterizer::Rasterizer::markVisible`, dùng công thức frustum-culling đã mô tả ở `MATH/cuda/auxiliary.md` (mục "Kiểm tra điểm trong frustum").
  - `adamUpdate` → gọi `ADAM::adamUpdate` (xem `MATH/cuda/adam.md` và chi tiết đầy đủ ở `adam_and_bindings.md` §2).

---

## 1. `RasterizeGaussiansCUDA` — forward binding (`rasterize_points.cu`, dòng 52–184)

### 1.1. Kiểm tra shape và các đại lượng cơ bản (dòng 79–85)

```cpp
if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
}

const int P = means3D.size(0);
const int H = image_height;
const int W = image_width;
```

Ràng buộc shape: $\mu\in\mathbb{R}^{P\times3}$ (không phải công thức số học, nhưng là tiền điều kiện bắt buộc để mọi công thức ở `forward.cu` hợp lệ). $P$ là số Gaussian, $H,W$ là chiều cao/rộng ảnh render.

### 1.2. Khởi tạo các tensor đầu ra (dòng 90–92)

```cpp
torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
torch::Tensor cov2D = torch::full({P, 7}, 0.0, float_opts);
```

$$
\text{out\_color}\in\mathbb R^{C\times H\times W},\ C=\texttt{NUM\_CHAFFELS}\ (=3),\qquad
\text{radii}\in\mathbb Z^{P},\qquad
\text{cov2D}\in\mathbb R^{P\times7}
$$

tất cả khởi tạo bằng 0 (hoặc `false`/0 cho `radii`), được `CudaRasterizer::Rasterizer::forward` ghi đè sau đó.

### 1.3. Buffer byte thô cấp phát động (dòng 94–103, 28–50)

```cpp
torch::Device device(torch::kCUDA);
torch::TensorOptions options(torch::kByte);
torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);
```

và hàm resize dùng chung (dòng 28–34, mẫu cho cả ba biến thể `resizeFunctional`/`resizeIntFunctional`/`resizeFloatFunctional`, dòng 28–50):

```cpp
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}
```

Đây không phải công thức toán mà là cơ chế cấp phát trễ (lazy allocation): kích thước byte thật sự cần ($N$ trong `resize_({N})`) do `CudaRasterizer::Rasterizer::forward` quyết định nội bộ (qua `rasterizer_impl.cu`) và gọi lại con trỏ hàm này khi biết kích thước, không được tính trong `rasterize_points.cu`.

### 1.4. `metricCount` (đếm cờ metric, dòng 105–113)

```cpp
int* accum_metric_counts_ptr = nullptr;

torch::Tensor metricCount = torch::empty({0}, int_opts);

if(get_flag)
{
	metricCount = torch::full({P}, 0, int_opts);
	accum_metric_counts_ptr = metricCount.contiguous().data<int>();
}
```

$$
\text{metricCount}\in\mathbb Z^{P}\ (\text{chỉ khi }\texttt{get\_flag}=\text{true, khởi tạo }=0)
$$

### 1.5. Các bản đồ phụ trợ depth/opacity/normal (dòng 115–130)

```cpp
torch::Tensor depth_map = torch::empty({0}, float_opts);
torch::Tensor opacity_map = torch::empty({0}, float_opts);
torch::Tensor normal_map = torch::empty({0}, float_opts);
float* depth_map_ptr = nullptr;
float* opacity_map_ptr = nullptr;
float* normal_map_ptr = nullptr;
if (compute_extra)
{
    depth_map = torch::zeros({H, W}, float_opts);
    opacity_map = torch::zeros({H, W}, float_opts);
    normal_map = torch::zeros({3, H, W}, float_opts);
    depth_map_ptr = depth_map.contiguous().data_ptr<float>();
    opacity_map_ptr = opacity_map.contiguous().data_ptr<float>();
    normal_map_ptr = normal_map.contiguous().data_ptr<float>();
}
```

$$
\text{depth\_map}\in\mathbb R^{H\times W},\quad \text{opacity\_map}\in\mathbb R^{H\times W},\quad \text{normal\_map}\in\mathbb R^{3\times H\times W}\quad(\text{chỉ khi } \texttt{compute\_extra}=\text{true})
$$

Nếu `compute_extra=false`, cả ba vẫn tồn tại dưới dạng tensor rỗng `{0}` và các con trỏ tương ứng là `nullptr` — `forward.cu` phải tự kiểm tra `nullptr` trước khi ghi.

### 1.6. Suy luận $M$ (số hệ số SH) và gọi vào `Rasterizer::forward` (dòng 134–183)

```cpp
int rendered = 0;
int num_buckets = 0;
if(P != 0)
{
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  auto tup = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		sampleFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		dc.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(),
		opacity.contiguous().data<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),
		metric_map.contiguous().data<int>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
        mult,
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		cov2D.contiguous().data_ptr<float>(),
		debug,
		get_flag,
		accum_metric_counts_ptr,
		compute_extra,
		depth_map_ptr,
		opacity_map_ptr,
		normal_map_ptr);

		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
}
return std::make_tuple(rendered, num_buckets, out_color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, metricCount, cov2D, depth_map, opacity_map, normal_map);
```

$$
M=\texttt{sh.size}(1)\ \text{nếu}\ \texttt{sh.size}(0)\neq0,\qquad M=0\ \text{ngược lại (trường hợp dùng}\ \texttt{colors\_precomp}\text{)}
$$

Nếu $P=0$ (không có Gaussian nào), toàn bộ khối `if(P != 0)` bị bỏ qua, hàm trả về các tensor rỗng/0 đã khởi tạo — không gọi CUDA kernel nào, tránh lỗi grid/block size = 0.

---

## 2. `RasterizeGaussiansBackwardCUDA` — backward binding (`rasterize_points.cu`, dòng 186–272)

### 2.1. Kích thước cơ bản và $M$ (dòng 213–221)

```cpp
const int P = means3D.size(0);
const int H = dL_dout_color.size(1);
const int W = dL_dout_color.size(2);

int M = 0;
if(sh.size(0) != 0)
{	
	M = sh.size(1);
}
```

Khác với forward, $H,W$ ở backward được suy ngược từ shape của gradient đầu vào `dL_dout_color` (không truyền `image_height`/`image_width` trực tiếp).

### 2.2. Khởi tạo các tensor gradient đầu ra (dòng 223–232)

```cpp
torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());  // abs
torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
torch::Tensor dL_ddc = torch::zeros({P, 1, 3}, means3D.options());
torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
```

$$
\text{dL\_dmeans3D}\in\mathbb R^{P\times3},\quad
\text{dL\_dmeans2D}\in\mathbb R^{P\times4}\ (\text{2 chiều gradient chuẩn}+\text{2 chiều "abs-gradient" cho densify, xem}\ \texttt{add\_densification\_stats}\ \text{ở}\ \texttt{gaussian\_model.py})
$$

$$
\text{dL\_dcolors}\in\mathbb R^{P\times C},\quad
\text{dL\_dconic}\in\mathbb R^{P\times2\times2},\quad
\text{dL\_dopacity}\in\mathbb R^{P\times1},\quad
\text{dL\_dcov3D}\in\mathbb R^{P\times6}\ (\text{tam giác trên đối xứng, khớp}\ \texttt{strip\_symmetric})
$$

$$
\text{dL\_ddc}\in\mathbb R^{P\times1\times3},\quad
\text{dL\_dsh}\in\mathbb R^{P\times M\times3},\quad
\text{dL\_dscales}\in\mathbb R^{P\times3},\quad
\text{dL\_drotations}\in\mathbb R^{P\times4}
$$

Tất cả khởi tạo bằng 0, và nếu $P=0$ được trả về nguyên trạng (khối `if(P != 0)` ở dòng 235–270 bị bỏ qua, không gọi `Rasterizer::backward`).

### 2.3. Gọi vào `Rasterizer::backward` (dòng 235–271)

```cpp
if(P != 0)
{  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  dc.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_ddc.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
}

return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_ddc, dL_dsh, dL_dscales, dL_drotations);
```

Đây thuần là truyền con trỏ dữ liệu thô vào hàm backward thật — toàn bộ công thức chain-rule ($\partial\mathcal L/\partial\theta$ cho từng $\theta\in\{\mu,\Sigma,c,\alpha,\text{SH},\dots\}$) nằm ở `backward.cu`/`rasterizer_impl.cu`, không ở đây. $R$ (tổng số instance đã render) và $B$ (số bucket) là hai giá trị vô hướng được forward trả về trước đó và truyền lại nguyên vẹn.

---

## 3. `markVisible` — frustum culling binding (`rasterize_points.cu`, dòng 274–293)

```cpp
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
```

$$
\text{present}\in\{\text{true},\text{false}\}^{P},\qquad \text{present}_i = \text{true} \iff \text{Gaussian } i \text{ nằm trong view frustum của camera (}viewmatrix, projmatrix\text{)}
$$

Công thức kiểm tra điểm trong frustum (phép chiếu homogeneous $p_{\text{proj}} = \text{projmatrix}\cdot[\mu_i,1]^\top$ và so sánh $w$ với ngưỡng) được cài đặt thật trong `CudaRasterizer::Rasterizer::markVisible`, mô tả chi tiết ở `MATH/cuda/auxiliary.md` (mục "Kiểm tra điểm trong frustum") — file này chỉ cấp phát `present` và chuyển con trỏ.

---

## 4. `adamUpdate` — wrapper gọi kernel Adam (`rasterize_points.cu`, dòng 295–320)

```cpp
void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}
```

Thuần tuý chuyển tiếp 11 tham số sang `ADAM::adamUpdate` (định nghĩa trong `cuda_rasterizer/adam.cu`/`.h`), không thêm phép biến đổi nào. Công thức Adam đầy đủ (EMA bậc 1/2, bỏ bias-correction, mask theo `tiles_touched`/`visible`) đã trình bày chi tiết ở `MATH/cuda/adam.md` và đối chiếu kỹ hơn ở `adam_and_bindings.md` §2.1–2.2:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

áp dụng theo mask Gaussian-wise: chỉ cập nhật $(\theta,m,v)$ của Gaussian $i$ nếu `visible[i] = true`; ngược lại giữ nguyên tuyệt đối.

---

## 5. Khai báo header `rasterize_points.h`

Toàn bộ `rasterize_points.h` (dòng 18–90) chỉ là **signature** (không có thân hàm) của 5 hàm: `RasterizeGaussiansCUDA` (dòng 19–43), `RasterizeGaussiansBackwardCUDA` (dòng 46–70), `markVisible` (dòng 72–75), `conv2DForward` (dòng 77, **chỉ khai báo, không có định nghĩa/triển khai trong các file đã đọc, và không được `ext.cpp` đăng ký qua `m.def` — không export sang Python**), và `adamUpdate` (dòng 79–90). Không có công thức toán học nào trong file header.

### Bảng tóm tắt (đối chiếu `adam_and_bindings.md` §2.4)

| Hàm C++ (`rasterize_points.h`) | Vai trò | Gọi vào |
|---|---|---|
| `RasterizeGaussiansCUDA` | Forward rasterization: render ảnh RGB + (tuỳ chọn) depth/opacity/normal map từ $P$ Gaussian 3D | `CudaRasterizer::Rasterizer::forward` |
| `RasterizeGaussiansBackwardCUDA` | Backward: lan truyền $\partial\mathcal L/\partial(\text{out\_color})$ ngược về gradient của mọi tham số Gaussian | `CudaRasterizer::Rasterizer::backward` |
| `markVisible` | Frustum-culling: đánh dấu Gaussian nào nằm trong view frustum của camera | `CudaRasterizer::Rasterizer::markVisible` |
| `adamUpdate` | Wrapper gọi kernel Adam | `ADAM::adamUpdate` |
| `conv2DForward` | Khai báo trong header nhưng **không** export qua `ext.cpp`, không dùng trong pipeline Python hiện tại | (không rõ, không có định nghĩa trong các file đã đọc) |

---

## 6. Kiểm chứng tính đúng sai

So với bản cũ của file này (trước khi đồng bộ): bản cũ chỉ liệt kê shape các tensor ở dạng bảng tóm tắt, **không trích dẫn code thật kèm số dòng** cho từng công thức/khởi tạo, và không tách riêng phần `RasterizeGaussiansBackwardCUDA`/`markVisible`/`adamUpdate` thành các mục có code trích dẫn. Sau khi đọc lại trực tiếp `rasterize_points.cu` (320 dòng) và `rasterize_points.h` (90 dòng) thật:

- Không phát hiện sai sót toán học nào trong nội dung cũ — các shape tensor ($P\times3$, $P\times4$, $P\times6$, $C\times H\times W$, v.v.) đã liệt kê đều khớp chính xác với khai báo `torch::zeros`/`torch::full` thật trong code.
- Bổ sung: phần `RasterizeGaussiansBackwardCUDA` (trước đây chỉ nhắc tên, chưa trích code), `markVisible` (công thức frustum present/absent), `adamUpdate` (wrapper, liên kết công thức Adam đầy đủ), và ghi chú $H,W$ ở backward được suy từ `dL_dout_color.size()` chứ không truyền trực tiếp — chi tiết này bản cũ chưa nêu.
- Xác nhận lại: `conv2DForward` khai báo ở header nhưng không có định nghĩa trong hai file nguồn này và không được `ext.cpp` đăng ký — khớp với ghi chú ở `adam_and_bindings.md`.

Không có công thức toán học nào trong `rasterize_points.cu`/`.h` mà sai khác so với các file CUDA nó gọi vào (`forward.cu`, `backward.cu`, `adam.cu`, `auxiliary.md`) — đây đúng như mô tả ban đầu, là lớp glue thuần tuý.
