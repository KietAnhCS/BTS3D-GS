# Công thức toán học của Adam CUDA (`adam.cu`/`.h`) và các hàm binding (`rasterize_points.cu`/`.h`, `ext.cpp`)

Tài liệu này kiểm chứng lại nội dung đã có ở `MATH/cuda/adam.md`, `MATH/cuda/rasterize_points.md`, `MATH/cuda/bindings.md` bằng cách đọc lại trực tiếp mã nguồn thật trong submodule `diff-gaussian-rasterization_structgs`, và hợp nhất vào một file duy nhất theo cấu trúc mirror `MATH/submodules/...`.

Nguồn đã đọc:

- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/adam.cu`
- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/adam.h`
- `submodules/diff-gaussian-rasterization_structgs/rasterize_points.cu`
- `submodules/diff-gaussian-rasterization_structgs/rasterize_points.h`
- `submodules/diff-gaussian-rasterization_structgs/ext.cpp`
- `submodules/diff-gaussian-rasterization_structgs/diff_gaussian_rasterization_structgs/__init__.py` (lớp Python `SparseGaussianAdam`, gọi vào binding)
- `scene/gaussian_model.py` (nơi tạo `SparseGaussianAdam` và giá trị `eps` thật dùng trong SADGS)
- `arguments/__init__.py` (giá trị mặc định `adam_eps_order=8`)

---

## 1. Ký hiệu

- $\theta$: một thành phần tham số có thể học (ví dụ một toạ độ của `_xyz`, một hệ số của SH, …), lưu phẳng trong tensor `param`.
- $g_t=\dfrac{\partial\mathcal L}{\partial\theta}$: gradient tại bước thời gian $t$ (một bước `step()` của optimizer), lưu trong `param_grad`.
- $m_t, v_t$: trung bình động mũ bậc 1 và bậc 2 của gradient (`exp_avg`, `exp_avg_sq`).
- $\beta_1,\beta_2$: hệ số suy giảm của EMA.
- $\eta$ (code gọi là `lr`): tốc độ học.
- $\epsilon$: hằng số ổn định số học chống chia 0.
- $N$: số Gaussian (`P` ở nơi khác trong codebase).
- $M$: số thành phần số học trên mỗi Gaussian của tensor tham số đang được cập nhật (`_xyz`→$M=3$, `_scaling`→$M=3$, `_rotation`→$M=4$, `_opacity`→$M=1$, hệ số SH→$M=$ số coeff $\times$ 3 kênh màu, v.v., vì mọi tensor bị `view(-1)`/làm phẳng).
- `tiles_touched[g_idx]` (kiểu `bool*`): mask "Gaussian $g_{idx}$ có phủ ít nhất một tile màn hình trong lượt render hiện tại" — Python truyền vào đây chính là `visibility` (xem §2.3).
- $p_{idx}$: chỉ số luồng phẳng trên lưới $N\times M$; $g_{idx}=\lfloor p_{idx}/M\rfloor$ là chỉ số Gaussian tương ứng.

---

## 2. Công thức Adam suy từ code

### 2.1. Kernel `adamUpdateCUDA` (`adam.cu`, dòng 9–38)

Trích nguyên văn phần tính toán (dòng 23–37):

```cpp
auto p_idx = cg::this_grid().thread_rank();
const uint32_t g_idx = p_idx / M;
if (g_idx >= N) return;
if (tiles_touched[g_idx]) {
    float Register_param_grad = param_grad[p_idx];
    float Register_exp_avg = exp_avg[p_idx];
    float Register_exp_avg_sq = exp_avg_sq[p_idx];
    Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
    Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
    float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);

    param[p_idx] += step;
    exp_avg[p_idx] = Register_exp_avg;
    exp_avg_sq[p_idx] = Register_exp_avg_sq;
}
```

Dịch sang toán học, với $g_{idx}$ là chỉ số Gaussian của luồng $p_{idx}$:

**Cập nhật EMA (mọi $p_{idx}$ có `tiles_touched[g_idx] = true`):**

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t
\tag{dòng 30}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2
\tag{dòng 31}
$$

**Bước cập nhật tham số (KHÔNG có hệ số hiệu chỉnh thiên lệch $\hat m_t,\hat v_t$):**

$$
\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
\tag{dòng 32, 34}
$$

Viết gộp theo chỉ số luồng $p_{idx}$, với $g_{idx}=\lfloor p_{idx}/M\rfloor$ (dòng 24):

$$
p_{idx} = \text{thread\_rank},\qquad g_{idx}=\Big\lfloor\frac{p_{idx}}{M}\Big\rfloor,\qquad\text{bỏ qua nếu } g_{idx}\ge N
\tag{dòng 23–25}
$$

Lưới và block khởi chạy (`ADAM::adamUpdate`, `adam.cu` dòng 53–66):

$$
\text{cnt}=N\cdot M,\qquad \text{grid}=\Big\lceil\frac{N\cdot M}{256}\Big\rceil,\qquad \text{block}=256
$$

### 2.2. Masked / sparse update theo visibility — mục riêng

Đây là điểm khác biệt cốt lõi so với `torch.optim.Adam` chuẩn, đặc trưng cho 3DGS/SADGS. Mask được áp dụng **theo Gaussian**, không theo từng phần tử: toàn bộ $M$ thành phần của một Gaussian cùng được cập nhật hoặc cùng bị bỏ qua, vì điều kiện `if (tiles_touched[g_idx])` chỉ tra theo $g_{idx}=\lfloor p_{idx}/M\rfloor$ chứ không theo $p_{idx}$.

$$
(\theta_i,m_i,v_i)_t=
\begin{cases}
\Big(\theta_{i,t-1}-\eta\dfrac{m_{i,t}}{\sqrt{v_{i,t}}+\epsilon},\ m_{i,t},\ v_{i,t}\Big) & \text{nếu tiles\_touched}_i=\text{true}\\[2mm]
\big(\theta_{i,t-1},\,m_{i,t-1},\,v_{i,t-1}\big) & \text{nếu tiles\_touched}_i=\text{false}
\end{cases}
$$

Ý nghĩa: nếu Gaussian $i$ không phủ tile nào trong batch/view hiện tại (không "nhìn thấy"), **cả ba đại lượng $\theta,m,v$ đều được giữ nguyên tuyệt đối** — không suy giảm theo $\beta_1,\beta_2$ (khác với việc "coi gradient bằng 0" — nếu coi $g=0$ thì $m,v$ vẫn bị nhân suy giảm $\beta_1,\beta_2$; ở đây dòng code bỏ qua hoàn toàn khối `if`, nên $m,v$ không đổi). Đây chính là "sparse Adam": với 3DGS, số Gaussian thường rất lớn ($10^5$–$10^7$) nhưng mỗi lượt render/backward chỉ một tập con nhỏ "nhìn thấy" trong camera hiện tại có gradient khác 0 có ý nghĩa; cập nhật toàn bộ $N$ Gaussian mỗi bước sẽ lãng phí băng thông bộ nhớ. Việc chỉ cập nhật tập con visible giúp tăng tốc đáng kể so với Adam dày đặc.

### 2.3. Giá trị $\beta_1,\beta_2,\epsilon$ thật từ code

Phía Python, `SparseGaussianAdam.step` (`diff_gaussian_rasterization_structgs/__init__.py`, dòng 249–276):

```python
class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)

    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            ...
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)
```

Xác nhận:

- $\beta_1=0.9,\ \beta_2=0.999$ **hard-code trực tiếp tại lệnh gọi** (dòng 276), không đọc từ `group["betas"]` — dù lớp kế thừa `torch.optim.Adam` (vốn có `betas`), giá trị `betas` của `torch.optim.Adam` không hề được dùng tới trong `step()` ghi đè này.
- $\eta=$ `group["lr"]` (đọc động mỗi bước, cho phép LR-scheduler cập nhật `param_groups[i]["lr"]` theo iteration).
- $\epsilon=$ `group["eps"]`, được gán khi khởi tạo optimizer. Trong SADGS thật, tại `scene/gaussian_model.py` dòng 315, 321, 324:

```python
adam_eps = 10**(-training_args.adam_eps_order)
...
self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=adam_eps)   # optimizer_type == "sparse_adam"
...
self.shoptimizer = SparseGaussianAdam(sh_l, lr=0.0, eps=adam_eps)      # optimizer_type == "hybrid"
```

với `adam_eps_order` mặc định $=8$ (`arguments/__init__.py` dòng 156), tức:

$$
\boxed{\ \beta_1=0.9,\quad \beta_2=0.999,\quad \epsilon=10^{-8}\ \text{(mặc định)}\ }
$$

(`lr=0.0` chỉ là giá trị khởi tạo placeholder cho `param_groups`; LR thật được scheduler ghi đè theo iteration trước khi gọi `step()`, giống `torch.optim.Adam` chuẩn — không ảnh hưởng tới công thức.) `adam_eps_order` có thể truyền `--adam_eps_order 10` từ dòng lệnh (ví dụ Mip-NeRF 360 indoor trong `run_train.sh`), cho $\epsilon=10^{-10}$.

`state['step']` được khởi tạo (`state['step'] = torch.tensor(0.0, ...)`, dòng 267) nhưng **không bao giờ được tăng hay đọc** trong phần còn lại của `step()` — xác nhận thêm rằng không có cơ chế bias-correction theo $t$ nào tồn tại trong đường đi sparse Adam này (không giống `torch.optim.Adam` gốc của PyTorch, nơi `state['step']` được dùng để tính $1-\beta^t$).

### 2.4. Tóm tắt các hàm binding

#### `rasterize_points.cu` / `.h`

File này là lớp **glue** C++/CUDA ↔ PyTorch, không chứa công thức toán mới — chỉ kiểm tra shape, cấp phát/resize tensor đệm, và chuyển con trỏ dữ liệu thô vào các hàm CUDA đã có công thức đầy đủ ở nơi khác (`forward.cu`, `backward.cu`, `rasterizer_impl.cu`, `adam.cu`).

| Hàm C++ (`rasterize_points.h`) | Vai trò | Tham số chính | Gọi vào |
|---|---|---|---|
| `RasterizeGaussiansCUDA` | Forward rasterization: render ảnh RGB + (tuỳ chọn) depth/opacity/normal map từ $N$ Gaussian 3D | `means3D, colors, opacity, scales, rotations, scale_modifier, cov3D_precomp, metric_map, viewmatrix, projmatrix, tan_fovx/fovy, H, W, dc, sh, degree, campos, mult, prefiltered, debug, get_flag, compute_extra` | `CudaRasterizer::Rasterizer::forward` |
| `RasterizeGaussiansBackwardCUDA` | Backward: lan truyền $\partial\mathcal L/\partial(\text{out\_color})$ ngược về gradient của mọi tham số Gaussian | `dL_dout_color, geomBuffer, binningBuffer, imageBuffer, sampleBuffer, R, B, ...` (các tensor forward lưu lại) | `CudaRasterizer::Rasterizer::backward` |
| `markVisible` | Frustum-culling: đánh dấu Gaussian nào nằm trong view frustum của camera | `means3D, viewmatrix, projmatrix` → trả `present` (bool mỗi Gaussian) | `CudaRasterizer::Rasterizer::markVisible` |
| `adamUpdate` | Wrapper gọi kernel Adam đã mô tả ở §2.1–2.2 | `param, param_grad, exp_avg, exp_avg_sq, visible, lr, b1, b2, eps, N, M` | `ADAM::adamUpdate` |

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

#### `ext.cpp`

Macro `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` (dòng 15–20) đăng ký 4 hàm Python-callable, chỉ là ánh xạ tên thuần tuý, không có logic toán học:

| Tên gọi từ Python | Hàm C++ tương ứng |
|---|---|
| `_C.rasterize_gaussians` | `RasterizeGaussiansCUDA` |
| `_C.rasterize_gaussians_backward` | `RasterizeGaussiansBackwardCUDA` |
| `_C.mark_visible` | `markVisible` |
| `_C.adamUpdate` | `adamUpdate` (wrapper trong `rasterize_points.cu` dòng 295–320, gọi `ADAM::adamUpdate`) |

(Lưu ý: `rasterize_points.h` còn khai báo `conv2DForward`, nhưng hàm này **không** được `ext.cpp` đăng ký qua `m.def`, nên không export sang Python từ file `ext.cpp` này.)

---

## 3. Kiến thức toán nền tảng

**Stochastic Gradient Descent (SGD).** Cập nhật tham số ngược chiều gradient của hàm mất mát trên minibatch: $\theta_t=\theta_{t-1}-\eta g_t$. Nhược điểm: cùng một $\eta$ cho mọi chiều, dao động mạnh khi gradient có độ lớn khác nhau giữa các tham số hoặc theo thời gian.

**Động lượng (momentum).** Thay gradient tức thời bằng trung bình trượt của các gradient quá khứ để làm mượt hướng đi: $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ — đây là **trung bình động mũ (exponential moving average — EMA)** của $g_t$, với "cửa sổ hiệu dụng" xấp xỉ $1/(1-\beta_1)$ bước gần nhất.

**Adaptive moment estimation (Adam, Kingma & Ba 2015).** Kết hợp hai EMA:

- Bậc 1 ($m_t$): ước lượng kỳ vọng gradient (hướng trung bình) — đóng vai trò động lượng.
- Bậc 2 ($v_t$): ước lượng kỳ vọng $g_t^2$ (độ lớn bình phương trung bình) — dùng để **chuẩn hoá tốc độ học theo từng tham số** (adaptive learning rate): tham số có gradient lớn/biến động mạnh bị chia cho $\sqrt{v_t}$ lớn ⇒ bước đi nhỏ lại; tham số có gradient nhỏ/ổn định được bước lớn hơn tương đối.

**Hiệu chỉnh thiên lệch (bias correction).** Vì $m_0=v_0=0$, các EMA ở bước đầu bị thiên lệch về 0 (kỳ vọng $\mathbb E[m_t]\ne\mathbb E[g_t]$ khi $t$ nhỏ). Kingma & Ba chia cho $(1-\beta^t)\to$ hiệu chỉnh:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Khi $t\to\infty$, $\beta_1^t,\beta_2^t\to0$ nên $\hat m_t\to m_t,\ \hat v_t\to v_t$ — hiệu chỉnh chỉ có tác dụng đáng kể ở **những bước đầu** của quá trình tối ưu (hoặc mỗi khi EMA "khởi động lại" từ 0, ví dụ sau khi một Gaussian mới được tạo ra bởi densification — state Adam của nó bắt đầu lại từ $m=v=0$).

**$\epsilon$ (hằng số ổn định số học).** Tránh chia cho 0 khi $v_t\approx0$ (gradient liên tục rất nhỏ hoặc bằng 0): $\theta_t=\theta_{t-1}-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$.

**Sparsification theo visibility (đặc trưng 3DGS).** Trong Gaussian Splatting, số điểm $N$ rất lớn nhưng mỗi view/batch chỉ một tập con "nhìn thấy" (visible, phủ ít nhất 1 tile) có gradient có ý nghĩa — gradient của Gaussian không nhìn thấy đúng bằng 0 hoặc không được tính. Dense Adam (cập nhật mọi $N$ tham số mỗi bước, kể cả khi $g_t=0$) lãng phí băng thông bộ nhớ GPU vì phải đọc/ghi state của toàn bộ $N$ điểm dù phần lớn không đổi. "Sparse Adam" (như cài đặt ở `adam.cu`) giới hạn vùng nhớ đọc/ghi chỉ ở tập con visible, giữ nguyên tuyệt đối state của phần còn lại — một kỹ thuật tối ưu hoá hiệu năng (hardware-level), không làm thay đổi bản chất thuật toán Adam áp dụng *trên tập con đó*.

---

## 4. Kiểm chứng tính đúng sai

Đối chiếu trực tiếp với công thức Adam **gốc** trong Kingma & Ba, *"Adam: A Method for Stochastic Optimization"*, ICLR 2015 (Algorithm 1):

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=m_t/(1-\beta_1^t),\qquad \hat v_t=v_t/(1-\beta_2^t)
$$

$$
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

| Thành phần | Paper gốc (Kingma & Ba 2015) | Code (`adam.cu` dòng 30–34) | Khớp? |
|---|---|---|---|
| Cập nhật $m_t$ | $\beta_1 m_{t-1}+(1-\beta_1)g_t$ | `b1*exp_avg + (1-b1)*grad` | **Khớp** |
| Cập nhật $v_t$ | $\beta_2 v_{t-1}+(1-\beta_2)g_t^2$ | `b2*exp_avg_sq + (1-b2)*grad*grad` | **Khớp** |
| Bias correction $\hat m_t,\hat v_t$ | **Có**, chia cho $1-\beta^t$ | **Không có** — dùng thẳng $m_t,v_t$ | **Khác biệt chính** |
| Thứ tự $\epsilon$: trong hay ngoài căn | Paper: $\epsilon$ cộng **sau** $\sqrt{\hat v_t}$ (ngoài căn): $\sqrt{\hat v_t}+\epsilon$ | `sqrt(Register_exp_avg_sq) + eps` — cộng **sau** căn, cùng vị trí | **Khớp** (không phải biến thể "epsilon trong căn" như một số cài đặt khác, ví dụ TensorFlow Adam mặc định) |
| Dấu bước cập nhật | $\theta_t=\theta_{t-1}-\eta(\cdot)$ | `step = -lr*(...)`, `param += step` | **Khớp** (trừ đi, không phải cộng) |
| $\beta_1,\beta_2$ | Khuyến nghị mặc định $0.9,\,0.999$ | hard-code $0.9,\,0.999$ tại lệnh gọi Python | **Khớp giá trị khuyến nghị** |
| $\epsilon$ | Khuyến nghị $10^{-8}$ | `adam_eps = 10**(-adam_eps_order)`, mặc định `adam_eps_order=8` ⇒ $10^{-8}$ | **Khớp giá trị mặc định** (có thể đổi qua CLI, ví dụ $10^{-10}$ cho Mip-NeRF 360 indoor) |
| Cập nhật dày đặc mọi $\theta$ mỗi bước | Có (không có khái niệm visibility) | **Không** — chỉ cập nhật Gaussian có `tiles_touched[g_idx]=true`; phần còn lại giữ nguyên $\theta,m,v$ | **Mở rộng có chủ đích** so với paper gốc, không phải lỗi — đặc trưng 3DGS/SADGS |

**Kết luận:** công thức cập nhật $m_t, v_t$ và bước gradient $\theta_t=\theta_{t-1}-\eta\,m_t/(\sqrt{v_t}+\epsilon)$ trong `adam.cu` khớp chính xác với phần "không hiệu chỉnh" của Adam gốc; khác biệt duy nhất và có chủ đích là **bỏ hoàn toàn bias correction** ($\hat m_t,\hat v_t$ không tồn tại trong code — không phải lỗi đọc nhầm, dòng `float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);` dùng thẳng $m_t,v_t$ chưa chia $(1-\beta^t)$). Đây **không phải bug code**: đây là biến thể đã biết, triển khai y hệt trong bản gốc 3DGS chính thức (Taming 3DGS / gsplat "SparseAdam") — lý do là 3DGS huấn luyện hàng chục nghìn bước, nên với $t$ đủ lớn, $1-\beta^t\to1$ và sai số do bỏ bias-correction tiệm cận 0; đồng thời mỗi Gaussian có thể được tạo ra giữa chừng (qua densification/split/clone) nên "bước $t$ toàn cục" không khớp với "số lần cập nhật cục bộ" của từng Gaussian — `state['step']` bị bỏ không tăng chính là hệ quả hợp lý của việc không dùng bias correction theo $t$ toàn cục nữa.

**Rủi ro cần lưu ý (không phải lỗi, nhưng là điểm cần hiểu đúng):** ở những bước đầu tiên sau khi $m=v=0$ (lúc khởi tạo optimizer, hoặc sau khi một Gaussian mới được densify và state Adam của nó reset về 0), việc bỏ bias-correction có thể làm bước cập nhật **lệch đáng kể** so với Adam chuẩn — không nhất thiết luôn nhỏ hơn, vì tỉ số $m_t/\sqrt{v_t}$ không bị chia đều bởi cùng một hệ số cho tử và mẫu như $\hat m_t/\sqrt{\hat v_t}$. Ví dụ số ở §5 minh hoạ cụ thể mức lệch này.

**So với 3 bản phân tích cũ (`MATH/cuda/adam.md`, `rasterize_points.md`, `bindings.md`):** nội dung toán học (công thức EMA, bỏ bias-correction, mask theo `tiles_touched`, $\beta_1=0.9,\beta_2=0.999$ hard-code, lưới $N\times M$) đã **đúng và khớp hoàn toàn** với code đọc lại lần này — không phát hiện sai sót cần sửa. Bổ sung so với bản cũ trong file này:

1. Xác nhận thêm giá trị $\epsilon$ **thật dùng trong SADGS** (không chỉ "tham số truyền vào"): $\epsilon=10^{-8}$ mặc định, suy từ `scene/gaussian_model.py` + `arguments/__init__.py` (`adam_eps_order=8`), có thể là $10^{-10}$ nếu chạy với cờ `--adam_eps_order 10`.
2. Đối chiếu tường minh với bảng thuật toán gốc Kingma & Ba 2015, chỉ rõ vị trí cộng $\epsilon$ (ngoài căn, khớp paper) — điều 3 bản cũ không nêu.
3. Làm rõ ngữ nghĩa "giữ nguyên tuyệt đối" của Gaussian không visible (không phải "coi gradient = 0 rồi vẫn suy giảm $m,v$").
4. Nêu rõ `conv2DForward` khai báo trong `rasterize_points.h` nhưng không được `ext.cpp` export.
5. Không phát hiện bug trong logic Adam hay binding.

---

## 5. Ví dụ số

Tham số $\theta_0=1.0$. Hai bước liên tiếp, Gaussian này **visible cả hai lần** (`tiles_touched=true`), $g_1=0.5,\ g_2=-0.2$. Dùng đúng giá trị mặc định thật của SADGS: $\beta_1=0.9,\ \beta_2=0.999,\ \eta=0.01,\ \epsilon=10^{-8}$. Khởi tạo $m_0=v_0=0$.

### Bước 1 ($g_1=0.5$) — theo code (không bias correction)

$$
m_1=0.9\times0+0.1\times0.5=0.05
$$
$$
v_1=0.999\times0+0.001\times0.5^2=0.00025
$$
$$
\sqrt{v_1}=0.0158114
$$
$$
\Delta\theta_1=-0.01\times\frac{0.05}{0.0158114+10^{-8}}=-0.01\times3.16228=-0.0316228
$$
$$
\theta_1=1.0-0.0316228=0.9683772
$$

### Bước 2 ($g_2=-0.2$) — theo code

$$
m_2=0.9\times0.05+0.1\times(-0.2)=0.045-0.02=0.025
$$
$$
v_2=0.999\times0.00025+0.001\times0.04=0.00024975+0.00004=0.00028975
$$
$$
\sqrt{v_2}=0.0170220
$$
$$
\Delta\theta_2=-0.01\times\frac{0.025}{0.0170220+10^{-8}}=-0.01\times1.46842=-0.0146842
$$
$$
\theta_2=0.9683772-0.0146842=0.9536930
$$

**Kết quả theo code:** $\theta_0=1.0\to\theta_1=0.9683772\to\theta_2=0.9536930$ — khớp chính xác với công thức $\theta_t=\theta_{t-1}-\eta\, m_t/(\sqrt{v_t}+\epsilon)$ trích từ `adam.cu` dòng 30–34.

### Đối chứng: cùng $m_t,v_t$ nhưng CÓ bias correction (Adam gốc Kingma & Ba)

$$
\hat m_1=\frac{0.05}{1-0.9^1}=\frac{0.05}{0.1}=0.5,\qquad
\hat v_1=\frac{0.00025}{1-0.999^1}=\frac{0.00025}{0.001}=0.25,\qquad\sqrt{\hat v_1}=0.5
$$
$$
\Delta\theta_1^{\text{std}}=-0.01\times\frac{0.5}{0.5+10^{-8}}=-0.0099999...\approx-0.01,\qquad \theta_1^{\text{std}}=0.99
$$

$$
\hat m_2=\frac{0.025}{1-0.9^2}=\frac{0.025}{0.19}=0.131579,\qquad
\hat v_2=\frac{0.00028975}{1-0.999^2}=\frac{0.00028975}{0.001999}=0.144947,\qquad\sqrt{\hat v_2}=0.380719
$$
$$
\Delta\theta_2^{\text{std}}=-0.01\times\frac{0.131579}{0.380719+10^{-8}}=-0.0034560,\qquad \theta_2^{\text{std}}=0.99-0.0034560=0.9865440
$$

**So sánh:** $\theta_2$ theo code $=0.9536930$ so với $\theta_2^{\text{std}}$ (Adam có bias-correction) $=0.9865440$ — lệch khoảng $0.033$, tức **không hề nhỏ** ở bước thứ 2. Điều này xác nhận nhận định ở §4: việc bỏ bias-correction có tác động rõ rệt ở các bước đầu (ngay sau khi $m=v=0$), và chỉ "biến mất" khi $t$ đủ lớn để $\beta_1^t,\beta_2^t\approx0$ — đúng với giả định vận hành của SADGS (hàng chục nghìn iteration), nhưng cũng đúng là mỗi Gaussian mới sinh ra giữa chừng (densify) sẽ trải qua giai đoạn lệch này cục bộ.

### Minh hoạ masked update (Gaussian không visible ở bước 2)

Nếu giả sử Gaussian trên **không visible** ở bước 2 (`tiles_touched[g_idx]=false`), nhánh `if` trong kernel bị bỏ qua hoàn toàn:

$$
\theta_2=\theta_1=0.9683772,\qquad m_2=m_1=0.05,\qquad v_2=v_1=0.00025
$$

tức là **đứng yên tuyệt đối** — không suy giảm theo $\beta_1,\beta_2$ như thể "gradient bằng 0", đúng với phân tích ở §2.2.
