# Công thức toán học trong `cuda_rasterizer/adam.h` và `cuda_rasterizer/adam.cu`

Tài liệu mô tả bộ tối ưu Adam cài đặt trực tiếp trên CUDA, dùng khi `optimizer_type="sparse_adam"` trong `gaussian_model.py` (thông qua lớp Python `SparseGaussianAdam` ở `diff_gaussian_rasterization_structgs/__init__.py`, gọi hàm binding `_C.adamUpdate` → `ADAM::adamUpdate` → kernel `adamUpdateCUDA`).

Nguồn: `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/adam.cu` và `adam.h`. Đã đọc lại trực tiếp hai file này để kiểm chứng số dòng (xem thêm bản đối chiếu đầy đủ hơn, bao gồm cả các hàm binding, tại `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md` — nội dung công thức của file đó và file này khớp nhau hoàn toàn).

---

## 0. Ký hiệu

- $\theta$: một thành phần tham số có thể học, lưu phẳng trong tensor `param`.
- $g_t=\partial\mathcal L/\partial\theta$: gradient tại bước $t$, lưu trong `param_grad`.
- $m_t, v_t$: EMA bậc 1 và bậc 2 của gradient (`exp_avg`, `exp_avg_sq`).
- $\beta_1,\beta_2$: hệ số suy giảm EMA (`b1`, `b2`).
- $\eta$: tốc độ học (`lr`).
- $\epsilon$: hằng số ổn định số học.
- $N$: số Gaussian; $M$: số thành phần số học trên mỗi Gaussian của tensor tham số đang cập nhật.
- $p_{idx}$: chỉ số luồng phẳng trên lưới $N\times M$; $g_{idx}=\lfloor p_{idx}/M\rfloor$ là chỉ số Gaussian tương ứng.
- `tiles_touched[g_idx]`: mask "Gaussian $g_{idx}$ có phủ ít nhất 1 tile màn hình ở lượt render hiện tại" (Python truyền `visibility` vào đây).

## 1. Adam chuẩn (tham chiếu lý thuyết)

Với tham số $\theta$, gradient $g_t$ tại bước $t$, hệ số $\beta_1,\beta_2$, tốc độ học $\eta$, hằng số ổn định $\epsilon$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

## 2. Đối chiếu với code — biến thể "visible-only", KHÔNG bias-correction

### 2.1. Kernel `adamUpdateCUDA` (`adam.cu`, dòng 9–38)

Trích nguyên văn toàn bộ thân kernel (dòng 9–38):

```cpp
__global__
void adamUpdateCUDA(
    float* __restrict__ param,
    const float* __restrict__ param_grad,
    float* __restrict__ exp_avg,
    float* __restrict__ exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M) {

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
}
```

**Chỉ số luồng và chỉ số Gaussian** (dòng 23–25):

```cpp
	auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / M;
    if (g_idx >= N) return;
```

$$
p_{idx} = \text{thread\_rank},\qquad g_{idx} = \left\lfloor \frac{p_{idx}}{M} \right\rfloor,\qquad \text{(bỏ qua nếu } g_{idx}\ge N)
$$

**Cập nhật EMA và bước gradient** (dòng 30–34, chỉ thực hiện khi `tiles_touched[g_idx]` đúng — dòng 26):

```cpp
        Register_exp_avg = b1 * Register_exp_avg + (1.0f - b1) * Register_param_grad;
        Register_exp_avg_sq = b2 * Register_exp_avg_sq + (1.0f - b2) * Register_param_grad * Register_param_grad;
        float step = -lr * Register_exp_avg / (sqrt(Register_exp_avg_sq) + eps);
        param[p_idx] += step;
```

Tức là:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\tag{dòng 30}
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
\tag{dòng 31}
$$

$$
\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
\tag{dòng 32, 34}
$$

**Khác biệt quan trọng so với Adam chuẩn:**

1. **Không có bias-correction** ($\hat m_t, \hat v_t$ không xuất hiện trong code). Kernel dùng thẳng $m_t, v_t$ chưa hiệu chỉnh thay vì $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$. Điều này hợp lý về mặt thực nghiệm vì 3DGS thường chạy hàng chục nghìn bước nên $\beta_1^t,\beta_2^t \to 0$ rất nhanh, sai số do bỏ hiệu chỉnh chỉ đáng kể ở vài bước đầu (xem ví dụ số ở §5).
2. **Mask "visible-only" qua `tiles_touched`** (tham số `const bool* tiles_touched`, dòng 15; điều kiện `if (tiles_touched[g_idx])`, dòng 26): mỗi Gaussian $i$ chỉ được cập nhật $m,v,\theta$ nếu `tiles_touched[g_idx]` là `true` — nghĩa là Gaussian đó có phủ ít nhất 1 tile màn hình (visible) ở lượt render hiện tại. Nếu không visible, **toàn bộ 3 đại lượng $\theta,m,v$ được giữ nguyên, không bị suy giảm hay reset** — mask được tra theo chỉ số Gaussian $g_{idx}$ (không theo $p_{idx}$), nên toàn bộ $M$ thành phần của một Gaussian cùng được cập nhật hoặc cùng bị bỏ qua. Đây chính là ý nghĩa "sparse Adam".

$$
\big(\theta_i, m_i, v_i\big)_t =
\begin{cases}
\Big(\theta_{i,t-1} - \eta\dfrac{m_{i,t}}{\sqrt{v_{i,t}}+\epsilon},\ m_{i,t},\ v_{i,t}\Big) & \text{nếu } \text{tiles\_touched}_i = \text{true} \\[2mm]
\big(\theta_{i,t-1},\ m_{i,t-1},\ v_{i,t-1}\big) & \text{nếu } \text{tiles\_touched}_i = \text{false}
\end{cases}
$$

### 2.2. Lưới chỉ số (N, M) và lệnh gọi kernel (`ADAM::adamUpdate`, `adam.cu`, dòng 40–67)

Trích nguyên văn (dòng 53–66):

```cpp
    const uint32_t cnt = N * M;
    adamUpdateCUDA<<<(cnt + 255) / 256, 256>>> (
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        tiles_touched,
        lr,
        b1,
        b2,
        eps,
        N,
        M
    );
```

Kernel chạy trên lưới phẳng kích thước $N\times M$ với $N$ = số Gaussian, $M$ = số phần tử trên mỗi Gaussian của tensor tham số (ví dụ `_xyz` có $M=3$, `_scaling` có $M=3$, `_rotation` có $M=4$, `_opacity` có $M=1$):

$$
\text{cnt}=N\cdot M,\qquad \text{grid} = \left\lceil \frac{N\cdot M}{256} \right\rceil,\qquad \text{block} = 256
$$

### 2.3. Khai báo chữ ký hàm (`adam.h`, dòng 12–23)

```cpp
void adamUpdate(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    const bool* tiles_touched,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M);
```

File `adam.h` chỉ khai báo chữ ký hàm trong namespace `ADAM` (không chứa logic/công thức); toàn bộ công thức nằm trong phần định nghĩa ở `adam.cu` đã trích ở trên.

## 3. Phía Python (`SparseGaussianAdam.step`, trong `diff_gaussian_rasterization_structgs/__init__.py`)

```python
M = param.numel() // N
_C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)
```

Xác nhận: $\beta_1=0.9,\ \beta_2=0.999$ được **hard-code** khi gọi (không đọc từ `group["betas"]` của `torch.optim.Adam`), và `visibility` chính là mảng mask truyền vào làm `tiles_touched`. `state['step']` được khởi tạo (`state['step'] = torch.tensor(0.0, ...)`) nhưng **không được tăng lên hay dùng ở đâu khác** trong `step()` — củng cố việc bias-correction bị bỏ qua hoàn toàn trong biến thể sparse Adam này. Giá trị $\epsilon$ thật dùng trong SADGS là $10^{-8}$ mặc định (`adam_eps = 10**(-training_args.adam_eps_order)`, `adam_eps_order=8` theo `arguments/__init__.py`), có thể đổi thành $10^{-10}$ qua cờ CLI `--adam_eps_order 10`.

---

## 4. Kiến thức nền tảng

**SGD → momentum → Adam.** SGD: $\theta_t=\theta_{t-1}-\eta g_t$, cùng tốc độ học cho mọi chiều. Momentum làm mượt hướng đi bằng EMA của gradient: $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$. Adam (Kingma & Ba, 2015) thêm EMA bậc 2 $v_t$ của $g_t^2$ để chuẩn hoá tốc độ học theo từng tham số (adaptive learning rate): tham số có gradient lớn/biến động mạnh bị chia cho $\sqrt{v_t}$ lớn ⇒ bước nhỏ lại; tham số có gradient nhỏ/ổn định được bước lớn hơn tương đối.

**Bias correction.** Vì $m_0=v_0=0$, EMA ở bước đầu bị thiên lệch về 0. Kingma & Ba chia cho $(1-\beta^t)$ để hiệu chỉnh: $\hat m_t=m_t/(1-\beta_1^t),\ \hat v_t=v_t/(1-\beta_2^t)$. Khi $t\to\infty$, hiệu chỉnh mất tác dụng ($\beta^t\to0$) — chỉ quan trọng ở các bước đầu (hoặc sau khi EMA "khởi động lại" từ 0, ví dụ Gaussian mới sinh ra bởi densification).

**$\epsilon$.** Tránh chia cho 0 khi $v_t\approx0$.

**Sparsification theo visibility (đặc trưng 3DGS).** Số Gaussian $N$ rất lớn nhưng mỗi view/batch chỉ một tập con "nhìn thấy" có gradient có ý nghĩa. Dense Adam (cập nhật mọi $N$ mỗi bước) lãng phí băng thông bộ nhớ GPU. "Sparse Adam" giới hạn đọc/ghi chỉ ở tập con visible, giữ nguyên tuyệt đối state phần còn lại.

---

## 5. Kiểm chứng tính đúng sai

Đối chiếu với Adam gốc (Kingma & Ba, ICLR 2015, Algorithm 1):

| Thành phần | Paper gốc | Code (`adam.cu` dòng 30–34) | Khớp? |
|---|---|---|---|
| Cập nhật $m_t$ | $\beta_1 m_{t-1}+(1-\beta_1)g_t$ | `b1*exp_avg + (1-b1)*grad` | Khớp |
| Cập nhật $v_t$ | $\beta_2 v_{t-1}+(1-\beta_2)g_t^2$ | `b2*exp_avg_sq + (1-b2)*grad*grad` | Khớp |
| Bias correction | Có, chia cho $1-\beta^t$ | Không có — dùng thẳng $m_t,v_t$ | Khác biệt chính, có chủ đích |
| Vị trí $\epsilon$ | Cộng sau $\sqrt{\hat v_t}$ | `sqrt(...) + eps`, cùng vị trí | Khớp |
| Dấu bước cập nhật | $\theta_t=\theta_{t-1}-\eta(\cdot)$ | `step = -lr*(...)`, `param += step` | Khớp |
| $\beta_1,\beta_2$ | Khuyến nghị $0.9,\,0.999$ | hard-code $0.9,\,0.999$ ở lệnh gọi Python | Khớp |
| Cập nhật dày đặc mọi $\theta$ | Có | Không — chỉ Gaussian có `tiles_touched`, phần còn lại giữ nguyên | Mở rộng có chủ đích, không phải lỗi |

**Kết luận:** công thức $m_t,v_t,\theta_t$ trong `adam.cu` khớp chính xác phần "không hiệu chỉnh" của Adam gốc. Khác biệt duy nhất và có chủ đích là bỏ hoàn toàn bias-correction — không phải bug, đây là biến thể "SparseAdam" đã biết (Taming 3DGS / gsplat), hợp lý vì 3DGS huấn luyện hàng chục nghìn bước ($1-\beta^t\to1$) và mỗi Gaussian có thể sinh ra giữa chừng qua densification nên "bước $t$ toàn cục" không khớp "số lần cập nhật cục bộ" của từng Gaussian.

**So với bản nội dung cũ của chính file này:** không phát hiện sai sót toán học — công thức EMA, việc bỏ bias-correction, cơ chế mask theo `tiles_touched`, và $\beta_1=0.9,\beta_2=0.999$ hard-code đều đã đúng trong bản trước; lần cập nhật này chỉ bổ sung trích dẫn nguyên văn code kèm số dòng chính xác trước mỗi công thức, theo đúng phong cách chuẩn ở `MATH/submodules/diff-gaussian-rasterization_structgs/adam_and_bindings.md`.

---

## 6. Ví dụ số

Tham số $\theta_0=1.0$. Hai bước liên tiếp, Gaussian này **visible cả hai lần** (`tiles_touched=true`), $g_1=0.5,\ g_2=-0.2$. Dùng giá trị mặc định thật của SADGS: $\beta_1=0.9,\ \beta_2=0.999,\ \eta=0.01,\ \epsilon=10^{-8}$. Khởi tạo $m_0=v_0=0$.

### Bước 1 ($g_1=0.5$)

$$
m_1=0.9\times0+0.1\times0.5=0.05,\qquad v_1=0.999\times0+0.001\times0.5^2=0.00025
$$

$$
\sqrt{v_1}=0.0158114,\qquad \Delta\theta_1=-0.01\times\frac{0.05}{0.0158114+10^{-8}}=-0.0316228
$$

$$
\theta_1=1.0-0.0316228=0.9683772
$$

### Bước 2 ($g_2=-0.2$)

$$
m_2=0.9\times0.05+0.1\times(-0.2)=0.025,\qquad v_2=0.999\times0.00025+0.001\times0.04=0.00028975
$$

$$
\sqrt{v_2}=0.0170220,\qquad \Delta\theta_2=-0.01\times\frac{0.025}{0.0170220+10^{-8}}=-0.0146842
$$

$$
\theta_2=0.9683772-0.0146842=0.9536930
$$

**Kết quả theo code:** $\theta_0=1.0\to\theta_1=0.9683772\to\theta_2=0.9536930$, khớp chính xác với công thức trích từ `adam.cu` dòng 30–34.

### Đối chứng: cùng $m_t,v_t$ nhưng CÓ bias correction (Adam gốc)

$$
\hat m_1=\frac{0.05}{1-0.9}=0.5,\qquad \hat v_1=\frac{0.00025}{1-0.999}=0.25,\qquad\sqrt{\hat v_1}=0.5
$$

$$
\Delta\theta_1^{\text{std}}=-0.01\times\frac{0.5}{0.5+10^{-8}}\approx-0.01,\qquad \theta_1^{\text{std}}=0.99
$$

$$
\hat m_2=\frac{0.025}{1-0.81}=0.131579,\qquad \hat v_2=\frac{0.00028975}{1-0.998001}=0.144947,\qquad\sqrt{\hat v_2}=0.380719
$$

$$
\Delta\theta_2^{\text{std}}=-0.01\times\frac{0.131579}{0.380719+10^{-8}}=-0.0034560,\qquad \theta_2^{\text{std}}=0.9865440
$$

**So sánh:** $\theta_2$ theo code $=0.9536930$ so với $\theta_2^{\text{std}}$ (có bias-correction) $=0.9865440$ — lệch khoảng $0.033$ ở bước thứ 2, xác nhận việc bỏ bias-correction có tác động rõ rệt ở các bước đầu.

### Minh hoạ masked update (Gaussian không visible ở bước 2)

Nếu Gaussian trên **không visible** ở bước 2 (`tiles_touched[g_idx]=false`), nhánh `if` (dòng 26–37) bị bỏ qua hoàn toàn:

$$
\theta_2=\theta_1=0.9683772,\qquad m_2=m_1=0.05,\qquad v_2=v_1=0.00025
$$

tức đứng yên tuyệt đối — không suy giảm theo $\beta_1,\beta_2$ như thể "gradient bằng 0".
