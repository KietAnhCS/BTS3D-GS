# Từ điển Cú pháp PyTorch ↔ Công thức Toán học

Mục tiêu file này: với người **chưa quen cú pháp PyTorch**, mỗi dòng code trong
`scene/gaussian_model.py` (và các file liên quan) đều có thể tra ngược ra đúng
khái niệm toán học tương ứng. File `gaussian_model.md` đã giải thích công thức
theo **từng hàm**; file này bổ sung theo chiều ngược lại — tra theo **từng mẫu
cú pháp** (syntax pattern) xuất hiện lặp đi lặp lại khắp codebase.

Quy ước ký hiệu dùng xuyên suốt:
- Chữ thường in đậm $\mathbf{x}$: vector/tensor 1 điểm (ví dụ 1 Gaussian).
- Chữ hoa $A, R, S, \Sigma$: ma trận.
- Chỉ số $i$ chạy trên **số điểm** $N$ (số Gaussian), trừ khi nói khác.
- "raw"/"thô" = giá trị `nn.Parameter` lưu trực tiếp, trước khi qua activation.

---

## 1. Tạo tensor (Tensor creation)

| Cú pháp | Toán học | Giải thích |
|---|---|---|
| `torch.empty(0)` | $\mathbf{x} \in \mathbb{R}^0$ (rỗng) | Tensor rỗng, dùng làm placeholder trước khi có dữ liệu thật. |
| `torch.zeros((N,))` | $\mathbf{0}_N = [0,\dots,0]^T \in \mathbb{R}^N$ | Vector không, độ dài $N$. |
| `torch.zeros((N, 3))` | $\mathbf{0}_{N\times3} \in \mathbb{R}^{N\times 3}$ | Ma trận không — dùng khởi tạo bộ tích lũy (accumulator) cho $N$ điểm, mỗi điểm 3 giá trị. |
| `torch.ones((N,)) * c` | $c\cdot\mathbf 1_N = [c,c,\dots,c]^T$ | Vector hằng số $c$, độ dài $N$. |
| `torch.ones_like(x)` | $\mathbf 1 \in \mathbb{R}^{\text{shape}(\mathbf x)}$ | Vector/ma trận toàn số 1, **cùng kích thước** với `x`. |
| `torch.zeros_like(x)` | $\mathbf 0 \in \mathbb{R}^{\text{shape}(\mathbf x)}$ | Tương tự, toàn số 0. |
| `torch.arange(n)` | $[0, 1, 2, \dots, n-1]^T$ | Dãy số nguyên liên tiếp — dùng để sinh chỉ số (index). |
| `torch.tensor(np_array)` | nhúng mảng NumPy $\mathbf a\in\mathbb R^n$ vào tensor | Không đổi giá trị, chỉ đổi kiểu dữ liệu để autograd/CUDA dùng được. |

---

## 2. Toán tử phần tử (Element-wise ops)

Tất cả các hàm dưới đều áp dụng **độc lập lên từng phần tử** của tensor —
tức nếu $\mathbf x = [x_1,\dots,x_n]^T$ thì $f(\mathbf x) = [f(x_1),\dots,f(x_n)]^T$.

| Cú pháp | Công thức (áp dụng từng phần tử) |
|---|---|
| `torch.exp(x)` / `x.exp()` | $e^{x_i}$ |
| `torch.log(x)` | $\ln(x_i)$ |
| `torch.sqrt(x)` | $\sqrt{x_i}$ |
| `torch.square(x)` | $x_i^2$ |
| `torch.abs(x)` | $\lvert x_i \rvert$ |
| `torch.sigmoid(x)` | $\sigma(x_i) = \dfrac{1}{1+e^{-x_i}}$ |
| `torch.clamp(x, min=a, max=b)` | $\max(a,\min(x_i,b))$ — ép $x_i$ nằm trong $[a,b]$ |
| `torch.clamp_min(x, a)` | $\max(x_i, a)$ |
| `a / b` (tensor chia tensor cùng shape) | $x_i / y_i$ (chia **từng phần tử**, không phải chia ma trận) |
| `a * b` (tensor nhân tensor cùng shape) | $x_i \cdot y_i$ (nhân Hadamard $\mathbf a \odot \mathbf b$, **không** phải tích vô hướng/ma trận) |
| `a + b`, `a - b` | $x_i + y_i$, $x_i - y_i$ |
| `k * x` (số nhân tensor) | $k\cdot\mathbf x = [k x_1,\dots,k x_n]^T$ — nhân vô hướng |

> **Lưu ý quan trọng**: `*` và `/` giữa hai tensor trong PyTorch **luôn là
> element-wise**, khác với ký hiệu nhân ma trận toán học $AB$. Muốn nhân ma
> trận phải dùng `@` hoặc `torch.matmul`/`torch.bmm` (xem mục 4).

---

## 3. Reduction — thu gọn chiều (sum, max, prod, norm...)

Các phép này **giảm số chiều** của tensor, tương ứng với phép tổng/tích/chuẩn
trong toán học.

| Cú pháp | Công thức | Giải thích |
|---|---|---|
| `x.sum()` | $\sum_i x_i$ | Tổng toàn bộ phần tử → một số vô hướng. |
| `x.sum(dim=1)` (với `x` shape $[N,3]$) | $\sum_{j=1}^{3} x_{ij}$ cho mỗi hàng $i$ | Tổng theo trục cột, kết quả shape $[N]$. |
| `x.prod(dim=1)` | $\prod_{j=1}^{3} x_{ij}$ | Ví dụ $\det(\text{diag}) = \prod_i s_i$ — dùng tính định thức ma trận đường chéo (mục 14 trong `gaussian_model.md`). |
| `torch.max(x, dim=1)` | $\max_j x_{ij}$ (trả cả `.values` và `.indices`) | $\arg\max$ và giá trị lớn nhất theo trục. |
| `torch.min(a, b)` (2 tensor) | $\min(x_i, y_i)$ từng phần tử | Khác với `x.min()` (thu gọn 1 tensor). |
| `torch.norm(x, dim=1)` | $\lVert \mathbf{x}_i \rVert_2 = \sqrt{\textstyle\sum_j x_{ij}^2}$ | Chuẩn Euclid ($L_2$-norm) của từng vector hàng. |
| `x.mean()` | $\dfrac{1}{n}\sum_i x_i$ | Trung bình cộng. |

---

## 4. Đại số tuyến tính — ma trận & vector

| Cú pháp | Công thức | Giải thích |
|---|---|---|
| `A @ B` hoặc `torch.matmul(A, B)` | $C = AB$, $C_{ik}=\sum_j A_{ij}B_{jk}$ | **Nhân ma trận** chuẩn (2D). |
| `torch.bmm(A, B)` | $C_p = A_p B_p\ \ \forall p=1..N$ | **Batched matmul**: nhân ma trận **song song** cho $N$ cặp ma trận cùng lúc, shape $[N,i,j]\times[N,j,k]\to[N,i,k]$. Dùng khi mỗi Gaussian có một ma trận quay $R_p$ riêng. |
| `A.transpose(1, 2)` | $A^T$ (chuyển vị 2 chiều cuối) | Với $A$ shape $[N,3,3]$ (batch $N$ ma trận $3\times3$), đổi hàng↔cột trong từng ma trận, **giữ nguyên** chiều batch (chiều `0`). |
| `x.unsqueeze(-1)` | $\mathbf x \in \mathbb R^{3} \to \mathbf x \in \mathbb R^{3\times 1}$ | Biến vector hàng thành **vector cột** (thêm 1 chiều) — cần thiết trước khi nhân ma trận `bmm`. |
| `x.squeeze(-1)` | $\mathbb R^{3\times1}\to\mathbb R^{3}$ | Ngược lại `unsqueeze` — bỏ chiều kích thước 1. |
| `x[..., None]` | giống `unsqueeze(-1)` | Thêm 1 chiều cuối cùng, dùng để broadcasting (xem mục 6). |
| `torch.diag`/ `S = diag(s)` (khái niệm, không phải literal code) | $S = \begin{bmatrix}s_x&0&0\\0&s_y&0\\0&0&s_z\end{bmatrix}$ | Ma trận đường chéo từ vector scale — xây dựng ngầm bên trong `build_scaling_rotation`. |
| `R @ eps` hay `torch.bmm(R, eps)` | $\mathbf y = R\mathbf x$ | **Phép quay** vector $\mathbf x$ bằng ma trận quay $R\in SO(3)$ ($R^TR=I,\ \det R=1$). |

---

## 5. Chuẩn hoá & activation (reparameterization)

Đây là nhóm hàm "ép" tham số thô (không giới hạn, $\in\mathbb R$) về đúng miền
vật lý hợp lệ — gọi là **reparameterization trick**.

| Cú pháp | Công thức xuôi | Công thức ngược (inverse) | Miền giá trị |
|---|---|---|---|
| `torch.exp` ↔ `torch.log` | $s=e^{x}$ | $x=\ln s$ | $\mathbb R \to (0,+\infty)$ — đảm bảo scale luôn dương |
| `torch.sigmoid` ↔ `inverse_sigmoid` | $\alpha=\sigma(x)=\frac{1}{1+e^{-x}}$ | $x=\ln\frac{\alpha}{1-\alpha}$ | $\mathbb R \to (0,1)$ — đảm bảo opacity là xác suất |
| `torch.nn.functional.normalize(q)` | $\hat{\mathbf q}=\dfrac{\mathbf q}{\lVert \mathbf q\rVert_2}$ | (không cần nghịch đảo — luôn chiếu lên mặt cầu) | $\mathbb R^4 \to S^3$ (quaternion đơn vị, đại diện phép quay 3D) |
| `torch.abs` ↔ `identity_gate` | $\alpha=\lvert x\rvert$ | $x=\alpha$ (gate đồng nhất) | $\mathbb R\to[0,+\infty)$ — biến thể opacity không bão hoà như sigmoid |

---

## 6. Broadcasting (lan truyền kích thước)

PyTorch tự động "nhân bản" một tensor nhỏ hơn để khớp shape khi tính toán —
không có công thức tường minh riêng, nhưng về mặt toán học nó **tương đương
nhân với vector $\mathbf 1$**.

| Cú pháp | Shape trước | Shape sau khi broadcast | Toán học tương đương |
|---|---|---|---|
| `opacity * coef[..., None]` | `opacity`:$[N,1]$, `coef`:$[N]\to[N,1]$ | $[N,1]$ | $\alpha_i \cdot c_i$ — nhân vô hướng riêng cho **từng điểm** $i$ |
| `xyz_cam + T[None, :]` | `T`: $[3]\to[1,3]$ | cộng vào mọi hàng của $[N,3]$ | $\mathbf x_{cam,i} + \mathbf T\ \ \forall i$ — cộng cùng 1 vector tịnh tiến cho mọi điểm |
| `x[..., np.newaxis]` (NumPy, tương tự `None`) | $[N]\to[N,1]$ | — | Thêm 1 chiều để khớp shape khi ghép cột (`concatenate`) |

---

## 7. Indexing / Masking (lọc theo điều kiện)

| Cú pháp | Công thức / khái niệm | Giải thích |
|---|---|---|
| `x[mask]` với `mask` là tensor bool | $\{x_i : m_i = \text{True}\}$ | **Lọc tập con** — chỉ giữ các phần tử có mask đúng. Số phần tử kết quả = $\sum_i m_i$. |
| `x[~mask]` | $\{x_i : m_i = \text{False}\}$ | Phần bù tập hợp (logic NOT). |
| `mask1 & mask2` hoặc `torch.logical_and(mask1, mask2)` | $m_i = m^{(1)}_i \land m^{(2)}_i$ | Giao hai tập điều kiện (AND luận lý). |
| `torch.logical_or(mask1, mask2)` | $m_i = m^{(1)}_i \lor m^{(2)}_i$ | Hợp hai tập điều kiện (OR luận lý). |
| `torch.where(cond, a, b)` | $y_i = \begin{cases}a_i & \text{nếu } cond_i\\ b_i & \text{nếu không}\end{cases}$ | Hàm chọn có điều kiện — tương đương $y = cond\cdot a + (1-cond)\cdot b$. |
| `torch.nonzero(mask).squeeze(1)` | $\{i : m_i = \text{True}\}$ | Trả về **chỉ số** (index) thay vì giá trị — tập $I\subseteq\{1,\dots,N\}$. |
| `x.isnan()` | $\{i : x_i = \text{NaN}\}$ | Mask các vị trí "không xác định" (ví dụ $0/0$ khi `denom=0`). |
| `x[mask] = v` (gán có điều kiện) | $x_i \leftarrow v\ \ \forall i \in \{i:m_i\}$ | Cập nhật tại chỗ chỉ những phần tử thoả điều kiện. |

---

## 8. Ghép / nhân bản tensor (concat, repeat, stack)

| Cú pháp | Công thức | Giải thích |
|---|---|---|
| `torch.cat((a, b), dim=0)` | $\mathbf c = [\mathbf a \,;\, \mathbf b] \in \mathbb R^{(n_a+n_b)\times\dots}$ | Nối theo chiều **hàng** (thêm điểm mới) — như "append" vào danh sách điểm. |
| `torch.cat((a, b), dim=1)` | $\mathbf c = [\mathbf a \,\Vert\, \mathbf b]$ | Nối theo chiều **cột** (ghép đặc trưng, ví dụ $F=[F_{dc}\Vert F_{rest}]$). |
| `torch.stack([a, b, c], dim=1)` | $M = \begin{bmatrix}a_1&b_1&c_1\\ \vdots&\vdots&\vdots\end{bmatrix}$ | Tạo **chiều mới** từ nhiều tensor cùng shape (khác `cat` là không tạo chiều mới). |
| `x.repeat(N, 1)` | $[\underbrace{\mathbf x;\mathbf x;\dots;\mathbf x}_{N}]$ | Lặp lại **toàn bộ khối** $N$ lần liên tiếp (nối đuôi). |
| `x.repeat_interleave(reps, dim=0)` | điểm $i$ được lặp $r_i$ lần **tại chỗ** trước khi sang điểm $i+1$ | Khác `repeat`: dùng khi mỗi điểm cha có **số con khác nhau** $N_i = k_{x,i}k_{y,i}k_{z,i}$ (xem `densify_and_split_structgs`). |

So sánh `repeat` vs `repeat_interleave` với $\mathbf x=[x_1,x_2]$, lặp mỗi phần tử 2 lần:
- `x.repeat(2)` → $[x_1,x_2,x_1,x_2]$
- `x.repeat_interleave(2)` → $[x_1,x_1,x_2,x_2]$

---

## 9. Phân bố xác suất (random sampling)

| Cú pháp | Công thức | Giải thích |
|---|---|---|
| `torch.normal(mean=mu, std=sigma)` | $\boldsymbol\epsilon \sim \mathcal N(\mu,\ \sigma^2)$ | Lấy mẫu phân phối chuẩn (Gaussian) — dùng sinh vị trí Gaussian con quanh tâm cha. |
| `torch.multinomial(w, k, replacement=False)` | lấy mẫu $k$ chỉ số **không hoàn lại**, với $P(i) = \dfrac{w_i}{\sum_j w_j}$ | Lấy mẫu có trọng số — dùng chọn ngẫu nhiên điểm để xoá theo xác suất tỉ lệ nghịch với điểm số quan trọng. |

---

## 10. Vi phân / Gradient / Optimizer

| Cú pháp | Công thức | Giải thích |
|---|---|---|
| `nn.Parameter(x.requires_grad_(True))` | đánh dấu $x$ là biến cần tính $\dfrac{\partial \mathcal L}{\partial x}$ | Biến tối ưu — PyTorch sẽ tự tính gradient qua backprop. |
| `tensor.grad` | $\nabla_x \mathcal L = \dfrac{\partial \mathcal L}{\partial x}$ | Gradient của hàm loss $\mathcal L$ theo biến `tensor`, tính bởi autograd sau `.backward()`. |
| `optimizer.zero_grad()` | $\nabla_x \mathcal L \leftarrow 0$ | Xoá gradient tích lũy từ bước trước (PyTorch cộng dồn gradient mặc định). |
| `optimizer.step()` | Adam: $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$, $\theta_t=\theta_{t-1}-\text{lr}\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ | Một bước cập nhật tham số theo quy tắc Adam. |
| `exp_avg`, `exp_avg_sq` (trong `optimizer.state`) | $m_t$ (moment bậc 1), $v_t$ (moment bậc 2, "exponential moving average") | Trạng thái nội bộ Adam — bị reset/cắt/nối mỗi khi thêm/xoá điểm Gaussian (`replace_tensor_to_optimizer`, `_prune_optimizer`, `cat_tensors_to_optimizer`). |
| `with torch.no_grad(): ...` | tính toán **ngoài** đồ thị tính đạo hàm | $\theta^{new} = f(\theta^{old})$ áp dụng trực tiếp, không lưu để backprop (dùng khi cập nhật thủ công như `expand_undersized_gs`). |

---

## 11. Ánh xạ cụ thể trong `build_covariance_from_scaling_rotation`

(Trường hợp chi tiết nhất — tổng hợp lại và mở rộng từ bảng người dùng đã có)

| Cú pháp | Công thức | Ý nghĩa |
|---|---|---|
| `scaling_modifier * scaling` | $k\cdot\mathbf s=[ks_x,ks_y,ks_z]^T$ | Co giãn đều cả 3 trục theo hệ số $k$ (`scaling_modifier`). |
| `build_scaling_rotation(s, q)` | $L = R(q)\cdot\text{diag}(\mathbf s)$ | Dựng ma trận $R\in SO(3)$ từ quaternion $q$, nhân với ma trận đường chéo scale. |
| `L.transpose(1, 2)` | $L^T$ | Chuyển vị trong không gian 2 chiều cuối (chiều 0 là batch $N$ Gaussian). |
| `L @ L.transpose(1, 2)` | $\Sigma = LL^T = RSS^TR^T$ | Ma trận hiệp phương sai $3\times3$, luôn đối xứng & nửa xác định dương. |
| `strip_symmetric(Σ)` | $\text{vech}(\Sigma)=[\Sigma_{00},\Sigma_{01},\Sigma_{02},\Sigma_{11},\Sigma_{12},\Sigma_{22}]^T$ | Chỉ lưu 6 giá trị độc lập (do $\Sigma_{ij}=\Sigma_{ji}$) thay vì 9. |

---

## 12. Bảng tra nhanh theo "hình dạng tensor" (shape) hay gặp trong file

| Shape | Ý nghĩa vật lý |
|---|---|
| `[N]` | Một đại lượng vô hướng **cho mỗi điểm** (opacity thô 1 chiều đã squeeze, denom, eta...) |
| `[N, 1]` | Giống trên nhưng giữ chiều cột (thường để broadcasting hoặc khớp PLY) |
| `[N, 3]` | Một vector 3 chiều cho mỗi điểm: toạ độ `xyz`, scale `sx,sy,sz`, hoặc `max_eta_3ch` (eta theo từng trục) |
| `[N, 4]` | Quaternion `(w,x,y,z)` cho mỗi điểm |
| `[N, 3, 3]` | Ma trận $3\times3$ cho mỗi điểm (ma trận quay $R$, hoặc hiệp phương sai $\Sigma$ trước khi `strip_symmetric`) |
| `[N, F, K]` | Hệ số cầu điều hoà (SH): $F=3$ kênh màu RGB, $K=(\ell+1)^2$ số hệ số SH tới bậc $\ell$ |

---

## Ghi chú

- Bảng này **bổ sung**, không thay thế `gaussian_model.md` — hãy đọc file đó
  trước để biết công thức theo *ngữ cảnh từng hàm*, rồi quay lại đây để tra
  theo *từng mẩu cú pháp* khi gặp dòng code lạ.
- Khi gặp một dòng code mới không có trong bảng, cách tra nhanh:
  1. Có `@` hoặc `bmm` → chắc chắn là **nhân ma trận**.
  2. Có `*` hoặc `/` giữa 2 tensor → **element-wise**, không phải ma trận.
  3. Có `[mask]` → **lọc tập con** theo điều kiện logic.
  4. Có `None`/`unsqueeze`/`[...,None]` → chỉ đổi shape để **broadcasting**,
     không đổi giá trị toán học.
