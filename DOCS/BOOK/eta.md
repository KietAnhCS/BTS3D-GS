# Công thức toán học của `update_freq_stats_online`

## Ký hiệu

- $V$: tập Gaussian nhìn thấy trong view hiện tại. $U\subseteq V$: tập "active & visible".
- Ảnh view kích thước $H\times W$. $\mathcal{S}(\cdot)$: structure tensor map 3 kênh $(S_{xx},S_{xy},S_{yy})$ của view, lấy từ cache. Nó có thể là $T$ của multiscale ở các câu trước.
- Mỗi hàng của `cov2D` gồm $(\Sigma_{xx},\Sigma_{xy},\Sigma_{yy},\mu_x,\mu_y,d,T^{\max})$.

---

## Bước 1: Chọn Gaussian nhìn thấy

$$
V=\{\,i:\ \text{visibility}_i=\text{True}\,\}
$$

Nếu $V=\emptyset$ thì thoát.

## Bước 2: Trích dữ liệu từ cov2D

Với $i\in V$:

$$
\boldsymbol\mu_i=(\mu_{x,i},\mu_{y,i}),\qquad d_i,\qquad T_i^{\max},\qquad
\Sigma_i=\begin{pmatrix}\Sigma_{xx,i}&\Sigma_{xy,i}\\ \Sigma_{xy,i}&\Sigma_{yy,i}\end{pmatrix}
$$

## Bước 3: Mask active

Với $\alpha_i$ là opacity, $\tau_T=0$ (`transmittance_threshold`), $\tau_\alpha=0.05$ (`opacity_threshold`):

$$
A_i=\mathbb{1}[T_i^{\max}>\tau_T]\ \wedge\ \mathbb{1}[\alpha_i>\tau_\alpha]
$$

Nếu có `viewspace_point_tensor.grad` và `grad_threshold` $=\tau_g$, thêm điều kiện gradient:

$$
g_i=\Big\|\frac{\partial\mathcal L}{\partial\boldsymbol\mu_i}\Big\|_2,\qquad
A_i\leftarrow A_i\ \wedge\ \mathbb{1}[g_i>\tau_g]
$$

Phần lọc theo `densify_count` đã bị comment nên không tính. Tập cuối cùng:

$$
U=\{\,i\in V:\ A_i=1\,\}
$$

Trọng số transmittance (thực tế đang bằng hằng số):

$$
\omega_i=1\quad\forall i\in U
$$

---

## Bước 4: Lấy mẫu jitter (stochastic sampling)

Với $i\in U$, đặt $a=\Sigma_{xx},\ b=\Sigma_{xy},\ c=\Sigma_{yy}$. Cholesky $\Sigma=LL^\top$ với

$$
L_{11}=\sqrt{\max(a,10^{-6})},\qquad
L_{21}=\frac{b}{L_{11}},\qquad
L_{22}=\sqrt{\max(c-L_{21}^2,\,10^{-6})}
$$

$$
L=\begin{pmatrix}L_{11}&0\\ L_{21}&L_{22}\end{pmatrix}
$$

Lấy nhiễu chuẩn $\boldsymbol\epsilon=(\epsilon_1,\epsilon_2)^\top\sim\mathcal N(0,I_2)$:

$$
\boldsymbol\delta_i=L\boldsymbol\epsilon=
\begin{pmatrix}L_{11}\epsilon_1\\ L_{21}\epsilon_1+L_{22}\epsilon_2\end{pmatrix}
\ \sim\ \mathcal N(\mathbf 0,\Sigma_i)
$$

Điểm lấy mẫu trong ảnh (mẫu ngẫu nhiên theo chính phân bố Gaussian 2D của splat):

$$
\mathbf p_i=\boldsymbol\mu_i+\boldsymbol\delta_i=(p_x,p_y)
$$

## Bước 5: Lấy mẫu structure tensor

Chuẩn hóa toạ độ về $[-1,1]$:

$$
g_x=\frac{2p_x}{W-1}-1,\qquad g_y=\frac{2p_y}{H-1}-1
$$

Với `align_corners=True`, phép này đảo ngược đúng về toạ độ pixel $p_x,p_y$. `grid_sample` là nội suy song tuyến tính, và ngoài biên thì padding bằng 0 (mặc định):

$$
\mathbf S_i=\big(S_{xx},S_{xy},S_{yy}\big)_i=\mathcal S(\mathbf p_i)
=\sum_{(m,n)\in\{0,1\}^2}\ w_{mn}\ \mathcal S\big(\lfloor p_x\rfloor+m,\ \lfloor p_y\rfloor+n\big)
$$

trong đó $w_{mn}$ là các trọng số song tuyến. Ma trận tương ứng:

$$
S_i=\begin{pmatrix}S_{xx}&S_{xy}\\ S_{xy}&S_{yy}\end{pmatrix}_i
$$

## Bước 6: Ba trục Gaussian chiếu lên ảnh

Đây là code thật của `compute_projected_axes_subset` (nhận `means2D`, `depths`, `scales`, `rotations` đã tính sẵn cho tập con $U$):

```python
def compute_projected_axes_subset(means2D, depths, scales, rotations, viewpoint_camera):
    with torch.no_grad():
        fx = viewpoint_camera.focal_x
        fy = viewpoint_camera.focal_y
        W = viewpoint_camera.image_width
        H = viewpoint_camera.image_height

        vec_x = (means2D[:, 0] - W * 0.5) * (depths / fx)
        vec_y = (means2D[:, 1] - H * 0.5) * (depths / fy)

        inv_z = 1.0 / (depths + 1e-7)
        inv_z2 = inv_z * inv_z

        J_00 = fx * inv_z
        J_02 = -fx * vec_x * inv_z2
        J_11 = fy * inv_z
        J_12 = -fy * vec_y * inv_z2

        W_view = viewpoint_camera.world_view_transform.transpose(0, 1)
        R_view = W_view[:3, :3]

        R_local = build_rotation(rotations)
        R_view_batch = R_view.unsqueeze(0).expand(scales.shape[0], -1, -1)
        R_total = torch.bmm(R_view_batch, R_local)

        axes_cam = R_total * scales.unsqueeze(1)

        ax_x = axes_cam[:, 0, :]
        ax_y = axes_cam[:, 1, :]
        ax_z = axes_cam[:, 2, :]

        u_vec = J_00.unsqueeze(1) * ax_x + J_02.unsqueeze(1) * ax_z
        v_vec = J_11.unsqueeze(1) * ax_y + J_12.unsqueeze(1) * ax_z

        return torch.stack([u_vec, v_vec], dim=2)
```

Diễn giải theo từng bước toán:

**6.1. Toạ độ camera-space của tâm $i$ (back-project từ pixel).** Với $(m_{x,i},m_{y,i})=\text{means2D}_i$ (toạ độ pixel, đã có sẵn từ rasterizer) và độ sâu $d_i=\text{depths}_i$:

$$
x_i=(m_{x,i}-\tfrac{W}{2})\cdot\frac{d_i}{f_x},\qquad
y_i=(m_{y,i}-\tfrac{H}{2})\cdot\frac{d_i}{f_y}
$$

Đây là nghịch đảo phép chiếu pinhole $u=f_x x/z+c_x$ với giả định $c_x=W/2,\ c_y=H/2$.

**6.2. Jacobian phối cảnh $J_i$ tại điểm đó.** Đạo hàm của phép chiếu $(x,y,z)\mapsto(u,v)$ theo $(x,y,z)$:

$$
J_i=\begin{pmatrix}
\dfrac{f_x}{d_i} & 0 & -\dfrac{f_x x_i}{d_i^2}\\[6pt]
0 & \dfrac{f_y}{d_i} & -\dfrac{f_y y_i}{d_i^2}
\end{pmatrix}
=\begin{pmatrix}J^{00}_i&0&J^{02}_i\\0&J^{11}_i&J^{12}_i\end{pmatrix}
$$

(code cộng $10^{-7}$ vào $d_i$ ở mẫu số để tránh chia 0).

**6.3. Xoay và co giãn 3 trục trong camera-space.** Gọi $R_{\text{view}}$ là khối quay $3\times3$ của `world_view_transform` (world→view), $R(q_i)$ là ma trận quay từ quaternion $q_i$ (`build_rotation`). Đặt:

$$
R^{\text{tot}}_i=R_{\text{view}}\,R(q_i)
$$

3 trục ellipsoid trong world-local là $s_{i,k}\mathbf e_k$ ($k=1,2,3$, $\mathbf e_k$ là vector đơn vị trục toạ độ, $s_{i,k}$ là scale tương ứng). Trong camera-space:

$$
\mathbf a^{\text{cam}}_{i,k}=R^{\text{tot}}_i\,(s_{i,k}\mathbf e_k)=s_{i,k}\,R^{\text{tot}}_{i,:,k}
$$

tức cột thứ $k$ của $R^{\text{tot}}_i$ nhân với scale $s_{i,k}$ (đúng với dòng `axes_cam = R_total * scales.unsqueeze(1)` — nhân element-wise theo cột).

**6.4. Chiếu lên ảnh qua Jacobian.** Với $\mathbf a^{\text{cam}}_{i,k}=(a^x_{i,k},a^y_{i,k},a^z_{i,k})$:

$$
\mathbf a_{i,k}=\begin{pmatrix}u_{i,k}\\ v_{i,k}\end{pmatrix}
=J_i\,\mathbf a^{\text{cam}}_{i,k}
=\begin{pmatrix}J^{00}_i a^x_{i,k}+J^{02}_i a^z_{i,k}\\ J^{11}_i a^y_{i,k}+J^{12}_i a^z_{i,k}\end{pmatrix},\qquad k=1,2,3
$$

Kết quả là 3 vector 2D $(u_{i,k},v_{i,k})$ — độ dài và hướng của 3 trục ellipsoid sau khi chiếu lên mặt phẳng ảnh, dùng cho $\eta_{i,k}$ ở Bước 7.

---

## Bước 7: Tính $\eta$ (hai chế độ)

### Chế độ `wavelength`

Trị riêng lớn nhất của $S$:

$$
\lambda_1=\frac{\operatorname{tr}S}{2}+\sqrt{\max\!\Big(\big(\tfrac{\operatorname{tr}S}{2}\big)^2-\det S,\ 0\Big)}
=\frac{S_{xx}+S_{yy}}{2}+\sqrt{\Big(\frac{S_{xx}-S_{yy}}{2}\Big)^2+S_{xy}^2}
$$

Bước sóng nhỏ nhất (đơn vị pixel) và độ dài trục chiếu:

$$
\ell_i=\frac{1}{\sqrt{\lambda_{1,i}}+10^{-5}},\qquad
r_{i,k}=\sqrt{u_{i,k}^2+v_{i,k}^2+10^{-8}}
$$

$$
\boxed{\ \eta_{i,k}=\frac{r_{i,k}}{\ell_i}=r_{i,k}\big(\sqrt{\lambda_{1,i}}+10^{-5}\big)\ }
$$

Đây là tỉ lệ (kích thước Gaussian) / (bước sóng texture). Nếu $S$ là $T$ multiscale thì $\sqrt{\lambda_1}\approx f$ (tần số chủ đạo), nên $\eta\approx r\cdot f$. $\eta>1$ nghĩa là Gaussian lớn hơn một chu kỳ texture, tức aliasing.

### Chế độ `projection`

$$
\boxed{\ \eta_{i,k}=\sqrt{\mathbf a_{i,k}^\top S_i\,\mathbf a_{i,k}}=\sqrt{S_{xx}u^2+2S_{xy}uv+S_{yy}v^2}\ }
$$

Khai triển theo eigen-decomposition $S=\lambda_1\mathbf e_1\mathbf e_1^\top+\lambda_2\mathbf e_2\mathbf e_2^\top$:

$$
\eta_{i,k}=\sqrt{\lambda_1(\mathbf a\cdot\mathbf e_1)^2+\lambda_2(\mathbf a\cdot\mathbf e_2)^2}
$$

Chế độ này có tính hướng: trục nằm dọc theo hướng gradient mạnh thì $\eta$ lớn, trục song song với cạnh thì $\eta\approx0$.

### Nhân trọng số

$$
\eta_{i,k}\leftarrow\omega_i\,\eta_{i,k},\qquad
\boldsymbol\eta_i=(\eta_{i,1},\eta_{i,2},\eta_{i,3}),\qquad
\eta_i^{\text{tot}}=\sum_{k=1}^{3}\eta_{i,k}
$$

---

## Bước 8: Cập nhật bộ tích lũy (với $i\in U$)

Tích lũy trên nhiều view:

$$
E_i\mathrel{+}=\eta_i^{\text{tot}},\qquad
n_i\mathrel{+}=1,\qquad
\Omega_i\mathrel{+}=\omega_i
$$

Max theo từng trục (elementwise):

$$
\mathbf M_i\leftarrow\max\big(\mathbf M_i,\ \boldsymbol\eta_i\big)\quad\Rightarrow\quad M_{i,k}\leftarrow\max(M_{i,k},\eta_{i,k})
$$

## Bước 9: Phân loại nhất quán đa view

Với $\tau_{\text{high}}=1.0$ và $\tau_{\text{low}}=0.1$, đặt $\bar\eta_i=\max_k\eta_{i,k}$:

$$
\text{high}:\ \bar\eta_i>\tau_{\text{high}},\qquad
\text{low}:\ \bar\eta_i\le\tau_{\text{low}},\qquad
\text{mid}:\ \tau_{\text{low}}<\bar\eta_i\le\tau_{\text{high}}
$$

Cập nhật số đếm:

$$
N_i^{c}\mathrel{+}=\mathbb{1}[\,i\in c\,],\qquad c\in\{\text{high},\text{mid},\text{low}\}
$$

Cập nhật tổng vector 3 kênh (chỉ cho high và mid, không có cho low):

$$
\mathbf H_i\mathrel{+}=\boldsymbol\eta_i\,\mathbb{1}[i\in\text{high}],\qquad
\mathbf D_i\mathrel{+}=\boldsymbol\eta_i\,\mathbb{1}[i\in\text{mid}]
$$

---

## Tóm tắt luồng

$$
\text{cov2D}\to\underbrace{U}_{\text{lọc}}\to\underbrace{\mathbf p=\boldsymbol\mu+L\boldsymbol\epsilon}_{\text{jitter}}\to\underbrace{S=\mathcal S(\mathbf p)}_{\text{sample}}\to\underbrace{\eta_k}_{\text{axis vs. tensor}}\to\{E,n,\Omega,\mathbf M,N^c,\mathbf H,\mathbf D\}
$$

## Nhận xét

- $\omega_i\equiv1$ nên trọng số transmittance hiện không có tác dụng (dòng `torch.ones_like` thay cho $T^{\max}$).
- $T^{\max}$ chỉ dùng để lọc (ngưỡng $\tau_T=0$), nên với ngưỡng 0 nó gần như chỉ loại các Gaussian có $T^{\max}\le0$.
- Vì jitter là ngẫu nhiên, $\eta$ của cùng một Gaussian thay đổi giữa các lần chạy. Giá trị trung bình qua nhiều view (chia cho $n_i$) mới ổn định, chẳng hạn $\bar E_i=E_i/n_i$.
- Chế độ `wavelength` có $\eta$ không có hướng (chỉ dùng $\lambda_1$), còn `projection` phân biệt trục theo hướng cạnh. Đây là lý do `max_eta_3ch` có ý nghĩa khác nhau giữa hai chế độ.
- Nếu $S$ được nội suy ra ngoài biên ảnh (jitter vượt $[0,W-1]\times[0,H-1]$) thì $S\to$ giá trị pha trộn với 0, làm $\eta$ bị kéo thấp. Đoạn "mirror jitter" đã bị comment nên chưa xử lý trường hợp này.

# Ví dụ số cho `update_freq_stats_online`

Tôi chạy đúng các bước ở trên với input tự đặt. Có ba điểm cần nói trước:

- `compute_projected_axes_subset` không có trong code bạn gửi, nên tôi **cho sẵn trực tiếp** `axes_2d` (đơn vị pixel).
- $\epsilon_1,\epsilon_2$ của jitter được **cố định** để bạn kiểm tra lại được. Chạy thật thì chúng ngẫu nhiên.
- `st_map` là output của multiscale v2 ($L=3,\ \sigma_0=1,\ s=1.5,\ p=3$) trên ảnh 64×64 có **cạnh chéo** $x+y=64$ (0.2 → 0.8). Cạnh chéo cho $S_{xy}\ne0$, nên hai chế độ `wavelength` và `projection` cho kết quả khác nhau.

## Input

Ngưỡng: $\tau_T=0,\ \tau_\alpha=0.05,\ \tau_g=2\times10^{-4}$. Ảnh $W=H=64$.

| Gaussian | $\Sigma_{xx},\Sigma_{xy},\Sigma_{yy}$ | $\mu_x,\mu_y$ | $T^{\max}$ | opacity | $\lVert\nabla\rVert$ | visible |
|---|---|---|---|---|---|---|
| G0 | 4, 0, 4 | 30, 32 | 0.90 | 0.60 | 0.00100 | ✓ |
| G1 | 9, 2, 6 | 48, 20 | 0.70 | 0.80 | 0.00050 | ✓ |
| G2 | 4, 0, 4 | 10, 10 | 0.95 | **0.02** | 0.00127 | ✓ |
| G3 | 4, 1, 5 | 20, 44 | 0.80 | 0.50 | **0.00005** | ✓ |
| G4 | 4, 0, 4 | 5, 5 | 0.50 | 0.70 | 0.00141 | **✗** |

Trục chiếu cho sẵn $(u,v)$ cho ba trục:

- G0: $(3,0),\ (0,3),\ (0.5,0.5)$
- G1: $(4,1),\ (-1,4),\ (0.3,0.3)$

Nhiễu cố định: G0 có $(\epsilon_1,\epsilon_2)=(0.5,-1.0)$, G1 có $(-0.8,0.6)$.

## Bước 1: Gaussian nhìn thấy

$V=\{0,1,2,3\}$ (G4 bị loại vì không visible).

## Bước 3: Mask active

| | $T^{\max}>0$ | $\alpha>0.05$ | $g>2\times10^{-4}$ | $A_i$ |
|---|---|---|---|---|
| G0 | ✓ | ✓ | ✓ | **1** |
| G1 | ✓ | ✓ | ✓ | **1** |
| G2 | ✓ | ✗ | ✓ | 0 |
| G3 | ✓ | ✓ | ✗ | 0 |

$U=\{0,1\}$, $\omega_i=1$.

## Bước 4: Jitter

**G0.** $L_{11}=\sqrt4=2,\ L_{21}=0/2=0,\ L_{22}=\sqrt{4-0}=2$

$$\boldsymbol\delta=(2\cdot0.5,\ 0\cdot0.5+2\cdot(-1))=(1,\,-2),\qquad \mathbf p=(30+1,\ 32-2)=(31,\,30)$$

**G1.** $L_{11}=3,\ L_{21}=2/3=0.6667,\ L_{22}=\sqrt{6-0.4444}=2.3570$

$$\boldsymbol\delta=(3\cdot(-0.8),\ 0.6667\cdot(-0.8)+2.357\cdot0.6)=(-2.4,\,0.8809)$$

$$\mathbf p=(45.6,\ 20.8809)$$

## Bước 5: Lấy mẫu structure tensor

Toạ độ chuẩn hóa:

- G0: $(g_x,g_y)=(\tfrac{2\cdot31}{63}-1,\ \tfrac{2\cdot30}{63}-1)=(-0.0159,\,-0.0476)$
- G1: $(0.4476,\ -0.3371)$

Nội suy song tuyến từ `st_map`:

| | $S_{xx}$ | $S_{xy}$ | $S_{yy}$ | $\lambda_1$ | $\sqrt{\lambda_1}$ |
|---|---|---|---|---|---|
| G0 | 0.32464 | 0.32464 | 0.32464 | 0.64929 | 0.80578 |
| G1 | 0.23716 | 0.23717 | 0.23719 | 0.47435 | 0.68873 |

$S_{xx}\approx S_{xy}\approx S_{yy}$ nghĩa là tensor hạng 1, $S\approx\lambda_1\mathbf e\mathbf e^\top$ với $\mathbf e=(1,1)/\sqrt2$. Đúng với cạnh chéo: gradient hướng theo $(1,1)$. Trace $=0.649$ và $0.474$ nằm trong khoảng $[f_2^2,f_0^2]=[0.20,\,1]$.

## Bước 7a: Chế độ `wavelength`

$$\ell=\frac{1}{\sqrt{\lambda_1}+10^{-5}},\qquad r_k=\sqrt{u_k^2+v_k^2+10^{-8}},\qquad \eta_k=\frac{r_k}{\ell}$$

**G0.** $\ell=1/0.80579=1.241$ px, $r=(3,\ 3,\ 0.7071)$:

$$\boldsymbol\eta=(2.4174,\ 2.4174,\ 0.5698),\quad \textstyle\sum=5.4046,\quad \bar\eta=2.4174$$

**G1.** $\ell=1.452$ px, $r=(4.1231,\ 4.1231,\ 0.4243)$:

$$\boldsymbol\eta=(2.8398,\ 2.8398,\ 0.2922),\quad \textstyle\sum=5.9717,\quad \bar\eta=2.8398$$

Hai trục đầu có cùng $\eta$ vì chế độ này chỉ dùng độ dài trục $r$, không xét hướng.

## Bước 7b: Chế độ `projection`

$$\eta_k=\sqrt{S_{xx}u^2+2S_{xy}uv+S_{yy}v^2}=\sqrt{\lambda_1}\,\big|\mathbf a_k\cdot\mathbf e\big|,\qquad \mathbf e=\tfrac{(1,1)}{\sqrt2}$$

**G0.**

| Trục | $(u,v)$ | $\lvert\mathbf a\cdot\mathbf e\rvert$ | $\eta$ |
|---|---|---|---|
| 1 | (3, 0) | $3/\sqrt2=2.1213$ | 0.80578 × 2.1213 = **1.7093** |
| 2 | (0, 3) | 2.1213 | **1.7093** |
| 3 | (0.5, 0.5) | 0.7071 | **0.5698** |

$$\textstyle\sum=3.9884,\qquad \bar\eta=1.7093$$

**G1.**

| Trục | $(u,v)$ | $\lvert\mathbf a\cdot\mathbf e\rvert$ | $\eta$ |
|---|---|---|---|
| 1 | (4, 1) | $5/\sqrt2=3.5355$ | 0.68873 × 3.5355 = **2.4350** |
| 2 | (−1, 4) | $3/\sqrt2=2.1213$ | **1.4611** |
| 3 | (0.3, 0.3) | 0.4243 | **0.2922** |

$$\textstyle\sum=4.1883,\qquad \bar\eta=2.4350$$

Ở G1, trục 1 và trục 2 khác nhau vì trục 1 lệch nhiều hơn về hướng gradient. Đây là điểm `projection` có mà `wavelength` không có.

## Bước 8–9: Cập nhật bộ tích lũy (giả sử bắt đầu từ 0, sau đúng một view)

Với $\omega_i=1$: $E_i\mathrel{+}=\eta^{\text{tot}}_i,\ n_i=1,\ \Omega_i=1$, và $\mathbf M_i=\boldsymbol\eta_i$ (vì $\max(0,\eta)=\eta$).

| | Chế độ | `accum_eta` $E$ | `max_eta_3ch` $\mathbf M$ | $\bar\eta$ | Lớp |
|---|---|---|---|---|---|
| G0 | wavelength | 5.4046 | (2.4174, 2.4174, 0.5698) | 2.4174 | **high** |
| G1 | wavelength | 5.9717 | (2.8398, 2.8398, 0.2922) | 2.8398 | **high** |
| G0 | projection | 3.9884 | (1.7093, 1.7093, 0.5698) | 1.7093 | **high** |
| G1 | projection | 4.1883 | (2.4350, 1.4611, 0.2922) | 2.4350 | **high** |

Vì cả hai đều thuộc lớp **high** ($\bar\eta>1$):

- `eta_high_count` += 1 cho G0 và G1.
- `eta_high_sum_3ch` += $\boldsymbol\eta$ tương ứng.
- `eta_mid_*` và `eta_low_*` không đổi.
- G2, G3, G4 không được cập nhật gì.

## Cách đọc

- Cả hai Gaussian có $\eta>1$: chiều rộng của chúng (khoảng 3–4 px) lớn hơn bước sóng texture cục bộ (1.2–1.5 px). Chúng bị coi là quá to so với chi tiết ảnh và là ứng viên để split.
- `wavelength` cho $\eta$ lớn hơn `projection` ở cùng Gaussian (2.42 vs 1.71 ở G0). Nó bỏ qua hướng, coi mọi trục đều "cắt ngang" texture, và có xu hướng ước lượng cao hơn.
- `projection` chỉ tính phần trục cắt qua hướng gradient. Trục song song cạnh (vuông góc $\mathbf e$) sẽ cho $\eta\approx0$. Trong ví dụ này không trục nào vuông góc chính xác với $\mathbf e$ nên chưa thấy trường hợp đó.
- Nếu đổi $\epsilon$ khác, $\mathbf p$ dịch đi và $S$ nội suy ra giá trị khác, kéo theo $\eta$ đổi. Vì vậy trong thực tế phải chia cho $n_i$ để lấy trung bình qua nhiều view.

Nếu bạn muốn, tôi có thể chạy tiếp nhiều view liên tiếp (ví dụ 5 view với $\epsilon$ ngẫu nhiên) để xem `accum_eta / accum_view_count` hội tụ thế nào, hoặc thử một Gaussian có trục song song với cạnh để thấy `projection` cho $\eta\approx0$.