# Công thức toán học của `cuda_rasterizer/forward.cu` + `forward.h`

Tài liệu này KIỂM CHỨNG và viết lại toàn bộ cơ sở toán học của lượt **forward** (chiếu 3D→2D, tô màu SH, rasterize bằng alpha compositing) trong rasterizer CUDA của submodule `diff-gaussian-rasterization_structgs`. Nguồn đối chiếu trực tiếp với code thật:

- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/forward.cu`
- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/forward.h`
- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/auxiliary.h` (hàm phụ trợ)
- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/config.h` (hằng số `BLOCK_X=16, BLOCK_Y=16, NUM_CHAFFELS=3`)

Bản phân tích cũ (`MATH/cuda/forward.md`) đã được đối chiếu từng công thức với code thật — kết quả: **toàn bộ công thức toán học trong bản cũ là ĐÚNG**, không có hằng số hay thứ tự phép tính nào sai. Tài liệu này giữ nguyên các công thức đã kiểm chứng, trình bày lại theo thứ tự code với đầy đủ "Ký hiệu", "Kiến thức toán nền tảng", mục kiểm chứng so với bài báo 3DGS gốc, và "Ví dụ số" tính tay. Các điểm từng được bản cũ ghi chú là "bất thường nhưng không phải lỗi toán" (floor $0.1$ khi tính eigenvalue, floor $0.1$ khi dilate hiệp phương sai) được giữ nguyên và giải thích rõ hơn ở mục 4 (Kiểm chứng).

---

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mu_i\in\mathbb R^3$ | Tâm Gaussian $i$ trong world-space (`orig_points`/`means3D`) |
| $s_i\in\mathbb R^3,\ q_i\in\mathbb R^4$ | Scale và quaternion $(r,x,y,z)$ của Gaussian $i$ |
| $m$ | `scale_modifier`, hệ số nhân scale toàn cục |
| $\Sigma_{3D,i}\in\mathbb R^{3\times3}$ | Hiệp phương sai 3D world-space |
| $W_{view}\in\mathbb R^{3\times3}$, $\mathbf t_{view}\in\mathbb R^3$ | Phần quay/tịnh tiến của `viewmatrix` (world→camera) |
| $t=(t_x,t_y,t_z)$ | Tâm Gaussian trong camera-space |
| $f_x,f_y$ | Tiêu cự pixel theo trục $x,y$ |
| $J\in\mathbb R^{3\times3}$ | Jacobian xấp xỉ affine của phép chiếu phối cảnh tại $t$ |
| $\Sigma'\in\mathbb R^{2\times2}$ | Hiệp phương sai 2D màn hình (sau dilation) |
| $\mathrm{conic}=\Sigma'^{-1}$ | Nghịch đảo hiệp phương sai 2D, lưu dạng $(\mathrm{conic}.x,\mathrm{conic}.y,\mathrm{conic}.z)=(\Sigma'^{-1}_{00},\Sigma'^{-1}_{01},\Sigma'^{-1}_{11})$ |
| $o_i\in(0,1)$ | Opacity gốc của Gaussian $i$ (`opacities`) |
| $\mathrm{coef}_i$ | Hệ số bù năng lượng của bộ lọc thông thấp 2D (anti-aliasing) |
| $c_i\in\mathbb R^3$ | Màu RGB của Gaussian $i$ sau khi giải mã SH |
| $\mathbf d_i(\mathbf x)=\mathrm{xy}_i-\mathbf x$ | Vector từ tâm chiếu $i$ đến pixel $\mathbf x$ |
| $\alpha_i(\mathbf x)$ | Độ mờ (alpha) của Gaussian $i$ tại pixel $\mathbf x$ |
| $T_k$ | Độ truyền qua (transmittance) còn lại sau $k$ Gaussian đầu tiên, $T_0=1$ |
| $C(\mathbf x)$ | Màu tích lũy (chưa cộng nền) tại pixel $\mathbf x$ |
| $\mathbf{bg}$ | Màu nền (background) |
| $D$ | $\deg$, bậc SH tối đa được dùng (0–3) |

---

## 2. `computeColorFromSH` — giải mã màu từ Spherical Harmonics (dòng 24–76)

### Bước 2.1 — Hướng nhìn

$$
\mathbf d_i=\frac{\mu_i-\mathbf o_{cam}}{\lVert\mu_i-\mathbf o_{cam}\rVert_2},\qquad (x,y,z)=(d_{i,x},d_{i,y},d_{i,z})
$$

```cpp
glm::vec3 dir = pos - campos;
dir = dir / glm::length(dir);
```

### Bước 2.2 — Hằng số SH thực chuẩn hoá (`auxiliary.h`)

$$
C_0=0.28209479177387814,\quad C_1=0.4886025119029199
$$

$$
C_2=[1.0925484305920792,\,-1.0925484305920792,\,0.31539156525252005,\,-1.0925484305920792,\,0.5462742152960396]
$$

$$
C_3=[-0.5900435899266435,\,2.890611442640554,\,-0.4570457994644658,\,0.3731763325901154,\,-0.4570457994644658,\,1.445305721320277,\,-0.5900435899266435]
$$

Đây đúng là các hệ số chuẩn hoá thực $K_l^m$ của hàm cầu điều hoà thực bậc $l\le3$ (xem mục 3.5). Không có sửa đổi so với 3DGS gốc.

### Bước 2.3 — Cộng dồn theo bậc

**Bậc 0** (DC, lưu tách riêng trong `dc`):
$$\mathbf c=C_0\cdot\mathrm{sh}_{dc}$$

**Bậc 1** ($\deg>0$):
$$\mathbf c\mathrel{+}=-C_1\,y\,\mathrm{sh}_0+C_1\,z\,\mathrm{sh}_1-C_1\,x\,\mathrm{sh}_2$$

**Bậc 2** ($\deg>1$, $xx=x^2,yy=y^2,zz=z^2,xy=xy,yz=yz,xz=xz$):
$$\mathbf c\mathrel{+}=C_2[0]xy\,\mathrm{sh}_3+C_2[1]yz\,\mathrm{sh}_4+C_2[2](2zz-xx-yy)\mathrm{sh}_5+C_2[3]xz\,\mathrm{sh}_6+C_2[4](xx-yy)\mathrm{sh}_7$$

**Bậc 3** ($\deg>2$):
$$
\begin{aligned}
\mathbf c\mathrel{+}=\ &C_3[0]y(3xx-yy)\mathrm{sh}_8+C_3[1]xyz\,\mathrm{sh}_9+C_3[2]y(4zz-xx-yy)\mathrm{sh}_{10}\\
&+C_3[3]z(2zz-3xx-3yy)\mathrm{sh}_{11}+C_3[4]x(4zz-xx-yy)\mathrm{sh}_{12}\\
&+C_3[5]z(xx-yy)\mathrm{sh}_{13}+C_3[6]x(xx-3yy)\mathrm{sh}_{14}
\end{aligned}
$$

### Bước 2.4 — Dịch offset và clamp dương

$$\mathbf c_{final}=\max(\mathbf c+0.5,\ 0)$$

`clamped[3i+k] = (c_k<0)` được lưu để backward áp dụng gradient $0$ trên nhánh bị clamp (straight-through ReLU).

---

## 3. `computeCov3D` — Hiệp phương sai 3D (dòng 135–169)

Với $q=(r,x,y,z)$ (**không chuẩn hoá lại trong hàm này**, dòng 144 `q = rot;` — comment chia `/glm::length(rot)` đã bị tắt, giả định Python đã chuẩn hoá trước) và $S=\mathrm{diag}(ms_x,ms_y,ms_z)$:

$$
R(q)=\begin{pmatrix}
1-2(y^2+z^2)&2(xy-rz)&2(xz+ry)\\
2(xy+rz)&1-2(x^2+z^2)&2(yz-rx)\\
2(xz-ry)&2(yz+rx)&1-2(x^2+y^2)
\end{pmatrix}
$$

$$
M=SR(q),\qquad \Sigma_{3D}=M^\top M
$$

**Lưu ý quy ước GLM** (đã kiểm chứng, không phải lỗi): `glm::mat3(a0,a1,...,a8)` nạp dữ liệu **theo cột**, nên ma trận `R` gõ trong code theo thứ tự dòng ở trên thực chất được GLM lưu là $R^\top$ của công thức toán chuẩn. Kết hợp với $\Sigma_{3D}=M^\top M=(SR)^\top(SR)=R^\top S^\top S R$, và vì GLM hoán đổi vai trò dòng/cột một lần nữa khi nhân, kết quả trị số cuối cùng trùng khớp với công thức chuẩn $\Sigma_{3D}=R S S^\top R^\top$ của bài báo 3DGS gốc (Kerbl et al. 2023, mục 3 Differentiable Point-Based Splatting). Đây là cách viết code gốc của 3DGS (Inria), **không phải đặc thù của StructGS** và không sai.

Chỉ lưu nửa trên tam giác đối xứng: `cov3D[0..5]` $=(\Sigma_{00},\Sigma_{01},\Sigma_{02},\Sigma_{11},\Sigma_{12},\Sigma_{22})$.

---

## 4. `computeCov2D` — Chiếu 3D→2D kiểu EWA splatting (dòng 79–130)

### Bước 4.1 — Tâm Gaussian trong camera-space

$$t=W_{view}\mu_i+\mathbf t_{view}\qquad(\texttt{transformPoint4x3})$$

### Bước 4.2 — Clamp góc nhìn (tránh biến dạng Jacobian ở rìa FOV)

$$
\mathrm{limx}=1.3\tan\!\big(\tfrac{\mathrm{fov}_x}{2}\big),\quad \mathrm{limy}=1.3\tan\!\big(\tfrac{\mathrm{fov}_y}{2}\big)
$$

$$
t_x\leftarrow\mathrm{clip}\Big(\frac{t_x}{t_z},-\mathrm{limx},\mathrm{limx}\Big)t_z,\qquad
t_y\leftarrow\mathrm{clip}\Big(\frac{t_y}{t_z},-\mathrm{limy},\mathrm{limy}\Big)t_z
$$

### Bước 4.3 — Jacobian xấp xỉ affine (EWA splatting, Zwicker et al. 2002, eq. 29)

$$
J=\begin{pmatrix}f_x/t_z&0&-f_xt_x/t_z^2\\0&f_y/t_z&-f_yt_y/t_z^2\\0&0&0\end{pmatrix}
$$

### Bước 4.4 — Chiếu hiệp phương sai (EWA splatting eq. 31)

Với $W=$ khối quay $3\times3$ trích từ `viewmatrix`, $T=WJ$:

$$
\Sigma'_{3\times3}=T^\top\Sigma_{3D}^\top T=JW\Sigma_{3D}W^\top J^\top
$$

Chỉ giữ khối $2\times2$ trên-trái (hàng/cột 3 luôn $=0$ vì hàng cuối của $J$ bằng $0$).

### Bước 4.5 — Bộ lọc thông thấp màn hình (anti-aliasing) + hệ số bù

```cpp
float det_0 = max(1e-6f, cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1]);
float kernel_size = 0.1f;
cov[0][0] += kernel_size; cov[1][1] += kernel_size;
float det_1 = max(1e-6f, cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1]);
float coef = sqrt(det_0 / (det_1 + 1e-6f));
```

$$
\det_0=\max(10^{-6},\Sigma'_{00}\Sigma'_{11}-{\Sigma'_{01}}^2)
$$

$$
\Sigma''_{00}=\Sigma'_{00}+0.1,\qquad \Sigma''_{11}=\Sigma'_{11}+0.1
$$

$$
\det_1=\max(10^{-6},\Sigma''_{00}\Sigma''_{11}-{\Sigma'_{01}}^2),\qquad
\mathrm{coef}=\sqrt{\frac{\det_0}{\det_1+10^{-6}}}
$$

Hàm trả về `float4 = (Σ''_00, Σ'_01, Σ''_11, coef)`. $\mathrm{coef}\le1$ bù lại việc "phình" Gaussian thêm $0.1\,\mathrm{px}^2$ phương sai (đảm bảo mọi Gaussian rộng ít nhất ~1 pixel, giảm alias khi lấy mẫu) mà vẫn giữ gần đúng tích phân năng lượng ban đầu — $\mathrm{coef}$ sẽ nhân trực tiếp vào opacity (mục 6).

---

## 5. `preprocessCUDA` — Nghịch đảo hiệp phương sai & bán kính màn hình

### Bước 5.1 — Nghịch đảo (conic), dòng 249–254

$$
\det=\Sigma''_{00}\Sigma''_{11}-{\Sigma'_{01}}^2
$$

Nếu $\det=0$: loại Gaussian (không render). Ngược lại:

$$
\Sigma'^{-1}=\mathrm{conic}=\frac1{\det}\begin{pmatrix}\Sigma''_{11}&-\Sigma'_{01}\\-\Sigma'_{01}&\Sigma''_{00}\end{pmatrix}
\ \Rightarrow\
(\mathrm{conic}.x,\mathrm{conic}.y,\mathrm{conic}.z)=\Big(\frac{\Sigma''_{11}}{\det},\,\frac{-\Sigma'_{01}}{\det},\,\frac{\Sigma''_{00}}{\det}\Big)
$$

### Bước 5.2 — Bán kính màn hình qua trị riêng, dòng 256–263

$$
\mathrm{mid}=\frac{\Sigma''_{00}+\Sigma''_{11}}{2}=\frac{\operatorname{tr}\Sigma'}{2}
$$

$$
\lambda_1=\mathrm{mid}+\sqrt{\max(0.1,\ \mathrm{mid}^2-\det)},\qquad
\lambda_2=\mathrm{mid}-\sqrt{\max(0.1,\ \mathrm{mid}^2-\det)}
$$

$$
\boxed{\ r^{screen}=\Big\lceil 3\sqrt{\max(\lambda_1,\lambda_2)}\Big\rceil\ }
$$

Công thức này **tương đương về đại số** với công thức trị riêng chuẩn của ma trận đối xứng $2\times2$ $\lambda_{1,2}=\dfrac{\operatorname{tr}\pm\sqrt{\operatorname{tr}^2-4\det}}{2}$, vì $\mathrm{mid}=\operatorname{tr}/2$ và $\mathrm{mid}^2-\det=(\operatorname{tr}^2-4\det)/4\Rightarrow\sqrt{\mathrm{mid}^2-\det}=\sqrt{\operatorname{tr}^2-4\det}/2$. Điểm khác biệt duy nhất — và là điểm cố ý, không phải lỗi — là code **chặn dưới phần dưới căn ở $0.1$** thay vì $0$ thuần tuý (xem mục kiểm chứng §9.1).

### Bước 5.3 — Opacity hiệu dụng & lọc theo tile, dòng 266–270

$$
\mathrm{con\_o}=(\mathrm{conic}.x,\mathrm{conic}.y,\mathrm{conic}.z,\ o_i\cdot\mathrm{coef}_i)
$$

`duplicateToTilesTouched` (trong `auxiliary.h`) tìm hình chữ nhật các tile mà ellipse mức $\alpha\ge1/255$ của Gaussian chạm tới, dùng ngưỡng

$$
t_{thresh}=\mathrm{mult}\cdot2\ln(255\cdot o_i\cdot\mathrm{coef}_i)
$$

(với `mult` là hệ số mở rộng tile, mặc định $1$). Nếu số tile chạm $=0$: loại Gaussian khỏi pipeline.

### Bước 5.4 — Pháp tuyến từ quaternion, dòng 291–300

Cột thứ 3 của $R(q)$ (trục $z$ cục bộ sau xoay), dùng làm normal xấp xỉ cho Gaussian phẳng (StructGS/2DGS-style):

$$
\mathbf n_i=\big(2(xz+ry),\ 2(yz-rx),\ 1-2(x^2+y^2)\big)
$$

---

## 6. `renderCUDA` — Rasterize bằng alpha compositing (dòng 306–532)

### Bước 6.1 — Độ mờ Gaussian 2D tại pixel, dòng 424–439

Với $\mathbf d=(\mathrm{xy}_i-\mathbf x_{pix})$:

$$
\mathrm{power}=-\tfrac12\big(\mathrm{conic}.x\,d_x^2+\mathrm{conic}.z\,d_y^2\big)-\mathrm{conic}.y\,d_xd_y=-\tfrac12\mathbf d^\top\Sigma'^{-1}\mathbf d
$$

Nếu $\mathrm{power}>0$: bỏ qua pixel (số học, $\mathrm{conic}$ mất PSD do làm tròn).

$$
G(\mathbf d)=\exp(\mathrm{power})
$$

$$
\boxed{\ \alpha_i=\min\big(0.99,\ \mathrm{con\_o}.w\cdot G(\mathbf d)\big)=\min\big(0.99,\ o_i\,\mathrm{coef}_i\exp(-\tfrac12\mathbf d^\top\Sigma'^{-1}\mathbf d)\big)\ }
$$

Nếu $\alpha_i<1/255$: bỏ qua (đóng góp không đáng kể, ngưỡng chuẩn 3DGS).

### Bước 6.2 — Tích luỹ alpha compositing theo thứ tự depth tăng dần, dòng 440–487

Danh sách Gaussian của mỗi tile đã được sort theo khoá `(tile, depth)` trước khi vào kernel này (ở bước duyệt/sort toàn cục, không nằm trong `forward.cu`), nên vòng lặp `j` duyệt đúng theo thứ tự **gần → xa camera**. Với $T_0=1$:

$$
T_{test}=T_{i-1}(1-\alpha_i)
$$

Nếu $T_{test}<10^{-4}$: `done=true`, dừng tích luỹ cho pixel này (các Gaussian xa hơn bị cắt vì đóng góp không đáng kể).

$$
C\mathrel{+}=c_i\,\alpha_i\,T_{i-1}\qquad\Longrightarrow\qquad
\boxed{\ C=\sum_i c_i\,\alpha_i\,T_{i-1},\quad T_{i-1}=\prod_{j<i}(1-\alpha_j)\ }
$$

cập nhật $T_i\leftarrow T_{test}$. Đây đúng là **Eq. (3)** của bài báo 3DGS gốc (Kerbl et al., 2023) — xem kiểm chứng §9.2.

### Bước 6.3 — Pha trộn nền & kênh phụ, dòng 490–523

$$
C_{final}=C+T_{final}\cdot\mathbf{bg}
$$

Kênh phụ (`compute_extra`, cùng trọng số $w_i=\alpha_iT_{i-1}$):

$$
D=\sum_i d_i\,\alpha_i\,T_{i-1},\qquad
\mathbf n_{raw}=\sum_i\mathbf n_i\,\alpha_i\,T_{i-1},\qquad
\mathbf n_{out}=\frac{\mathbf n_{raw}}{\lVert\mathbf n_{raw}\rVert_2}
$$

$$
O=1-T_{final}=1-\prod_i(1-\alpha_i)
$$

($O$ là độ phủ/alpha tích luỹ tổng — công thức volume rendering chuẩn.)

### Bước 6.4 — Theo dõi $T^{\max}$ và contributor lớn nhất

```cpp
atomicMax((int*)&cov2Ds[collected_id[j]*7+6], __float_as_int(T));
```

Dùng trick "so sánh bit pattern của IEEE-754 float dương như int" để lấy max $T$ qua tất cả pixel/view mà Gaussian này góp mặt — đây là $T^{\max}_i$ dùng ở bước lọc `update_freq_stats_online` (xem `DOCS/BOOK/eta.md`). `max_weight`/`max_id` theo dõi Gaussian đóng góp trọng số $\alpha_iT_{i-1}$ lớn nhất cho mỗi pixel.

---

## 7. Kiến thức toán nền tảng

### 7.1 Trị riêng ma trận đối xứng $2\times2$

Với $A=\begin{pmatrix}a&b\\b&c\end{pmatrix}$, đa thức đặc trưng $\det(A-\lambda I)=\lambda^2-(\operatorname{tr}A)\lambda+\det A=0$ cho:

$$
\lambda_{1,2}=\frac{\operatorname{tr}A\pm\sqrt{(\operatorname{tr}A)^2-4\det A}}{2}=\frac{a+c}{2}\pm\sqrt{\Big(\frac{a-c}{2}\Big)^2+b^2}
$$

Biểu thức dưới căn luôn $\ge0$ (vì $A$ đối xứng thực nên trị riêng thực), bằng $0$ khi và chỉ khi $A$ là ma trận vô hướng ($a=c,\ b=0$). Với ma trận hiệp phương sai (bán xác định dương, $\det A\ge0$), $\lambda_1\ge\lambda_2\ge0$.

### 7.2 Phân phối Gaussian 2D và ellipse mức

Gaussian 2D chuẩn hoá: $p(\mathbf x)=\dfrac1{2\pi\sqrt{\det\Sigma}}\exp\big(-\tfrac12(\mathbf x-\mu)^\top\Sigma^{-1}(\mathbf x-\mu)\big)$. Rasterizer 3DGS **không chuẩn hoá** theo $1/(2\pi\sqrt{\det\Sigma})$ — chỉ dùng phần "hạt nhân" (kernel) $G(\mathbf d)=\exp(-\tfrac12\mathbf d^\top\Sigma^{-1}\mathbf d)$ với đỉnh $G(\mathbf 0)=1$, nhân thêm opacity $o_i\in(0,1)$ đóng vai trò "biên độ đục" thay vì xác suất. Ellipse mức $\{\mathbf d:\mathbf d^\top\Sigma^{-1}\mathbf d=k^2\}$ có bán trục theo hướng vector riêng $\mathbf e_j$ bằng $k\sqrt{\lambda_j}$ — đây là lý do bán kính màn hình $r^{screen}=3\sqrt{\lambda_{\max}}$ (mục 5.2) ứng với ellipse mức $k=3$ (khoảng $3\sigma$, phủ $>98.9\%$ khối lượng Gaussian 2D).

### 7.3 Alpha compositing / phương trình volume rendering rời rạc

Volume rendering liên tục (NeRF, Mildenhall et al. 2020): $C=\int_{t_n}^{t_f}T(t)\sigma(t)c(t)\,dt$, $T(t)=\exp\big(-\int_{t_n}^t\sigma(s)ds\big)$. Rời rạc hoá theo "quadrature rule" (Max, 1995) với $N$ mẫu và khoảng cách $\delta_i$:

$$
C=\sum_{i=1}^N T_i\big(1-\exp(-\sigma_i\delta_i)\big)c_i,\qquad T_i=\prod_{j<i}\exp(-\sigma_j\delta_j)
$$

Đặt $\alpha_i:=1-\exp(-\sigma_i\delta_i)\in[0,1]$ ("độ mờ rời rạc" của mẫu $i$), công thức trên trở thành đúng dạng splatting:

$$
C=\sum_iT_i\alpha_ic_i,\qquad T_i=\prod_{j<i}(1-\alpha_j)
$$

3DGS thay "mẫu dọc tia" bằng "Gaussian chồng lên nhau tại pixel", và $\alpha_i$ được tính trực tiếp từ hình dạng Gaussian ($\alpha_i=o_i\,G(\mathbf d)$) thay vì từ mật độ $\sigma_i$ — đây chính là Eq. (2)–(3) trong bài báo 3DGS (Kerbl et al., 2023), kiểm chứng ở mục 9.2.

### 7.4 Hình học chiếu phối cảnh & Jacobian EWA splatting

Phép chiếu pinhole $u=f_x x/z+c_x,\ v=f_y y/z+c_y$ là phi tuyến, nên ảnh của một Gaussian 3D qua phép chiếu không phải Gaussian 2D chính xác. Zwicker et al. (2002, "EWA Splatting") tuyến tính hoá quanh tâm $t=(t_x,t_y,t_z)$ bằng khai triển Taylor bậc 1 (Jacobian), biến đổi affine gần đúng:

$$
\begin{pmatrix}u\\v\end{pmatrix}\approx J\cdot(\mathbf x-t)+\begin{pmatrix}u_0\\v_0\end{pmatrix},\qquad
J=\frac{\partial(u,v)}{\partial(x,y,z)}\Big|_t=\begin{pmatrix}f_x/t_z&0&-f_xt_x/t_z^2\\0&f_y/t_z&-f_yt_y/t_z^2\end{pmatrix}
$$

Với phép biến đổi tuyến tính $\mathbf y=A\mathbf x$, hiệp phương sai biến đổi theo $\Sigma_y=A\Sigma_xA^\top$ — áp dụng liên tiếp world→camera ($W$) rồi camera→screen ($J$) cho $\Sigma'=JW\Sigma_{3D}W^\top J^\top$ (mục 4.4).

### 7.5 Spherical Harmonics thực bậc thấp

SH thực chuẩn hoá $Y_l^m(\theta,\phi)$ là cơ sở trực chuẩn trên mặt cầu, dùng biểu diễn hàm màu phụ thuộc hướng nhìn gọn (giống Fourier trên mặt cầu). 3DGS dùng SH thực tới bậc $l\le3$ ($16$ hệ số: $1+3+5+7$), viết dưới dạng đa thức theo $(x,y,z)=\mathbf d$ (hướng nhìn đơn vị) nhân hằng số chuẩn hoá $K_l^m$ — chính là các hằng số $C_0,C_1,C_2,C_3$ ở mục 2.2. Vì SH là hàm trực chuẩn, thêm một bậc không làm hỏng các bậc thấp hơn — đúng với cấu trúc "cộng dồn theo bậc" của `computeColorFromSH`.

---

## 8. Kiến trúc song song (bối cảnh, không phải công thức toán nhưng ảnh hưởng số học)

- `preprocessCUDA`: 1 thread / 1 Gaussian, lưới `((P+255)/256, 256)`.
- `renderCUDA`: 1 thread / 1 pixel, 1 block / 1 tile $16\times16$ (`BLOCK_X=BLOCK_Y=16`, `NUM_CHAFFELS=3` kênh RGB). Dữ liệu Gaussian được nạp theo lô (`BLOCK_SIZE=256`) vào shared memory, mọi thread trong block cùng duyệt một Gaussian tại một thời điểm (`collected_id/xy/conic_opacity`) rồi `block.sync()` — đây là lý do mọi pixel trong cùng tile thấy đúng cùng một thứ tự Gaussian theo depth.

---

## 9. Kiểm chứng tính đúng sai

### 9.1 Floor $0.1$ trong công thức trị riêng — khác biệt có chủ đích, không phải lỗi toán

Công thức trị riêng toán học thuần tuý là $\lambda_{1,2}=\mathrm{mid}\pm\sqrt{\mathrm{mid}^2-\det}$ (mục 7.1, tương đương dạng $\operatorname{tr}\pm\sqrt{\operatorname{tr}^2-4\det}$ chia 2). Code thay $\sqrt{\mathrm{mid}^2-\det}$ bằng $\sqrt{\max(0.1,\ \mathrm{mid}^2-\det)}$. Vì $\mathrm{mid}^2-\det\ge0$ luôn đúng về mặt toán (ma trận hiệp phương sai PSD), floor $0.1$ **chỉ có tác dụng khi biểu thức rất gần $0$** (Gaussian gần tròn đều, $\lambda_1\approx\lambda_2$) — nó **nới rộng khoảng cách giữa $\lambda_1$ và $\lambda_2$ một chút**, không làm sai giá trị $\max(\lambda_1,\lambda_2)$ dùng để tính bán kính (vì $\lambda_1$ chỉ tăng thêm tối đa $\approx\sqrt{0.1}-0\approx0.316$ so với giá trị đúng khi discriminant xấp xỉ $0$). Đây là biện pháp an toàn số học/chống Gaussian bán kính $0$, không phải lỗi logic. **Kết luận: bản cũ (`MATH/cuda/forward.md`) đã mô tả đúng công thức này, không cần sửa.**

### 9.2 Alpha compositing so với phương trình volume rendering gốc (3DGS, Kerbl et al. 2023)

So khớp trực tiếp:

| Bài báo 3DGS (Eq. 1–3) | Code `renderCUDA` | Khớp? |
|---|---|---|
| $\alpha_i=1-\exp(-\sigma_i\delta_i)$, thực hành $=o_i\cdot G_i'(\mathbf x)$ với $G'$ là Gaussian 2D chiếu | `alpha = min(0.99f, con_o.w * exp(power))`, $\mathrm{con\_o}.w=o_i\mathrm{coef}_i$ | **Khớp**, có thêm trần $0.99$ (ổn định số, chuẩn 3DGS gốc) và hệ số $\mathrm{coef}_i$ (riêng của biến thể anti-aliasing, không có trong bài báo gốc 2023 mà xuất hiện ở bản mip-splatting sau này — không sai, là mở rộng hợp lệ) |
| $C=\sum_iT_i\alpha_ic_i$ | `C[ch] += features[...]*alpha*T` (dùng $T$ **trước** khi cập nhật, tức $T_{i-1}$) | **Khớp chính xác** |
| $T_i=\prod_{j<i}(1-\alpha_j)$ | `T = test_T` cập nhật **sau** khi cộng màu | **Khớp**, đúng thứ tự: dùng $T_{i-1}$ để cộng màu rồi mới gán $T_i$ |
| Dừng sớm khi $T\to0$ | `if (test_T < 0.0001f) done = true;` | **Khớp**, $\epsilon=10^{-4}$ là ngưỡng chuẩn của code gốc Inria |
| Ngưỡng bỏ qua $\alpha$ nhỏ | `if (alpha < 1.0f/255.0f) continue;` | **Khớp**, $1/255$ = độ phân giải 1 mức xám 8-bit, chuẩn 3DGS gốc |
| Composite với nền | $C_{final}=C+T_{final}\mathbf{bg}$ | `out_color = C[ch] + T*bg_color[ch]` — **Khớp**, đúng kiểu "nền ở vô cực" trong NeRF ($T_N\cdot c_{bg}$) |

**Kết luận: không phát hiện bug trong công thức alpha compositing.** Thứ tự thao tác (cộng màu trước, cập nhật $T$ sau) là điểm dễ viết sai nhất khi tự cài đặt lại — code hiện tại đúng.

### 9.3 Spherical Harmonics so với chuẩn toán học

Đối chiếu từng hệ số $C_0,C_1,C_2,C_3$ và từng số hạng đa thức với bảng hàm cầu điều hoà thực chuẩn $l\le3$ (dùng trong Ramamoorthi & Hanrahan 2001, và chính là bảng mà 3DGS gốc/`Zhang et al. 2022` dùng) — **khớp tuyệt đối, không lệch dấu, không lệch hệ số**. Không phát hiện sai sót.

### 9.4 Sự không nhất quán forward/backward — đã biết, ghi nhận lại

So với `backward.cu` (`computeCov2DCUDA`), hằng số cộng vào đường chéo khi dilate hiệp phương sai 2D ở **forward là $0.1$** (mục 4.5) trong khi **backward dùng $0.3$** (theo `MATH/cuda/forward.md` mục 4, đã verify từ bản phân tích backward trước đó). Đây là **điểm không khớp thật sự trong codebase** — có khả năng là tàn dư khi hợp nhất code anti-aliasing 3DGS 2024 vào một backward cũ hơn. Về mặt toán, nó khiến **gradient lan truyền ngược không khớp chính xác với hàm forward thực tế đã chạy** (gradient được tính như thể $\mathrm{kernel\_size}=0.3$ trong khi forward dùng $0.1$) — đây là một **bug tiềm ẩn ảnh hưởng độ chính xác gradient**, cần sửa ở `backward.cu` để khớp $0.1$ nếu muốn gradient chính xác tuyệt đối, nhưng **không nằm trong phạm vi `forward.cu`/`forward.h`** đang xét ở đây nên chỉ ghi nhận, không sửa trong tài liệu này.

### 9.5 Tổng kết đối chiếu với bản cũ

Không có công thức nào trong `MATH/cuda/forward.md` bị sai so với code thật. Các bổ sung của bản mới này so với bản cũ:
1. Thêm phần Ký hiệu tường minh.
2. Thêm phần Kiến thức toán nền tảng (trị riêng $2\times2$, Gaussian 2D, volume rendering liên tục→rời rạc, Jacobian phối cảnh, SH) để người đọc hiểu "vì sao" công thức đúng, không chỉ "nó là gì".
3. Thêm bảng đối chiếu trực tiếp với Eq. (1)-(3) của bài báo 3DGS gốc (mục 9.2).
4. Làm rõ floor $0.1$ trong eigenvalue không phải lỗi (mục 9.1) — bản cũ chỉ liệt kê công thức mà chưa phân tích tác động.
5. Thêm Ví dụ số tính tay đầy đủ pipeline $\Sigma'\to$ conic $\to\alpha\to$ alpha compositing $\to$ màu pixel cuối (mục 10), bản cũ chưa có.

---

## 10. Ví dụ số

### 10.1 Bán kính màn hình từ hiệp phương sai 2D (kiểm chứng Bước 5.2)

**Gaussian đẳng hướng** $\Sigma'=\begin{pmatrix}4&0\\0&4\end{pmatrix}$:

$$
\mathrm{mid}=4,\quad \det=16-0=16,\quad \mathrm{mid}^2-\det=16-16=0
$$

$$
\sqrt{\max(0.1,0)}=\sqrt{0.1}=0.3162
$$

$$
\lambda_1=4.3162,\quad\lambda_2=3.6838\quad(\text{so với giá trị toán học đúng }\lambda_1=\lambda_2=4\text{ nếu không floor})
$$

$$
r^{screen}=\lceil3\sqrt{4.3162}\rceil=\lceil3\times2.0775\rceil=\lceil6.2325\rceil=7\ \text{px}
$$

(So với $\lceil3\sqrt4\rceil=\lceil6\rceil=6$ nếu không có floor — floor $0.1$ làm bán kính lớn hơn **đúng 1 pixel** trong trường hợp biên này, một sai số nhỏ, có chủ đích, chấp nhận được vì mục đích là tránh bỏ sót tile biên.)

**Gaussian dị hướng** $\Sigma'=\begin{pmatrix}9&2\\2&4\end{pmatrix}$:

$$
\mathrm{mid}=6.5,\quad\det=36-4=32,\quad\mathrm{mid}^2-\det=42.25-32=10.25
$$

$$
\sqrt{\max(0.1,10.25)}=\sqrt{10.25}=3.2016\quad(\text{floor không tác dụng vì đã}\gg0.1)
$$

$$
\lambda_1=9.7016,\ \lambda_2=3.2984,\qquad r^{screen}=\lceil3\sqrt{9.7016}\rceil=\lceil9.344\rceil=10\ \text{px}
$$

### 10.2 Pipeline đầy đủ: 3 Gaussian → alpha compositing → màu pixel

Pixel đang render: $\mathbf x_{pix}=(50,50)$. Nền $\mathbf{bg}=(0.1,0.1,0.1)$ (xám nhạt). Giả sử `coef`$=1$ cho cả 3 Gaussian (bỏ qua bộ lọc 2D để đơn giản hoá số). Ba Gaussian đã **sort theo depth tăng dần** (G1 gần nhất, G3 xa nhất) — đúng thứ tự mà `renderCUDA` thấy.

| Gaussian | $\Sigma'$ | $\mu$ (pixel) | $o_i$ | $c_i$ (RGB) |
|---|---|---|---|---|
| G1 | $\mathrm{diag}(4,4)$ | $(50,50)$ | $0.6$ | $(1,0,0)$ đỏ |
| G2 | $\mathrm{diag}(4,4)$ | $(52,50)$ | $0.9$ | $(0,1,0)$ lục |
| G3 | $\mathrm{diag}(4,4)$ | $(55,50)$ | $0.8$ | $(0,0,1)$ lam |

**Conic chung** (vì $\Sigma'$ giống hệt nhau): $\det=16$, $\mathrm{conic}=(4/16,0,4/16)=(0.25,0,0.25)$.

**G1**: $\mathbf d=(50-50,50-50)=(0,0)$.
$$\mathrm{power}=-\tfrac12(0.25\cdot0+0.25\cdot0)-0=0,\quad G=1$$
$$\alpha_1=\min(0.99,\ 0.6\times1)=0.6$$
$$T_0=1,\qquad C\mathrel{+}=(1,0,0)\times0.6\times1=(0.6,0,0)$$
$$T_1=T_0(1-\alpha_1)=1\times0.4=0.4$$

**G2**: $\mathbf d=(52-50,50-50)=(2,0)$.
$$\mathrm{power}=-\tfrac12(0.25\times4+0.25\times0)-0=-0.5,\quad G=e^{-0.5}=0.60653$$
$$\alpha_2=\min(0.99,\ 0.9\times0.60653)=\min(0.99,0.54588)=0.54588$$
$$C\mathrel{+}=(0,1,0)\times0.54588\times0.4=(0,0.21835,0)\ \Rightarrow\ C=(0.6,0.21835,0)$$
$$T_2=T_1(1-\alpha_2)=0.4\times0.45412=0.18165$$

**G3**: $\mathbf d=(55-50,50-50)=(5,0)$.
$$\mathrm{power}=-\tfrac12(0.25\times25)=-3.125,\quad G=e^{-3.125}=0.04394$$
$$\alpha_3=\min(0.99,\ 0.8\times0.04394)=0.03515$$
$$C\mathrel{+}=(0,0,1)\times0.03515\times0.18165=(0,0,0.006386)\ \Rightarrow\ C=(0.6,0.21835,0.006386)$$
$$T_3=T_2(1-\alpha_3)=0.18165\times0.96485=0.17526$$

Hết danh sách Gaussian của pixel này (không có $T<10^{-4}$ nên vòng lặp kết thúc tự nhiên, không bị cắt sớm).

**Pha màu nền** (dòng 499): $C_{final}=C+T_3\cdot\mathbf{bg}$

$$
C_{final}=(0.6,0.21835,0.006386)+0.17526\times(0.1,0.1,0.1)
$$

$$
\boxed{\ C_{final}=(0.61753,\ 0.23588,\ 0.02391)\ }
$$

**Kiểm tra bảo toàn năng lượng** (xem mục 7.3): tổng trọng số phải $\le1$:
$$
\alpha_1T_0+\alpha_2T_1+\alpha_3T_2+T_3=0.6+0.21835+0.006386+0.17526=1.00000\ \checkmark
$$

khớp đúng $1$ (trong sai số làm tròn), xác nhận alpha compositing bảo toàn tổng trọng số $=1$ như lý thuyết volume rendering rời rạc yêu cầu (mục 7.3) — đây là một kiểm tra đúng-sai thực nghiệm cho thấy code không có lỗi chuẩn hoá.

**Opacity tích luỹ** $O=1-T_3=1-0.17526=0.82474$, khớp với $1-\prod(1-\alpha_i)=1-(0.4\times0.45412\times0.96485)=1-0.17526=0.82474\ \checkmark$.

### 10.3 Kiểm tra màu từ SH (bậc 0 only, minh hoạ nhanh)

Với $\deg=0$, chỉ còn $\mathbf c=C_0\cdot\mathrm{sh}_{dc}$. Lấy $\mathrm{sh}_{dc}=(1.0,0.5,-0.2)$:

$$
\mathbf c=0.28209\times(1.0,0.5,-0.2)=(0.28209,\,0.14105,\,-0.05642)
$$

$$
\mathbf c_{final}=\max(\mathbf c+0.5,0)=\max((0.78209,\,0.64105,\,0.44358),\,0)=(0.78209,\,0.64105,\,0.44358)
$$

Không kênh nào âm nên `clamped=(false,false,false)` — minh hoạ đúng công thức mục 2.4, không có clamp xảy ra trong ví dụ này.
