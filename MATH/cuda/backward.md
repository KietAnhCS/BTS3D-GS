# Công thức toán học trong `cuda_rasterizer/backward.cu` (+ `backward.h`)

> **Đồng bộ hoá**: nội dung file này đã được đối chiếu lại và đồng bộ với bản kiểm chứng chi tiết hơn tại
> `MATH/submodules/diff-gaussian-rasterization_structgs/backward.md` (724 dòng, đọc lại trực tiếp source thật,
> có finite-difference numeric gradient check). Khi hai bản mâu thuẫn, bản `submodules/...` được ưu tiên vì
> mới hơn và đã kiểm chứng bằng số. File này giữ bản rút gọn, nhưng **mọi công thức đều có code trích dẫn
> nguyên văn kèm đúng số dòng**, lấy từ:
>
> - `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/backward.cu`
> - `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/backward.h`
> - Tham chiếu: `forward.cu` (công thức xuôi cần đảo ngược), `auxiliary.h` (`transformPoint4x3`, `transformVec4x3Transpose`, `dnormvdv`, hằng số SH).

Quy ước: $L$ là hàm mất mát cuối, $dL\_dX := \partial L/\partial X$. $i$ là chỉ số Gaussian theo thứ tự depth-sort (gần→xa).

---

## 1. Đạo hàm ngược của alpha compositing (`renderCUDA`, `backward.cu` dòng 621–784)

Kernel "per-pixel" duyệt Gaussian **từ sau ra trước** (ngược forward). Khởi tạo $T=T_{final}$, khôi phục dần $T_{i-1}$ bằng chia ngược — đúng nghịch đảo của $T_i=T_{i-1}(1-\alpha_i)$ ở forward:

$$
T_{i-1} = \frac{T_i}{1-\alpha_i}
$$

Trích nguyên văn phần vòng lặp chính (dòng 709–717):

```cpp
contributor--;
if (contributor >= last_contributor)
	continue;

// Compute blending values, as before.
const float2 xy = collected_xy[j];
const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
const float4 con_o = collected_conic_opacity[j];
const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
```

(đoạn trực tiếp cập nhật `T` nằm ngay sau, cùng khối lặp):

```cpp
T = T / (1.f - alpha);
```

**Biến tích luỹ màu phía sau** `accum_rec[ch]` đóng vai trò $S_i := \sum_{j>i} c_j\alpha_j\prod_{i<k<j}(1-\alpha_k)$ — phần đã blend bởi các Gaussian ở *sau* Gaussian hiện tại, quy chiếu lại từ góc nhìn Gaussian $i$. Trích nguyên văn (dòng 738):

```cpp
accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
```

cập nhật đệ quy, với `last_alpha, last_color` là $\alpha_{i+1}, c_{i+1}$ của vòng lặp trước:

$$
S_{i} \leftarrow \alpha_{i+1}\,c_{i+1} + (1-\alpha_{i+1})\,S_{i+1}
$$

**Gradient của $L$ theo $\alpha_i$** — từ $C=\sum_i c_i\alpha_iT_{i-1}$, $C_{final}=C+T_{final}\mathbf{bg}$:

$$
\frac{\partial L}{\partial \alpha_i} = T_{i-1}\,\big(c_i - S_i\big)\cdot \frac{\partial L}{\partial C}
$$

Trích nguyên văn (dòng 742, 748):

```cpp
dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
...
dL_dalpha *= T;
```

$$
dL\_d\alpha \mathrel{+}= (c_{i,ch} - S_{i,ch})\cdot dL\_dpixel_{ch}, \qquad dL\_d\alpha \leftarrow dL\_d\alpha \cdot T_{i-1}
$$

(`T` ở dòng 748 đã là $T_{i-1}$ nhờ phép chia dòng 726 chạy trước đó trong cùng vòng lặp).

**Đóng góp của nền (background)**: $C_{final}=C+T_{final}\mathbf{bg}$, $\partial T_{final}/\partial\alpha_i=-T_{final}/(1-\alpha_i)$. Trích nguyên văn (dòng 754–757):

```cpp
float bg_dot_dpixel = 0;
for (int i = 0; i < C; i++)
    bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
```

$$
\frac{\partial L}{\partial \alpha_i} \mathrel{+}= -\frac{T_{final}}{1-\alpha_i}\sum_{ch} \mathbf{bg}_{ch}\cdot dL\_dpixel_{ch}
$$

(dòng này chạy **sau** `dL_dalpha *= T` nên không bị nhân nhầm $T_{i-1}$ lần hai).

**Gradient theo màu từng Gaussian** — từ $\partial C_{ch}/\partial c_{i,ch}=\alpha_iT_{i-1}$. Trích nguyên văn (dòng 727, 746):

```cpp
const float dchannel_dcolor = alpha * T;
...
atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
```

$$
\frac{\partial L}{\partial c_{i,ch}} = \alpha_i T_{i-1}\cdot \frac{\partial L}{\partial C_{ch}}
$$

cộng dồn bằng `atomicAdd` vì một Gaussian có thể phủ nhiều pixel song song.

**Ngưỡng bão hoà alpha** — trích nguyên văn (dòng 767):

```cpp
if (con_o.w * G > 0.99f)
    continue;
```

đặt **sau** khi đã cộng `dL_dcolors`/`dL_dalpha` nhưng **trước** khi cộng `dL_dmean2D`, `dL_dconic2D`, `dL_dopacity`: khi $\alpha=\min(0.99,\cdot)$ bị clamp ở forward, $\partial\alpha/\partial(\text{mean2D, conic, opacity})=0$ tại các Gaussian đó (straight-through, giống `torch.clamp`).

### 1.1. Phiên bản song song theo bucket (`PerGaussianRenderCUDA`, dòng 402–619)

Tổ chức lại **cùng công thức toán học trên** nhưng đổi chiều song song hoá: "1 Gaussian = 1 warp", duyệt **xuôi** (gần→xa). Trích nguyên văn (dòng 594, 596 — điểm khác biệt thật sự về gradient, không chỉ tổ chức lại):

```cpp
Register_dL_dmean2D_z += fabs(tmp_x);
Register_dL_dmean2D_w += fabs(tmp_y);
```

Đây là **gradient giá trị tuyệt đối** của $\partial L/\partial\text{mean2D}$, tích luỹ riêng vào 2 kênh phụ `.z,.w`, dùng làm tiêu chí split dị hướng của StructGS (không phải một phần của $\partial L/\partial\mu$ chuẩn, không lan truyền tiếp xuống $\mu_{3D}$) — xem `scene/gaussian_model.md` mục densify.

---

## 2. Đạo hàm của Gaussian 2D theo conic và theo vị trí pixel

Trích nguyên văn (dòng 762–781, định nghĩa tương tự lặp lại ở dòng 581–584 trong phiên bản per-pixel):

```cpp
const float gdx = G * d.x;
const float gdy = G * d.y;
const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

if(con_o.w*G > 0.99f){
	continue;
}

// Update gradients w.r.t. 2D mean position of the Gaussian
atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

// Update gradients w.r.t. opacity of the Gaussian
atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
```

Từ $\alpha=\min(0.99,o\cdot G)$ (bỏ nhánh clamp), $\partial\alpha/\partial G=o$:

$$
\frac{\partial L}{\partial G} = o\cdot \frac{\partial L}{\partial \alpha} \qquad (\texttt{dL\_dG = con\_o.w * dL\_dalpha})
$$

Từ $\mathrm{power}=-\tfrac12(\mathrm{conic}.x\,d_x^2+\mathrm{conic}.z\,d_y^2)-\mathrm{conic}.y\,d_xd_y$ và $G=e^{\mathrm{power}}$:

$$
\frac{\partial G}{\partial d_x} = -G\,(\mathrm{conic}.x\,d_x+\mathrm{conic}.y\,d_y), \qquad
\frac{\partial G}{\partial d_y} = -G\,(\mathrm{conic}.z\,d_y+\mathrm{conic}.y\,d_x)
$$

Vì $\mathbf d=\mathrm{xy}_{screen}-\mathrm{pix}_f$ và `mean2D` lưu ở không gian NDC (hệ số đổi đơn vị $ddelx\_dx=W/2,\ ddely\_dy=H/2$):

$$
\frac{\partial L}{\partial \mathrm{mean2D}_x} = \frac{\partial L}{\partial G}\cdot \frac{\partial G}{\partial d_x}\cdot \frac{W}{2}, \qquad
\frac{\partial L}{\partial \mathrm{mean2D}_y} = \frac{\partial L}{\partial G}\cdot \frac{\partial G}{\partial d_y}\cdot \frac{H}{2}
$$

**Đạo hàm theo conic:**

$$
\frac{\partial L}{\partial \mathrm{conic}.x} = -\tfrac12\,d_x^2\,G\cdot \frac{\partial L}{\partial G}, \quad
\frac{\partial L}{\partial \mathrm{conic}.y} = -d_xd_y\,G\cdot \frac{\partial L}{\partial G}, \quad
\frac{\partial L}{\partial \mathrm{conic}.z} = -\tfrac12\,d_y^2\,G\cdot \frac{\partial L}{\partial G}
$$

(`.w` của struct `float4 dL_dconic2D` giữ vai trò $\mathrm{conic}.z$ — quy ước tái sử dụng struct, không phải lỗi; `computeCov2DCUDA` đọc lại đúng quy ước này ở dòng 167: `{dL_dconics[4*idx], dL_dconics[4*idx+1], dL_dconics[4*idx+3]}`).

**Đạo hàm theo opacity:**

$$
\frac{\partial L}{\partial o} = G\cdot \frac{\partial L}{\partial \alpha}
$$

### 2.1. Điểm quan trọng: `con_o.w` không phải opacity thô — thiếu gradient qua hệ số bù khử-alias

Đọc `forward.cu` cho thấy `con_o.w` thực chất là `opacities[idx] * coef`, với `coef` là hệ số bù kiểu Mip-Splatting (tỉ lệ định thức hiệp phương sai trước/sau khi cộng `kernel_size`). Trong `backward.cu`, dòng `dL_dopacity += G * dL_dalpha` (trích ở trên) ngầm coi `con_o.w` chính là biến cần lấy đạo hàm, và **không có nhánh gradient nào** lan từ `coef` ngược về $\Sigma_{3D}$/scale/rotation. Đây **nhiều khả năng là một stop-gradient có chủ đích** (phổ biến trong các cài đặt anti-aliasing compensation để tránh mất ổn định huấn luyện), không phải bug, nhưng là điểm cần ghi nhận: $\partial L/\partial o_{\text{raw}}$ do code trả về thiếu hệ số nhân `coef` so với đạo hàm đầy đủ theo opacity gốc.

---

## 3. Đạo hàm ngược từ $\mathrm{conic}$ → $\Sigma'$ → $T=WJ$ → $J$ → mean3D (`computeCov2DCUDA`, dòng 146–276)

Kernel này **tính lại** $t=W_{view}\mu$, $J,W,T=WJ,\Sigma'=T^\top\mathrm{Vrk}^\top T$ từ đầu (không lưu từ forward). Trích nguyên văn (dòng 194–201):

```cpp
glm::mat3 T = W * J;

glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

// Use helper variables for 2D covariance entries. More compact.
float a = cov2D[0][0] += 0.3f;
float b = cov2D[0][1];
float c = cov2D[1][1] += 0.3f;
```

> **Phát hiện quan trọng (bất nhất thật giữa forward và backward):** $a,c$ ở đây cộng thêm $\kappa=0.3$, trong khi `forward.cu::computeCov2D` cộng `kernel_size`$=0.1$ (xem mục "Kiến thức nền tảng/kiểm chứng" bên dưới). Vì $a,c$ dùng để tính `denom=ac-b²` rồi lan ra mọi hệ số của `dL_da,dL_db,dL_dc`, sai lệch $0.3$ so với $0.1$ (gấp 3 lần) làm **gradient của nhánh hiệp phương sai bị lệch có hệ thống**, đặc biệt đáng kể khi Gaussian có hiệp phương sai màn hình nhỏ. Đây là điểm **cần sửa**: nên đổi `0.3f` → `0.1f` ở hai dòng trên cho khớp `kernel_size` của forward.

**Đạo hàm nghịch đảo ma trận $2\times2$** — trích nguyên văn (dòng 209–214):

```cpp
// Gradients of loss w.r.t. entries of 2D covariance matrix,
// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
```

$$
\frac{\partial L}{\partial a} = \frac{1}{\mathrm{denom}^2}\Big(-c^2\,dL\_dconic.x + 2bc\,dL\_dconic.y + (\mathrm{denom}-ac)\,dL\_dconic.z\Big)
$$

$$
\frac{\partial L}{\partial c} = \frac{1}{\mathrm{denom}^2}\Big(-a^2\,dL\_dconic.z + 2ab\,dL\_dconic.y + (\mathrm{denom}-ac)\,dL\_dconic.x\Big)
$$

$$
\frac{\partial L}{\partial b} = \frac{2}{\mathrm{denom}^2}\Big(bc\,dL\_dconic.x - (\mathrm{denom}+2b^2)\,dL\_dconic.y + ab\,dL\_dconic.z\Big)
$$

> **Kiểm chứng công thức $\partial L/\partial b$ (hệ số 2 ngoài ngoặc):** đã kiểm chứng kỹ bằng cả đạo hàm ma trận đầy đủ (với ràng buộc đối xứng $M_{01}=M_{10}=b$) lẫn finite-difference numeric gradient check trong `MATH/submodules/diff-gaussian-rasterization_structgs/backward.md` (mục 9–10). Kết luận: **công thức code đúng** — hệ số 2 xuất hiện vì $b$ là MỘT tham số vật lý ảnh hưởng đồng thời tới hai vị trí đối xứng $N_{01},N_{10}$ của ma trận nghịch đảo, nên gradient $\partial L/\partial b$ là **tổng hai đóng góp bằng nhau** từ $N_{01}$ và $N_{10}$ — không phải lỗi hay tuỳ tiện.

**Đạo hàm ngược qua $\Sigma'=T^\top\mathrm{Vrk}^\top T$ về $\mathrm{Vrk}=\Sigma_{3D}$** — trích nguyên văn (dòng 219–229):

```cpp
dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

// Off-diagonal elements appear twice --> double the gradient.
dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
```

$$
\frac{\partial L}{\partial \mathrm{Vrk}_{00}} = T_{00}^2\,dL\_da + T_{00}T_{10}\,dL\_db + T_{10}^2\,dL\_dc
$$

$$
\frac{\partial L}{\partial \mathrm{Vrk}_{01}} = 2T_{00}T_{01}\,dL\_da + (T_{00}T_{11}+T_{01}T_{10})\,dL\_db + 2T_{10}T_{11}\,dL\_dc
$$

(tương tự hoán vị chỉ số cho $\mathrm{Vrk}_{11},\mathrm{Vrk}_{22},\mathrm{Vrk}_{02},\mathrm{Vrk}_{12}$; phần tử off-diagonal nhân đôi vì xuất hiện 2 lần trong dạng toàn phương $\mathbf t^\top\mathrm{Vrk}\,\mathbf t$).

**Gradient ngược qua $T$** — trích nguyên văn (dòng 239–240):

```cpp
float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
    (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
```

$$
\frac{\partial L}{\partial T_{00}} = 2\big(T_{00}\mathrm{Vrk}_{00}+T_{01}\mathrm{Vrk}_{01}+T_{02}\mathrm{Vrk}_{02}\big)\,dL\_da + \big(T_{10}\mathrm{Vrk}_{00}+T_{11}\mathrm{Vrk}_{01}+T_{12}\mathrm{Vrk}_{02}\big)\,dL\_db
$$

(hệ số 2 cho $dL\_da$ vì $a$ là dạng toàn phương theo cột 0 của $T$; hệ số 1 cho $dL\_db$ vì $b$ song tuyến tính, không nhân đôi; tương tự cho $T_{01},T_{02},T_{10},T_{11},T_{12}$ — 6 phần tử khác 0 của $T$ do hàng cuối $J$ luôn 0).

**Gradient ngược qua $J$** (chỉ 4 phần tử khác 0: $J_{00},J_{02},J_{11},J_{12}$):

$$
\frac{\partial L}{\partial J_{00}} = \sum_k W_{0k}\,dL\_dT_{0k}, \qquad
\frac{\partial L}{\partial J_{02}} = \sum_k W_{2k}\,dL\_dT_{0k}
$$

$$
\frac{\partial L}{\partial J_{11}} = \sum_k W_{1k}\,dL\_dT_{1k}, \qquad
\frac{\partial L}{\partial J_{12}} = \sum_k W_{2k}\,dL\_dT_{1k}
$$

**Gradient ngược qua điểm camera-space $t$** — trích nguyên văn (dòng 264–266, kèm phần clamp mask dòng 174–178):

```cpp
const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;
...
float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
```

Từ $J_{00}=f_x/t_z,\ J_{02}=-f_xt_x/t_z^2,\ J_{11}=f_y/t_z,\ J_{12}=-f_yt_y/t_z^2$:

$$
\frac{\partial L}{\partial t_x} = -f_x\,t_z^{-2}\,\frac{\partial L}{\partial J_{02}}\cdot \mathbb{1}[\text{không clamp } x], \qquad
\frac{\partial L}{\partial t_y} = -f_y\,t_z^{-2}\,\frac{\partial L}{\partial J_{12}}\cdot \mathbb{1}[\text{không clamp } y]
$$

$$
\frac{\partial L}{\partial t_z} = -f_x\,t_z^{-2}\,\frac{\partial L}{\partial J_{00}} - f_y\,t_z^{-2}\,\frac{\partial L}{\partial J_{11}} + 2f_x t_x\,t_z^{-3}\,\frac{\partial L}{\partial J_{02}} + 2f_y t_y\,t_z^{-3}\,\frac{\partial L}{\partial J_{12}}
$$

(mask `x_grad_mul,y_grad_mul` $\in\{0,1\}$ triệt tiêu gradient khi $t_x/t_z,t_y/t_z$ đã bị clamp ở forward — straight-through, giống `torch.clamp`).

**Gradient ngược qua mean3D** — trích nguyên văn (dòng 270, 275):

```cpp
float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);
...
dL_dmeans[idx] = dL_dmean;
```

$$
\frac{\partial L}{\partial \mu}\Big|_{\text{qua }\Sigma'} = W_{view}^\top \begin{pmatrix}dL\_dt_x\\ dL\_dt_y\\ dL\_dt_z\end{pmatrix}
$$

(dùng transpose vì $t=W_{view}\mu$ tuyến tính; **ghi đè** `=` chứ không `+=` vì `computeCov2DCUDA` chạy **trước** `preprocessCUDA` — gradient từ SH và từ `mean2D` được cộng thêm sau đó).

---

## 4. Đạo hàm ngược qua $\Sigma_{3D}$ về lại $(s,q)$ — scale & rotation (`computeCov3D`, dòng 280–343)

Dựng ma trận gradient đối xứng đầy đủ từ 6 giá trị nén `dL_dcov3D` (nhân $\tfrac12$ cho phần tử off-diagonal vì mỗi giá trị nén đã gộp sẵn 2 đóng góp — xem mục 3):

$$
dL\_d\Sigma_{3D} = \begin{pmatrix}
dL\_d\mathrm{cov}_0 & \tfrac12 dL\_d\mathrm{cov}_1 & \tfrac12 dL\_d\mathrm{cov}_2\\
\tfrac12 dL\_d\mathrm{cov}_1 & dL\_d\mathrm{cov}_3 & \tfrac12 dL\_d\mathrm{cov}_4\\
\tfrac12 dL\_d\mathrm{cov}_2 & \tfrac12 dL\_d\mathrm{cov}_4 & dL\_d\mathrm{cov}_5
\end{pmatrix}
$$

**Qua $\Sigma_{3D}=M^\top M$** — trích nguyên văn (dòng 318):

```cpp
glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
```

$$
dL\_dM = 2\,M\cdot dL\_d\Sigma_{3D}
$$

(đạo hàm Fréchet $f(M)=M^\top M$ cho $df=(dM)^\top M+M^\top dM$, dùng tính đối xứng của $dL\_d\Sigma_{3D}$ để gộp hai số hạng).

**Qua $M=SR$ về $S$ (scale)** — trích nguyên văn (dòng 325–327):

```cpp
dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
```

$$
\frac{\partial L}{\partial s_k} = \big\langle R^\top_{k,:},\ (dL\_dM)^\top_{k,:}\big\rangle, \quad k\in\{x,y,z\}
$$

> **Lưu ý quan trọng (sửa diễn giải so với bản cũ):** biến `s = mod * scale` (dòng 297, `mod` = `scale_modifier`) được dùng để dựng $S=\mathrm{diag}(ms)$, và $M=SR$ dùng trực tiếp $S$ này — nên công thức trên cho ra $\partial L/\partial(ms_k)$, **không tự động** bằng $\partial L/\partial s_k$ (gradient của tham số `scale` thật sự dùng ở phía PyTorch), trừ khi $m=1$. Vì $\partial(ms_k)/\partial s_k=m$, nói chung cần nhân thêm $m$ để ra đúng $\partial L/\partial s_k$, nhưng **code không nhân $m$ ở bước này**. Trong thực tế $m\equiv1$ là giá trị phổ biến khi không áp dụng annealing scale, nên sai số này thường không xuất hiện, nhưng về tổng quát (khi gọi với `scale_modifier`$\ne1$) gradient trả về bị thiếu hệ số $m$. Đây **không phải điều bản cũ đã nêu đúng** — bản cũ từng viết "khớp vì độc lập hệ số mod", nhận định đó sai về logic đạo hàm.

**Qua $M=SR$ về $R$ rồi về quaternion** — trích nguyên văn (dòng 329–338):

```cpp
dL_dMt[0] *= s.x;
dL_dMt[1] *= s.y;
dL_dMt[2] *= s.z;

glm::vec4 dL_dq;
dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
```

$$
\begin{aligned}
\frac{\partial L}{\partial r} &= 2z(dL\_dM^\top_{01}-dL\_dM^\top_{10}) + 2y(dL\_dM^\top_{20}-dL\_dM^\top_{02}) + 2x(dL\_dM^\top_{12}-dL\_dM^\top_{21})\\
\frac{\partial L}{\partial x} &= 2y(dL\_dM^\top_{10}+dL\_dM^\top_{01}) + 2z(dL\_dM^\top_{20}+dL\_dM^\top_{02}) + 2r(dL\_dM^\top_{12}-dL\_dM^\top_{21}) - 4x(dL\_dM^\top_{22}+dL\_dM^\top_{11})\\
\frac{\partial L}{\partial y} &= 2x(dL\_dM^\top_{10}+dL\_dM^\top_{01}) + 2r(dL\_dM^\top_{20}-dL\_dM^\top_{02}) + 2z(dL\_dM^\top_{12}+dL\_dM^\top_{21}) - 4y(dL\_dM^\top_{22}+dL\_dM^\top_{00})\\
\frac{\partial L}{\partial z} &= 2r(dL\_dM^\top_{01}-dL\_dM^\top_{10}) + 2x(dL\_dM^\top_{20}+dL\_dM^\top_{02}) + 2y(dL\_dM^\top_{12}+dL\_dM^\top_{21}) - 4z(dL\_dM^\top_{11}+dL\_dM^\top_{00})
\end{aligned}
$$

(đạo hàm trực tiếp từng số hạng đa thức bậc 2 của $R(q)$, gộp theo biến; `dL_dMt` ở bước này đã nhân $S$ ở trên nên chính là $dL\_dR^\top$).

**Không chuẩn hoá lại quaternion** — trích nguyên văn (dòng 342, và `forward.cu` dòng 144):

```cpp
*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
```

```cpp
// forward.cu, computeCov3D:
glm::vec4 q = rot;// / glm::length(rot);
```

Lời gọi `dnormvdv(rot, dL_dq)` (lẽ ra cần nếu $q$ được chuẩn hoá trước khi dùng) bị **comment out** ở cả forward lẫn backward — nhất quán: cả hai giả định quaternion đầu vào **đã chuẩn hoá sẵn ở phía Python**. Không phải lỗi.

---

## 5. Đạo hàm ngược của SH color theo hướng nhìn và về hệ số SH (`computeColorFromSH`, dòng 20–141)

Trích nguyên văn mask clamp màu (dòng 32–35):

```cpp
glm::vec3 dL_dRGB = dL_dcolor[idx];
dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;
```

$$
dL\_dRGB_k \leftarrow dL\_dRGB_k \cdot \mathbb{1}[\neg \mathrm{clamped}_k]
$$

(straight-through giống `torch.clamp(x,min=0)`).

**Gradient về hệ số SH** — trích nguyên văn bậc 0 và bậc 1 (dòng 49–58):

```cpp
float dRGBdsh0 = SH_C0;
dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
if (deg > 0)
{
    float dRGBdsh1 = -SH_C1 * y;
    float dRGBdsh2 = SH_C1 * z;
    float dRGBdsh3 = -SH_C1 * x;
    dL_dsh[0] = dRGBdsh1 * dL_dRGB;
    dL_dsh[1] = dRGBdsh2 * dL_dRGB;
    dL_dsh[2] = dRGBdsh3 * dL_dRGB;
```

Vì $\mathbf c=\sum_\ell f_\ell(\mathbf d)\,\mathrm{sh}_\ell$ tuyến tính theo mỗi $\mathrm{sh}_\ell\in\mathbb R^3$:

$$
\frac{\partial L}{\partial \mathrm{sh}_\ell} = f_\ell(\mathbf d)\cdot dL\_dRGB
$$

với bậc 1: $f_0=-C_1y,\ f_1=C_1z,\ f_2=-C_1x$ (bậc 2, 3 tương tự cơ học, dòng 69–99 trong code, không suy lại chi tiết vì lặp lại cùng quy tắc).

**Gradient về hướng nhìn** — trích nguyên văn (dòng 60–62, 132):

```cpp
dRGBdx = -SH_C1 * sh[2];
dRGBdy = -SH_C1 * sh[0];
dRGBdz = SH_C1 * sh[1];
...
glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
```

$$
\frac{\partial L}{\partial x} = \langle dRGBdx,\, dL\_dRGB\rangle,\quad
\frac{\partial L}{\partial y} = \langle dRGBdy,\, dL\_dRGB\rangle,\quad
\frac{\partial L}{\partial z} = \langle dRGBdz,\, dL\_dRGB\rangle
$$

**Lan truyền qua chuẩn hoá hướng nhìn** $\mathbf d=\mathbf d_{orig}/\|\mathbf d_{orig}\|$ — trích nguyên văn (dòng 24–25, 135, 140):

```cpp
glm::vec3 dir_orig = pos - campos;
glm::vec3 dir = dir_orig / glm::length(dir_orig);
...
float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });
...
dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
```

$$
\frac{\partial L}{\partial \mathbf d_{orig}} = \frac{(\mathrm{sum2}\, I - \mathbf d_{orig}\mathbf d_{orig}^\top)\, \nabla_{\mathbf d} L}{\mathrm{sum2}^{3/2}}, \qquad \mathrm{sum2}=\|\mathbf d_{orig}\|^2
$$

(hàm `dnormvdv` trong `auxiliary.h` cài chính xác công thức đạo hàm chuẩn hoá vector này; vì $\mathbf d_{orig}=\mu_i-\mathbf{campos}$ tuyến tính hệ số $+1$ theo $\mu_i$, gradient này cộng thẳng `+=` vào `dL_dmeans[idx]` — nguồn gradient vị trí thứ hai, sau nguồn qua $\Sigma'$ ở mục 3).

---

## 6. Gradient còn lại của `preprocessCUDA` — chain rule mean2D → mean3D (dòng 348–400)

Trích nguyên văn (dòng 385, 391):

```cpp
dL_dmeans[idx].x += (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
...
dL_dmeans[idx] += dL_dmean;
```

Từ $p_{hom}=P\mu$, $p_w=1/(p_{hom,w}+\epsilon)$, $\mathrm{mean2D}=(p_{hom,x}p_w,\,p_{hom,y}p_w)$, đặt $\mathrm{mul}_1=p_{hom,x}p_w^2,\ \mathrm{mul}_2=p_{hom,y}p_w^2$:

$$
\frac{\partial L}{\partial \mu_x} = (P_{00}p_w - P_{03}\mathrm{mul}_1)\,dL\_d\mathrm{mean2D}_x + (P_{01}p_w-P_{03}\mathrm{mul}_2)\,dL\_d\mathrm{mean2D}_y
$$

(tương tự cho $\mu_y,\mu_z$ dùng hàng tương ứng của $P$; đây là **nguồn gradient vị trí thứ ba**, `+=` cộng dồn — tổng 3 nguồn độc lập vì $\mu$ ảnh hưởng $L$ qua 3 đường: $\Sigma'\to\mathrm{conic}$, SH$\to$color, và trực tiếp qua `mean2D`$\to\alpha$).

---

## 7. Ghi chú: không có gradient `eta`/multiview trong file này

Không có biến hay phép tính nào tên `eta`, `accum_eta` trong `backward.cu`/`backward.h`. Hai kernel ở đây chỉ sinh gradient chuẩn của pipeline rasterize: `dL_dmean2D` (gồm 2 kênh phụ "abs" ở `.z,.w`, mục 1.1), `dL_dconic2D`, `dL_dopacity`, `dL_dcolors`, `dL_dmeans`, `dL_dcov3D`, `dL_dscale`, `dL_drot`, `dL_ddc`, `dL_dsh`. Tín hiệu $\eta$ (năng lượng tần số, xem `MATH/BOOK/eta.md`) được tích luỹ ở một khâu Python riêng, ngoài phạm vi 2 file CUDA này.

---

## 8. Kiến thức toán nền tảng

- **Chain rule nhiều biến**: $dL\_d\mathbf x=J_{\mathbf y}(\mathbf x)^\top\,dL\_d\mathbf y$. Khi một biến ảnh hưởng $L$ qua nhiều đường (ví dụ $\mu$ qua $\Sigma'$, SH, `mean2D`), gradient tổng là **tổng** các đóng góp.
- **Dạng toàn phương/song tuyến**: $\partial(\mathbf x^\top A\mathbf x)/\partial\mathbf x=(A+A^\top)\mathbf x=2A\mathbf x$ nếu $A$ đối xứng; $\partial(\mathbf x^\top A\mathbf y)/\partial\mathbf x=A\mathbf y$ (song tuyến, không nhân đôi).
- **Đạo hàm Fréchet** của $f(M)=M^\top M$: $df=(dM)^\top M+M^\top dM$, dẫn tới $dL\_dM=2MG$ với $G=dL\_d\Sigma_{3D}$ đối xứng.
- **Hàm vector tuyến tính** $\mathbf y=A\mathbf x$: $dL\_d\mathbf x=A^\top dL\_d\mathbf y$ (dùng cho $t=W_{view}\mu\Rightarrow dL\_d\mu=W_{view}^\top dL\_dt$).
- **Đạo hàm nghịch đảo ma trận**: từ $M^{-1}M=I$, $d(M^{-1})=-M^{-1}(dM)M^{-1}$.
- **Đạo hàm chuẩn hoá vector** $\mathbf d=\mathbf v/\|\mathbf v\|$: $\partial L/\partial\mathbf v = \big[(\mathrm{sum2}\,I-\mathbf v\mathbf v^\top)\mathbf g\big]/\mathrm{sum2}^{3/2}$.
- **Backpropagation = reverse-mode AD**: toàn bộ `backward.cu` là cài đặt tay (không autograd) của thuật toán lan truyền ngược trên đồ thị tính toán $\mu,s,q,\mathrm{sh}\to\Sigma_{3D}\to\Sigma'\to\mathrm{conic}\to\alpha\to C\to L$ (và $\mu\to\mathrm{dir}\to$ SH color; $\mu\to\mathrm{mean2D}\to\alpha$).

---

## 9. Kiểm chứng tính đúng sai — tóm tắt (chi tiết đầy đủ ở `MATH/submodules/diff-gaussian-rasterization_structgs/backward.md` mục 9–10)

Công thức $\partial L/\partial b$ (mục 3) được tự suy lại bằng hai cách — (a) đạo hàm riêng "ngây thơ" của $N_{01}=-b/\mathrm{denom}$ (chỉ coi $b$ ở một vị trí), (b) đạo hàm ma trận đầy đủ với ràng buộc đối xứng $M_{01}=M_{10}=b$ — và xác nhận bằng finite-difference numeric gradient check tại $a=2,b=1,c=3$ ($\mathrm{denom}=5$):

- Phép (a) cho $\partial N_{01}/\partial b=-0.28$, khớp finite-difference của **riêng một phần tử** $N_{01}(b)=-b/(6-b^2)$.
- Nhưng $\partial L/\partial b$ mà code cần là tổng gradient qua **cả** $N_{01}$ và $N_{10}$ (ràng buộc bằng nhau): $2\times(-0.28)=-0.56$ — **khớp chính xác** với công thức code `denom2inv*2*(...)` tại cùng điểm số ($g_x=g_z=0,g_y=1$).

**Kết luận: công thức `dL_db` trong code là ĐÚNG**, hệ số 2 không phải tuỳ tiện mà là hệ quả của $b$ ảnh hưởng đồng thời hai vị trí đối xứng của ma trận nghịch đảo.

---

## 10. Ví dụ số

Lấy $a=2,b=1,c=3\Rightarrow\mathrm{denom}=ac-b^2=5$. Với $dL\_d\mathrm{conic}=(g_x,g_y,g_z)=(1,0,0)$:

$$
dL\_da = \frac{1}{25}(-9\cdot1+0+0) = -0.36,\qquad
dL\_db = \frac{2}{25}(3\cdot1+0+0) = 0.24,\qquad
dL\_dc = \frac{1}{25}(0+0+(5-6)\cdot1) = -0.04
$$

(kiểm tra bằng đạo hàm giải tích trực tiếp $\mathrm{conic}.x(b)=c/(ac-b^2)$ tại $a,c$ cố định, $g_x=1$: $\partial\mathrm{conic}.x/\partial a = -c^2/\mathrm{denom}^2=-9/25=-0.36$ — khớp $dL\_da$ ở trên).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX | Trạng thái |
|---|---|---|
| `T = T/(1.f-alpha)` | $T_{i-1} = T_i/(1-\alpha_i)$ | đúng |
| `accum_rec[ch] = last_alpha*last_color[ch] + (1-last_alpha)*accum_rec[ch]` | $S_i = \alpha_{i+1}c_{i+1} + (1-\alpha_{i+1})S_{i+1}$ | đúng |
| `dL_dalpha += (c - accum_rec[ch]) * dL_dchannel` rồi `dL_dalpha *= T` | $\partial L/\partial\alpha_i = T_{i-1}(c_i-S_i)\,dL/dC$ | đúng |
| `dL_dalpha += (-T_final/(1-alpha)) * bg_dot_dpixel` | đóng góp nền qua $T_{final}$ | đúng |
| `atomicAdd(dL_dcolors, dchannel_dcolor*dL_dchannel)` | $\partial L/\partial c_{i,ch}=\alpha_iT_{i-1}\,\partial L/\partial C_{ch}$ | đúng |
| `Register_dL_dmean2D_z/w += fabsf(...)` (PerGaussianRenderCUDA) | gradient "abs" phụ cho tiêu chí split StructGS | đúng (đặc thù StructGS, không phải $\partial L/\partial\mu$ chuẩn) |
| `dL_dG = con_o.w * dL_dalpha` | $\partial L/\partial G = o\cdot\partial L/\partial\alpha$ | đúng, nhưng $o=$opacity×`coef` — xem mục 2.1 |
| `dL_dopacity += G*dL_dalpha` | $\partial L/\partial o = G\cdot\partial L/\partial\alpha$ | **thiếu gradient qua hệ số bù `coef`** — khả năng là chủ ý detach |
| `a=cov2D[0][0]+=0.3f; c=cov2D[1][1]+=0.3f` | $\kappa=0.3$ trong backward | **SAI — lệch với `kernel_size=0.1f` ở forward.cu, cần sửa 0.3→0.1** |
| `dL_db = denom2inv*2*(bc\,g_x-(denom+2b²)g_y+ab\,g_z)` | $\partial L/\partial b$ | **đúng** — đã kiểm chứng bằng ma trận đầy đủ + finite-difference (mục 9) |
| `dL_dM = 2.0f*M*dL_dSigma` | $dL\_dM = 2M\cdot dL\_d\Sigma_{3D}$ | đúng |
| `dL_dscale->x = dot(Rt[0], dL_dMt[0])` | $\partial L/\partial(ms_x)$ | nhãn đúng công thức nhưng là đạo hàm theo $ms_x$, chỉ bằng $\partial L/\partial s_x$ khi $m=1$ |
| `*dL_drot = {dL_dq...}` (dnormvdv comment out) | không lan truyền qua chuẩn hoá quaternion | đúng — nhất quán với forward |
| `dL_dsh[0] = -SH_C1*y*dL_dRGB` | $\partial L/\partial\mathrm{sh}_0 = -C_1y\cdot dL\_dRGB$ | đúng |
| `dL_dmean = dnormvdv(dir_orig, dL_ddir)` | lan truyền qua chuẩn hoá hướng nhìn | đúng |
| `dL_dmeans[idx].x += (P00*m_w-P03*mul1)*dL_dmean2D.x + ...` | $\partial L/\partial\mu_x$ qua phép chia phối cảnh | đúng |

### Các điểm đã sửa/bổ sung so với bản cũ của chính file này

1. **Bổ sung mọi code trích dẫn nguyên văn kèm số dòng** trước từng công thức (yêu cầu định dạng mới).
2. **Bổ sung phát hiện mới** (không có ở bản cũ): `con_o.w` là `opacity * coef` (hệ số bù khử-alias), backward không lan truyền gradient qua `coef` — mục 2.1.
3. **Giữ nguyên, xác nhận lại**: lệch hằng số dilation $\kappa=0.3$ (backward) vs `kernel_size=0.1` (forward) — bug thật, cần sửa.
4. **Sửa diễn giải sai của bản cũ** về gradient scale: công thức cho $\partial L/\partial(ms_k)$, không tự động là $\partial L/\partial s_k$ trừ khi $m=1$.
5. **Bổ sung kiểm chứng chi tiết** cho hệ số 2 trong `dL_db` bằng đạo hàm ma trận đầy đủ + finite-difference — xác nhận công thức code đúng (xem bản đầy đủ ở `MATH/submodules/.../backward.md` mục 9–10 nếu cần xem lại toàn bộ quá trình truy vết).
