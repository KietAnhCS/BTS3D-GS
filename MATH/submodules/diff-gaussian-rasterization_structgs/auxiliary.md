# Công thức toán học trong `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/auxiliary.h`

File nguồn: `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/auxiliary.h` (393 dòng). Đây là header chứa các hàm `__device__ inline`/`__forceinline__` dùng chung bởi `forward.cu`, `backward.cu`, `rasterizer_impl.cu`: hằng số Spherical Harmonics, các phép biến đổi toạ độ thuần nhất, đạo hàm của chuẩn hoá vector, sigmoid, kiểm tra frustum, và thuật toán xác định tile bị một Gaussian phủ (AccuTile/SNUGBox).

**Lưu ý quan trọng ngay từ đầu (chỉnh lại giả định của đề bài):** `computeCov3D` và `computeCov2D` — phép dựng $\Sigma_{3D}=R S S^\top R^\top$ từ quaternion/scale, và phép chiếu EWA $\Sigma'=JW\Sigma W^\top J^\top$ — **không nằm trong `auxiliary.h`**. Hai hàm này được định nghĩa trong `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/forward.cu`, dòng 79–130 (`computeCov2D`) và dòng 135–169 (`computeCov3D`). `auxiliary.h` chỉ cung cấp các hàm nguyên thuỷ mà `computeCov2D` gọi lại (`transformPoint4x3`, dòng 71–79 của `auxiliary.h`). Vì đề bài yêu cầu kiểm chứng kỹ hai hàm này (bao gồm hằng số low-pass filter), tài liệu này vẫn trình bày đầy đủ ở §2.8–2.9 và §5 (Ví dụ số) — có trích rõ nguồn là `forward.cu`, không phải `auxiliary.h` — để không bỏ sót yêu cầu kiểm chứng, đồng thời sửa lại giả định sai trong đề bài (hằng số low-pass filter **không phải $0.3$** như suy đoán ban đầu — xem §4).

---

## 1. Ký hiệu toán học

- $M$: ma trận $4\times4$ lưu dạng mảng phẳng `matrix[0..15]`, **column-major** (cột $j$ nằm ở `matrix[4j..4j+3]`). Khi nhân với điểm $p=(x,y,z)$, các hàm `transformPoint*` tính $M^\top$ tác động lên $p$ theo kiểu "hàng $i$ của kết quả $=$ cột $i$ của $M$ $\cdot\,p$" — tương đương $p' = M^\top \tilde p$ nếu coi $M$ là ma trận biến đổi chuẩn (xem §2.4).
- $p_{orig}=(x,y,z)$: toạ độ thế giới (world-space) của tâm Gaussian.
- $p_{view}=(t_x,t_y,t_z)$: toạ độ camera-space (sau `viewmatrix`).
- $p_{hom}=(x',y',z',w')$: toạ độ thuần nhất sau `projmatrix` (clip space).
- $p_{proj}$: toạ độ NDC sau chia $w'$.
- $\Sigma_{3D}\in\mathbb R^{3\times3}$: ma trận hiệp phương sai thế giới của Gaussian (đối xứng, lưu 6 số `cov3D[0..5]`).
- $\Sigma_{2D}\in\mathbb R^{2\times2}$: ma trận hiệp phương sai sau chiếu lên ảnh (`float4` `(a,b,c,coef)` với $\Sigma_{2D}=\begin{pmatrix}a&b\\b&c\end{pmatrix}$).
- $\text{con\_o}=(A,B,C,\alpha)$: hệ số conic $= \Sigma_{2D}^{-1}$ cộng kênh opacity $\alpha$, dùng trong §2.9.
- $q=(r,x,y,z)$: quaternion đơn vị (không nhất thiết đã chuẩn hoá trong code, xem §4), biểu diễn phép quay $R(q)\in SO(3)$.
- $S=\mathrm{diag}(s_x,s_y,s_z)$: ma trận co giãn (scale) đường chéo.
- $f_x,f_y$: tiêu cự (focal length) tính bằng pixel; $\tan\!\text{fovx},\tan\!\text{fovy}$: nửa góc nhìn.
- $C_0,C_1,C_2[\cdot],C_3[\cdot]$: hằng số chuẩn hoá hàm cầu điều hoà thực (real Spherical Harmonics).
- $\text{BLOCK\_X},\text{BLOCK\_Y}$: kích thước tile rasterization (từ `config.h`).
- $\mathrm{grid}=(\mathrm{grid}_x,\mathrm{grid}_y)$: số tile theo mỗi chiều.
- $\mathbb 1[\cdot]$: hàm chỉ thị (indicator).

---

## 2. Suy công thức từng hàm (bám sát `auxiliary.h`)

### 2.1. Hằng số chuẩn hoá Spherical Harmonics (dòng 22–40)

```cpp
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = { 1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f };
__device__ const float SH_C3[] = { -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f };
```

Đây là các hệ số chuẩn hoá của hàm cầu điều hoà thực (real SH) bậc $\ell=0,1,2,3$, định nghĩa sao cho (quy ước chuẩn trong đồ hoạ máy tính, giống `utils/sh_utils.py` phía Python):

$$
C_0=\sqrt{\frac{1}{4\pi}},\qquad C_1=\sqrt{\frac{3}{4\pi}}
$$

Kiểm tra số: $\sqrt{1/(4\pi)}=0.2820947917738782\ldots$ — khớp đến 16 chữ số với `SH_C0`. $\sqrt{3/(4\pi)}=0.4886025119029199\ldots$ — khớp chính xác với `SH_C1`.

$C_2$ (5 hệ số ứng với $m=-2,\ldots,2$ của $\ell=2$) và $C_3$ (7 hệ số của $\ell=3$) là các hằng số chuẩn (xem §3.4) nhân với đa thức toạ độ phương $(x,y,z)$ đã chuẩn hoá, ví dụ số hạng bậc 2 đầu tiên dùng trong `computeColorFromSH` (ở `forward.cu`, không phải trong file này) có dạng $C_2[0]\,xy$. File `auxiliary.h` **chỉ chứa 4 bậc này** ($\ell=0..3$), không có $C_4$ — đúng như bản cũ đã ghi nhận.

### 2.2. `ndc2Pix` (dòng 42–45)

```cpp
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
    return ((v + 1.0) * S - 1.0) * 0.5;
}
```

$$
\boxed{\ \text{pix}(v,S)=\frac{(v+1)\,S-1}{2}\ },\qquad v\in[-1,1]\ (\text{NDC}),\ S\in\{W,H\}
$$

Đây là ánh xạ affine chuẩn NDC $\to$ pixel với quy ước **pixel-center sampling** (pixel $0$ có tâm tại toạ độ liên tục $0$, không phải $-0.5$): tại $v=-1\Rightarrow \text{pix}=-0.5$; tại $v=1\Rightarrow\text{pix}=S-0.5$. Khoảng ảnh $[-0.5,S-0.5]$ chia đều cho $S$ pixel, mỗi pixel rộng 1, đúng quy ước pixel-center ($-1$ ở tử số, không phải $-1/S$) khớp với cách OpenGL/PyTorch3D định nghĩa NDC half-pixel offset.

### 2.3. `getRect` — hình chữ nhật tile bao quanh (dòng 47–69, 2 overload)

Overload 1 (bán kính tròn `max_radius`), dòng 47–57:

$$
\text{rect}_{min}=\Big(\mathrm{clip}\big(\lfloor (p_x-r)/B_X\rfloor,0,\mathrm{grid}_x\big),\ \mathrm{clip}\big(\lfloor (p_y-r)/B_Y\rfloor,0,\mathrm{grid}_y\big)\Big)
$$
$$
\text{rect}_{max}=\Big(\mathrm{clip}\big(\lfloor (p_x+r+B_X-1)/B_X\rfloor,0,\mathrm{grid}_x\big),\ \mathrm{clip}\big(\lfloor (p_y+r+B_Y-1)/B_Y\rfloor,0,\mathrm{grid}_y\big)\Big)
$$

với $r=\text{max\_radius}$, $B_X=\text{BLOCK\_X}$, $B_Y=\text{BLOCK\_Y}$. Số hạng $+B_X-1$ trong $\text{rect}_{max}$ là làm tròn lên (ceiling-division nguyên): $\lceil n/B_X\rceil=\lfloor (n+B_X-1)/B_X\rfloor$.

Overload 2 (dòng 59–69) giống hệt nhưng dùng `ext_rect`$=(r_x,r_y)$ thay vì bán kính tròn đơn — tức hình chữ nhật bao (không phải hình tròn):

$$
\text{rect}_{min}=\Big(\mathrm{clip}\big(\lfloor (p_x-r_x)/B_X\rfloor,0,\mathrm{grid}_x\big),\ \ldots\Big),\quad \text{rect}_{max}=\Big(\mathrm{clip}\big(\lfloor (p_x+r_x+B_X-1)/B_X\rfloor,0,\mathrm{grid}_x\big),\ \ldots\Big)
$$

### 2.4. Các phép biến đổi toạ độ thuần nhất (dòng 71–110)

`matrix` là mảng 16 phần tử **column-major** của ma trận $4\times4$ $M$ (cột $j$: `matrix[4j]..matrix[4j+3]`). Biểu thức CUDA

```cpp
matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z + matrix[12]
```

lấy phần tử `matrix[0]=M[0][0]`, `matrix[4]=M[0][1]` (cột 1, hàng 0 — vì column-major nên chỉ số phẳng $4\cdot\text{col}+\text{row}$, ở đây hàng $=0$ cho cả 3 số hạng đầu), `matrix[8]=M[0][2]`, `matrix[12]=M[0][3]`. Tức là dòng này tính **hàng 0** của $M$ nhân với $(x,y,z,1)^\top$. Nói cách khác with quy ước toán học thông thường ($M$ là ma trận biến đổi, điểm là vector cột), các hàm này tính đúng:

$$
\text{transformPoint4x3}(p)=\big(M_{0,:}\cdot\tilde p,\ M_{1,:}\cdot\tilde p,\ M_{2,:}\cdot\tilde p\big),\qquad \tilde p=(x,y,z,1)^\top
$$

(bỏ hàng $w$, dùng cho world→camera, giả định $w=1$ không cần chia lại — dòng 71–79).

$$
\text{transformPoint4x4}(p)=(x',y',z',w')=M\,\tilde p
$$

(dùng cho chiếu phối cảnh đầy đủ, dòng 81–90; cần chia $w'$ ở nơi gọi).

$$
\text{transformVec4x3}(p)=M_{3\times3}\,p\qquad\text{(bỏ phần tịnh tiến — dùng cho vector, dòng 92–100)}
$$

$$
\text{transformVec4x3Transpose}(p)=M_{3\times3}^\top\,p\qquad\text{(dòng 102–110)}
$$

Ở `transformVec4x3Transpose`, biểu thức `matrix[0]*p.x+matrix[1]*p.y+matrix[2]*p.z` lấy 3 phần tử **liên tiếp trong bộ nhớ** (tức cột 0 đầy đủ của $M$, vì column-major $\{matrix[0],matrix[1],matrix[2]\}=\{M_{00},M_{10},M_{20}\}$) — đúng là hàng 0 của $M^\top$, xác nhận công thức $M^\top p$.

### 2.5. `dnormvdz`, `dnormvdv` — đạo hàm của chuẩn hoá vector (dòng 112–145)

Cho $v\in\mathbb R^3$, $\hat v=v/\lVert v\rVert$, $s=\lVert v\rVert^2=v_x^2+v_y^2+v_z^2$. Đạo hàm chuẩn của phép chuẩn hoá:

$$
\frac{\partial \hat v_k}{\partial v_j}=\frac{s\,\delta_{kj}-v_kv_j}{s^{3/2}}
$$

Các hàm này tính **vector-Jacobian product** ngược $dv\mapsto \big(\partial\hat v/\partial v\big)^\top\! dv$ (nhận gradient theo $\hat v$, trả gradient theo $v$) — ký hiệu $dv$ ở đây đóng vai trò gradient đến, không phải vi phân tiến:

$$
\text{dnormvdv}_j=\sum_k \frac{\partial \hat v_k}{\partial v_j}\,dv_k=\frac{(s-v_j^2)\,dv_j-v_j\sum_{k\ne j}v_k\,dv_k}{s^{3/2}}
$$

khớp dòng 126–128:
```cpp
dnormvdv.x = ((+sum2 - v.x*v.x)*dv.x - v.y*v.x*dv.y - v.z*v.x*dv.z) * invsum32;
```

`dnormvdz` (dòng 112–118) chỉ trả **thành phần $z$**:

$$
\text{dnormvdz}=\frac{-v_xv_z\,dv_x-v_yv_z\,dv_y+(s-v_z^2)\,dv_z}{s^{3/2}}
$$

Bản `dnormvdv(float4,float4)` (dòng 132–145) mở rộng cho $\mathbb R^4$ (dùng cho gradient chuẩn hoá quaternion $q\to\hat q$), $s=\sum_{k\in\{x,y,z,w\}}v_k^2$, $\text{vdv}_k=v_k\,dv_k$:

$$
\text{dnormvdv}_k=\frac{(s-v_k^2)\,dv_k-v_k\big(\sum_j \text{vdv}_j-\text{vdv}_k\big)}{s^{3/2}}
$$

khớp chính xác dòng 140–143 của code.

### 2.6. `sigmoid` (dòng 147–150)

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

### 2.7. `in_frustum` — kiểm tra điểm trong frustum (dòng 152–177)

```cpp
float4 p_hom = transformPoint4x4(p_orig, projmatrix);
float p_w = 1.0f / (p_hom.w + 0.0000001f);
float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
p_view = transformPoint4x3(p_orig, viewmatrix);
if (p_view.z <= 0.2f) { ... return false; }
return true;
```

$$
(x',y',z',w')=\text{transformPoint4x4}(p_{orig},\text{projmatrix}),\qquad w_{inv}=\frac{1}{w'+10^{-7}}
$$
$$
p_{proj}=(x'w_{inv},\,y'w_{inv},\,z'w_{inv}),\qquad p_{view}=\text{transformPoint4x3}(p_{orig},\text{viewmatrix})
$$

Điều kiện loại bỏ (near-plane culling **duy nhất**, dòng 167):

$$
\boxed{\ \text{visible}=\big(p_{view,z}>0.2\big)\ }
$$

Dòng kiểm tra biên NDC $p_{proj}\in[-1.3,1.3]^2$ **bị comment** (`// || ((p_proj.x < -1.3 ...`) — tức hiện tại hàm **không** làm frustum culling theo $x,y$, chỉ cắt theo near-plane $z>0.2$. $p_{proj}$ được tính nhưng không dùng ở điều kiện cuối. Đây đúng là hành vi thực của code, trùng khớp hoàn toàn với bản `MATH/cuda/auxiliary.md` cũ.

### 2.8. (Tham chiếu ngoài `auxiliary.h`) `computeCov3D` — `forward.cu` dòng 135–169

Không thuộc `auxiliary.h`, nhưng dùng trực tiếp `S`, `R(q)`; trình bày ở đây để phục vụ Ví dụ số §5 và vì đề bài yêu cầu kiểm chứng.

```cpp
glm::mat3 S = glm::mat3(1.0f);
S[0][0] = mod * scale.x; S[1][1] = mod * scale.y; S[2][2] = mod * scale.z;

glm::vec4 q = rot; // / glm::length(rot);   <-- chuẩn hoá bị COMMENT OUT
float r = q.x; float x = q.y; float y = q.z; float z = q.w;

glm::mat3 R = glm::mat3(
    1.f - 2.f*(y*y+z*z), 2.f*(x*y-r*z),     2.f*(x*z+r*y),
    2.f*(x*y+r*z),       1.f - 2.f*(x*x+z*z), 2.f*(y*z-r*x),
    2.f*(x*z-r*y),       2.f*(y*z+r*x),     1.f - 2.f*(x*x+y*y)
);
glm::mat3 M = S * R;
glm::mat3 Sigma = glm::transpose(M) * M;
cov3D[0]=Sigma[0][0]; cov3D[1]=Sigma[0][1]; cov3D[2]=Sigma[0][2];
cov3D[3]=Sigma[1][1]; cov3D[4]=Sigma[1][2]; cov3D[5]=Sigma[2][2];
```

**Chú ý quan trọng — quy ước thành phần quaternion:** biến `rot` (`glm::vec4 rot`) được gán $r=q.x,\ x=q.y,\ y=q.z,\ z=q.w$ — nghĩa là **thành phần đầu của `vec4`(`.x`) là phần thực $r$ (w của quaternion toán học)**, còn 3 thành phần ảo nằm ở `.y,.z,.w`. Đây là quy ước `(w,x,y,z)` chứ không phải `(x,y,z,w)` chuẩn — cần truyền `rotations` theo đúng thứ tự này từ phía Python (`gaussian_model.py`), nếu không rotation sẽ sai.

**Chuẩn hoá quaternion bị tắt** (dòng `glm::vec4 q = rot; // / glm::length(rot);`) — tức $R(q)$ chỉ **thực sự là ma trận trực giao** nếu `rotations` đưa vào đã được chuẩn hoá từ trước (ở phía Python, `gaussian_model.py` dùng `torch.nn.functional.normalize` trước khi truyền cho CUDA — không kiểm chứng trong phạm vi file này).

$$
R(q)=\begin{pmatrix}
1-2(y^2+z^2) & 2(xy-rz) & 2(xz+ry)\\
2(xy+rz) & 1-2(x^2+z^2) & 2(yz-rx)\\
2(xz-ry) & 2(yz+rx) & 1-2(x^2+y^2)
\end{pmatrix},\qquad
S=\mathrm{diag}(m\,s_x,\ m\,s_y,\ m\,s_z)
$$

($m=$ `scale_modifier`). Với $M=SR$ (không phải $RS$) và $\Sigma_{3D}=M^\top M$:

$$
\boxed{\ \Sigma_{3D}=M^\top M=(SR)^\top(SR)=R^\top S^\top S R=R^\top S^2 R\ }
$$

(vì $S$ đối xứng, $S^\top S=S^2$). Công thức **lý thuyết chuẩn** của ellipsoid-covariance là $\Sigma=R S^2 R^\top$ (quay $S^2$ bằng $R$ **theo chiều thuận**, không phải $R^\top S^2 R$). Đây **không phải bug** — do quy ước truy cập `glm::mat3` column-major kết hợp cách `computeCov2D` cũng dựng lại $W$ bằng cách đọc `viewmatrix` theo kiểu "transpose" (xem §2.9), toàn bộ pipeline dùng nhất quán quy ước $A^\top$ thay cho $A$ ở mọi nơi, nên kết quả cuối (sau khi nhân với $W,J$ ở `computeCov2D`) vẫn đúng về mặt hình học. Đây là hành vi **giống hệt repo gốc Inria 3DGS** (`diff-gaussian-rasterization`), không phải lỗi riêng của `structgs`. Xem minh hoạ số cụ thể ở §5.

### 2.9. (Tham chiếu ngoài `auxiliary.h`) `computeCov2D` — `forward.cu` dòng 79–130

```cpp
float3 t = transformPoint4x3(mean, viewmatrix);     // auxiliary.h, dòng 71
const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
const float txtz = t.x / t.z, tytz = t.y / t.z;
t.x = min(limx, max(-limx, txtz)) * t.z;
t.y = min(limy, max(-limy, tytz)) * t.z;

glm::mat3 J = glm::mat3(
    focal_x/t.z, 0, -(focal_x*t.x)/(t.z*t.z),
    0, focal_y/t.z, -(focal_y*t.y)/(t.z*t.z),
    0, 0, 0);
glm::mat3 W = glm::mat3(
    viewmatrix[0], viewmatrix[4], viewmatrix[8],
    viewmatrix[1], viewmatrix[5], viewmatrix[9],
    viewmatrix[2], viewmatrix[6], viewmatrix[10]);
glm::mat3 T = W * J;
glm::mat3 Vrk = glm::mat3(cov3D[0],cov3D[1],cov3D[2], cov3D[1],cov3D[3],cov3D[4], cov3D[2],cov3D[4],cov3D[5]);
glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

float det_0 = max(1e-6f, cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1]);
float kernel_size = 0.1f;
cov[0][0] += kernel_size; cov[1][1] += kernel_size;
float det_1 = max(1e-6f, cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1]);
float coef = sqrt(det_0 / (det_1 + 1e-6f));
return { cov[0][0], cov[0][1], cov[1][1], coef };
```

**6.1. "Clamp" toạ độ camera-space (anti-shear clipping).** $t=\text{transformPoint4x3}(\text{mean},\text{viewmatrix})$ (gọi hàm ở §2.4). Giới hạn:

$$
\frac{t_x}{t_z},\frac{t_y}{t_z}\ \text{bị clip vào}\ [-1.3\tan\!\text{fovx},\,1.3\tan\!\text{fovx}]\times[-1.3\tan\!\text{fovy},\,1.3\tan\!\text{fovy}]
$$

rồi nhân lại với $t_z$ — đúng kỹ thuật "EWA Splatting" (Zwicker et al. 2002, mô tả ở comment dòng 81–83) để tránh Jacobian xấp xỉ sai khi Gaussian gần rìa frustum (mở rộng 30% ngoài $[-1,1]$ để không cắt cứng).

**6.2. Jacobian phép chiếu phối cảnh** $J$:

$$
J=\begin{pmatrix}\dfrac{f_x}{t_z}&0&-\dfrac{f_x t_x}{t_z^2}\\[4pt]0&\dfrac{f_y}{t_z}&-\dfrac{f_y t_y}{t_z^2}\\[2pt]0&0&0\end{pmatrix}
$$

Đây là đạo hàm của phép chiếu pinhole $(t_x,t_y,t_z)\mapsto (f_x t_x/t_z,\,f_y t_y/t_z)$ theo $(t_x,t_y,t_z)$, hàng thứ 3 $=0$ vì chỉ cần Jacobian của 2 toạ độ ảnh (xấp xỉ affine bậc nhất quanh $t$, đúng công thức (29) trong Zwicker et al. 2002 — xem §3.5).

**6.3. Ma trận quay view** $W$: lấy từ 3 cột đầu, 3 hàng đầu của `viewmatrix` theo đúng cách đọc "transpose" mô tả ở §2.4 — $W$ ở đây chính là phần quay $3\times3$ của ma trận view (world→camera), cùng quy ước transpose như `computeCov3D`.

**6.4. Ghép Jacobian và view:** $T=WJ$, rồi

$$
\Sigma_{2D}^{\text{raw}}=T^\top \Sigma_{3D}^\top\, T = T^\top \Sigma_{3D} T
$$

(vì $\Sigma_{3D}$ đối xứng). Do quy ước transpose nhất quán (§2.8), biểu thức này **tương đương về mặt toán học** với công thức chuẩn EWA splatting:

$$
\boxed{\ \Sigma'=J\,W\,\Sigma_{3D}\,W^\top J^\top\ }
$$

(chỉ lấy khối $2\times2$ trên-trái, bỏ hàng/cột thứ 3 vì hàng 3 của $J$ bằng 0) — đây chính là công thức (31) của Zwicker et al. 2002.

**6.5. Low-pass filter — ĐIỂM CẦN SỬA SO VỚI GIẢ ĐỊNH BAN ĐẦU.** Code **không** cộng $0.3\,I$ như trong bản 3DGS gốc nguyên thuỷ (Kerbl et al. 2023, `cov[0][0]+=0.3f; cov[1][1]+=0.3f;`, không có hệ số bù). Biến thể `structgs` này dùng kỹ thuật kiểu **Mip-Splatting** (Yu et al. 2024): cộng $\texttt{kernel\_size}=\mathbf{0.1}$ (không phải $0.3$) vào đường chéo, **và tính thêm hệ số bù suy giảm độ mờ (dilation compensation)** nhân vào alpha:

$$
\det_0=\max\!\big(10^{-6},\,ac-b^2\big)\quad(\Sigma_{2D}^{\text{raw}}=\begin{pmatrix}a&b\\b&c\end{pmatrix})
$$
$$
a\leftarrow a+0.1,\qquad c\leftarrow c+0.1
$$
$$
\det_1=\max\!\big(10^{-6},\,ac-b^2\big)\quad\text{(sau khi cộng kernel)}
$$
$$
\boxed{\ \text{coef}=\sqrt{\dfrac{\det_0}{\det_1+10^{-6}}}\ }
$$

Hàm trả về `float4` $=(a_{\text{new}},b,c_{\text{new}},\text{coef})$. Ở nơi gọi (`forward.cu`, không trong `auxiliary.h`), `coef` được nhân vào alpha hiệu dụng của splat để **bù lại** việc Gaussian bị "phồng" thêm bởi $0.1\,I$ — mục đích là đảm bảo mỗi Gaussian rộng tối thiểu khoảng 1 pixel để chống alias (aliasing) khi render ở độ phân giải thấp, trong khi vẫn giữ gần đúng năng lượng/diện tích tích phân ban đầu bằng hệ số bù $\sqrt{\det_0/\det_1}$ (tỉ lệ nghịch căn bậc hai diện tích ellipse, vì $\det\Sigma_{2D}\propto(\text{diện tích})^2$).

### 2.10. Hình học ellipse mức opacity & AccuTile/SNUGBox (dòng 179–393)

Thuật toán chi tiết (vòng lặp quét tile) đã mô tả ở `MATH/cuda/rasterizer_impl.md` mục 4; ở đây suy lại công thức cốt lõi bám theo code thật trong `auxiliary.h`.

**`computeEllipseIntersection`** (dòng 179–194): tìm giao điểm của biên ellipse mức ngưỡng $t$ (xem dưới) với đường thẳng $u=\text{coord}$ (hoặc $v=\text{coord}$ nếu `isY`). Biên ellipse ứng với $\text{con\_o}=(A,B,C,\alpha)=\Sigma_{2D}^{-1}$ kèm $\alpha$:

$$
A\,dx^2+2B\,dx\,dy+C\,dy^2=t
$$

Với $dx=h=\text{coord}-p_u$ cố định, giải phương trình bậc 2 theo $dy$: $C\,dy^2+2Bh\,dy+(Ah^2-t)=0$

$$
dy=\frac{-Bh\pm\sqrt{B^2h^2-C(Ah^2-t)}}{C}=\frac{-Bh\pm\sqrt{(B^2-AC)h^2+Ct}}{C}=\frac{-Bh\pm\sqrt{\text{disc}\cdot h^2+Ct}}{C}
$$

khớp chính xác code (dòng 187–193, với `coeff`$=C$ khi không `isY`, $=A$ khi `isY` do vai trò $u,v$ hoán đổi):

```cpp
float h = coord - p_u;
float sqrt_term = sqrt(disc * h * h + t * coeff);
return { (-con_o.y*h - sqrt_term)/coeff + p_v, (-con_o.y*h + sqrt_term)/coeff + p_v };
```

**`duplicateToTilesTouched`** (dòng 313–382) — ngưỡng và discriminant:

$$
\text{disc}=B^2-AC\qquad(\text{dòng 324: } \texttt{con\_o.y*con\_o.y - con\_o.x*con\_o.z})
$$

Ellipse hợp lệ (xác định dương) khi $A>0,\,C>0,\,\text{disc}<0$ (dòng 327). Ngưỡng năng lượng ứng với độ trong suốt $1/255$:

$$
t=\texttt{mult}\cdot 2\ln(255\,\alpha)\qquad(\text{dòng 332–333})
$$

Suy ra từ điều kiện biên: $\alpha\exp(-\tfrac12(Adx^2+2Bdxdy+Cdy^2))=\tfrac1{255}\iff Adx^2+2Bdxdy+Cdy^2=2\ln(255\alpha)$.

Điểm cực trị bbox theo trục chéo (dòng 335–341):

$$
x_{\text{term}}=\mathrm{sgn}(-B)\sqrt{\frac{-B^2t}{\text{disc}\cdot A}},\qquad y_{\text{term}}=\mathrm{sgn}(-B)\sqrt{\frac{-B^2t}{\text{disc}\cdot C}}
$$

(dấu lấy theo code: `(con_o.y < 0) ? x_term : -x_term`), dùng làm toạ độ $\arg\max/\arg\min$ của ellipse để xác định bounding box chính xác $\text{bbox}_{min},\text{bbox}_{max}$ qua `computeEllipseIntersection`, rồi `getRect`-kiểu chuyển sang chỉ số tile (dòng 352–360). Phần duyệt tile theo từng lát quét (`processTiles`, dòng 196–310) đã có đầy đủ ở `rasterizer_impl.md`.

---

## 3. Kiến thức toán nền tảng

### 3.1. Đại số tuyến tính cơ bản
- Ma trận đối xứng xác định dương $\Sigma$ có thể phân tích $\Sigma=R\Lambda R^\top$ (eigen-decomposition, $R$ trực giao, $\Lambda$ đường chéo không âm) — đây là cơ sở để hiểu $\Sigma_{3D}=R S^2 R^\top$ (ellipsoid: trục chính $=$ cột của $R$, bán trục $=\sqrt{\Lambda_{ii}}=s_i$).
- Với ma trận trực giao $R$ ($R^\top R=I$), $R^\top=R^{-1}$: quay ngược chiều.
- Dạng toàn phương (quadratic form) $x^\top A x$ với $A$ đối xứng xác định dương định nghĩa một ellipsoid $\{x: x^\top A x = t\}$; $\det A$ tỉ lệ nghịch với bình phương thể tích ellipsoid.
- Biến đổi tuyến tính của biến ngẫu nhiên Gaussian: nếu $X\sim\mathcal N(\mu,\Sigma)$ và $Y=AX+b$ thì $Y\sim\mathcal N(A\mu+b,\,A\Sigma A^\top)$ — chính là quy tắc "sandwich" $\Sigma'=A\Sigma A^\top$ dùng xuyên suốt (rotation-scale, rồi Jacobian chiếu).

### 3.2. Hình học chiếu phối cảnh (pinhole camera)
Mô hình pinhole: điểm camera-space $(t_x,t_y,t_z)$ chiếu lên ảnh qua

$$
u=f_x\frac{t_x}{t_z}+c_x,\qquad v=f_y\frac{t_y}{t_z}+c_y
$$

Đây là ánh xạ **phi tuyến** (chia cho $t_z$), nên một ellipsoid 3D không chiếu thành ellipse 2D chính xác — EWA splatting (§3.5) xấp xỉ **tuyến tính hoá cục bộ** bằng Jacobian $J=\partial(u,v)/\partial(t_x,t_y,t_z)$ tại điểm $t$:

$$
J=\begin{pmatrix}f_x/t_z & 0 & -f_xt_x/t_z^2\\0 & f_y/t_z & -f_yt_y/t_z^2\end{pmatrix}
$$

(đúng công thức ở §2.9 — đây là phép tính Taylor bậc nhất của phép chiếu phối cảnh quanh tâm Gaussian).

### 3.3. Ma trận quay từ quaternion
Quaternion đơn vị $q=r+xi+yj+zk$ ($r^2+x^2+y^2+z^2=1$) biểu diễn phép quay 3D không suy biến (tránh gimbal lock của Euler angles). Công thức chuẩn:

$$
R(q)=I+2r[q_v]_\times+2[q_v]_\times^2,\qquad q_v=(x,y,z)
$$

khai triển thành ma trận $3\times3$ đúng như ở §2.8. Nếu $\lVert q\rVert\ne1$ thì $R(q)$ không còn trực giao (các cột không còn đơn vị/trực giao với nhau) — đây là lý do việc **chuẩn hoá quaternion trước khi dựng $R$** là bắt buộc về mặt toán học; code tắt bước này (§2.8), nên tính đúng đắn phụ thuộc hoàn toàn vào input đã chuẩn hoá từ phía gọi.

### 3.4. Hàm cầu điều hoà thực (Real Spherical Harmonics)
SH là cơ sở trực chuẩn của hàm trên mặt cầu $S^2$, dùng để biểu diễn màu phụ thuộc hướng nhìn (view-dependent color) gọn nhẹ. Bậc $\ell$ có $2\ell+1$ hệ số. Dạng tường minh (toạ độ hướng đơn vị $(x,y,z)$):

$$
Y_0^0=C_0,\qquad Y_1^{-1,0,1}=C_1\cdot(-y,\,z,\,-x)
$$

$C_2[\cdot],C_3[\cdot]$ là các hằng số chuẩn hoá tương tự cho $\ell=2,3$, nhân với đa thức bậc 2, 3 của $(x,y,z)$ (các biểu thức đầy đủ nằm trong `computeColorFromSH` ở `forward.cu`, không trong `auxiliary.h`).

### 3.5. EWA Splatting (Zwicker et al. 2001/2002)
"EWA Volume Splatting" (Zwicker, Pfister, van Baar, Gross — Vis. 2001) và bản mở rộng "Surface Splatting" (2001)/"EWA Splatting" (TVCG 2002) đưa ra công thức chiếu một Gaussian 3D (hoặc elliptical reconstruction kernel) sang không gian ảnh bằng:

1. Biến đổi affine cục bộ (tuyến tính hoá phép chiếu phối cảnh bằng Jacobian $J$ tại tâm Gaussian) — công thức (29).
2. "Sandwich rule": $\Sigma'=JW\Sigma W^\top J^\top$ — công thức (31), với $W$ là phần quay của phép biến đổi camera.
3. Cộng một "low-pass filter" (reconstruction kernel, thường $\approx$ Gaussian phương sai bằng 1 pixel) vào $\Sigma'$ để tránh aliasing khi lấy mẫu rời rạc — nguyên bản Zwicker dùng kernel $h(x)=\mathcal N(x;0,I)$ cộng trực tiếp ($\Sigma'_{\text{lọc}}=\Sigma'+I$ nếu kernel phương sai 1 pixel toàn phần; bản 3DGS gốc dùng hệ số $0.3$ thay vì $1$ như một xấp xỉ thực nghiệm nhẹ hơn). Bản `structgs` đi xa hơn: dùng $0.1$ **và bù lại bằng hệ số $\text{coef}$** để giữ gần đúng năng lượng — đây là kỹ thuật của **Mip-Splatting** (Yu, Chen, Huang, Sattler, Geiger — CVPR 2024, "Mip-Splatting: Alias-free 3D Gaussian Splatting"), không phải EWA/3DGS gốc nguyên bản thuần tuý.

### 3.6. Phân rã ma trận trong `computeCov3D`
$\Sigma_{3D}=M^\top M$ với $M=SR$ là một dạng **phân rã Cholesky suy rộng / phân rã căn bậc hai** của ma trận hiệp phương sai đối xứng xác định dương: bất kỳ $\Sigma$ xác định dương nào cũng viết được $\Sigma=M^\top M$ cho một $M$ nào đó (không duy nhất — ở đây chọn $M=SR$ cụ thể, tương ứng phân rã ellipsoid thành quay $\times$ co giãn theo 3 trục chính).

---

## 4. Kiểm chứng tính đúng sai

So với lý thuyết chuẩn (EWA splatting, SH thực chuẩn) và liệt kê khác biệt so với `MATH/cuda/auxiliary.md` (bản cũ):

| Mục | Lý thuyết chuẩn / kỳ vọng ban đầu | Code thực tế | Khớp? |
|---|---|---|---|
| $C_0,C_1$ | $\sqrt{1/4\pi},\sqrt{3/4\pi}$ | `0.28209479177387814f`, `0.4886025119029199f` | **Khớp**, đến đúng số chữ số thập phân |
| $C_2,C_3$ | Hằng số SH thực chuẩn bậc 2,3 | Giá trị trong code khớp bảng SH thực chuẩn dùng trong `utils/sh_utils.py` | **Khớp** |
| `ndc2Pix` | $\text{pix}=((v+1)S-1)/2$ | Đúng vậy, dòng 44 | **Khớp** |
| `in_frustum` | Cắt theo near-plane và NDC range | Chỉ cắt near-plane $p_{view,z}>0.2$; kiểm tra NDC $[-1.3,1.3]$ **bị vô hiệu hoá** (comment) | **Khớp với code thật**, nhưng khác kỳ vọng lý thuyết "full frustum culling" — đã nêu đúng ở bản cũ |
| `computeCov3D`: $\Sigma=RS^2R^\top$ | Công thức chuẩn ellipsoid-covariance | Code tính $\Sigma=R^\top S^2 R$ (do $M=SR$, $\Sigma=M^\top M$) | **Khác công thức "chuẩn" theo nghĩa chặt, nhưng tự hợp lý (self-consistent)** nhờ quy ước transpose xuyên suốt cả `computeCov2D` — xem §2.8, không phải bug, giống hệt repo gốc Inria |
| `computeCov3D`: chuẩn hoá quaternion | Bắt buộc $\lVert q\rVert=1$ để $R$ trực giao | Dòng chuẩn hoá `/ glm::length(rot)` **bị comment out** | **Rủi ro tiềm ẩn**: nếu `rotations` đưa vào CUDA chưa được chuẩn hoá từ phía Python, $R(q)$ không trực giao, $\Sigma_{3D}$ sai |
| `computeCov2D`: $\Sigma'=JW\Sigma W^\top J^\top$ | Công thức (31) Zwicker et al. 2002 | `cov = transpose(T)*transpose(Vrk)*T`, $T=WJ$ — tương đương công thức chuẩn qua quy ước transpose nhất quán | **Khớp về bản chất hình học** |
| **Hằng số low-pass filter** | Đề bài giả định có thể là $+0.3I$ | **$\texttt{kernel\_size}=0.1$**, cộng có **kèm hệ số bù** $\text{coef}=\sqrt{\det_0/(\det_1+10^{-6})}$ | **KHÔNG khớp giả định $0.3$.** Đây là điểm quan trọng nhất cần sửa: bản gốc 3DGS (Kerbl 2023) dùng $0.3$ không có hệ số bù; bản `structgs` dùng $0.1$ kiểu Mip-Splatting có hệ số bù alpha. Đề bài đưa ra "$+0.3I$" chỉ là một khả năng cần kiểm tra — **đã xác minh và kết luận là không đúng với code thực tế của submodule này** |
| `getRect`, `transformPoint*`, `dnormvdz/dnormvdv`, `sigmoid`, `computeEllipseIntersection`, `disc`, `t = mult*2ln(255α)` | — | Đã đối chiếu từng dòng ở §2 | **Khớp hoàn toàn với bản cũ** |

**Các điểm đã sửa / bổ sung so với `MATH/cuda/auxiliary.md` (bản cũ):**

1. Bản cũ **không đề cập** `computeCov3D`/`computeCov2D` — đúng về phạm vi (hai hàm này thật sự không nằm trong `auxiliary.h`), nhưng vì đề bài yêu cầu kiểm chứng rõ ràng, tài liệu mới bổ sung toàn bộ §2.8–2.9, §3.5–3.6 và nêu rõ nguồn là `forward.cu` chứ không phải `auxiliary.h`, tránh gây hiểu lầm là các công thức đó thuộc file này.
2. Xác minh và chốt số liệu chính xác cho "low-pass filter": **$0.1$, kèm hệ số bù `coef`**, không phải $0.3$ (3DGS gốc) và không phải một con số tuỳ ý nào khác — lấy trực tiếp từ dòng 118–127 của `forward.cu`.
3. Làm rõ quy ước quaternion `(r,x,y,z) = (q.x,q.y,q.z,q.w)` và việc chuẩn hoá bị tắt trong `computeCov3D` — bản cũ không có mục này (vì bản cũ không xét `computeCov3D`).
4. Diễn giải lại $\Sigma_{3D}=M^\top M=R^\top S^2 R$ (không phải $RS^2R^\top$ "chuẩn sách giáo khoa") và giải thích đây là do quy ước row-major/transpose nhất quán trong toàn bộ pipeline CUDA, không phải lỗi.
5. Nội dung còn lại (SH constants, `ndc2Pix`, `getRect`, `transformPoint*`, `in_frustum`, `dnormvdz/dnormvdv`, `sigmoid`, `computeEllipseIntersection`/`duplicateToTilesTouched`) đối chiếu lại với code thật và **xác nhận bản cũ đúng** — chỉ viết lại rõ ràng hơn, thêm số dòng cụ thể và phần suy công thức chi tiết hơn (ví dụ suy ra $dy$ từ phương trình bậc 2 thay vì chỉ liệt kê công thức cuối).

---

## 5. Ví dụ số (tính tay, kiểm chứng công thức khớp code)

### Input tự đặt

- Quaternion (đã chuẩn hoá, quay $90^\circ$ quanh trục $z$), theo đúng quy ước code $(q.x,q.y,q.z,q.w)=(r,x,y,z)$:
$$
q=(r,x,y,z)=\Big(\tfrac{\sqrt2}{2},\,0,\,0,\,\tfrac{\sqrt2}{2}\Big)\approx(0.70711,\,0,\,0,\,0.70711)
$$
- Scale: $s=(s_x,s_y,s_z)=(0.02,\,0.01,\,0.005)$, $\text{mod}=1$.
- View matrix: camera trùng world frame, Gaussian đặt tại $(0,0,5)$ trước camera $\Rightarrow$ `viewmatrix` $=I_4$ (không quay, không tịnh tiến), $W=I_3$.
- Camera: $f_x=f_y=500$ px, $\tan\!\text{fovx}=\tan\!\text{fovy}=0.5$ (đủ lớn để không bị clip ở bước 6.1).

### Bước A: Dựng $R(q)$

Với $r=0.70711,x=0,y=0,z=0.70711$:

$$
1-2(y^2+z^2)=1-2(0.5)=0,\quad 2(xy-rz)=-1,\quad 2(xz+ry)=0
$$
$$
2(xy+rz)=1,\quad 1-2(x^2+z^2)=0,\quad 2(yz-rx)=0
$$
$$
2(xz-ry)=0,\quad 2(yz+rx)=0,\quad 1-2(x^2+y^2)=1
$$

$$
R=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}\quad\text{(quay }90^\circ\text{ quanh }z\text{)}
$$

### Bước B: $\Sigma_{3D}=R^\top S^2 R$ (đúng công thức code, $M=SR,\ \Sigma=M^\top M$)

$S=\mathrm{diag}(0.02,0.01,0.005)$. $M=SR$: nhân mỗi hàng của $R$ với $s$ tương ứng:

$$
M=\begin{pmatrix}0&-0.02&0\\0.01&0&0\\0&0&0.005\end{pmatrix}
$$

Cột của $M$: $\text{col}_0=(0,0.01,0)$, $\text{col}_1=(-0.02,0,0)$, $\text{col}_2=(0,0,0.005)$. $\Sigma_{3D}=M^\top M$, $(\Sigma_{3D})_{ij}=\text{col}_i\cdot\text{col}_j$:

$$
\Sigma_{3D}=\begin{pmatrix}0.0001&0&0\\0&0.0004&0\\0&0&0.000025\end{pmatrix}
$$

(Mọi phần tử ngoài đường chéo $=0$ vì $R$ chỉ hoán vị trục $x\leftrightarrow y$.) Kiểm tra bằng trực giác hình học: trục cục bộ $x$ (scale $0.02$) sau khi quay $90^\circ$ quanh $z$ chỉ về hướng world-$y$ $\Rightarrow\Sigma_{yy}=0.02^2=0.0004$ ✓; trục cục bộ $y$ (scale $0.01$) chỉ về hướng world-$(-x)$ $\Rightarrow\Sigma_{xx}=0.01^2=0.0001$ ✓; trục $z$ giữ nguyên $\Rightarrow\Sigma_{zz}=0.005^2=0.000025$ ✓.

Lưu trữ 6 số: `cov3D` $=[0.0001,\ 0,\ 0,\ 0.0004,\ 0,\ 0.000025]$.

### Bước C: `computeCov2D`

$t=\text{transformPoint4x3}((0,0,5),I_4)=(0,0,5)$ (viewmatrix đơn vị). $t_x/t_z=0$, $t_y/t_z=0$, cả hai nằm trong $[\pm1.3\cdot0.5]=[\pm0.65]$ nên không bị clip, $t$ không đổi.

$$
J=\begin{pmatrix}500/5&0&0\\0&500/5&0\\0&0&0\end{pmatrix}=\begin{pmatrix}100&0&0\\0&100&0\\0&0&0\end{pmatrix}
$$

$W=I_3\Rightarrow T=WJ=J$. Vì $T$ đối xứng ở đây ($T=T^\top$, do dạng đường chéo với hàng/cột 3 bằng 0):

$$
\Sigma_{2D}^{\text{raw}}=T^\top\Sigma_{3D}T=T\,\Sigma_{3D}\,T
$$

$T\Sigma_{3D}=\mathrm{diag}(100,100,0)\cdot\mathrm{diag}(0.0001,0.0004,0.000025)=\mathrm{diag}(0.01,0.04,0)$

$(T\Sigma_{3D})T=\mathrm{diag}(0.01,0.04,0)\cdot\mathrm{diag}(100,100,0)=\mathrm{diag}(1,\,4,\,0)$

Lấy khối $2\times2$ trên-trái:

$$
\Sigma_{2D}^{\text{raw}}=\begin{pmatrix}1&0\\0&4\end{pmatrix}\qquad(a=1,\ b=0,\ c=4)
$$

**Kiểm tra chéo bằng công thức EWA chuẩn** $\Sigma'=JW\Sigma_{3D}W^\top J^\top=J\Sigma_{3D}J^\top$ (vì $W=I$): $J$ hàng 0 $=(100,0,0)$, hàng 1 $=(0,100,0)$. $J\Sigma_{3D}J^\top$ phần tử $(0,0)=100^2\cdot\Sigma_{3D,00}=10000\cdot0.0001=1$ ✓, $(1,1)=10000\cdot0.0004=4$ ✓, $(0,1)=0$ ✓ — khớp hoàn toàn với kết quả tính qua code ở trên (xác nhận quy ước transpose không gây sai số trong trường hợp này, đúng như lập luận ở §2.9).

### Bước D: Low-pass filter + hệ số bù

$$
\det_0=\max(10^{-6},\,1\cdot4-0^2)=4
$$
$$
a\leftarrow1+0.1=1.1,\qquad c\leftarrow4+0.1=4.1
$$
$$
\det_1=\max(10^{-6},\,1.1\cdot4.1-0^2)=4.51
$$
$$
\text{coef}=\sqrt{\frac{4}{4.51+10^{-6}}}=\sqrt{0.88692\ldots}=0.94180\ldots
$$

**Kết quả cuối `computeCov2D` trả về:**

$$
\boxed{\ \Sigma_{2D}=\begin{pmatrix}1.1&0\\0&4.1\end{pmatrix},\quad \text{coef}\approx0.9418\ }
$$

(nghĩa là alpha hiệu dụng của splat này bị nhân thêm $\approx0.9418$ ở nơi gọi trong `forward.cu` để bù lại việc "phồng" ellipse do kernel $0.1$.)

### Bước E: Kiểm tra chéo bằng các hàm còn lại của `auxiliary.h`

- **`in_frustum`:** $p_{view,z}=t_z=5>0.2\Rightarrow$ visible $=$ true.
- **Conic** $\text{con\_o}=\Sigma_{2D}^{-1}$: $\det=1.1\cdot4.1-0^2=4.51$, $\text{con\_o}=(A,B,C)=(4.1/4.51,\,0,\,1.1/4.51)=(0.9091,\,0,\,0.2439)$.
- **`duplicateToTilesTouched`** với $\alpha=0.8,\ \texttt{mult}=1$: $\text{disc}=B^2-AC=-0.9091\cdot0.2439=-0.2218<0$ ✓ (ellipse hợp lệ vì $A>0,C>0,\text{disc}<0$). $t=2\ln(255\cdot0.8)=2\ln(204)=2\cdot5.3181=10.636$. Vì $B=0$ (ellipse không nghiêng, trục chính trùng $x,y$), bán trục bbox tại $dy=0$: $dx_{\max}=\sqrt{t/A}=\sqrt{10.636/0.9091}=\sqrt{11.70}=3.421$ px; tại $dx=0$: $dy_{\max}=\sqrt{t/C}=\sqrt{10.636/0.2439}=\sqrt{43.61}=6.604$ px. Kết quả $dy_{\max}>dx_{\max}$ khớp với $\Sigma_{2D,yy}=4.1>\Sigma_{2D,xx}=1.1$ (ellipse rộng theo trục $y$) — xác nhận tính nhất quán end-to-end từ quaternion/scale $\to\Sigma_{3D}\to\Sigma_{2D}\to$ bounding box tile.
- **`ndc2Pix`** (minh hoạ công thức độc lập, không phụ thuộc Gaussian trên): với $v=0$ (tâm NDC) và ảnh rộng $S=800$: $\text{pix}=((0+1)\cdot800-1)\cdot0.5=399.5$ — đúng là pixel giữa của trục $800$ pixel theo quy ước pixel-center.

Toàn bộ chuỗi tính tay ở trên khớp với từng dòng code đã trích ở §2, xác nhận các công thức trong tài liệu này là đúng với `auxiliary.h` (và `forward.cu` ở phần tham chiếu §2.8–2.9) tại thời điểm đọc mã nguồn.
