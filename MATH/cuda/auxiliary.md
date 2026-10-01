# Công thức toán học trong `cuda_rasterizer/auxiliary.h`

Nguồn: `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/auxiliary.h` (393 dòng). Header này chứa các hàm `__device__ inline`/`__forceinline__` dùng chung bởi `forward.cu`, `backward.cu`, `rasterizer_impl.cu`: hằng số Spherical Harmonics, biến đổi toạ độ thuần nhất, chuẩn hoá vector (và đạo hàm của chuẩn hoá), sigmoid, kiểm tra frustum, và thuật toán xác định tile bị phủ bởi một Gaussian (AccuTile/SNUGBox — đã mô tả chi tiết thuật toán ở `MATH/cuda/rasterizer_impl.md`, ở đây chỉ nêu lại công thức hình học cốt lõi).

**Đồng bộ với bản đối chiếu kỹ hơn:** nội dung dưới đây đã được kiểm chứng lại và đồng bộ với `MATH/submodules/diff-gaussian-rasterization_structgs/auxiliary.md` (đọc trực tiếp source mới nhất). Bản đó còn trình bày thêm `computeCov3D`/`computeCov2D` (định nghĩa ở `forward.cu`, không nằm trong `auxiliary.h`) kèm ví dụ số chi tiết — xem file đó để biết đầy đủ, kể cả phát hiện quan trọng là **hằng số low-pass filter của biến thể `structgs` là $0.1$ (kèm hệ số bù `coef`), không phải $0.3$ như bản 3DGS gốc**.

---

## 1. Hằng số chuẩn hoá Spherical Harmonics (dòng 22–40)

```cpp
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};
```

$$
C_0 = 0.28209479177387814 = \sqrt{\tfrac{1}{4\pi}}
$$
$$
C_1 = 0.4886025119029199 = \sqrt{\tfrac{3}{4\pi}}
$$
$$
C_2 = \big[1.0925\ldots,\ -1.0925\ldots,\ 0.3154\ldots,\ -1.0925\ldots,\ 0.5463\ldots\big]
$$
$$
C_3 = \big[-0.5900\ldots,\ 2.8906\ldots,\ -0.4570\ldots,\ 0.3732\ldots,\ -0.4570\ldots,\ 1.4453\ldots,\ -0.5900\ldots\big]
$$

Đây đúng là các hệ số chuẩn hoá SH thực (real SH) bậc 0–3 theo quy ước chuẩn trong đồ hoạ máy tính (khớp với `utils/sh_utils.py` ở phía Python — bậc 4 ($C_4$) không xuất hiện trong file header này, có nghĩa biến thể CUDA của dự án giới hạn SH tối đa bậc 3, dù `gaussian_model.py`/`sh_utils.py` phía Python có thể hỗ trợ bậc cao hơn). Kiểm tra số: $\sqrt{1/(4\pi)}=0.28209479177387814\ldots$ khớp đến 16 chữ số với `SH_C0`.

## 2. Chuyển đổi NDC sang toạ độ pixel (`ndc2Pix`, dòng 42–45)

```cpp
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}
```

$$
\text{pix} = \frac{(v+1)\,S - 1}{2}, \qquad v\in[-1,1]\ (\text{NDC}),\ S = \text{kích thước ảnh (W hoặc H)}
$$

## 3. Hình chữ nhật tile bao quanh (`getRect`, dòng 47–69, 2 overload)

```cpp
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}
```

Với tâm chiếu $p=(p_x,p_y)$ và bán kính ảnh hưởng $r=$ `max_radius`, kích thước tile $(\text{BLOCK\_X},\text{BLOCK\_Y})$:

$$
\text{rect}_{min} = \left(\mathrm{clip}\!\left(\left\lfloor\frac{p_x-r}{\text{BLOCK\_X}}\right\rfloor,0,\mathrm{grid}_x\right),\ \mathrm{clip}\!\left(\left\lfloor\frac{p_y-r}{\text{BLOCK\_Y}}\right\rfloor,0,\mathrm{grid}_y\right)\right)
$$
$$
\text{rect}_{max} = \left(\mathrm{clip}\!\left(\left\lfloor\frac{p_x+r+\text{BLOCK\_X}-1}{\text{BLOCK\_X}}\right\rfloor,0,\mathrm{grid}_x\right),\ \mathrm{clip}\!\left(\left\lfloor\frac{p_y+r+\text{BLOCK\_Y}-1}{\text{BLOCK\_Y}}\right\rfloor,0,\mathrm{grid}_y\right)\right)
$$

Overload thứ hai (dòng 59–69) giống hệt nhưng dùng `ext_rect`$=(r_x,r_y)$ (hình chữ nhật bao, không phải bán kính tròn) thay cho $r$:

```cpp
__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}
```

## 4. Phép biến đổi toạ độ thuần nhất (dòng 71–110)

```cpp
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}
```

Nhân điểm $p=(x,y,z)$ với ma trận $4\times4$ lưu dạng cột (`matrix[0..15]`, column-major):

$$
\text{transformPoint4x3}(p) = \begin{pmatrix}M_{00}x+M_{10}y+M_{20}z+M_{30}\\ M_{01}x+M_{11}y+M_{21}z+M_{31}\\ M_{02}x+M_{12}y+M_{22}z+M_{32}\end{pmatrix}
$$

(bỏ hàng $w$, dùng cho world→camera, giả định $w=1$ không chia lại)

```cpp
__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}
```

$$
\text{transformPoint4x4}(p) = \begin{pmatrix}M_{00}x+M_{10}y+M_{20}z+M_{30}\\ \vdots\\ M_{03}x+M_{13}y+M_{23}z+M_{33}\end{pmatrix} = (x',y',z',w')
$$

(dùng cho chiếu phối cảnh đầy đủ, cần chia $w'$ ở nơi gọi, ví dụ `in_frustum`)

```cpp
__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}
```

$$
\text{transformVec4x3}(p) = M_{3\times3}\,p \qquad \text{(bỏ phần dịch chuyển — dùng cho vector, không phải điểm)}
$$

```cpp
__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}
```

$$
\text{transformVec4x3Transpose}(p) = M_{3\times3}^T\,p
$$

## 5. Kiểm tra điểm trong frustum (`in_frustum`, dòng 152–177)

```cpp
float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

// Bring points to screen space
float4 p_hom = transformPoint4x4(p_orig, projmatrix);
float p_w = 1.0f / (p_hom.w + 0.0000001f);
float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
p_view = transformPoint4x3(p_orig, viewmatrix);

if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
{
	if (prefiltered)
	{
		printf("Point is filtered although prefiltered is set. This shouldn't happen!");
		__trap();
	}
	return false;
}
return true;
```

$$
(x',y',z',w') = \text{transformPoint4x4}(p_{orig}, \text{projmatrix}), \qquad w_{inv} = \frac{1}{w'+10^{-7}}
$$
$$
p_{proj} = (x'w_{inv},\, y'w_{inv},\, z'w_{inv}), \qquad p_{view} = \text{transformPoint4x3}(p_{orig}, \text{viewmatrix})
$$

Điều kiện loại bỏ (near-plane culling — dòng kiểm tra biên NDC $p_{proj}\in[-1.3,1.3]^2$ **bị comment**, nên hiện tại hàm không làm frustum culling theo $x,y$, chỉ cắt theo near-plane $z$):

$$
\text{visible} = \big(p_{view,z} > 0.2\big)
$$

## 6. Sigmoid (dòng 147–150)

```cpp
__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}
```

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

## 7. Đạo hàm của phép chuẩn hoá vector (`dnormvdz`, `dnormvdv`, dòng 112–145)

```cpp
__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}
```

Cho $\hat v = v/\lVert v\rVert$, với $s=\lVert v\rVert^2=\sum_k v_k^2$, đạo hàm ngược (vector-Jacobian product) của chuẩn hoá theo từng thành phần — công thức chuẩn của đạo hàm $\partial \hat v_k/\partial v_j = \big(s\,\delta_{kj}-v_kv_j\big)/s^{3/2}$, áp cho $d\hat v\cdot (\partial \hat v/\partial v)$:

$$
\frac{\partial \hat v}{\partial v}\Big|_j \cdot dv \;\equiv\; \text{dnormvdv}_j = \frac{\big(s-v_j^2\big)\,dv_j \;-\; v_j\sum_{k\ne j} v_k\,dv_k}{s^{3/2}}
$$

Riêng `dnormvdz` chỉ trả về thành phần $z$ của công thức trên (dùng khi chỉ cần gradient theo trục $z$, ví dụ pháp tuyến):

$$
\text{dnormvdz} = \frac{-v_xv_z\,dv_x - v_yv_z\,dv_y + (s-v_z^2)\,dv_z}{s^{3/2}}
$$

Bản `dnormvdv(float4,float4)` mở rộng công thức này cho vector 4 chiều (dòng 132–145, dùng cho gradient quaternion $q\to \hat q$):

```cpp
__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}
```

với $s=\sum_{k\in\{x,y,z,w\}} v_k^2$ và $\text{vdv}_k = v_k\,dv_k$:

$$
\text{dnormvdv}_k = \frac{(s-v_k^2)\,dv_k - v_k\big(\textstyle\sum_j \text{vdv}_j - \text{vdv}_k\big)}{s^{3/2}}
$$

## 8. Hình học ellipse mức opacity & thuật toán AccuTile/SNUGBox (dòng 179–393)

Đã trình bày đầy đủ ở `MATH/cuda/rasterizer_impl.md` mục 4 (các hàm `computeEllipseIntersection`, `processTiles`, `duplicateToTilesTouched` thực chất định nghĩa tại đây, `rasterizer_impl.cu` chỉ gọi lại). Trích công thức cốt lõi:

```cpp
// duplicateToTilesTouched, dòng 324, 332–333
float disc = con_o.y * con_o.y - con_o.x * con_o.z;
...
float t = 2.0f * log(con_o.w * 255.0f);
t = mult * t;
```

$$
\text{power}(dx,dy) = \tfrac12\big(A\,dx^2+C\,dy^2\big)+B\,dx\,dy, \qquad \Sigma_{2D}^{-1}=\begin{pmatrix}A&B\\B&C\end{pmatrix}=\text{con\_o}_{xyz}
$$
$$
\text{disc}=B^2-AC,\qquad t = \texttt{mult}\cdot 2\ln(255\,\alpha)
$$

Suy ra từ điều kiện biên (ngưỡng năng lượng ứng với độ trong suốt $1/255$): $\alpha\exp\!\big(-\tfrac12(Adx^2+2Bdxdy+Cdy^2)\big)=\tfrac1{255}\iff Adx^2+2Bdxdy+Cdy^2=2\ln(255\alpha)$.

```cpp
// computeEllipseIntersection, dòng 187–193
float h = coord - p_u;  // h = y - p.y for y, x - p.x for x
float sqrt_term = sqrt(disc * h * h + t * coeff);

return {
  (-con_o.y * h - sqrt_term) / coeff + p_v,
  (-con_o.y * h + sqrt_term) / coeff + p_v
};
```

Với $dx=h=\text{coord}-p_u$ cố định, giải phương trình bậc 2 theo $dy$: $C\,dy^2+2Bh\,dy+(Ah^2-t)=0$, nghiệm:

$$
dy=\frac{-Bh\pm\sqrt{B^2h^2-C(Ah^2-t)}}{C}=\frac{-Bh\pm\sqrt{(B^2-AC)h^2+Ct}}{C}=\frac{-Bh\pm\sqrt{\text{disc}\cdot h^2+ t\cdot\text{coeff}}}{\text{coeff}}
$$

(`coeff`$=C$ khi không `isY`, $=A$ khi `isY` do vai trò $u,v$ hoán đổi — đúng tham số `coeff` trong code trích ở trên).

---

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc (dòng) | Công thức toán học tương ứng |
|---|---|
| `SH_C0 = 0.28209479177387814f` (dòng 23) | $C_0=\sqrt{1/(4\pi)}$ |
| `SH_C1 = 0.4886025119029199f` (dòng 24) | $C_1=\sqrt{3/(4\pi)}$ |
| `SH_C2[]` (dòng 25–31), `SH_C3[]` (dòng 32–40) | hệ số chuẩn hoá SH thực bậc 2, bậc 3 |
| `((v + 1.0) * S - 1.0) * 0.5` (dòng 44) | $\text{pix}=((v+1)S-1)/2$ |
| `min(grid.x, max(0, (int)((p.x - max_radius)/BLOCK_X)))` (dòng 50) | $\text{rect}_{min,x}=\mathrm{clip}(\lfloor (p_x-r)/\text{BLOCK\_X}\rfloor,0,\mathrm{grid}_x)$ |
| `min(grid.x, max(0, (int)((p.x + max_radius + BLOCK_X - 1)/BLOCK_X)))` (dòng 54) | $\text{rect}_{max,x}=\mathrm{clip}(\lfloor (p_x+r+\text{BLOCK\_X}-1)/\text{BLOCK\_X}\rfloor,0,\mathrm{grid}_x)$ |
| `matrix[0]*p.x + matrix[4]*p.y + matrix[8]*p.z + matrix[12]` (dòng 74, transformPoint4x3) | hàng đầu của $M_{col-major}\cdot (x,y,z,1)^T$ |
| `matrix[0]*p.x+matrix[4]*p.y+matrix[8]*p.z` (dòng 95, transformVec4x3) | $M_{3\times3}\cdot p$, bỏ phần dịch chuyển |
| `matrix[0]*p.x+matrix[1]*p.y+matrix[2]*p.z` (dòng 105, transformVec4x3Transpose) | $M_{3\times3}^T\cdot p$ |
| `p_w = 1.0f/(p_hom.w + 1e-7)` (dòng 163) | $w_{inv}=1/(w'+10^{-7})$ |
| `p_proj = {p_hom.x*p_w, p_hom.y*p_w, p_hom.z*p_w}` (dòng 164) | $p_{proj}=(x'w_{inv},y'w_{inv},z'w_{inv})$ |
| `if (p_view.z <= 0.2f)... return false;` (dòng 167) | $\text{visible} = (p_{view,z}>0.2)$ |
| `1.0f / (1.0f + expf(-x))` (dòng 149) | $\sigma(x)=1/(1+e^{-x})$ |
| `invsum32 = 1.0f/sqrt(sum2*sum2*sum2)` (dòng 115, 123) | $s^{-3/2}$ |
| `dnormvdz = (-v.x*v.z*dv.x - v.y*v.z*dv.y + (sum2-v.z*v.z)*dv.z) * invsum32` (dòng 116) | $\text{dnormvdz}=\dfrac{-v_xv_zdv_x-v_yv_zdv_y+(s-v_z^2)dv_z}{s^{3/2}}$ |
| `dnormvdv.x = ((+sum2-v.x*v.x)*dv.x - v.y*v.x*dv.y - v.z*v.x*dv.z)*invsum32` (dòng 126) | $\text{dnormvdv}_x=\dfrac{(s-v_x^2)dv_x-v_xv_ydv_y-v_xv_zdv_z}{s^{3/2}}$ |
| `float4` overload: `vdv_sum`, `(sum2-v.x*v.x)*dv.x - v.x*(vdv_sum-vdv.x)` (dòng 138–143) | $\text{dnormvdv}_k=\dfrac{(s-v_k^2)dv_k-v_k(\sum_j\text{vdv}_j-\text{vdv}_k)}{s^{3/2}}$ |
| `disc = con_o.y*con_o.y - con_o.x*con_o.z` (dòng 324) | $\text{disc}=B^2-AC$ |
| `t = 2.0f*log(con_o.w*255.0f); t = mult*t;` (dòng 332–333) | $t=\texttt{mult}\cdot 2\ln(255\alpha)$ |
| `sqrt_term = sqrt(disc * h * h + t * coeff)` (dòng 188) | $\sqrt{\text{disc}\cdot h^2+t\cdot\text{coeff}}$ |

**Lưu ý kiểm chứng:** toàn bộ số dòng và trích dẫn ở trên lấy trực tiếp từ `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/auxiliary.h` (393 dòng), đối chiếu khớp với `MATH/submodules/diff-gaussian-rasterization_structgs/auxiliary.md`. Không phát hiện sai lệch công thức nào trong nội dung vốn có của file này; phần bổ sung duy nhất là trích dẫn code + số dòng chính xác, và ghi chú tham chiếu tới phát hiện quan trọng về hằng số low-pass filter ($0.1$, không phải $0.3$) đã nêu ở bản submodules (thuộc `computeCov2D` trong `forward.cu`, ngoài phạm vi `auxiliary.h`).
