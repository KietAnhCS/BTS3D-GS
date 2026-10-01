# Công thức toán học trong `utils/sh_utils.py`

File này cài đặt các hàm cầu điều hoà thực (real Spherical Harmonics) dùng để biểu diễn màu phụ thuộc hướng nhìn (view-dependent color) của mỗi Gaussian trong 3DGS, cùng các phép chuyển đổi RGB ↔ SH cho hệ số DC (bậc 0).

---

## 1. Hằng số chuẩn hoá SH (hardcoded, chính xác theo code)

Trích nguyên văn (`utils/sh_utils.py`, dòng 26–54):

```python
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   
```

$$
C_0 = 0.28209479177387814
$$

$$
C_1 = 0.4886025119029199
$$

$$
C_2 = [\,1.0925484305920792,\ -1.0925484305920792,\ 0.31539156525252005,\ -1.0925484305920792,\ 0.5462742152960396\,]
$$

$$
C_3 = [\,-0.5900435899266435,\ 2.890611442640554,\ -0.4570457994644658,\ 0.3731763325901154,\ -0.4570457994644658,\ 1.445305721320277,\ -0.5900435899266435\,]
$$

$$
C_4 = [\,2.5033429417967046,\ -1.7701307697799304,\ 0.9461746957575601,\ -0.6690465435572892,\ 0.10578554691520431,\ -0.6690465435572892,\ 0.47308734787878004,\ -1.7701307697799304,\ 0.6258357354491761\,]
$$

Đây đúng là các hằng số chuẩn hoá của real spherical harmonics tới bậc $\ell=4$ (dù docstring của `eval_sh` chỉ nói "0-3 supported", code thực tế hỗ trợ `deg<=4`, bao gồm cả nhánh $C_4$ cho $\ell=4$).

---

## 2. Đánh giá SH tại hướng đơn vị $(x,y,z)$ (`eval_sh`)

Cho bậc tối đa `deg` $=\ell_{max}\in\{0,1,2,3,4\}$, số hệ số cần có $\ge(\ell_{max}+1)^2$, và hướng nhìn đơn vị $\mathbf{d}=(x,y,z)$, $\lVert\mathbf{d}\rVert=1$.

### Bậc 0 ($\ell=0$, 1 hệ số, chỉ số `sh[...,0]`)

Trích nguyên văn (`utils/sh_utils.py`, dòng 74):

```python
result = C0 * sh[..., 0]
```

$$
f_0 = C_0\, c_0
$$

### Bậc 1 ($\ell=1$, cộng thêm 3 hệ số `sh[...,1:4]`)

Trích nguyên văn (dòng 76–80):

```python
x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
result = (result -
        C1 * y * sh[..., 1] +
        C1 * z * sh[..., 2] -
        C1 * x * sh[..., 3])
```

$$
f_1 = f_0 \;-\; C_1\, y\, c_1 \;+\; C_1\, z\, c_2 \;-\; C_1\, x\, c_3
$$

tương ứng đúng 3 hàm cơ sở bậc 1 của real SH: $Y_1^{-1}\propto y$, $Y_1^{0}\propto z$, $Y_1^{1}\propto x$ (với dấu âm trước $y$ và $x$ theo đúng quy ước trong code).

### Bậc 2 ($\ell=2$, cộng thêm 5 hệ số `sh[...,4:9]`), với $xx=x^2, yy=y^2, zz=z^2, xy=xy, yz=yz, xz=xz$:

Trích nguyên văn (dòng 83–90):

```python
xx, yy, zz = x * x, y * y, z * z
xy, yz, xz = x * y, y * z, x * z
result = (result +
        C2[0] * xy * sh[..., 4] +
        C2[1] * yz * sh[..., 5] +
        C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
        C2[3] * xz * sh[..., 7] +
        C2[4] * (xx - yy) * sh[..., 8])
```

$$
f_2 = f_1 + C_{2,0}\,xy\,c_4 + C_{2,1}\,yz\,c_5 + C_{2,2}\,(2zz-xx-yy)\,c_6 + C_{2,3}\,xz\,c_7 + C_{2,4}\,(xx-yy)\,c_8
$$

### Bậc 3 ($\ell=3$, cộng thêm 7 hệ số `sh[...,9:16]`):

Trích nguyên văn (dòng 93–100):

```python
result = (result +
C3[0] * y * (3 * xx - yy) * sh[..., 9] +
C3[1] * xy * z * sh[..., 10] +
C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
C3[5] * z * (xx - yy) * sh[..., 14] +
C3[6] * x * (xx - 3 * yy) * sh[..., 15])
```

$$
\begin{aligned}
f_3 = f_2 \;&+\; C_{3,0}\,y(3xx-yy)\,c_9 \;+\; C_{3,1}\,xy\,z\,c_{10} \;+\; C_{3,2}\,y(4zz-xx-yy)\,c_{11} \\
&+\; C_{3,3}\,z(2zz-3xx-3yy)\,c_{12} \;+\; C_{3,4}\,x(4zz-xx-yy)\,c_{13} \;+\; C_{3,5}\,z(xx-yy)\,c_{14} \\
&+\; C_{3,6}\,x(xx-3yy)\,c_{15}
\end{aligned}
$$

### Bậc 4 ($\ell=4$, cộng thêm 9 hệ số `sh[...,16:25]`):

Trích nguyên văn (dòng 103–111):

```python
result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
        C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
        C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
        C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
        C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
        C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
        C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
        C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
        C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
```

$$
\begin{aligned}
f_4 = f_3 \;&+\; C_{4,0}\,xy(xx-yy)\,c_{16} \;+\; C_{4,1}\,yz(3xx-yy)\,c_{17} \;+\; C_{4,2}\,xy(7zz-1)\,c_{18} \\
&+\; C_{4,3}\,yz(7zz-3)\,c_{19} \;+\; C_{4,4}\,\big(zz(35zz-30)+3\big)\,c_{20} \;+\; C_{4,5}\,xz(7zz-3)\,c_{21} \\
&+\; C_{4,6}\,(xx-yy)(7zz-1)\,c_{22} \;+\; C_{4,7}\,xz(xx-3yy)\,c_{23} \\
&+\; C_{4,8}\,\big(xx(xx-3yy)-yy(3xx-yy)\big)\,c_{24}
\end{aligned}
$$

Giá trị trả về là $f_{\ell_{max}}$ ứng với bậc được yêu cầu — màu quan sát được theo hướng $\mathbf{d}$ là tổng có trọng số của các hệ số SH $c_0,\dots,c_{(\ell_{max}+1)^2-1}$ với các hàm cơ sở cầu điều hoà thực tương ứng.

**Lưu ý nhỏ về điều kiện bậc (dòng 70, 92, 102):** code dùng `assert deg <= 4 and deg >= 0` (dòng 70) rồi 3 khối `if deg > 0 / deg > 1 / deg > 2 / deg > 3` lồng nhau (dòng 75, 82, 92, 102) — tức bậc 4 chỉ được cộng vào nếu đã đi qua đủ 3 lớp `if` trước đó, khớp đúng với việc $f_4$ build dần từ $f_0\to f_1\to f_2\to f_3\to f_4$ như trình bày ở trên, không có đường tắt bỏ qua bậc trung gian.

---

## 3. Chuyển đổi RGB ↔ SH bậc 0 (hệ số DC)

### `RGB2SH`

Trích nguyên văn (`utils/sh_utils.py`, dòng 114–115):

```python
def RGB2SH(rgb):
    return (rgb - 0.5) / C0
```

$$
c_{SH} = \frac{\mathrm{RGB} - 0.5}{C_0}
$$

### `SH2RGB` (nghịch đảo chính xác của `RGB2SH`)

Trích nguyên văn (dòng 117–118):

```python
def SH2RGB(sh):
    return sh * C0 + 0.5
```

$$
\mathrm{RGB} = c_{SH}\cdot C_0 + 0.5
$$

Hai công thức này đúng là nghịch đảo của nhau:

$$
\mathrm{SH2RGB}\big(\mathrm{RGB2SH}(\mathrm{RGB})\big) = \left(\frac{\mathrm{RGB}-0.5}{C_0}\right)C_0 + 0.5 = \mathrm{RGB}
$$

Về mặt vật lý: màu "trung tính" (không phụ thuộc hướng, chỉ có hệ số DC) ứng với hướng nhìn bất kỳ cho ra đúng $f_0 = C_0\cdot c_{SH,0}$, và khi khởi tạo Gaussian từ point cloud, công thức $c_{SH,0}=(\mathrm{RGB}-0.5)/C_0$ đảm bảo $f_0 + 0.5 = \mathrm{RGB}$ ngay từ bước khởi tạo (do $f_0$ cộng thêm offset $0.5$ ở bước render màu cuối cùng, nằm ngoài file này).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `C0 = 0.28209479177387814` | $C_0 = 0.28209479177387814$ |
| `C1 = 0.4886025119029199` | $C_1 = 0.4886025119029199$ |
| `C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]` | $C_2 = [C_{2,0},\dots,C_{2,4}]$ như liệt kê ở mục 1 |
| `C3 = [-0.5900435899266435, 2.890611442640554, ...]` | $C_3=[C_{3,0},\dots,C_{3,6}]$ như liệt kê ở mục 1 |
| `C4 = [2.5033429417967046, -1.7701307697799304, ...]` | $C_4=[C_{4,0},\dots,C_{4,8}]$ như liệt kê ở mục 1 |
| `result = C0 * sh[..., 0]` | $f_0 = C_0\,c_0$ |
| `result - C1*y*sh[...,1] + C1*z*sh[...,2] - C1*x*sh[...,3]` | $f_1=f_0-C_1 y c_1+C_1 z c_2-C_1 x c_3$ |
| `C2[0]*xy*sh[...,4] + C2[1]*yz*sh[...,5] + C2[2]*(2*zz-xx-yy)*sh[...,6] + C2[3]*xz*sh[...,7] + C2[4]*(xx-yy)*sh[...,8]` | $f_2=f_1+C_{2,0}xy\,c_4+C_{2,1}yz\,c_5+C_{2,2}(2zz-xx-yy)c_6+C_{2,3}xz\,c_7+C_{2,4}(xx-yy)c_8$ |
| `C3[0]*y*(3*xx-yy)*sh[...,9] + ... + C3[6]*x*(xx-3*yy)*sh[...,15]` | $f_3=f_2+\sum_{k=0}^{6} C_{3,k}\cdot(\text{đa thức bậc 3})\cdot c_{9+k}$ (mục 2, bậc 3) |
| `C4[0]*xy*(xx-yy)*sh[...,16] + ... + C4[8]*(...)*sh[...,24]` | $f_4=f_3+\sum_{k=0}^{8} C_{4,k}\cdot(\text{đa thức bậc 4})\cdot c_{16+k}$ (mục 2, bậc 4) |
| `(rgb - 0.5) / C0` | $c_{SH} = (\mathrm{RGB}-0.5)/C_0$ |
| `sh * C0 + 0.5` | $\mathrm{RGB} = c_{SH}\cdot C_0 + 0.5$ |
