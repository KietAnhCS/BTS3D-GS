# Công thức toán học trong `utils/graphics_utils.py`

Tài liệu tổng hợp cơ sở toán học của các hàm dựng ma trận camera, phép chiếu phối cảnh và các phép biến đổi toạ độ thuần nhất dùng xuyên suốt pipeline 3DGS, bám sát thứ tự xuất hiện trong code.

---

## 1. `BasicPointCloud`

Trích nguyên văn (`utils/graphics_utils.py`, dòng 17–20):

```python
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
```

Một cấu trúc dữ liệu thuần tuý (không có công thức), biểu diễn đám mây điểm khởi tạo:

$$
\mathcal{P} = \{(\mathbf{p}_i, \mathbf{c}_i, \mathbf{n}_i)\}_{i=1}^{N}, \qquad \mathbf{p}_i \in \mathbb{R}^3,\ \mathbf{c}_i \in \mathbb{R}^3,\ \mathbf{n}_i \in \mathbb{R}^3
$$

với $\mathbf{p}_i$ là toạ độ thế giới, $\mathbf{c}_i$ là màu RGB $\in[0,1]^3$, $\mathbf{n}_i$ là pháp tuyến.

---

## 2. Toạ độ thuần nhất và phép biến đổi điểm (`geom_transform_points`)

Trích nguyên văn (`utils/graphics_utils.py`, dòng 22–29):

```python
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)
```

Với một tập điểm $P\in\mathbb{R}^{N\times 3}$, mỗi điểm được nâng lên toạ độ thuần nhất (homogeneous coordinates) bằng cách thêm thành phần $w=1$ (dòng 24–25):

$$
\mathbf{x}_i^{hom} = \begin{bmatrix}\mathbf{x}_i \\ 1\end{bmatrix} \in \mathbb{R}^4
$$

Nhân với ma trận biến đổi $M\in\mathbb{R}^{4\times4}$ (`transf_matrix`), theo đúng quy ước nhân hàng trong code (`points_hom @ M`, dòng 26):

$$
\mathbf{y}_i^{hom} = M^\top \mathbf{x}_i^{hom} \quad \text{(dưới dạng hàng: } \mathbf{x}_i^{hom\top} M\text{)}
$$

Chia phối cảnh (perspective divide) bằng thành phần thứ 4, có cộng thêm epsilon chống chia 0 (dòng 28–29):

$$
\mathbf{y}_i = \frac{\mathbf{y}_{i,0:3}^{hom}}{\mathbf{y}_{i,3}^{hom} + \epsilon}, \qquad \epsilon = 10^{-7}
$$

---

## 3. Ma trận World-to-View (`getWorld2View`)

Trích nguyên văn (`utils/graphics_utils.py`, dòng 31–36):

```python
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)
```

Với ma trận xoay $R\in SO(3)$ và vector tịnh tiến $\mathbf{t}\in\mathbb{R}^3$ (từ hệ thế giới sang hệ camera), ma trận thuần nhất $4\times4$ (dòng 33–35):

$$
R_t =
\begin{bmatrix}
R^\top & \mathbf{t} \\
\mathbf{0}^\top & 1
\end{bmatrix}
$$

Đây chính là ma trận world-to-camera $W2C$, áp dụng cho điểm thế giới $\mathbf{x}_w$ (toạ độ thuần nhất) để được toạ độ camera:

$$
\mathbf{x}_{cam} = R_t\, \mathbf{x}_w = \begin{bmatrix} R^\top \mathbf{x}_w^{(3)} + \mathbf{t} \\ 1 \end{bmatrix}
$$

(code lưu $R^\top$ ở khối trên-trái vì quy ước COLMAP/3DGS: $R$ truyền vào là camera-to-world rotation, nên transpose cho world-to-camera).

---

## 4. Ma trận World-to-View có dịch/co tâm camera (`getWorld2View2`)

Trích nguyên văn (`utils/graphics_utils.py`, dòng 38–49):

```python
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
```

Cho phép dịch chuyển và co giãn lại "tâm cảnh" (scene center) — hữu ích khi chuẩn hoá scale của toàn bộ scene.

Bước 1 — dựng $R_t$ như mục 3 (dòng 40–42):

$$
R_t = \begin{bmatrix} R^\top & \mathbf{t} \\ \mathbf{0}^\top & 1\end{bmatrix}
$$

Bước 2 — nghịch đảo để lấy ma trận camera-to-world $C2W = R_t^{-1}$, trích tâm camera trong hệ thế giới (dòng 44–45):

$$
C2W = R_t^{-1}, \qquad \mathbf{c}_{cam} = C2W_{[0:3,\,3]}
$$

Bước 3 — áp dụng dịch chuyển (`translate`) và tỉ lệ (`scale`) lên tâm camera (dòng 46):

$$
\mathbf{c}_{cam}^{new} = \big(\mathbf{c}_{cam} + \mathbf{t}_{translate}\big)\cdot s
$$

Bước 4 — ghi lại vào $C2W$ rồi nghịch đảo ngược lại để có $W2C$ mới (dòng 47–48):

$$
C2W_{[0:3,\,3]} \leftarrow \mathbf{c}_{cam}^{new}, \qquad R_t^{new} = C2W^{-1}
$$

---

## 5. Ma trận chiếu phối cảnh (`getProjectionMatrix`)

Trích nguyên văn (`utils/graphics_utils.py`, dòng 51–71):

```python
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
```

Cho near plane $z_{near}$, far plane $z_{far}$, góc nhìn ngang/dọc $fov_X, fov_Y$ (radian, full-angle). Tang nửa góc (dòng 52–53):

$$
\tan\!\left(\frac{fov_Y}{2}\right), \qquad \tan\!\left(\frac{fov_X}{2}\right)
$$

Biên trên/dưới/phải/trái của frustum tại mặt phẳng near (symmetric frustum, dòng 55–58):

$$
top = \tan\!\left(\frac{fov_Y}{2}\right) z_{near}, \qquad bottom = -top
$$

$$
right = \tan\!\left(\frac{fov_X}{2}\right) z_{near}, \qquad left = -right
$$

Với $z_{sign}=1.0$ (dòng 62), ma trận chiếu $P\in\mathbb{R}^{4\times4}$ (OpenGL-style, hàng/cột theo đúng chỉ số code, $P[\text{row},\text{col}]$, dòng 64–70):

$$
P_{00} = \frac{2 z_{near}}{right-left}, \qquad
P_{11} = \frac{2 z_{near}}{top-bottom}
$$

$$
P_{02} = \frac{right+left}{right-left}, \qquad
P_{12} = \frac{top+bottom}{top-bottom}
$$

$$
P_{32} = z_{sign} = 1, \qquad
P_{22} = z_{sign}\cdot\frac{z_{far}}{z_{far}-z_{near}}
$$

$$
P_{23} = -\frac{z_{far}\, z_{near}}{z_{far}-z_{near}}
$$

Các phần tử còn lại bằng 0 (khởi tạo `torch.zeros(4,4)`, dòng 60). Áp dụng cho điểm camera-space $\mathbf{x}_{cam}=(x,y,z,1)^\top$ (nhân cột $P\,\mathbf{x}_{cam}$), kết quả clip-space có thành phần $w_{clip}=z$ (do $P_{32}=1$), dùng để chia phối cảnh sau đó:

$$
\mathbf{x}_{clip} = P\,\mathbf{x}_{cam}, \qquad \mathbf{x}_{ndc} = \mathbf{x}_{clip}\,/\,w_{clip}
$$

---

## 6. Chuyển đổi FOV ↔ tiêu cự (`fov2focal`, `focal2fov`)

Trích nguyên văn (`utils/graphics_utils.py`, dòng 73–77):

```python
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
```

Quan hệ hình học pinhole camera giữa trường nhìn (field of view, radian) và tiêu cự tính theo pixel, với $pixels$ là kích thước ảnh (chiều rộng hoặc cao tương ứng, dòng 73–74):

$$
f = \frac{pixels}{2\tan\!\left(\dfrac{fov}{2}\right)}
$$

Nghịch đảo (dòng 76–77):

$$
fov = 2\arctan\!\left(\frac{pixels}{2f}\right)
$$

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `points_hom = torch.cat([points, ones], dim=1)` | $\mathbf{x}_i^{hom} = \begin{bmatrix}\mathbf{x}_i \\ 1\end{bmatrix}$ |
| `points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))` | $\mathbf{y}_i^{hom} = \mathbf{x}_i^{hom\top} M$ |
| `denom = points_out[..., 3:] + 0.0000001` | $w' = y^{hom}_{i,3} + 10^{-7}$ |
| `points_out[..., :3] / denom` | $\mathbf{y}_i = \mathbf{y}_{i,0:3}^{hom} / w'$ |
| `Rt[:3, :3] = R.transpose()` | $R_t^{(0:3,0:3)} = R^\top$ |
| `Rt[:3, 3] = t` | $R_t^{(0:3,3)} = \mathbf{t}$ |
| `Rt[3, 3] = 1.0` | $R_t^{(3,3)} = 1$ |
| `getWorld2View(R, t)` | $R_t = \begin{bmatrix} R^\top & \mathbf{t} \\ \mathbf{0}^\top & 1\end{bmatrix}$ |
| `C2W = np.linalg.inv(Rt)` | $C2W = R_t^{-1}$ |
| `cam_center = C2W[:3, 3]` | $\mathbf{c}_{cam} = C2W_{[0:3,3]}$ |
| `cam_center = (cam_center + translate) * scale` | $\mathbf{c}_{cam}^{new} = (\mathbf{c}_{cam}+\mathbf{t}_{translate})\cdot s$ |
| `C2W[:3, 3] = cam_center` | $C2W_{[0:3,3]} \leftarrow \mathbf{c}_{cam}^{new}$ |
| `Rt = np.linalg.inv(C2W)` | $R_t^{new} = C2W^{-1}$ |
| `tanHalfFovY = math.tan((fovY / 2))` | $\tan(fov_Y/2)$ |
| `tanHalfFovX = math.tan((fovX / 2))` | $\tan(fov_X/2)$ |
| `top = tanHalfFovY * znear` | $top = \tan(fov_Y/2)\,z_{near}$ |
| `bottom = -top` | $bottom=-top$ |
| `right = tanHalfFovX * znear` | $right=\tan(fov_X/2)\,z_{near}$ |
| `left = -right` | $left=-right$ |
| `z_sign = 1.0` | $z_{sign}=1$ |
| `P[0, 0] = 2.0 * znear / (right - left)` | $P_{00}=\dfrac{2z_{near}}{right-left}$ |
| `P[1, 1] = 2.0 * znear / (top - bottom)` | $P_{11}=\dfrac{2z_{near}}{top-bottom}$ |
| `P[0, 2] = (right + left) / (right - left)` | $P_{02}=\dfrac{right+left}{right-left}$ |
| `P[1, 2] = (top + bottom) / (top - bottom)` | $P_{12}=\dfrac{top+bottom}{top-bottom}$ |
| `P[3, 2] = z_sign` | $P_{32}=1$ |
| `P[2, 2] = z_sign * zfar / (zfar - znear)` | $P_{22}=\dfrac{z_{far}}{z_{far}-z_{near}}$ |
| `P[2, 3] = -(zfar * znear) / (zfar - znear)` | $P_{23}=-\dfrac{z_{far}z_{near}}{z_{far}-z_{near}}$ |
| `fov2focal(fov, pixels)` | $f = \dfrac{pixels}{2\tan(fov/2)}$ |
| `focal2fov(focal, pixels)` | $fov = 2\arctan\!\left(\dfrac{pixels}{2f}\right)$ |
