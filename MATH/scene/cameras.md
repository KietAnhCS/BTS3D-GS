# Công thức toán học trong `scene/cameras.py`

Nguồn đã đọc:

- `scene/cameras.py` (lớp `Camera`, `MiniCam`)
- `utils/graphics_utils.py` (hàm `getWorld2View2`, `getProjectionMatrix` — được `Camera` gọi trực tiếp để dựng ma trận view/projection)

---

## 1. Ký hiệu

- $R\in SO(3)$: ma trận xoay thế giới→camera ước lượng từ COLMAP, lưu ở dạng **đã chuyển vị** theo quy ước của codebase 3DGS (xem §2).
- $T\in\mathbb R^3$: vector tịnh tiến thế giới→camera.
- $t_{\text{extra}}\in\mathbb R^3$ (`trans`): tịnh tiến bổ sung áp dụng lên tâm camera (mặc định $(0,0,0)$).
- $s$ (`scale`): hệ số co giãn áp dụng lên tâm camera (mặc định $1.0$).
- $W_{2}C\in\mathbb R^{4\times4}$: ma trận world-to-view (`world_view_transform`, đã chuyển vị).
- $P\in\mathbb R^{4\times4}$: ma trận chiếu phối cảnh OpenGL-style (`projection_matrix`, đã chuyển vị).
- $M_{\text{full}}=W_2C\cdot P$: ma trận world-to-clip hợp nhất (`full_proj_transform`).
- $c\in\mathbb R^3$: tâm camera trong không gian thế giới (`camera_center`).
- $\text{FoV}_x,\text{FoV}_y$: góc nhìn ngang/dọc theo radian.
- $z_{\text{near}}=0.01,\ z_{\text{far}}=100.0$: mặt cắt gần/xa của frustum (hằng số hard-code).
- $f_x,f_y$: tiêu cự tính bằng pixel (`focal_x`, `focal_y`).
- $I_{gt}\in[0,1]^{3\times H\times W}$: ảnh ground-truth sau khi clamp và nhân mask alpha.

---

## 2. Ma trận world-to-view — `getWorld2View2` (gọi từ `Camera.__init__`)

Trích nguyên văn lời gọi (`scene/cameras.py`, dòng 54):

```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
```

Trích nguyên văn định nghĩa hàm (`utils/graphics_utils.py`, dòng 38–49):

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

Diễn giải: hàm trước tiên dựng ma trận world-to-camera "thô"

$$
Rt=\begin{pmatrix} R^\top & T\\ 0^\top & 1\end{pmatrix}\in\mathbb R^{4\times4}
$$

(lưu ý: tham số đầu vào tên `R` nhưng bị chuyển vị lại — `R.transpose()` — nên quy ước thực sự trong code là $R$ được **lưu sẵn ở dạng camera-to-world** tại nơi gọi, và `getWorld2View2` chuyển nó về world-to-camera bằng $R^\top$).

Sau đó lấy nghịch đảo để ra camera-to-world, áp dụng tịnh tiến/co giãn **lên tâm camera** (không phải lên toàn ma trận):

$$
C2W = Rt^{-1},\qquad c_{\text{raw}} = C2W[:3,3]
$$

$$
c' = (c_{\text{raw}} + t_{\text{extra}})\cdot s
$$

rồi ghi $c'$ trở lại cột tịnh tiến của $C2W$ và nghịch đảo lần nữa để có world-to-view cuối cùng:

$$
W_2C^{\text{raw}} = C2W'^{-1},\qquad C2W'[:3,3]=c',\ C2W'[:3,:3]=C2W[:3,:3]
$$

Tại nơi gọi, kết quả còn bị **chuyển vị thêm một lần** (`.transpose(0,1)`, dòng 54), cho ra quy ước hàng-vector dùng xuyên suốt rasterizer:

$$
W_2C = \big(W_2C^{\text{raw}}\big)^\top
$$

Ý nghĩa hình học: một điểm thế giới $x_w\in\mathbb R^4$ (toạ độ thuần nhất) được chuyển sang không gian camera bằng tích **hàng-vector bên trái**: $x_{\text{cam}} = x_w^\top \cdot W_2C$ (khác với quy ước cột-vector $x_{\text{cam}}=W_2C^{\text{raw}}\cdot x_w$ thường gặp trong OpenGL/computer-vision cổ điển — đây là lý do code phải `.transpose(0,1)` tường minh).

---

## 3. Ma trận chiếu phối cảnh — `getProjectionMatrix` (gọi từ `Camera.__init__`)

Trích nguyên văn lời gọi (`scene/cameras.py`, dòng 48–49, 55):

```python
self.zfar = 100.0
self.znear = 0.01
...
self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
```

Trích nguyên văn định nghĩa hàm (`utils/graphics_utils.py`, dòng 51–71):

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

Với $\tan(\text{FoV}_x/2),\tan(\text{FoV}_y/2)$ xác định cạnh của mặt cắt gần (frustum symmetric quanh trục quang):

$$
\text{top}=z_{\text{near}}\tan\!\Big(\frac{\text{FoV}_y}{2}\Big),\quad \text{bottom}=-\text{top},\quad
\text{right}=z_{\text{near}}\tan\!\Big(\frac{\text{FoV}_x}{2}\Big),\quad \text{left}=-\text{right}
$$

vì frustum đối xứng ($\text{left}=-\text{right}$, $\text{top}=-\text{bottom}$), các số hạng $(\text{right}+\text{left})$ và $(\text{top}+\text{bottom})$ triệt tiêu về $0$ — tức $P[0,2]=P[1,2]=0$ trong mọi trường hợp thực tế của file này (không có camera lệch tâm/off-axis). Ma trận (dạng cột-vector chuẩn OpenGL, $z_{\text{sign}}=1$):

$$
P^{\text{raw}} = \begin{pmatrix}
\dfrac{2z_{\text{near}}}{\text{right}-\text{left}} & 0 & \dfrac{\text{right}+\text{left}}{\text{right}-\text{left}} & 0\\[2mm]
0 & \dfrac{2z_{\text{near}}}{\text{top}-\text{bottom}} & \dfrac{\text{top}+\text{bottom}}{\text{top}-\text{bottom}} & 0\\[2mm]
0 & 0 & \dfrac{z_{\text{far}}}{z_{\text{far}}-z_{\text{near}}} & -\dfrac{z_{\text{far}}z_{\text{near}}}{z_{\text{far}}-z_{\text{near}}}\\[2mm]
0 & 0 & 1 & 0
\end{pmatrix}
$$

Thay $\text{right}-\text{left}=2z_{\text{near}}\tan(\text{FoV}_x/2)$ và $\text{top}-\text{bottom}=2z_{\text{near}}\tan(\text{FoV}_y/2)$, hai phần tử đường chéo rút gọn còn:

$$
P^{\text{raw}}_{00}=\frac{1}{\tan(\text{FoV}_x/2)},\qquad P^{\text{raw}}_{11}=\frac{1}{\tan(\text{FoV}_y/2)}
$$

— đây chính là dạng chuẩn "cotangent" của ma trận chiếu phối cảnh OpenGL. Hàng cuối $P[3,2]=1$ (thay vì $0,0,-1,0$ như một số tài liệu) khiến $w_{\text{clip}}=z_{\text{view}}$ (độ sâu camera dùng trực tiếp làm hệ số chia phối cảnh).

Tại nơi gọi, ma trận này cũng bị chuyển vị (`.transpose(0,1)`, dòng 55) để khớp quy ước hàng-vector:

$$
P = \big(P^{\text{raw}}\big)^\top
$$

---

## 4. Ma trận world-to-clip hợp nhất và tâm camera

Trích nguyên văn (`scene/cameras.py`, dòng 56–57):

```python
self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
self.camera_center = self.world_view_transform.inverse()[3, :3]
```

$$
M_{\text{full}} = W_2C \cdot P
$$

(nhân ma trận theo đúng thứ tự hàng-vector: một điểm world $x_w^\top$ chiếu ra clip-space bằng $x_w^\top\cdot M_{\text{full}} = x_w^\top\cdot W_2C\cdot P$.)

Tâm camera trong không gian thế giới được lấy từ **hàng thứ 4** (chỉ số 3) của nghịch đảo $W_2C$ — vì $W_2C$ ở dạng hàng-vector nên cột tịnh tiến gốc trở thành hàng thứ 4 sau transpose:

$$
c = \big(W_2C^{-1}\big)[3,\,:3]
$$

---

## 5. Tiêu cự pixel từ góc nhìn (FoV → focal length)

Trích nguyên văn (`scene/cameras.py`, dòng 59–62):

```python
tan_fovx = np.tan(self.FoVx / 2.0)
tan_fovy = np.tan(self.FoVy / 2.0)
self.focal_y = self.image_height / (2.0 * tan_fovy)
self.focal_x = self.image_width / (2.0 * tan_fovx)
```

Công thức tiêu cự chuẩn của mô hình pinhole, suy ngược từ định nghĩa góc nhìn $\text{FoV}=2\arctan\!\big(\tfrac{\text{kích thước cảm biến}}{2f}\big)$:

$$
f_x = \frac{W}{2\tan(\text{FoV}_x/2)},\qquad f_y = \frac{H}{2\tan(\text{FoV}_y/2)}
$$

với $W=$ `image_width`, $H=$ `image_height` (lấy từ shape của `original_image`, xem §6).

---

## 6. Ảnh ground-truth: clamp và áp mask alpha

Trích nguyên văn (`scene/cameras.py`, dòng 39–46):

```python
self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
self.image_width = self.original_image.shape[2]
self.image_height = self.original_image.shape[1]

if gt_alpha_mask is not None:
    self.original_image *= gt_alpha_mask.to(self.data_device)
else:
    self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
```

$$
I_{gt} = \operatorname{clip}(I,\,0,\,1)\ \odot\ A
$$

với $A=$ `gt_alpha_mask` nếu có (nhân Hadamard theo từng pixel, broadcast trên 3 kênh màu), ngược lại $A=\mathbf 1^{1\times H\times W}$ (không thay đổi gì — phép nhân với toàn 1 chỉ để thống nhất luồng code, không có ý nghĩa toán học khác).

---

## 7. `MiniCam` — không tính lại ma trận, chỉ suy tâm camera

Trích nguyên văn (`scene/cameras.py`, dòng 64–75):

```python
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
```

`MiniCam` (dùng cho `network_gui.py`, xem `MATH/gaussian_renderer/network_gui.md`) **không** gọi `getWorld2View2`/`getProjectionMatrix` — nó nhận thẳng $W_2C$ và $M_{\text{full}}$ đã được client GUI tính sẵn (sau khi đổi quy ước trục, xem file đó), và chỉ suy lại tâm camera bằng đúng công thức ở §4:

$$
c = \big(W_2C^{-1}\big)[3,\,:3]
$$

---

## 8. Kiến thức nền tảng

**Quy ước hàng-vector vs cột-vector.** Trong đồ hoạ máy tính cổ điển (OpenGL gốc), điểm được biểu diễn là vector cột và phép biến đổi nhân bên trái: $x'=M x$. PyTorch/3DGS ở đây dùng quy ước ngược — vector hàng nhân bên phải: $x'^\top = x^\top M^\top$ — nên mọi ma trận dựng theo công thức OpenGL chuẩn đều phải `.transpose(0,1)` trước khi dùng, đúng như cả `world_view_transform` và `projection_matrix` đều bị transpose tại nơi gọi.

**Mô hình pinhole và FoV.** Với tiêu cự $f$ (pixel) và nửa kích thước cảm biến $d/2$ (pixel), góc nhìn thoả $\tan(\text{FoV}/2)=\dfrac{d/2}{f}$, suy ra $f = \dfrac{d}{2\tan(\text{FoV}/2)}$ — đúng công thức ở §5.

**Frustum đối xứng.** Khi camera không có "principal point offset" (quang tâm trùng tâm ảnh), frustum đối xứng quanh trục quang ⇒ $\text{left}=-\text{right},\ \text{bottom}=-\text{top}$, khiến các số hạng lệch tâm trong ma trận chiếu triệt tiêu.

**Vai trò $z_{\text{near}},z_{\text{far}}$ hard-code.** Khác với $R,T,\text{FoV}$ (đọc từ dữ liệu COLMAP mỗi camera), $z_{\text{near}}=0.01,\,z_{\text{far}}=100.0$ là hằng số cố định cho mọi scene trong `Camera.__init__` — giới hạn này ảnh hưởng tới độ chính xác số học của buffer độ sâu (depth non-linearity) nhưng không ảnh hưởng tới rasterization Gaussian (vốn không dùng z-buffer rời rạc).

---

## 9. Kiểm chứng tính đúng sai

| Thành phần | Kỳ vọng lý thuyết | Code | Khớp? |
|---|---|---|---|
| $Rt[:3,:3]=R^\top$ (world-to-cam từ cam-to-world lưu sẵn) | Đúng quy ước COLMAP (R lưu dạng cam→world cục bộ theo cột) | `Rt[:3, :3] = R.transpose()` (`graphics_utils.py` dòng 40) | **Khớp** |
| Tịnh tiến/co giãn áp dụng lên **tâm camera**, không lên cả ma trận | Đúng ý đồ "dịch chuyển/co giãn toàn bộ scene quanh tâm" dùng khi chuẩn hoá scale cảnh (`scene_scale`) | `cam_center = (cam_center + translate) * scale` rồi set lại trước khi nghịch đảo lần 2 (dòng 46–48) | **Khớp** |
| $P[0,0]=1/\tan(\text{FoV}_x/2)$ sau rút gọn | Công thức cotangent chuẩn OpenGL perspective | `P[0,0] = 2*znear/(right-left)` với `right-left=2*znear*tanHalfFovX` → rút gọn còn $1/\tan(\text{FoV}_x/2)$ | **Khớp** |
| $P[3,2]=1$ (không phải $-1$) | Quy ước $w_{\text{clip}}=z_{\text{view}}$ (dương, do $z_{\text{sign}}=1$) — ngược dấu quy ước OpenGL "nhìn theo trục $-z$" cổ điển nhưng nhất quán nội bộ với cách rasterizer CUDA dùng độ sâu dương | `z_sign = 1.0; P[3, 2] = z_sign` (dòng 62, 68) | **Khớp với quy ước nội bộ của 3DGS** (đã biết, không phải lỗi) |
| $f_x=W/(2\tan(\text{FoV}_x/2))$ | Suy ngược đúng định nghĩa FoV của mô hình pinhole | `self.focal_x = self.image_width / (2.0 * tan_fovx)` (dòng 62) | **Khớp** |
| `camera_center` lấy từ hàng 3 (0-based), không phải cột 3 | Hệ quả trực tiếp của quy ước hàng-vector: sau transpose, cột tịnh tiến gốc nằm ở hàng cuối | `self.world_view_transform.inverse()[3, :3]` (dòng 57) và tương tự trong `MiniCam` (dòng 75) | **Khớp, nhất quán giữa `Camera` và `MiniCam`** |
| `full_proj_transform = world_view_transform @ projection_matrix` (không phải chiều ngược) | Điểm world → view → clip, đúng thứ tự áp dụng phép biến đổi liên tiếp dưới quy ước hàng-vector | `self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))` (dòng 56) | **Khớp** |

**Kết luận:** không phát hiện sai sót toán học trong `scene/cameras.py`. Điểm dễ gây nhầm lẫn nhất (và đã được làm rõ ở trên) là việc **transpose hai lần ở hai nơi khác nhau** — một lần bên trong `getWorld2View2` (nghịch đảo hai lần quanh phép dịch tâm) và một lần bên ngoài tại `Camera.__init__` (`.transpose(0,1)`) để đổi từ quy ước cột-vector sang hàng-vector — dễ bị đọc nhầm là lỗi trùng lặp nếu không đối chiếu cả hai file nguồn.

---

## 10. Ví dụ số

Giả sử camera nhìn thẳng dọc trục $z$ thế giới, không xoay ($R=I_3$), đặt tại $T=(0,0,-5)$ (tức tâm camera cách gốc toạ độ $5$ đơn vị dọc trục $z$ âm theo quy ước $Rt$ world→cam), không có `trans`/`scale` bổ sung ($t_{\text{extra}}=0,\,s=1$), $\text{FoV}_x=\text{FoV}_y=90^\circ=\pi/2$ rad, ảnh $W=H=800$ px, $z_{\text{near}}=0.01,\,z_{\text{far}}=100.0$.

**Bước 1 — $Rt$ thô:**

$$
Rt=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&-5\\0&0&0&1\end{pmatrix}
$$

**Bước 2 — $C2W=Rt^{-1}$:** vì $R=I$, nghịch đảo chỉ đổi dấu tịnh tiến: $C2W[:3,3]=(0,0,5)$.

**Bước 3 — áp `translate`/`scale`:** $t_{\text{extra}}=0,\,s=1$ ⇒ không đổi: $c'=(0,0,5)$.

**Bước 4 — nghịch đảo lại ⇒ $W_2C^{\text{raw}}=Rt$ (không đổi vì bước 3 không thay đổi gì).** Sau `.transpose(0,1)`: vì $Rt$ đối xứng ở khối $3\times3$ và chỉ có cột tịnh tiến khác 0, transpose đưa $(0,0,-5)$ từ cột 3 sang **hàng** 3:

$$
W_2C=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&-5&1\end{pmatrix}
$$

**Kiểm tra `camera_center`:** $W_2C^{-1}$ phải cho lại $Rt^{-1}$ dạng hàng-vector, hàng thứ 4 (chỉ số 3), 3 cột đầu $=(0,0,5)$ — đúng bằng tâm camera kỳ vọng trong thế giới. $\Rightarrow c=(0,0,5)$. **Khớp trực giác hình học** (camera đặt tại $z=5$, nhìn về gốc toạ độ).

**Bước 5 — tiêu cự:** $\tan(45^\circ)=1$ ⇒

$$
f_x=f_y=\frac{800}{2\times1}=400\ \text{px}
$$

**Bước 6 — $P[0,0]=P[1,1]$:** $\text{right}-\text{left}=2\times0.01\times1=0.02$ ⇒ $P[0,0]=2\times0.01/0.02=1=1/\tan(45^\circ)$ — khớp công thức rút gọn ở §3.

**Bước 7 — $P[2,2],P[2,3]$:**

$$
P[2,2]=\frac{100}{100-0.01}=1.0001000,\qquad P[2,3]=-\frac{100\times0.01}{100-0.01}=-0.0100010
$$

Các giá trị này xác nhận trực tiếp công thức đã trích từ `getProjectionMatrix` dòng 69–70.
