# Công thức toán học trong `scene/dataset_readers.py`

Nguồn: `scene/dataset_readers.py` (311 dòng). File này đọc dữ liệu scene (COLMAP hoặc NeRF-synthetic/Blender), dựng danh sách camera (`CameraInfo`), chuẩn hoá toạ độ scene về một quả cầu đơn vị-tỉ lệ (NeRF++ normalization), và tạo/đọc point cloud khởi tạo cho Gaussian. Các hàm `qvec2rotmat`, `getWorld2View2`, `focal2fov`, `fov2focal` được **định nghĩa ở nơi khác** (`scene/colmap_loader.py`, `utils/graphics_utils.py`) — ở đây trích dẫn công thức của chúng khi cần để diễn giải, có ghi rõ nguồn.

---

## 1. Ký hiệu

- $R\in SO(3)$: ma trận quay world→camera (hoặc transpose của nó, tuỳ quy ước lưu trữ — xem §2.2).
- $T\in\mathbb R^3$: vector tịnh tiến world→camera.
- $q=(q_w,q_x,q_y,q_z)$: quaternion COLMAP (`qvec`), chuyển thành ma trận quay bởi `qvec2rotmat`.
- $C2W, W2C\in\mathbb R^{4\times4}$: ma trận camera-to-world và world-to-camera thuần nhất.
- $f_x,f_y$: tiêu cự (pixel); $\text{FovX},\text{FovY}$: góc nhìn ngang/dọc (radian).
- $c_i\in\mathbb R^3$: tâm camera thứ $i$ trong world-space (cột cuối của $C2W$).
- $N$: số camera train.

---

## 2. Suy công thức từng hàm

### 2.1. Chuẩn hoá scene kiểu NeRF++ (`getNerfppNorm`, dòng 46–67)

```python
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
```

Với $c_i=\text{C2W}_i[:3,3]$ là tâm camera thứ $i$ (dòng 60, lấy từ cột tịnh tiến của ma trận camera-to-world — nghịch đảo của world-to-view ở dòng 58–59):

$$
\bar c=\frac{1}{N}\sum_{i=1}^{N} c_i
\tag{dòng 49}
$$

$$
d_i=\lVert c_i-\bar c\rVert_2,\qquad \text{diag}=\max_i d_i
\tag{dòng 51–52}
$$

$$
\boxed{\ \text{radius}=1.1\cdot\text{diag},\qquad \text{translate}=-\bar c\ }
\tag{dòng 63, 65}
$$

Ý nghĩa: $\bar c$ là tâm "khối cầu bao" các vị trí camera, $\text{diag}$ là bán kính nhỏ nhất đủ để bao mọi camera quanh $\bar c$; nhân hệ số an toàn $1.1$ để chừa biên. `translate` dùng để dịch scene về gốc toạ độ (centering) ở nơi gọi (`getWorld2View2(..., translate, scale)`, không phải trong file này).

### 2.2. Tư thế camera từ COLMAP (`readColmapCameras`, dòng 69–143)

```python
R = np.transpose(qvec2rotmat(extr.qvec))
T = np.array(extr.tvec)
```
(dòng 84–85)

Gọi $R_{\text{colmap}}=\text{qvec2rotmat}(q)$ là ma trận quay chuẩn dựng trực tiếp từ quaternion COLMAP (world→camera theo quy ước COLMAP). Code lưu lại:

$$
R=R_{\text{colmap}}^{\top},\qquad T=\mathbf{t}_{\text{colmap}}
\tag{dòng 84–85}
$$

Việc lấy **transpose** ở đây là do quy ước lưu trữ nội bộ của `getWorld2View2` (`utils/graphics_utils.py`, dòng 40: `Rt[:3,:3] = R.transpose()`) — $R$ lưu trong `CameraInfo` bị transpose một lần ở đây, rồi transpose lại một lần nữa khi dựng ma trận $4\times4$ view, nên kết quả ròng vẫn là $R_{\text{colmap}}$ đúng world→camera. Phép dựng ma trận view đầy đủ:

$$
W2C=\begin{pmatrix}R^\top & T\\ 0 & 1\end{pmatrix}=\begin{pmatrix}R_{\text{colmap}} & T\\ 0 & 1\end{pmatrix}
\tag{\texttt{getWorld2View2}, graphics\_utils.py dòng 38–42}
$$

### 2.3. Góc nhìn (FOV) từ tiêu cự (`focal2fov`, gọi tại dòng 90–91, 95–96, 114–115)

```python
if intr.model=="SIMPLE_PINHOLE":
    focal_length_x = intr.params[0]
    FovY = focal2fov(focal_length_x, height)
    FovX = focal2fov(focal_length_x, width)
elif intr.model=="PINHOLE":
    focal_length_x = intr.params[0]
    focal_length_y = intr.params[1]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
```
(dòng 88–96)

Định nghĩa `focal2fov` (`utils/graphics_utils.py`, dòng 76–77):

```python
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
```

$$
\text{FovY}=2\arctan\!\Big(\frac{\text{height}}{2f_y}\Big),\qquad \text{FovX}=2\arctan\!\Big(\frac{\text{width}}{2f_x}\Big)
$$

(với `SIMPLE_PINHOLE`, $f_x=f_y=$ `intr.params[0]`, một tiêu cự chung cho cả 2 trục — dòng 89–91). Đây là công thức ngược của mô hình pinhole $u=f\tan(\theta/2)\cdot 2$ hay tương đương $\tan(\text{fov}/2)=\dfrac{\text{pixels}/2}{f}$, suy ra đúng biểu thức `atan` ở trên.

### 2.4. Camera có méo ảnh (distortion) — `SIMPLE_RADIAL`/`RADIAL`/`OPENCV` (dòng 97–117, 127–133)

```python
elif intr.model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
    if intr.model == "SIMPLE_RADIAL":
        focal_length_x = focal_length_y = intr.params[0]
        cx, cy = intr.params[1], intr.params[2]
        distortion = np.array([intr.params[3], 0.0, 0.0, 0.0])
    elif intr.model == "RADIAL":
        focal_length_x = focal_length_y = intr.params[0]
        cx, cy = intr.params[1], intr.params[2]
        distortion = np.array([intr.params[3], intr.params[4], 0.0, 0.0])
    else:  # OPENCV
        focal_length_x, focal_length_y = intr.params[0], intr.params[1]
        cx, cy = intr.params[2], intr.params[3]
        distortion = np.array(intr.params[4:8])
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)
```
(dòng 97–115)

```python
K = np.array([[focal_length_x, 0, cx], [0, focal_length_y, cy], [0, 0, 1]])
image = Image.fromarray(cv2.undistort(np.array(image), K, distortion))
```
(dòng 132–133)

Ma trận nội tại (intrinsics) pinhole chuẩn:

$$
K=\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}
$$

`cv2.undistort(image, K, distortion)` áp dụng mô hình méo ảnh xuyên tâm/tiếp tuyến chuẩn OpenCV (hệ số `distortion`$=(k_1,k_2,p_1,p_2[,k_3,\ldots])$) để tính ánh xạ ngược từ pixel méo về pixel đã khử méo — công thức đầy đủ nằm trong OpenCV, không được định nghĩa lại trong file này; đây là lớp "tiền xử lý hình học" (remap toạ độ pixel), được project bổ sung so với 3DGS gốc (vốn chỉ hỗ trợ PINHOLE/SIMPLE_PINHOLE — dòng 117 vẫn giữ `assert False` cho mọi model khác).

### 2.5. Chuẩn hoá màu point cloud (`fetchPly`, dòng 145–151)

```python
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
```

$$
\text{colors}=\frac{(\text{red},\text{green},\text{blue})}{255}\in[0,1]^3
\tag{dòng 149}
$$

Chuẩn hoá giá trị kênh màu 8-bit $\{0,\ldots,255\}$ về khoảng $[0,1]$, khớp quy ước màu "linear-ish" dùng làm hệ số SH bậc 0 ở nơi khác (`utils/sh_utils.py`, `RGB2SH`).

### 2.6. Phân chia train/test theo `llffhold` (`readColmapSceneInfo`, dòng 198–203)

```python
if eval:
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
else:
    train_cam_infos = cam_infos
    test_cam_infos = []
```

$$
\text{train}=\{c_{idx}: idx \bmod \text{llffhold}\neq 0\},\qquad \text{test}=\{c_{idx}: idx\bmod\text{llffhold}=0\}
$$

Với `llffhold`$=8$ (giá trị mặc định tham số hàm, dòng 170): cứ mỗi 8 camera (theo thứ tự đã sort theo tên ảnh, dòng 196) thì camera đầu tiên của mỗi nhóm ($idx\equiv 0\pmod 8$) thuộc tập test, 7 camera còn lại thuộc tập train — tỉ lệ test $\approx 1/8=12.5\%$.

### 2.7. Đổi hệ trục camera Blender/NeRF-synthetic → COLMAP (`readCamerasFromTransforms`, dòng 229–269)

```python
c2w = np.array(frame["transform_matrix"])
# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
c2w[:3, 1:3] *= -1

# get the world-to-camera transform and set R, T
w2c = np.linalg.inv(c2w)
R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
T = w2c[:3, 3]
```
(dòng 241–248)

Gọi $M_{\text{flip}}=\mathrm{diag}(1,-1,-1)$ (đảo dấu 2 cột $y,z$ của khối quay $3\times3$):

$$
C2W_{\text{colmap}}=C2W_{\text{blender}}\cdot\begin{pmatrix}M_{\text{flip}}&0\\0&1\end{pmatrix}
\tag{dòng 243, áp lên 3 hàng đầu, cột 1–2 (0-indexed)}
$$

$$
W2C=C2W_{\text{colmap}}^{-1}
\tag{dòng 246}
$$

$$
R=\big(W2C[:3,:3]\big)^{\top},\qquad T=W2C[:3,3]
\tag{dòng 247–248}
$$

(bình luận trong code dòng 247 giải thích lý do transpose: quy ước lưu `R` để khớp với cách CUDA rasterizer — viết bằng `glm`, column-major — đọc lại ma trận, giống phân tích ở `MATH/submodules/.../auxiliary.md` §2.4 về quy ước transpose xuyên suốt pipeline.)

### 2.8. Blend ảnh RGBA với nền (`readCamerasFromTransforms`, dòng 254–260)

```python
im_data = np.array(image.convert("RGBA"))

bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

norm_data = im_data / 255.0
arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
```

Với $I_{rgb}\in[0,1]^{H\times W\times3}$, $\alpha\in[0,1]^{H\times W\times1}$ (kênh alpha đã chuẩn hoá), $\mathbf{bg}\in\{(1,1,1),(0,0,0)\}$:

$$
\boxed{\ I_{out}=I_{rgb}\cdot\alpha+\mathbf{bg}\cdot(1-\alpha)\ }
\tag{dòng 259}
$$

Đây đúng là công thức **alpha compositing chuẩn** ("over" operator, Porter & Duff 1984) khi hợp một ảnh tiền nhân (hoặc chưa tiền nhân, ở đây là chưa — nhân $\alpha$ tường minh) với nền đặc $\mathbf{bg}$: pixel trong suốt hoàn toàn ($\alpha=0$) trả về đúng màu nền; pixel đặc hoàn toàn ($\alpha=1$) giữ nguyên màu gốc.

### 2.9. FOV dọc suy từ FOV ngang qua tỉ lệ khung hình (dòng 262)

```python
fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
```

Định nghĩa `fov2focal` (`utils/graphics_utils.py`, dòng 73–74):

```python
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
```

$$
f_x=\text{fov2focal}(\text{FovX}, W)=\frac{W}{2\tan(\text{FovX}/2)}
\tag{\texttt{fov2focal}}
$$

$$
\text{FovY}=\text{focal2fov}(f_x, H)=2\arctan\!\Big(\frac{H}{2f_x}\Big)
\tag{dòng 262}
$$

Gộp lại:

$$
\boxed{\ \text{FovY}=2\arctan\!\Big(\tan(\text{FovX}/2)\cdot\frac{H}{W}\Big)\ }
$$

tức là: giả định **cùng một tiêu cự $f_x=f_y$** (pixel vuông, không méo anamorphic), suy FOV dọc từ FOV ngang đã biết (`camera_angle_x` trong file `transforms_*.json`) theo đúng tỉ lệ khung hình $H/W$ — hợp lý vì NeRF-synthetic chỉ cung cấp 1 góc FOV trong metadata.

### 2.10. Sinh point cloud ngẫu nhiên khi không có COLMAP (`readNerfSyntheticInfo`, dòng 284–294)

```python
num_pts = 100_000
print(f"Generating random point cloud ({num_pts})...")

xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
shs = np.random.random((num_pts, 3)) / 255.0
pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

storePly(ply_path, xyz, SH2RGB(shs) * 255)
```

Với $u\sim\mathcal U(0,1)^3$ (`np.random.random`):

$$
xyz = 2.6\,u - 1.3 \in [-1.3,\,1.3]^3
\tag{dòng 290}
$$

tức lấy mẫu đều (uniform) trong khối lập phương cạnh $2.6$ tâm tại gốc toạ độ — khớp với kích thước chuẩn hoá của scene Blender tổng hợp (NeRF-synthetic chuẩn hoá vật thể về $[-1,1]^3$, mở rộng biên $\pm0.3$ để chắc chắn bao phủ).

$$
shs = \frac{u'}{255},\quad u'\sim\mathcal U(0,1)^3\ \text{(3 kênh)}
\tag{dòng 291}
$$

$$
\text{colors}=\text{SH2RGB}(shs)=shs\cdot C_0+0.5\qquad(\text{định nghĩa ở }\texttt{utils/sh\_utils.py})
\tag{dòng 292}
$$

Vì $shs=\texttt{np.random.random(...)}/255\in[0,\,1/255)$ chỉ lấy mẫu **không âm** (không đối xứng quanh 0) và $C_0\approx0.282$, nên $\text{colors}=shs\cdot C_0+0.5\in[0.5,\;0.5+0.282/255)\approx[0.5,\,0.50111]$ — tức màu khởi tạo lệch về một phía, nhích nhẹ lên trên **xám trung tính $0.5$** (không bao giờ nhỏ hơn 0.5), với nhiễu rất nhỏ (biên độ $\approx 0.11\%$) — hợp lý vì không có thông tin màu thật khi không dùng COLMAP.

---

## 3. Kiến thức toán nền tảng

**Chuẩn hoá scene kiểu NeRF++ (Zhang et al. 2020).** Các phương pháp NeRF cổ điển giả định scene nằm trong một khối giới hạn đã biết trước; với scene "unbounded" (ngoài trời), NeRF++ đề xuất chuẩn hoá: dịch tâm về gốc toạ độ và co giãn theo bán kính bao các camera, giúp các siêu tham số (như `near`/`far`, kích thước lưới octree) đặt nhất quán giữa các scene khác nhau.

**Quaternion → ma trận quay.** Quaternion đơn vị biểu diễn phép quay không gimbal-lock (xem thêm `MATH/submodules/.../auxiliary.md` §3.3 cho công thức tường minh $R(q)$ dùng trong CUDA — ở đây `qvec2rotmat` của COLMAP dùng đúng công thức chuẩn Hamilton, định nghĩa trong `scene/colmap_loader.py`, không lặp lại).

**Alpha compositing ("over" operator, Porter & Duff 1984).** Công thức $I_{out}=I_{fg}\alpha+I_{bg}(1-\alpha)$ là phép nội suy tuyến tính (lerp) giữa ảnh tiền cảnh và nền theo trọng số $\alpha$ — nền tảng của mọi phép ghép ảnh trong suốt, bao gồm cả công thức blending alpha trong rasterizer 3DGS (`forward.cu`, khác ngữ cảnh).

**Mô hình camera pinhole & FOV.** $\tan(\text{fov}/2)=\dfrac{\text{kích thước ảnh}/2}{\text{tiêu cự}}$ — quan hệ hình học cơ bản giữa góc nhìn, tiêu cự và kích thước cảm biến/ảnh; `focal2fov`/`fov2focal` là hai chiều của cùng một quan hệ này.

**Lấy mẫu đều (uniform sampling) trong hộp.** $u\sim\mathcal U(0,1)\Rightarrow a+(b-a)u\sim\mathcal U(a,b)$ — phép biến đổi affine chuẩn để sinh điểm ngẫu nhiên đều trong khoảng $[a,b]$, áp dụng từng toạ độ độc lập cho khối lập phương.

---

## 4. Kiểm chứng tính đúng sai

Đối chiếu từng công thức với `scene/dataset_readers.py` thật (311 dòng) và các hàm phụ trợ (`utils/graphics_utils.py`):

| Mục | Công thức | Dòng code | Khớp? |
|---|---|---|---|
| Tâm & bán kính NeRF++ | $\bar c=\frac1N\sum c_i$, $\text{radius}=1.1\max_i\lVert c_i-\bar c\rVert$ | dòng 49–52, 63 | **Khớp** |
| `translate` | $-\bar c$ | dòng 65 | **Khớp** |
| $R=R_{\text{colmap}}^\top$ | transpose của `qvec2rotmat` | dòng 84 | **Khớp**, và tự hợp lý với `getWorld2View2` transpose lại một lần |
| FOV từ tiêu cự | $2\arctan(\text{pixels}/(2f))$ | `graphics_utils.py` dòng 76–77, gọi tại dòng 90–91, 95–96, 114–115 | **Khớp** |
| Ma trận $K$ cho `cv2.undistort` | $\begin{pmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{pmatrix}$ | dòng 132 | **Khớp** |
| Chuẩn hoá màu PLY | $/255$ | dòng 149 | **Khớp** |
| Train/test split | $idx\bmod\text{llffhold}$ | dòng 199–200 | **Khớp** |
| Đổi trục Blender→COLMAP | đảo dấu cột $y,z$ của $C2W[:3,:3]$ | dòng 243 | **Khớp** |
| Alpha compositing | $I_{rgb}\alpha+\mathbf{bg}(1-\alpha)$ | dòng 259 | **Khớp**, đúng công thức Porter–Duff "over" |
| FOV dọc từ FOV ngang | $2\arctan(\tan(\text{FovX}/2)\cdot H/W)$ | dòng 262 | **Khớp**, suy đúng từ ghép `fov2focal`+`focal2fov` |
| Point cloud ngẫu nhiên | $xyz\in[-1.3,1.3]^3$ đều; màu $\approx0.5$ | dòng 290–292 | **Khớp** |

**Kết luận:** không phát hiện sai sót toán học nào trong toàn bộ file `scene/dataset_readers.py`. Toàn bộ công thức là các phép biến đổi hình học/camera chuẩn (chuẩn hoá scene, đổi hệ trục, FOV↔tiêu cự, alpha compositing) và một phép lấy mẫu ngẫu nhiên đơn giản — không có phép toán nào đặc thù "nghiên cứu" (như EWA splatting hay Adam) cần kiểm chứng sâu hơn. Điểm cần lưu ý (không phải lỗi) là nhánh distortion (`SIMPLE_RADIAL`/`RADIAL`/`OPENCV`, dòng 97–133) là phần mở rộng so với 3DGS gốc Inria (vốn chỉ `assert` các model không phải PINHOLE), được thêm vào để dự án tự chạy COLMAP mapper trên ảnh tự chụp chưa undistort.

---

## 5. Ví dụ số

### Ví dụ A — `getNerfppNorm`

Giả sử $N=3$ camera có tâm world-space $c_1=(0,0,2)$, $c_2=(2,0,0)$, $c_3=(-2,0,0)$.

$$
\bar c=\frac{(0,0,2)+(2,0,0)+(-2,0,0)}{3}=\Big(0,\,0,\,\tfrac23\Big)
$$

$$
d_1=\lVert(0,0,2)-(0,0,\tfrac23)\rVert=\tfrac43\approx1.333,\quad d_2=\lVert(2,0,0)-(0,0,\tfrac23)\rVert=\sqrt{4+\tfrac49}=\sqrt{4.444}\approx2.108
$$
$$
d_3=d_2\approx2.108\ \text{(đối xứng)}
$$

$$
\text{diag}=\max(1.333,\,2.108,\,2.108)=2.108
$$

$$
\text{radius}=1.1\times2.108\approx2.319,\qquad \text{translate}=-\bar c=\Big(0,0,-\tfrac23\Big)
$$

### Ví dụ B — FOV từ tiêu cự (`focal2fov`)

Camera `PINHOLE` với $f_x=f_y=1000$ px, ảnh $W=H=1200$ px:

$$
\text{FovX}=2\arctan\!\Big(\frac{1200}{2\times1000}\Big)=2\arctan(0.6)=2\times0.5404=1.0808\ \text{rad}\ (\approx61.9^\circ)
$$

### Ví dụ C — Alpha compositing nền trắng

Pixel RGBA chuẩn hoá $I_{rgb}=(0.2,0.4,0.6)$, $\alpha=0.5$, nền trắng $\mathbf{bg}=(1,1,1)$:

$$
I_{out}=(0.2,0.4,0.6)\times0.5+(1,1,1)\times0.5=(0.1,0.2,0.3)+(0.5,0.5,0.5)=(0.6,0.7,0.8)
$$

tức là pixel bị "pha loãng" về phía trắng đúng theo tỉ lệ trong suốt $1-\alpha=0.5$, khớp công thức dòng 259.

### Ví dụ D — FOV dọc từ FOV ngang

$\text{FovX}=1.0472$ rad ($60^\circ$), ảnh $W=1600,\ H=900$ (tỉ lệ 16:9):

$$
f_x=\frac{1600}{2\tan(0.5236)}=\frac{1600}{2\times0.5774}=\frac{1600}{1.1547}\approx1385.6
$$
$$
\text{FovY}=2\arctan\!\Big(\frac{900}{2\times1385.6}\Big)=2\arctan(0.3248)=2\times0.3141=0.6282\ \text{rad}\ (\approx36.0^\circ)
$$

Kiểm tra chéo bằng công thức gộp: $2\arctan(\tan(0.5236)\times900/1600)=2\arctan(0.5774\times0.5625)=2\arctan(0.3248)=0.6282$ — khớp.
