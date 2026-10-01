# Pipeline đọc dữ liệu Scene: `colmap_loader.py`, `dataset_readers.py`, `cameras.py`, `scene/__init__.py`

4 file này tạo thành pipeline: đọc dữ liệu thô COLMAP/Blender → dựng `CameraInfo`/
`SceneInfo` → dựng object `Camera` dùng khi train/render → quản lý vòng đời `Scene`
(load, save, lấy danh sách camera). Hầu hết hàm **không có công thức toán** (chỉ là
I/O, parsing binary/text, đóng gói dữ liệu) — tài liệu này liệt kê rõ hàm nào có công
thức và hàm nào thuần I/O.

---

# A. `scene/colmap_loader.py` — 11 hàm + 1 class

Đọc file COLMAP (`cameras.bin/txt`, `images.bin/txt`, `points3D.bin/txt`) ở định dạng
nhị phân/text gốc của COLMAP.

## 1. `qvec2rotmat(qvec)` (dòng 43–53)

Chuyển quaternion COLMAP $q=(q_w,q_x,q_y,q_z)$ thành ma trận quay 3×3 (công thức
quaternion → rotation matrix chuẩn):

$$
R=\begin{bmatrix}
1-2q_y^2-2q_z^2 & 2q_xq_y-2q_wq_z & 2q_zq_x+2q_wq_y\\
2q_xq_y+2q_wq_z & 1-2q_x^2-2q_z^2 & 2q_yq_z-2q_wq_x\\
2q_zq_x-2q_wq_y & 2q_yq_z+2q_wq_x & 1-2q_x^2-2q_y^2
\end{bmatrix}
$$

(giả định $\lVert q\rVert=1$; đây là quaternion đơn vị ⇒ rotation matrix trực giao).

## 2. `rotmat2qvec(R)` (dòng 55–66)

Chiều ngược lại: dựng ma trận đối xứng $K$ 4×4 từ 9 phần tử của $R$:

$$
K=\frac13\begin{bmatrix}
R_{xx}-R_{yy}-R_{zz} & 0 & 0 & 0\\
R_{yx}+R_{xy} & R_{yy}-R_{xx}-R_{zz} & 0 & 0\\
R_{zx}+R_{xz} & R_{zy}+R_{yz} & R_{zz}-R_{xx}-R_{yy} & 0\\
R_{yz}-R_{zy} & R_{zx}-R_{xz} & R_{xy}-R_{yx} & R_{xx}+R_{yy}+R_{zz}
\end{bmatrix}
$$

Quaternion tương ứng với $R$ là **eigenvector ứng với trị riêng lớn nhất** của $K$
(phương pháp Markley, ổn định số học hơn công thức Shepperd cổ điển):

$$
K\,q = \lambda_{max}\,q,\qquad q = \arg\max_{\lambda}\ \text{eigvec}(K)
$$

sau đó chuẩn hoá dấu để $q_w \ge 0$.

## 3. `Image.qvec2rotmat(self)` (dòng 68–70)

Method tiện ích gọi lại hàm #1 trên `self.qvec`. Không công thức mới.

## 4. `read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<")` (dòng 72–81)

Đọc và giải mã (`struct.unpack`) một chuỗi byte nhị phân theo định dạng C struct. Thuần I/O.

## 5. `read_points3D_text(path)` (dòng 83–123)

Đọc file text `points3D.txt` của COLMAP (`ID x y z r g b error ...`) thành 3 mảng
NumPy `xyz (N,3)`, `rgb (N,3)`, `error (N,1)`. Thuần parsing, không công thức.

## 6. `read_points3D_binary(path_to_model_file)` (dòng 125–154)

Tương tự #5 nhưng đọc định dạng nhị phân COLMAP (struct format `"QdddBBBd"` cho mỗi
điểm + track). Thuần I/O nhị phân.

## 7. `read_intrinsics_text(path)` (dòng 156–178)

Đọc `cameras.txt`, chỉ chấp nhận model `PINHOLE` (4 tham số $f_x, f_y, c_x, c_y$).
Thuần parsing.

## 8. `read_extrinsics_binary(path_to_model_file)` (dòng 180–212)

Đọc `images.bin`: với mỗi ảnh, đọc quaternion $q$, tịnh tiến $t$, `camera_id`, tên
ảnh (chuỗi kết thúc bằng `\x00`), và danh sách điểm 2D `(x,y,point3D_id)`. Thuần I/O.

## 9. `read_intrinsics_binary(path_to_model_file)` (dòng 215–241)

Đọc `cameras.bin`: với mỗi camera, đọc `model_id`, `width`, `height`, rồi đọc đúng
`num_params` tham số nội tại tương ứng model đó (tra bảng `CAMERA_MODEL_IDS`). Thuần I/O.

## 10. `read_extrinsics_text(path)` (dòng 244–270)

Bản text của #8 (`images.txt`, mỗi ảnh chiếm 2 dòng: dòng pose + dòng điểm 2D). Thuần parsing.

## 11. `read_colmap_bin_array(path)` (dòng 273–295)

Đọc bản đồ độ sâu/pháp tuyến dày đặc (`.bin` của COLMAP dense — `read_dense.py`),
header dạng text `"width&height&channels&"` rồi dữ liệu `float32` dạng Fortran-order,
chuyển về C-order bằng `transpose(1,0,2)`. Thuần I/O, không công thức.

---

# B. `scene/dataset_readers.py` — 10 hàm

Chuyển dữ liệu thô (COLMAP hoặc Blender/NeRF-synthetic) thành `CameraInfo` / `SceneInfo`
dùng chung cho pipeline train.

## 1. `getNerfppNorm(cam_info)` (dòng 46–67)

Tính tâm và bán kính chuẩn hoá cảnh (theo NeRF++), dùng làm `cameras_extent` — quy mô
không gian để đặt ngưỡng densify ở `gaussian_model.py`.

Với tâm camera $C_i$ (world-to-camera nghịch đảo, cột cuối của $C2W$):

$$
\bar C = \dfrac1N\sum_i C_i
$$

$$
d_i = \lVert C_i - \bar C\rVert_2,\qquad
\text{diag} = \max_i d_i
$$

$$
\text{radius} = 1.1 \cdot \text{diag}, \qquad \text{translate} = -\bar C
$$

Hàm con `get_center_and_diag(cam_centers)` thực hiện đúng 2 công thức trên.

## 2. `readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder)` (dòng 69–143)

Với mỗi ảnh COLMAP: lấy `R = qvec2rotmat(q)^T` (transpose vì CUDA rasterizer lưu R
theo quy ước glm/column-major), `T = t`. Tính trường nhìn (FOV) từ tiêu cự bằng:

$$
\text{FoVx} = 2\arctan\!\left(\dfrac{W}{2f_x}\right), \qquad
\text{FoVy} = 2\arctan\!\left(\dfrac{H}{2f_y}\right)
$$

(gọi hàm `focal2fov` ở `utils/graphics_utils.py`, xem mục D).

Với camera có méo ảnh (`SIMPLE_RADIAL/RADIAL/OPENCV`), xây ma trận nội tại

$$
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}
$$

và gọi `cv2.undistort(image, K, distCoeffs)` để khử méo ảnh (phép biến đổi hình học,
không có công thức closed-form viết tường minh trong file — nằm trong OpenCV). Nếu
file ảnh bị thiếu trên đĩa, camera đó được bỏ qua (không crash).

## 3. `fetchPly(path)` (dòng 145–151)

Đọc `.ply`, chuẩn hoá màu RGB về $[0,1]$ bằng $c = \dfrac{c_{byte}}{255}$. Không công
thức khác.

## 4. `storePly(path, xyz, rgb)` (dòng 153–168)

Ghi file `.ply` từ mảng toạ độ + màu (pháp tuyến gán 0). Thuần I/O.

## 5. `readColmapSceneInfo(path, images, eval, llffhold=8)` (dòng 170–227)

Điều phối: tìm thư mục `sparse/0` hoặc `sparse`, đọc extrinsics/intrinsics (binary ưu
tiên, fallback text), gọi `readColmapCameras`, sắp xếp theo tên ảnh.

Chia tập train/test theo **LLFF holdout** — cứ mỗi `llffhold` ảnh thì 1 ảnh làm test:

$$
\text{test} = \{c_i : i \bmod \text{llffhold} = 0\}, \qquad
\text{train} = \{c_i : i \bmod \text{llffhold} \ne 0\}
$$

Gọi `getNerfppNorm` để chuẩn hoá, và đọc/convert point cloud ban đầu (`points3D.bin/txt`
→ `.ply`).

## 6. `readCamerasFromTransforms(path, transformsfile, white_background, extension=".png")` (dòng 229–269)

Đọc file `transforms_*.json` (định dạng NeRF-synthetic/Blender).

Chuyển hệ trục camera Blender/OpenGL (Y lên, Z ra sau) sang hệ COLMAP (Y xuống, Z
tiến về trước) bằng cách đảo dấu 2 cột:

$$
c2w_{:3,\,1:3} \leftarrow -\,c2w_{:3,\,1:3}
$$

Tính world-to-camera bằng nghịch đảo ma trận camera-to-world:

$$
w2c = (c2w)^{-1}, \qquad R = (w2c_{:3,:3})^{T}, \qquad T = w2c_{:3,3}
$$

Trộn nền (alpha compositing) cho ảnh RGBA, với nền trắng hoặc đen:

$$
I_{rgb} = I_{rgb}\cdot \alpha + \text{bg}\cdot(1-\alpha), \qquad \alpha = \dfrac{A}{255}
$$

FOV theo trục dọc được suy từ FOV ngang qua tiêu cự pixel (ảnh không nhất thiết vuông):

$$
f = \text{fov2focal}(\text{fovx}, W), \qquad \text{fovy} = \text{focal2fov}(f, H)
$$

## 7. `readNerfSyntheticInfo(path, white_background, eval, extension=".png")` (dòng 271–305)

Điều phối đọc Blender dataset qua #6 cho cả train/test. Nếu không có point cloud khởi
tạo, sinh **point cloud ngẫu nhiên đều** trong hộp $[-1.3,1.3]^3$:

$$
\mathbf{x} \sim \mathcal U(-1.3,\,1.3)^3,\qquad 100{,}000\ \text{điểm}
$$

và màu khởi tạo ngẫu nhiên qua $SH \to RGB$ (hàm `SH2RGB`, bản chất là cộng hằng số
chuẩn hoá SH bậc 0: $c = s\cdot C_0 + 0.5$).

## 8–9. Nested: `get_center_and_diag` — đã nêu trong mục 1.

## 10. `sceneLoadTypeCallbacks` (dòng 307–310)

Không phải hàm — là dict điều phối (`"Colmap"` → #5, `"Blender"` → #7).

---

# C. `scene/cameras.py` — 1 lớp `Camera` + 1 lớp `MiniCam` (2 hàm `__init__`)

## 1. `Camera.__init__(...)` (dòng 17–62)

Dựng object camera đầy đủ dùng khi train (có ảnh ground-truth).

- Nhân ảnh với alpha mask (hoặc mask toàn 1 nếu không có):

$$
I = I_{gt}\odot M,\qquad M\in\{0,1\}^{H\times W}\ (\text{hoặc toàn } 1)
$$

- Ma trận **World-to-View** (xem công thức đầy đủ ở mục D.3 `getWorld2View2`):

$$
V = \text{getWorld2View2}(R,T,\text{trans},\text{scale})^{T}
$$

- Ma trận **chiếu phối cảnh (projection)** (xem D.4 `getProjectionMatrix`):

$$
P = \text{getProjectionMatrix}(z_{near}, z_{far}, \text{FoVx}, \text{FoVy})^{T}
$$

- Ma trận chiếu đầy đủ (world → clip space):

$$
M_{full} = V \cdot P
$$

- Tâm camera trong world space = cột tịnh tiến của ma trận camera-to-view nghịch đảo:

$$
C = \big(V^{-1}\big)_{[3,\,0:3]}
$$

- Tiêu cự pixel suy từ FOV (ngược với `focal2fov`):

$$
f_x = \dfrac{W}{2\tan(\text{FoVx}/2)}, \qquad
f_y = \dfrac{H}{2\tan(\text{FoVy}/2)}
$$

## 2. `MiniCam.__init__(...)` (dòng 64–76)

Phiên bản camera rút gọn (dùng khi render tự do, không có ảnh GT), nhận thẳng
`world_view_transform` và `full_proj_transform` đã tính sẵn. Chỉ tính:

$$
C = \big(V^{-1}\big)_{[3,\,0:3]}
$$

(giống công thức tâm camera ở trên).

---

# D. Công thức hình học dùng chung — `utils/graphics_utils.py`

(Không nằm trong 4 file được yêu cầu, nhưng là nơi định nghĩa các hàm `cameras.py` và
`dataset_readers.py` gọi tới — liệt kê ở đây để công thức đầy đủ, dễ tra cứu.)

### D.1 `geom_transform_points(points, transf_matrix)`

Biến đổi phối cảnh (homogeneous) rồi chia phối cảnh:

$$
\mathbf{p}_{hom} = [\mathbf{p}, 1],\qquad
\mathbf{p}' = \mathbf{p}_{hom}\,M,\qquad
\mathbf{p}_{out} = \dfrac{\mathbf{p}'_{0:3}}{\mathbf{p}'_{3}+\epsilon}
$$

### D.2 `getWorld2View(R, t)`

$$
Rt=\begin{bmatrix} R^{T} & t\\ 0 & 1\end{bmatrix}
$$

### D.3 `getWorld2View2(R, t, translate=0, scale=1)`

Giống D.2 nhưng dịch/co tâm camera trong world space trước khi quay lại view space
(dùng để chuẩn hoá scene về gốc toạ độ, bán kính đơn vị):

$$
Rt = \begin{bmatrix}R^T & t\\0&1\end{bmatrix},\quad
C2W = Rt^{-1},\quad
C' = (C2W_{[0:3,3]} + \text{translate})\cdot \text{scale}
$$
$$
C2W_{[0:3,3]} \leftarrow C', \qquad Rt_{new} = C2W^{-1}
$$

### D.4 `getProjectionMatrix(znear, zfar, fovX, fovY)`

Ma trận chiếu phối cảnh kiểu OpenGL (frustum đối xứng), với:

$$
\tan_x=\tan\!\left(\dfrac{fovX}{2}\right),\quad \tan_y=\tan\!\left(\dfrac{fovY}{2}\right)
$$
$$
\text{right}=\tan_x\cdot z_{near},\ \text{left}=-\text{right},\quad
\text{top}=\tan_y\cdot z_{near},\ \text{bottom}=-\text{top}
$$

$$
P=\begin{bmatrix}
\dfrac{2z_{near}}{r-l} & 0 & \dfrac{r+l}{r-l} & 0\\[4pt]
0 & \dfrac{2z_{near}}{t-b} & \dfrac{t+b}{t-b} & 0\\[4pt]
0 & 0 & \dfrac{z_{far}}{z_{far}-z_{near}} & -\dfrac{z_{far}z_{near}}{z_{far}-z_{near}}\\[4pt]
0 & 0 & 1 & 0
\end{bmatrix}
$$

### D.5 `fov2focal(fov, pixels)`

$$
f = \dfrac{\text{pixels}}{2\tan(\text{fov}/2)}
$$

### D.6 `focal2fov(focal, pixels)`

$$
\text{fov} = 2\arctan\!\left(\dfrac{\text{pixels}}{2f}\right)
$$

---

# E. `scene/__init__.py` — class `Scene` (4 hàm)

Không có công thức toán — đây là lớp điều phối vòng đời dữ liệu, chỉ gọi lại các hàm
đã mô tả ở trên.

## 1. `Scene.__init__(self, args, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0])` (dòng 25–83)

- Phát hiện loại dataset: có thư mục `sparse/` → gọi `readColmapSceneInfo` (mục B.5);
  có `transforms_train.json` → gọi `readNerfSyntheticInfo` (mục B.7).
- Nếu train từ đầu: copy point cloud gốc vào `input.ply`, ghi `cameras.json`
  (dump pose mọi camera qua `camera_to_JSON`).
- Xáo trộn ngẫu nhiên thứ tự camera train/test (`random.shuffle`) nếu `shuffle=True`.
- `cameras_extent` lấy trực tiếp từ `nerf_normalization["radius"]` (công thức ở mục B.1).
- Dựng danh sách `Camera` thật (ảnh đã load, resize theo `resolution_scale`) qua
  `cameraList_from_camInfos` (nằm ở `utils/camera_utils.py`, ngoài phạm vi 4 file này).
- Nếu có checkpoint (`load_iteration`): gọi `gaussians.load_ply(...)`.
  Nếu không và point cloud rỗng: gọi `gaussians.create_from_pcd(pcd, cameras_extent)`
  (xem `DOCS/BOOK/gaussian_model.md` mục 18).

## 2. `Scene.save(self, iteration)` (dòng 85–87)

Gọi `gaussians.save_ply(...)`. Không công thức.

## 3. `Scene.getTrainCameras(self, scale=1.0)` (dòng 89–90)

Trả về dict đã lưu sẵn theo scale. Không công thức.

## 4. `Scene.getTestCameras(self, scale=1.0)` (dòng 92–93)

Tương tự #3. Không công thức.

---

## Tổng kết số lượng hàm

| File | Số hàm/method |
|---|---|
| `scene/colmap_loader.py` | 11 (+ 1 class `Image` kế thừa `BaseImage`) |
| `scene/dataset_readers.py` | 8 hàm top-level + 2 hàm lồng (`get_center_and_diag`) |
| `scene/cameras.py` | 2 (`Camera.__init__`, `MiniCam.__init__`) |
| `scene/__init__.py` | 4 (`__init__`, `save`, `getTrainCameras`, `getTestCameras`) |

Phần lớn các hàm trong `colmap_loader.py` và `dataset_readers.py` là **I/O thuần tuý**
(đọc/ghi file nhị phân hoặc text, không có công thức toán); công thức toán tập trung ở
`qvec2rotmat`/`rotmat2qvec` (quaternion ↔ rotation matrix), `getNerfppNorm` (chuẩn hoá
scene), và các hàm hình học camera dùng chung trong `utils/graphics_utils.py` mà
`cameras.py` gọi tới (world-to-view, projection matrix, focal↔FOV).
