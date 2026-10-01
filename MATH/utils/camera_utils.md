# Công thức toán học trong `utils/camera_utils.py`

Tài liệu mô tả cơ sở toán học của việc dựng đối tượng `Camera` từ dữ liệu COLMAP (`cam_info`) và chuyển đổi sang định dạng JSON, bám sát thứ tự xuất hiện trong code.

---

## 1. Tính độ phân giải đích (`loadCam`, `utils/camera_utils.py` dòng 19–39)

Ảnh gốc có kích thước $(W_0, H_0)$ = `orig_w, orig_h`. Tuỳ theo `args.resolution`:

Trích nguyên văn (dòng 19–39):

```python
def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
```

### 1.1. Trường hợp hệ số nguyên $\in\{1,2,4,8\}$ (dòng 22–23)

$$
(W,H) = \left(\mathrm{round}\!\left(\frac{W_0}{r_{scale}\cdot R}\right),\ \mathrm{round}\!\left(\frac{H_0}{r_{scale}\cdot R}\right)\right)
$$

trong đó $R=$ `args.resolution`, $r_{scale}=$ `resolution_scale`.

### 1.2. Trường hợp $R=-1$ (tự động) (dòng 25–34)

Nếu $W_0 > 1600$:

$$
d_{global} = \frac{W_0}{1600}
$$

ngược lại $d_{global}=1$.

### 1.3. Trường hợp $R$ là số khác (ép kiểu float, chỉ định độ rộng đích) (dòng 35–36)

$$
d_{global} = \frac{W_0}{R}
$$

### 1.4. Áp dụng tỉ lệ tổng hợp (cho cả hai nhánh 1.2, 1.3) (dòng 38–39)

$$
s = d_{global}\cdot r_{scale}
$$

$$
(W,H) = \left(\left\lfloor \frac{W_0}{s}\right\rfloor,\ \left\lfloor \frac{H_0}{s}\right\rfloor\right)
$$

(phép ép kiểu `int()` trong Python làm tròn về 0, tương đương sàn với số dương)

---

## 2. Resize ảnh và tách kênh alpha (`loadCam` dòng 41–47, `PILtoTorch` trong `utils/general_utils.py` dòng 24–30)

Trích nguyên văn `PILtoTorch` (`utils/general_utils.py`, dòng 24–30):

```python
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
```

Ảnh PIL gốc được resize về $(W,H)$ (dùng phép nội suy mặc định của PIL — `Image.resize`, không chỉ định rõ bộ lọc):

$$
I_{resized} = \mathrm{resize}(I_{orig}, (W,H)), \qquad I_{tensor} = \frac{I_{resized}}{255}
$$

Ảnh được hoán vị trục kênh ra trước (dòng 27–30): nếu $I_{tensor}\in\mathbb{R}^{H\times W\times C}$ thì chuyển thành $\mathbb{R}^{C\times H\times W}$ (hoặc thêm chiều kênh rồi hoán vị nếu ảnh xám, chỉ có 2 chiều $H\times W$).

Trích nguyên văn phần tách kênh trong `loadCam` (dòng 41–47):

```python
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
```

Tách kênh:

$$
I_{gt} = I_{tensor}[0{:}3,\,\cdot,\,\cdot] \quad(\text{RGB})
$$

Nếu $C=4$ (có alpha):

$$
M_{mask} = I_{tensor}[3{:}4,\,\cdot,\,\cdot]
$$

**Lưu ý nhỏ phát hiện khi đọc lại code:** điều kiện kiểm tra alpha trong code là `resized_image_rgb.shape[1] == 4`, tức chiều thứ **1** (chỉ số 0-based) của tensor đã hoán vị $C\times H\times W$ — đây chính là chiều $H$ chứ không phải $C$ nếu đọc theo nghĩa đen "shape[1]". Tuy nhiên đọc kỹ `PILtoTorch`: với ảnh màu PIL $(H,W,C)$, sau `permute(2,0,1)` tensor có shape $(C,H,W)$, nên `shape[1]` thực chất là $H$, không phải $C$! Điều kiện `resized_image_rgb.shape[1] == 4` do đó về bản chất đang hỏi "$H=4$ hay không" — có vẻ như một lỗi chỉ-số trong code gốc 3DGS (Inria) kế thừa nguyên văn vào SADGS, lẽ ra phải là `shape[0] == 4` (kiểm tra số kênh $C=4$) để phát hiện ảnh RGBA. Đây là hành vi **y hệt bản gốc 3DGS chính thức** (không phải lỗi riêng của SADGS) nên tài liệu này ghi nhận lại nguyên trạng, không tự ý "sửa" công thức — trên thực tế logic này gần như luôn đúng vì ảnh RGBA được dùng trong 3DGS thường chỉ xảy ra khi pipeline tải mask cùng ảnh gốc, và trong đa số trường hợp huấn luyện mask không tồn tại nên nhánh này đơn giản không kích hoạt.

---

## 3. Chuyển đổi trường nhìn (FOV) $\leftrightarrow$ tiêu cự (`fov2focal`, `focal2fov` trong `graphics_utils.py`, dùng ở `camera_to_JSON`)

Với mô hình camera lỗ kim (pinhole), quan hệ giữa trường nhìn (radian) và tiêu cự tính theo pixel với độ phân giải $P$ (chiều rộng hoặc chiều cao ảnh).

Trích nguyên văn (`utils/graphics_utils.py`, dòng 73–77):

```python
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
```

$$
f = \frac{P}{2\tan(\mathrm{FoV}/2)} \qquad \text{(`fov2focal`, dòng 73–74)}
$$

$$
\mathrm{FoV} = 2\arctan\!\left(\frac{P}{2f}\right) \qquad \text{(`focal2fov`, dòng 76–77)}
$$

Áp dụng cho từng trục:

$$
f_x = \mathrm{fov2focal}(\mathrm{FoV}_x, W), \qquad f_y = \mathrm{fov2focal}(\mathrm{FoV}_y, H)
$$

---

## 4. Dựng ma trận biến đổi thế giới $\to$ camera và xuất JSON (`camera_to_JSON`, `utils/camera_utils.py` dòng 62–82)

Trích nguyên văn (dòng 62–82):

```python
def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
```

Từ ma trận xoay $R\in\mathbb{R}^{3\times3}$ và vector tịnh tiến $T\in\mathbb{R}^3$ (quy ước COLMAP/3DGS: $R,T$ biến đổi camera $\to$ thế giới dạng transpose), dựng ma trận đồng nhất $4\times4$ (dòng 63–66):

$$
Rt = \begin{bmatrix} R^\top & T \\ 0 & 1 \end{bmatrix}
$$

(code gán `Rt[:3,:3] = R.transpose()`, `Rt[:3,3] = T`, `Rt[3,3]=1`)

Nghịch đảo để ra ma trận World-to-Camera (W2C), rồi trích vị trí và hướng camera trong hệ thế giới (dòng 68–70):

$$
W2C = Rt^{-1}
$$

$$
\mathrm{pos} = W2C[0{:}3,\,3], \qquad \mathrm{rot} = W2C[0{:}3,\,0{:}3]
$$

Tiêu cự xuất ra JSON dùng công thức mục 3 (dòng 79–80):

$$
f_x^{json} = \mathrm{fov2focal}(\mathrm{FoV}_x, W_{cam}), \qquad f_y^{json} = \mathrm{fov2focal}(\mathrm{FoV}_y, H_{cam})
$$

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `orig_w, orig_h = cam_info.image.size` | $(W_0, H_0)$ |
| `round(orig_w/(resolution_scale * args.resolution))` | $\mathrm{round}\!\left(\dfrac{W_0}{r_{scale}\cdot R}\right)$ |
| `global_down = orig_w / 1600` | $d_{global} = \dfrac{W_0}{1600}$ (khi $W_0>1600$) |
| `global_down = orig_w / args.resolution` | $d_{global} = \dfrac{W_0}{R}$ |
| `scale = float(global_down) * float(resolution_scale)` | $s = d_{global}\cdot r_{scale}$ |
| `resolution = (int(orig_w / scale), int(orig_h / scale))` | $(W,H) = \left(\left\lfloor W_0/s\right\rfloor, \left\lfloor H_0/s\right\rfloor\right)$ |
| `PILtoTorch(cam_info.image, resolution)` | $I_{tensor} = \mathrm{resize}(I_{orig},(W,H))/255$, hoán vị trục kênh |
| `resized_image_rgb[:3, ...]` | $I_{gt} = I_{tensor}[0{:}3]$ |
| `resized_image_rgb[3:4, ...]` | $M_{mask} = I_{tensor}[3{:}4]$ |
| `fov2focal(fov, pixels)` → `pixels / (2*tan(fov/2))` | $f = \dfrac{P}{2\tan(\mathrm{FoV}/2)}$ |
| `focal2fov(focal, pixels)` → `2*atan(pixels/(2*focal))` | $\mathrm{FoV} = 2\arctan\!\left(\dfrac{P}{2f}\right)$ |
| `Rt[:3, :3] = camera.R.transpose()` | $Rt[0{:}3,0{:}3] = R^\top$ |
| `Rt[:3, 3] = camera.T` | $Rt[0{:}3,3] = T$ |
| `W2C = np.linalg.inv(Rt)` | $W2C = Rt^{-1}$ |
| `pos = W2C[:3, 3]` | $\mathrm{pos} = W2C[0{:}3,3]$ |
| `rot = W2C[:3, :3]` | $\mathrm{rot} = W2C[0{:}3,0{:}3]$ |
| `fy = fov2focal(camera.FovY, camera.height)` | $f_y = \mathrm{fov2focal}(\mathrm{FoV}_y, H)$ |
| `fx = fov2focal(camera.FovX, camera.width)` | $f_x = \mathrm{fov2focal}(\mathrm{FoV}_x, W)$ |
