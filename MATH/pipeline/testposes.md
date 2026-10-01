# Công thức toán học trong `pipeline/testposes.py`

File này **đọc `test_poses.csv`** (pose camera do ban tổ chức cấp sẵn) thành đối tượng `Camera` để render — **không có nội suy quỹ đạo** (không slerp, không spiral/orbit path): mỗi dòng CSV là một camera độc lập, dựng trực tiếp từ quaternion + translation + tiêu cự có sẵn, không nội suy giữa các khung hình.

Các công thức hình học xuất hiện trong file đều là **chuyển đổi tham số camera** (quaternion → ma trận xoay, tiêu cự → FoV), không phải phép toán tối ưu hay loss.

---

## 1. Quy ước pose (`pipeline/testposes.py`, dòng 8–9, docstring đầu file)

```python
Cột CSV: image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
Quy ước giống COLMAP: quaternion + translation là world-to-camera.
```

Quaternion + translation trong CSV là **world-to-camera**, cùng quy ước COLMAP:

$$
q = (q_w, q_x, q_y, q_z), \qquad T = (t_x, t_y, t_z)
$$

## 2. Quaternion → ma trận xoay (`load_test_cameras`, dòng 89–91)

```python
qvec = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
R = np.transpose(qvec2rotmat(qvec))            # giống readColmapCameras
T = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
```

Dùng hàm `qvec2rotmat` (định nghĩa ở `scene/colmap_loader.py`, dòng 43–53, trích nguyên văn):

```python
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
```

tức là ma trận xoay chuẩn từ quaternion đơn vị $q=(q_w,q_x,q_y,q_z)$:

$$
\mathrm{qvec2rotmat}(q) =
\begin{pmatrix}
1-2q_y^2-2q_z^2 & 2q_xq_y-2q_wq_z & 2q_zq_x+2q_wq_y\\
2q_xq_y+2q_wq_z & 1-2q_x^2-2q_z^2 & 2q_yq_z-2q_wq_x\\
2q_zq_x-2q_wq_y & 2q_yq_z+2q_wq_x & 1-2q_x^2-2q_y^2
\end{pmatrix}
$$

sau đó `testposes.py` **chuyển vị** kết quả này để khớp quy ước dựng `Camera` của 3DGS (giống `readColmapCameras`):

$$
R = \big(\mathrm{qvec2rotmat}(q)\big)^{\top}
$$

## 3. Tiêu cự → trường nhìn (FoV) (`load_test_cameras`, dòng 92–93)

```python
fov_x = focal2fov(float(row["fx"]), width)
fov_y = focal2fov(float(row["fy"]), height)
```

Dùng hàm `focal2fov` (định nghĩa ở `utils/graphics_utils.py`, dòng 76–77, trích nguyên văn):

```python
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
```

$$
\mathrm{FoV}_x = \mathrm{focal2fov}(f_x, W) = 2\arctan\!\left(\frac{W}{2f_x}\right), \qquad
\mathrm{FoV}_y = \mathrm{focal2fov}(f_y, H) = 2\arctan\!\left(\frac{H}{2f_y}\right)
$$

(Ghi chú: công thức $\mathrm{FoV}=2\arctan(\text{pixels}/(2f))$ ở trên đã được xác nhận trực tiếp từ nguồn thật `utils/graphics_utils.py` dòng 77, không còn là suy luận từ tên hàm như ghi chú ở bản cũ.)

## 4. Độ lệch principal point (`load_test_cameras`, dòng 95–97, 113–117)

```python
# 3DGS chỉ dựng ma trận chiếu từ FoV, không có chỗ cho principal point lệch
offsets.append((abs(float(row["cx"]) - width / 2.0),
                abs(float(row["cy"]) - height / 2.0)))
```

```python
max_dx = max(o[0] for o in offsets)
max_dy = max(o[1] for o in offsets)
if max_dx > 1.0 or max_dy > 1.0:
    print(f"  ⚠ principal point lệch tâm tối đa ({max_dx:.1f}, {max_dy:.1f}) px — "
          "3DGS dựng ma trận chiếu chỉ từ FoV nên ảnh render sẽ bị dịch đúng lượng đó")
```

3DGS chỉ dựng ma trận chiếu từ FoV (giả định principal point ở đúng tâm ảnh $(W/2, H/2)$), nên file tính độ lệch so với tâm để cảnh báo sai số render:

$$
\Delta c_{x} = \left|c_x - \frac{W}{2}\right|, \qquad \Delta c_{y} = \left|c_y - \frac{H}{2}\right|
$$

$$
\Delta c_x^{max} = \max_i \Delta c_{x,i}, \qquad \Delta c_y^{max} = \max_i \Delta c_{y,i}
$$

Cảnh báo được in nếu $\Delta c_x^{max} > 1$ hoặc $\Delta c_y^{max} > 1$ (pixel).

---

## 5. Luồng tổng thể (không phải công thức, nhưng cần để hiểu ngữ cảnh)

Toàn bộ vòng lặp dựng `Camera` (`load_test_cameras`, dòng 86–111):

```python
cams, missing_gt, offsets = [], [], []
for index, row in enumerate(rows):
    width, height = int(float(row["width"])), int(float(row["height"]))
    qvec = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
    R = np.transpose(qvec2rotmat(qvec))            # giống readColmapCameras
    T = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
    fov_x = focal2fov(float(row["fx"]), width)
    fov_y = focal2fov(float(row["fy"]), height)

    # 3DGS chỉ dựng ma trận chiếu từ FoV, không có chỗ cho principal point lệch
    offsets.append((abs(float(row["cx"]) - width / 2.0),
                    abs(float(row["cy"]) - height / 2.0)))

    name = row["image_name"]
    gt_path = _find_image(images_dir, name)
    if gt_path is None:
        missing_gt.append(name)
        image = torch.zeros((3, height, width))
    else:
        with Image.open(gt_path) as handle:
            image = PILtoTorch(handle.convert("RGB"), (width, height))

    cams.append(Camera(colmap_id=index, R=R, T=T, FoVx=fov_x, FoVy=fov_y,
                       image=image, gt_alpha_mask=None,
                       image_name=os.path.splitext(name)[0], uid=index,
                       data_device=data_device))
```

Mỗi dòng CSV độc lập sinh một đối tượng `Camera(R, T, FoVx, FoVy, ...)`; nếu không tìm thấy ảnh GT tương ứng (`_find_image` trả `None`), ảnh được thay bằng tensor 0 (`torch.zeros((3, height, width))`) và tên ảnh được ghi vào `missing_gt` — vẫn render được nhưng không chấm điểm được (không có GT để tính SSIM/PSNR/LPIPS, xem `MATH/metrics.md`).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Dòng | Công thức LaTeX |
|---|---|---|
| `qvec = np.array([qw, qx, qy, qz])` | 89 | $q=(q_w,q_x,q_y,q_z)$ |
| `R = np.transpose(qvec2rotmat(qvec))` | 90 | $R=\big(\mathrm{qvec2rotmat}(q)\big)^{\top}$ |
| `T = np.array([tx, ty, tz])` | 91 | $T=(t_x,t_y,t_z)$ |
| `fov_x = focal2fov(fx, width)` | 92 | $\mathrm{FoV}_x=\mathrm{focal2fov}(f_x,W)=2\arctan(W/2f_x)$ |
| `fov_y = focal2fov(fy, height)` | 93 | $\mathrm{FoV}_y=\mathrm{focal2fov}(f_y,H)=2\arctan(H/2f_y)$ |
| `abs(cx - width/2.0)` | 96 | $\Delta c_x = \lvert c_x - W/2\rvert$ |
| `abs(cy - height/2.0)` | 97 | $\Delta c_y = \lvert c_y - H/2\rvert$ |
| `max(o[0] for o in offsets)` | 113 | $\Delta c_x^{max}=\max_i \Delta c_{x,i}$ |
| `max(o[1] for o in offsets)` | 114 | $\Delta c_y^{max}=\max_i \Delta c_{y,i}$ |
| `if max_dx > 1.0 or max_dy > 1.0` | 115 | cảnh báo nếu $\Delta c_x^{max}>1 \lor \Delta c_y^{max}>1$ |

---

## Kiểm chứng tính đúng sai

Đã đọc lại thật `pipeline/testposes.py` (128 dòng), `scene/colmap_loader.py` (hàm `qvec2rotmat`, dòng 43–53) và `utils/graphics_utils.py` (hàm `focal2fov`, dòng 76–77). Kết quả:

- Công thức $R=(\mathrm{qvec2rotmat}(q))^\top$ khớp chính xác với code thật, số dòng đã cập nhật đúng (90, không phải suy diễn).
- Công thức FoV $=2\arctan(\text{pixels}/(2f))$ trước đây ghi là "suy luận từ tên hàm, không đọc trực tiếp" — nay đã đọc trực tiếp `utils/graphics_utils.py:77` và xác nhận khớp đúng 100%, không còn là suy luận.
- Khai triển tường minh ma trận $\mathrm{qvec2rotmat}(q)$ (trước đây chỉ viết dạng hàm trừu tượng) để người đọc thấy rõ công thức quaternion→rotation chuẩn (dòng 44–53 của `colmap_loader.py`).
- Không phát hiện sai sót toán học nào trong nội dung cũ.
