# Công thức toán học trong `render.py`

## Nhận định chung

`render.py` là một script điều phối: nó khởi tạo `GaussianModel`, nạp `Scene` đã huấn luyện, rồi gọi hàm render thực sự `render_structgs` (định nghĩa trong `gaussian_renderer/`, **không nằm trong file này**) cho từng camera, lưu ảnh ra đĩa và đo thời gian/FPS. Bản thân `render.py` **không định nghĩa công thức dựng ảnh (rasterization) hay alpha-blending** — những phép toán đó nằm bên trong `render_structgs` (CUDA rasterizer). File này chỉ chứa một vài phép tính hậu xử lý số học đơn giản để trực quan hoá (`depth_map`, `normal_map`, `error_map`) và đo hiệu năng (FPS).

Không có lời gọi PSNR/SSIM trong file này (các phép đo chất lượng ảnh được thực hiện riêng trong `metrics.py`).

---

## 1. Chuẩn hoá bản đồ độ sâu (depth) để hiển thị

Trích nguyên văn (`render.py`, dòng 57–61):

```python
depth = render_pkg["depth_map"]
if len(depth.shape) == 2:
    depth = depth.unsqueeze(0)
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
depth = depth.repeat(3, 1, 1) if depth.shape[0] == 1 else depth
```

$$
\widehat{D} = \frac{D - \min(D)}{\max(D) - \min(D) + \varepsilon}, \qquad \varepsilon = 10^{-5}
$$

Đây là chuẩn hoá min-max thuần tuý cho mục đích trực quan hoá, không phải độ sâu thực.

## 2. Chuẩn hoá bản đồ pháp tuyến (normal) để hiển thị

Trích nguyên văn (`render.py`, dòng 63–67):

```python
normal = render_pkg["normal_map"]
if len(normal.shape) == 2:
    normal = normal.unsqueeze(0)
normal = (normal + 1.0) / 2.0 # Assuming normals are in [-1, 1]
normal = normal.repeat(3, 1, 1) if normal.shape[0] == 1 else normal
```

Giả định các vector pháp tuyến $N \in [-1,1]^3$ (từ kết xuất), ánh xạ tuyến tính sang $[0,1]$ để lưu ảnh:

$$
\widehat{N} = \frac{N + 1}{2}
$$

## 3. Bản đồ sai số (error map)

Trích nguyên văn (`render.py`, dòng 69–70):

```python
error_map = torch.abs(rendering - gt)
# error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-5)
```

$$
E = \lvert I_{render} - I_{gt} \rvert
$$

(trị tuyệt đối theo từng pixel, theo đúng dòng `error_map = torch.abs(rendering - gt)`; dòng chuẩn hoá min-max cho `error_map` đã bị **comment out** trong code nên không được áp dụng).

## 4. Đo hiệu năng dựng ảnh (FPS)

Tích luỹ thời gian (`render.py`, dòng 40–44):

```python
start_time = time.time()
render_pkg = render_structgs(view, gaussians, pipeline, background, args.mult, compute_extra=args.render_extra)
rendering = render_pkg["render"]
end_time = time.time()
total_time += (end_time - start_time)
```

Tổng hợp cuối vòng lặp (`render.py`, dòng 81–84):

```python
num_frames = len(views)
avg_time = total_time / num_frames if num_frames > 0 else 0
fps = 1.0 / avg_time if avg_time > 0 else 0
print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")
```

$$
\overline{t} = \frac{T_{total}}{N_{frames}}, \qquad \mathrm{FPS} = \frac{1}{\overline{t}}
$$

với $T_{total} = \sum_{\text{frame}} (t_{end} - t_{start})$ đo bằng `time.time()` quanh lệnh gọi `render_structgs`, chỉ được tính khi $N_{frames} > 0$ và $\overline{t} > 0$ (nếu không thì FPS $=0$ theo code).

## 5. Màu ngẫu nhiên cho chế độ "extras"

Trích nguyên văn (`render.py`, dòng 50–53):

```python
# 1. Random color rendering
num_gaussians = gaussians.get_xyz.shape[0]
random_colors = torch.rand((num_gaussians, 3), device="cuda")
render_pkg_random = render_structgs(view, gaussians, pipeline, background, args.mult, override_color=random_colors)
```

Khi `args.render_extra`, mỗi Gaussian được gán một màu RGB ngẫu nhiên độc lập đồng nhất:

$$
c_i \sim \mathcal{U}([0,1]^3)
$$

rồi dựng lại ảnh với màu override này (không liên quan đến hệ số SH gốc) — chỉ phục vụ trực quan hoá phân bố điểm.

---

## Bảng hằng số/ngưỡng

| Ký hiệu | Giá trị | Vị trí |
|---|---|---|
| $\varepsilon$ (chuẩn hoá depth) | $10^{-5}$ | `render_set`, chuẩn hoá `depth_map` |
| background color | $(1,1,1)$ nếu `white_background` else $(0,0,0)$ | `render_sets` |
| `args.mult` | mặc định $0.7$ (giá trị CLI mặc định trong `render.py` là `0.5`, ghi đè giá trị mặc định $0.7$ của `OptimizationParams`) | truyền vào `render_structgs` làm hệ số nhân box tile |

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức |
|---|---|
| `depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)` | $\widehat{D} = \dfrac{D-\min D}{\max D-\min D+10^{-5}}$ |
| `normal = (normal + 1.0) / 2.0` | $\widehat{N} = \dfrac{N+1}{2}$ |
| `error_map = torch.abs(rendering - gt)` | $E = \lvert I_{render}-I_{gt}\rvert$ |
| `random_colors = torch.rand((num_gaussians, 3), device="cuda")` | $c_i \sim \mathcal{U}([0,1]^3)$ |
| `avg_time = total_time / num_frames if num_frames > 0 else 0` | $\overline{t} = T_{total}/N_{frames}$ |
| `fps = 1.0 / avg_time if avg_time > 0 else 0` | $\mathrm{FPS} = 1/\overline{t}$ |
| `total_time += (end_time - start_time)` | $T_{total} \mathrel{+}= t_{end}-t_{start}$ (đo bằng `time.time()`) |
