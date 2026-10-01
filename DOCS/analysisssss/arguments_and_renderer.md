# `arguments/__init__.py` và `gaussian_renderer/` — Giải thích chi tiết

---

# A. `arguments/__init__.py` — 6 hàm/class

File này **không chứa công thức toán** — chỉ định nghĩa các nhóm tham số dòng lệnh
(argparse) và cách parse/merge config. Liệt kê để đủ số lượng:

## 1. `GroupParams` (dòng 16–17)

Class rỗng, chỉ dùng làm "túi" chứa thuộc tính (namespace) khi `extract()` gọi `setattr`.

## 2. `ParamGroup.__init__(self, parser, name, fill_none=False)` (dòng 20–38)

Tự động quét các thuộc tính `self.xxx` đã gán trong `__init__` của lớp con
(`ModelParams`, `PipelineParams`, `OptimizationParams`), rồi đăng ký chúng thành
argparse argument (`--xxx`), suy luận kiểu dữ liệu (`bool`→cờ `store_true`, còn lại
dùng `type=t`). Tên bắt đầu bằng `_` được coi là có dạng rút gọn `-x` (shorthand).

## 3. `ParamGroup.extract(self, args)` (dòng 40–45)

Lấy lại giá trị đã parse từ `argparse.Namespace`, gán vào `GroupParams`. Không công thức.

## 4. `ModelParams` (dòng 47–62)

Khai báo tham số dữ liệu (đường dẫn, `sh_degree=3`, kích thước ảnh, v.v.). `extract()`
override để chuẩn hoá `source_path` thành đường dẫn tuyệt đối.

## 5. `PipelineParams` (dòng 64–71)

Khai báo cờ pipeline render (`convert_SHs_python`, `compute_cov3D_python`, `debug`,
`antialiasing`...).

## 6. `OptimizationParams` (dòng 73–160)

Khai báo toàn bộ siêu tham số huấn luyện: learning rate, ngưỡng densify/prune, trọng
số loss (`lambda_dssim`, `lambda_l2`, `lambda_freq`...), tham số StructGS (`tau_expand`,
`ks_scale_power`, `clone_target_eta`...). Các giá trị này được **dùng** trong công thức
ở `gaussian_model.py`, `loss_utils.py`, `freq_utils.py` (xem các tài liệu tương ứng)
nhưng bản thân file này chỉ khai báo hằng số mặc định, không tính toán.

## `get_combined_args(parser)` (dòng 162–182)

Merge tham số dòng lệnh với file config đã lưu (`cfg_args`), ưu tiên dòng lệnh nếu
khác `None`. Thuần logic, không công thức.

---

# B. `gaussian_renderer/` — 3 file

## B.1 `gaussian_renderer/__init__.py` — hàm `render_structgs(...)` (dòng 18–127)

Đây là **hàm cầu nối** giữa `GaussianModel` và rasterizer CUDA
(`diff_gaussian_rasterization_structgs`) — chuẩn bị toàn bộ tham số hình học rồi gọi
rasterizer để "vẽ" (splat) các Gaussian 3D lên ảnh 2D.

### Công thức / phép biến đổi chính trong hàm

- **Tiêu cự chuẩn hoá theo FOV** (tangent half-FOV, dùng nội bộ bởi rasterizer để tính
  Jacobian chiếu phối cảnh):

$$
\tan\!\left(\dfrac{\text{FoVx}}{2}\right), \qquad \tan\!\left(\dfrac{\text{FoVy}}{2}\right)
$$

- **Opacity đưa vào rasterizer** đã được bù bộ lọc 3D (công thức xem
  `DOCS/BOOK/gaussian_model.md` mục 14):

$$
\alpha_{in} = \texttt{get\_opacity\_with\_3D\_filter}
$$

- **Scale đưa vào rasterizer** cũng đã cộng bộ lọc 3D (mục 7 cùng tài liệu):

$$
S_{in} = \sqrt{S^2 + F_{3D}^2}
$$

- **(Tuỳ chọn) Chuyển SH → RGB bằng Python** thay vì để CUDA làm, dùng hướng nhìn chuẩn
  hoá từ camera đến mỗi Gaussian:

$$
\mathbf d = \dfrac{\mathbf x_{gaussian} - \mathbf c_{camera}}{\lVert \mathbf x_{gaussian} - \mathbf c_{camera}\rVert}
$$

  rồi gọi `eval_sh(deg, sh, d)` (công thức đầy đủ ở mục C.sh_utils bên dưới), và clamp
  màu dưới về 0 sau khi cộng offset $0.5$ (vì SH mã hoá quanh màu trung bình 0):

$$
c_{rgb} = \max(0,\ \text{eval\_sh}(d) + 0.5)
$$

- **Hàm "zero trick"**: `screenspace_points` được tạo là tensor 0 có `requires_grad=True`
  chỉ để PyTorch autograd có chỗ "móc" gradient của toạ độ màn hình 2D
  $\nabla_{xy}$ (dùng ở `densify_and_split`/`add_densification_stats`) — không có công
  thức toán, là một kỹ thuật lập trình autograd phổ biến trong 3DGS.

Phần rasterization thật sự (dựng ma trận hiệp phương sai 2D $\Sigma'$ bằng phép chiếu
Jacobian $\Sigma' = JW\Sigma W^T J^T$, alpha-blending theo thứ tự độ sâu
$C=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)$...) được thực hiện **trong CUDA**
(`diff_gaussian_rasterization_structgs`), không nằm trong file Python này — xem mục C.

## B.2 `gaussian_renderer/network_gui.py` — 5 hàm (dòng 26–86)

Giao tiếp socket TCP thô với GUI xem trực tiếp (SIBR viewer). Không công thức toán,
chỉ là I/O mạng (bind/listen/accept, đọc/ghi JSON theo giao thức độ dài-tiền tố).

- `init`, `try_connect`, `read`, `send`: thuần socket I/O.
- `receive()`: parse JSON nhận được thành `MiniCam`; có 1 phép biến đổi hệ trục khi
  nhận ma trận view từ GUI (Unity/OpenGL convention → convention renderer), bằng cách
  đảo dấu 2 trục:

$$
V_{:,1} \leftarrow -V_{:,1}, \qquad V_{:,2} \leftarrow -V_{:,2}
$$

(áp dụng cho cả `world_view_transform` và cột Y của `full_proj_transform`).

## B.3 `gaussian_renderer/network_gui_ws.py` — 4 hàm (dòng 27–55)

Biến thể WebSocket (thay cho TCP thô ở B.2) để stream kết quả render. Thuần network
I/O (asyncio, struct pack), không công thức toán.

---

# C. Rasterizer CUDA (tham chiếu, không phải code Python trong 2 thư mục trên)

Vì `gaussian_renderer/__init__.py` gọi trực tiếp `GaussianRasterizer` từ submodule
`diff_gaussian_rasterization_structgs` (đã tài liệu hoá đầy đủ phần Python wrapper ở
`DOCS/BOOK/submodules.md`), công thức splatting lõi (chiếu Gaussian 3D→2D, alpha
compositing) nằm trong mã nguồn CUDA (`.cu`) không có trong phạm vi các file `.py`
được liệt kê ở yêu cầu này.

---

## Tổng kết số lượng hàm

| File | Số hàm/class |
|---|---|
| `arguments/__init__.py` | 2 class (`GroupParams`, `ParamGroup` với 2 method) + 3 subclass tham số + 1 hàm `get_combined_args` |
| `gaussian_renderer/__init__.py` | 1 hàm (`render_structgs`) |
| `gaussian_renderer/network_gui.py` | 5 hàm |
| `gaussian_renderer/network_gui_ws.py` | 4 hàm |
