# `utils/` — Giải thích chi tiết toàn bộ (trừ `graphics_utils.py`)

`utils/graphics_utils.py` đã được giải thích đầy đủ ở `DOCS/BOOK/scene_data_pipeline.md`
(mục D). File này bao phủ 7 file còn lại: `general_utils.py`, `loss_utils.py`,
`freq_utils.py`, `gaussian_sampling.py`, `sh_utils.py`, `image_utils.py`,
`camera_utils.py`, `system_utils.py`.

---

# A. `utils/general_utils.py` — 10 hàm

## 1. `identity_gate(x)` (dòng 18–19)

$$
f(x) = x
$$

Dùng làm `inverse_opacity_activation` thay thế khi `modify_functions()` đổi opacity
activation sang `abs` (xem `gaussian_model.py` mục 2).

## 2. `inverse_sigmoid(x)` (dòng 21–22)

$$
\sigma^{-1}(x) = \ln\!\left(\dfrac{x}{1-x}\right)
$$

## 3. `PILtoTorch(pil_image, resolution)` (dòng 24–30)

Resize ảnh PIL rồi chuẩn hoá về $[0,1]$ và đổi layout `HWC → CHW`:

$$
I_{torch} = \dfrac{I_{PIL}}{255}, \qquad \text{permute}(2,0,1)
$$

## 4. `get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1e6)` (dòng 32–65)

Trả về closure `helper(step)` tính learning rate suy giảm **log-tuyến tính** (tương
đương suy giảm mũ), có tuỳ chọn "delay" (khởi động chậm) kiểu cosine ngược.

Chuẩn hoá bước huấn luyện:

$$
t = \text{clip}\!\left(\dfrac{\text{step}}{\text{max\_steps}},\,0,\,1\right)
$$

Nội suy log-tuyến tính giữa `lr_init` và `lr_final` (tương đương suy giảm mũ theo step):

$$
\text{lr}_{base}(t) = \exp\!\big((1-t)\ln(\text{lr}_{init}) + t\ln(\text{lr}_{final})\big)
= \text{lr}_{init}^{1-t}\cdot \text{lr}_{final}^{\,t}
$$

Nếu có `lr_delay_steps > 0`, nhân thêm hệ số "trễ" dạng nửa-cosine, bắt đầu từ
`lr_delay_mult` rồi tiến về 1 khi `step` vượt `lr_delay_steps`:

$$
\tau = \text{clip}\!\left(\dfrac{\text{step}}{\text{lr\_delay\_steps}},0,1\right)
$$
$$
\text{delay\_rate}(\tau) = \text{lr\_delay\_mult} + (1-\text{lr\_delay\_mult})\sin\!\left(\dfrac{\pi}{2}\tau\right)
$$

Learning rate cuối cùng:

$$
\text{lr}(\text{step}) = \text{delay\_rate}\cdot \text{lr}_{base}(t)
$$

## 5. `strip_lowerdiag(L)` (dòng 67–76)

Lấy 6 phần tử độc lập (tam giác trên) của ma trận đối xứng 3×3 $L$ để lưu gọn:

$$
[L_{00},\,L_{01},\,L_{02},\,L_{11},\,L_{12},\,L_{22}]
$$

## 6. `strip_symmetric(sym)` (dòng 78–79)

Alias gọi lại #5.

## 7. `build_rotation(r)` (dòng 81–102)

Chuẩn hoá quaternion rồi dựng ma trận quay — **cùng công thức** với `qvec2rotmat` ở
`colmap_loader.py` nhưng viết vector hoá cho batch (xem công thức ở
`DOCS/BOOK/scene_data_pipeline.md` mục A.1), với $q=(r,x,y,z)/\lVert q_{raw}\rVert$:

$$
R=\begin{bmatrix}
1-2(y^2+z^2) & 2(xy-rz) & 2(xz+ry)\\
2(xy+rz) & 1-2(x^2+z^2) & 2(yz-rx)\\
2(xz-ry) & 2(yz+rx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

## 8. `matrix_to_quaternion(R)` (dòng 104–144)

Chiều ngược của #7, dùng **thuật toán Shepperd** (chọn 1 trong 4 công thức tuỳ phần tử
đường chéo nào lớn nhất, để tránh chia cho số gần 0 — ổn định số học hơn công thức đơn
nhất). Với vết ma trận $tr=R_{00}+R_{11}+R_{22}$:

- Nếu $tr>0$: $s=2\sqrt{tr+1}$, $q=\left(\dfrac{s}{4},\ \dfrac{R_{21}-R_{12}}{s},\ \dfrac{R_{02}-R_{20}}{s},\ \dfrac{R_{10}-R_{01}}{s}\right)$
- Nếu $R_{00}$ lớn nhất trên đường chéo: $s=2\sqrt{1+R_{00}-R_{11}-R_{22}}$, v.v. (tương tự hoán vị cho $R_{11}$, $R_{22}$ lớn nhất).

Sau đó chuẩn hoá lại: $q \leftarrow q/\lVert q\rVert$.

## 9. `build_scaling_rotation(s, r)` (dòng 146–155)

Dựng ma trận $L=R\cdot\text{diag}(s)$ dùng để tính hiệp phương sai $\Sigma=LL^T$
(xem `gaussian_model.py` mục 1):

$$
L = R\begin{bmatrix}s_x&0&0\\0&s_y&0\\0&0&s_z\end{bmatrix}
$$

## 10. `safe_state(silent)` (dòng 157–180)

Gắn timestamp vào mỗi dòng `print`, cố định toàn bộ seed ngẫu nhiên
($\text{random}, \text{numpy}, \text{torch}, \text{torch.cuda}$) bằng giá trị 0 để tái
lập kết quả. Không công thức toán.

---

# B. `utils/loss_utils.py` — 14 hàm

## 1. `l1_loss(network_output, gt)` (dòng 22–23)

$$
\mathcal L_1 = \dfrac1N\sum |I_{pred}-I_{gt}|
$$

## 2. `l2_loss(network_output, gt)` (dòng 25–26)

$$
\mathcal L_2 = \dfrac1N\sum (I_{pred}-I_{gt})^2
$$

## 3. `tone_curve_loss(network_output, gt, epsilon=1e-6, norm=2)` (dòng 28–34)

Loss "gradient supervision" — chuẩn hoá sai số theo cường độ dự đoán (stop-gradient ở
mẫu số, ký hiệu $\text{sg}(\cdot)$) để tránh vùng tối/sáng chi phối gradient:

$$
\mathcal L_{tone} = \dfrac1N\sum \left(\dfrac{I_{pred}-I_{gt}}{\text{sg}(I_{pred})+\epsilon}\right)^{norm}
$$

## 4. `gaussian(window_size, sigma)` (dòng 36–38)

Cửa sổ Gaussian 1D chuẩn hoá tổng = 1 (công thức giống mục `submodules.md` phần test):

$$
g_i = \dfrac{\exp\!\left(-\dfrac{(i-\lfloor w/2\rfloor)^2}{2\sigma^2}\right)}{\sum_j(\cdot)}
$$

## 5. `create_window(window_size, channel)` (dòng 40–44)

$$
W = g\,g^{T}\ (\text{2D}), \quad \text{lặp cho mỗi channel (depthwise)}
$$

## 6. `ssim(img1, img2, window_size=11, size_average=True)` + `_ssim(...)` (dòng 46–76)

Công thức SSIM chuẩn (Wang et al., 2004), với $\mu,\sigma^2,\sigma_{12}$ là trung bình/
phương sai/hiệp phương sai cục bộ ước lượng bằng conv2D cửa sổ Gaussian $W$:

$$
\mu_1=W*I_1,\quad \mu_2=W*I_2
$$
$$
\sigma_1^2=W*I_1^2-\mu_1^2,\quad \sigma_2^2=W*I_2^2-\mu_2^2,\quad
\sigma_{12}=W*(I_1 I_2)-\mu_1\mu_2
$$
$$
\text{SSIM}=\dfrac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)},
\qquad C_1=0.01^2,\ C_2=0.03^2
$$

(loss huấn luyện thường dùng $\mathcal L_{D\text{-}SSIM}=1-\text{SSIM}$, trộn với L1 theo
trọng số `lambda_dssim` ở `arguments/__init__.py`).

## 7. `frequency_loss(means2D, cov2D, st_map, H, W)` (dòng 80–110)

Phạt các Gaussian có bán kính màn hình lớn hơn bước sóng kết cấu địa phương (alias).

Chuẩn hoá toạ độ pixel về $[-1,1]$ để lấy mẫu song tuyến tính (`grid_sample`):

$$
n_x = \dfrac{2x}{W-1}-1,\qquad n_y=\dfrac{2y}{H-1}-1
$$

Lấy mẫu structure tensor $S_{xx}, S_{xy}, S_{yy}$ tại vị trí Gaussian. Với $a,b,c$ là 3
thành phần hiệp phương sai 2D $\Sigma'=\begin{bmatrix}a&b\\b&c\end{bmatrix}$:

$$
\det(\Sigma') = ac-b^2
$$
$$
\sigma_{area} = \sqrt{\det(\Sigma')},\qquad \sigma_{linear}=\sqrt{\sigma_{area}}
$$

(diện tích ellipse $\sim \sigma_{area}$, "bán kính" tuyến tính $\sim \sigma_{linear}$).

Tần số đặc trưng cục bộ và giới hạn bước sóng (Nyquist-like):

$$
f_{target} = \sqrt{S_{xx}+S_{yy}}, \qquad
\lambda_{limit} = \dfrac{1}{f_{target}+\epsilon}
$$

Loss phạt khi kích thước Gaussian vượt quá bước sóng cho phép:

$$
\mathcal L_{freq} = \dfrac1N\sum \text{ReLU}(\sigma_{linear} - \lambda_{limit})
$$

## 8. `fast_gaussian_blur(img, kernel_size, sigma)` (dòng 113–168)

Gaussian blur có tối ưu: nếu $\sigma$ lớn, downsample ảnh trước khi blur rồi upsample
lại (giảm chi phí tính toán vì kernel size tỉ lệ với $\sigma$). Khi downsample theo hệ
số `scale`, $\sigma$ tương ứng trong không gian đã co:

$$
\sigma_{down} = \dfrac{\sigma}{H/H_{target}}
$$

Kích thước kernel suy ra theo quy tắc $\pm4\sigma$ (bán kính $4\sigma$ đủ phủ $>99.99\%$
năng lượng Gaussian):

$$
k = \lceil 8\sigma+1\rceil \,|\, 1 \quad (\text{làm lẻ bằng OR bit 1})
$$

Nếu $\sigma$ vượt quá kích thước ảnh, fallback về giá trị trung bình toàn ảnh (xem như
blur với $\sigma\to\infty$):

$$
I_{blur} \approx \bar I = \dfrac1{HW}\sum_{x,y} I(x,y)
$$

## 9. `get_structure_tensor_torch(image_tensor, sigma=1.0, rho=1.0)` (dòng 170–230)

Tính **Structure Tensor đa kênh** (phương pháp Di Zenzo) — nền tảng cho toàn bộ loss
tần số StructGS.

Làm mịn ảnh trước (giảm nhiễu đạo hàm):

$$
I_{smooth} = G_\sigma * I
$$

Đạo hàm Sobel theo x, y (riêng từng kênh màu):

$$
I_x = S_x * I_{smooth}, \qquad I_y = S_y * I_{smooth}
$$
$$
S_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\quad
S_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}
$$

Tích đạo hàm, **cộng dồn năng lượng qua các kênh RGB** (Di Zenzo's method — tránh bỏ
sót biên "iso-luminant" chỉ đổi màu không đổi độ sáng):

$$
I_{xx}=\sum_c (I_x^{(c)})^2,\quad I_{yy}=\sum_c (I_y^{(c)})^2,\quad I_{xy}=\sum_c I_x^{(c)}I_y^{(c)}
$$

Tích phân cửa sổ (window integration) bằng blur Gaussian $\sigma=\rho$:

$$
S_{xx}=G_\rho * I_{xx},\quad S_{xy}=G_\rho * I_{xy},\quad S_{yy}=G_\rho * I_{yy}
$$

Chuẩn hoá theo giá trị lớn nhất của vết ma trận (per-batch) để ngưỡng không trôi theo
độ sáng ảnh:

$$
m = \max_{x,y}(S_{xx}+S_{yy}) + \epsilon,\qquad
S_{xx}\leftarrow \dfrac{S_{xx}}{m},\ S_{xy}\leftarrow\dfrac{S_{xy}}{m},\ S_{yy}\leftarrow\dfrac{S_{yy}}{m}
$$

Structure tensor tại mỗi pixel là:

$$
J=\begin{bmatrix}S_{xx}&S_{xy}\\S_{xy}&S_{yy}\end{bmatrix}
$$

(trị riêng của $J$ cho biết năng lượng gradient theo 2 hướng chính — dùng để ước lượng
tần số/định hướng kết cấu địa phương).

## 10. `get_multiscale_structure_tensor_v1(...)` (dòng 232–313)

Tổng hợp structure tensor qua nhiều "octave" tần số (giống kim tự tháp Laplacian),
nhưng **blur lại tensor cơ sở** ở mỗi scale thay vì tính lại từ ảnh (nhanh hơn v2,
kém chính xác hơn — xem `st_mode="v1"` trong `OptimizationParams`).

Với tầng $i=0,\dots,\text{levels}-1$, tần số dải và $\sigma$ mục tiêu:

$$
f_i = \dfrac{1}{\text{octave\_step}^{\,i}}, \qquad \sigma_i = \sigma_0\cdot\text{octave\_step}^{\,i}
$$

Làm mịn tăng dần (chỉ cần thêm $\sigma$ bù để đạt $\sigma_i$ từ $\sigma_{i-1}$ đã có,
nhờ tính chất cộng phương sai của Gaussian):

$$
\sigma_{inc} = \sqrt{\sigma_i^2 - \sigma_{i-1}^2}
$$

Đáp ứng dải tần (Difference of Gaussians — xấp xỉ Laplacian-of-Gaussian, đo năng lượng
ở dải tần đó):

$$
R_i = \sqrt{\sum_c (I_{smooth,i-1}-I_{smooth,i})^2}
$$

Chuẩn hoá tensor cơ sở về "chỉ hướng" (loại bỏ độ lớn, giữ tỉ lệ dị hướng) tại mỗi scale
bằng cách chia cho vết:

$$
\hat S^{(i)}_{xx} = \dfrac{S^{(i)}_{xx}}{S^{(i)}_{xx}+S^{(i)}_{yy}+\epsilon}\quad(\text{tương tự cho } xy, yy)
$$

Trọng số theo năng lượng dải ($p=$`power_factor`, mặc định 3) và dồn theo tỉ lệ nghịch
bước sóng bình phương $f_i^2$ (để tổng các tầng có vết $\approx$ bình phương tần số
tổng hợp):

$$
w_i = R_i^{\,p}, \qquad
S_{xx} \mathrel{+}= \hat S^{(i)}_{xx}\cdot w_i \cdot f_i^2 \quad (\text{tương tự } xy, yy)
$$

Chuẩn hoá cuối theo tổng trọng số:

$$
S_{xx}^{final} = \dfrac{\sum_i \hat S^{(i)}_{xx}w_if_i^2}{\sum_i w_i + \epsilon}
$$

## 11. `get_multiscale_structure_tensor_v2(...)` (dòng 315–387)

Biến thể "true multi-scale": **tính lại structure tensor trực tiếp trên ảnh đã làm mờ**
ở mỗi scale (gọi `get_structure_tensor_torch(next_smooth, sigma=target_sigma,
rho=3·target_sigma)`) thay vì blur lại tensor cơ sở như v1 — chính xác hơn về định
hướng/tần số tại từng scale, chi phí tính toán cao hơn. Công thức tổng hợp trọng số
và chuẩn hoá **giống hệt v1** (xem mục 10), chỉ khác nguồn $S_{xx}^{(i)}, S_{xy}^{(i)}, S_{yy}^{(i)}$.

## 12. `frequency_loss_simple(rendered_image, st_map)` (dòng 389–392)

So khớp trực tiếp structure tensor của ảnh render với ảnh GT bằng L1:

$$
\mathcal L = \text{L1}\big(\text{ST}(I_{render}),\ \text{ST}_{gt}\big)
$$

## 13. `estimate_required_gaussians(image, base_density=0.01, detail_sensitivity=0.5)` (dòng 394–437)

Heuristic ước lượng **số Gaussian cần thiết** cho 1 view, dựa trên tổng năng lượng
structure tensor (không phải loss, dùng để khởi tạo/chẩn đoán).

Vết tensor (năng lượng gradient cục bộ) tích phân toàn ảnh:

$$
E_{total} = \sum_{x,y}\big(S_{xx}(x,y)+S_{yy}(x,y)\big)
$$

Ước lượng tổng số Gaussian = phần "phủ nền" (tỉ lệ diện tích) + phần "chi tiết" (tỉ lệ
năng lượng kết cấu):

$$
N_{coverage} = \text{base\_density}\cdot H W, \qquad
N_{detail} = \text{detail\_sensitivity}\cdot E_{total}
$$
$$
N_{estimate} = \lfloor N_{coverage} + N_{detail} \rfloor
$$

## 14. `C1, C2` (hằng số module, dòng 19–20)

$$
C_1 = 0.01^2,\qquad C_2 = 0.03^2
$$

(hằng số ổn định SSIM — không phải hàm, nhưng xuất hiện lặp lại ở mọi công thức SSIM
trong file này và ở `fused-ssim`).

---

# C. `utils/freq_utils.py` — 6 hàm

Đây là nơi **định nghĩa chỉ số η** (dùng xuyên suốt `gaussian_model.py` cho split/expand
theo tần số — xem `DOCS/BOOK/gaussian_model.md` mục 31–32).

## 1. `sampling_cameras(my_viewpoint_stack, mode="fps", num_cams=60, weights=None)` (dòng 12–87)

Chọn tập con camera để tích luỹ thống kê mỗi "cửa sổ" densify.

- `mode="random"`: lấy mẫu đều ngẫu nhiên không hoàn lại.
- `mode="fps"` (**Farthest Point Sampling**): chọn camera sao cho khoảng cách tới tập
  đã chọn luôn lớn nhất, đảm bảo phủ đều không gian thay vì tụ cụm.

  Vị trí camera trong world space suy từ ma trận world-to-view $[R|t]$ (xem
  `cameras.py`):

$$
\mathbf c = -R^{T}\mathbf t
$$

  Thuật toán FPS lặp: chọn 1 camera ngẫu nhiên làm điểm đầu, sau đó ở mỗi bước chọn
  điểm có khoảng cách Euclid tới **tập đã chọn** (đo bằng $\min$ khoảng cách tới từng
  điểm đã chọn) là lớn nhất:

$$
d(\mathbf c_i,\,\mathcal S) = \min_{\mathbf c_j\in\mathcal S} \lVert \mathbf c_i-\mathbf c_j\rVert_2
$$
$$
\mathbf c_{next} = \arg\max_i\ d(\mathbf c_i,\,\mathcal S)
$$

## 2. `get_loss(reconstructed_image, original_image)` (dòng 92–96)

Bản đồ lỗi L1 theo pixel (trung bình qua kênh màu), chuẩn hoá về $[0,1]$ bằng
min-max scaling:

$$
e(x,y) = \dfrac1C\sum_c |I_{pred}(x,y,c)-I_{gt}(x,y,c)|
$$
$$
\hat e = \dfrac{e-\min(e)}{\max(e)-\min(e)}
$$

## 3. `compute_photometric_loss(viewpoint_cam, image)` (dòng 98–102)

Loss ảnh tổng hợp chuẩn 3DGS, trộn L1 và $1-\text{SSIM}$ với trọng số cố định $0.2$:

$$
\mathcal L = 0.8\cdot \mathcal L_1 + 0.2\cdot\big(1-\text{SSIM}\big)
$$

## 4. `normalize(config_value, value_tensor)` (dòng 104–114)

Chuẩn hoá một tensor giá trị (vd. điểm quan trọng) theo **trung vị** của các giá trị
dương, nhân với hệ số cấu hình — bền với outlier hơn chuẩn hoá theo mean/max:

$$
\hat v_i = \text{multiplier}\cdot \dfrac{v_i}{\text{median}(\{v_j : v_j>0\})} \quad (\text{nếu } v_i>0,\ \text{ngược lại } 0)
$$

## 5. `compute_projected_axes_subset(means2D, depths, scales, rotations, viewpoint_camera)` (dòng 116–178)

Tính độ dài hình chiếu 2D của 3 trục chính Gaussian (dùng để so sánh với bước sóng kết
cấu, suy ra η — mục 6). Đây là **xấp xỉ tuyến tính hoá phép chiếu phối cảnh** (giống
Jacobian $J$ trong rasterizer CUDA, nhưng viết lại ở Python cho một tập con điểm).

Toạ độ camera-space suy ngược từ pixel + độ sâu (pinhole ngược):

$$
x_{cam} = \dfrac{(u-W/2)\cdot z}{f_x}, \qquad y_{cam} = \dfrac{(v-H/2)\cdot z}{f_y}
$$

Jacobian phép chiếu phối cảnh tại điểm đó:

$$
J=\begin{bmatrix}
f_x/z & 0 & -f_x x_{cam}/z^2\\
0 & f_y/z & -f_y y_{cam}/z^2
\end{bmatrix}
$$

Xoay 3 trục chính của Gaussian (đã nhân scale) từ hệ local sang hệ camera:

$$
R_{total} = R_{view}\cdot R_{local}, \qquad
\mathbf a_k = R_{total}\cdot(s_k\,\hat e_k) \quad (k=1,2,3\ \text{trục})
$$

Chiếu mỗi trục 3D sang 2D bằng $J$ (bỏ chiều $z$ trong Jacobian hàng, chỉ giữ $x,z$ và
$y,z$):

$$
\begin{bmatrix}u_k\\v_k\end{bmatrix} = J\!\begin{bmatrix}a_{k,x}\\a_{k,y}\\a_{k,z}\end{bmatrix}
$$

## 6. `update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, ...)` (dòng 181–418)

Hàm **cốt lõi nhất** của cơ chế StructGS — tích luỹ online chỉ số vi phạm tần số $\eta$
cho mỗi Gaussian nhìn thấy trong 1 view, dùng để quyết định split (η cao) hay giữ
nguyên/mở rộng (η thấp) ở `gaussian_model.py`.

### Bước 1 — Lọc Gaussian "đang hoạt động" (active & visible)

Chỉ xét Gaussian có độ truyền qua (transmittance) và opacity đủ lớn, tuỳ chọn thêm điều
kiện gradient cao:

$$
\text{active} = (T_{max} > \tau_T) \wedge (\alpha > \tau_\alpha) \wedge \big(\lVert\nabla_{xy}\rVert > \tau_{grad}\big)_{\text{nếu có}}
$$

### Bước 2 — Lấy mẫu "jitter" ngẫu nhiên trong ellipse Gaussian (thay vì chỉ lấy tâm)

Dùng phân rã Cholesky của hiệp phương sai 2D $\Sigma'=\begin{bmatrix}\sigma_{xx}&\sigma_{xy}\\\sigma_{xy}&\sigma_{yy}\end{bmatrix}=LL^T$:

$$
L_{11}=\sqrt{\sigma_{xx}},\qquad L_{21}=\dfrac{\sigma_{xy}}{L_{11}},\qquad L_{22}=\sqrt{\sigma_{yy}-L_{21}^2}
$$

Sinh nhiễu Gaussian 2 chiều $\mathcal N(0,\Sigma')$ bằng biến ngẫu nhiên chuẩn độc lập
$\epsilon_1,\epsilon_2\sim\mathcal N(0,1)$:

$$
\Delta x = L_{11}\epsilon_1, \qquad \Delta y = L_{21}\epsilon_1 + L_{22}\epsilon_2
$$

(đúng công thức lấy mẫu phân phối chuẩn đa biến qua Cholesky: $\mathbf z=L\boldsymbol\epsilon\sim\mathcal N(0,LL^T)$).

Toạ độ lấy mẫu = tâm Gaussian + nhiễu, rồi lấy mẫu `grid_sample` structure tensor tại
đó (thay vì chỉ tại tâm — giảm thiên lệch khi Gaussian lớn phủ lên vùng không đồng nhất).

### Bước 3 — Tính η theo 1 trong 2 chế độ (`eta_compute_mode`)

**Chế độ `"wavelength"` (mặc định, so sánh kích thước và bước sóng)**:

Trị riêng lớn nhất (năng lượng tần số cao nhất) của structure tensor $2\times2$
$\begin{bmatrix}S_{xx}&S_{xy}\\S_{xy}&S_{yy}\end{bmatrix}$:

$$
\text{tr}=S_{xx}+S_{yy},\quad \det=S_{xx}S_{yy}-S_{xy}^2
$$
$$
\lambda_1 = \dfrac{\text{tr}}{2} + \sqrt{\max\!\left(0,\ \left(\dfrac{\text{tr}}{2}\right)^2-\det\right)}
$$

Bước sóng tối thiểu (nghịch đảo căn bậc hai năng lượng — xấp xỉ $f\sim\sqrt\lambda$,
$\lambda_{wave}\sim1/f$):

$$
\lambda_{min} = \dfrac{1}{\sqrt{\lambda_1}+\epsilon}
$$

Độ dài hình chiếu từng trục Gaussian trên màn hình (từ mục 5):

$$
L_k = \sqrt{u_k^2+v_k^2}
$$

**Chỉ số vi phạm Nyquist** = tỉ lệ kích thước Gaussian / bước sóng kết cấu — nếu $\eta>1$,
Gaussian "to hơn" chi tiết cần biểu diễn ⇒ alias ⇒ cần split:

$$
\eta_k = \dfrac{L_k}{\lambda_{min}}
$$

**Chế độ `"projection"` (chiếu trực tiếp structure tensor lên từng trục)**:

$$
\eta_k = \sqrt{S_{xx}u_k^2 + 2S_{xy}u_kv_k + S_{yy}v_k^2}
$$

(dạng toàn phương $\mathbf a_k^T J\,\mathbf a_k$ — năng lượng tần số chiếu theo đúng
hướng trục Gaussian thứ $k$, thay vì theo hướng trị riêng chính như chế độ trên).

### Bước 4 — Trọng số theo transmittance và tích luỹ

$$
\eta_k \leftarrow \eta_k \cdot w_{valid}, \qquad \eta_{total}=\sum_k \eta_k
$$

Tích luỹ "leaky running max" theo từng kênh và theo tổng, cùng bộ đếm số view:

$$
\text{accum\_eta} \mathrel{+}= \eta_{total}, \qquad
\text{max\_eta\_3ch} \leftarrow \max(\text{max\_eta\_3ch},\ \eta_{1:3})
$$

### Bước 5 — Phân loại nhất quán đa góc nhìn (multiview consistency)

Với $\eta_{max}=\max_k \eta_k$ và 2 ngưỡng cố định $\tau_{high}=1.0,\ \tau_{low}=0.1$:

$$
\text{high nếu } \eta_{max}>\tau_{high},\quad
\text{low nếu } \eta_{max}\le\tau_{low},\quad
\text{mid: còn lại}
$$

cập nhật bộ đếm `eta_high_count/eta_mid_count/eta_low_count` và tổng
`eta_high_sum_3ch/eta_mid_sum_3ch` tương ứng — dùng ở nơi khác (ngoài phạm vi các file
được liệt kê) để quyết định split/prune theo đa số phiếu từ nhiều view thay vì 1 view
đơn lẻ.

---

# D. `utils/gaussian_sampling.py` — 1 hàm

## `sample_anisotropic_gaussians_2d(image_tensor, num_samples=2000, sigma=1.0, rho=1.5, seed=42)` (dòng 11–129)

Lấy mẫu vị trí + hướng Gaussian 2D dị hướng từ 1 ảnh, dựa trên phân tích structure
tensor — dùng cho khởi tạo/thử nghiệm fit ảnh 2D (không phải pipeline 3D chính).

Bản đồ "năng lượng" dùng làm xác suất lấy mẫu (vết structure tensor, giống mục C.5
`gaussian_sampling` nhưng đây là bước **sampling theo xác suất**, không phải loss):

$$
E(x,y) = S_{xx}(x,y)+S_{yy}(x,y), \qquad
p(x,y) = \dfrac{E(x,y)}{\sum_{x',y'}E(x',y')}
$$

Lấy mẫu toạ độ không hoàn lại theo phân phối $p$ (importance sampling — vùng nhiều chi
tiết được lấy mẫu dày hơn).

Tại mỗi điểm lấy mẫu, dựng ma trận structure tensor $2\times2$ (đã cộng $\epsilon$ vào
đường chéo để đảm bảo xác định dương):

$$
J=\begin{bmatrix}S_{xx}+\epsilon & S_{xy}\\ S_{xy} & S_{yy}+\epsilon\end{bmatrix}
$$

**Phân rã trị riêng** (`eigh`, vì $J$ đối xứng):

$$
J = V\Lambda V^{T}, \qquad \lambda_1\ge\lambda_2
$$

$\lambda_1$ (lớn) ứng với hướng gradient mạnh nhất (vuông góc biên), $\lambda_2$ (nhỏ)
ứng với **hướng dọc biên** (edge direction) — dùng vector riêng $v_2$ để định hướng
Gaussian dị hướng sẽ đặt tại điểm đó. Góc định hướng:

$$
\theta = \text{atan2}(v_{2,y},\, v_{2,x})
$$

---

# E. `utils/sh_utils.py` — 3 hàm

## 1. `eval_sh(deg, sh, dirs)` (dòng 57–112)

Khai triển cầu điều hòa (Spherical Harmonics) thực, bậc $0$–$4$, đánh giá tại hướng đơn
vị $\mathbf d=(x,y,z)$, dùng hệ số hằng số đã rút gọn sẵn (tránh tính Legendre tổng quát):

$$
c(\mathbf d) = \sum_{\ell=0}^{deg}\sum_{m=-\ell}^{\ell} k_{\ell m}\cdot Y_{\ell m}(\mathbf d)\cdot sh_{\ell m}
$$

Bậc 0 (hằng số):

$$
c_0 = C_0\cdot sh_0, \qquad C_0=0.282094791...
$$

Bậc 1 (tuyến tính theo $x,y,z$):

$$
c_1 = -C_1\,y\,sh_1 + C_1\,z\,sh_2 - C_1\,x\,sh_3,\qquad C_1=0.488602512...
$$

Bậc 2 (bậc hai theo $x,y,z$, hệ số $C_2[0..4]$):

$$
c_2 = C_2[0]xy\,sh_4+C_2[1]yz\,sh_5+C_2[2](2z^2-x^2-y^2)sh_6+C_2[3]xz\,sh_7+C_2[4](x^2-y^2)sh_8
$$

Bậc 3, 4 dùng công thức đa thức cụ thể tương ứng, hệ số cố định $C_3[0..6]$, $C_4[0..8]$
(xem mã nguồn dòng 93–111 — không viết lại khai triển đầy đủ ở đây vì chỉ là đa thức
cầu điều hoà chuẩn, không có cấu trúc rút gọn thêm).

Kết quả cuối cùng: $c(\mathbf d)=\sum_{\ell\le deg} c_\ell$ (tổng dồn theo bậc).

## 2. `RGB2SH(rgb)` (dòng 114–115)

$$
sh_0 = \dfrac{rgb - 0.5}{C_0}
$$

## 3. `SH2RGB(sh)` (dòng 117–118)

$$
rgb = sh\cdot C_0 + 0.5
$$

(2 hàm này là nghịch đảo của nhau, dùng để khởi tạo màu DC ban đầu từ RGB và chiều ngược
lại khi cần — ví dụ sinh point cloud ngẫu nhiên ở `dataset_readers.py`).

---

# F. `utils/image_utils.py` — 2 hàm

## 1. `mse(img1, img2)` (dòng 14–15)

$$
\text{MSE} = \dfrac1N\sum (I_1-I_2)^2
$$

## 2. `psnr(img1, img2)` (dòng 17–19)

Công thức PSNR chuẩn cho ảnh đã chuẩn hoá $[0,1]$ (MAX = 1):

$$
\text{PSNR} = 20\log_{10}\!\left(\dfrac{1}{\sqrt{\text{MSE}}}\right) = -10\log_{10}(\text{MSE})
$$

---

# G. `utils/camera_utils.py` — 3 hàm

## 1. `loadCam(args, id, cam_info, resolution_scale)` (dòng 19–52)

Tính độ phân giải đích theo hệ số scale (chia đôi/4/8 hoặc tự động co về $\le1600$px
chiều rộng), resize ảnh, tách kênh alpha làm mask nếu có (RGBA). Không công thức phức
tạp — chỉ là số học chia tỉ lệ:

$$
(W',H') = \left(\dfrac{W}{s},\ \dfrac{H}{s}\right), \qquad s = \text{global\_down}\times\text{resolution\_scale}
$$

## 2. `cameraList_from_camInfos(cam_infos, resolution_scale, args)` (dòng 54–60)

Lặp `loadCam` cho toàn bộ danh sách camera. Không công thức.

## 3. `camera_to_JSON(id, camera)` (dòng 62–82)

Chuyển `Camera` sang dict JSON để lưu (`cameras.json`), gồm vị trí/hướng world (nghịch
đảo ma trận world-to-view — cùng công thức `getWorld2View` ở
`DOCS/BOOK/scene_data_pipeline.md` mục D.2) và tiêu cự pixel suy từ FOV:

$$
W2C = \begin{bmatrix}R^T & T\\0&1\end{bmatrix}, \qquad
C2W = W2C^{-1}, \qquad \text{pos} = C2W_{[0:3,3]},\ \text{rot}=C2W_{[0:3,0:3]}
$$
$$
f_x = \text{fov2focal}(\text{FoVx}, W), \qquad f_y = \text{fov2focal}(\text{FoVy}, H)
$$

---

# H. `utils/system_utils.py` — 2 hàm

## 1. `mkdir_p(folder_path)` (dòng 16–24)

Tương đương `mkdir -p` Unix — tạo thư mục đệ quy, bỏ qua lỗi nếu đã tồn tại. Không công thức.

## 2. `searchForMaxIteration(folder)` (dòng 26–28)

Tìm số iteration checkpoint lớn nhất trong thư mục (`iteration_<N>`):

$$
N_{max} = \max_i \{N : \text{"iteration\_"+N} \in \text{folder}\}
$$

---

## Tổng kết số lượng hàm theo file

| File | Số hàm |
|---|---|
| `general_utils.py` | 10 |
| `loss_utils.py` | 13 hàm + 2 hằng số module (`C1`, `C2`) |
| `freq_utils.py` | 6 |
| `gaussian_sampling.py` | 1 |
| `sh_utils.py` | 3 |
| `image_utils.py` | 2 |
| `camera_utils.py` | 3 |
| `system_utils.py` | 2 |
| **Tổng** | **40 hàm** (không tính `graphics_utils.py`, đã có ở tài liệu khác) |

**Trọng tâm toán học của toàn bộ `utils/`**: `loss_utils.py` (SSIM, structure tensor đa
scale, tone-curve loss) và `freq_utils.py` (chỉ số η Nyquist dùng để điều khiển
split/expand Gaussian theo tần số) là 2 file nặng công thức nhất — nền tảng lý thuyết
đằng sau toàn bộ cơ chế StructGS trong repo này.
