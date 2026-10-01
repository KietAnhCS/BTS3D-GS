# Pipeline theo code thật (`train.py` + `gaussian_model.py`)

Có code rồi nên mình sửa lại một số chỗ đoán sai ở câu trả lời trước:

- **SH degree** tăng **mỗi iteration** (`iteration % 1 == 0`), không phải mỗi 1000 iter.
- **Không có loss tần số.** `lambda_freq` và `lambda_tone` không được dùng. Phần "frequency" nằm ở **tiêu chí densify/prune**, không nằm ở loss.
- `lambda_l2 = 2` **có** dùng trong loss.
- LR của SH bậc cao là `highfeature_lr/20`, tức **0.00025**, không phải 0.005.
- Prune 6000–30000 mỗi 3000 iter **không tồn tại** trong code. Prune thật sự chạy theo cách khác (xem Bước 4 và 6).

Riêng `update_freq_stats_online` nằm trong `utils/freq_utils.py` mà mình chưa thấy. Chỗ nào phụ thuộc hàm đó mình ghi **[chưa thấy code]**.

---

## Bước 0. Khởi tạo Gaussian từ point cloud

Với mỗi điểm COLMAP $p_i$:

$$\mu_i=p_i,\qquad \sigma_i=\sqrt{\tfrac13\textstyle\sum_{j\in kNN_3(i)}\|p_i-p_j\|^2}\ \ (\text{cho cả 3 trục}),\qquad R_i=I,\qquad \alpha_i=0.1$$

Tham số lưu trong bộ nhớ:

- Scale: $s_i=\ln\sigma_i$ (log-scale).
- Opacity: $o_i=\mathrm{logit}(0.1)$.
- Màu: $f^{dc}_i=\mathrm{RGB2SH}(c_i)$, còn $f^{rest}_i=0$.

Hiệp phương sai:

$$\Sigma_i=R_i\,\mathrm{diag}(\sigma_i)^2R_i^\top,\qquad \sigma_i=e^{s_i}$$

---

## Bước 1. Tiền xử lý: structure tensor (1 lần trước khi train)

Với mỗi ảnh train $I$, tính structure tensor **đa tỉ lệ** ($L=$ `st_levels`$=4$ mức). Theo comment trong code, lưu 3 kênh $(S_{xx},S_{xy},S_{yy})$:

$$S(p)=G_\rho*\begin{bmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2\end{bmatrix}(p)$$

Trị riêng của $S$ đo **năng lượng gradient** (độ "nhiều chi tiết") cục bộ, tức $\omega^2$ của tín hiệu ảnh. Kết quả được cache theo tên ảnh. `st_mode` chọn `v1` hoặc `v2`, hai cách tính khác nhau **[chưa thấy code]**.

---

## Bước 2. Mỗi iteration $t=1\dots30000$

### 2a. Learning rate

Chỉ $\mu$ có lịch giảm (`update_learning_rate`); với `scale_rotation_scheduler=False` thì scale và rotation cố định:

$$\mathrm{lr}_\mu(t)=\underbrace{r_{scene}}_{\text{cameras\_extent}}\cdot\delta(t)\cdot\exp\big((1-w)\ln 1.6\!\times\!10^{-4}+w\ln 1.6\!\times\!10^{-6}\big),\quad w=\tfrac{t}{30000}$$

Ở đây $\delta(t)$ là hệ số delay (mặc định $=1$).

| Nhóm | LR | Optimizer |
|---|---|---|
| $\mu$ | công thức trên | Adam (amsgrad) |
| $f^{dc}$ | 0.0025 | Adam |
| $o$ | 0.05 | Adam |
| $s$ | 0.01 | Adam |
| $q$ | 0.002 | Adam |
| $f^{rest}$ | $0.005/20=0.00025$ | SparseGaussianAdam (chỉ cập nhật Gaussian nhìn thấy, `radii>0`) |

Đây là chế độ `hybrid`, với $\epsilon=10^{-8}$.

### 2b. SH degree

$$\deg_{active}(t)=\min(t,\,3)$$

Nghĩa là sau đúng 3 iteration đã dùng full SH bậc 3.

### 2c. Chọn camera + render

Xáo trộn toàn bộ camera, pop lần lượt (`camera_sampling="random"`). Render $\hat I$ bằng `render_structgs` (rasterizer có `mult=0.7` cho compact box, và trả thêm `cov2D`). Splatting vẫn theo công thức chuẩn:

$$\hat C(p)=\sum_i c_i\,\alpha_i(p)\prod_{j<i}(1-\alpha_j(p))$$

### 2d. Loss (đúng như code)

$$\boxed{\mathcal L=0.8\,\|\hat I-I\|_1+0.2\,\big(1-\mathrm{SSIM}(\hat I,I)\big)+2.0\,\|\hat I-I\|_2^2}$$

Mình giả định `l2_loss` là MSE (`loss_utils.py` chưa thấy). Sau đó `loss.backward()`. Vì `batch_size=1` nên mỗi iteration tương ứng một ảnh, còn nếu tăng batch thì gradient được **cộng dồn (không chia trung bình)**.

---

## Bước 3. Thu thập thống kê (để dùng cho densify)

**(a) Gradient, mỗi iteration.** Với các Gaussian nhìn thấy, $g$ là gradient theo viewspace, $g_{1:2}$ là 2 thành phần đầu và $g_{3:4}$ là 2 thành phần sau:

$$G_i\mathrel{+}=\|g_{1:2}\|,\qquad G^{abs}_i\mathrel{+}=\|g_{3:4}\|,\qquad n_i\mathrel{+}=1$$

Về sau dùng $\bar G_i=G_i/n_i$ và $\bar G^{abs}_i=G^{abs}_i/n_i$.

**(b) Tần số, mỗi 10 iteration** (chỉ khi $t<15000$). Với camera $v$ vừa render, mỗi Gaussian nhìn thấy trên **3 trục cục bộ** $a\in\{x,y,z\}$ có một chỉ số

$$\eta^{(v)}_{i,a}\;\approx\;(\sigma_{i,a}\,\omega_v)^2$$

Đây là tỉ số giữa kích thước Gaussian và tần số tín hiệu ảnh tại vị trí Gaussian chiếu lên (lấy từ structure tensor). Theo comment trong `expand_undersized_gs`, $\eta=1$ tương ứng ngưỡng Nyquist. Công thức chính xác nằm trong `update_freq_stats_online`. Gaussian được ghi nhận vào một trong 3 nhóm high / mid / low theo ngưỡng (ngưỡng ở `freq_utils` **[chưa thấy code]**), và cập nhật:

- $V_i$ = số view đã ghi nhận (`accum_view_count`)
- $H_i$, $L_i$ = số view mà $\eta$ thuộc nhóm high, low
- $\eta^{max}_{i,a}=\max_v\eta^{(v)}_{i,a}$ (`max_eta_3ch`, gần như chắc chắn là max theo view)

Vì densify cách nhau 100 iter và cứ 10 iter lấy 1 view, mỗi lần densify có khoảng **10 view** để thống kê.

---

## Bước 4. Densify + prune: mỗi 100 iter, với $500<t<15000$ ($t=600,700,\dots,14900$)

### 4.1 Tính các mask

$$\rho^{H}_i=\frac{H_i}{V_i},\qquad \rho^{L}_i=\frac{L_i}{V_i}\quad(V_i>0)$$

- **Split tần số:** $M^{freq}_i=\mathbb 1[\rho^H_i>0.8]\ \wedge\ \mathbb 1[\|\bar G_i\|\ge 10^{-5}]$ (ngưỡng $10^{-5}$ hard-code trong `train.py`)
- **Prune tần số:** $M^{low}_i=\mathbb 1[\rho^L_i>0.8]\ \wedge\ \mathbb 1[V_i>0]$
- **Điều kiện gradient:** $A_i=\mathbb 1[\|\bar G_i\|\ge 0.0002]$, $A^{abs}_i=\mathbb 1[\|\bar G^{abs}_i\|\ge 0.0002]$
- **Kích thước:** nhỏ nếu $\max_a\sigma_{i,a}\le 0.001\,r_{scene}$, lớn nếu ngược lại
- **Mask "đã được thấy":** $M^{seen}_i=\mathbb 1[V_i>0.5]$, tức Gaussian phải xuất hiện ít nhất 1 lần trong các view mẫu (không phải điểm quan trọng theo lỗi như mình đoán trước)

Kết hợp:

$$\text{Split}_i=\big(M^{freq}_i\vee A^{abs}_i\big)\wedge \text{lớn}_i\wedge M^{seen}_i$$

$$\text{Clone}_i=\big(M^{freq}_i\vee A_i\big)\wedge \text{nhỏ}_i\wedge M^{seen}_i$$

### 4.2 Clone

Sao chép nguyên xi Gaussian (cùng $\mu,s,q,o,f$), không dịch chuyển. Gradient các bước sau sẽ tự tách chúng ra.

### 4.3 Split giải tích (phần đặc trưng của method này)

Số mảnh mỗi trục, dựa trên định lý lấy mẫu ($\sigma/k$ phải thoả $(\sigma\omega/k)^2\le1\Rightarrow k\ge\sqrt\eta$):

$$k_{i,a}=\Big\lceil\sqrt{\max(\eta^{max}_{i,a},\,1)}\Big\rceil\ \ge 1$$

Tổng số con: $N_i=k_{i,x}\,k_{i,y}\,k_{i,z}$. Ví dụ $\eta^{max}=(4,1,9)\Rightarrow k=(2,1,3)\Rightarrow 6$ con.

Scale mới của con (với $p=$ `ks_scale_power`$=1$):

$$\sigma'_{i,a}=\frac{\sigma_{i,a}}{k_{i,a}^{\,p}}$$

Vị trí con là lưới đều trong hệ tọa độ cục bộ của cha, rồi xoay ra world:

$$g_a=j_a-\tfrac{k_{i,a}-1}{2},\ \ j_a\in\{0,\dots,k_{i,a}-1\},\qquad \mu'=\mu_i+R_i\,\Big(\sqrt{12}\,\tfrac{\sigma_{i,a}}{k_{i,a}}\,g_a\Big)_{a}$$

Hệ số $\sqrt{12}$ là quan hệ giữa độ rộng và độ lệch chuẩn của phân bố đều. Con **thừa kế nguyên** opacity, SH, rotation của cha (không chia opacity). Cha bị xóa.

Lưu ý: comment ghi "max 8 mỗi trục" nhưng code chỉ chặn `min=1`. `max_clones_per_axis` **không được dùng**, nên số con có thể rất lớn nếu $\eta$ lớn.

### 4.4 Prune (ngay sau đó)

Xóa Gaussian $i$ nếu **một trong** các điều kiện:

$$\alpha_i<0.1\quad\vee\quad M^{low}_i\quad\vee\quad\big(t>3000\ \wedge\ (r^{2D}_{max,i}>20\text{px}\ \vee\ \max_a\sigma_{i,a}>0.1\,r_{scene})\big)$$

Ngưỡng opacity $0.1$ khá cao so với 3DGS gốc ($0.005$).

### 4.5 Clamp opacity

$$\alpha_i\leftarrow\min(\alpha_i,\,0.8)$$

Bước này cũng **reset trạng thái Adam** của opacity về 0.

Cuối cùng reset toàn bộ bộ đếm ($V,H,L,\eta^{max}$, và $G,G^{abs},n$ được reset trong `densification_postfix`).

---

## Bước 5. Reset opacity: $t\in\{3000,6000,9000,12000\}$

Reset chạy sau khi densify trong cùng iteration:

$$\alpha_i\leftarrow 0.1\,\alpha_i$$

Vì opacity đã bị clamp $\le0.8$ trước đó, sau reset mọi Gaussian có $\alpha\le0.08<0.1$. Ở lần densify kế tiếp (100 iter sau), Gaussian nào **không hồi opacity lên trên 0.1** kịp sẽ bị prune (điều kiện $\alpha<0.1$ ở 4.4). Đây chính là cơ chế loại "floaters", và lr opacity lớn ($0.05$) giúp Gaussian hữu ích hồi phục nhanh.

---

## Bước 6. Prune cứng theo danh sách iteration

Tại $t\in$ `prune_iterations` (mặc định **$\{4000,8000\}$**, đổi bằng `--prune_iterations`):

$$\text{xóa nếu }\alpha_i<0.1$$

Các tham số `prune_from_iter`, `prune_interval`, `densify_prune_ratio`, `after_densify_prune_ratio`, `min_weight`... **không được dùng** trong hai file này.

---

## Bước 7. Cập nhật tham số

$$\theta\leftarrow\theta-\mathrm{lr}\cdot\frac{\hat m}{\sqrt{\hat v_{max}}+10^{-8}}$$

Dùng Adam với amsgrad cho nhóm chính và SparseGaussianAdam cho $f^{rest}$, mỗi iteration (trừ iteration cuối). Nhánh lịch bước 15000/20000 chỉ chạy khi `optimizer_type="default"`, không phải `hybrid`.

---

## Timeline thực tế

| Iter | Việc xảy ra |
|---|---|
| 1–3 | SH degree lên 3 |
| mỗi iter | render, loss (L1 + SSIM + 2·L2), backward, cộng gradient, Adam |
| mỗi 10 (<15000) | cập nhật thống kê $\eta$ |
| 600…14900, mỗi 100 | clone / split giải tích / prune / clamp $\alpha\le0.8$ |
| 3000, 6000, 9000, 12000 | reset opacity $\times0.1$ |
| 4000, 8000 | prune $\alpha<0.1$ |
| 15000–30000 | không đổi cấu trúc, chỉ tinh chỉnh tham số |

---

## Điểm lạ / cần kiểm tra

1. `max_clones_per_axis=8` không được áp dụng, nên $k$ có thể vượt 8 và tốn VRAM (comment trong code cũng nhắc "7M points").
2. Nhánh `warmup_densification` dùng `densify_and_prune` kiểu 3DGS gốc với `densify_grad_abs_threshold=0.0004`. Tham số này **chỉ** dùng ở nhánh đó, còn nhánh chính dùng `grad_abs_thresh=0.0002`.
3. `max_radii2D` bị đặt về 0 trong `densification_postfix` trước khi prune, nên điều kiện "quá to trên màn hình" ở 4.4 gần như không có tác dụng (tính chất này cũng có ở 3DGS gốc).
4. `compute_3d_filter=False` nên bộ lọc 3D kiểu Mip-Splatting tắt; scale/opacity dùng nguyên giá trị gốc.
5. Ngưỡng high/mid/low của $\eta$ và công thức $\eta$ cụ thể nằm ở `freq_utils.py`. Bạn gửi file đó (và `loss_utils.py` cho structure tensor + `l2_loss`) thì mình viết nốt Bước 1 và 3b thành công thức chính xác.