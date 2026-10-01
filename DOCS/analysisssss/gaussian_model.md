q# `scene/gaussian_model.py` — Giải thích chi tiết từng hàm

File này định nghĩa class `GaussianModel`, đại diện cho tập hợp các 3D Gaussian
(point cloud các Gaussian dị hướng) dùng trong 3D Gaussian Splatting (3DGS) và
các biến thể mở rộng (StructGS, bộ lọc 3D, thống kê đa góc nhìn η...).

Tổng cộng class có **39 hàm** (tính cả hàm lồng bên trong `setup_functions`,
các `@property` getter, và các hàm "private" bắt đầu bằng `_`).

Danh sách nhóm theo chức năng:

- Khởi tạo & activation functions: `setup_functions`, `modify_functions`, `__init__`
- Save/restore checkpoint: `capture`, `restore`
- Properties (getter có áp activation): `get_scaling`, `get_scaling_with_3D_filter`,
  `get_rotation`, `get_xyz`, `get_features`, `get_features_dc`, `get_features_rest`,
  `get_opacity`, `get_opacity_with_3D_filter`, `get_covariance`
- Bộ lọc 3D (mip-splatting): `compute_3D_filter`
- Khởi tạo từ point cloud & huấn luyện: `oneupSHdegree`, `create_from_pcd`,
  `training_setup`, `update_learning_rate`, `optimizer_step`
- Lưu/đọc PLY: `construct_list_of_attributes`, `save_ply`, `load_ply`, `reset_opacity`
- Quản lý optimizer khi thêm/xoá điểm: `replace_tensor_to_optimizer`,
  `_prune_optimizer`, `prune_points`, `cat_tensors_to_optimizer`, `densification_postfix`
- Densify / Prune (chuẩn 3DGS): `densify_and_split`, `densify_and_clone`,
  `densify_and_prune`, `add_densification_stats`
- Densify / Prune (StructGS mở rộng theo η tần số): `densify_and_split_structgs`,
  `expand_undersized_gs`, `densify_and_clone_structgs`, `densify_and_prune_structgs`,
  `final_prune_structgs`

---

## 1. `setup_functions(self)` (dòng 32–46)

Thiết lập các hàm activation áp lên tham số thô (raw parameter) để đảm bảo ràng buộc
vật lý (scale > 0, opacity ∈ [0,1], rotation là quaternion đơn vị).

### Hàm con `build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation)`

Công thức ma trận hiệp phương sai 3D của một Gaussian dị hướng:

$$
\Sigma = R\,S\,S^{T}R^{T} = L L^{T}, \qquad L = R \cdot (s_{mod}\cdot S)
$$

trong đó `S = diag(sx, sy, sz)` là ma trận scale, `R` là ma trận quay dựng từ quaternion,
`s_mod` là `scaling_modifier`. Sau đó chỉ giữ phần tam giác trên (đối xứng) của `Σ`
(`strip_symmetric`) để lưu gọn thành 6 giá trị.

### Các activation được gán

| Thuộc tính | Activation | Công thức |
|---|---|---|
| `scaling_activation` | `exp` | $s = e^{s_{raw}}$ |
| `scaling_inverse_activation` | `log` | $s_{raw} = \ln(s)$ |
| `covariance_activation` | hàm trên | $\Sigma = LL^T$ |
| `opacity_activation` | `sigmoid` | $\alpha = \dfrac{1}{1+e^{-o_{raw}}}$ |
| `inverse_opacity_activation` | `inverse_sigmoid` | $o_{raw} = \ln\!\left(\dfrac{\alpha}{1-\alpha}\right)$ |
| `rotation_activation` | `normalize` | $q = \dfrac{q_{raw}}{\lVert q_{raw}\rVert_2}$ |

---

## 2. `modify_functions(self)` (dòng 48–52)

Đổi activation của opacity từ sigmoid sang `abs` (giá trị tuyệt đối) — một biến thể
không bị bão hòa như sigmoid:

$$
\alpha = |o_{raw}|
$$

và áp dụng ngay để giữ nguyên opacity hiện có:
$o_{raw}^{new} = \text{abs}^{-1}(\alpha_{old}) = \alpha_{old}$ (identity_gate).

---

## 3. `__init__(self, sh_degree, optimizer_type)` (dòng 54–92)

Không có công thức toán — chỉ khởi tạo tensor rỗng cho các thuộc tính: vị trí `_xyz`,
hệ số cầu điều hòa (`_features_dc`, `_features_rest`), `_scaling`, `_rotation`,
`_opacity`, các bộ tích lũy thống kê (gradient, η đa góc nhìn...), rồi gọi `setup_functions`.

---

## 4. `capture(self, optimizer_type)` (dòng 94–127)

Đóng gói toàn bộ trạng thái model thành tuple để lưu checkpoint. Không có công thức.

## 5. `restore(self, model_args, training_args)` (dòng 129–149)

Ngược lại với `capture`: nạp lại trạng thái, gọi `training_setup` để tái tạo optimizer.

---

## 6. `get_scaling` (property, dòng 151–153)

$$
S = \exp(S_{raw})
$$

## 7. `get_scaling_with_3D_filter` (property, dòng 155–161)

Kết hợp scale Gaussian với bộ lọc 3D chống alias (ý tưởng từ Mip-Splatting):

$$
S_{filtered} = \sqrt{S^2 + F_{3D}^2}
$$

với `S = get_scaling`, `F_3D = self.filter_3D`.

## 8. `get_rotation` (property, dòng 163–165)

$$
q = \dfrac{q_{raw}}{\lVert q_{raw} \rVert_2}
$$

## 9. `get_xyz` (property, dòng 167–169)

Trả về trực tiếp `_xyz` (không activation): $x = x_{raw}$.

## 10. `get_features` (property, dòng 171–175)

Ghép hệ số SH bậc 0 (DC) và các bậc cao (rest):

$$
F = [F_{dc} \,\Vert\, F_{rest}]
$$

## 11. `get_features_dc` / 12. `get_features_rest` (dòng 177–183)

Trả về trực tiếp từng phần, không biến đổi.

## 13. `get_opacity` (property, dòng 185–187)

$$
\alpha = \sigma(o_{raw}) = \dfrac{1}{1+e^{-o_{raw}}}
$$

## 14. `get_opacity_with_3D_filter` (property, dòng 189–201)

Điều chỉnh opacity khi tích phân Gaussian 3D bị "phình to" bởi bộ lọc chống alias,
sao cho tích phân năng lượng (khối lượng) Gaussian không đổi. Với
$\det_1 = \prod_i s_i^2$ (scale gốc) và $\det_2=\prod_i (s_i^2+f_i^2)$ (scale sau lọc):

$$
\text{coef} = \sqrt{\dfrac{\det \Sigma}{\det \Sigma_{filtered}}}
            = \sqrt{\dfrac{\prod_i s_i^2}{\prod_i (s_i^2+f_i^2)}}
$$

$$
\alpha_{filtered} = \alpha \cdot \text{coef}
$$

## 15. `get_covariance(self, scaling_modifier=1)` (dòng 203–204)

$$
\Sigma = R\big(S_{mod}\,S\big)\big(S_{mod}\,S\big)^T R^T
$$

(gọi lại `build_covariance_from_scaling_rotation` ở mục 1).

---

## 16. `compute_3D_filter(self, cameras)` (dòng 206–255)

Tính bán kính lọc 3D chống aliasing cho mỗi điểm, dựa trên khoảng cách nhỏ nhất
tới các camera nhìn thấy điểm đó (ý tưởng Mip-Splatting).

Biến đổi điểm sang hệ camera (R được lưu transpose sẵn nên dùng trực tiếp):

$$
\mathbf{x}_{cam} = R^T\mathbf{x}_{world} + T
$$

Chiếu phối cảnh lên màn hình (pinhole camera):

$$
u = \dfrac{x_{cam}}{z_{cam}} f_x + \dfrac{W}{2}, \qquad
v = \dfrac{y_{cam}}{z_{cam}} f_y + \dfrac{H}{2}
$$

Một điểm hợp lệ nếu $z_{cam} > 0.2$ và nằm trong vùng mở rộng màn hình
$[-0.15W,\,1.15W]\times[-0.15H,\,1.15H]$. Với mỗi điểm, lấy khoảng cách Z nhỏ nhất
qua mọi camera hợp lệ:

$$
d_i = \min_{\text{cam hợp lệ}} z_{cam,i}
$$

Bán kính lọc cuối cùng (xấp xỉ góc nhìn pixel tối thiểu để không bị alias, hệ số
$\sqrt{0.2}$ lấy theo kinh nghiệm bài báo Mip-Splatting):

$$
F_{3D} = \dfrac{d}{f_{max}} \sqrt{0.2}
$$

với $f_{max}$ là độ dài tiêu cự lớn nhất trong các camera.

---

## 17. `oneupSHdegree(self)` (dòng 257–259)

$$
\ell_{active} \leftarrow \min(\ell_{active}+1,\ \ell_{max})
$$

---

## 18. `create_from_pcd(self, pcd, spatial_lr_scale)` (dòng 261–284)

Khởi tạo tham số Gaussian từ point cloud SfM ban đầu.

- Màu RGB → hệ số SH bậc 0: $F_{dc} = \text{RGB2SH}(c) = \dfrac{c - 0.5}{C_0}$ (với $C_0$
  là hằng số chuẩn hóa SH bậc 0, $C_0 = 0.28209...$).
- Khoảng cách bình phương trung bình tới 3 láng giềng gần nhất (`distCUDA2`), dùng làm
  scale khởi tạo đẳng hướng:

$$
s_{raw} = \ln\!\big(\sqrt{d^2}\big) \quad\text{(lặp lại cho 3 trục)}
$$

- Quaternion khởi tạo là đơn vị: $q = (1,0,0,0)$.
- Opacity khởi tạo $\alpha_0 = 0.1$:

$$
o_{raw} = \sigma^{-1}(0.1) = \ln\!\left(\dfrac{0.1}{0.9}\right)
$$

---

## 19. `training_setup(self, training_args)` (dòng 286–339)

Không công thức toán phức tạp — khởi tạo bộ tích lũy gradient bằng 0, tạo optimizer
Adam (hoặc SparseGaussianAdam) với các learning rate khác nhau cho từng nhóm tham số,
và các bộ lịch học suy giảm theo hàm mũ (`get_expon_lr_func`):

$$
\text{lr}(t) = \text{lr}_{init} \cdot \left(\dfrac{\text{lr}_{final}}{\text{lr}_{init}}\right)^{\,t/t_{max}}
\cdot \text{delay\_rate}(t)
$$

(công thức đầy đủ nằm trong `utils/general_utils.py`).

---

## 20. `update_learning_rate(self, iteration)` (dòng 341–355)

Gọi các scheduler ở trên tại bước `iteration`:

$$
\text{lr}_{xyz}(t),\quad \text{lr}_{scaling}(t),\quad \text{lr}_{rotation}(t)
$$

và gán vào các param group tương ứng của optimizer.

---

## 21. `optimizer_step(self, iteration)` (dòng 357–377)

Chiến lược cập nhật thưa (sparse update) theo lịch:

$$
\text{step nếu}\quad
\begin{cases}
\text{mọi iteration}, & t \le 15000\\
t \bmod 32 = 0, & 15000 < t \le 20000\\
t \bmod 64 = 0, & t > 20000
\end{cases}
$$

Không có công thức tối ưu riêng — dùng rule Adam chuẩn:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
$$
\theta_t = \theta_{t-1} - \text{lr}\cdot\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

---

## 22. `construct_list_of_attributes(self)` (dòng 379–392)

Chỉ tạo danh sách tên thuộc tính PLY — không công thức.

## 23. `save_ply(self, path)` (dòng 394–413)

Ghi tham số thô (chưa qua activation) ra file `.ply`. Không công thức.

## 24. `reset_opacity(self, decay_factor=0.1)` (dòng 415–433)

Giảm opacity hiệu dụng (đã tính theo bộ lọc 3D) theo hệ số suy giảm, rồi quy đổi
ngược về giá trị thô:

$$
\alpha' = \alpha_{filtered} \cdot \text{decay\_factor}
$$

$$
\alpha'_{unfiltered} = \dfrac{\alpha'}{\text{coef}}, \qquad
\text{coef} = \sqrt{\dfrac{\det_1}{\det_2}} \ \text{(như mục 14)}
$$

$$
o_{raw}^{new} = \sigma^{-1}(\alpha'_{unfiltered})
$$

## 25. `load_ply(self, path)` (dòng 435–483)

Đọc ngược lại các tham số thô từ `.ply`. Không công thức toán.

---

## 26. `replace_tensor_to_optimizer(self, tensor, name)` (dòng 485–501)

Thay 1 tham số trong optimizer và reset trạng thái Adam (moment bậc 1 & 2) về 0:

$$
m = 0,\quad v = 0 \quad(\text{và } \hat v_{max}=0 \text{ nếu dùng amsgrad})
$$

## 27. `_prune_optimizer(self, mask)` (dòng 503–526)

Lọc (indexing bằng `mask`) cả tham số lẫn trạng thái Adam $m, v$ (và $\hat v_{max}$)
tương ứng với các điểm được giữ lại. Không công thức mới.

## 28. `prune_points(self, mask)` (dòng 528–564)

Xoá các Gaussian bị đánh dấu `mask=True` khỏi mọi tensor liên quan
(`_xyz`, features, scaling, rotation, opacity, bộ đếm, bộ tích lũy η...). Không công thức.

## 29. `cat_tensors_to_optimizer(self, tensors_dict)` (dòng 566–593)

Nối thêm tham số mới vào optimizer, với trạng thái Adam khởi tạo 0 cho phần mới:

$$
\theta_{new} = [\theta_{old} \,\Vert\, \theta_{append}], \qquad
m_{new} = [m_{old}\,\Vert\,0],\quad v_{new} = [v_{old}\,\Vert\,0]
$$

## 30. `densification_postfix(...)` (dòng 595–636)

Gộp các Gaussian mới (do split/clone) vào model và reset các bộ tích lũy thống kê
(gradient, η, denom...) về 0 cho toàn bộ tập điểm. Không công thức mới, chủ yếu là
ghép tensor (`torch.cat`).

---

## 31. `densify_and_split_structgs(self, metric_mask, max_eta_3ch, scale_power)` (dòng 638–831)

Thuật toán **split dị hướng theo tần số** (StructGS) — phần nhiều công thức nhất file.

Lý thuyết lấy mẫu (sampling theory): định nghĩa năng lượng tần số vi phạm Nyquist

$$
\eta = (\sigma \cdot \omega_{max})^2
$$

Để hết aliasing cần giảm $\sigma$ đi hệ số $k$ sao cho $\dfrac{\sigma}{k}\omega_{max}\le 1$, tức:

$$
\dfrac{\eta}{k^2} \le 1 \;\Longrightarrow\; k \ge \sqrt{\eta}
$$

Số lần chia theo từng trục (x,y,z), làm tròn lên và kẹp trong $[1, \dots]$:

$$
k_{axis} = \big\lceil \sqrt{\max(\eta_{axis},\,1)} \big\rceil
$$

(nếu không có `max_eta_3ch`, fallback kiểu 3DGS gốc: chỉ chia đôi trục scale lớn nhất, $k=2$.)

Tổng số Gaussian con sinh ra từ 1 Gaussian cha:

$$
N = k_x \cdot k_y \cdot k_z
$$

Scale mới của Gaussian con (chia theo $k$, có thể lũy thừa `scale_power`$=p$):

$$
s_{new} = \dfrac{s_{old}}{k^{p}}, \qquad s_{raw,new} = \ln(s_{new})
$$

Toạ độ lưới con (centered index), với $i \in \{0,\dots,k-1\}$:

$$
g_i = i - \dfrac{k-1}{2}
$$

Độ lệch tiêu chuẩn mới và khoảng cách giữa các Gaussian con (giả định phân bố đều
trên khoảng $[-\sqrt3\,\sigma, \sqrt3\,\sigma]$, nên nhân $\sqrt{12}$):

$$
\sigma_{new} = \dfrac{\sigma_{old}}{k}, \qquad
\text{separation} = \sigma_{new}\sqrt{12}
$$

Offset cục bộ (trong hệ local của Gaussian cha) cho Gaussian con tại chỉ số $(i_x,i_y,i_z)$:

$$
\Delta_{local} = \big(g_x,\,g_y,\,g_z\big)\odot\text{separation}
$$

Xoay offset sang hệ world bằng ma trận quay $R$ dựng từ quaternion, rồi cộng vào tâm cha:

$$
\Delta_{world} = R\,\Delta_{local}, \qquad
\mathbf{x}_{child} = \mathbf{x}_{parent} + \Delta_{world}
$$

Sau đó các Gaussian cha bị xoá (`prune_points`), chỉ còn các Gaussian con.

---

## 32. `expand_undersized_gs(self, tau_expand, max_eta_3ch)` (dòng 833–864)

Phóng to (ngược với split) các Gaussian quá nhỏ so với tần số cần biểu diễn
(dưới ngưỡng Nyquist), cập nhật **trực tiếp trong không gian log** để đạt chính xác
$\eta_{new}=1$:

Từ $\eta=(\sigma\omega)^2$, muốn $\eta_{new}=1$ thì:

$$
\sigma_{target} = \dfrac{\sigma_{old}}{\sqrt{\eta}}
$$

$$
\ln(\sigma_{target}) = \ln(\sigma_{old}) - \dfrac12\ln(\eta)
\quad\Longrightarrow\quad
\Delta(\log s) = -\dfrac12 \ln(\eta)
$$

Áp dụng: $s_{raw}^{new} = s_{raw}^{old} + \Delta(\log s)$, chỉ tại các trục có
$0 < \eta < \tau_{expand}$ (undersized).

---

## 33. `densify_and_split(self, grads, grad_threshold, scene_extent, N=2)` (dòng 866–891)

Thuật toán split chuẩn của 3DGS gốc (không dùng η).

Điều kiện chọn điểm để split: gradient view-space đủ lớn **và** scale đủ lớn:

$$
\lVert \nabla_{xy} \rVert \ge \tau_{grad}
\quad\text{và}\quad
\max_i(s_i) > p_{dense}\cdot \text{extent}
$$

Sinh $N$ Gaussian con bằng lấy mẫu Gaussian 3D quanh tâm cha trong hệ local rồi xoay
sang world:

$$
\boldsymbol{\epsilon} \sim \mathcal N(0,\; \text{diag}(s_x^2,s_y^2,s_z^2))
$$
$$
\mathbf{x}_{child} = R\,\boldsymbol{\epsilon} + \mathbf{x}_{parent}
$$

Scale mới được chia nhỏ (hệ số kinh nghiệm $0.8N$):

$$
s_{new} = \dfrac{s_{old}}{0.8\,N}
$$

Gaussian cha bị xoá sau khi tạo con (giống mục 31 nhưng không theo trục riêng lẻ).

---

## 34. `densify_and_clone(self, grads, grad_threshold, scene_extent)` (dòng 893–909)

Nhân bản (clone) Gaussian nhỏ nhưng có gradient lớn — không tách scale, chỉ copy
nguyên trạng:

$$
\lVert \nabla_{xy}\rVert \ge \tau_{grad}
\quad\text{và}\quad
\max_i(s_i) \le p_{dense}\cdot\text{extent}
$$

Không công thức biến đổi — các tham số con = tham số cha (copy).

---

## 35. `densify_and_prune(self, max_grad, max_grad_abs, min_opacity, extent, max_screen_size, radii)` (dòng 911–932)

Gradient trung bình tích lũy theo số lần nhìn thấy (denom):

$$
\bar g = \dfrac{\sum \nabla_{xy}}{\text{denom}}, \qquad
\bar g_{abs} = \dfrac{\sum \nabla_{z}}{\text{denom}}
$$

(NaN do chia 0 được gán về 0). Gọi `densify_and_clone` với $\bar g$ và
`densify_and_split` với $\bar g_{abs}$.

Điều kiện xoá (prune):

$$
\alpha < \alpha_{min}
\;\lor\; r_{2D} > r_{max}
\;\lor\; \max_i(s_i) > 0.1\cdot\text{extent}
$$

---

## 36. `densify_and_clone_structgs(self, metric_mask, filter)` (dòng 934–954)

Giống `densify_and_clone` nhưng điều kiện chọn điểm là mask tùy ý (kết hợp
importance score và điều kiện clone), và kế thừa `densify_count` của cha
(không cộng thêm 1).

---

## 37. `densify_and_prune_structgs(...)` (dòng 957–1052)

Pipeline densify/prune đầy đủ cho StructGS, kết hợp nhiều tiêu chí:

- Gradient chuẩn hoá: $\bar g = \dfrac{\sum\nabla_{xy}}{\text{denom}}$,
  $\bar g_{abs} = \dfrac{\sum\nabla_z}{\text{denom}}$
- Điều kiện split theo gradient-abs: $\lVert \bar g_{abs}\rVert \ge \tau_{abs}$
- Điều kiện clone theo gradient: $\lVert \bar g\rVert \ge \tau_{grad}$
- Điều kiện kích thước: clone nếu $\max_i(s_i) \le \text{dense}\cdot\text{extent}$,
  split nếu $\max_i(s_i) > \text{dense}\cdot\text{extent}$
- Mask cuối cùng là hợp/giao logic (OR/AND) giữa mask tùy chỉnh (`custom_split_mask`,
  `custom_prune_mask`), mask từ importance score, và điều kiện kích thước/gradient.
- Gọi `densify_and_clone_structgs` và `densify_and_split_structgs` (công thức η như mục 31).
- Prune theo cùng điều kiện ở mục 35, cộng thêm `full_prune_mask`.
- Nếu có `pruning_score`, lấy mẫu **có trọng số nghịch đảo điểm số** để chỉ xoá một
  phần ngân sách ($50\%$ số điểm định xoá):

$$
w_i = \dfrac{1}{10^{-6} + (1-\text{pruning\_score}_i)}, \qquad
\text{budget} = \lfloor 0.5 \cdot N_{to\_remove} \rfloor
$$

  lấy mẫu `multinomial(w, budget)` không hoàn lại.

- Cuối cùng giới hạn trần opacity về $0.8$:

$$
\alpha_{new} = \min(\alpha,\ 0.8), \qquad o_{raw}^{new} = \sigma^{-1}(\alpha_{new})
$$

---

## 38. `add_densification_stats(self, viewspace_point_tensor, update_filter)` (dòng 1054–1057)

Tích lũy độ lớn gradient 2D (vị trí màn hình) và "abs" (kênh thứ 3 trở đi, dùng cho
split theo StructGS) cho mỗi điểm được nhìn thấy:

$$
G_{xy} \mathrel{+}= \lVert \nabla_{(0:2)} \rVert_2, \qquad
G_{abs} \mathrel{+}= \lVert \nabla_{(2:)} \rVert_2, \qquad
\text{denom} \mathrel{+}=1
$$

---

## 39. `final_prune_structgs(self, min_opacity, pruning_score=None)` (dòng 1059–1067)

Prune cuối cùng dựa trên opacity thấp **hoặc** điểm số nhất quán đa góc nhìn cao
(gợi ý điểm không ổn định/outlier):

$$
\text{prune} = (\alpha < \alpha_{min}) \;\lor\; (\text{score} > 0.9)
$$

---

## Ghi chú chung

- "raw" nghĩa là giá trị lưu trực tiếp trong `nn.Parameter` (trước activation);
  giá trị thực tế dùng khi render luôn đi qua activation tương ứng (`exp`, `sigmoid`,
  `normalize`...).
- `η` (eta) là đại lượng đặc trưng cho "năng lượng tần số vi phạm Nyquist" được tính
  ở nơi khác trong codebase (không phải trong file này) và truyền vào qua tham số
  `max_eta_3ch` / accumulator `max_eta_3ch`, `accum_eta`...
- Các công thức Adam optimizer không được định nghĩa lại trong file này — chúng dùng
  trực tiếp `torch.optim.Adam` hoặc `SparseGaussianAdam` (import ngoài).
