[← Mục lục](00-muc-luc.md) · Chương 5/15

# Chương 5 — Ký hiệu & Nền tảng toán học chung

> Ký hiệu trong chương này được rút ra trực tiếp từ code (`scene/gaussian_model.py`, `gaussian_renderer/`,
> `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/`), dùng chung cho các chương 6–13.

Từ chương này trở đi, sách trình bày lại toàn bộ pipeline 3D Gaussian Splatting theo đúng thứ tự các khối trong sơ đồ, đối chiếu công thức trực tiếp với code trong repo (không lấy số liệu từ paper upstream). Mỗi chương (6–13) tương ứng một khối trong sơ đồ pipeline bên dưới, và có phần kiểm định số kèm theo (chạy trên cùng một "cảnh đồ chơi").

## 5.1 Sơ đồ pipeline & mục lục các chương tiếp theo

> Các chương sau đi theo **đúng thứ tự các khối trong sơ đồ pipeline** của 3D Gaussian Splatting, và ở mỗi khối chỉ ra
> phần nào là 3DGS gốc (giữ nguyên) và phần nào là **SADGS** (Structure-Aware
> Densification, Lyu et al., SIGGRAPH 2026) thay đổi. SADGS kế thừa nguyên vẹn cơ chế compact-box/AccuTile ở
> khối Projection (đổi CUDA extension thành `diff-gaussian-rasterization_structgs`), nhưng **thay thế toàn bộ Adaptive
> Density Control gốc** bằng densify/split/prune dựa trên so sánh screen-space extent với cấu trúc texture đa
> tỉ lệ (multiscale image structure) và một metric multi-view reconstruction consistency. Mọi công thức đối chiếu với
> code trong `SADGS/`, không lấy con số nào từ paper upstream. Chi tiết cơ chế densification: xem
> [Chương 17](17-sadgs-structure-aware-densification.md).

*[hình sơ đồ pipeline sẽ được cập nhật]*

```
SfM Points ──► Initialization ──► 3D Gaussians ──► Projection ──► Differentiable ──► Image
                                       ▲    ▲          ▲           Tile Rasterizer      │
                        Camera ────────┼────┼──────────┘                 ▲              │
                                       │    │                            │              │
                                       │    └──── gradient ◄─────────────┴── gradient ◄─┘
                                       │
                              Adaptive Density Control
```

Mũi tên đen = **Operation Flow** (chương 1 → 5). Mũi tên xanh = **Gradient Flow** (chương 6). Vòng quay về từ
**Adaptive Density Control** (chương 7).

## Mục lục

| Chương | Khối trong sơ đồ | File | SADGS thay đổi gì |
|---|---|---|---|
| 5 | Ký hiệu chung | (chương này) | — |
| 6 | SfM Points → Initialization | [06-initialization.md](06-initialization.md) | giữ nguyên |
| 7 | 3D Gaussians | [07-3d-gaussians.md](07-3d-gaussians.md) | giữ nguyên (kể cả 3D filter Mip-Splatting) |
| 8 | Camera + Projection | [08-projection-compact-box.md](08-projection-compact-box.md) | **compact box, lọc tile theo ellipse** (giảm $K$) — kế thừa nguyên vẹn |
| 9 | Differentiable Tile Rasterizer | [09-differentiable-tile-rasterizer.md](09-differentiable-tile-rasterizer.md) | giữ nguyên toán; đầu vào nhỏ hơn |
| 10 | Image → Loss & Metrics | [10-anh-den-loss-metrics.md](10-anh-den-loss-metrics.md) | giữ nguyên loss; ghi chú metrics |
| 11 | Gradient Flow | [11-gradient-flow-backprop.md](11-gradient-flow-backprop.md) | **gradient trị tuyệt đối, Adam thưa, lr SH** — kế thừa nguyên vẹn |
| 12 | Adaptive Density Control | [12-adaptive-density-control.md](12-adaptive-density-control.md) | **thay thế bằng structure-aware densification**: `densify_and_split_structgs`, `densify_and_clone_structgs`, `densify_and_prune_structgs`, `final_prune_structgs` — so sánh screen-space extent với cấu trúc texture đa tỉ lệ, split dị hướng (anisotropic), pruning theo multi-view reconstruction consistency (giảm $N$) |
| 13 | Tổng hợp: mô hình chi phí | [13-tong-hop-chi-phi-sadgs.md](13-tong-hop-chi-phi-sadgs.md) | ba tỉ số nhân nhau |

## Kết luận trước khi đọc

SADGS **không đổi một dòng nào của toán render** — $G(x)$, $\Sigma=RSS^\top R^\top$, $\Sigma'=JW\Sigma W^\top J^\top$,
alpha-blend, backward qua blend, update rule của Adam, loss $\mathcal L=(1-\lambda)\mathcal L_1+\lambda\mathcal L_{\text{D-SSIM}}$
đều giữ nguyên, và cũng giữ nguyên cơ chế compact-box/AccuTile ở khối Projection. Cái bị thay là:

| Điều khiển | Khối | Hiệu ứng lên chi phí | Nguồn gốc |
|---|---|---|---|
| Cặp (tile, Gaussian) nào được đưa vào sort/blend | Projection | giảm $K$ | kế thừa nguyên vẹn từ 3DGS gốc |
| Gaussian nào được tồn tại (densify/split/prune structure-aware) | Adaptive Density Control | giảm $N$ | **SADGS thay đổi** |
| Khi nào gọi `optimizer.step()` | Gradient Flow | giảm số lần cập nhật tham số | kế thừa nguyên vẹn từ 3DGS gốc |

*(Xem sơ đồ pipeline gốc tại `../../pipe.png` — tức file `pipe.png` ở gốc repo `digital-twin-gs/`.)*

## 5.2 Ký hiệu dùng chung

| Ký hiệu | Ý nghĩa | Nơi xuất hiện trong code |
|---|---|---|
| $N$ | số Gaussian hiện có | `gaussians.get_xyz.shape[0]` |
| $i$ | chỉ số Gaussian, $i=1..N$ | |
| $v$ | chỉ số camera / góc nhìn | `viewpoint_cam` |
| $\theta_i=(\mu_i,\tilde q_i,\tilde s_i,\tilde\alpha_i,k_i)$ | 59 tham số học được của Gaussian $i$ | `_xyz, _rotation, _scaling, _opacity, _features_dc/_rest` |
| $\mu_i\in\mathbb R^3$ | tâm (world space) | `_xyz` |
| $q_i,\ R_i=R(q_i)$ | quaternion đơn vị và ma trận xoay | `get_rotation` |
| $s_i\in\mathbb R^3_{>0},\ S_i=\operatorname{diag}(s_i)$ | scale | `get_scaling` |
| $\alpha_i\in(0,1)$ | opacity (sau sigmoid); code gọi là $o$ | `get_opacity` |
| $k_{i,lm}\in\mathbb R^3$ | hệ số SH, $l=0..3$, 16 hệ số/kênh | `get_features` |
| $\Sigma_i$ | covariance 3D | `computeCov3D` |
| $\Sigma'_i$ | covariance 2D sau chiếu | `computeCov2D` |
| $M_i=\Sigma_i'^{-1}=\begin{pmatrix}A&B\B&C\end{pmatrix}$ | conic | `con_o.x, .y, .z` |
| $\mu'_i\in\mathbb R^2$ | tâm trên ảnh (pixel) | `points_xy_image` |
| $\Delta=x-\mu'_i$ | độ lệch pixel–tâm | `d` trong `renderCUDA` |
| $t_i$ | ngưỡng level-set của compact box | `auxiliary.h:312-314` |
| $\mathcal K_i,\ K_i=\lvert\mathcal K_i\rvert$ | tập tile Gaussian $i$ chạm, và số tile | `tiles_touched[i]` |
| $P=\sum_iK_i$ | tổng số cặp (tile, Gaussian) | `num_rendered` |
| $T_n$ | transmittance tích luỹ trước Gaussian thứ $n$ | `T` |
| $C(x)$ | màu pixel $x$ | `out_color` |
| $\mathcal L$ | loss huấn luyện | `loss` |
| $\text{extent}$ | bán kính cảnh ước lượng từ camera | `scene.cameras_extent` |
| $\text{Importance}_i,\ \text{Pruning}_i$ | hai điểm số multi-view của SADGS (structure-aware densification / multi-view reconstruction consistency) | `importance_score, pruning_score` |
| $\sigma(\cdot),\ \sigma^{-1}(\cdot)$ | sigmoid và nghịch đảo | `torch.sigmoid, inverse_sigmoid` |

Hằng số cứng hay gặp (đều nằm trong kernel hoặc `arguments/__init__.py`):

| Hằng | Giá trị | Ý nghĩa |
|---|---|---|
| tile | $16\times16$ px | `BLOCK_X, BLOCK_Y` |
| low-pass | $0.3$ | cộng vào đường chéo $\Sigma'$ |
| clamp Jacobian | $1.3\tan(\text{fov}/2)$ | |
| ngưỡng alpha | $1/255$ | bỏ splat mờ hơn 1 bước 8-bit |
| dừng sớm | $10^{-4}$ | dừng blend khi $T<10^{-4}$ |
| `mult` | $0.7$ | hệ số compact box (`arguments/__init__.py`, dòng định nghĩa `self.mult`) |
| $\tau_{\text{grad}}$ | $2\times10^{-4}$ | `densify_grad_threshold` (đường ADC kiểu 3DGS gốc, vẫn còn trong code nhưng không phải nhánh chính) |
| $\tau^{\text{abs}}_{\text{grad}}$ | $2\times10^{-4}$ | `grad_abs_thresh` |
| $\tau_{\text{loss}}$ | $0.02$ | `loss_thresh` |
| $\delta$ | $0.001$ | `dense` / `percent_dense` — ngưỡng clone/split theo scale |
| $\lambda$ | $0.2$ | `lambda_dssim` |

Nhánh **structure-aware densification** thật sự của SADGS (chương 12, 17) không gate bằng $\tau_{\text{grad}},\tau_{\text{loss}}$ ở trên mà bằng một bộ ngưỡng khác, cũng nằm trong `arguments/__init__.py`: `freq_grad_threshold` ($2\times10^{-5}$), `importance_score_threshold` ($0.5$), `min_contribution_threshold` ($0.1$), `importance_error_threshold` ($0.06$), `tau_expand` ($1.0$), `split_ratio_threshold`/`prune_ratio_threshold` ($0.8$). Bảng trên chỉ liệt kê các hằng số dùng chung ở chương 6–8 (initialization, 3D Gaussians, projection); các hằng số densification thuộc phạm vi chương 12.

## Bài tập (Exercise)

**Bài tập 5.1.** Giải thích ý nghĩa của $\mathcal K_i$, $K_i=\lvert\mathcal K_i\rvert$ và $P=\sum_iK_i$. Vì sao $P$ — chứ không phải $N$ riêng lẻ — là đại lượng thực sự quyết định số phép tính trong bước sort/blend của rasterizer?

**Bài tập 5.2.** Từ bảng hằng số, ngưỡng alpha là $1/255$. Viết ngưỡng này dưới dạng số thập phân (4 chữ số có nghĩa) và giải thích tại sao nó gắn với "1 bước 8-bit" thay vì một hằng số tuỳ ý như $10^{-3}$.

**Bài tập 5.3.** Dựa vào bảng ký hiệu, $M_i=\Sigma_i'^{-1}=\begin{pmatrix}A&B\\B&C\end{pmatrix}$ là conic ứng với $\Sigma'_i$ (covariance 2D sau chiếu, lưu trong code ở `con_o.x, .y, .z`). Nêu ý nghĩa hình học của $A$, $B$, $C$ (liên hệ với trục và độ nghiêng của ellipse mà Gaussian chiếu lên ảnh), và giải thích vì sao rasterizer dùng nghịch đảo $\Sigma_i'^{-1}$ thay vì $\Sigma_i'$ trực tiếp khi tính $\Delta^\top M_i\Delta$.

**Bài tập 5.4.** So sánh vai trò của hai điểm số $\text{Importance}_i$ và $\text{Pruning}_i$ trong bảng ký hiệu (chỉ dựa trên tên và vị trí xuất hiện trong code — `importance_score, pruning_score` — chưa cần công thức chi tiết ở chương 7): vì sao SADGS cần *hai* điểm số multi-view riêng biệt thay vì một điểm số duy nhất để vừa densify vừa prune?

**Bài tập 5.5.** Trace code: liệt kê tất cả các ký hiệu trong bảng 5.2 mà nơi xuất hiện trong code nằm ở `gaussian_model.py` (dưới dạng `get_*` hoặc `_*`), và tách riêng nhóm ký hiệu chỉ tồn tại trong kernel CUDA (`renderCUDA`, `auxiliary.h`). Từ đó giải thích ranh giới giữa phần "Python/PyTorch" và phần "CUDA kernel" trong pipeline.

---

[← Chương 4](04-luu-render-cham-diem-phan-3.md) | [Mục lục](00-muc-luc.md) | [Chương 6 →](06-initialization.md)
