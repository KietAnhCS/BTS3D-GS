## 8. Cấu hình thật đã dùng để train scene HCM0539 (`cfg_args`)

Nguồn duy nhất của mục này: `output/26sepoutput/sadgs_models/HCM0539/cfg_args` — một dòng `Namespace(...)` do chính lần chạy train thật ghi ra, đối chiếu ý nghĩa tham số SADGS với `DOCS/BOOK/17-sadgs-structure-aware-densification.md`, `DOCS/BOOK/18-sadgs-co-che-chuyen-sau/09-sieu-tham-so-va-rasterizer.md` và `DOCS/BOOK/12-adaptive-density-control.md` (bảng tổng kết tham số SADGS §12).

### 8.1 Bảng toàn bộ tham số thật

| Tham số | Giá trị thật (HCM0539) | Ghi chú ý nghĩa (chỉ ghi cho tham số SADGS đặc thù) |
|---|---|---|
| `sh_degree` | `3` | |
| `source_path` | `/content/drive/MyDrive/VAI_NVS_DATA_ROUND2/HCM0539/train` | |
| `model_path` | `/content/output/HCM0539` | |
| `images` | `'images'` | |
| `resolution` | `2` | |
| `white_background` | `False` | |
| `data_device` | `'cuda'` | |
| `eval` | `True` | |
| `iterations` | `30000` | |
| `opacity_lr` | `0.05` | |
| `scaling_lr` | `0.01` | |
| `rotation_lr` | `0.002` | |
| `position_lr_init` | `0.00016` | |
| `position_lr_final` | `1.6e-06` | |
| `position_lr_delay_mult` | `0.01` | |
| `position_lr_max_steps` | `30000` | |
| `feature_lr` | `0.0025` | Khai báo trong `OptimizationParams` nhưng **không có tham chiếu** ở bất kỳ đâu ngoài `arguments/` — tham số chết (xác nhận ở 18.9.1 Nhóm E). |
| `shfeature_lr` | `0.005` | Tham số chết, cùng nhóm với `feature_lr` ở trên. |
| `percent_dense` | `0.001` | Tham số 3DGS gốc (ranh giới clone/split theo scale so với extent) — SADGS có bản riêng là `dense` (xem dưới). |
| `lambda_dssim` | `0.25` | |
| `densification_interval` | `500` | Chu kỳ gọi vòng densify+prune hợp nhất (`densify_and_prune_structgs`). |
| `opacity_reset_interval` | `6000` | Mốc reset opacity định kỳ, cũng là mốc bật `size_threshold=20` trong pruning. |
| `opacity_reset_decay` | `0.1` | SADGS thay công thức 3DGS gốc ($\alpha\leftarrow\min(\alpha,0.01)$) bằng $\alpha\leftarrow 0.1\,\alpha$. |
| `densify_from_iter` | `500` | Cửa sổ densify **và** cửa sổ bắt đầu cập nhật thống kê $\eta$. |
| `densify_until_iter` | `15000` | Mốc dừng densify/cập nhật $\eta$. |
| `densify_grad_threshold` | `0.0002` | Chỉ thực sự dùng khi `warmup_densification=True`; ở run này `warmup_densification=False` nên tham số này **không tác động**. |
| `densify_grad_abs_threshold` | `0.0004` | Cùng điều kiện như trên — inert vì `warmup_densification=False`. |
| `prune_until_iter` | `30000` | Bị khai báo/ghi đè hai lần trong `arguments/__init__.py` (25000 rồi 30000) và **không được dùng ở đâu trong code** — tham số chết. |
| `min_weight` | `0.7` | Khai báo nhưng không có tham chiếu ngoài `arguments/` — tham số chết (khác với `min_opacity=0.1` hard-code trực tiếp trong `train.py:364`). |
| `prune_from_iter` | `6000` | Tham số chết — không được đọc ở bất kỳ đâu ngoài `arguments/`. |
| `prune_interval` | `3000` | Tham số chết, cùng nhóm với `prune_from_iter`. |
| `densify_prune_ratio` | `0.45` | Tham số SADGS đặc thù nhưng **không có tham chiếu nào** trong `train.py`/`gaussian_model.py` — vestigial, không ảnh hưởng gì tới việc chạy thật dù mang giá trị "có vẻ hợp lý" 0.45. |
| `after_densify_prune_ratio` | `0.01` | Cùng tình trạng: khai báo trong `OptimizationParams` nhưng không được đọc ở đâu — tham số chết. |
| `loss_thresh` | `0.07` | Tham số chết. |
| `grad_abs_thresh` | `0.0012` | Ngưỡng gradient tuyệt đối cho nhánh split kiểu 3DGS gốc kế thừa; **giá trị tài liệu ghi mặc định là 0.0002** — run thật đặt cao gấp 6 lần. |
| `highfeature_lr` | `0.02` | LR nhóm optimizer "hybrid" cho tham số tần số cao; tài liệu ghi mặc định `0.005` — run thật cao gấp 4 lần. |
| `lowfeature_lr` | `0.0025` | Khớp giá trị mặc định tài liệu ghi (`0.0025`). |
| `grad_thresh` | `0.0002` | Ngưỡng gradient dùng thật trong `densify_and_prune_structgs` — khớp mặc định tài liệu. |
| `dense` | `0.001` | Ranh giới clone/split của SADGS: `max σ ≶ dense·extent`; khớp mặc định tài liệu. |
| `mult` | `0.5` | Hệ số thu nhỏ hộp bao compact của rasterizer riêng SADGS (điều khiển số tile mỗi splat, ảnh hưởng tốc độ render/VRAM, gần như không ảnh hưởng chất lượng theo tài liệu); mặc định tài liệu ghi `0.7` — run thật đặt thấp hơn (hộp bao nhỏ hơn, nhanh hơn). |
| `lambda_l2` | `2.0` | Khớp mặc định tài liệu. |
| `lambda_tone` | `0.0` | Tắt (loss tone-curve không dùng), khớp mặc định. |
| `lambda_freq` | `0.0` | Tắt (loss tần số không dùng dù có tính `frequency_loss_simple`), khớp mặc định. |
| `st_levels` | `4` | Số mức kim tự tháp DoG/Laplacian trong structure tensor đa tỉ lệ; khớp mặc định. |
| `st_mode` | `'v1'` | Chọn biến thể "rẻ hơn nhưng xấp xỉ" (`v1`) thay vì "đúng hơn" (`v2`, theo chính docstring tác giả) cho multiscale structure tensor; khớp mặc định `v1`. |
| `freq_grad_threshold` | `2e-05` | Chỉ tích luỹ thống kê $\eta$ khi $\lVert\nabla_{uv}\rVert$ vượt ngưỡng này; khớp mặc định. |
| `importance_score_threshold` | `0.5` | Ngưỡng `metric_mask` trong `densify_and_prune_structgs`: `importance_score > 0.5`. Đại lượng truyền vào thực chất là `accum_view_count` (số nguyên đếm view đã quan sát) chứ không phải điểm lỗi màu — nên ngưỡng 0.5 tương đương `accum_view_count ≥ 1`, cực kỳ lỏng. Khớp giá trị mặc định tài liệu, đang **thật sự chạy**. |
| `min_contribution_threshold` | `0.1` | Tham số SADGS đặc thù nhưng **không được tham chiếu ở bất kỳ đâu khác trong `SADGS/`** — tham số chết, còn sót lại từ một khung điểm số (scoring) thử nghiệm trước đó. |
| `importance_error_threshold` | `0.06` | Cùng tình trạng: khai báo nhưng không tham chiếu ở đâu — tham số chết. |
| `random_background` | `False` | |
| `optimizer_type` | `'hybrid'` | Chia tham số thành 2 optimizer riêng (tần số cao/thấp) — cơ chế tối ưu hoá đặc thù SADGS, khớp mặc định. |
| `sample_bbox_faces` | `False` | Không tìm thấy giải thích chi tiết trong hai file đã đọc lướt — không xác định được vai trò cụ thể trong lần đọc này. |
| `warmup_densification` | `False` | Cờ bật/tắt nhánh densify "warmup" (gọi thẳng `densify_and_prune` gốc 3DGS thay vì `densify_and_prune_structgs`); `False` = TẮT, khớp mặc định — nghĩa là toàn bộ quá trình densify dùng nhánh SADGS chính (anisotropic split theo $\eta$), không dùng nhánh 3DGS gốc. |
| `camera_sampling` | `'random'` | Không tìm thấy giải thích chi tiết trong hai file đã đọc lướt. |
| `compute_3d_filter` | `False` | Liên quan bộ lọc 3D chống alias (mục 18.7, không đọc sâu trong lần này) — tắt. |
| `tau_expand` | `1.0` | Ngưỡng Nyquist cho cơ chế "nở" Gaussian dưới cỡ (`expand_undersized_gs`, $\Delta\log\sigma=-\tfrac12\log\eta$); khớp mặc định, nhưng lời gọi hàm này **bị comment-out** trong `train.py:356-359` — cơ chế không chạy trong lịch train mặc định, nên giá trị này **không có tác dụng thực tế** ở lần chạy này. |
| `adaptive_clone` | `False` | Cờ dự định bật `expand_undersized_gs` trước khi clone; vì lời gọi expand đã bị comment nên cờ này hiện không đổi hành vi gì dù đặt giá trị nào — khớp mặc định. |
| `expansion_speed` | `0.1` | Khai báo nhưng không được dùng trong code hiện tại của `expand_undersized_gs` (hàm áp công thức phân tích trực tiếp, không có "tốc độ") — tham số chết. |
| `ks_scale_power` | `1.0` | Số mũ $p$ trong công thức co scale Gaussian con khi split: $\sigma'=\sigma/k^{p}$. Tài liệu kết luận rõ "giữ `ks_scale_power=1.0`" vì đây là giá trị duy nhất khiến công thức scale và công thức vị trí con mô tả cùng một hình học — run thật đúng bằng giá trị khuyến nghị này. |
| `sample_far_plane` | `False` | Cùng nhóm tham số chết `far_plane_*` bên dưới. |
| `far_plane_dist` | `10.0` | Tham số chết — không tham chiếu ngoài `arguments/`. |
| `far_plane_res` | `32` | Tham số chết, cùng nhóm. |
| `densification_window_width` | `200` | Tham số chết — không tham chiếu ngoài `arguments/`. |
| `freq_opacity_threshold` | `0.05` | Bỏ qua Gaussian gần trong suốt khi tích luỹ thống kê $\eta$; khớp mặc định. |
| `freq_transmittance_threshold` | `0.0` | Bỏ qua Gaussian bị che khi tích luỹ $\eta$; giá trị `0.0` nghĩa là **không lọc theo transmittance**, khớp mặc định. |
| `batch_size` | `1` | |
| `split_ratio_threshold` | `0.8` | Ngưỡng đồng thuận đa view để **split**: split khi tỉ lệ view thấy $\eta$ cao trên tổng view $> 0.8$. Khớp mặc định; theo bảng nhiệt độ nhạy trong tài liệu, đây là một trong hai tham số ảnh hưởng trực tiếp mạnh nhất tới ngân sách Gaussian. |
| `prune_ratio_threshold` | `0.8` | Ngưỡng đồng thuận đa view để **prune**: prune khi tỉ lệ view thấy $\eta$ thấp trên tổng view $> 0.8$. Khớp mặc định — cùng mức 0.8 như split, nhưng bản chất bất đối xứng (xem mục 8.2). |
| `eta_compute_mode` | `'wavelength'` | Cách tính $\eta$ = so trục chiếu Gaussian với bước sóng texture cục bộ, diễn giải trị riêng lớn nhất của structure tensor như bước sóng nhỏ nhất còn phân biệt được ($w_{\min}=1/(\sqrt{\lambda_1}+10^{-5})$); mặc định, lựa chọn khác là `'projection'`. |
| `clone_target_eta` | `1.0` | Tham số SADGS đặc thù nhưng **không được đọc ở bất kỳ đâu trong `SADGS/`** — tham số chết. |
| `scale_rotation_scheduler` | `False` | Khớp mặc định. |
| `max_clones_per_axis` | `8` | Tham số SADGS đặc thù dự định làm **trần** số lát cắt mỗi trục khi split ($k_a\le 8$), nhưng code thật của `densify_and_split_structgs` chỉ có `clamp(min=1)` — **không có trần trên nào được áp dụng**, và bản thân tham số này **không được đọc ở đâu trong code**. Giá trị `8` hoàn toàn không có tác dụng ràng buộc thực tế. |
| `adam_eps_order` | `8` | |
| `separate_sh` | `True` | |
| `convert_SHs_python` | `False` | |
| `compute_cov3D_python` | `False` | |
| `debug` | `False` | |
| `antialiasing` | `False` | |

### 8.2 Nhận định

**Tham số nào lệch khỏi mặc định thường thấy trong tài liệu.** Trong số các tham số SADGS đặc thù còn thật sự chạy (không phải tham số chết), bốn tham số lệch rõ so với giá trị mặc định mà tài liệu (18.9.1) ghi lại: `densification_interval=500` (mặc định tài liệu ghi `100` — thưa hơn 5 lần), `opacity_reset_interval=6000` (mặc định `3000` — gấp đôi), `grad_abs_thresh=0.0012` (mặc định `0.0002` — gấp 6 lần), `highfeature_lr=0.02` (mặc định `0.005` — gấp 4 lần), và `mult=0.5` (mặc định `0.7`, hộp bao compact nhỏ hơn — ảnh hưởng tốc độ/VRAM nhiều hơn chất lượng theo tài liệu). Ngược lại, các tham số cấu trúc cốt lõi của cơ chế SADGS — `st_levels=4`, `st_mode='v1'`, `dense=0.001`, `grad_thresh=0.0002`, `freq_grad_threshold=2e-5`, `importance_score_threshold=0.5`, `ks_scale_power=1.0`, `split_ratio_threshold=0.8`, `prune_ratio_threshold=0.8`, `eta_compute_mode='wavelength'`, `tau_expand=1.0` — đều giữ đúng giá trị mặc định. Với các tham số `sample_bbox_faces`, `camera_sampling`, `compute_3d_filter`: không tìm thấy giá trị mặc định để so sánh trong hai file đã đọc lướt.

**Tham số nào ảnh hưởng lớn nhất tới việc số Gaussian cuối cùng lên tới ~2.97 triệu.** Đây là điểm cần nói thẳng: `densify_prune_ratio=0.45` và `after_densify_prune_ratio=0.01` **không giải thích được** con số 2.97 triệu, vì cả hai đã được xác nhận (12-adaptive-density-control.md, bảng tổng kết SADGS) là **không có tham chiếu nào** trong `train.py`/`gaussian_model.py` — dù giá trị 0.45/0.01 "trông" như một tỉ lệ prune hợp lý mỗi vòng, code thật không bao giờ đọc chúng, nên chúng có tác dụng đúng bằng 0 tới số Gaussian cuối. Tương tự, `min_contribution_threshold=0.1` cũng là tham số chết, không góp phần lọc bớt Gaussian nào. `importance_score_threshold=0.5` là tham số **duy nhất trong nhóm này thật sự chạy**, nhưng vì đại lượng nó so sánh (`accum_view_count`, một số nguyên đếm view) chứ không phải sai số render, ngưỡng 0.5 trên thực tế chỉ tương đương "đã được nhìn thấy ít nhất 1 lần" — một điều kiện gần như luôn đúng với bất kỳ Gaussian nào vừa được tạo, tức gần như không lọc được gì. Nói cách khác: cơ chế lẽ ra phải đóng vai trò "phanh" (prune theo importance/contribution/error) đã **mất tác dụng gần như hoàn toàn**, do một phần bị chết code (`densify_prune_ratio`, `after_densify_prune_ratio`, `min_contribution_threshold`, `importance_error_threshold`), một phần vì ngưỡng còn hoạt động (`importance_score_threshold`) quá lỏng so với ý nghĩa tên gọi. Trong khi đó, nhánh sinh Gaussian mới không hề bị giới hạn tương xứng: `max_clones_per_axis=8` — tham số lẽ ra là trần số lát cắt mỗi trục khi split — cũng là tham số chết, và code thật chỉ `clamp(min=1)` chứ không có trần trên nào, nghĩa là số con mỗi lần split ($k_x k_y k_z$) có thể lớn tuỳ ý theo $\eta$ đo được, không bị chặn. Cộng thêm `split_ratio_threshold=prune_ratio_threshold=0.8` (đối xứng về ngưỡng nhưng không đối xứng về hệ quả, vì tách/split không có trần còn prune bị vô hiệu hoá một phần) và `warmup_densification=False` (luôn dùng nhánh SADGS chính chứ không phải nhánh 3DGS gốc dè dặt hơn), kết quả hợp lý là: cơ chế split anisotropic không giới hạn trên kết hợp với cơ chế prune gần như bất hoạt (do tham số chết + ngưỡng quá lỏng) là tổ hợp nguyên nhân khả dĩ nhất khiến ngân sách Gaussian phình lên tới ~2.97 triệu, chứ không phải bốn tham số `densify_prune_ratio`, `after_densify_prune_ratio`, `importance_score_threshold`, `min_contribution_threshold` tự thân "được set cao" — ba trong bốn tham số đó thực ra không hề được thực thi.
