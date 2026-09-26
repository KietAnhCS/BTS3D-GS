# Script thuyết trình 45 phút — BOT 09 (SADGS chuyên sâu)

Bao phủ hai slide: `part45_sadgsx_13.tex` và `part45_sadgsx_14.tex`.
Chương sách tương ứng: `DOCS/BOOK/18-sadgs-co-che-chuyen-sau/09-sieu-tham-so-va-rasterizer.md`.

---

## Slide: Siêu tham số & lịch trình huấn luyện của SADGS
*(part45_sadgsx_13.tex)*

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này trả lời câu hỏi rất thực tế: muốn chạy SADGS thì phải vặn những nút nào. SADGS kế thừa toàn bộ tham số của 3DGS rồi mở rộng lên hơn năm mươi cờ, nhưng chỉ khoảng một nửa thực sự được nối vào code. Bảng bên trái liệt kê những cờ quan trọng nhất kèm giá trị mặc định thật, lấy từ file `arguments/__init__.py`: `mult` bằng 0.7, `eta_compute_mode` là "wavelength", hai ngưỡng đồng thuận đa view `split_ratio_threshold` và `prune_ratio_threshold` cùng bằng 0.8, `dense` bằng 0.001 — tức nhỏ hơn `percent_dense` của 3DGS gốc mười lần, `opacity_reset_decay` bằng 0.1, và optimizer mặc định là "hybrid". Nhưng điểm em muốn nhấn mạnh nhất nằm ngay dưới bảng: hai ngưỡng quyết định nhất của cả phương pháp — tau-high bằng 1.0 và tau-low bằng 0.1 — lại **không** nằm trong file tham số mà bị hard-code trong `freq_utils.py` dòng 395 đến 396. Muốn đổi chúng phải sửa mã nguồn, không có cờ dòng lệnh nào cả. Hình bên phải là sơ đồ Gantt dựng từ đúng các điều kiện `if` trong `train.py`: bậc SH được nâng mỗi iteration nên chạm bậc ba ngay ở iteration thứ ba; thống kê tần số eta cập nhật mỗi mười iteration và chỉ chạy đến mốc mười lăm nghìn; densify mỗi trăm iteration rồi lập tức xoá sạch bộ đếm eta — nghĩa là cái gọi là "đồng thuận đa view" thực chất chỉ dựa trên khoảng mười quan sát; reset opacity mỗi ba nghìn iteration bằng cách nhân alpha với 0.1; prune thô tại bốn nghìn và tám nghìn; còn mười lăm nghìn iteration cuối thì chỉ tinh chỉnh tham số chứ không thêm bớt Gaussian nữa. Cuối cùng, xin lưu ý ba nhánh đã bị comment-out trong code: `expand_undersized_gs` khiến cờ `tau_expand` hiện vô hiệu, prune trong nhánh warmup, và `final_prune_structgs` ở cuối huấn luyện.

**Chuyển tiếp:** Một trong những cờ vừa nêu — `mult` — không chạy trong Python mà được truyền thẳng xuống CUDA, nên slide tiếp theo ta sẽ mở nắp capô xem rasterizer riêng của SADGS đã sửa những gì.

### Nếu bị hỏi

**Hỏi: Cấu hình mặc định trong `arguments` có phải là cấu hình dùng để báo cáo kết quả không?**
Không ạ. `run_train.sh` mới là cấu hình thực nghiệm thật, và nó khác khá nhiều: số iteration chỉ 3000 cho Mip-NeRF 360 và Deep Blending, 7000 cho Tanks and Temples, chứ không phải 30000; `mult` dùng 0.25 ở ba trên bốn cấu hình; `freq_transmittance_threshold` đặt 0.8 thay vì 0.0; và cả bốn cấu hình đều bật `--warmup_densification` dù mặc định trong code là `False`.

**Hỏi: Những cờ nào chỉnh cũng vô ích?**
Em đã quét toàn bộ file `.py` ngoài thư mục `arguments` và tìm được mười tám cờ có đúng không lần sử dụng: `clone_target_eta`, `lambda_freq`, `lambda_tone`, `min_weight`, `prune_from_iter`, `prune_interval`, `densify_prune_ratio`, `after_densify_prune_ratio`, `loss_thresh`, `sample_far_plane`, `far_plane_dist`, `far_plane_res`, `densification_window_width`, `max_clones_per_axis`, `expansion_speed`, `adaptive_clone`, `feature_lr`, `shfeature_lr`. Ngoài ra `prune_until_iter` bị khai báo trùng hai lần ở dòng 96 và 100, giá trị 25000 bị ghi đè thành 30000, và cũng không dùng ở đâu.

**Hỏi: Tham số nào SADGS đổi so với 3DGS gốc mà đáng chú ý nhất?**
Ba cái. Một, learning rate hình học đều cao gấp đôi: `opacity_lr` 0.05 so với 0.025, `scaling_lr` 0.01 so với 0.005, `rotation_lr` 0.002 so với 0.001. Hai, reset opacity là phép **nhân** alpha với 0.1 chứ không phải kẹp về 0.01 như 3DGS — nhẹ tay hơn nhiều. Ba, optimizer "hybrid" tách làm đôi: Adam có amsgrad cho hình học, còn `SparseGaussianAdam` fused CUDA riêng cho hệ số SH bậc cao, chỉ cập nhật Gaussian đang nhìn thấy được.

---

## Slide: Rasterizer riêng của SADGS — `mult`, `cov2D` và $T^{\max}$
*(part45_sadgsx_14.tex)*

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

SADGS không dùng được rasterizer CUDA gốc của 3DGS mà phải fork thành một submodule riêng tên `diff-gaussian-rasterization_structgs`, gọi qua hàm `render_structgs`. Lý do rất cụ thể: cơ chế densify theo tần số cần hai đại lượng mà bản gốc không hề xuất ra. Phần nền tảng thì không đổi — vẫn chia ảnh thành tile mười sáu nhân mười sáu, vẫn sort theo cặp tile-và-độ-sâu, vẫn alpha blending với alpha bằng opacity nhân hàm mũ, transmittance là tích của một trừ alpha các lớp phía trước, bỏ qua đóng góp nhỏ hơn một phần hai trăm năm mươi lăm và dừng sớm khi T xuống dưới mười mũ trừ bốn. Cái mới thứ nhất là `mult`: thay vì gán tile theo hộp bao vuông ba sigma, SADGS xác định chính xác vùng ellipse mà d chuyển vị nhân nghịch đảo sigma hai D nhân d nhỏ hơn hoặc bằng t, với t bằng `mult` nhân hai lần log của hai trăm năm mươi lăm nhân opacity — đúng hai dòng code trong `auxiliary.h`. Hình bên phải cho thấy ellipse nghiêng nằm gọn hẳn trong hình vuông ba sigma, và số tile phải xử lý mỗi splat giảm rõ rệt khi `mult` nhỏ đi. Điều quan trọng là `mult` **không** đụng vào công thức alpha — nó chỉ quyết định Gaussian nào được đưa vào danh sách tile nào, nên giảm `mult` là giảm khối lượng công việc chứ không làm sai blending; tại `mult` bằng một, phần bị cắt có alpha đúng bằng ngưỡng một phần hai trăm năm mươi lăm mà nhân render vốn đã bỏ qua. Cái mới thứ hai là tensor `cov2D` kích thước N nhân bảy: ba cột đầu là ma trận hiệp phương sai hai chiều, cột ba bốn là tâm theo pixel, cột năm là độ sâu, và cột sáu là T-max — transmittance lớn nhất mà Gaussian đó đạt được trên toàn ảnh, ghi bằng thủ thuật `atomicMax` trên bit-pattern của số thực, hợp lệ vì T luôn không âm. Python lấy tensor này ra, lọc những Gaussian có T-max vượt ngưỡng và opacity đủ lớn, rồi mới lấy mẫu Cholesky từ hiệp phương sai để tính eta. Nói ngắn gọn, SADGS đã biến rasterizer thành một cảm biến: nó vừa vẽ ảnh vừa đo đạc trạng thái từng Gaussian trong cùng một lượt forward.

**Chuyển tiếp:** Như vậy ta đã có đủ cả tham số lẫn hạ tầng CUDA; phần tiếp theo sẽ ghép chúng lại để xem kết quả thực nghiệm SADGS đạt được.

### Nếu bị hỏi

**Hỏi: `cov2D` có tham gia vào lan truyền ngược không?**
Không. Nó được `save_for_backward` ở dòng 112 của `diff_gaussian_rasterization_structgs/__init__.py`, nhưng chữ ký hàm `backward` ở dòng 116 nhận nó dưới tên `_cov2D` và không dùng — gradient bị bỏ qua hoàn toàn. Đây là kênh thống kê một chiều, GPU đo rồi trả về Python, không can thiệp vào tối ưu.

**Hỏi: Đặt `mult` nhỏ có làm giảm chất lượng ảnh không?**
Về lý thuyết alpha bị cắt tại biên hộp bằng opacity nhân với, hai trăm năm mươi lăm nhân opacity, mũ trừ `mult`. Tại `mult` bằng 1 thì đúng bằng một phần hai trăm năm mươi lăm — không mất gì. Tại 0.7, sai số chỉ gấp vài lần ngưỡng đó, vẫn dưới một mức lượng tử tám bit, trong khi chi phí giảm khoảng hai mươi đến hai mươi lăm phần trăm. Nhưng `run_train.sh` dùng `mult` bằng 0.25 — lúc đó sai số cắt đã lên hai bậc log, nên cấu hình đó chỉ hợp lý khi đi kèm số iteration ngắn và ngân sách Gaussian bị siết.

**Hỏi: `metric_map` và `compute_extra` dùng để làm gì, có chạy trong lúc train không?**
`metric_map` là mask nhị phân người dùng đưa vào; khi bật `get_flag`, nhân render sẽ `atomicAdd` đếm số lần mỗi Gaussian đóng góp vào vùng mask đó — `forward.cu` dòng 461 đến 467. `compute_extra` thì xuất thêm depth map, opacity map và normal map trong cùng lượt render — `rasterize_points.cu` dòng 116 đến 129. Cả hai đều **không** được bật trong `train.py`: lời gọi `render_structgs` ở dòng 250 không truyền `get_flag`, nên nó về `False`; còn `compute_extra` chỉ được dùng ở `render.py` dòng 41 qua cờ `--render_extra`. Đây là hạ tầng đã có sẵn nhưng chưa dùng trong vòng lặp huấn luyện.

**Hỏi: Vì sao rasterizer trả về tới mười ba tensor?**
Ngoài bảy tensor xuất ra Python (`color`, `radii`, `accum_metric_counts`, `cov2D`, và ba bản đồ phụ), còn bốn buffer nội bộ cùng `num_rendered` và `num_buckets`. `num_buckets` và `sampleBuffer` là do SADGS lưu mẫu transmittance mỗi ba mươi hai Gaussian vào `sampled_T` — `forward.cu` dòng 412 đến 420 — để backward chạy theo bucket song song thay vì duyệt ngược tuần tự toàn tile.
