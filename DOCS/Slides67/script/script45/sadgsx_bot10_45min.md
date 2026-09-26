# Kịch bản 45 phút — BOT 10 (sadgsx): Tổng hợp đầu-cuối SADGS

> Slide phụ trách: `part45_sadgsx_15.tex` — *"Tổng hợp đầu-cuối: pipeline SADGS, chi phí, và phần chưa bật"* (1 frame, slide khép lại phần SADGS).
> Chương MD đi kèm: [`DOCS/BOOK/18-sadgs-co-che-chuyen-sau/10-tong-hop-dau-cuoi.md`](../../../BOOK/18-sadgs-co-che-chuyen-sau/10-tong-hop-dau-cuoi.md)
> Tổng thời lượng slide: **~2 phút 15** — chia làm ba nhịp, mỗi nhịp ~45 giây.

---

## Slide: Tổng hợp đầu-cuối — pipeline SADGS, chi phí, và phần chưa bật

### Nhịp 1 — Sơ đồ pipeline (cột trái)
**Thời lượng: ~45 giây**

Đến đây em đã trình bày từng cơ chế riêng lẻ, nên slide cuối này ghép tất cả lại thành một sơ đồ chạy được. Hình bên trái là mười ba bước của **một** iteration SADGS, đúng thứ tự thực thi trong `train.py` từ dòng 202 đến 440. Màu xanh lá là phần lõi kế thừa từ 3DGS và FastGS: cập nhật learning rate, chọn camera, render, tính loss, backward, rồi bước optimizer. Màu cam là tầng tần số riêng của SADGS. Màu hồng là khối densify và prune. Còn dải màu xanh nhạt trên cùng là phần tiền xử lý chạy **một lần duy nhất** ngoài vòng lặp — cache structure tensor cho toàn bộ ảnh train, ở dòng 160 đến 200. Khối xám nét đứt bên phải là hai nhánh mặc định TẮT: `compute_3D_filter` và `sample_bbox_faces`. Một chi tiết nhỏ mà em muốn các thầy cô để ý: dòng 216 viết là `if iteration % 1 == 0` — điều kiện luôn đúng, nên bậc SH tăng **mỗi vòng**, chạm trần bậc ba chỉ sau ba iteration, thay vì sau ba nghìn vòng như 3DGS gốc.

**Chuyển tiếp:** Vấn đề là mười ba bước này không chạy với cùng một nhịp — và chính điều đó quyết định chi phí.

---

### Nhịp 2 — Mô hình chi phí và khác biệt cốt lõi (cột phải, phần trên)
**Thời lượng: ~45 giây**

Chi phí trung bình một vòng có năm số hạng: $a$ nhân $N$ cho forward-backward-Adam, $b$ nhân $P$ cho rasterize theo số pixel, $c$ là overhead cố định, rồi hai số hạng riêng của SADGS — $a_\eta N$ chia 10 cho phần thống kê tần số, và $a_D N$ chia 100 cho densify và prune. Hai mẫu số 10 và 100 không phải em tự đặt: số 10 đến từ điều kiện `iteration % 10 == 0` ở dòng 275, số 100 là `densification_interval` mặc định trong `arguments`. Nghĩa là hai cơ chế đắt nhất về mặt thuật toán lại chỉ chiếm phần nhỏ trong tổng chi phí, vì chúng được chia cho chu kỳ kích hoạt. Và vì **bốn trên năm** số hạng đều tuyến tính theo $N$, kết luận thực hành là: cắt được ba mươi phần trăm số Gaussian thì cắt được xấp xỉ ba mươi phần trăm chi phí, trên **mọi** iteration còn lại. Đó là lý do prune đa view — một dòng code so `low_ratio` với 0.8 ở dòng 353 — là đòn bẩy mạnh nhất, mạnh hơn nhiều so với tối ưu vi mô một CUDA kernel, vì nó tác động lên **biến** $N$ chứ không phải lên **hệ số** $a$.

So với FastGS, khác biệt cốt lõi gói trong ba điểm: tín hiệu densify là phép **AND** của gradient với `high_ratio` lớn hơn 0.8; split là **dị hướng**, mỗi trục chia $k$ bằng trần của căn $\eta$; và prune có thêm tiêu chí đa view. Nền rasterization — compact box `mult` bằng 0.7, Fused và Sparse Adam — giữ nguyên không đổi.

**Chuyển tiếp:** Nhưng nếu chỉ dừng ở đây thì bài trình bày sẽ không trung thực. Em muốn nói rõ phần còn lại.

---

### Nhịp 3 — Phần đã cài nhưng chưa bật, và rủi ro đo đạc (cột phải, phần dưới)
**Thời lượng: ~45 giây**

Hai ô cuối của slide là phần em nghĩ quan trọng nhất. Có năm lời gọi trong `train.py` đã được viết đầy đủ nhưng **bị comment**, nên không bao giờ chạy: `expand_undersized_gs` ở dòng 356, `final_prune_structgs` ở 418, prune đa view trong nhánh warmup ở 393, `training_report` ở 306 — nên không có log PSNR nào lúc train — và `scene.save` ở dòng 442. Cộng thêm mười cờ mặc định `False` và hai nhánh chết do truyền `None` vào đối số. Phần **thực sự chạy** của SADGS chỉ là một chuỗi bốn khâu: cache structure tensor, `update_freq_stats_online`, tính `high_ratio` với `low_ratio`, rồi `densify_and_prune_structgs`.

Về đo đạc, có ba rủi ro cần nói: `render.py` dòng 40 đến 44 đo FPS bằng `time.time()` mà **không** gọi `cuda.synchronize()`, nên con số FPS báo cáo có thể lạc quan; `metrics.py` dòng 74 dùng hàm SSIM khác với `fast_ssim` dùng lúc train; và `full_eval.py` ở các dòng 90, 95, 121 truyền ba tham số `--budget`, `--mode`, `--sh_lower` mà `argparse` chưa hề khai báo, nên script đó không chạy được như đang viết. Cuối cùng — và em xin nhấn mạnh — **repo SADGS hiện không chứa bất kỳ số đo PSNR, SSIM, LPIPS hay thời gian train nào**. Bảng số liệu trong `DOCS/README.md` là của pipeline FastGS-lite cũ, trước khi tích hợp structure-aware densification; đó **không** phải kết quả của SADGS, và em không trình bày nó như vậy.

**Chuyển tiếp:** Em xin kết thúc phần SADGS ở đây và chuyển sang phần hỏi đáp.

---

## Nếu bị hỏi

**Hỏi: "Em nói SADGS train 30 nghìn iteration à?"**
Dạ không. 30 000 chỉ là mặc định trong `arguments/__init__.py` dòng 75. Script thật sinh số liệu paper là `run_train.sh`, và nó dùng `--iterations 3000` cho Mip-NeRF 360 với Deep Blending, `--iterations 7000` cho Tanks & Temples, với `densify_until_iter` tương ứng là 1900 và 4000. Thêm nữa `run_train.sh` truyền `--densification_interval 500`, nên khối densify chỉ chạy khoảng ba đến bốn lần trong toàn bộ quá trình train — toàn bộ cơ chế structure-aware phải phát huy tác dụng trong vài lần gọi đó.

**Hỏi: "Vậy các cờ mặc định TẮT nghĩa là paper không dùng chúng?"**
Không hẳn, và đây là điểm em phải nói rõ. `sample_bbox_faces` và `warmup_densification` mặc định là `False` ở dòng 129 và 130, nhưng **cả bốn cấu hình** trong `run_train.sh` đều truyền `--sample_bbox_faces --warmup_densification` để bật lại. Cấu hình 2 và 4 còn truyền `--batch_size 2`. Nên "cấu hình mặc định" và "cấu hình paper" là hai chế độ khác nhau — bảng mặc định không mô tả lần chạy sinh ra số liệu công bố.

**Hỏi: "Split dị hướng có nguy cơ bùng nổ số Gaussian không?"**
Có, và đây là một bug thật em tìm thấy. Docstring ở `gaussian_model.py` dòng 697 viết "clamp min=2 and max=8 to limit VRAM usage", nhưng dòng thực thi 701 chỉ là `torch.clamp(ks, min=1)` — **không có trần 8**. Mà số con sinh ra là tích $k_x k_y k_z$, nên một Gaussian có $\eta$ cao ở cả ba trục có thể sinh ra rất nhiều con. Chốt chặn duy nhất còn lại là ngưỡng gradient $10^{-5}$ ở `train.py` dòng 333, mà chính comment trong code gọi là `[CRITICAL] We MUST use this to prevent 7M points`.

**Hỏi: "Với `batch_size 2` thì loss có được trung bình không?"**
Không. `train.py` dòng 262 có `#/ opt.batch_size` bị comment, và `loss.backward()` ở dòng 263 nằm **trong** vòng lặp batch, nên gradient cộng dồn chứ không trung bình — learning rate hiệu dụng tăng gấp `batch_size` lần. Vì `run_train.sh` thật sự dùng `--batch_size 2`, đây là thứ cần sửa trước khi so sánh công bằng giữa các cấu hình.

**Hỏi: "File `.ply` xuất ra nặng bao nhiêu?"**
`construct_list_of_attributes` ở `gaussian_model.py` dòng 379 sinh ra 63 trường `float32`: 3 toạ độ, 3 normal luôn bằng 0, 3 hệ số SH bậc 0, 45 hệ số SH bậc cao, 1 opacity, 3 scale, 4 quaternion, và 1 trường `filter_3D`. Vậy là 252 byte mỗi Gaussian — nhiều hơn đúng 4 byte so với record 3DGS chuẩn, vì SADGS ghi thêm `filter_3D`.
