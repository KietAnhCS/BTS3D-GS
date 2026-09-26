## Tài nguyên hệ thống trong quá trình train thật (HCM0539, 30000 iterations, Colab T4)

Toàn bộ số liệu trong mục này lấy trực tiếp từ log thật của lần train scene `HCM0539` trên Colab (GPU T4, ~15GB VRAM khả dụng), file nguồn: `output/26sepoutput/sadgs_models/history.csv` (30 checkpoint, log mỗi 1000 iteration) và `output/26sepoutput/sadgs_models/leaderboard.csv` (tổng kết cuối phiên).

![Resource usage vs iteration](figures/04_resource_usage_vs_iter.png)

Hình trên gồm 3 subplot: (a) thời gian tích lũy (giây) theo iteration cùng đạo hàm rời rạc "giây / 1000 iter" (trục phải) để quan sát tốc độ train có chậm dần khi số Gaussian tăng hay không; (b) RAM hệ thống (GB) theo iteration; (c) VRAM GPU (GB) theo iteration, có kẻ đường tham chiếu ~15GB là giới hạn VRAM thực tế của T4.

### Bảng tóm tắt số liệu đo được

| Chỉ số | Giá trị thực đo | Nguồn |
|---|---|---|
| Tổng thời gian train | 4234.8 s ≈ **1 giờ 10 phút 35 giây** | `leaderboard.csv: train_s` |
| Tốc độ train trung bình | 30000 / 4234.8 ≈ **7.08 iteration/giây** | tính từ `train_s` và `iters` |
| Tốc độ train (giây/1000 iter, trung bình theo log) | ≈ 142.5 s/1000 iter | tính từ `history.csv: elapsed_s` |
| Tốc độ train chậm nhất quan sát được | ≈ 200.3 s/1000 iter (quanh iter 18000–19000) | `history.csv` (đạo hàm rời rạc) |
| Tốc độ train nhanh nhất quan sát được | ≈ 87.1 s/1000 iter (giai đoạn đầu, iter 1000–2000) | `history.csv` |
| RAM đỉnh thực đo | **6.24 GB** (tại iter 22000) | `history.csv: ram_gb` |
| VRAM đỉnh thực đo (báo cáo cuối phiên) | **7.21 GB** | `leaderboard.csv: peak_vram_gb` |
| VRAM đỉnh quan sát trong log định kỳ | 5.58 GB (tại iter 12000, đúng thời điểm density-reset gây rớt điểm PSNR) | `history.csv: vram_gb` |
| Số Gaussian ổn định (từ iter ~15000) | 2,968,520 | `history.csv: n_gauss` / `leaderboard.csv: n_gauss` |

### Nhận định về tốc độ train và số Gaussian

Đường "giây/1000-iter" trong subplot (a) không tăng đơn điệu theo n_gauss như kỳ vọng thuần tuý lý thuyết (số Gaussian càng nhiều thì mỗi bước rasterization/backward càng tốn), mà dao động mạnh: giai đoạn đầu (iter 1000–5000, n_gauss tăng từ ~1.07M lên ~2.6M) tốc độ khá nhanh và ổn định quanh 90–150 s/1000 iter; sau đó xuất hiện 2 đỉnh bất thường rõ rệt quanh iter 6000 và iter 12000 (~170–200 s/1000 iter), trùng đúng với hai lần `d_score` âm mạnh trong `history.csv` (density reset/prune-densify của 3DGS làm PSNR rớt tạm thời xuống ~12), tức chi phí thời gian tăng đột biến không chỉ do n_gauss mà chủ yếu do các thao tác adaptive density control (clone/split/prune) diễn ra định kỳ. Từ iter 15000 trở đi, n_gauss bão hoà ở mức 2,968,520 (không đổi đến hết 30000 iter) và tốc độ train ổn định trở lại quanh 120–140 s/1000 iter — cho thấy sau khi ngừng densify, chi phí mỗi iteration gần như hằng định, không tiếp tục leo thang.

### Nhận định về khả năng vận hành lặp lại trên Colab free-tier T4

- **RAM**: đỉnh 6.24 GB — nằm rất xa giới hạn RAM hệ thống của runtime Colab free (thường ~12–13GB khả dụng), không phải điểm nghẽn.
- **VRAM**: đỉnh thực đo 7.21GB so với ~15GB VRAM của T4 — dùng khoảng **48%** VRAM, còn dư biên an toàn khá lớn (~7.8GB) để chịu được biến động do dataset/scene khác có mật độ Gaussian cao hơn, miễn không tăng đột biến quá 2 lần mức hiện tại.
- **Thời gian**: tổng thời gian train thật chỉ **~1h10m35s**, so với giới hạn phiên Colab free-tier (thường ~12 giờ liên tục hoặc bị ngắt sớm hơn nếu không tương tác/idle, thường 30–90 phút idle timeout). Vì thời gian train thực đo (~1h10m) **nhỏ hơn nhiều** so với trần 12 giờ, rủi ro bị Colab ngắt do vượt giới hạn session gần như không đáng kể trong một lần chạy đơn — miễn phiên được giữ active (tránh idle-timeout, ví dụ không đóng tab hoặc có cơ chế giữ kết nối/heartbeat).
- **Rủi ro thực tế lớn hơn không nằm ở tài nguyên GPU/RAM mà ở tính liên tục của phiên**: Colab free có thể ngắt kết nối bất ngờ (idle, giới hạn sử dụng linh động theo giờ cao điểm) trước khi chạm mốc 12 giờ. Với thời lượng train ~1h10m cho một scene, việc train 1 scene trong một phiên là khả thi ổn định; tuy nhiên nếu cần train nhiều scene liên tiếp trong cùng phiên (nhiều lần x ~1h10m), nguy cơ mất phiên giữa chừng tăng theo số lượng scene, nên cần cơ chế autosave checkpoint định kỳ (đã có, xem commit "autosave checkpoint to Drive before re-raising on mid-scene failure") để không mất toàn bộ tiến trình khi phiên bị ngắt.
- **Kết luận**: với mức tiêu thụ RAM/VRAM đo được, scene HCM0539 hoàn toàn khả thi để train lặp lại nhiều lần trên Colab T4 free-tier về mặt tài nguyên phần cứng; điểm cần quản lý là tính ổn định của phiên (idle-disconnect) chứ không phải giới hạn 12 giờ hay giới hạn VRAM/RAM.
