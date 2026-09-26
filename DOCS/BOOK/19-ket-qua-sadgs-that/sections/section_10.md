## Đánh giá tổng thể & khuyến nghị tiếp theo

Đây là lần đầu tiên pipeline có số liệu **thật** cho cấu hình SADGS (structure-aware
densification) — trước lần chạy này, `DOCS/README.md` chỉ ghi được số liệu tham khảo của
bản FastGS-lite cũ (chưa bật `densify_and_split_structgs` /
`densify_and_clone_structgs` / `densify_and_prune_structgs` / `final_prune_structgs`).
Kết quả dưới đây lấy từ `output/26sepoutput/sadgs_models/leaderboard.csv`, scene
`HCM0539`, 30 000 iteration, chạy trên Colab T4: **score = 0.8681, PSNR = 23.778 dB,
SSIM = 0.8897, LPIPS = 0.0915, số Gaussian cuối = 2 968 520, thời gian train = 4234.8 s
(~1 giờ 10 phút), VRAM đỉnh = 7.21 GB, 60 ảnh test, độ phân giải 1320×989**.

### 1. Ba điểm mạnh nổi bật nhất

- **Score tổng thể cải thiện so với mốc tham khảo cũ**: 0.8681 so với 0.8579 của bản
  FastGS-lite (chưa có densify/prune structure-aware). Đây là lần đầu con số này được đo
  thật với cơ chế SADGS, và nó nhỉnh hơn mốc cũ dù công thức score gộp nhiều chỉ số khác
  nhau (xem điểm rủi ro #2 bên dưới).
- **SSIM và LPIPS đều tốt hơn rõ rệt**: SSIM 0.8897 so với 0.8659, và đặc biệt LPIPS
  0.0915 so với 0.1364 (giảm ~33% sai số nhận thức). LPIPS là chỉ số nhạy với chi tiết cấu
  trúc/texture cục bộ, nên cải thiện này phù hợp với kỳ vọng của structure-aware
  densification: chèn thêm Gaussian đúng chỗ có cấu trúc phức tạp để tái tạo chi tiết tốt
  hơn.
- **VRAM đỉnh vẫn nằm trong ngân sách an toàn của T4**: 7.21 GB / 14.56 GB, dù số Gaussian
  tăng gần 9 lần so với lần chạy cũ (2 968 520 so với 344 484). Điều này cho thấy pipeline
  (Fused Adam, 3D anti-aliasing filter, Morton reordering) vẫn kiểm soát được bộ nhớ dù mật
  độ điểm tăng mạnh — không có dấu hiệu OOM, còn dư khoảng 7.3 GB.

### 2. Ba rủi ro/điểm yếu cần lưu ý nhất

- **Số Gaussian cuối tăng gần 9 lần (344 484 → 2 968 520)**, kéo theo thời gian train tăng
  gần 2.4 lần (1776 s → 4234.8 s) và VRAM đỉnh tăng gần 3.8 lần (1.92 GB → 7.21 GB). Đây là
  rủi ro thực sự về chi phí lưu trữ file `.ply` cuối và tốc độ inference/render sau này
  (chưa được đo trong lần chạy này — xem khuyến nghị #3). Nếu mở rộng ra scene lớn hơn hoặc
  chạy trên GPU yếu hơn T4, mức tăng này có thể chạm trần VRAM hoặc làm thời gian
  render/submission vượt giới hạn cuộc thi.
- **PSNR giảm so với mốc cũ**: 23.778 dB so với 25.27 dB (giảm ~1.5 dB), dù score tổng lại
  tốt hơn. Điều này cho thấy công thức score tổng hợp (kết hợp psnr_norm, ssim, lpips —
  xem cột `psnr_norm=0.7926` trong leaderboard) không coi PSNR là yếu tố quyết định duy
  nhất; SADGS đang đánh đổi PSNR để lấy SSIM/LPIPS tốt hơn. Đây có thể là đánh đổi chấp
  nhận được theo đúng thiết kế của thang điểm cuộc thi, nhưng cần xác nhận lại công thức
  score chính thức (trọng số giữa psnr_norm/ssim/lpips) trước khi kết luận đây là “thắng
  lợi” toàn diện — nếu công thức thật sự ưu tiên PSNR cao hơn thì kết quả này có thể bị
  đánh giá thấp hơn con số 0.8681 thể hiện.
- **Đây mới là một scene, một lần chạy duy nhất**: chưa có lặp lại (seed khác, scene
  khác) để kiểm tra độ ổn định của SADGS. Chưa thể kết luận SADGS tốt hơn 3DGS gốc hay
  FastGS-lite trên diện rộng — số liệu so sánh hiện tại chỉ là 1 điểm dữ liệu SADGS đối
  đầu với 1 điểm dữ liệu tham khảo cũ, khác nhau cả về cơ chế densify/prune lẫn có thể
  khác nhau về seed/checkpoint autosave giữa hai lần chạy.

### 3. Khuyến nghị cụ thể cho lần chạy tiếp theo

- Chạy thêm ít nhất 1–2 scene khác trong `VAI_NVS_DATA_ROUND2` với cùng cấu hình SADGS để
  có cơ sở so sánh chéo, tránh kết luận vội từ một scene duy nhất.
- Thử giảm mật độ Gaussian cuối bằng cách điều chỉnh `importance_score_threshold` (tăng
  ngưỡng để lọc chặt hơn) hoặc `densify_prune_ratio` (prune mạnh hơn sau mỗi vòng
  densify), rồi đo lại xem score/PSNR/SSIM/LPIPS đổi thế nào khi số Gaussian giảm về gần
  mức cũ — để xác định liệu 2.97M điểm có thực sự cần thiết hay có thể giảm mà không mất
  nhiều chất lượng.
- Đo riêng thời gian render/inference trên tập test (hiện `leaderboard.csv` chỉ có
  `train_s`, chưa có số liệu render time) — với gần 9 lần số Gaussian, thời gian render mỗi
  ảnh và dung lượng file `.ply` cuối cần được ghi lại để đánh giá tính khả thi khi triển
  khai/nộp bài trong giới hạn thời gian của cuộc thi.
- Xác nhận lại công thức score chính thức của ban tổ chức (trọng số giữa psnr_norm, ssim,
  lpips) để hiểu rõ việc PSNR giảm 1.5 dB đổi lấy SSIM/LPIPS tốt hơn có thực sự có lợi cho
  điểm số cuối cùng hay không.

### Kết luận ngắn gọn

Lần chạy SADGS thật đầu tiên trên `HCM0539` cho kết quả nhỉnh hơn mốc tham khảo FastGS-lite
cũ về score/SSIM/LPIPS, nhưng đánh đổi bằng PSNR thấp hơn và số Gaussian/thời gian
train/VRAM tăng đáng kể; đây là tín hiệu tích cực bước đầu chứ chưa phải bằng chứng đủ
mạnh để khẳng định SADGS vượt trội — cần thêm scene, thêm phép đo render time, và cần soát
lại công thức score trước khi đưa ra kết luận chắc chắn.
