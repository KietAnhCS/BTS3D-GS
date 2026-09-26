## Kết quả thực nghiệm SADGS thật (30 000 iter, HCM0539, Colab T4) so với FastGS-lite

Bảng dưới đối chiếu hai lần chạy trên cùng scene `HCM0539` (240 ảnh train / 60 ảnh test), 30 000
iterations, Colab T4:

- **FastGS-lite (cũ)** — số liệu tham khảo đã công bố trong `DOCS/README.md`, mục "Kết quả thực
  nghiệm mới nhất", chạy **trước khi** tích hợp cơ chế structure-aware densification thật của
  SADGS (`densify_and_split_structgs` / `densify_and_clone_structgs` / `densify_and_prune_structgs`
  / `final_prune_structgs`).
- **SADGS (mới)** — số liệu đọc từ `output/26sepoutput/sadgs_models/leaderboard.csv`, là lần chạy
  thật đầu tiên với cơ chế densify/prune structure-aware của SADGS.

| Pipeline | Score | PSNR (dB) | SSIM | LPIPS (thấp hơn = tốt hơn) | Gaussians cuối | Thời gian train | VRAM đỉnh |
|---|---|---|---|---|---|---|---|
| FastGS-lite (cũ) | 0.8579 | 25.27 | 0.8659 | 0.1364 | 344 484 | 1776 s (~29,6 phút) | 1,92 GB / 14,56 GB (T4) |
| SADGS (mới) | 0.8681 | 23.778 | 0.8897 | 0.0915 | 2 968 520 | 4234,8 s (~70,6 phút) | 7,21 GB |

![So sánh FastGS-lite vs SADGS trên HCM0539](figures/06_sadgs_vs_fastgs_lite_comparison.png)

### Chênh lệch % tính trực tiếp từ hai nguồn số liệu trên

- **Score**: (0,8681 − 0,8579) / 0,8579 = **+1,19%** — SADGS cao hơn.
- **PSNR**: (23,778 − 25,27) / 25,27 = **−5,90%** — SADGS thấp hơn khoảng 1,49 dB.
- **SSIM**: (0,8897 − 0,8659) / 0,8659 = **+2,75%** — SADGS cao hơn.
- **LPIPS**: (0,0915 − 0,1364) / 0,1364 = **−32,9%** — vì LPIPS càng thấp càng tốt, đây là một
  **cải thiện lớn**, không phải một điểm yếu. Không được đọc "0,0915 < 0,1364" thành "SADGS tệ
  hơn"; đúng ra SADGS cho ảnh nhận thức-giống-thật (perceptual similarity) tốt hơn rõ rệt.
- **Số Gaussian cuối**: 2 968 520 / 344 484 = **8,62 lần** nhiều hơn.
- **Thời gian train**: 4234,8 / 1776 = **2,38 lần** lâu hơn (tăng 138,4%).
- **VRAM đỉnh**: 7,21 GB so với 1,92 GB đã công bố cho FastGS-lite → **cao hơn 3,76 lần**
  (tăng ~275%), **không** thấp hơn. Đây là điểm cần đính chính: giả định "VRAM đỉnh mới vẫn thấp
  hơn FastGS-lite" là **sai** khi đối chiếu với số liệu thật trong `DOCS/README.md` (1,92 GB, không
  phải một con số khác) — số Gaussian tăng 8,6 lần kéo theo VRAM đỉnh tăng tương ứng, đúng như dự
  đoán vật lý (mỗi Gaussian chiếm bộ nhớ cố định cho vị trí, hiệp phương sai, SH, alpha).

### Nhận định

Bức tranh không đơn giản là "SADGS tốt hơn toàn diện" hay "tệ hơn toàn diện":

- **Điểm mạnh rõ ràng**: LPIPS giảm 32,9% và SSIM tăng 2,75% cho thấy cơ chế structure-aware
  densification thật sự đang đặt Gaussian đúng chỗ hơn về mặt cấu trúc/kết cấu ảnh — ảnh render
  giống thật hơn về mặt tri giác (perceptual), đúng như mục tiêu thiết kế: so khớp screen-space
  extent với cấu trúc texture đa tỉ lệ, tách dị hướng thay vì clone/split đẳng hướng.
- **Điểm yếu rõ ràng**: PSNR giảm 5,9% (mất ~1,49 dB) dù SSIM và LPIPS đều cải thiện — đây là dấu
  hiệu kinh điển của việc tối ưu hoá thiên về cấu trúc/tri giác thay vì sai số pixel-wise thuần
  tuý (MSE), không phải bug. Nhưng vì PSNR vẫn là một phần cấu thành trọng số của composite score
  (điểm thi đấu), việc PSNR giảm trong khi score tổng vẫn tăng cho thấy SSIM/LPIPS đang bù lại
  nhiều hơn phần PSNR mất đi.
- **Đánh đổi chi phí — rủi ro thật**: số Gaussian cuối tăng 8,62 lần (344K → 2,97M), thời gian
  train tăng 2,38 lần, và VRAM đỉnh tăng 3,76 lần (1,92 GB → 7,21 GB). Đây là bằng chứng trực tiếp
  rằng cơ chế densify/prune của SADGS đang "bung" mật độ Gaussian mạnh hơn hẳn ADC gốc — nhất
  quán với thiết kế lý thuyết (mật độ hoá theo cấu trúc → nhiều Gaussian hơn ở vùng chi tiết) chứ
  không phải chỉ ở vùng cần thiết. 7,21 GB vẫn nằm trong ngân sách 14,56 GB của T4 nên chưa OOM,
  nhưng biên độ tăng theo tỷ lệ 3,76 lần là đáng lo nếu scene phức tạp hơn hoặc chạy nhiều scene
  song song trên cùng GPU.

**Kết luận**: đây là bằng chứng SADGS *đang hoạt động đúng cơ chế được thiết kế* (structure-aware
densification tạo nhiều Gaussian hơn ở vùng cấu trúc chi tiết, cải thiện SSIM/LPIPS), nhưng đồng
thời cũng là **dấu hiệu over-densify cần điều chỉnh ngưỡng** — mức tăng 8,6 lần số Gaussian đi kèm
sụt giảm PSNR và chi phí train/VRAM tăng vài lần là quá lớn để coi là "chi phí chấp nhận được" mà
không cần tinh chỉnh. Bước tiếp theo hợp lý: rà lại ngưỡng gradient/độ nhạy cấu trúc trong
`densify_and_split_structgs` / `densify_and_clone_structgs` để siết bớt số lần split ở vùng đã đủ
mật độ, và bổ sung `final_prune_structgs` mạnh tay hơn để cắt bớt Gaussian dư thừa mà không đánh
đổi quá nhiều SSIM/LPIPS đã đạt được — mục tiêu là giữ được phần cải thiện tri giác trong khi kéo
PSNR và chi phí bộ nhớ/thời gian về gần mức FastGS-lite hơn.
