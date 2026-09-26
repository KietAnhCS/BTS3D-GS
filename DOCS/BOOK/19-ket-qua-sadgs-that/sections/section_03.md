## Diễn biến số lượng Gaussian theo iteration (scene HCM0539)

Hình dưới đây vẽ trực tiếp từ `history.csv` (30 dòng, mỗi 1000 iter một checkpoint đánh giá) toàn bộ quá trình huấn luyện 30000 iteration trên GPU T4 (Colab):

![Số lượng Gaussian vs iteration](figures/03_gaussian_count_vs_iter.png)

### Pha densify (iter 500 → 15000)

Theo `cfg_args`, densify được bật từ `densify_from_iter=500` đến `densify_until_iter=15000`, với `prune_from_iter=6000`, `prune_interval=3000`, `opacity_reset_interval=6000`. Dữ liệu thật khớp đúng với thiết kế này:

- Từ iter 1000 (1,072,940 Gaussian) đến iter 15000 (2,968,520 Gaussian), số lượng Gaussian tăng gần như đơn điệu — đây là pha clone/split đang hoạt động tích cực, đúng như kỳ vọng khi `densify_from_iter=500` đã kích hoạt.
- Có hai điểm sụt giảm rõ rệt, đúng vào bước ngay sau các mốc `opacity_reset_interval=6000`:
  - iter 6000 → 7000: n_gauss giảm từ 2,751,419 xuống 2,460,715 (−10.6%).
  - iter 12000 → 13000: n_gauss giảm từ 2,918,642 xuống 2,830,179 (−3.0%).
  
  Đây là hệ quả trực tiếp của cơ chế `opacity_reset`: khi opacity bị reset về gần 0, các Gaussian có đóng góp thấp bị `prune` (theo `min_weight=0.7`, `prune_interval=3000`) loại bỏ mạnh ở bước làm mới tiếp theo trước khi được "hồi phục" dần bởi densify. Điều này cũng trùng khớp với việc PSNR/SSIM rớt sâu đúng tại iter 6000 và iter 12000 (PSNR tụt còn ~12.15 dB, SSIM còn ~0.40) — dấu hiệu kinh điển của một chu kỳ opacity-reset trong 3DGS/SADGS.
- Ngược lại, hai mốc opacity_reset còn lại tại iter 18000 và 24000 **không** để lại dấu vết nào trên đường n_gauss, vì tại thời điểm đó densify đã dừng (`densify_until_iter=15000` đã qua) nên không còn Gaussian mới được sinh ra để "che lấp" phần bị prune, nhưng đồng thời cũng không quan sát được prune tiếp diễn ở các mốc này trong dữ liệu.

### Pha ổn định (sau iter 15000)

Đây là quan sát đáng chú ý nhất: từ iter 15000 đến iter 30000 (toàn bộ 16 checkpoint còn lại), n_gauss giữ nguyên **chính xác bằng nhau tuyệt đối ở mọi mốc: 2,968,520 Gaussian** — không tăng, không giảm dù chỉ một Gaussian nào, mặc dù cấu hình `prune_until_iter=30000` (đọc từ `cfg_args`) cho thấy cơ chế prune về lý thuyết vẫn còn "được phép" chạy đến hết iteration 30000. Điều này gợi ý rằng trong thực thi thật, việc prune bị gắn chặt vào cùng chu kỳ với densify (điều kiện kích hoạt phụ thuộc `densify_until_iter`) nên khi densify dừng ở iter 15000, nhánh prune cũng ngừng hoạt động theo, dẫn tới số Gaussian bị "đóng băng" hoàn toàn. Đây là điểm cần xác minh thêm ở source code huấn luyện nếu muốn tinh chỉnh lại lịch trình densify/prune độc lập nhau, vì hiện tại effectively không có "dọn dẹp" hậu-densify nào diễn ra trong 15000 iteration cuối.

### Nhận định về chi phí bộ nhớ so với VRAM quan sát

Số Gaussian cuối cùng ~2.97 triệu (làm tròn "gần 3M" như giả thuyết ban đầu là hợp lý) đi kèm VRAM đỉnh quan sát được trong log là **5.576 GB tại iter 14000** — đúng vào giai đoạn cuối densify khi n_gauss đạt đỉnh cục bộ (2,923,199 tại iter 14000) trước khi ổn định ở iter 15000. Sau đó, khi densify dừng, VRAM giảm và ổn định quanh **4.35 GB** (iter 16000–29000), chỉ tăng lại lên ~5.0 GB ở bước cuối 30000 (nhiều khả năng do overhead lưu checkpoint cuối cùng, chứ không phải do tăng số Gaussian vì n_gauss vẫn y hệt).

Với ~2.97 triệu Gaussian ở SH degree=3 (48 hệ số cầu điều hòa/Gaussian, cộng vị trí, scale, rotation, opacity — tổng ~59 giá trị float32/Gaussian ≈ 236 byte tham số thô/Gaussian), riêng phần tham số mô hình đã chiếm khoảng 0.65–0.7 GB. Optimizer Adam (lưu thêm 2 buffer momentum/variance cho mỗi tham số) đẩy chi phí thực tế lên khoảng gấp 2.5–3 lần con số đó (~1.7–2.1 GB), phần còn lại của 4.3–5.6 GB quan sát được là buffer render (rasterizer, ảnh 30 view, gradient tích lũy cho densify) và overhead khung PyTorch/CUDA. Với card T4 có 16 GB VRAM, mức sử dụng đỉnh 5.576 GB chỉ chiếm khoảng 35% dung lượng — biên an toàn còn khá rộng, nghĩa là cấu hình densify/prune hiện tại (dừng tăng trưởng ở ~3M Gaussian) là một lựa chọn **thận trọng hơn mức cần thiết** xét về giới hạn phần cứng: T4 vẫn còn dư địa để cho phép densify tiếp tục lâu hơn (nâng `densify_until_iter` hoặc nới lỏng ngưỡng prune) nếu mục tiêu là tăng chi tiết tái tạo, mà không lo tràn VRAM. Ngược lại, nếu mục tiêu ưu tiên là tốc độ render/kích thước file mô hình (mỗi Gaussian ~236 byte tham số × 2.97M ≈ 700 MB chỉ riêng phần lưu trữ, chưa kể metadata), thì việc dừng densify ở iter 15000 và giữ nguyên số lượng Gaussian trong nửa sau của quá trình huấn luyện là hợp lý để tránh mô hình phình to không cần thiết trong khi các chỉ số PSNR/SSIM/LPIPS ở giai đoạn 15000–30000 (theo `history.csv`) không còn cải thiện đáng kể nữa.
