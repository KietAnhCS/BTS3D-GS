## Sụt giảm score định kỳ tại các mốc opacity reset (iter=6000, iter=12000)

### Hiện tượng quan sát được

Trong log huấn luyện thật của scene `HCM0539` (30000 iterations, GPU T4 trên Colab, dữ liệu đọc từ
`output/26sepoutput/sadgs_models/history.csv`), cột `score` — điểm tổng hợp từ PSNR, SSIM, LPIPS —
đi lên đều đặn qua từng checkpoint 1000 iteration, nhưng có đúng hai lần rơi thẳng đứng rồi bật lại
ngay ở checkpoint kế tiếp:

| iter  | score  | d_score  | Giai đoạn |
|-------|--------|----------|-----------|
| 5000  | 0.8433 | +0.0058  | trước reset |
| **6000**  | **0.4557** | **-0.3875** | **ngay tại opacity reset #1** |
| 7000  | 0.8529 | +0.3972  | phục hồi sau reset |
| 8000  | 0.8587 | +0.0058  | ổn định trở lại |
| 11000 | 0.8642 | +0.0022  | trước reset |
| **12000** | **0.4548** | **-0.4094** | **ngay tại opacity reset #2** |
| 13000 | 0.8663 | +0.4115  | phục hồi sau reset |
| 18000 | 0.8712 | -0.0002  | mốc 6k tiếp theo — **không sụt** |
| 24000 | 0.8692 | -0.0005  | mốc 6k tiếp theo — **không sụt** |

PSNR tại iter=6000 và 12000 rơi từ ~23.2/23.7 dB xuống ~12.15 dB, SSIM từ ~0.86-0.89 xuống ~0.40,
LPIPS tăng từ ~0.10-0.12 lên ~0.47 — tức toàn bộ ba metric ảnh hưởng cùng lúc, không phải nhiễu đo
đơn lẻ ở một chỉ số.

### Nguyên nhân: cơ chế opacity reset trong 3DGS/SADGS

File cấu hình thực tế của lần train này (`output/26sepoutput/sadgs_models/HCM0539/cfg_args`) khai
báo:

- `opacity_reset_interval=6000`
- `prune_from_iter=6000`, `prune_interval=3000`
- `densify_until_iter=15000`

Cơ chế opacity reset là một bước tái cấu trúc có chủ đích của thuật toán Gaussian Splatting (kế
thừa từ 3DGS gốc, SADGS giữ nguyên logic): cứ mỗi `opacity_reset_interval` iteration, opacity của
**toàn bộ** Gaussian trong scene bị ghi đè về một giá trị thấp gần như đồng nhất (ở đây theo
`opacity_reset_decay=0.1`), bất kể Gaussian đó trước đó "quan trọng" hay "mờ nhạt" thế nào. Mục
đích là để lộ ra những Gaussian đang tồn tại chỉ nhờ opacity cao chứ không đóng góp gradient hay
tín hiệu tái tạo thực sự — sau khi reset, các Gaussian vô dụng sẽ có gradient thấp và bị cơ chế
`prune` (được kích hoạt cùng lúc qua `prune_from_iter=6000`, `prune_interval=3000`) loại bỏ ở vòng
lặp tiếp theo, giữ cho model gọn và tránh chồng lấn/overfitting cục bộ.

Vì opacity bị ghi về gần 0 đồng loạt, ảnh render ngay tại iteration đó gần như "trong suốt" cục bộ ở
nhiều vùng — PSNR/SSIM rơi tự do, LPIPS (đo khác biệt cảm nhận) tăng vọt. Đây là hệ quả trực tiếp,
tức thời của thao tác ghi đè tham số, không phải suy giảm khả năng biểu diễn của model. Ngay sau đó,
optimizer chỉ cần vài trăm đến ~1000 iteration để tăng lại opacity của các Gaussian còn sống sót
(gradient descent trên `opacity_lr=0.05` khá nhanh), nên checkpoint kế tiếp (7000, 13000) phục hồi
gần như hoàn toàn — thậm chí có xu hướng vượt nhẹ điểm trước reset vì phần Gaussian dư thừa đã bị
prune, cấu trúc scene "sạch" hơn.

**Điểm cần lưu ý về số liệu thật:** giả thuyết ban đầu là hiện tượng sẽ lặp lại ở mọi bội số 6000
(iter=18000, 24000). Số liệu thực tế cho thấy **không đúng** — d_score tại 18000 và 24000 chỉ là
-0.0002 và -0.0005, không có sụt giảm. Lý do nằm ở chính `cfg_args`: `densify_until_iter=15000`.
Trong 3DGS/SADGS, opacity reset chỉ được kích hoạt trong pha densification đang hoạt động (Gaussian
còn đang được clone/split/prune theo gradient); sau iter=15000, densification dừng lại nên vòng lặp
reset-opacity cũng dừng theo, dù `opacity_reset_interval=6000` về mặt cấu hình vẫn "đến hạn" ở
18000/24000. Do đó chỉ có đúng 2 lần crash trong toàn bộ 30000 iteration của scene này (tại 6000 và
12000), không phải 4 lần như suy đoán ban đầu.

### Nhận định

Đây **là hành vi kỳ vọng, có chủ đích của thuật toán, không phải lỗi** của pipeline SADGS hay của
quá trình train scene HCM0539. Việc quan sát thấy score rớt sâu rồi phục hồi ngay checkpoint sau là
dấu hiệu cơ chế prune-via-opacity-reset đang hoạt động đúng thiết kế.

Tuy nhiên, đây là **rủi ro vận hành thật** cần ghi nhận rõ trong quy trình đánh giá/chấm điểm:

- Nếu hệ thống chấm điểm, dashboard giám sát, hoặc early-stopping tự động lấy checkpoint **đúng
  tại iter=6000 hoặc iter=12000** làm căn cứ (ví dụ dừng train giữa chừng do timeout Colab, mất kết
  nối, hoặc một job giám sát kiểm tra "checkpoint mới nhất" ngay sau khi nó vừa ghi ra), kết quả sẽ
  cho điểm cực thấp (score ~0.455, PSNR ~12 dB) dù model thực chất đang ở trạng thái tốt nhất tính
  đến thời điểm đó (score ~0.84-0.86 một iteration trước).
- Ngược lại, nếu mốc dừng rơi đúng vào 18000/24000, số liệu thật cho thấy sẽ không gặp vấn đề này —
  vì densification (và do đó opacity reset) đã dừng từ iter=15000.
- Khuyến nghị vận hành: không dùng checkpoint tại đúng bội số của `opacity_reset_interval` nằm
  trong khoảng `[densify_from_iter, densify_until_iter]` làm căn cứ đánh giá cuối cùng hoặc để dừng
  sớm; nên chờ thêm ít nhất 1 checkpoint (1000 iteration) sau mốc reset trước khi ghi nhận kết quả,
  hoặc chọn checkpoint có `score` tốt nhất trong lịch sử thay vì checkpoint mới nhất theo thời gian.

![Score theo iteration, đánh dấu 2 điểm crash tại opacity reset](figures/02_score_opacity_reset_crash.png)
