## Tốc độ hội tụ và lợi ích biên giảm dần (scene HCM0539, 30000 iter)

![Convergence và diminishing returns](figures/05_convergence_diminishing_returns.png)

### Số liệu đo trực tiếp từ `history.csv`

| Iteration | score | % so với score cuối (iter=30000) |
|---|---|---|
| 10000 | 0.8620 | **99.31%** |
| 15000 | 0.8705 | 100.28% |
| 17000 | 0.8714 | 100.38% (điểm score cao nhất toàn bộ quá trình train) |
| 18000 | 0.8712 | 100.36% |
| 30000 (cuối) | 0.8681 | 100.00% |

Score cuối cùng tại iter=30000 là **0.8680734**. Score tại iter=10000 đã là 0.8620455, tức đã đạt **99.31%** giá trị cuối cùng chỉ sau 1/3 tổng số iteration (10000/30000). Đáng chú ý hơn: score không tăng đơn điệu đến cuối — nó đạt **đỉnh tại iter=17000 (score=0.87141)**, cao hơn cả giá trị tại iter=30000 khoảng 0.38%, rồi dao động đi ngang và giảm nhẹ trong suốt 13000 iteration còn lại (17000→30000). Nói cách khác, 13000 iteration cuối (43% tổng thời gian train, tương ứng khoảng 1600–1800 giây thực đo theo cột `elapsed_s`) không mang lại cải thiện ròng nào về score — thậm chí kết quả cuối thấp hơn đỉnh đã đạt được ở giữa quá trình train.

### Ba giai đoạn hội tụ (subplot b)

- **Giai đoạn 1 — tăng nhanh (iter < ~8000):** `|d_score|` (chênh lệch mỗi 1000 iter, bỏ qua các điểm outlier opacity-reset) giảm từ ~0.15 (iter 2000) xuống ~0.006 (iter 8000), tức giảm hơn 25 lần chỉ trong 6000 iteration đầu. Đây là giai đoạn "ăn điểm" chính.
- **Giai đoạn 2 — chậm dần (8000–20000):** `|d_score|` tiếp tục giảm xuống bậc 10⁻³–10⁻⁴ (ví dụ 0.0024 tại 9000, 0.0009 tại 10000, xuống ~0.0002–0.0007 quanh 16000–20000). Score vẫn nhích lên nhưng biên độ mỗi 1000 iter đã rất nhỏ.
- **Giai đoạn 3 — bão hòa/đi ngang (>20000):** `|d_score|` dao động trong khoảng 10⁻⁴, không có xu hướng tăng rõ ràng; score thực tế dao động quanh 0.868–0.871 và kết thúc thấp hơn đỉnh đã đạt ở iter=17000.

Ba điểm outlier lớn tại iter=6000, 12000, 18000 (d_score ≈ −0.39, −0.41, +0.41 tương ứng) được chú thích riêng trên subplot (a) — đây là do cơ chế opacity-reset của Gaussian Splatting gây sụt giảm chất lượng render tạm thời rồi phục hồi ngay sau đó (xem thêm mục về opacity reset), không phải nhiễu đo lường hay lỗi hội tụ.

### Nhận định và khuyến nghị

Dữ liệu cho thấy rõ ràng lợi ích biên (marginal gain) của việc train dài hơn 15000–18000 iteration là **rất thấp, thậm chí âm** đối với scene HCM0539 cụ thể này:

- Train đến 15000 iter (50% tổng chi phí) đã đạt 100.28% score so với train đủ 30000 iter — tức là **vượt** kết quả cuối cùng trong khi chỉ tốn một nửa thời gian/compute (khoảng 1963s so với 4191s theo `elapsed_s`, tiết kiệm ~53% thời gian).
- Train đến 17000–18000 iter (57–60% tổng chi phí) cho score cao nhất toàn bộ lần chạy (100.36–100.38%), cao hơn cả kết quả sau khi train đủ 30000 iter.
- 13000 iteration cuối (từ 17000 đến 30000, tức 43% tổng compute) không mang lại giá trị gia tăng — score thậm chí trôi giảm nhẹ, nhiều khả năng do dao động/overfit nhẹ ở giai đoạn bão hòa chứ không phải cải thiện thật.

**Khuyến nghị cụ thể:** với cấu hình và scene tương tự HCM0539, nên đặt điểm dừng sớm (early-stop) ở khoảng **iter 15000–18000** thay vì chạy đủ 30000 iter mặc định. Điều này giúp tiết kiệm 40–43% thời gian/compute train mà không đánh đổi chất lượng — thực tế còn cho kết quả tốt hơn nhẹ so với chạy đủ 30000 iter trong lần đo này. Nếu vẫn muốn giữ ngân sách 30000 iter cho an toàn (ví dụ scene khác có đặc tính hội tụ khác), nên bổ sung cơ chế checkpoint-best (lưu lại checkpoint có score cao nhất trong lịch sử, không chỉ checkpoint cuối) để không mất đi điểm hội tụ tốt nhất như trường hợp iter=17000 ở đây.
