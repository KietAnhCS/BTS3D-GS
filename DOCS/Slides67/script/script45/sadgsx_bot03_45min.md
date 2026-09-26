# Kịch bản 45 phút — Bot 03 (SADGS-X, slide `part45_sadgsx_03`)

> Chủ đề: Thống kê $\eta$ online đa view & tỉ lệ nhất quán (multiview consistency ratio).
> Slide nguồn: `DOCS/Slides67/Slides45min/part45_sadgsx_03.tex` (1 frame, 2 hình: `03_ratio_hist.png`, `03_timeline_reset.png`).
> Chương sách đi kèm: `DOCS/BOOK/18-sadgs-co-che-chuyen-sau/03-thong-ke-eta-da-view.md`.

---

## Slide: SADGS — Tỉ lệ nhất quán đa view cho densify/prune

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide trước ta đã có đại lượng $\eta$ — mức vi phạm tần số của một Gaussian. Nhưng $\eta$ được đo trên mặt phẳng ảnh, nên nó phụ thuộc góc nhìn: cùng một Gaussian, ở view gần thì trục chiếu dài hàng chục pixel nên $\eta$ rất lớn, ở view xa thì co lại dưới một pixel nên $\eta$ gần như bằng không. Vì vậy SADGS không bao giờ quyết định trên một view. Cứ mỗi 10 iteration, hàm `update_freq_stats_online` trong `utils/freq_utils.py` dòng 181 chạy một lần, đo $\eta$ và tích luỹ bộ đếm cho từng Gaussian — nhưng chỉ với những mẫu qua được chuỗi ba bộ lọc: transmittance lớn hơn 0, opacity lớn hơn 0.05, và gradient view-space lớn hơn 2 nhân 10 mũ trừ 5. Tầng gradient là tầng quan trọng nhất: nó bảo đảm ta chỉ đếm ở những view mà hình học **đang thực sự sai**. Mỗi quan sát được phân loại theo trục xấu nhất trong ba trục: gọi là HIGH nếu $\eta$ vượt 1.0 — tức Gaussian dài hơn bước sóng kết cấu, chắc chắn gây răng cưa; gọi là LOW nếu $\eta$ dưới 0.1 — tức nhỏ hơn chi tiết ảnh mười lần, tức là dư thừa. Rồi cứ mỗi 100 iteration, `train.py` dòng 337 đến 353 mới lấy hai bộ đếm đó chia cho `accum_view_count` để ra `high_ratio` và `low_ratio`. Dùng tỉ lệ chứ không dùng số đếm thô, vì mỗi Gaussian được nhìn ở số view rất khác nhau — chia cho mẫu số sẽ chuẩn hoá tất cả về thang không đến một, nên một ngưỡng duy nhất dùng được cho cả đám mây điểm. Ngưỡng đó là 0.8, đọc ra tiếng Việt là "ít nhất tám mươi phần trăm số view phải đồng ý". Nhìn hình trên bên phải: phân bố `high_ratio` và `low_ratio` trên bốn nghìn Gaussian, và vạch đứt 0.8 chỉ cắt đúng phần đuôi — đại đa số Gaussian không bị đụng tới, đó chính là cơ chế giữ số điểm không bùng nổ. Hai chi tiết cuối cần nhấn. Thứ nhất, SPLIT còn phải kèm điều kiện gradient trung bình lớn hơn 10 mũ trừ 5; comment ngay trong `train.py` dòng 331 ghi nguyên văn "we must use this to prevent 7 million points". Thứ hai, kích thước Gaussian con được định hình bằng `max_eta_3ch`, tức cực đại qua các view, chứ không phải trung bình — vì con phải đủ nhỏ để không aliasing ở **mọi** view, chia một lần là xong. Và như hình dưới cho thấy, ngay sau mỗi lần densify, cả chín bộ đếm bị `zero_()` ở `train.py` dòng 377 đến 386, nên `accum_view_count` có dạng răng cưa — mẫu số luôn là một cửa sổ trượt khoảng mười mẫu, chứ không tích luỹ từ đầu huấn luyện.

**Chuyển tiếp:** Đã biết Gaussian nào cần chia và chia theo trục nào bị vi phạm, phần tiếp theo sẽ xem SADGS thực sự sinh Gaussian con ra sao — cơ chế anisotropic split.

---

## Nếu bị hỏi

**H1: Tại sao lại là 0.8 mà không phải 0.5?**
Vì mẫu số rất nhỏ. Mỗi chu kỳ densify chỉ cho tối đa 100 chia 10, tức 10 mẫu cho một Gaussian (`train.py:275` và `densification_interval = 100` ở `arguments/__init__.py:87`), lại còn bị bộ lọc gradient cắt bớt. Với $M = 10$ và $p = 0.8$, nửa khoảng tin cậy 95% là $1.96\sqrt{0.8\cdot0.2/10} \approx 0.25$. Ngưỡng thấp như 0.5 sẽ cho rất nhiều split sai. Đặt cao ở 0.8, cộng thêm điều kiện gradient, là cách bù cho việc ước lượng còn nhiễu. Giá trị nằm ở `arguments/__init__.py:148-149`, cả `split_ratio_threshold` lẫn `prune_ratio_threshold` đều là 0.8.

**H2: Vì sao phải reset bộ đếm, giữ lại để có thống kê tốt hơn không được à?**
Không được, vì hai lý do. Một là sau khi split thì hình học đã đổi hẳn — Gaussian cha biến mất, con có scale mới, nên `high_ratio` cũ vô nghĩa và sẽ khiến chia tiếp ngay dù con đã đủ nhỏ. Hai là `max_eta_3ch` cập nhật bằng `torch.max` chạy dần (`freq_utils.py:391-392`), nó đơn điệu không giảm; không reset thì nó chỉ tăng và cuối cùng mọi Gaussian đều vượt ngưỡng 1.0. Ngoài ra, điểm mới sinh được nối bằng zeros (`gaussian_model.py:625-636`) nên tự động có `valid_mask` bằng False và không thể bị prune ngay ở chu kỳ kế tiếp.

**H3: `eta_mid_count` và `eta_high_sum_3ch` dùng vào việc gì?**
Nói thẳng là trong cấu hình mặc định hiện tại chúng không ảnh hưởng gì đến huấn luyện. `eta_high_sum_3ch` được dùng để tính `avg_high_eta_3ch` ở `train.py:348-350`, nhưng biến đó chỉ được truyền vào `expand_undersized_gs` — mà lời gọi này đang bị comment (`train.py:355-359`). Hàm chia nhận `max_eta_3ch` chứ không phải trung bình. Tương tự, `accum_eta` và `accum_weights_valid` được ghi mỗi chu kỳ nhưng không đọc ở nhánh quyết định; riêng `accum_weights_valid` hiện chỉ là bản sao của `accum_view_count` vì `weights_valid` được gán bằng `torch.ones_like` (`freq_utils.py:238`), tức trọng số transmittance đang bị vô hiệu hoá. Chúng là bộ đếm chẩn đoán, để sẵn cho các biến thể tương lai.

**H4: Việc lấy mẫu structure tensor có gì đặc biệt?**
Có — code không lấy mẫu tại tâm Gaussian mà tại một điểm ngẫu nhiên rút từ chính phân bố 2D của nó, bằng phân tích Cholesky của hiệp phương sai màn hình (`freq_utils.py:255-292`): $L_{11}=\sqrt{\sigma_{xx}}$, $L_{21}=\sigma_{xy}/L_{11}$, $L_{22}=\sqrt{\sigma_{yy}-L_{21}^2}$, rồi jitter bằng $L\varepsilon$ với $\varepsilon$ chuẩn. Lý do: $\eta$ phải phản ánh kết cấu trên toàn bộ vết loang của Gaussian, không phải một pixel tâm — một Gaussian dẹt vắt qua biên vật thể có tâm rơi vào vùng phẳng sẽ bị bỏ sót nếu chỉ lấy mẫu ở tâm. Cái giá là phương sai từng mẫu tăng, và đó chính là lý do thứ hai bắt buộc phải tích luỹ nhiều view trước khi quyết định.
