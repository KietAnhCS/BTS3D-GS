# Kịch bản 45 phút — BOT 06 (SADGS chuyên sâu)

Phụ trách hai slide: `part45_sadgsx_07.tex` và `part45_sadgsx_08.tex` (thứ tự 7 và 8 trong khối 15 slide SADGS chuyên sâu của `main45.tex`).

Chủ đề: **(A)** `densify_and_prune_structgs` — vòng densify + prune hợp nhất; **(B)** `final_prune_structgs` — tầng tỉa cuối theo pruning score và nhất quán đa view, **hiện đang bị tắt trong code**.

---

## Slide SADGSX-07: `densify_and_prune_structgs` — một vòng sinh + xoá hợp nhất

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Ở slide này chúng em đi vào hàm điều phối toàn bộ việc thay đổi số lượng Gaussian của SADGS. Cứ mỗi một trăm iteration, tính từ iteration năm trăm, `train.py` dòng 362 gọi đúng một hàm duy nhất: `densify_and_prune_structgs`. Điểm em muốn nhấn mạnh trước tiên là **thứ tự thật trong code**, nhìn vào sơ đồ khối bên phải: clone trước, rồi split, rồi mới prune, và bước cuối cùng không phải prune mà là áp trần opacity ở mức không phẩy tám. Vì prune chạy **sau** khi số Gaussian đã tăng, nên các Gaussian con vừa sinh ra cũng bị soi ngay bởi ngưỡng opacity, ngưỡng bán kính màn hình và ngưỡng scale — đây chính là cơ chế tự hãm giữ cho số Gaussian bão hoà thay vì bùng nổ. Về mặt nạ sinh: clone và split là hai tập **bù nhau tuyệt đối**, chia theo bán trục lớn nhất so với `dense` nhân `extent`, với `dense` bằng không phẩy không không một — nên một Gaussian không bao giờ vừa clone vừa split trong cùng một vòng. Còn mặt nạ xoá là **hợp OR của bốn tiêu chí**: opacity nhỏ hơn không phẩy một, bán kính màn hình lớn hơn hai mươi pixel, scale lớn hơn một phần mười extent, và tiêu chí thứ tư là đóng góp của SADGS — `low_ratio` lớn hơn không phẩy tám, tức Gaussian bất nhất về tần số trên phần lớn các view. Ba con số cần lưu ý. Thứ nhất, `min_opacity` thật là **không phẩy một**, gấp hai mươi lần mức không phẩy không không năm của 3DGS gốc — chính comment trong code ở `train.py` dòng 364 thừa nhận điều đó. Thứ hai, hai tiêu chí kích thước chỉ bật **sau iteration ba nghìn**, trước đó `max_screen_size` là `None` nên chúng bị bỏ qua hoàn toàn. Thứ ba, `pruning_score` được truyền là `None`, nên nhánh lấy mẫu multinomial chỉ xoá năm mươi phần trăm ứng viên **không bao giờ chạy** — code rơi vào nhánh `else`, xoá cứng toàn bộ.

**Chuyển tiếp:** Nhưng nếu nhánh multinomial đã chết, thì còn tầng tỉa thứ hai — `final_prune_structgs` — liệu có sống không? Slide sau trả lời câu đó.

**Nếu bị hỏi:**

- *Hỏi: `importance_score` là gì, tại sao ngưỡng lại là 0.5?*
  Trả lời: Nó **không phải** một điểm số quan trọng theo nghĩa FastGS. `train.py:368` truyền vào `gaussians.accum_view_count` — một bộ đếm **số view** đã nhìn thấy Gaussian trong cửa sổ 100 iteration, nhận giá trị nguyên 0, 1, 2… Ngưỡng `importance_score_threshold = 0.5` (`arguments/__init__.py:124`) vì thế tương đương đúng điều kiện "đã được ít nhất một camera quan sát". Nó là bộ lọc tính hợp lệ, không phải xếp hạng.

- *Hỏi: Gaussian mới sinh ra có bị prune ngay trong cùng vòng không?*
  Trả lời: Có, nhưng chỉ bởi ba trên bốn tiêu chí. `gaussian_model.py:1021-1025` nối thêm các giá trị 0 vào `full_prune_mask` cho đúng số Gaussian mới, nên chúng **miễn nhiễm với tiêu chí multi-view** ở chính vòng đó — hợp lý vì chưa camera nào quan sát nên `low_ratio` chưa có nghĩa. Còn ba tiêu chí opacity, bán kính và scale thì được tính lại trên tensor đã mở rộng nên vẫn áp dụng đầy đủ.

- *Hỏi: Xoá Gaussian thì trạng thái Adam xử lý thế nào?*
  Trả lời: `_prune_optimizer` (`gaussian_model.py:503-527`) duyệt **cả** `self.optimizer` lẫn `self.shoptimizer` — vì SADGS mặc định `optimizer_type = "hybrid"`, hệ số SH có optimizer riêng. Với mỗi nhóm, nó cắt `exp_avg`, `exp_avg_sq`, và `max_exp_avg_sq` nếu có amsgrad, rồi `del opt.state[p]` trước khi tạo `nn.Parameter` mới. Bước `del` là bắt buộc vì khoá của `opt.state` chính là đối tượng tensor tham số; không xoá thì state cũ dài N sẽ lệch chiều với tham số mới dài N phẩy ngay ở bước Adam kế tiếp.

---

## Slide SADGSX-08: Tỉa cuối `final_prune_structgs` — công thức có, lời gọi đang tắt

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này nói về tầng tỉa thứ hai, và đây cũng là phát hiện thẳng thắn nhất mà nhóm em muốn báo cáo. Toàn bộ hàm `final_prune_structgs`, ở `scene/gaussian_model.py` dòng 1059 đến 1066, chỉ vỏn vẹn bốn dòng thân, và là một phép OR **tất định**: xoá Gaussian nếu opacity nhỏ hơn tau alpha, **hoặc** nếu điểm số s lớn hơn tau s. Trong đó tau alpha bằng không phẩy một, do lời gọi quyết định; còn tau s bằng **không phẩy chín và được hard-code cứng ngay trong thân hàm**, không có tham số dòng lệnh nào chỉnh được — khác hẳn `prune_ratio_threshold` vốn khai báo đàng hoàng trong `arguments`. Đại lượng s ở đây là `pruning_score`, đo **độ bất nhất** tái dựng qua nhiều view: giá trị càng lớn thì Gaussian càng sai lệch giữa các góc nhìn. Nhìn hình bên phải, ngưỡng không phẩy chín chỉ cắt đúng cái đuôi phải của phân bố — tức đây là một thiết kế rất bảo thủ, chỉ Gaussian sai ở gần như mọi view mới bị loại. Điểm khác biệt so với prune trong vòng densify: ở đó có ngân sách `remove_budget` bằng một nửa, cộng với lấy mẫu `multinomial`, tức có phanh; còn ở đây cắt dứt khoát, không phanh. Và bây giờ là phần quan trọng nhất: **lời gọi hàm này đang bị comment** ở `train.py` dòng 418 và 419. Hơn thế nữa, nguồn sinh ra điểm số — hàm `compute_gaussian_score_structgs` — khi grep toàn bộ repo chỉ ra đúng một kết quả, chính là dòng comment đó; hàm **không hề được định nghĩa ở đâu cả**. Và nếu ai đó bỏ comment mà để `pruning_score` mặc định là `None` thì dòng 1064 sẽ so sánh `None` lớn hơn không phẩy chín và văng `TypeError`, vì hàm không kiểm tra None. Cái thực sự chạy tại iteration bốn nghìn và tám nghìn chỉ còn mỗi nhánh opacity nhỏ hơn không phẩy một. Hệ quả là các **floater** — Gaussian có opacity cao nhưng chỉ đúng ở vài góc nhìn — lọt lưới hoàn toàn, khiến mô hình cuối to hơn mức cần thiết và sinh vệt sương ở view mới.

**Chuyển tiếp:** Đó là toàn bộ cơ chế điều khiển mật độ của SADGS cùng những chỗ nó chưa được bật; phần tiếp theo chuyển sang khâu render và chấm điểm chất lượng.

**Nếu bị hỏi:**

- *Hỏi: Làm sao khẳng định nhánh đó bị tắt chứ không phải chưa từng có?*
  Trả lời: Bằng dấu vết còn lại trong chính khối code. `train.py:413-414` vẫn gọi `scene.getTrainCameras().copy()` rồi `sampling_cameras(...)` để dựng `camlist` gồm 60 camera lấy theo FPS (`utils/freq_utils.py:12`, `num_cams=60`). Nhưng sau đó **không dòng nào dùng `camlist`** — nó chỉ có ý nghĩa nếu hai dòng 418, 419 được bật. Đó là bằng chứng trực tiếp rằng nhánh multi-view từng chạy và đã bị comment đi.

- *Hỏi: Muốn bật lại thì cần sửa những gì?*
  Trả lời: Ba việc theo thứ tự. Một, cài đặt `compute_gaussian_score_structgs` trả về 6 giá trị, phần tử thứ hai là s thuộc đoạn 0–1 theo nghĩa **độ bất nhất**, để khớp chiều so sánh dấu lớn hơn ở dòng 1064. Hai, thêm bảo vệ None: nếu `pruning_score` là `None` thì `scores_mask` phải là tensor toàn `False`. Ba, đưa hằng số 0.9 ra `arguments/__init__.py` cạnh `prune_ratio_threshold` ở dòng 149 để có thể quét ngưỡng. Nếu muốn bật luôn cả nhánh multinomial trong vòng densify thì thêm việc thứ tư: sửa `gaussian_model.py:1036`, tensor `padded_importance` đang được tạo **trên CPU** trong khi dòng ngay sau gán vào nó dữ liệu CUDA — sẽ lỗi thiết bị.

- *Hỏi: Tín hiệu nhất quán đa view có còn sống ở chỗ nào không?*
  Trả lời: Còn đúng một đường. `low_ratio` bằng `eta_low_count` chia `accum_view_count` (`train.py:341-342`), so với `prune_ratio_threshold = 0.8`, rồi đi vào `custom_prune_mask` của vòng densify (`train.py:353, 371`). Nhưng nó đo bất nhất về **tần số eta**, không đo bất nhất về **màu tái dựng** — nên nó không thay thế được vế `s > 0.9` trong việc khử floater. Cần nói rõ: cả hai đường dùng `pruning_score` trong repo hiện đều vô hiệu, đường này là thứ duy nhất còn hoạt động.
