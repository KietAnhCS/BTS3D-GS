# Kịch bản 45 phút — BOT 07 (SADGSX): Optimizer / LR / opacity reset & Bộ lọc 3D chống alias

Bao gồm hai slide: `part45_sadgsx_09.tex` và `part45_sadgsx_10.tex`.
Chương sách đi kèm: [`DOCS/BOOK/18-sadgs-co-che-chuyen-sau/07-optimizer-lr-opacity-reset-va-3d-filter.md`](../../../BOOK/18-sadgs-co-che-chuyen-sau/07-optimizer-lr-opacity-reset-va-3d-filter.md).

---

## Slide `part45_sadgsx_09`: Optimizer, lịch learning rate & reset opacity trong SADGS

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này trả lời câu hỏi: SADGS điều khiển quá trình tối ưu khác 3DGS gốc ở chỗ nào. Có bốn điểm. Thứ nhất, lịch learning rate cho vị trí vẫn là nội suy log-tuyến tính từ 1,6 nhân 10 mũ trừ 4 xuống 1,6 nhân 10 mũ trừ 6 qua 30 nghìn bước — nhưng khi đọc `training_setup` dòng 325 đến 328, nhóm em phát hiện code chỉ truyền `lr_delay_mult` mà quên truyền `lr_delay_steps`, nên tham số này giữ mặc định bằng 0 và **nhánh warm-up không bao giờ chạy**. Thứ hai, năm nhóm tham số còn lại giữ learning rate hằng số suốt 30 nghìn bước, trong đó opacity, scaling và rotation đều được **nhân đôi** so với 3DGS — chính comment trong source cũng ghi lại giá trị cũ. Thứ ba, optimizer mặc định là chế độ `hybrid`: Adam kèm `amsgrad` cho hình học, còn Spherical Harmonics bậc cao — nhóm tham số nặng nhất với 45 hệ số mỗi Gaussian — dùng `SparseGaussianAdam`, chỉ cập nhật những Gaussian đang hiển thị. Thứ tư, và đây là điểm thú vị nhất: reset opacity. Nhìn vào hình bên phải, mỗi 3000 iteration opacity bị kéo tụt đúng 10 lần rồi hồi phục dần, tạo ra mẫu răng cưa này. Điều đáng nói là trong code, hệ số bù của bộ lọc 3D được nhân vào rồi chia ngược ra nên **triệt tiêu hoàn toàn** — kết quả cuối cùng chỉ là phép nhân opacity với 0,1. Khác hẳn 3DGS gốc vốn kẹp về hằng số 0,01, SADGS **giữ nguyên thứ tự** độ đục giữa các Gaussian, nên phép prune ở ngưỡng 0,1 chọn lọc chính xác hơn nhiều.

**Chuyển tiếp:** Đường đứt nằm ngang trong hình chính là ngưỡng prune 0,1 — và slide tiếp theo sẽ cho thấy còn một cơ chế thứ hai cũng đẩy Gaussian xuống dưới ngưỡng đó, lần này xuất phát từ lý thuyết lấy mẫu chứ không phải từ lịch huấn luyện.

**Nếu bị hỏi:**

- *"Tại sao lại tăng gấp đôi learning rate của opacity, scaling, rotation?"* — Vì SADGS reset opacity theo phép **nhân** 0,1 mỗi 3000 bước. Opacity phải kịp hồi phục trong 3000 bước, nếu không thì sau n chu kỳ nó chỉ còn 0,1 mũ n nhân giá trị ban đầu — hai chu kỳ đã giảm 100 lần. `opacity_lr = 0.05` so với 0,025 của 3DGS (`arguments/__init__.py:76`) chính là để bù cho điều đó. Scaling và rotation tăng theo để hình học hội tụ sớm, cho tín hiệu $\eta$ của densify rõ ràng từ giai đoạn đầu.

- *"Hàm `optimizer_step` giãn tần suất cập nhật 1 / 32 / 64 có thực sự chạy không?"* — Không, ở cấu hình mặc định. Hàm đó nằm ở `gaussian_model.py:357-378`, nhưng `train.py:422-424` chỉ gọi nó khi `optimizer_type == "default"`. Mặc định là `"hybrid"` (`arguments:128`), và nhánh hybrid ở `train.py:432-437` gọi `.step()` mỗi iteration bình thường.

- *"Bậc SH tăng dần như 3DGS chứ?"* — Không. `train.py:216-217` gọi `oneupSHdegree()` với điều kiện `iteration % 1 == 0`, tức **mỗi iteration**, nên bậc SH đạt tối đa 3 ngay tại iteration thứ 3, thay vì tại 3000 như 3DGS gốc. Cơ chế warm-up SH thực chất đã bị vô hiệu hoá; thay vào đó SADGS kiềm chế bằng learning rate rất nhỏ cho `f_rest`, chỉ 2,5 nhân 10 mũ trừ 4.

---

## Slide `part45_sadgsx_10`: Bộ lọc 3D chống alias — chặn dưới kích thước Gaussian theo Nyquist

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Vấn đề đặt ra rất cụ thể: một Gaussian 3D có thể co nhỏ tuỳ ý, và khi camera lùi xa, hình chiếu của nó xuống ảnh nhỏ hơn một pixel — vượt giới hạn Nyquist của lưới pixel, gây nhấp nháy và moiré. Lời giải theo lý thuyết lấy mẫu là tiền lọc, và may mắn là tích chập Gaussian với Gaussian vẫn là Gaussian nên có thể lọc giải tích, không cần mipmap. SADGS làm ba bước. Bước một, tính ngưỡng lọc: `filter_3D` bằng độ sâu z chia tiêu cự f nhân căn 0,2 — trong đó z là độ sâu tới camera **gần nhất** nhìn thấy Gaussian, còn f là tiêu cự **lớn nhất**, tức camera nét nhất. Ý nghĩa: z chia f chính là kích thước thế giới của một pixel tại độ sâu đó, nên `filter_3D` xấp xỉ 0,447 lần một pixel quy về 3D — đúng là giới hạn Nyquist chiếu ngược vào không gian. Bước hai, scale hiệu dụng bằng căn của s bình phương cộng filter bình phương, tức cộng phương sai vào mỗi trục: Gaussian nhỏ bị **kéo lên sàn** filter, Gaussian lớn không bị đụng chạm. Bước ba là điểm mà nhiều người bỏ sót: nếu chỉ phình scale mà giữ nguyên opacity thì tổng năng lượng tăng lên, vật thể xa sẽ sáng và đậm hơn thực tế. Nên phải nhân thêm hệ số `coef` bằng căn của tỉ số hai định thức, để bảo toàn tích phân. Hình bên phải chính là hệ số đó: với r bằng filter chia s, ở trường hợp đẳng hướng ta có coef bằng một cộng r bình phương, mũ trừ ba phần hai. Đọc hình: khi r bằng 1, coef còn khoảng 0,354; khi r bằng 2, chỉ còn 0,089. Gaussian nhỏ hơn pixel sẽ **mờ hẳn đi** thay vì nhấp nháy. Và đây chính là chỗ bộ lọc khớp vào cơ chế densify theo $\eta$: chia theo $\eta$ làm s nhỏ dần, nhưng khi s tụt xuống dưới filter thì opacity hiệu dụng sụp đổ và Gaussian bị prune — bộ lọc 3D đóng vai trò **phanh vật lý** cho densification.

**Chuyển tiếp:** Cần nói rõ một điều: cờ `--compute_3d_filter` mặc định là `False`, nên đây là cơ chế tuỳ chọn, không nằm trong đường chạy mặc định của SADGS — phần tiếp theo sẽ quay lại các cơ chế luôn hoạt động.

**Nếu bị hỏi:**

- *"Vì sao lấy camera gần nhất chứ không phải xa nhất?"* — Vì lấy `distance` nhỏ nhất là chọn ràng buộc **chặt nhất** qua mọi view: nếu có view nào nhìn rất gần, Gaussian được phép nhỏ tương ứng, và ở các view xa hơn thì khâu chiếu 2D tự lọc thêm. Ngược lại `focal_length` lấy **lớn nhất** (`gaussian_model.py:247-248`) để chuẩn theo camera phân giải cao nhất, tránh làm mờ quá mức ảnh nét nhất trong tập train. Gaussian không camera nào thấy thì nhận giá trị lớn nhất trong nhóm nhìn thấy (`:250`) — tức chịu mức lọc mạnh nhất, an toàn.

- *"Hệ số căn 0,2 ở đâu ra?"* — Đó là hằng số hard-code ở `gaussian_model.py:254`, ngay phía trên có hai dòng TODO của chính tác giả: `#TODO remove hard coded value` và `#TODO box to gaussian transform`. Về ý nghĩa, nó quy đổi hộp lấy mẫu rộng một pixel thành độ lệch chuẩn Gaussian tương đương. Đây là điểm thừa kế từ Mip-Splatting chứ không phải đóng góp riêng của SADGS.

- *"Bộ lọc có ảnh hưởng tới quyết định densify không?"* — Không, và đây là điểm dễ hiểu nhầm. Điều kiện split và clone dùng `get_scaling` **thô** (`gaussian_model.py:683` và `:896-897`), không dùng `get_scaling_with_3D_filter`. Bộ lọc chỉ được tiêu thụ ở khâu render — `opacity` tại `gaussian_renderer/__init__.py:63` và `scales` tại `:74`. Vòng phản hồi giữa hai cơ chế là **gián tiếp**, đi qua opacity rồi tới prune ở ngưỡng 0,1, chứ không đi qua tiêu chí densify.

- *"Trong pha densify, `filter_3D` có được cập nhật theo không?"* — Không. `train.py:405-409` chỉ chạy lại `compute_3D_filter` khi iteration chia hết 100 **và** lớn hơn `densify_until_iter` bằng 15 nghìn **và** còn cách cuối hơn 100 bước. Nghĩa là suốt toàn bộ pha densification, Gaussian con kế thừa nguyên `filter_3D` của cha (`:728, :886, :907`) dù bản thân nó đã nhỏ hơn — tỉ số r tăng, coef giảm, nên con mới sinh ra đã bị mờ sẵn.
