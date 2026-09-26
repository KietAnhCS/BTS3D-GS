# Kịch bản 45 phút — SADGS Bot 02: Tỉ số cấu trúc $\eta$

> Slide nguồn: `DOCS/Slides67/Slides45min/part45_sadgsx_02.tex` (1 frame).
> Code nguồn: `SADGS/utils/freq_utils.py:116-178, 313-348, 395-409`, `SADGS/train.py:337-353`,
> `SADGS/arguments/__init__.py:148-150`, `SADGS/scene/gaussian_model.py:699`.

---

## Slide: SADGS — Tỉ số cấu trúc $\eta$: Gaussian có "quá to" so với texture?

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này trả lời đúng một câu hỏi: một Gaussian có đang quá to so với chi tiết ảnh mà nó phải tái tạo hay không? Câu hỏi đó không có lời giải trong không gian ba chiều, vì cùng một quả Gaussian sẽ là thô khi camera đứng gần và là mịn khi camera lùi ra xa — nên SADGS đem cả hai vế về mặt phẳng ảnh, đơn vị pixel. Vế thứ nhất, bên trái slide: ta lấy ba trục chính của ellipsoid, xoay về camera space rồi chiếu qua Jacobian phối cảnh, được ba vector hai chiều; độ dài của chúng ở tâm ảnh rút gọn thành f nhân s chia z — tỉ lệ nghịch với khoảng cách. Vế thứ hai: từ structure tensor của ảnh gốc, ta lấy trị riêng lớn nhất lambda-một, là năng lượng tần số cực đại, rồi nghịch đảo căn của nó ra lambda-min — bước sóng nhỏ nhất mà texture ở chỗ đó chứa. Chia hai vế cho nhau ta được eta, một phân số không thứ nguyên: pixel chia pixel. Nhìn heatmap bên phải: eta lớn hơn một là Gaussian rộng hơn cả chu kỳ texture, nó sẽ xoá mất chi tiết, phải SPLIT; eta nhỏ hơn một phần mười là Gaussian nhỏ hơn chi tiết hàng chục lần, dư thừa, ứng viên PRUNE. Và vì eta đổi theo góc nhìn, SADGS không quyết định theo một view, mà đòi tám mươi phần trăm số view đồng thuận.

**Chuyển tiếp:** Có ba phiếu bầu high, mid, low như vậy cho mỗi view — slide sau sẽ xem chúng được cộng dồn và chốt thành quyết định split hay prune như thế nào.

---

### Nếu bị hỏi

**Hỏi: Vì sao ngưỡng lại đúng bằng 1.0, có phải chọn tuỳ tiện không?**
Không ạ. Một Gaussian khi tô lên ảnh hoạt động như bộ lọc thông thấp, nên biên độ texture còn lại bằng A-in nhân e mũ trừ một phần hai, nhân bình phương độ rộng, nhân bình phương tần số. Thay omega bằng hai pi chia lambda-min thì toàn bộ rút gọn về $A_{out} = A_{in}e^{-2\pi^2\eta^2}$ — chỉ còn eta, độ dài trục và bước sóng đều biến mất. Tại eta bằng 1, hệ số đó là e mũ trừ 2 pi bình, khoảng ba phần tỉ, tức chi tiết bị xoá sạch. Nên 1.0 đã là mốc khoan dung chứ không hề khắt khe. Trong code, `TAU_HIGH = 1.0` và `TAU_LOW = 0.1` hard-code tại `freq_utils.py:395-396`.

**Hỏi: Tại sao là ba kênh eta? Có phải ba kênh màu RGB không?**
Không ạ, ba kênh màu đã bị cộng gộp từ trước rồi, ở bước structure tensor kiểu Di Zenzo — `loss_utils.py:206-208`. Ba kênh trong `eta_3ch` là ba **trục chính** của ellipsoid, tức ba cột của R nhân diag(s) sau khi chiếu, trả về tensor $[N,3,2]$ ở `freq_utils.py:178`. Giữ riêng ba con số đó là thứ duy nhất cho phép split bất đẳng hướng: ở `gaussian_model.py:699`, số lát cắt tính riêng từng trục theo $k_j = \lceil\sqrt{\max(\eta_j,1)}\rceil$. Nếu chỉ có một eta vô hướng thì ta chỉ chia đều được cả ba chiều — đúng hành vi của 3DGS gốc, là thứ SADGS muốn vượt qua.

**Hỏi: Đã có $\Sigma_{2D}$ từ rasterizer rồi, sao không dùng luôn mà phải chiếu lại ba trục?**
Vì $\Sigma_{2D}$ chỉ cho một ellipse 2D — một hình dạng đã trộn, không nói được Gaussian vi phạm tần số theo hướng nào trong ba trục riêng của chính nó. Chi phí cũng không phải lý do, vì SADGS vẫn dùng lại kết quả của rasterizer chứ không chiếu lại từ đầu: tensor `cov2D` bảy cột đã mang sẵn means2D, depth và transmittance (`forward.cu:241-247`), hàm chỉ dựng thêm Jacobian và nhân ba cột — toàn bộ chạy trong `torch.no_grad()` ở `freq_utils.py:120`, không tham gia backward.

**Hỏi: Có mode tính eta nào khác không?**
Có ạ, `eta_compute_mode` mặc định là `"wavelength"` (`arguments/__init__.py:150`), còn mode `"projection"` ở `freq_utils.py:350-373` chiếu structure tensor lên đúng hướng từng trục dưới dạng toàn phương $\sqrt{a_j^\top S a_j}$. Mode đó có tính hướng thật — với một đường ngang thì trục dọc cho eta cao, trục ngang cho eta thấp — nhưng nhạy hơn với nhiễu, và hai ngưỡng 1.0 / 0.1 vẫn dùng chung nên đổi mode là phải chỉnh lại ngưỡng.

---

### Ghi chú cho người trình bày

- Chỉ vào **heatmap** khi nói tới ba vùng: hai đường đẳng trị là đường **thẳng chéo** vì cả hai trục đều thang log — nhấn mạnh rằng không có một "kích thước Gaussian xấu" cố định nào cả, tất cả phụ thuộc texture bên dưới.
- Nếu còn thời gian, thêm một câu: trọng số transmittance ở `freq_utils.py:380` hiện là phép nhân với vector toàn số 1 (`freq_utils.py:238`), nên transmittance chỉ còn tác dụng ở mặt nạ lọc chứ chưa phải trọng số liên tục như tên biến gợi ý.
- Nếu bị ép rút còn ~25 giây, bỏ phần Jacobian, nói gọn: "Chiếu ba trục Gaussian ra pixel, chia cho bước sóng texture cục bộ cũng tính bằng pixel, được một tỉ số không thứ nguyên eta. Lớn hơn 1 thì Gaussian đang xoá mất chi tiết — SPLIT; nhỏ hơn 0.1 thì dư thừa — PRUNE; và phải 80% số view đồng ý mới hành động."
