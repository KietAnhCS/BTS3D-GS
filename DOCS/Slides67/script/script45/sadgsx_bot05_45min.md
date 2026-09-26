# Kịch bản nói 45 phút — BOT 05

**Phạm vi:** hai slide `part45_sadgsx_05.tex` (Clone trong SADGS và cách ghép tensor vào optimizer) và
`part45_sadgsx_06.tex` (Giãn giải tích Gaussian dưới cỡ — và trạng thái *bị tắt*).

**Nguồn code:** `SADGS/scene/gaussian_model.py:934-955` (`densify_and_clone_structgs`), `:566-593`
(`cat_tensors_to_optimizer`), `:485-501` (`replace_tensor_to_optimizer`), `:833-864`
(`expand_undersized_gs`), `SADGS/train.py:355-359` (lời gọi bị comment out),
`SADGS/arguments/__init__.py:112-113, 124, 135, 148, 156`, `SADGS/utils/freq_utils.py:348`.

**Chương sách đi kèm:** `DOCS/BOOK/18-sadgs-co-che-chuyen-sau/05-clone-va-expand.md`.

---

## Slide: Clone trong SADGS và cách ghép tensor vào optimizer

**Thời lượng: ~75 giây**

**Lời thuyết trình:**

Chúng ta vừa xem split, giờ sang thao tác còn lại của densify: clone. Điểm đầu tiên cần nhấn mạnh là
trong SADGS, hàm clone **không tự quyết định** nhân bản cái gì. Nếu mở `gaussian_model.py` dòng 934,
các thầy cô sẽ thấy hàm chỉ có đúng một dòng logic: lấy giao của `metric_mask` và `filter` — cả hai đều
được truyền từ ngoài vào. Đây là khác biệt kiến trúc so với 3DGS gốc, nơi điều kiện chọn nằm chôn cứng
bên trong hàm ở dòng 895. Nhờ tách ra như vậy, tiêu chí densify của SADGS có thể đến từ tín hiệu tần số
đa góc nhìn chứ không chỉ từ gradient. Cụ thể, điều kiện là: gradient trung bình vượt ngưỡng hai nhân
mười mũ trừ bốn, **hoặc** Gaussian vi phạm tần số nhất quán qua nhiều view; **và** kích thước phải nhỏ
hơn `args.dense` nhân extent, với `args.dense` bằng không phẩy không không một — chặt hơn 3DGS gốc đúng
mười lần; **và** Gaussian đó phải đã được ít nhất một camera nhìn thấy trong chu kỳ vừa rồi. Điều kiện
thứ ba chính là `metric_mask`, nó chặn việc sinh điểm cho những Gaussian vô hình.

Việc nhân bản thì cực kỳ đơn giản: sao chép nguyên xi sáu tham số. Không dịch vị trí, không chia scale,
không đổi opacity. Bản sao trùng khít một trăm phần trăm với Gaussian cha. Và đây là chỗ nhiều người
thắc mắc: hai điểm chồng nhau hoàn toàn thì làm sao tách ra được? Câu trả lời nằm ở phần dưới slide —
cơ chế ghép tensor vào optimizer.

Khi thêm M điểm mới, ta không chỉ nối tham số. Adam còn giữ hai tensor trạng thái song song là `exp_avg`
và `exp_avg_sq`, tức moment bậc một và bậc hai. Cả ba đều phải nối lên N cộng M. Nhưng chú ý: tham số
thì nhận **bản sao giá trị**, còn hai moment nhận **toàn số không**. Hình bên phải minh hoạ đúng điều
đó — ô đỏ đặc là giá trị copy, ô gạch chéo là số không.

Và chính việc reset moment về không tạo ra hiệu ứng thú vị. Bộ đếm bước `step` của Adam là scalar dùng
chung cho cả tensor, nó không reset. Nên bias-correction vẫn chia cho một trừ beta mũ t với t rất lớn,
gần bằng một. Kết quả là bước cập nhật đầu tiên của điểm mới bằng một trừ beta một, chia căn của một trừ
beta hai — bằng ba phẩy mười sáu lần bước bình thường. Cú hích gấp ba lần này chính là thứ tách hai bản
sao trùng khít ra khỏi nhau. Nếu chúng kế thừa moment của cha thì sẽ đi cùng hướng, cùng tốc độ, và mãi
mãi không tách.

Một chi tiết kỹ thuật nhỏ nhưng quan trọng: không thể gán tensor mới vào chỗ cũ, vì khoá của
`optimizer.state` chính là đối tượng `Parameter`. Bắt buộc phải xoá bản ghi cũ, tạo `nn.Parameter` mới,
rồi gắn lại state — đúng ba dòng, đúng thứ tự.

**Chuyển tiếp:** Clone và split đều làm tăng số Gaussian. Nhưng SADGS còn thiết kế một thao tác thứ ba
hoàn toàn khác — sửa thẳng tham số mà không thêm điểm nào. Slide sau nói về nó, và về một sự thật khá
bất ngờ.

---

### Nếu bị hỏi

**Hỏi: Tại sao clone của SADGS không dịch vị trí bản sao như tài liệu 3DGS thường mô tả?**
Vì code thật không dịch. `gaussian_model.py:937` gán thẳng `new_xyz = self._xyz[selected_pts_mask]`,
không cộng nhiễu, không cộng gradient. Bản 3DGS gốc ở dòng 899 cũng làm y hệt. Việc dịch vị trí chỉ có ở
`densify_and_split`, nơi lấy mẫu `torch.normal` theo hiệp phương sai của cha. Hai bản sao tách nhau nhờ
cú hích Adam ba phẩy mười sáu lần đã nói, chứ không nhờ dịch thủ công.

**Hỏi: Ngưỡng scale của SADGS chặt hơn mười lần thì clone còn tác dụng gì?**
Đúng là phần lớn Gaussian rơi vào nhánh split, vì `split_qualifiers` ở dòng 991 là phần bù đúng của
`clone_qualifiers` ở dòng 990. Clone trở thành thao tác dành riêng cho Gaussian rất nhỏ — dưới không phẩy
không không năm đơn vị thế giới với extent bằng năm. Giá trị của nó là tăng gấp đôi khối lượng tại chỗ ở
vùng under-reconstruction, rồi để optimizer phân bổ lại.

**Hỏi: Nếu reset moment về không thì các Gaussian cũ có bị ảnh hưởng không?**
Với `cat_tensors_to_optimizer` thì không — phần N dòng cũ giữ nguyên moment, chỉ M dòng mới là số không
(dòng 578-579). Nhưng có một hàm khác, `replace_tensor_to_optimizer` ở dòng 485, thì xoá moment của
**toàn bộ** param group. Hàm đó được gọi mỗi chu kỳ densify để ép opacity về tối đa không phẩy tám
(dòng 1046-1048), nên cú sốc động lượng toàn cục đó là có thật.

---

## Slide: Giãn giải tích Gaussian dưới cỡ (`expand_undersized_gs`) — và trạng thái *bị tắt*

**Thời lượng: ~75 giây**

**Lời thuyết trình:**

SADGS thiết kế ba thao tác thích nghi theo tần số, không phải hai. Split cho Gaussian quá to so với
texture, prune cho đóng góp thấp, và thao tác thứ ba — expand — cho Gaussian **quá nhỏ** so với texture.
Trực giác rất rõ: nếu một Gaussian chỉ phủ một phần mười chi tiết mà nó cần biểu diễn, thì để phủ hết
chi tiết đó ta cần hàng trăm Gaussian. Đó là lãng phí. Thay vì sinh thêm điểm, SADGS sửa thẳng tham số
kích thước.

Điều kiện chọn là eta nhỏ hơn `tau_expand` và eta lớn hơn không. Eta ở đây là tỉ số không thứ nguyên
giữa độ dài trục Gaussian chiếu lên ảnh và bước sóng texture cục bộ, định nghĩa ở `freq_utils.py` dòng
348. Eta lớn hơn một nghĩa là Gaussian to hơn chi tiết, gây aliasing, cần split. Eta nhỏ hơn một là
ngược lại. `tau_expand` mặc định bằng một chấm không, chính là ngưỡng Nyquist. Mask này là per-axis trên
tensor N nhân ba, tức mỗi trục chính được xét độc lập. Điều kiện eta lớn hơn không là để loại Gaussian
chưa từng được quan sát — nếu thiếu nó, log của không sẽ là âm vô cực và Gaussian bị giãn vô hạn.

Công thức cập nhật rất gọn: delta log scale bằng trừ một phần hai nhân log của eta. Vì hàm kích hoạt
scale là `torch.exp`, nhân scale trở thành cộng trong không gian log. Với eta nhỏ hơn một thì log âm,
nên delta dương, nên scale tăng — đúng chiều. Hệ số giãn chính là eta mũ trừ một phần hai, đúng đường
cong trên hình bên phải. Có một clamp ở mười mũ trừ sáu, nghĩa là một lần gọi có thể nhân scale tới
khoảng một nghìn lần. Đây là con số rất lớn, và là một rủi ro.

Ưu điểm lớn nhất: N không đổi. Không gọi `densification_postfix`, không cat tensor nào. Để đạt cùng độ
phủ bằng split thì phải sinh k mũ ba Gaussian con, với k là trần của eta mũ trừ một phần hai. Eta bằng
không phẩy không một cho k bằng mười, tức gấp một nghìn lần số điểm. Expand đạt cùng kết quả với không
byte thêm. Đây đúng nghĩa là van giảm áp VRAM.

Nhưng em phải nói rõ hai điều. Thứ nhất, docstring của hàm giả định eta tỉ lệ với bình phương scale, nên
khẳng định một lần gọi đưa eta về đúng một. Code thật thì đo eta tuyến tính theo scale, nên một lần gọi
chỉ đưa eta về căn của eta, không phải một. Hội tụ vẫn nhanh, nhưng comment trong code là sai.

Thứ hai, và quan trọng hơn: **nhánh này đang bị tắt**. Lời gọi duy nhất tới `expand_undersized_gs` nằm ở
`train.py` dòng 356 đến 359, và nó đã bị comment out. Hàm tồn tại đầy đủ, chạy được, nhưng không nằm
trên đường thực thi nào. Nghĩa là mặc định SADGS chỉ split và prune. Các tham số `tau_expand`,
`adaptive_clone`, `expansion_speed` là tham số ngủ đông — đổi giá trị không ảnh hưởng gì tới kết quả.
Mọi số liệu nhóm em báo cáo đều không bao gồm cơ chế giãn giải tích này. Nhóm em nêu ra vì đây là khoảng
trống triển khai rõ ràng nhất của codebase, và cũng là hướng phát triển tiếp theo.

**Chuyển tiếp:** Như vậy đã xong nhóm thao tác tăng mật độ. Phần tiếp theo chuyển sang chiều ngược lại —
cơ chế pruning nhiều tầng của SADGS.

---

### Nếu bị hỏi

**Hỏi: Nếu bật lại nhánh expand thì có chạy đúng không?**
Không, có ít nhất ba vấn đề. Một, `train.py:358` truyền `avg_high_eta_3ch`, mà tensor này chỉ khác không
ở các Gaussian có `eta_high_count` lớn hơn không — tức nhóm **đã** vi phạm Nyquist và đang chờ split.
Kết hợp với điều kiện eta lớn hơn không, expand sẽ chạm đúng nhóm đang chờ split: giãn và chia cùng lúc,
mâu thuẫn trực tiếp. Hai, `replace_tensor_to_optimizer` ở dòng 489-490 dùng `stored_state` mà không kiểm
tra `None`, nên nếu gọi trước bước optimizer đầu tiên sẽ crash. Ba, clamp mười mũ trừ sáu cho phép nhân
scale một nghìn lần trong một bước, đủ để phá tile-sorting.

**Hỏi: Sao biết chắc là bị comment out chứ không phải gọi ở chỗ khác?**
Nhóm em đã grep toàn bộ `SADGS/` với từ khoá `expand_undersized_gs`. Chỉ có hai kết quả: định nghĩa hàm
tại `gaussian_model.py:833`, và lời gọi bị comment tại `train.py:356`. Không có chỗ nào khác. Tham số
`tau_expand` cũng chỉ xuất hiện đúng một lần ở `arguments/__init__.py:135`.

**Hỏi: Muốn sửa cho đúng lý thuyết thì phải đổi gì?**
Đổi hệ số từ trừ không phẩy năm thành trừ một chấm không. Vì `freq_utils.py:348` định nghĩa eta bằng độ
dài trục chia bước sóng — bậc nhất theo scale — nên để eta mới bằng một thì delta log phải bằng trừ log
eta, không phải trừ một nửa log eta. Với hệ số hiện tại, một lần gọi cho eta mới bằng căn eta, và cần
khoảng bốn lần gọi để đưa eta từ mười mũ trừ bốn lên trên không phẩy năm.

---
