# Kịch bản 45 phút — SADGS chuyên sâu, Bot 01

**Slide nguồn:** `DOCS/Slides67/Slides45min/part45_sadgsx_01.tex` (1 frame)
**Chủ đề:** Structure tensor & multiscale image structure — nền tảng đo cấu trúc ảnh của SADGS
**Chương sách đi kèm:** [18.1 — Structure tensor đa tỉ lệ](../../../BOOK/18-sadgs-co-che-chuyen-sau/01-structure-tensor-da-ti-le.md)

---

## Slide: Tensor cấu trúc đa tỉ lệ — nền tảng đo cấu trúc ảnh của SADGS

**Thời lượng: ~75 giây**

**Lời thuyết trình:**

Trước khi nói về việc SADGS chia nhỏ Gaussian như thế nào, phải trả lời câu hỏi nền: làm sao máy biết chỗ nào trong ảnh là chi tiết mịn, chỗ nào là mảng phẳng. Câu trả lời của SADGS là tensor cấu trúc, công thức ở góc trên bên trái. Với mỗi pixel, ta làm mượt ảnh rồi lấy đạo hàm Sobel theo hai hướng x và y, nhân chéo hai đạo hàm đó thành một ma trận hai nhân hai, rồi làm mượt lần nữa bằng một cửa sổ Gauss bán kính rho. Có hai chi tiết đáng chú ý ở đây. Thứ nhất, đạo hàm chạy riêng trên từng kênh đỏ, lục, lam rồi mới cộng **năng lượng** lại — đây là phương pháp Di Zenzo, giúp không bỏ sót những cạnh mà hai màu khác nhau nhưng độ sáng bằng nhau. Thứ hai, bước làm mượt bằng rho là bắt buộc chứ không phải tuỳ chọn: nếu bỏ nó đi thì ma trận luôn có hạng một, định thức bằng không ở mọi điểm, và ta không còn phân biệt được cạnh thẳng với góc hay vân texture nữa. Toàn bộ nằm ở `loss_utils.py`, dòng 170 đến 230.

Ma trận hai nhân hai này đối xứng nên có hai trị riêng, tính bằng công thức đóng ở dòng thứ hai của slide. Trị riêng lớn hơn, lambda một, là năng lượng biến thiên theo hướng mạnh nhất; nghịch đảo căn bậc hai của nó cho ta một đại lượng mang đơn vị **pixel**, gọi là bước sóng nhỏ nhất w-min — hiểu nôm na là "chi tiết nhỏ nhất mà chỗ này còn nhìn thấy được". Đây chính là giới hạn tốc độ cho kích thước Gaussian: Gaussian nào chiếu lên màn hình mà dài hơn w-min thì đang làm mờ chi tiết, cần tách. Vector riêng thứ hai, vuông góc với vector riêng thứ nhất, là hướng mà cạnh chạy dọc theo — tức hướng duy nhất Gaussian được phép kéo dài mà không mất chi tiết. Hình quiver ở chương sách cho thấy rất rõ điều này quanh cung tròn: vector thứ hai luôn bám theo tiếp tuyến.

Nhưng một giá trị sigma duy nhất chỉ nhìn thấy một dải tần. Nên SADGS quét bốn octave, mỗi octave nhân sigma với một-phẩy-năm, và gán cho octave đó một tần số danh nghĩa f bằng một-phẩy-năm mũ trừ i. Ở mỗi octave, ta đo band response bằng hiệu hai ảnh mờ liên tiếp — tức Difference-of-Gaussian, đo xem làm mờ thêm thì mất bao nhiêu chi tiết — rồi luỹ thừa ba để làm trọng số. Công thức gộp ở dòng thứ ba là chỗ tinh tế nhất: mỗi octave bị **chia cho vết của chính nó** trước khi cộng vào, nên nó chỉ đóng góp **hướng**, còn **độ lớn** thì do f bình phương áp đặt. Hệ quả là vết của bản đồ cuối cùng không còn là năng lượng gradient nữa, mà chính là trung bình có trọng số của bình phương tần số trội. Thay số vào cấu hình mặc định, vết này nằm gọn trong khoảng từ không-phẩy-không-tám-tám đến một, nên w-min luôn rơi trong khoảng một đến bốn-phẩy-tám pixel. Đây là một ràng buộc thực tế đáng nhớ: ngay cả một bức tường trắng trơn cũng chỉ cho phép Gaussian dài tối đa cỡ năm pixel trên màn hình.

Cuối cùng, hai biến thể v1 và v2 khác nhau đúng một chỗ: v1 tính tensor một lần ở thang mịn nhất rồi chỉ làm mờ lại cho các thang sau, còn v2 tính lại tensor trên ảnh đã mờ ở từng thang. Hình dưới bên phải so trace của hai bản: chúng gần như trùng nhau ở vùng sọc thẳng, và chỉ lệch tại cung tròn cùng các ranh giới vùng — tức đúng những nơi hướng cấu trúc thay đổi theo thang. Mặc định huấn luyện chọn v1, bản rẻ hơn, dù chính docstring của tác giả gọi v2 là bản "đúng". Lý do hợp lý: bản đồ này chỉ dùng làm tín hiệu cổng cho quyết định tách hay cắt, không đi vào hàm mất mát. Và nó được tính **một lần duy nhất trước vòng huấn luyện** rồi cache theo tên ảnh, ở `train.py` dòng 160 đến 199.

**Chuyển tiếp:** Bản đồ cấu trúc đã sẵn sàng. Việc tiếp theo là biến nó thành một con số cho từng Gaussian ở từng góc nhìn — đó là đại lượng eta, và slide sau sẽ nói về cách chiếu ba trục của Gaussian xuống màn hình để tính nó.

---

## Nếu bị hỏi

**Hỏi: Tính tensor cấu trúc cho toàn bộ ảnh huấn luyện thì có tốn không? Sao không tính lại mỗi vòng?**
Không tốn theo thời gian, nhưng tốn theo bộ nhớ. Nó chạy **một lần** trước vòng lặp, trong `torch.no_grad()`, gom camera theo độ phân giải rồi xử lý theo lô 100 ảnh (`train.py:168-190`). Vì ảnh GT không đổi trong suốt quá trình huấn luyện nên không có lý do tính lại. Cái phải trả là cache: mỗi view giữ thường trú một tensor `(1,3,H,W)` float32 trên GPU (`train.py:193-194`) — ở độ phân giải 1600×1200 là khoảng 12 MB mỗi view, nên 200 view tốn cỡ 4,6 GB. Đây là ràng buộc VRAM thực tế của SADGS, không phải chi tiết phụ.

**Hỏi: Nhóm chỉnh được những tham số nào của phần này?**
Thực tế chỉ hai: `st_levels` (mặc định 4) và `st_mode` (mặc định `"v1"`), khai báo ở `arguments/__init__.py:120-121`. Lý do là `train.py:188` và `:190` chỉ truyền đúng `levels`; `power_factor=3.0`, `octave_step=1.5`, `base_sigma=1.0` giữ nguyên mặc định trong chữ ký hàm, muốn đổi phải sửa code. Ngoài ra có hai tham số trông như chỉnh được nhưng không: `aggregation_mode` không được đọc ở bất kỳ dòng nào (tham số chết), còn `smoothing_factor` chỉ dùng làm **cờ bật/tắt** trong điều kiện `if i > 0 and smoothing_factor > 0` — độ mượt thật bị hard-code là `2*target_sigma` (`loss_utils.py:273-275`), nên đặt 0,1 hay 100 đều cho kết quả y hệt.

**Hỏi: Có hàm loss nào dùng tensor cấu trúc này không?**
Có code, nhưng **không chạy**. `lambda_freq` mặc định bằng 0 (`arguments/__init__.py:119`) và không được đọc ở đâu trong `SADGS/`. `frequency_loss_simple` (`loss_utils.py:389`) được `import` ở `freq_utils.py:5` nhưng không được gọi lần nào; `frequency_loss` (`loss_utils.py:80`) cũng không có người gọi; còn `estimate_required_gaussians` (`loss_utils.py:394`) thì gọi một hàm tên `get_multiscale_structure_tensor` **không tồn tại** ở dòng 411, sẽ ném `NameError` nếu ai đó thực sự chạy nó. Nói cách khác, `st_map` hiện chỉ đóng vai trò tín hiệu cổng cho densify/prune, hoàn toàn nằm ngoài đồ thị gradient.

**Hỏi: Vì sao mặc định lại chọn v1 nếu tác giả tự nhận v2 mới đúng?**
Vì đánh đổi có lợi. v2 phải gọi đầy đủ `get_structure_tensor_torch` ở từng octave (`loss_utils.py:357`), tức thêm ba lần Sobel trên ảnh RGB đầy đủ so với v1 chỉ làm mờ lại một tensor một kênh (`:282-284`). Đổi lại, hình so sánh cho thấy chênh lệch giữa hai bản **chỉ tập trung ở vùng cong và ranh giới vùng**, gần như bằng 0 ở vùng sọc thẳng. Vì kết quả chỉ dùng để quyết định tách hay không tách chứ không lan vào gradient, sai số hướng ở octave thô không tích luỹ — nên chọn bản rẻ là hợp lý.
