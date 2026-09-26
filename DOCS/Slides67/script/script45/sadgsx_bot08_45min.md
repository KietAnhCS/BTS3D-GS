# Kịch bản 45 phút — SADGS Bot 08

Bao phủ hai slide: `part45_sadgsx_11.tex` (hàm mất mát) và `part45_sadgsx_12.tex` (lấy mẫu camera FPS & lấy mẫu Gaussian dị hướng 2D).

Tổng thời lượng dự kiến: ~90 giây.

---

## Slide — Hàm mất mát của SADGS: $L_1$ + SSIM + $L_2$ (và hai loss chưa dùng)

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Sang phần hàm mất mát. Em muốn nói thẳng một điều trước: toàn bộ tín hiệu gradient của SADGS đi qua đúng một dòng code, dòng 259 của train.py, và nó là loss photometric quen thuộc — 0,8 nhân L1, cộng 0,2 nhân một trừ SSIM, cộng 2,0 nhân L2. Ba hệ số này lấy từ arguments dòng 86 và 117, không phải em tự đặt. So với 3DGS gốc, khác biệt duy nhất ở tầng photometric là số hạng L2 với trọng số 2,0. Và trọng số 2,0 nghe thì lớn nhưng thực ra chỉ để kéo L2 về cùng bậc độ lớn với L1 thôi — vì khi sai số nhỏ hơn 1 thì bình phương luôn nhỏ hơn trị tuyệt đối. SSIM ở đây dùng kernel CUDA fused_ssim chứ không phải hàm ssim thuần PyTorch trong loss_utils. Bây giờ đến phần quan trọng: trong file loss_utils có **hai hàm loss mang tinh thần structure-aware rất rõ nhưng chưa bao giờ được gọi**. Thứ nhất là tone_curve_loss, dòng 28: nó chia sai số cho chính độ sáng dự đoán, có stop-gradient ở mẫu số, nên hệ thống học trên **sai số tương đối** thay vì tuyệt đối — hệ quả là vùng tối được khuếch đại gradient mạnh, đúng tinh thần Weber-Fechner. Thứ hai là frequency_loss, dòng 80: nó không tính trên pixel mà tính **trên từng Gaussian** — lấy mẫu structure tensor tại tâm Gaussian, suy ra bước sóng texture cho phép, rồi phạt ReLU một phía mỗi khi bán kính Gaussian vượt quá bước sóng đó. Hình bên phải minh hoạ đúng cơ chế này trên một ảnh chirp: bước sóng cho phép co lại dần khi texture rậm hơn, và vùng tô đỏ chính là nơi Gaussian bị phạt. Nhưng cả hai hàm này đều có trọng số bằng 0 và **không được tham chiếu ở bất kỳ đâu** — nghĩa là đặt tham số qua dòng lệnh cũng vô hiệu, vì dòng nối chưa tồn tại. Kết luận cần nhớ: loss của SADGS vẫn là loss 3DGS cộng một số hạng L2 nặng; toàn bộ tính structure-aware nằm ở nhánh densification, qua thống kê eta không gradient, chứ không nằm trong hàm mất mát.

**Chuyển tiếp:** Đã rõ tần số không đi vào loss, vậy nó đi vào đâu — câu trả lời là qua thống kê tích luỹ theo từng view, và điều đó đặt ra câu hỏi: chọn view nào để tích luỹ? Đó là nội dung slide tiếp theo.

### Nếu bị hỏi

**H: Tại sao nhóm lại thêm L2 với trọng số tận 2,0? Có làm ảnh mờ đi không?**
Đ: Vì L2 nhỏ hơn L1 hẳn một bậc khi sai số dưới 1 — ví dụ sai số 0,1 thì L1 bằng 0,1 còn L2 chỉ 0,01. Hệ số 2,0 chỉ đưa hai số hạng về cùng bậc, không phải ưu tiên L2. Tác dụng thực là khuếch đại gradient ở các pixel sai nhiều, giúp hội tụ nhanh giai đoạn đầu. Còn nguy cơ mờ thì được số hạng D-SSIM giữ lại — nó phạt rất nặng việc mất tương quan cục bộ, là thứ L1 và L2 đều bỏ qua.

**H: Làm sao khẳng định tone_curve_loss và frequency_loss chưa dùng, chứ không phải nhóm đọc sót?**
Đ: Ba bằng chứng độc lập. Một, train.py dòng 17 chỉ import l1_loss, l2_loss và hai hàm structure tensor — hai hàm kia không có trong danh sách import. Hai, lambda_tone và lambda_freq đều bằng 0 ở arguments dòng 118 và 119. Ba, grep toàn repo không tìm thấy lời gọi nào. Ngoài ra còn một bằng chứng gián tiếp: hàm estimate_required_gaussians ở dòng 394 gọi get_multiscale_structure_tensor tại dòng 411, mà tên đó không tồn tại trong file — chỉ có hậu tố _v1 và _v2. Nếu hàm đó từng chạy, nó đã ném NameError rồi.

**H: Vậy ý tưởng "so cỡ Gaussian với bước sóng texture" có được dùng không?**
Đ: Có, nhưng không ở dạng loss. Nó chạy ở freq_utils dòng 181, hàm update_freq_stats_online, dưới dạng thống kê eta bằng độ dài trục chiếu chia cho bước sóng cực tiểu, tính riêng cho cả ba trục, và chạy trong torch.no_grad. Khác biệt cốt lõi: frequency_loss là đẳng hướng và khả vi, còn eta là dị hướng ba trục và không khả vi. SADGS chọn đường thứ hai — điều khiển tần số qua densification chứ không qua gradient.

---

## Slide — Lấy mẫu camera bằng FPS & lấy mẫu Gaussian dị hướng 2D

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này nói về hai bộ lấy mẫu trong SADGS. Thứ nhất là sampling_cameras ở freq_utils dòng 12, mặc định chế độ fps với 60 camera. Nó lấy vị trí camera bằng công thức trừ R chuyển vị nhân t — tức giải ngược ma trận view ra toạ độ thế giới — rồi chạy vòng lặp tham lam kinh điển: mỗi bước cập nhật khoảng cách tới điểm gần nhất đã chọn, rồi chọn camera **xa nhất** so với mọi camera đã chọn. Đây chính là thuật toán xấp xỉ 2 của bài toán k-center. Hình bên phải cho thấy hiệu quả: em cố tình dồn 70 phần trăm camera vào một phần ba quỹ đạo, vậy mà tập được chọn vẫn trải gần như đều quanh vòng tròn — vì quy tắc arg-max luôn nhảy sang vùng đang trống nhất, không quan tâm mật độ gốc. Điều này quan trọng với SADGS vì thống kê eta tích luỹ theo từng view: nếu camera dồn về một phía, các Gaussian phía đối diện gần như không bao giờ được cập nhật, và ước lượng vi phạm tần số sẽ thiên lệch. Có hai điểm em phải nói thẳng. Một, metric chỉ dùng **vị trí** camera, hoàn toàn không dùng hướng nhìn — hai camera đứng cùng chỗ nhìn ngược nhau có khoảng cách bằng 0. Hai, chi phí là N nhân num_cams, nên phải lấy tập con mới rẻ; nhưng ở train.py dòng 224 và 240, hàm lại được gọi với num_cams bằng đúng số camera trong stack, cộng với việc code sắp lại kết quả theo chỉ số gốc tăng dần, nên FPS ở đó **suy biến thành phép tắt xáo trộn** kèm chi phí N bình phương — và mặc định camera_sampling còn là "random" nên nhánh đó thậm chí không chạy. Bộ lấy mẫu thứ hai là sample_anisotropic_gaussians_2d: đặt Gaussian theo xác suất tỉ lệ với vết của structure tensor, hình dạng lấy từ hai trị riêng, trục dài nằm dọc vector riêng ứng với trị riêng nhỏ — tức bám theo cạnh. Ý tưởng rất đẹp, nhưng sự thật là hàm này không được gọi ở đâu cả; khởi tạo point cloud thật vẫn là create_from_pcd từ COLMAP, ở scene dòng 83.

**Chuyển tiếp:** Như vậy cả tầng loss lẫn tầng lấy mẫu đều cho thấy cùng một điều — hạ tầng đã có, nhưng cơ chế thực sự tạo ra chất lượng của SADGS nằm ở nhánh densification; phần tiếp theo sẽ đi vào optimizer và lịch học.

### Nếu bị hỏi

**H: Nếu FPS ở train.py bị suy biến thì tại sao còn giữ nó?**
Đ: Vì nó vẫn đúng thuật toán, chỉ sai ở tham số gọi. Sửa num_cams từ len(stack) xuống một tập con, ví dụ 60 như mặc định của chữ ký hàm, là lấy lại được lợi ích ngay. Trên thực nghiệm mô phỏng 220 camera, bán kính phủ của FPS luôn thấp hơn trung bình random, và chênh nhiều nhất đúng ở vùng num_cams nhỏ. Random còn có phương sai lớn, nghĩa là một lần rút xấu có thể bỏ trống cả một phía cảnh.

**H: Chi phí FPS có đáng lo không khi dataset lớn?**
Đ: Chi phí là O của N nhân num_cams so với O của N cho random. Với N bằng 1000, lấy k bằng 60 rẻ hơn lấy k bằng N khoảng 17 lần. Nhưng lưu ý FPS chỉ chạy khi stack cạn, không phải mỗi iteration, nên ngay cả trường hợp N bình phương thì chi phí phân bổ trên cả một epoch cũng không phải nút cổ chai — vấn đề chính là nó không mang lại lợi ích gì khi num_cams bằng N.

**H: Vậy lấy mẫu Gaussian dị hướng 2D để làm gì nếu không dùng?**
Đ: Docstring nói nó được tách ra từ fit_2d_image.py để tái sử dụng — mà file đó không còn trong repo. Vai trò đúng của nó là tiện ích nghiên cứu cho bài toán khớp ảnh 2D, đồng thời là bản mẫu cho ý tưởng cốt lõi của SADGS: dùng structure tensor để quyết định kích thước và hướng Gaussian. Con số thuyết phục là tỉ lệ mẫu rơi vào vùng phẳng vô ích: lấy đều cho khoảng 25 phần trăm, lấy theo structure tensor chỉ 0,3 phần trăm. Ý tưởng đó đã được hiện thực hoá trong huấn luyện 3D qua eta của update_freq_stats_online, chứ không qua hàm này.
