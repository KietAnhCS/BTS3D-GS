# Kịch bản thuyết trình — Quy trình Render 3D Gaussian Splatting và Cải tiến Structure-Aware (SADGS)

> Bài giảng 45 phút, 51 slide. Lời thuyết trình dưới đây đối chiếu trực tiếp với code thật trong repo
> (`train.py`, `scene/gaussian_model.py`, `utils/freq_utils.py`, `utils/loss_utils.py`, `arguments/__init__.py`,
> `cuda_rasterizer/{forward.cu,auxiliary.h}`) và với slide tương ứng trong `DOCS/Slides67/Slides45min/part45_0{1..9}.tex`.
> Đọc tuần tự từ Slide 1 đến Slide 51.

## Slide 1: Bìa
Xin chào mọi người. Hôm nay mình sẽ trình bày về 3D Gaussian Splatting, gọi tắt là 3DGS, và về SADGS, viết tắt của Structure-Aware Densification for Gaussian Splatting. Đây là một cải tiến nhắm vào đúng khâu "thêm và bớt Gaussian" trong lúc train. Bài giảng này có một đặc điểm mình muốn nói trước: mọi công thức và con số mình đưa ra đều được đối chiếu với code thật của pipeline, chứ không chỉ chép lại từ paper hay các bài blog. Ở vài chỗ, code thật chạy khác với cách người ta hay mô tả, và mình sẽ nói rõ từng chỗ đó. Mình sẽ đi từ những khái niệm nền tảng nhất: một Gaussian 3D là gì, nó được khởi tạo ra sao, màu của nó được biểu diễn thế nào. Sau đó mình mới đi tới phần chính là cơ chế densification. Giờ mình bắt đầu với câu hỏi: vì sao lại cần tới 3D Gaussian Splatting?

## Slide 2: Vì sao 3D Gaussian Splatting?
Trước hết là bài toán. Ta có một tập ảnh chụp cùng một cảnh từ nhiều góc khác nhau. Vị trí và hướng của camera ở mỗi ảnh, tức là camera pose, đã được SfM, cụ thể là COLMAP, ước lượng sẵn. Việc của ta là dựng lại một biểu diễn 3D của cảnh để render ra ảnh từ một góc nhìn hoàn toàn mới mà vẫn trông thật. Bài toán này gọi là novel view synthesis.

NeRF, ra đời năm 2020, giải bài này bằng một mạng MLP ẩn. Mạng nhận vào vị trí và hướng nhìn, trả ra màu và mật độ. Để render một pixel, NeRF bắn một tia sáng qua pixel đó rồi ray-marching, tức là lấy hàng trăm điểm mẫu dọc theo tia. Mỗi điểm mẫu lại phải chạy một lần forward qua mạng. Nhân con số đó lên cho cả bức ảnh, ta hiểu vì sao NeRF cho chất lượng cao nhưng rất chậm: train mất hàng giờ tới hàng ngày, còn render cũng mất cỡ vài giây cho mỗi khung hình.

3D Gaussian Splatting của Kerbl và các cộng sự, công bố ở SIGGRAPH 2023, thay đổi tận gốc cách biểu diễn cảnh. Không còn mạng ẩn nào cả. Cảnh được mô tả bằng hàng trăm nghìn tới hàng triệu Gaussian 3D tường minh. Mỗi Gaussian có bộ tham số cụ thể gồm mu, Sigma, alpha và c. Để render, ta chỉ cần chiếu các Gaussian này lên mặt phẳng ảnh rồi rasterize, tức là tô chúng lên màn hình, giống hệt cách GPU vẽ tam giác trong game. Vì vậy 3DGS train nhanh hơn nhiều lần, và quan trọng nhất là render được theo thời gian thực, từ hàng chục tới hàng trăm khung hình mỗi giây.

Tuy vậy vẫn còn một câu hỏi chưa có lời giải trọn vẹn. Cần bao nhiêu Gaussian, đặt chúng ở đâu, và mỗi cái to cỡ nào? Ta không biết trước, nên phải để thuật toán tự điều chỉnh trong lúc train. Cơ chế đó gọi là Adaptive Density Control, viết tắt ADC. ADC gốc chỉ dựa vào gradient để quyết định thêm hay bớt Gaussian. SADGS can thiệp đúng vào chỗ này: nó bổ sung một tín hiệu đo trực tiếp cấu trúc tần số cục bộ của ảnh, để densify đúng chỗ, đúng lúc và đúng trục hơn.

Lộ trình bài giảng gồm bốn phần. Phần một là biểu diễn Gaussian và pipeline render. Phần hai là loss, gradient và ADC nguyên bản. Phần ba là SADGS, bắt đầu từ việc xây dựng tín hiệu tần số eta từ đầu. Phần bốn là nhìn lại một cách trung thực xem cái gì thật sự đang chạy. Ta bắt đầu với phần một, cụ thể là viên gạch cơ bản nhất: một Gaussian 3D.

## Slide 3: 3D Gaussian: đơn vị biểu diễn cơ bản
Toàn bộ cảnh chỉ là một tập hợp các Gaussian 3D. Mỗi Gaussian là một hàm mật độ liên tục trong không gian, cho bởi công thức đóng khung trên slide: G của x bằng exp của trừ một phần hai nhân (x trừ mu) chuyển vị, nhân Sigma nghịch đảo, nhân (x trừ mu).

Mình đọc công thức này theo trực giác. Đại lượng (x trừ mu) chuyển vị nhân Sigma nghịch đảo nhân (x trừ mu) là một kiểu "khoảng cách bình phương" từ điểm x tới tâm mu, nhưng được đo theo hình dạng mà Sigma quy định. Tại tâm, khoảng cách này bằng không nên G bằng một, là giá trị lớn nhất. Càng đi ra xa, khoảng cách càng lớn và hàm mũ âm làm G giảm dần về không. Vì Sigma có thể kéo dài theo hướng này và ép dẹt theo hướng khác, mặt đồng mức của G không phải mặt cầu mà là mặt ellipsoid. Hình minh hoạ trên slide cho thấy đúng dạng "cái bướu" đó: đậm nhất ở giữa, mờ dần ra biên. Vì vậy ta có thể hình dung mỗi Gaussian như một cục mây mờ hình ellipsoid.

Mỗi Gaussian mang bốn nhóm tham số, và tất cả đều học bằng gradient descent.
- Mu là vị trí tâm, gồm 3 số.
- Sigma là hình dạng và hướng của ellipsoid. Sigma không được lưu trực tiếp mà được tham số hoá qua scale s gồm 3 số và quaternion q gồm 4 số. Slide sau sẽ giải thích lý do.
- Alpha là độ mờ đục, cho biết Gaussian "đặc" hay "trong suốt". Alpha được lưu dưới dạng logit rồi đi qua hàm sigmoid, nhờ vậy nó luôn nằm trong khoảng từ 0 tới 1 dù optimizer đẩy logit đi đâu.
- C là màu. C không phải một bộ RGB cố định mà là các hệ số Spherical Harmonics, để màu có thể đổi theo góc nhìn.

Cộng lại, mỗi Gaussian có 59 số thực: 3 cho xyz, 3 cho scaling, 4 cho rotation, 1 cho opacity, 3 cho f_dc và 45 cho f_rest. Riêng phần màu đã chiếm 48 trên 59 số, tức là phần lớn bộ nhớ. Ta sẽ quay lại con số này ở slide về SH. Trước đó, ta xem vì sao Sigma phải đi qua scale và quaternion.

## Slide 4: Tham số hoá hiệp phương sai: luôn xác định dương
Về mặt toán học, một ma trận hiệp phương sai hợp lệ phải đối xứng và xác định dương. Nếu điều kiện này bị vi phạm, Sigma nghịch đảo trong công thức G có thể làm hàm "phát tán" vô nghĩa thay vì tắt dần ra biên. Một ma trận 3×3 đối xứng có 6 số độc lập. Cách ngây thơ là cho gradient descent tối ưu thẳng 6 số đó. Nhưng gradient descent không biết gì về ràng buộc xác định dương, nên chỉ vài bước cập nhật là ma trận đã có thể rơi ra khỏi vùng hợp lệ.

Giải pháp của 3DGS là phân rã Sigma thành R nhân S nhân S chuyển vị nhân R chuyển vị. Ý tưởng là không tối ưu Sigma trực tiếp mà tối ưu các thành phần mà mọi giá trị của chúng đều cho ra một Sigma hợp lệ.
- S là ma trận đường chéo chứa s bằng exp của biến _scaling. Hàm mũ luôn dương, nên dù _scaling là số thực bất kỳ thì s vẫn dương. Ta không cần thêm ràng buộc nào khi tối ưu.
- R là ma trận quay dựng từ quaternion q sau khi chuẩn hoá về độ dài 1. Một quaternion đơn vị luôn cho ra một ma trận quay trực giao hợp lệ.

Kết quả là với bất kỳ s và q nào, Sigma tự động đối xứng và xác định dương. Ta đã biến một bài toán tối ưu có ràng buộc thành một bài toán tối ưu tự do. Đây là một mẹo rất phổ biến trong học sâu, và nó cũng giống cách opacity dùng sigmoid ở slide trước.

Về hình học, cách phân rã này dễ hình dung. Đầu tiên ta lấy một hình cầu đơn vị, kéo giãn nó theo ba trục toạ độ với hệ số s_x, s_y, s_z. Sau đó ta xoay nó bằng R. Vì vậy bán trục của ellipsoid theo mỗi hướng chính chính là s_j. Có một điểm hay bị nhầm: trị riêng của Sigma không phải s_j mà là s_j bình phương, vì Sigma chứa S hai lần. Phân biệt rõ điều này rất quan trọng, vì về sau các ngưỡng to/nhỏ trong ADC đều được so trên s. Tiếp theo, ta xem các giá trị s và mu được gán ban đầu như thế nào.

## Slide 5: Khởi tạo (1/2): scale ban đầu theo mật độ điểm SfM
Trước khi train, ta cần một điểm xuất phát. 3DGS đặt mỗi Gaussian đúng tại vị trí của một điểm 3D trong sparse point cloud mà SfM/COLMAP đã tái dựng từ các ảnh input. Nhưng biết vị trí thôi thì chưa đủ, ta còn phải cho mỗi Gaussian một kích thước ban đầu hợp lý. Nếu chọn quá nhỏ, giữa các điểm sẽ còn nhiều khoảng trống và ảnh render ban đầu lỗ chỗ. Nếu chọn quá to, các Gaussian chồng lấn lên nhau ngay từ đầu, làm nhiễu tín hiệu học.

Ý tưởng ở đây là để chính dữ liệu cho biết mật độ cục bộ. Với mỗi điểm, ta tìm 3 điểm lân cận gần nhất, tính khoảng cách bình phương tới từng điểm, lấy trung bình rồi lấy căn bậc hai. Con số thu được là "khoảng cách điển hình giữa các điểm hàng xóm" tại vị trí đó. Cuối cùng ta lấy log, tức là s ngã bằng log của căn trung bình d bình phương knn3. Lý do lấy log là vì scale được lưu và tối ưu trong không gian log. Như slide trước, s bằng exp của _scaling, nên giá trị khởi tạo cho _scaling phải là log của kích thước mong muốn.

Trực giác của cách làm này rất tự nhiên. Ở vùng point cloud dày đặc, thường là nơi có nhiều chi tiết hình học mà COLMAP bắt được nhiều điểm đặc trưng, các hàng xóm ở gần nhau, nên Gaussian khởi tạo nhỏ. Ở vùng thưa, hàng xóm ở xa hơn, nên Gaussian khởi tạo to hơn để lấp khoảng trống. Hình minh hoạ trên slide cho thấy đúng mối liên hệ đó giữa khoảng cách tới láng giềng và kích thước ban đầu. Nhờ vậy cảnh được phủ kín mà ta không cần biết trước hình dạng vật thể.

Có một chi tiết cần để ý: phép tính này dùng khoảng cách tuyệt đối trong đơn vị của point cloud. Để các quyết định về sau không phụ thuộc vào việc cảnh to hay nhỏ, ta cần thêm một thước đo cho quy mô cả cảnh. Đó là scene extent ở slide tiếp theo.

## Slide 6: Khởi tạo (2/2): scene extent
Rất nhiều quyết định trong ADC cần trả lời câu hỏi "Gaussian này to hay nhỏ so với quy mô của cả cảnh?". Ví dụ, ngưỡng percent_dense nhân extent được dùng để phân biệt khi nào nên clone và khi nào nên split, sẽ nói ở phần sau. Nếu so với một ngưỡng tuyệt đối, cùng một cấu hình siêu tham số sẽ chạy rất khác nhau giữa một cảnh chụp trên bàn và một cảnh chụp cả toà nhà. Vì vậy ta cần một đơn vị đo gắn với chính cảnh đang xét. Đơn vị đó là scene extent, và nó chỉ được tính một lần lúc khởi tạo.

Công thức là extent bằng 1,1 nhân max theo v của chuẩn (c_v trừ c ngang). Trong đó c_v là vị trí của camera train thứ v, còn c ngang là tâm trung bình của tất cả camera train. Diễn giải ra lời: ta lấy trọng tâm của các camera, tìm camera xa trọng tâm nhất, lấy khoảng cách đó làm bán kính rồi nhân thêm hệ số an toàn 1,1. Kết quả là bán kính của một khối cầu tưởng tượng bao quanh toàn bộ vị trí các camera. Hình minh hoạ trên slide thể hiện đúng khối cầu này.

Mình muốn nhấn mạnh một điểm hay bị hiểu nhầm: extent không phải là đường chéo của hộp bao quanh point cloud. Nó được tính từ vị trí camera, không phải từ các điểm 3D. Vậy vì sao lại hợp lý? Vì trong các bộ dữ liệu kiểu này, người ta thường đi vòng quanh một đối tượng trung tâm để chụp. Bán kính của vòng camera vì thế phản ánh khá tốt kích thước tổng thể của cảnh. Từ giờ, mỗi khi nghe nói tới ngưỡng "to" hay "nhỏ" trong ADC, mọi người hãy hiểu là nó được đo tương đối theo extent này.

Đến đây ta đã có hình học ban đầu: vị trí, kích thước và hướng. Thành phần còn lại là màu, và 3DGS xử lý màu theo một cách khá tinh tế.

## Slide 7: Màu phụ thuộc góc nhìn: hệ số Spherical Harmonics
Ngoài đời, nhiều vật liệu như kim loại, mặt nước hay sứ bóng phản chiếu ánh sáng khác nhau tuỳ theo góc ta đứng nhìn. Nếu mỗi Gaussian chỉ có một màu RGB cố định, ta không thể tái hiện hiệu ứng đó. 3DGS giải quyết bằng cách coi màu là một hàm của hướng nhìn d. Hướng nhìn là một điểm trên mặt cầu đơn vị, nên màu là một hàm xác định trên mặt cầu. Để biểu diễn một hàm như vậy bằng một số hữu hạn tham số, ta khai triển nó theo cơ sở Spherical Harmonics.

Có thể hiểu SH như chuỗi Fourier, nhưng dành cho mặt cầu thay vì một đường thẳng. Chuỗi Fourier phân tích một tín hiệu thành các sóng sin từ tần số thấp tới tần số cao. SH cũng phân tích một hàm trên mặt cầu thành các thành phần từ "trơn" tới "gợn" nhiều hơn, đánh số bằng bậc l. Công thức c(d) bằng tổng các k_lm nhân Y_lm(d) nói rằng màu là tổ hợp tuyến tính của các hàm cơ sở. Mỗi Gaussian chỉ cần lưu các hệ số k_lm, và đó chính là những gì được học.

- Hệ số bậc 0, gọi là hệ số DC, tương ứng với hàm cơ sở hằng số. Vì vậy nó chính là màu trung bình theo mọi góc nhìn. Lúc khởi tạo, màu RGB của điểm SfM được đổi sang hệ số DC bằng hàm RGB2SH: k_00 bằng (c trừ 0,5) chia C_0, với C_0 bằng 1 chia 2 căn pi, xấp xỉ 0,282. Phép trừ 0,5 dời màu về quanh 0, còn phép chia C_0 bù cho hằng số chuẩn hoá của hàm cơ sở bậc 0. Hình nhỏ trên slide minh hoạ phép chuyển đổi này.
- Các hệ số bậc từ 1 trở lên mã hoá mức độ màu thay đổi theo hướng nhìn. Càng nhiều bậc thì hiệu ứng phản chiếu càng chi tiết, nhưng số tham số phải học cũng càng nhiều.

Dùng tới bậc D thì mỗi kênh màu cần (D+1) bình phương hệ số. Với D bằng 3 là 16 hệ số mỗi kênh, nhân 3 kênh ra 48 số: 3 số DC, tức f_dc, và 45 số bậc cao, tức f_rest. Đây chính là 48 trên 59 số mình nhắc ở slide 3. Học ngần ấy tham số màu ngay từ đầu có an toàn không? Slide sau sẽ trả lời câu hỏi này.

## Slide 8: Lịch tăng bậc SH
Ý tưởng như sau. Ở những iteration đầu, vị trí và hình dạng của các Gaussian còn rất thô. Nếu lúc đó cho mô hình học ngay đủ 48 hệ số SH, nó có thể dùng các hệ số bậc cao để "chữa cháy". Tức là thay vì dời Gaussian về đúng chỗ, nó cho màu thay đổi theo góc nhìn để bù lại phần sai do hình học chưa chuẩn. Đó là overfit vào những biến thiên màu giả, không phải phản chiếu thật, và nó gây nhiễu cho quá trình tối ưu. Vì vậy chiến lược hợp lý là tăng bậc SH dần dần: để hình học ổn định trước, rồi mới cho học chi tiết phản chiếu sau.

Nhưng đây là chỗ mình phải đối chiếu với code thật. Trong train.py, dòng 216 tới 217, điều kiện là: if iteration chia lấy dư cho 1 bằng 0 thì gọi gaussians.oneupSHdegree(). Bất kỳ số nguyên nào chia cho 1 cũng dư 0, nên điều kiện này đúng ở mọi iteration. Nghĩa là bậc SH tăng thêm 1 sau mỗi iteration cho tới khi chạm bậc tối đa D_max bằng 3. Nói cách khác, chỉ sau 3 iteration đầu tiên trong tổng số 30.000 iteration, toàn bộ Gaussian đã dùng bậc SH tối đa. Biểu đồ trên slide thể hiện đúng điều đó: bậc SH nhảy lên 3 gần như ngay lập tức rồi giữ nguyên đến hết quá trình train.

Mình nêu rõ chỗ này vì nó rất dễ gây hiểu lầm. Nhiều tài liệu về 3DGS mô tả lịch tăng bậc là "cứ vài nghìn iteration thì tăng một bậc". Con số đó không khớp với đoạn code thực tế của chính pipeline mà ta đang phân tích. Vì vậy, trong pipeline này, lợi ích "hình học ổn định trước, phản chiếu học sau" mà ta vừa nói ở trên thực tế gần như không được tận dụng. Đây là tinh thần mà cả bài giảng sẽ giữ: luôn phân biệt điều lý thuyết mô tả với điều code thật sự chạy.

Đến đây ta đã có đầy đủ bộ tham số của một Gaussian: vị trí, hiệp phương sai, độ mờ đục và màu SH, cùng cách khởi tạo chúng. Bước tiếp theo là xem các Gaussian này được chiếu và tô lên ảnh như thế nào trong pipeline render.

## Slide 9: Chiếu điểm 3D lên màn hình: mô hình pinhole
Chúng ta bắt đầu phần rasterization, tức là quá trình biến các Gaussian 3D thành một bức ảnh 2D. Muốn vẽ được một Gaussian lên ảnh, câu hỏi đầu tiên rất đơn giản: tâm của nó rơi vào pixel nào?

Để trả lời, ta dùng mô hình camera pinhole, mô hình camera tiêu chuẩn trong thị giác máy tính. Một điểm có toạ độ (x, y, z) trong hệ toạ độ camera được chiếu thành pixel (u, v) theo công thức trong khung: u bằng f_x nhân x chia z cộng c_x, v cũng làm tương tự. Trong đó f_x, f_y là tiêu cự tính theo đơn vị pixel, phụ thuộc độ phân giải ảnh và góc nhìn của camera. Còn c_x, c_y là điểm chính, thường nằm ở giữa ảnh, tức W/2 và H/2. Nhìn hình minh hoạ, các bạn thấy tia nối điểm 3D với tâm camera cắt mặt phẳng ảnh ở đâu thì đó chính là vị trí pixel.

Điểm mấu chốt ở đây là phép chia cho z, tức độ sâu. Vì có phép chia này nên phép chiếu là phi tuyến: vật càng xa thì z càng lớn và hình chiếu càng nhỏ, đúng với cảm nhận thị giác bình thường.

Nhưng chính tính phi tuyến lại gây rắc rối. Gaussian không phải một điểm mà là cả một ellipsoid 3D. Khi đưa cả một khối hình qua phép chiếu phi tuyến, kết quả nói chung không còn là một ellipse chuẩn. Cách giải quyết thực dụng là tuyến tính hoá cục bộ: ngay tại tâm Gaussian, ta xấp xỉ phép chiếu bằng đạo hàm bậc nhất của nó, tức ma trận Jacobian J. Nếu nhìn vào J, hai cột đầu là f/z, nghĩa là dịch ngang trong không gian thì dịch ngang trên ảnh theo tỉ lệ nghịch với độ sâu. Cột thứ ba mang dấu trừ và chia cho z bình phương, mô tả việc điểm tiến ra xa thì hình chiếu co dần về tâm ảnh.

Các bạn nên ghi nhớ ma trận J này, vì nó được dùng ở hai nơi: dùng để chiếu hiệp phương sai sang 2D ở slide tiếp theo, và sau này dùng để chiếu trục Gaussian trong cơ chế của SADGS. Giờ ta sẽ xem J giúp chiếu cả một ellipsoid như thế nào.

## Slide 10: EWA splatting: chiếu ellipsoid 3D thành ellipse 2D
Có Jacobian J rồi, ta chiếu cả hình dạng của Gaussian, tức ma trận hiệp phương sai Σ, chứ không chỉ riêng tâm. Kỹ thuật này có tên là EWA, Elliptical Weighted Average, của Zwicker và cộng sự năm 2001. Đây là nền tảng toán học của mọi hệ splatting hiện đại.

Công thức trong khung là Σ' bằng J nhân W nhân Σ nhân W chuyển vị nhân J chuyển vị, lấy khối 2×2, rồi cộng thêm κ nhân ma trận đơn vị. Có thể hiểu công thức theo từng lớp. W là ma trận quay từ hệ world sang hệ camera, nên W Σ Wᵀ là hình dạng Gaussian nhìn từ góc camera. Tiếp đó J (…) Jᵀ là cách một phép biến đổi tuyến tính tác động lên hiệp phương sai: nếu biến ngẫu nhiên bị biến đổi bởi ma trận A thì hiệp phương sai của nó trở thành A Σ Aᵀ. Cuối cùng, ta chỉ giữ khối 2×2 trên-trái vì chỉ cần ellipse trên mặt phẳng ảnh, chiều độ sâu được bỏ đi.

Vậy vì sao phải cộng κI? Khi Gaussian ở rất xa camera, hoặc khi ta nhìn một Gaussian dẹt gần như từ cạnh, ellipse chiếu ra có thể suy biến thành một đường gần như không có bề rộng. Lúc đó sẽ xảy ra chia cho 0 và răng cưa. Cộng κI giúp ellipse luôn rộng tối thiểu khoảng 1 pixel theo mọi hướng. Đây là một bộ lọc low-pass đơn giản và luôn được áp dụng, không phải tuỳ chọn. Một chi tiết đối chiếu code: trong CUDA kernel của repo này, file forward.cu dòng 113 đến 117, κ bằng 0.1, chứ không phải con số 0.3 hay gặp trong một số tài liệu 3DGS khác.

Việc "làm phồng" hiệp phương sai lại làm sai tổng năng lượng của Gaussian. Vì thế renderer nhân thêm vào α một hệ số hiệu chỉnh, bằng căn bậc hai của định thức Σ' trước khi cộng chia cho định thức sau khi cộng. Hệ số này nhỏ hơn 1, bù lại phần diện tích bị nới rộng để năng lượng được giữ gần đúng. Hình minh hoạ cho thấy ellipsoid 3D sau khi chiếu thành ellipse 2D trên ảnh. Có ellipse rồi, bước tiếp theo là xác định nó phủ lên vùng nào của màn hình.

## Slide 11: Compact box: xác định Gaussian phủ những tile nào
Renderer không xử lý từng pixel riêng lẻ mà chia màn hình thành các tile 16×16 pixel. Với mỗi ellipse 2D, ta phải biết nó chạm tới những tile nào. Bước này quyết định renderer nhanh hay chậm: Gaussian chạm càng nhiều tile thì càng nhiều cặp Gaussian–tile phải sắp xếp và xử lý.

3DGS gốc bao mỗi Gaussian bằng một hình vuông cố định bán kính 3σ. Với ellipse dài và mảnh, hình vuông này chứa rất nhiều diện tích thừa. Repo này dùng kỹ thuật SnugBox, nằm ở auxiliary.h dòng 313 đến 340. Tôi xin lưu ý rõ: đây là kỹ thuật kế thừa từ dòng nghiên cứu Taming và Speedy-Splat, không phải đóng góp mới của SADGS. SnugBox dựng một hộp bao khít theo đúng hình dạng ellipse, tại ngưỡng mà α nhân G bằng 1/255. Dưới mức này, đóng góp nhỏ hơn một bậc màu 8-bit nên coi như không nhìn thấy.

Công thức trong khung là t bằng mult nhân 2 ln(255α). Trực giác như sau: G có dạng mũ của âm một nửa khoảng cách Mahalanobis bình phương. Giải phương trình α·G = 1/255 ta được khoảng cách bình phương bằng 2 ln(255α). Đó chính là "bán kính" ellipse tại ngưỡng nhìn thấy. Gaussian nào có α càng lớn thì hộp càng rộng, điều này hợp lý vì nó đậm hơn nên đuôi còn thấy được ở xa hơn.

Hệ số mult cho phép siết hộp chặt thêm, mặc định là 0.7, cấu hình tại arguments/__init__.py dòng 114. Khi mult bằng 1, hộp khớp đúng ngưỡng 1/255 như SnugBox nguyên bản. Khi mult nhỏ hơn 1, ta cắt sớm hơn: ngưỡng thực ở biên trở thành σ_o nhân (255σ_o) mũ âm mult, lớn hơn 1/255. Nghĩa là ta chấp nhận bỏ một phần đuôi Gaussian vẫn còn nhìn thấy được để đổi lấy tốc độ, vì mỗi Gaussian chạm ít tile hơn. Hình minh hoạ so sánh hộp bao khít với hộp vuông truyền thống.

Biết Gaussian phủ vùng nào rồi, ta cần biết tại mỗi pixel trong vùng đó nó che phủ bao nhiêu.

## Slide 12: Trường alpha: mỗi Gaussian che phủ pixel bao nhiêu?
Một Gaussian không đặc đều như một đĩa màu. Nó đậm nhất ở tâm và nhạt dần ra biên theo đúng hàm G. Vì thế tại mỗi pixel x, độ mờ đục hiệu dụng của Gaussian thứ n là α_n(x) bằng min của 0.99 và α_n nhân G_n(x).

Ta cần tách bạch hai đại lượng. α_n không có đối số x là tham số học được, cố định cho cả Gaussian, thể hiện độ "đặc" tối đa của nó. G_n(x) là giá trị hàm Gaussian 2D sau khi chiếu, thay đổi theo từng pixel: bằng 1 tại tâm ellipse và giảm dần khi đi ra ngoài. Nhân hai đại lượng này lại, ta được một "trường alpha" trên màn hình, đúng như hình minh hoạ: một vùng sáng ở tâm và mờ dần theo hình ellipse.

Chi tiết đáng chú ý là chặn trên ở 0.99 thay vì 1. Nếu cho một Gaussian đạt α bằng đúng 1 thì nó che kín hoàn toàn pixel. Mọi Gaussian phía sau khi đó bị nhân với 1 − 1 = 0, không đóng góp vào ảnh, và quan trọng hơn là gradient của chúng biến mất hẳn. Chúng không thể học được nữa. Giữ lại một khe hở nhỏ 0.01 giúp tín hiệu gradient vẫn lan truyền được về phía sau. Đây là một mẹo nhỏ nhưng có ý nghĩa cho tính ổn định của quá trình tối ưu.

Tôi nhấn mạnh lại một lần: đại lượng α_n(x) phụ thuộc pixel này mới là thứ đi vào công thức alpha blending ở slide tiếp theo, không phải hằng số α_n của Gaussian. Vậy khi nhiều Gaussian cùng chồng lên một pixel thì màu cuối cùng được tính ra sao? Đó là nội dung của slide tiếp theo.

## Slide 13: Alpha blending: trộn nhiều lớp Gaussian theo thứ tự trước-sau
Tại một pixel thường có nhiều Gaussian chồng lên nhau theo chiều sâu. Màu cuối cùng được trộn tuần tự từ Gaussian gần camera nhất ra xa nhất. Kỹ thuật này gọi là front-to-back alpha blending, giống như ta xếp chồng nhiều tấm kính màu bán trong suốt rồi nhìn xuyên qua chúng.

Công thức: C(x) bằng tổng theo n của c_n nhân α_n(x) nhân T_n(x), cộng T_final(x) nhân màu nền C_bg. Mỗi Gaussian góp phần màu c_n, với trọng số gồm hai yếu tố. α_n(x) là mức nó tự che phủ pixel, như ta vừa thấy ở slide trước. T_n(x) là transmittance, tức tỉ lệ ánh sáng còn lại chưa bị các lớp phía trước hấp thụ. T_n bằng tích của (1 − α_j) với mọi j đứng trước n. Mỗi lớp phía trước "giữ lại" một phần α_j và cho đi qua phần 1 − α_j. Như vậy Gaussian thứ n chỉ đóng góp đúng phần ánh sáng mà nó nhận được sau khi đi qua n − 1 lớp trước. Hình minh hoạ cho thấy các lớp xếp chồng và phần đóng góp nhỏ dần theo độ sâu.

Số hạng màu nền T_final nhân C_bg là bắt buộc, không được bỏ qua. Ở những vùng thưa Gaussian, tổng độ che phủ dọc tia nhìn có thể không đạt 1, nên vẫn còn một phần ánh sáng "lọt" qua hết các lớp. Phần đó phải được lấp bằng màu nền. Nếu bỏ đi, những vùng này sẽ bị tối hơn thực tế và ảnh render sai.

Nhìn công thức ta thấy T_n là tích của nhiều thừa số nhỏ hơn 1. Tính chất này dẫn tới một tối ưu quan trọng, là chủ đề của slide tiếp theo.

## Slide 14: Transmittance suy giảm: cơ sở cho việc dừng sớm
Quay lại định nghĩa T_n là tích của các thừa số (1 − α_j(x)), mỗi thừa số nhỏ hơn hoặc bằng 1. Nhân càng nhiều thừa số như vậy thì T_n càng nhỏ, và giảm theo cấp số nhân. Hiện tượng này đặc biệt rõ khi từng α_j đều đáng kể. Đường cong trong hình minh hoạ cho thấy điều đó: T lao xuống rất nhanh sau vài lớp Gaussian đầu tiên.

Hệ quả là sau đủ nhiều lớp, T_n gần như bằng 0. Các Gaussian phía sau dù có màu gì cũng bị nhân với một hệ số gần 0, nên hầu như không ảnh hưởng tới màu pixel C(x). Nói theo ngôn ngữ thường ngày: khi phía trước đã có một bức tường gần như đặc thì những gì sau bức tường không còn nhìn thấy nữa.

Renderer tận dụng điều này để dừng sớm, gọi là early termination. Tại mỗi pixel, ngay khi T nhân (1 − α) nhỏ hơn 10 mũ âm 4, renderer ngừng xử lý thêm Gaussian cho pixel đó. Ta kiểm tra T sau khi đã trừ lớp hiện tại, tức lượng ánh sáng sẽ còn lại cho các lớp tiếp theo. Nếu lượng đó đã nhỏ hơn một phần mười nghìn thì mọi đóng góp tiếp theo đều nằm dưới ngưỡng nhận biết. Nhờ vậy tiết kiệm được đáng kể thời gian tính toán mà hầu như không ảnh hưởng chất lượng ảnh.

Điều này cũng giải thích vì sao thứ tự theo độ sâu rất quan trọng. Dừng sớm chỉ đúng khi ta thật sự duyệt từ gần ra xa. Nếu sắp sai thứ tự, renderer có thể dừng trước khi gặp một Gaussian lẽ ra ở gần camera hơn và vẫn còn đóng góp đáng kể, và ảnh sẽ bị sai. Đến đây ta đã có trọn chuỗi render: chiếu tâm, chiếu hình dạng, xác định tile, tính trường alpha, trộn màu và dừng sớm. Chúng ta chuyển sang phần tiếp theo.

## Slide 15: Kiến trúc rasterizer theo tile: vì sao render nhanh
Ở các slide trước, chúng ta đã biết cách chiếu một Gaussian 3D xuống màn hình, tính khung bao compact box cho nó, và trộn màu bằng alpha blending. Giờ đến câu hỏi thực tế: làm sao làm tất cả việc đó thật nhanh? Một cảnh có thể có tới hàng triệu Gaussian, và một khung hình cũng có hàng triệu pixel. Cách làm ngây thơ là với mỗi pixel, duyệt qua toàn bộ Gaussian để xem cái nào ảnh hưởng tới nó. Làm vậy thì không thể nào đạt tốc độ thời gian thực.

3DGS giải quyết bằng cách chia màn hình thành các ô vuông 16×16 pixel, gọi là tile, rồi xử lý theo ba bước.

Bước một là xác định phạm vi. Với mỗi Gaussian, ta dùng compact box ở slide trước để biết nó phủ lên những tile nào. Mỗi tile bị phủ sẽ sinh ra một cặp khoá gồm chỉ số tile và độ sâu. Như vậy một Gaussian lớn phủ nhiều tile sẽ sinh ra nhiều cặp khoá, mỗi tile một cặp.

Bước hai là sắp xếp, và chỉ sắp xếp một lần. Toàn bộ danh sách cặp khoá được sắp theo tile trước, rồi đến độ sâu. Về lý thuyết thì đây là bước tốn kém nhất. Nhưng mấu chốt là nó chỉ chạy một lần cho cả khung hình, chứ không phải mỗi pixel tự sắp xếp một lần như cách làm ngây thơ.

Bước ba là xử lý song song theo tile. Trên GPU, mỗi thread block phụ trách đúng một tile. Vì danh sách của tile đó đã được sắp sẵn từ gần ra xa, mỗi thread, tức mỗi pixel, chỉ cần đi lần lượt và áp dụng công thức alpha blending đúng thứ tự. Khi độ truyền qua T đã đủ nhỏ, nghĩa là pixel gần như đã bị che kín, thread dừng sớm luôn.

Kết quả là tổng chi phí chỉ còn xấp xỉ O(N·K), trong đó K là số tile trung bình mà một Gaussian chạm tới. K lại phụ thuộc trực tiếp vào hệ số mult của compact box: box càng chặt thì K càng nhỏ. So với O(N·H·W) khi mỗi Gaussian phải quét cả ảnh, đây là khác biệt rất lớn, và cũng là lý do cốt lõi giúp 3DGS render được theo thời gian thực.

Render được ảnh rồi, câu hỏi tiếp theo là: ta đo xem ảnh đó đúng hay sai như thế nào để huấn luyện?

## Slide 16: Hàm mất mát: so khớp ảnh render với ảnh thật
Sau khi rasterizer tạo ra ảnh render, ta cần một con số cho biết ảnh này khác ảnh thật I_gt bao nhiêu. Con số đó là hàm mất mát, và gradient của nó sẽ lan ngược qua rasterizer để cập nhật vị trí, hình dạng, màu sắc và độ mờ của từng Gaussian.

Trong repo này, hàm mất mát nằm ở train.py, dòng 256 đến 259, gồm ba số hạng: (1 − λ_dssim) nhân L1, cộng λ_dssim nhân (1 − SSIM), cộng λ_L2 nhân L2. Ở đây λ_dssim bằng 0.2 và λ_L2 bằng 2.0. 3DGS gốc chỉ có hai số hạng đầu. Bản của chúng tôi cộng thêm số hạng L2, định nghĩa ở utils/loss_utils.py dòng 25.

Vì sao lại cần cả ba? Mỗi số hạng lo một việc khác nhau.

L1 là trung bình trị tuyệt đối của sai số. Vì gradient của nó có độ lớn không đổi, vài pixel sai rất nhiều do nhiễu hay che khuất sẽ không kéo cả quá trình tối ưu lệch đi. Nói cách khác, L1 ổn định trước các điểm ngoại lai.

SSIM không so từng pixel riêng lẻ mà so theo cửa sổ Gaussian 11×11 với σ bằng 1.5. Nó xét độ chói, độ tương phản và cấu trúc của từng vùng nhỏ, nên gần với cách mắt người cảm nhận hơn. Ta dùng 1 − SSIM vì SSIM bằng 1 khi hai ảnh khớp hoàn toàn, còn mất mát thì cần nhỏ khi ảnh khớp.

L2 bình phương sai số, nên sai số lớn bị phạt nặng hơn hẳn. Ở giai đoạn đầu, khi ảnh render còn rất khác ảnh thật, số hạng này giúp mô hình hội tụ nhanh hơn.

Hình minh hoạ trên slide cho thấy cách các thành phần này được pha trộn với nhau.

Có một lưu ý quan trọng khi đối chiếu với code. loss_utils.py còn định nghĩa thêm hai hàm là tone_curve_loss và frequency_loss, trong đó frequency_loss dựa trên structure tensor mà ta sẽ gặp ở phần SADGS. Tuy nhiên, không có dòng nào trong train.py gọi tới hai hàm này. Chúng là dead code: có trong file nhưng hoàn toàn không ảnh hưởng tới hàm mất mát đang được tối ưu.

Hàm mất mát phục vụ việc huấn luyện. Còn để đánh giá kết quả một cách khách quan, ta cần những chỉ số riêng, và chỉ số đầu tiên là PSNR.

## Slide 17: Đánh giá chất lượng (1/3): PSNR
Khi huấn luyện xong, ta không thể lấy chính hàm mất mát để khẳng định mô hình tốt. Mô hình đã được tối ưu trực tiếp trên đại lượng đó, và trên chính những ảnh đó. Vì vậy ta cần những chỉ số độc lập, đo trên tập test, tức là những góc nhìn mà mô hình chưa từng thấy khi train. Chỉ số đầu tiên và đơn giản nhất là PSNR, viết tắt của Peak Signal-to-Noise Ratio, tức tỉ số tín hiệu đỉnh trên nhiễu.

Công thức là PSNR = 10·log₁₀(1/MSE), đơn vị là decibel. Ảnh ở đây được chuẩn hoá về khoảng [0, 1], nên giá trị đỉnh của tín hiệu là 1, và tử số trong công thức chính là 1. MSE là sai số bình phương trung bình trên các pixel. Trực giác rất đơn giản: MSE nằm ở mẫu số, nên sai số càng nhỏ thì PSNR càng cao. Hàm log đưa chỉ số về thang logarit. Vì thế chỉ cần chênh lệch vài dB là đã ứng với một khác biệt đáng kể về sai số. Khi đọc bảng kết quả, đừng xem nhẹ những chênh lệch trông có vẻ nhỏ.

Nhưng PSNR có một hạn chế cơ bản: nó coi mọi pixel như nhau. Nó không phân biệt được sai ở vùng nhiều chi tiết với sai ở vùng phẳng, đơn giản. Hệ quả là hai ảnh có cùng PSNR vẫn có thể trông rất khác nhau. Ví dụ, một ảnh bị mờ đều trên toàn khung, còn ảnh kia sắc nét gần như hoàn toàn nhưng có vài vệt lỗi rõ rệt. Tính trung bình bình phương thì hai sai số có thể bằng nhau, nhưng mắt người đánh giá hai ảnh hoàn toàn khác. Hình minh hoạ trên slide giúp ta hình dung mối quan hệ giữa sai số và PSNR.

Chính vì hạn chế này, PSNR gần như không bao giờ được báo cáo một mình, mà luôn đi kèm SSIM và LPIPS. Ta chuyển sang SSIM trước, chỉ số giải quyết đúng điểm yếu "chỉ nhìn từng pixel" của PSNR.

## Slide 18: Đánh giá chất lượng (2/3): SSIM
SSIM, tức Structural Similarity Index, chính là công thức mà ta vừa gặp trong hàm mất mát. Điểm khác là ở đây ta dùng nó để đánh giá, không phải để lấy gradient.

Ý tưởng của SSIM là không so từng pixel riêng lẻ, mà trượt một cửa sổ cục bộ lên ảnh. Cửa sổ này là Gaussian kích thước 11×11 với σ bằng 1.5. Tại mỗi vị trí, SSIM so hai ảnh theo ba khía cạnh. Thứ nhất là độ chói, tức vùng này sáng hay tối như nhau không. Thứ hai là độ tương phản, tức mức dao động sáng tối trong vùng có giống nhau không. Thứ ba là cấu trúc, tức các thay đổi sáng tối có đi cùng hướng, cùng hình dạng không. Hình trên slide minh hoạ cửa sổ trượt này. Cửa sổ dùng trọng số Gaussian nên các pixel ở tâm được coi trọng hơn các pixel ở rìa.

Giá trị SSIM nằm trong khoảng [−1, 1]. Với ảnh tự nhiên, nó thường được báo cáo trong khoảng [0, 1], và bằng 1 nghĩa là khớp hoàn hảo.

Vì sao SSIM gần với cảm nhận của con người hơn PSNR? Vì nó nhìn theo vùng. Khi một đường nét bị mờ đi hay một hoạ tiết bị lệch, cấu trúc cục bộ thay đổi, và SSIM phát hiện ra ngay. Ngược lại, nếu cả ảnh chỉ sáng hơn hoặc tối hơn một chút, cấu trúc gần như giữ nguyên, nên SSIM không phạt nặng. Mắt người cũng hoạt động giống vậy: ta để ý đường nét và hoạ tiết nhiều hơn là một chút lệch độ sáng tổng thể.

Trong 3DGS, SSIM đóng vai trò kép. Lúc train, nó là một phần của hàm mất mát và khuyến khích mô hình khớp cấu trúc. Lúc test, nó là một chỉ số đánh giá.

Tuy vậy, SSIM vẫn là một công thức toán học viết tay. Để có bức tranh đầy đủ, ta cần thêm một chỉ số "học" được cách con người nhìn ảnh. Đó là nội dung của slide tiếp theo.

## Slide 19: Đánh giá chất lượng (3/3): kết hợp nhiều chỉ số
Slide này ghép ba chỉ số lại thành một bức tranh chung. Ngoài PSNR và SSIM, chỉ số thứ ba là LPIPS, viết tắt của Learned Perceptual Image Patch Similarity. Thay vì so pixel, LPIPS đưa cả hai ảnh qua một mạng CNN đã được huấn luyện trước, rồi đo khoảng cách giữa hai ảnh trong không gian đặc trưng của mạng đó. Mạng này học từ dữ liệu đánh giá thị giác của con người, nên khoảng cách LPIPS phản ánh "con người thấy hai ảnh khác nhau đến mức nào".

Ba chỉ số này đo những khía cạnh khác nhau và bổ sung cho nhau.

PSNR đo sai số pixel thô. Nó dễ tính và dễ hiểu, nhưng ít tương quan với cảm nhận thị giác, như ta đã thấy ở ví dụ ảnh mờ đều so với ảnh có vệt lỗi.

SSIM đo cấu trúc cục bộ và tương quan với cảm nhận tốt hơn. Nhưng nó vẫn là một công thức toán học tường minh, không học gì từ cách con người nhìn ảnh.

LPIPS học trực tiếp từ dữ liệu đánh giá của con người, nên bắt được những khác biệt mà mắt người nhận ra rõ nhưng PSNR và SSIM có thể bỏ sót. Ví dụ, một chi tiết bị làm mờ nhẹ nhưng cấu trúc trung bình vẫn đúng: về mặt con số, PSNR và SSIM có thể vẫn đẹp, nhưng người xem sẽ thấy ảnh kém sắc nét.

Hình minh hoạ trên slide tách điểm số tổng hợp thành các thành phần, để thấy mỗi chỉ số đóng góp một góc nhìn riêng.

Thông điệp chính của slide là không có chỉ số nào đủ để dùng một mình. Một báo cáo kết quả nghiêm túc cần nêu cả ba chỉ số. Ngoài ra phải ghi rõ scene nào, checkpoint ở iteration bao nhiêu, và cấu hình đo ra sao. Nếu thiếu những thông tin này, việc so sánh giữa các phương pháp rất dễ trở nên khập khiễng, vì các con số được đo trong những điều kiện khác nhau.

Đến đây ta đã có đủ công cụ nền tảng: biểu diễn, render, huấn luyện và đánh giá. Tiếp theo, ta sẽ xem 3DGS điều chỉnh số lượng Gaussian trong quá trình huấn luyện như thế nào.

## Slide 20: Lan truyền gradient: từ loss ngược về từng tham số Gaussian
Ở mấy slide trước, ta đã đi hết chiều thuận: chiếu Gaussian từ 3D xuống màn hình, xấp xỉ hiệp phương sai 2D bằng EWA, rồi alpha blending để ra màu cho từng pixel. Có một tính chất của cả chuỗi này mà ta chưa nhấn mạnh: mọi bước đều khả vi. Nhờ đó ta không cần tự tay viết đạo hàm cho từng tham số, vì PyTorch cùng renderer CUDA sẽ áp dụng quy tắc chuỗi để tính.

Công thức trên slide nói rằng đạo hàm của loss theo một tham số θ (có thể là vị trí μ, scale s, quaternion q, opacity α hay hệ số màu cầu điều hoà k) bằng tổng qua mọi pixel x. Mỗi số hạng trong tổng là tích của hai thứ: loss nhạy thế nào với màu C(x) của pixel, và màu pixel đó nhạy thế nào với θ. Vì sao lại là tổng? Vì một Gaussian phủ lên nhiều pixel, nên mỗi pixel "góp một phiếu" vào hướng tham số cần thay đổi. Thứ tự lan ngược đúng bằng chiều thuận đi ngược lại: từ alpha blending về α_n(x) = min(0.99, α_n·G_n(x)), về giá trị Gaussian G_n(x), về Σ′ và μ′ trên màn hình, về Σ và μ trong 3D, cuối cùng về s và q.

Điều quan trọng nhất của slide nằm ở gạch đầu dòng thứ hai. Gradient theo vị trí chiếu μ′ không rải đều trên footprint mà tập trung ở vùng biên của Gaussian trong ảnh, tức nơi cường độ ảnh thay đổi nhanh nhất. Hình minh hoạ cho thấy đúng điều đó. Trực giác khá đơn giản: ở vùng màu phẳng, dịch Gaussian đi một chút thì màu render gần như không đổi, nên gradient nhỏ. Ở cạnh sắc, chỉ cần lệch một chút là màu thay đổi ngay, nên gradient lớn.

Đại lượng ∂L/∂μ′ này chính là tín hiệu mà Adaptive Density Control dùng để đánh giá một Gaussian đang thiếu hay thừa hình học. Tuy vậy tín hiệu này có một vấn đề khá tinh vi. Trước khi nói đến vấn đề đó, ta cần xem gradient này được tích luỹ ra sao qua nhiều view.

## Slide 21: Tích luỹ gradient qua nhiều view: lấy CHUẨN trước, cộng sau
Trong một epoch, tức một lượt đi qua toàn bộ camera train, cùng một Gaussian thường được nhìn thấy từ nhiều góc khác nhau. Mỗi lần render cho ra một gradient khác nhau, và để quyết định có densify hay không thì ta phải gộp các gradient đó lại. Việc này nằm trong hàm add_densification_stats, ở file scene/gaussian_model.py, dòng 1054 đến 1057.

Công thức đóng khung nói rằng mỗi lần Gaussian i được render, ta lấy chuẩn L2 của gradient theo vị trí chiếu μ′_i rồi cộng vào biến xyz_gradient_accum. Đồng thời denom đếm số lần Gaussian đó được nhìn thấy. Giá trị cuối cùng là trung bình ḡ_i bằng accum chia denom. Nhìn hình minh hoạ, bạn có thể hình dung hai "thùng chứa": một thùng cộng dồn độ lớn gradient, một thùng đếm số lần xuất hiện.

Chi tiết cần nhớ là thứ tự của hai phép toán: chuẩn được lấy ngay tại mỗi view, sau đó mới cộng. Code không cộng các vector gradient của nhiều view lại rồi mới lấy chuẩn một lần ở cuối. Hệ quả toán học rất rõ: chuẩn của một vector luôn lớn hơn hoặc bằng 0, nên mọi số hạng được cộng vào accum đều không âm. Cộng các số không âm thì tổng chỉ tăng chứ không giảm, nghĩa là gradient từ các view khác nhau không thể triệt tiêu nhau trong phép cộng này. Đây là tiền đề để ta sửa một hiểu lầm phổ biến ở slide sau.

Ngoài ra, repo này còn tích luỹ song song một kênh thứ hai là xyz_gradient_accum_abs, lấy trị tuyệt đối. Kênh này làm tín hiệu cho ngưỡng grad_abs_thresh khi quyết định split, còn ngưỡng grad_thresh thì dùng cho clone, hai ngưỡng hoàn toàn tách biệt. Nếu đã có kênh chuẩn rồi thì cần thêm kênh thứ hai để làm gì? Slide tiếp theo sẽ trả lời câu hỏi này.

## Slide 22: Vấn đề "triệt tiêu gradient": xảy ra GIỮA CÁC PIXEL, không phải giữa các view
Slide này sửa một điểm mà nhiều tài liệu hiểu sai, nên tôi đề nghị các bạn tập trung.

Cách hiểu sai thường gặp là thế này: "camera A kéo Gaussian sang trái, camera B kéo sang phải, cộng lại thì triệt tiêu, nên gradient tích luỹ bị nhỏ." Nghe có lý, nhưng với pipeline ta vừa xem thì điều này không thể xảy ra. Như slide trước đã nói, chuẩn được lấy tại từng view trước khi cộng. Từ camera A ta nhận một số dương, từ camera B cũng nhận một số dương. Đến lúc cộng thì thông tin về hướng đã bị bỏ đi, không còn "trái" hay "phải" nữa, chỉ còn độ lớn. Tổng các số không âm không thể nhỏ hơn bất kỳ số hạng nào, nên nói gradient "triệt tiêu giữa các view" là sai.

Vậy triệt tiêu thật sự xảy ra ở đâu? Nó xảy ra bên trong một lần render duy nhất, giữa các pixel thuộc footprint của cùng một Gaussian. Nhớ lại công thức ở slide 20: gradient theo μ′ là tổng theo pixel, và tổng này được tính trên các vector có dấu, trước khi lấy chuẩn. Lấy ví dụ trên hình: ảnh thật có một cạnh chạy ngang qua giữa Gaussian. Các pixel nửa bên trái muốn kéo μ′ sang trái, các pixel nửa bên phải muốn kéo sang phải. Riêng từng pixel thì gradient đều lớn, nhưng khi cộng vector lại thì hai lực ngược chiều gần như bù trừ nhau, và vector tổng rất nhỏ. Đến lúc lấy chuẩn thì đã muộn, vì sự triệt tiêu đã xảy ra rồi.

Hệ quả khá nguy hiểm. Một Gaussian thực sự đang thiếu chi tiết hình học, nhưng lại đối xứng theo một trục nào đó, sẽ bị đánh giá nhầm là "gradient thấp, không cần densify". Đây là điểm mù của cơ chế chỉ dùng gradient có dấu.

Cách khắc phục là kỹ thuật AbsGS. Ngay trong backward pass, renderer CUDA tính thêm một kênh cộng trị tuyệt đối gradient của từng pixel, trước khi các dấu đối lập kịp triệt tiêu nhau. Đó chính là kênh xyz_gradient_accum_abs đã nhắc ở slide trước. Tóm lại, thứ cần sửa là phép cộng giữa các pixel, còn phép cộng giữa các view vốn không có vấn đề gì.

## Slide 23: Adaptive Density Control (ADC): khung chung
Giờ ta đã có hai tín hiệu: gradient trung bình ḡ và gradient trị tuyệt đối ḡ_abs. Câu hỏi tiếp theo là dùng chúng như thế nào. Đó là việc của Adaptive Density Control, gọi tắt là ADC.

Về lịch chạy, ADC không chạy ở mọi iteration. Cứ mỗi densification_interval bằng 100 iteration, 3DGS mới xét lại mật độ Gaussian một lần. Quá trình này bắt đầu từ densify_from_iter bằng 500, nên lần densify đầu tiên thực sự rơi vào iteration 600, và kéo dài tới densify_until_iter bằng 15.000. Sau mốc đó số Gaussian ổn định, optimizer chỉ còn tinh chỉnh tham số.

Mỗi lần chạy, ADC có ba thao tác như trong bảng. Clone áp dụng khi gradient cao và Gaussian đang nhỏ, để bù under-reconstruction, tức vùng đang thiếu hình học. Split áp dụng khi gradient cao nhưng Gaussian đang to, để xử lý over-reconstruction, tức một Gaussian quá to làm nhoè chi tiết. Prune thì loại bỏ những Gaussian có opacity quá thấp hoặc kích thước quá lớn, vì chúng không còn đóng góp gì.

"To" hay "nhỏ" được so với ngưỡng percent_dense nhân extent, trong đó extent là kích thước cảnh đã định nghĩa ở phần đầu. Gaussian có trục dài nhất không vượt quá ngưỡng này thì thuộc diện clone, lớn hơn thì thuộc diện split.

Trực giác phía sau khá gần với chuyện điều quân. Một Gaussian nhỏ mà gradient vẫn cao nghĩa là ở đó còn thiếu chi tiết và một mình nó không đủ, nên ta nhân bản để tăng quân số. Một Gaussian to mà gradient vẫn cao nghĩa là nó đang cố phủ một vùng có nhiều chi tiết hơn khả năng của một hình elip, nên ta chia nhỏ nó ra. Gradient cho biết chỗ nào cần can thiệp, còn kích thước cho biết nên can thiệp theo cách nào. Tiếp theo ta xem chi tiết thao tác đầu tiên là clone.

## Slide 24: Clone: sao chép y nguyên, KHÔNG lệch vị trí
Khi một Gaussian nhỏ có gradient vượt ngưỡng, hàm densify_and_clone làm một việc đơn giản đến mức nhiều người bất ngờ: nó sao chép y hệt mọi tham số sang một Gaussian con mới. Con có cùng vị trí, cùng scale, cùng rotation, cùng màu với cha. Công thức trên slide ghi θ_con bằng θ_cha, không cộng thêm epsilon nào và cũng không lấy mẫu ngẫu nhiên. Nhiều người cứ nghĩ clone sẽ đặt bản sao lệch đi một chút, nhưng trong code thì không có bước đó.

Từ đây nảy ra một câu hỏi rất tự nhiên: hai bản sao giống hệt nhau thì làm sao tách ra để bổ sung chi tiết, thay vì trùng khít nhau mãi? Câu trả lời nằm ở optimizer. Ngay sau khi sinh ra, hai Gaussian được tối ưu độc lập, mỗi Gaussian có trạng thái Adam riêng gồm exp_avg và exp_avg_sq. Ở các bước gradient descent sau đó, dù xuất phát từ cùng một điểm, chúng chịu nhiễu số học và các tương tác nhỏ khác nhau với những Gaussian lân cận. Dần dần hai Gaussian bị đẩy ra hai hướng khác nhau một cách tự nhiên và cùng phủ tốt hơn vùng đang thiếu chi tiết. Hình trước và sau trên slide minh hoạ quá trình này: lúc đầu chỉ có một Gaussian, sau khi clone và train tiếp thì có hai Gaussian cùng lấp vùng trống.

Chi tiết cuối cùng là về các bộ đếm thống kê như xyz_gradient_accum và denom. Của bản sao mới được reset về 0 ngay khi sinh ra. Từ thời điểm đó, lịch sử theo dõi của bản sao tách biệt hoàn toàn với cha, và nó phải tự tích luỹ gradient của riêng mình trước khi được xét densify tiếp.

Clone là cách xử lý Gaussian nhỏ. Còn với Gaussian to thì sao chép y nguyên không giải quyết được gì, ta phải chia nó ra. Đó là nội dung của thao tác split ở slide tiếp theo.

## Slide 25: Split nguyên bản: lấy mẫu ngẫu nhiên quanh Gaussian cha
Ở slide trước ta đã thấy clone, tức là nhân bản những Gaussian nhỏ đang thiếu chi tiết. Slide này nói về trường hợp ngược lại: Gaussian đã to mà vẫn khớp chưa tốt. Cụ thể, trục dài nhất của nó vượt ngưỡng percent_dense nhân với extent của cảnh, và gradient trị tuyệt đối vẫn còn cao. Tình huống này giống như dùng một nét cọ quá to để vẽ một vùng có nhiều chi tiết: nhân bản nét cọ đó cũng vô ích, phải thay nó bằng những nét nhỏ hơn. Việc này do hàm densify_and_split làm, nằm ở dòng 866 đến 889 của gaussian_model.py.

Công thức trên slide có hai phần. Phần thứ nhất là vị trí của con: lấy tâm cha μ cộng với R(q) nhân ε, trong đó ε được lấy mẫu từ một phân phối chuẩn có độ lệch chuẩn đúng bằng scale của cha. Vì ε còn được xoay theo R(q), các con không rơi đều trong một hình cầu mà nằm rải rác bên trong đúng hình ellipsoid của cha: trục nào của cha dài thì con có thể rơi xa theo trục đó. Hình minh hoạ bên dưới cho thấy điều này, hai con nằm trong vùng mà cha từng chiếm.

Phần thứ hai là scale: mỗi con có scale bằng scale của cha chia cho 0.8 nhân N. Với N = 2, hệ số này là 1.6. Điểm cần chú ý là hệ số 1.6 được áp như nhau trên cả ba trục, dù trục gây ra vấn đề có thể chỉ là một trục. Đây là một cách thu nhỏ đẳng hướng khá "thô", và cũng chính là chỗ mà cơ chế split dị hướng của SADGS ở phần sau sẽ cải tiến.

Cuối cùng, cha bị xoá ngay sau khi sinh hai con, nên mỗi lần split tổng số Gaussian chỉ tăng thêm đúng một.

Vậy là ta đã có đủ công cụ để thêm Gaussian. Câu hỏi tiếp theo là làm sao bớt những Gaussian vô dụng, và để trả lời câu hỏi đó ta cần nhìn kỹ hơn vào opacity.

## Slide 26: Vì sao lưu opacity dưới dạng logit? Và cơ chế reset định kỳ
Trước hết là một chi tiết cài đặt nhỏ nhưng quan trọng. Opacity α phải nằm trong khoảng từ 0 đến 1. Nếu tối ưu α trực tiếp, gradient descent có thể đẩy nó vượt ra ngoài khoảng này và ta sẽ phải clamp thủ công. Thay vào đó, code lưu logit o, là nghịch đảo sigmoid của α, tức ln của α chia cho 1 trừ α. Logit có thể nhận bất kỳ giá trị thực nào. Mỗi khi cần dùng α, ta chỉ việc cho o đi qua sigmoid. Cách làm này giống hệt lý do ta dùng hàm exp cho scale: mọi số thực đều ánh xạ ra một giá trị hợp lệ, nên optimizer được hoạt động tự do. Hình bên dưới vẽ cặp hàm sigmoid và sigmoid nghịch đảo để minh hoạ phép ánh xạ này.

Phần thứ hai là reset opacity định kỳ. Nếu cứ để tự nhiên, nhiều Gaussian sẽ "chết dần": opacity giảm rất thấp, chúng gần như không đóng góp gì cho ảnh nhưng vẫn tồn tại lâu và chiếm bộ nhớ. Vì vậy cứ mỗi 3000 iteration, với điều kiện iteration vẫn còn nhỏ hơn densify_until_iter là 15.000, toàn bộ opacity bị nhân với 0.1. Cần lưu ý điều kiện này: reset chỉ xảy ra đúng bốn lần, ở các mốc 3k, 6k, 9k và 12k.

Trực giác đằng sau là một phép thử. Sau khi mọi Gaussian cùng bị làm mờ, Gaussian nào thực sự giúp giảm loss sẽ được gradient kéo opacity lên lại chỉ sau vài trăm bước. Gaussian nào không còn đóng góp thì opacity nằm yên ở mức thấp và sẽ bị prune ở đợt dọn dẹp kế tiếp.

Một hệ quả cần nhớ: sau iteration 15.000, reset dừng hẳn, nên nửa sau của quá trình train không còn cơ chế nào chủ động sàng lọc lại opacity nữa.

Để thấy phép thử này diễn ra thế nào trong thực tế, ta hãy theo dõi opacity của một Gaussian cụ thể.

## Slide 27: Quan sát quỹ đạo opacity của một Gaussian qua nhiều lần reset
Hình trên slide vẽ α theo thời gian của một Gaussian trong suốt quá trình train, và ta có thể thấy rõ "nhịp điệu" của cơ chế reset. Mỗi lần chạm một mốc reset (3k, 6k, 9k, 12k), đường cong bị "đập" xuống đột ngột, chỉ còn 10% giá trị ngay trước đó. Sau đó đường cong leo dần trở lại, với điều kiện Adam vẫn còn nhận được gradient khuyến khích tăng opacity tại vị trí đó. Hình dạng răng cưa này chính là dấu vết của reset.

Có hai kịch bản khác nhau. Với một Gaussian quan trọng, tức là đang thực sự giúp giảm loss, α hồi phục nhanh sau mỗi lần reset và thường vượt lại mức cũ chỉ trong vài trăm iteration. Lý do là loss "cần" Gaussian này: khi nó mờ đi thì ảnh render bị sai, nên gradient đẩy opacity lên mạnh.

Ngược lại, với một Gaussian dư thừa, chẳng hạn vì một Gaussian lân cận đã "lấn sân" và đảm nhận vùng đó, việc làm mờ nó hầu như không ảnh hưởng đến loss. Gradient kéo nó lên rất yếu, nên α hồi phục rất chậm hoặc gần như không hồi phục. Kết quả là ở lần dọn tiếp theo, nó nằm dưới ngưỡng prune và bị loại bỏ.

Vì vậy có thể xem reset là một dạng "kiểm tra sức khỏe" định kỳ cho cả quần thể Gaussian. Điều quan trọng là cơ chế này tách biệt hẳn với clone, split và prune đã nói ở các slide trước. Những cơ chế kia dựa trên gradient vị trí hoặc kích thước, còn reset thì kiểm tra xem mỗi Gaussian có còn "đáng giữ" hay không.

Nhưng một cá thể chưa đủ để kết luận. Slide tiếp theo sẽ nhìn trên toàn bộ quần thể.

## Slide 28: Nhìn trên toàn bộ quần thể: phân bố opacity trước và sau reset
Bây giờ ta không theo dõi một Gaussian nữa mà nhìn histogram opacity của tất cả Gaussian, lấy ở ngay trước và ngay sau một lần reset. Hình trên slide so sánh hai thời điểm này.

Ngay sau reset, cả phân bố bị nén mạnh về phía các giá trị thấp. Điều này dễ hiểu: mọi α đều bị nhân với 0.1, nên kể cả Gaussian gần như đục hoàn toàn cũng chỉ còn opacity khoảng 0.1. Nói cách khác, gần như toàn bộ cảnh tạm thời "mờ đi" đồng loạt. Ở thời điểm này, ta chưa phân biệt được Gaussian nào tốt, Gaussian nào xấu, vì tất cả đều bị đối xử như nhau.

Điều thú vị nằm ở các iteration sau reset. Phân bố dần tách thành hai nhóm. Nhóm thứ nhất hồi phục nhanh về vùng opacity cao, đó là những Gaussian hữu ích. Nhóm thứ hai tiếp tục nằm lại ở vùng thấp, và đó là các ứng viên cho prune. Chính quá trình tối ưu đã tự làm việc phân loại: phép reset chỉ tạo ra một điểm xuất phát chung, còn gradient quyết định ai được "sống lại".

Hiện tượng này chính là hiệu ứng sàng lọc ta đã mô tả ở hai slide trước, chỉ khác là giờ ta nhìn nó dưới góc độ thống kê quần thể thay vì một cá thể. Về mặt thực hành, đây là một công cụ chẩn đoán khá tiện: khi chạy huấn luyện một cảnh cụ thể, ta có thể vẽ histogram này để kiểm chứng xem reset có thực sự đang phân loại Gaussian hữu ích và dư thừa hay không. Nếu phân bố không tách thành hai nhóm thì có lẽ cơ chế này không phát huy tác dụng như mong đợi.

Đến đây ta đã đi qua gần hết các thành phần của 3DGS gốc. Hãy ghép chúng lại thành một bức tranh chung.

## Slide 29: Tổng kết: pipeline 3DGS gốc, đầu-cuối
Slide này gộp toàn bộ phần vừa trình bày thành một vòng lặp duy nhất. Ta hãy đọc chuỗi mũi tên từ trái sang phải. Đầu tiên, SfM/COLMAP cho ta một đám mây điểm thưa để khởi tạo tập Gaussian, mỗi Gaussian có vị trí μ, hiệp phương sai Σ, opacity α và màu c. Tiếp theo, phép chiếu EWA biến mỗi Gaussian 3D thành một ellipse 2D trên ảnh. Compact box xác định ellipse đó phủ lên những tile nào, rồi trong từng tile các Gaussian được alpha-blend để ra màu pixel C(x). Ta so ảnh render với ảnh thật qua hàm loss, lan truyền ngược để lấy gradient theo mọi tham số θ, và Adam cập nhật θ.

Vòng lặp này chạy hàng chục nghìn lần. Chen vào giữa có hai nhịp phụ: cứ mỗi 100 iteration có một bước ADC (clone, split, prune) dựa trên gradient tích luỹ, và mỗi 3000 iteration, cho tới mốc 15.000, có một lần reset opacity.

Điểm mạnh của thiết kế này là nó hoàn toàn tường minh và khả vi từ đầu đến cuối. Không có hộp đen nào: mọi bước đều có thể kiểm tra và gỡ lỗi, và ngay khi train xong ta có thể render thời gian thực. Ảnh bên dưới là một ảnh render minh hoạ để các bạn hình dung chất lượng. Lưu ý đây chỉ là minh hoạ định tính, không kèm số đo benchmark.

Tuy nhiên có một hạn chế, và đây là động lực cho phần tiếp theo của bài. Tín hiệu quyết định densify chỉ dựa vào gradient. Gradient phản ánh trạng thái hội tụ hiện tại của loss, chứ không trực tiếp đo xem ảnh còn thiếu bao nhiêu chi tiết tần số cao và thiếu ở tỉ lệ nào. Hai Gaussian có cùng độ lớn gradient có thể đang thiếu hình học ở hai quy mô rất khác nhau: một cái thiếu chi tiết rất mịn, một cái thiếu cấu trúc thô hơn nhiều. Chỉ nhìn gradient thì không phân biệt được hai trường hợp này. Hãy nhớ thêm điểm ta đã thấy ở slide split: scale bị chia đều cho 1.6 trên cả ba trục.

Đó chính là hai câu hỏi mà SADGS đặt ra: làm sao để việc densify nhận biết được cấu trúc và tỉ lệ của chi tiết còn thiếu, và làm sao để split theo đúng hướng cần tách. Chúng ta sẽ bước sang phần SADGS ngay sau đây.

## Slide 30: SADGS: bổ sung tín hiệu tần số cho Adaptive Density Control
Đến đây chúng ta bước vào phần trọng tâm của bài: SADGS, viết tắt của Structure-Aware Densification. Trước khi xem công thức, tôi nhắc lại hạn chế đã nêu ở cuối phần trước. Cơ chế Adaptive Density Control gốc quyết định clone hay split một Gaussian dựa vào gradient. Gradient cho biết Gaussian đó "đang bị kéo mạnh" trong quá trình tối ưu, nhưng nó không trả lời được một câu hỏi rất tự nhiên: Gaussian này đang to hay nhỏ so với chi tiết ảnh mà nó phải biểu diễn, và chênh lệch theo tỉ lệ nào?

Ý tưởng cốt lõi của SADGS là đo thẳng đại lượng đó. Ta lấy kích thước của Gaussian sau khi chiếu lên màn hình, chia cho kích thước chi tiết ảnh cục bộ tại đúng vị trí ấy. Kích thước chi tiết này có thể hình dung như một "bước sóng" của texture. Tỉ số thu được gọi là eta.

Cách đọc eta khá trực quan. Nếu eta lớn hơn 1, Gaussian to hơn chi tiết. Nó "nuốt chửng" chi tiết, làm mờ vùng đó, nên cần split thành các mảnh nhỏ hơn. Điểm đáng chú ý là SADGS split đúng theo trục bị vi phạm, chứ không chia đều mọi trục như cơ chế gốc. Ngược lại, nếu eta nhỏ hơn hoặc bằng 0.1, Gaussian nhỏ hơn rất nhiều so với chi tiết cục bộ. Nó đang "phục vụ thừa" một vùng gần như không đổi, nơi một Gaussian lớn hơn cũng đủ sức biểu diễn, nên nó là ứng viên để prune.

Tôi muốn nhấn mạnh một điều: SADGS không thay thế khung ADC mà chúng ta vừa học. Nó kế thừa toàn bộ clone, split và prune, rồi chỉ bổ sung thêm một điều kiện kích hoạt dựa trên eta. Điều kiện mới này được kết hợp với điều kiện gradient cũ bằng phép OR. Phần sau sẽ trình bày chi tiết cách kết hợp này.

Vậy eta được tính ra sao? Bốn slide tiếp theo sẽ dựng nó từ con số không theo một chuỗi: ảnh gốc, kim tự tháp đa tỉ lệ, structure tensor, trị riêng, bước sóng cục bộ, rồi chiếu trục Gaussian lên màn hình. Ta bắt đầu với bước thứ nhất: làm sao đo được chi tiết ảnh có kích thước bao nhiêu.

## Slide 31: Bước 1: kim tự tháp đa tỉ lệ (multiscale) trên ảnh GT
Muốn so sánh kích thước Gaussian với kích thước chi tiết ảnh, trước hết ta cần một cách đo xem tại mỗi vị trí, chi tiết ảnh lớn cỡ nào. Trong code, việc này nằm ở hàm get_multiscale_structure_tensor_v1 trong file utils/loss_utils.py, bắt đầu từ dòng 232.

Hàm này làm mờ dần ảnh ground truth qua nhiều tầng, đánh số i từ 0 đến levels trừ 1, mặc định là 3 tầng. Độ mờ tăng theo cấp số nhân: sigma_i bằng sigma_0 nhân với octave_step mũ i, với octave_step bằng 1.5. Như vậy mỗi tầng mờ hơn tầng trước theo cùng một tỉ lệ cố định.

Tại mỗi tầng, ta lấy hiệu giữa hai ảnh mờ liên tiếp rồi tính chuẩn, được b_i. Đây là kỹ thuật Difference-of-Gaussians, hay DoG, một kỹ thuật kinh điển trong thị giác máy tính, chẳng hạn được dùng trong SIFT, và nó xấp xỉ toán tử Laplacian. Trực giác rất đơn giản. Ở vùng phẳng, làm mờ thêm một chút cũng không thay đổi gì, nên hiệu gần bằng 0. Ở vùng có cạnh hoặc chi tiết rõ, làm mờ thêm sẽ thay đổi ảnh đáng kể, nên hiệu lớn. Vì thế b_i đóng vai trò bản đồ năng lượng cạnh tại tầng i.

Mỗi tầng còn được gán một tần số f_i bằng 1 chia octave_step mũ i. Tầng càng sâu thì càng mờ, f_i càng nhỏ, và tầng đó ứng với chi tiết càng thô.

Tại sao phải dùng nhiều tỉ lệ thay vì một? Một hoạ tiết lặp như vân gỗ hay gạch lát có thể hiện rõ ở độ phân giải mịn nhưng gần như biến mất ở độ phân giải thô. Cấu trúc lớn như viền một toà nhà thì ngược lại. Nếu chỉ nhìn ở một tỉ lệ, ta sẽ bỏ sót những chi tiết chỉ tồn tại rõ ràng ở một quy mô khác. Hình trên slide minh hoạ các tầng của kim tự tháp này.

Giờ mỗi tầng đã cho ta một góc nhìn riêng. Câu hỏi tiếp theo là gộp chúng thành một đại lượng duy nhất như thế nào.

## Slide 32: Bước 2: gộp thành một structure tensor duy nhất Ĵ
Tại mỗi tầng i, ta tính một structure tensor J_i. Đây là một ma trận 2x2 quen thuộc trong xử lý ảnh, gồm ba thành phần S_xx, S_xy và S_yy. Nó được dựng từ tích ngoài của vector gradient ảnh với chính nó, sau đó làm mờ nhẹ để ổn định. Ma trận này mô tả cả hướng lẫn cường độ biến thiên cục bộ: ảnh đang thay đổi mạnh theo hướng nào, và mạnh đến đâu.

Việc gộp các tầng nằm ở loss_utils.py, từ dòng 283 đến 300, theo công thức đóng khung trên slide. Ta lấy J_i chia cho vết của nó, nhân với trọng số w_i và với f_i bình phương, cộng qua mọi tầng, rồi chia cho tổng các trọng số. Mỗi thành phần có lý do riêng.

Thứ nhất, chia J_i cho vết là chuẩn hoá theo vết. Sau bước này, mỗi tầng chỉ còn đóng góp thông tin về hướng và hình dạng, tức mức độ dị hướng của biến thiên. Nhờ vậy, một tầng tình cờ có năng lượng tuyệt đối lớn hơn sẽ không lấn át các tầng khác.

Thứ hai, trọng số w_i bằng b_i mũ p, với p mặc định là 3. Tầng nào có cạnh rõ hơn, tức b_i lớn hơn, sẽ được tin cậy nhiều hơn khi gộp. Luỹ thừa bậc 3 khuếch đại khá mạnh sự chênh lệch này, để các tầng ít cấu trúc, chủ yếu là nhiễu, không làm loãng thông tin từ tầng có cấu trúc rõ.

Thứ ba, hệ số f_i bình phương là hệ số quy đổi. Các tầng bị làm mờ ở những sigma khác nhau, nên cần được đưa về cùng một thang tần số vật lý trước khi cộng lại với nhau.

Kết quả là ma trận Ĵ với ba kênh S_xx, S_xy và S_yy, như trong hình trên slide. Đây là nguyên liệu dùng xuyên suốt các bước tính eta phía sau. Tuy vậy, một ma trận vẫn chưa phải là một con số để so sánh. Slide tiếp theo sẽ rút gọn nó thành một bước sóng.

## Slide 33: Bước 3: trị riêng của Ĵ → bước sóng cục bộ w_min
Từ ma trận Ĵ, ta chỉ cần trích ra một đại lượng vô hướng duy nhất là trị riêng lớn nhất lambda_1. Phần code này nằm ở utils/freq_utils.py, dòng 313 đến 334. Vì Ĵ là ma trận 2x2 nên trị riêng có công thức đóng: lambda_1 bằng một nửa vết, cộng căn bậc hai của bình phương một nửa vết trừ định thức. Đó chính là nghiệm lớn của phương trình đặc trưng bậc hai, nên không cần gọi một bộ giải trị riêng tổng quát nào.

Ý nghĩa của lambda_1 là gì? Nó tỉ lệ với năng lượng gradient bình phương theo hướng biến thiên mạnh nhất tại điểm đó. Vùng có cạnh sắc, chi tiết dày đặc cho lambda_1 lớn. Vùng phẳng cho lambda_1 xấp xỉ 0.

Tiếp theo, ta chuyển từ năng lượng sang bước sóng bằng công thức đóng khung: w_min bằng 1 chia cho căn lambda_1 cộng 10 mũ trừ 5. Trực giác như sau. Năng lượng gradient cao nghĩa là ảnh đổi nhanh theo từng pixel, tức tần số không gian cao. Tần số cao lại đồng nghĩa với bước sóng ngắn. Vì vậy lambda_1 lớn cho w_min nhỏ. Ngược lại, ở vùng phẳng, lambda_1 tiến về 0 nên w_min tiến tới khoảng 10 mũ 5, một giá trị rất lớn. Có thể hiểu đó là cách công thức nói rằng ở đây không có chi tiết nào cần bám theo cả. Hằng số 10 mũ trừ 5 ở mẫu có tác dụng tránh phép chia cho 0. Hình trên slide minh hoạ quan hệ giữa trị riêng và bước sóng này.

Như vậy w_min là "cây thước" mô tả chi tiết ảnh tại mỗi vị trí dài bao nhiêu pixel. Trong công thức eta, nó sẽ nằm ở mẫu số. Phần còn thiếu là tử số: kích thước Gaussian, đo bằng cùng đơn vị pixel. Đó là nội dung của slide tiếp theo.

## Slide 34: Bước 4: chiếu 3 trục chính của Gaussian lên màn hình
Đã có cây thước w_min, giờ ta cần đo kích thước Gaussian bằng một đơn vị tương thích, tức pixel trên màn hình. Việc này do hàm compute_projected_axes_subset đảm nhận, trong utils/freq_utils.py, dòng 116 đến 178. Điểm hay là hàm này không dùng công cụ gì mới. Nó dùng lại đúng Jacobian J của phép chiếu pinhole mà chúng ta đã học ở phần đầu bài giảng.

Đọc công thức đóng khung từ phải sang trái. Đầu tiên, R_i nhân diag(s) nhân e_j cho ra véc-tơ trục chính thứ j của ellipsoid trong không gian world, với j bằng 1, 2, 3. Đây là ba trục vuông góc của ellipsoid, và độ dài của trục thứ j đúng bằng s_j. Sau đó, R_view xoay véc-tơ này sang hệ toạ độ camera. Cuối cùng, J tuyến tính hoá phép chiếu phối cảnh tại đúng vị trí mu của Gaussian, giống hệt cách J được dùng trong EWA splatting ở phần render. Kết quả là véc-tơ a_j trên mặt phẳng ảnh, với hai thành phần u_j và v_j.

Độ dài ell_j bằng căn của u_j bình phương cộng v_j bình phương cộng 10 mũ trừ 8. Đây là độ dài tính bằng pixel của trục thứ j sau khi chiếu, và nó chính là tử số của eta. Hằng số nhỏ cộng thêm giúp phép lấy căn ổn định về số học.

Cần lưu ý một điểm dễ nhầm: mỗi Gaussian có ba giá trị ell_1, ell_2, ell_3, ứng với ba trục chính hình học. Chúng không phải là ba kênh màu R, G, B. Chính vì giữ riêng từng trục, SADGS biết được Gaussian đang vi phạm theo hướng cụ thể nào. Nhờ đó, về sau nó có thể split đúng trục bị vi phạm thay vì chia đều mọi hướng, đúng như lời hứa ở slide mở đầu phần này. Hình trên slide minh hoạ ba trục của ellipsoid sau khi được chiếu qua Jacobian.

Bây giờ ta đã có đủ hai vế: tử số ell_j và mẫu số w_min. Bước tiếp theo là ghép chúng lại thành tỉ số eta.

## Slide 35: Bước 5: η=ℓ/w_min — trực giác Nyquist
Thưa các bạn, đây là slide quan trọng nhất của cả bài, vì mọi quyết định của SADGS về sau đều xoay quanh đại lượng η trên slide này.

Ở hai bước trước, ta đã chuẩn bị đủ hai thành phần. Tử số ℓ là kích thước của Gaussian sau khi chiếu, tức là "cái chổi" ta dùng để vẽ ảnh to đến mức nào. Mẫu số w_min là bước sóng của chi tiết cục bộ, tức là "nét vẽ" nhỏ nhất ở vùng đó dài bao nhiêu. Chia hai con số này cho nhau, ta được η_j = ℓ_j / w_min, tính riêng cho từng trục j. Trong code, đây là chế độ mặc định eta_compute_mode = "wavelength", ở dòng 348 của freq_utils.py.

Tôi xin nhấn mạnh một điều: η là một ngưỡng hình học thuần tuý. Nó chỉ so sánh hai độ dài với nhau, và không phải là một mô hình vật lý mô tả việc biên độ tín hiệu bị suy hao ra sao. Chữ "Nyquist" trong tên gọi chỉ mượn ý tưởng chung của lý thuyết lấy mẫu: muốn biểu diễn đúng một dao động thì đơn vị lấy mẫu phải đủ nhỏ so với chu kỳ của dao động đó. Ở đây, đơn vị lấy mẫu chính là Gaussian.

Từ đó có ba vùng. Khi η lớn hơn 1, Gaussian rộng hơn cả một chu kỳ chi tiết, nên một mình nó không thể bám theo dao động, và nó trở thành ứng viên SPLIT. Khi η nằm trong khoảng lớn hơn 0.1 và nhỏ hơn hoặc bằng 1, kích thước tương xứng với chi tiết, nên ta GIỮ NGUYÊN. Khi η nhỏ hơn hoặc bằng 0.1, Gaussian nhỏ hơn w_min tới hơn mười lần, tức là nó đang phủ thừa một vùng gần như không đổi, và nó trở thành ứng viên PRUNE. Hình minh hoạ bên dưới thể hiện chính trực giác này: so kích thước Gaussian với chu kỳ của chi tiết.

Có một lưu ý quan trọng cho ai đọc code. Docstring ở gaussian_model.py, dòng 837 đến 840, mô tả η tỉ lệ với (σω) bình phương, tức là quan hệ bậc hai. Tuy nhiên, công thức thực sự được tính và được dùng là quan hệ tuyến tính theo kích thước, như trên slide. Toàn bộ bài giảng bám theo công thức thực chạy, vì docstring có thể là tài liệu chưa được cập nhật theo code.

Có η cho từng lần render rồi, câu hỏi tiếp theo là: ta dùng các ngưỡng này thế nào khi có rất nhiều góc nhìn khác nhau?

## Slide 36: Bước 6: ngưỡng phân loại và tích luỹ NHẤT QUÁN ĐA VIEW
Hai ngưỡng mà ta vừa nói được cố định ngay trong code: TAU_HIGH bằng 1.0 và TAU_LOW bằng 0.1, ở dòng 395–396 của freq_utils.py.

Cách dùng như sau. Mỗi lần render một view, với mỗi Gaussian ta lấy η_max, là giá trị lớn nhất trong ba trục. Lý do là chỉ cần một trục quá to so với chi tiết thì Gaussian đó đã có vấn đề. Sau đó ta cộng 1 vào đúng một trong ba bộ đếm. Nếu η_max lớn hơn 1 thì cộng vào eta_high_count. Nếu η_max nhỏ hơn hoặc bằng 0.1 thì cộng vào eta_low_count. Trường hợp còn lại thì cộng vào eta_mid_count. Như vậy, mỗi view giống như một lá phiếu bầu cho một trong ba trạng thái.

Tới kỳ densify, tức là cứ mỗi 100 iteration, các bộ đếm được quy ra tỉ lệ, như ở dòng 341–353 của train.py. high_ratio bằng eta_high_count chia cho accum_view_count, còn low_ratio bằng eta_low_count chia cho accum_view_count. Chỉ khi tỉ lệ vượt 0.8, tức là hơn 80% số góc nhìn đồng thuận, quyết định split hoặc prune mới được kích hoạt.

Đây là khác biệt lớn nhất về triết lý so với 3DGS gốc. 3DGS gốc chỉ nhìn vào gradient tích luỹ tại một thời điểm cuối cùng để quyết định, và không phân biệt rõ có bao nhiêu góc nhìn đồng ý với nhau. SADGS thì yêu cầu đa số áp đảo. Tại sao phải thận trọng như vậy? Vì một góc chụp đơn lẻ có thể cho η rất bất thường: vật bị che khuất, ánh sáng bị loá, hoặc ta nhìn gần như ngay cạnh của ellipsoid khiến kích thước chiếu méo đi. Ngưỡng 80% giúp hệ thống không phản ứng thái quá theo nhiễu của một view như thế. Hình dòng thời gian bên dưới minh hoạ quá trình tích luỹ qua nhiều view trước mỗi lần ra quyết định.

Vậy khi một Gaussian đã bị "kết án" là quá to, ta chia nó ra sao? Đó là nội dung slide tiếp theo.

## Slide 37: Split dị hướng (1/2): chia đúng trục, đặt con trên lưới tất định
Hàm thực hiện việc này là densify_and_split_structgs, bắt đầu từ dòng 638 của gaussian_model.py. Đây là phiên bản split mới của SADGS, được dùng khi đã có sẵn max_eta_3ch, tức là η của từng trục.

Công thức cốt lõi được đóng khung trên slide: k_axis bằng trần của căn bậc hai của max(η_axis, 1). Số con sinh ra là N_con = k_x · k_y · k_z. Kích thước mỗi con bằng kích thước cha chia cho k_axis mũ p, với p = ks_scale_power = 1.0. Nghĩa là trục nào bị chia thành k phần thì kích thước con theo trục đó nhỏ đi k lần.

Điểm khác biệt lớn đầu tiên so với split nguyên bản ở phần 1 là k được tính riêng cho từng trục. Nhờ phép max với 1, trục nào "khoẻ mạnh", tức η nhỏ hơn hoặc bằng 1, sẽ có k bằng 1 và được giữ nguyên. Chỉ trục thực sự vi phạm mới bị chia. Chẳng hạn, một Gaussian dẹt chỉ quá dài theo một hướng sẽ được cắt đúng theo hướng đó, thay vì bị chia đều mọi phía.

Điểm khác biệt thứ hai là vị trí của các con. Cơ chế gốc lấy mẫu ngẫu nhiên, còn SADGS đặt các con trên một lưới tất định. Khoảng cách tâm-tâm giữa hai con liền kề bằng σ_con nhân căn 12, và toàn bộ lưới được xoay theo đúng hướng R của Gaussian cha, như ở dòng 766–780. Hình bên dưới minh hoạ lưới này trong 2D.

Con số căn 12 không phải được chọn tuỳ ý. Ta có thể kiểm chứng rằng lưới này bảo toàn phương sai: tổng phương sai của các con đặt cách đều theo công thức trên đúng bằng phương sai ban đầu của cha. Nói cách khác, sau khi split, "độ trải" tổng thể của vùng mà cha từng phủ vẫn được giữ, nên việc chia có cơ sở thống kê rõ ràng.

Nhưng tại sao lại lấy căn bậc hai của η và làm tròn lên? Ta sẽ xem hệ quả của lựa chọn này ở slide sau.

## Slide 38: Split dị hướng (2/2): k tăng theo bậc thang khi η tăng
Vì k_axis được tính bằng hàm trần, tức là làm tròn lên, quan hệ giữa k và η không phải một đường cong mượt mà là một bậc thang. k chỉ nhảy lên mức mới khi η vượt qua các ngưỡng 1, 4, 9, 16, và cứ thế tiếp tục, chính là các số k bình phương. Hình bên dưới vẽ đúng dạng bậc thang này: η chạy trên trục ngang, k chạy trên trục đứng, và mỗi bậc kéo dài cho tới số chính phương kế tiếp.

Hãy xem hai ví dụ cụ thể. Nếu η vừa mới vượt 1, ví dụ η = 1.2, thì căn của nó xấp xỉ 1.1, làm tròn lên thành k = 2, tức là sinh 2 con trên trục đó. Đây là mức tăng thận trọng, tương xứng với mức vi phạm còn nhẹ. Còn nếu η lớn hơn nhiều, ví dụ η = 20, thì k bằng trần của căn 20, tức là 5, và sinh hẳn 5 con trên trục đó. Đây là phản ứng mạnh tay hơn, tương xứng với mức vi phạm nghiêm trọng hơn.

So sánh với split nguyên bản của 3DGS: split gốc luôn cố định N = 2, bất kể Gaussian quá to một chút hay quá to gấp nhiều lần. Như vậy sẽ có hai rủi ro. Nếu Gaussian quá to nhiều, chia đôi là không đủ, các con vẫn còn η lớn hơn 1 và phải chờ thêm nhiều vòng densify nữa. Ngược lại, nếu tăng số con một cách cố định cho mọi trường hợp, số lượng Gaussian sẽ bùng nổ không cần thiết. Cách làm bậc thang cho phép hệ thống tự điều chỉnh cường độ phản ứng theo đúng quy mô của vấn đề, tránh cả hai rủi ro này.

Đó là phía "quá to". Còn phía ngược lại, những Gaussian quá nhỏ và thừa thãi, được xử lý bằng prune, ở slide tiếp theo.

## Slide 39: Prune theo cấu trúc: OR với tiêu chí gốc, không thay thế
Việc prune nằm trong hàm densify_and_prune_structgs, ở dòng 1009–1019 của gaussian_model.py. Điều tôi muốn các bạn nắm là: tiêu chí cấu trúc không thay thế tiêu chí gốc mà được cộng thêm vào bằng phép OR.

Mặt nạ prune gồm bốn điều kiện, và một Gaussian bị xoá nếu bất kỳ điều kiện nào đúng. Thứ nhất, opacity thấp, α nhỏ hơn 0.1. Thứ hai, quá to trên màn hình, tức bán kính 2D lớn hơn max_screen_size. Thứ ba, quá to trong không gian, tức scale lớn nhất vượt 0.1 lần extent của cảnh. Ba điều kiện này là những gì ta đã học ở phần ADC gốc. Điều kiện thứ tư là điều kiện mới: low_ratio lớn hơn 0.8, tức hơn 80% số view đồng thuận rằng Gaussian này có η nhỏ hơn hoặc bằng 0.1, đang phủ thừa một vùng phẳng. Hình minh hoạ bên dưới thể hiện cách các mặt nạ này được gộp lại bằng OR.

Có hai chi tiết cần lưu ý. Thứ nhất, trong nhánh này min_opacity bằng 0.1, cao hơn nhiều so với 0.005 ở nhánh 3DGS gốc. Nghĩa là SADGS prune mạnh tay hơn không chỉ nhờ thêm điều kiện cấu trúc, mà ngay cả tiêu chí opacity đơn thuần cũng đã khắt khe hơn. Khi so sánh số lượng Gaussian giữa hai phương pháp, cần nhớ yếu tố này.

Thứ hai, cả clone lẫn split mà ta vừa bàn còn bị chặn thêm bởi metric_mask, tức là điều kiện accum_view_count lớn hơn 0.5. Hiểu đơn giản, Gaussian phải được quan sát ít nhất một lần trong đợt tích luỹ hiện tại. Một Gaussian vừa mới sinh ra, chưa có lá phiếu nào, sẽ không bị động tới ở đợt densify đó. Cũng giống tinh thần đồng thuận đa view ở slide trước, SADGS tránh ra quyết định vội vàng khi dữ liệu còn quá ít.

Đến đây ta đã đi hết vòng quyết định của SADGS: tính η, bỏ phiếu qua nhiều view, split dị hướng theo bậc thang, và prune kết hợp OR với tiêu chí gốc.

## Slide 40: Luồng quyết định đầy đủ: densify_and_prune_structgs
Đến đây ta đã có đủ các mảnh ghép riêng lẻ. Slide này ghép chúng lại thành một quy trình duy nhất. Hàm đang chạy mặc định là densify_and_prune_structgs, nằm ở gaussian_model.py từ dòng 957 đến 1050, và được gọi cứ mỗi 100 iteration.

Bước đầu tiên là phân loại Gaussian theo kích thước, trước khi xét có cần làm dày hay không. Ta lấy trục dài nhất của Gaussian, tức max(s), rồi so với dense nhân extent, với dense bằng 0.001 và extent là kích thước đặc trưng của cảnh. Gaussian lớn hơn ngưỡng này thuộc diện split_qual: nó đã đủ to, nên muốn thêm chi tiết thì phải chẻ nó thành các Gaussian nhỏ hơn. Gaussian nhỏ hơn hoặc bằng ngưỡng thuộc diện clone_qual: nó đã nhỏ, chẻ ra không có ý nghĩa, nên ta nhân bản nó để phủ thêm vùng còn thiếu. Hai điều kiện này loại trừ nhau, nên mỗi Gaussian chỉ có thể rơi vào một trong hai nhánh.

Tiếp theo là hai công thức đóng khung. Mỗi công thức có hai tầng. Tầng thứ nhất hỏi Gaussian này có cần làm dày không, và trả lời bằng phép OR giữa hai nguồn tín hiệu. Nguồn thứ nhất là split_mask theo eta, tức tín hiệu cấu trúc tần số mà ta đã xây trong năm slide trước. Nguồn thứ hai là tín hiệu gradient quen thuộc từ ADC gốc. Chỉ cần một trong hai nguồn báo cần làm dày là đủ. Nhờ phép OR, SADGS bổ sung cho cơ chế cũ chứ không thay thế nó: những gì 3DGS gốc sẽ làm dày thì SADGS vẫn làm dày, và SADGS còn bắt thêm những vùng giàu cấu trúc mà gradient có thể bỏ sót. Tầng thứ hai là phép AND với split_qual hoặc clone_qual, dùng để quyết định cách làm dày: chẻ hay nhân bản.

Có một chi tiết tinh tế: nhánh split dùng chuẩn của gradient tuyệt đối, còn nhánh clone dùng chuẩn của gradient thường. Trong nhánh này, cả hai ngưỡng đều bằng 0.0002. Cặp giá trị 0.0002/0.0004 mà ta từng thấy chỉ dùng ở nhánh 3DGS gốc.

Sau hai công thức trên, cả hai luồng còn phải AND thêm với metric_mask ở slide trước rồi mới được thực thi. Sau khi clone và split xong, hàm chạy bước prune. Cuối cùng, hàm áp trần opacity, alpha bằng min(alpha, 0.8), lên toàn bộ Gaussian còn lại. Trực giác của bước này như sau: một Gaussian gần như đặc hoàn toàn sẽ che khuất mọi thứ phía sau. Khi đó các Gaussian phía sau gần như không nhận được gradient và không học được gì. Giữ opacity dưới 0.8 để lại một chút "khe hở" cho gradient đi qua.

Vậy trong toàn bộ luồng này, phần nào thật sự là của SADGS? Slide tiếp theo sẽ phân định rõ.

## Slide 41: Đâu là điểm MỚI của SADGS, đâu là KẾ THỪA — trong số những gì ĐANG CHẠY
Khi đọc một codebase thực tế, một câu hỏi rất dễ bị bỏ qua là: trong những gì đang chạy, cái nào là đóng góp mới, còn cái nào đã có sẵn từ trước? Bảng này chia các cơ chế chạy mặc định thành hai nhóm, ngăn cách bằng đường kẻ ở giữa.

Nửa trên là ba cơ chế kế thừa. Thứ nhất là compact box, tức SnugBox với mult bằng 0.7, lấy từ Taming và Speedy-Splat. Nó giúp khoanh vùng ảnh hưởng của mỗi Gaussian trên màn hình chặt hơn, nhờ đó rasterize nhanh hơn. Thứ hai là bộ lọc low-pass 2D, cộng thêm kappa nhân I với kappa bằng 0.1 vào covariance 2D. Ý tưởng này có từ EWA splatting của Zwicker và cộng sự, và giúp tránh việc một Gaussian chiếu xuống nhỏ hơn cả một pixel. Thứ ba là optimizer hybrid, kế thừa từ Taming. Nó là cấu hình mặc định và không bị tắt.

Nửa dưới là bốn cơ chế thật sự mới của SADGS. Một là structure tensor đa tỉ lệ, dùng để suy ra bước sóng w_min rồi suy ra eta. Hai là tích luỹ nhất quán qua nhiều view bằng high_ratio và low_ratio. Ba là split dị hướng, với hệ số k_x, k_y, k_z riêng cho từng trục. Bốn là prune theo cấu trúc, kết hợp OR với tiêu chí prune gốc.

Thông điệp chính cần nhớ: cái mới của SADGS là toàn bộ chuỗi xây dựng và sử dụng eta, từ phân tích ảnh cho đến quyết định densify và prune. Cái mới không nằm ở compact box hay optimizer. Đó là các kỹ thuật tăng tốc đã có sẵn trước khi SADGS được thêm vào. Nếu nhầm điểm này, ta sẽ dễ gán sai công lao, ví dụ cho rằng SADGS nhanh hơn là nhờ phát minh ra optimizer mới.

Ba slide tiếp theo đi vào phần tối ưu hoá, gồm optimizer, learning rate và lịch reset. Cả ba cơ chế này áp dụng giống hệt nhau cho nhánh 3DGS gốc lẫn nhánh SADGS. Ta học chúng để hiểu toàn bộ hệ thống, không phải vì chúng tạo ra khác biệt giữa hai nhánh. Bắt đầu với optimizer.

## Slide 42: Optimizer: hybrid là MẶC ĐỊNH — chạy song song hai bộ tối ưu
Nhìn vào arguments/__init__.py dòng 128, ta thấy optimizer_type được đặt là "hybrid". Chữ "hybrid" có nghĩa là trong mỗi iteration, ở train.py từ dòng 429 đến 434, có hai lệnh step cùng chạy chứ không phải chọn một trong hai.

Lệnh thứ nhất là gaussians.optimizer.step(), một Adam thường có bật amsgrad=True. Nó cập nhật các nhóm tham số chính: vị trí xyz, opacity, scaling, rotation và f_dc, tức màu trung bình.

Lệnh thứ hai là gaussians.shoptimizer.step(visible, radii.shape[0]), một Sparse hay Fused Gaussian Adam riêng. Điểm khác biệt là nó chỉ cập nhật tham số của những Gaussian đang nhìn thấy trong view hiện tại, xác định bằng visible = radii > 0, tức Gaussian có bán kính chiếu lớn hơn 0 trên màn hình. Trực giác đằng sau là thế này: trong một cảnh lớn, tại mỗi thời điểm, một camera đơn lẻ chỉ nhìn thấy một phần nhỏ của cảnh. Phần lớn Gaussian nằm ngoài khung hình và không nhận gradient nào từ view đó. Cập nhật cả những Gaussian này là tốn phép tính vô ích, nên bản sparse bỏ qua chúng.

Code còn hai chế độ khác nhưng không phải mặc định. Chế độ "default" chỉ chạy optimizer_step, với lịch cập nhật thưa dần từ 1, lên 32, rồi 64 iteration một lần. Chế độ "sparse_adam" chỉ chạy phần sparse và bỏ hẳn Adam thường.

Hình minh hoạ bên dưới cho thấy một chi tiết kỹ thuật đi kèm optimizer. Khi densify thêm M Gaussian mới, hàm cat_tensors_to_optimizer phải ghép M dòng mới vào cả tham số lẫn trạng thái của Adam. Tham số của Gaussian mới được sao chép. Hai moment exp_avg và exp_avg_sq của các dòng mới được khởi tạo bằng 0, còn N dòng cũ giữ nguyên. Nói cách khác, Gaussian mới bắt đầu với "trí nhớ" optimizer trống.

Kết luận thực dụng: cả hai nhánh dùng chung cấu hình hybrid này, nên optimizer không phải chỗ để so sánh 3DGS gốc với SADGS. Tiếp theo, ta xem mỗi nhóm tham số học với tốc độ bao nhiêu.

## Slide 43: Lịch learning rate: mỗi nhóm tham số một tốc độ riêng
Mỗi nhóm tham số có một learning rate riêng, và việc chọn các con số này có lý do hẳn hoi.

Trước hết là vị trí mu. Vị trí là nhóm duy nhất dùng learning rate thay đổi theo thời gian: suy giảm mũ từ position_lr_init bằng 1.6 nhân 10 mũ trừ 4 xuống position_lr_final bằng 1.6 nhân 10 mũ trừ 6, trải qua 30.000 bước. Đó là mức giảm 100 lần. Trên hình, trục tung là thang log nên lịch mũ hiện ra thành một đường thẳng dốc xuống, và công thức ở tiêu đề hình là nội suy tuyến tính trong không gian log giữa lr_init và lr_final. Hình còn có một đường đứt đoạn màu đỏ, minh hoạ trường hợp giả định nếu đặt lr_delay_steps bằng 3000: khi đó learning rate bắt đầu rất nhỏ rồi mới tăng dần lên. Theo chú thích trên hình, SADGS thực tế đặt lr_delay_steps bằng 0, tức đường màu xanh, nên learning rate bắt đầu ngay ở mức lớn nhất.

Vì sao vị trí lại được học theo cách này? Ở giai đoạn đầu, hình học của cảnh còn rất thô, và các Gaussian cần di chuyển nhiều để "định hình" cảnh, nên learning rate lớn. Khi gần hội tụ, chỉ còn cần tinh chỉnh nhẹ. Một learning rate lớn lúc này sẽ làm Gaussian dao động quanh vị trí đúng, nên ta giảm mạnh.

Các nhóm còn lại dùng learning rate cố định: opacity 0.05, scaling 0.01, rotation 0.002, f_dc 0.0025 và f_rest 0.00025. Hãy để ý cặp f_dc và f_rest. f_dc là màu trung bình, thành phần nền tảng. f_rest là các hệ số spherical harmonics bậc cao, mô tả màu thay đổi theo góc nhìn. f_rest học chậm hơn f_dc đúng 10 lần. Nguyên tắc chung là học nhanh các tham số "thô", mang tính nền tảng, còn các tham số "tinh chỉnh chi tiết" thì học chậm và thận trọng. Nếu để SH bậc cao học nhanh ngay từ đầu, chúng dễ hấp thụ nhiễu và che đi lỗi hình học thay vì biểu diễn hiệu ứng góc nhìn thật.

Sau optimizer và learning rate, mảnh cuối của phần tối ưu hoá là lịch reset opacity.

## Slide 44: Xác nhận lại: lịch reset opacity đúng 4 lần, không hơn
Slide này nhằm chốt lại một con số cụ thể, và qua đó minh hoạ một nguyên tắc quan trọng.

Hình là dòng thời gian thực tế của quá trình huấn luyện, lấy từ train.py dòng 202 đến 440. Hãy nhìn thanh reset_opacity(0.1). Điều kiện gọi hàm là iteration chia hết cho 3000. Tuy nhiên lệnh gọi còn nằm trong điều kiện iteration nhỏ hơn densify_until_iter, bằng 15.000. Ghép hai điều kiện lại, reset chỉ xảy ra tại đúng bốn mốc: 3000, 6000, 9000 và 12000. Mốc 15.000 không được tính vì điều kiện là "nhỏ hơn" chứ không phải "nhỏ hơn hoặc bằng". Sau 15.000 cũng không còn lần reset nào nữa. Trên hình, bạn thấy cả cụm thanh densify, size_threshold và reset opacity đều kết thúc tại vạch 15.000. Từ đó đến 30.000 là giai đoạn tinh chỉnh, chỉ tối ưu tham số chứ không densify.

Vì sao phải nhấn mạnh con số này? Một vài tài liệu tổng hợp trước đây ghi rằng có "10 chu kỳ trải đều tới 30.000". Con số đó không khớp với điều kiện iteration nhỏ hơn 15.000 nằm ngay trong đoạn code gọi reset_opacity. Đây là ví dụ cụ thể cho nguyên tắc xuyên suốt bài giảng: luôn đối chiếu với code thật trước khi phát biểu một con số, thay vì suy luận hay chép lại từ tài liệu khác. Một con số sai nghe có vẻ hợp lý rất dễ được lan truyền.

Cuối cùng, cũng như optimizer và learning rate, lịch reset này được cả nhánh 3DGS gốc lẫn nhánh SADGS dùng chung. Nó không phải điểm khác biệt giữa hai cơ chế densify. Như vậy ta đã khép lại phần tối ưu hoá, và có thể chuyển sang phần tiếp theo của bài.

## Slide 45: Toàn cảnh lịch trình huấn luyện (30.000 iteration)
Trước khi so sánh hai phương pháp, chúng ta ghép tất cả các cơ chế đã học vào một trục thời gian duy nhất, từ iteration đầu tiên đến iteration 30.000. Bảng trên slide đọc từ trên xuống, theo thứ tự thời gian.

Mốc đầu tiên là việc tăng bậc hàm cầu điều hoà (SH). Bậc SH được nâng dần tới bậc tối đa D_max bằng 3. Nhờ vậy, mô hình học màu cơ bản trước rồi mới học đến hiệu ứng phụ thuộc hướng nhìn. Tại iteration 500, ứng với tham số densify_from_iter, quá trình densify chính thức bắt đầu. Từ iteration 600, rồi 700, và cứ mỗi 100 iteration một lần, hệ thống thực hiện clone, split và prune. Ở đây cần nhớ lại điểm cốt lõi của SADGS: một Gaussian được chọn để densify khi gradient vượt ngưỡng HOẶC tín hiệu tần số η báo vi phạm. Đó là phép OR, không phải thay thế.

Song song với densify, opacity được reset bằng cách nhân với 0,1 tại các iteration 3.000, 6.000, 9.000 và 12.000, đúng 4 lần và chỉ khi iteration còn dưới 15.000. Ngoài ra, tại iteration 4.000 và 8.000 (tham số prune_iterations) có thêm một đợt prune cứng, loại mọi Gaussian có α nhỏ hơn 0,1.

Mốc quan trọng nhất là iteration 15.000, tức densify_until_iter. Tại đây densify, prune và reset opacity đều dừng hẳn. Từ 15.000 đến 30.000 chỉ còn Adam tối ưu giá trị tham số, còn số Gaussian N được giữ cố định.

Điều này dẫn đến một hệ quả khi so sánh. Sau iteration 15.000 không còn cơ chế nào chủ động thay đổi N. Vì vậy, mọi khác biệt giữa 3DGS gốc và SADGS về chất lượng hay tốc độ hội tụ chủ yếu hình thành trong 15.000 iteration đầu. Không nên hiểu rằng hai phương pháp ra số Gaussian cuối cùng khác nhau vì một bên tiếp tục tỉa bớt về sau.

Đã có lịch trình, câu hỏi tự nhiên tiếp theo là chi phí tính toán. Slide sau sẽ trả lời câu hỏi này một cách thận trọng.

## Slide 46: Chi phí tính toán: mô hình định tính, KHÔNG có số đo thật
Slide này nói về chi phí tính toán, và điều đầu tiên tôi muốn nói là một lời thừa nhận: repo này không chứa script benchmark thời gian thực nào. Không có phép đo wall-clock nào so sánh SADGS với 3DGS gốc trên cùng phần cứng, cùng scene và với số lần đo lặp lại. Vì vậy, những gì tôi trình bày ở đây là một mô hình định tính, dùng để lập luận, không phải kết quả thực nghiệm.

Mô hình ước lượng chi phí mỗi iteration như sau: T_iter xấp xỉ bằng a nhân N, cộng b nhân N nhân K, cộng c nhân N khi có bước Adam, cộng F. Trong đó N là số Gaussian. K là số tile trung bình mà mỗi Gaussian phủ lên, và K phụ thuộc vào hệ số mult của compact box đã học ở phần rasterizer. F là chi phí cố định mỗi khung hình, gồm forward pass của loss, sắp xếp và các bước tương tự.

Từ mô hình này ta xét hai chiều ảnh hưởng. Chiều có thể làm tăng chi phí: cứ mỗi 10 iteration, SADGS cập nhật η, nghĩa là phải tính thêm một lượt structure tensor và chiếu lên các trục. Thêm vào đó, ở chế độ hybrid có hai optimizer chạy song song. Chiều có thể làm giảm chi phí: mult bằng 0,7, nhỏ hơn giá trị 1 của SnugBox nguyên bản, nên hộp bao của mỗi Gaussian nhỏ hơn và K giảm, tức mỗi Gaussian chạm ít tile hơn.

Hai chiều này bù trừ nhau đến mức nào thì chúng ta chưa biết, vì chưa có số đo. Do đó bài giảng này giữ một nguyên tắc: con số kiểu "nhanh hơn X lần" hay "bao nhiêu giây mỗi iteration" nếu không kèm phương pháp đo rõ ràng (phần cứng nào, scene nào, đo lặp lại bao nhiêu lần) thì sẽ không được trích dẫn như một kết quả benchmark đã kiểm chứng. Nói "chưa đo" một cách trung thực có giá trị hơn một con số đẹp không có cơ sở.

Tiếp theo, chúng ta tóm tắt lại những gì thực sự đang chạy mặc định trong repo.

## Slide 47: Tóm tắt: những gì ĐANG CHẠY mặc định trong repo này
Slide này gom lại mọi cơ chế đã trình bày trong bài, mỗi cơ chế kèm vị trí cụ thể trong code để các bạn tự đối chiếu.

Dòng thứ nhất là trục chính của SADGS. Structure tensor đa tỉ lệ được dùng để tính η, rồi η được dùng để quyết định split, clone hoặc prune. Phần tính toán nằm trong utils/freq_utils.py, còn chỗ gọi nằm ở train.py, dòng 276 và các dòng 341 đến 372. Dòng thứ hai là split dị hướng. Mỗi trục có hệ số chia riêng k_x, k_y, k_z, được cài đặt tại scene/gaussian_model.py dòng 638. Dòng thứ ba là tích luỹ nhất quán đa view qua hai tỉ lệ high và low ratio, nằm ở train.py dòng 341 đến 353. Cơ chế này giúp quyết định dựa trên nhiều góc nhìn chứ không chỉ một.

Ba dòng tiếp theo là các cơ chế điều tiết. Reset opacity nhân 0,1 đúng 4 lần, như đã thấy ở slide lịch trình, nằm tại gaussian_model.py dòng 415. Trần opacity lấy min của α và 0,8, tại dòng 1043. Optimizer hybrid, tức Adam thường chạy cùng Sparse hoặc Fused Adam, nằm ở train.py dòng 429 đến 434.

Hai dòng cuối thuộc về CUDA rasterizer. Compact box SnugBox với mult bằng 0,7 nằm trong auxiliary.h dòng 313. Bộ lọc low-pass 2D với κ bằng 0,1 nằm trong forward.cu dòng 113.

Chú thích dưới bảng cũng quan trọng. Đây là danh sách đầy đủ những gì bài giảng đã đi sâu. Trong code còn các nhánh tuỳ chọn khác, như nhánh 3DGS gốc dùng làm baseline ở phần đầu, và một vài cơ chế đã được thiết kế nhưng đang tắt theo mặc định. Chúng tôi không đào sâu các nhánh đó vì chúng không ảnh hưởng đến hành vi mặc định của pipeline.

Lưu ý rằng không phải mọi dòng trong bảng này đều là điểm khác biệt so với 3DGS gốc. Slide tiếp theo sẽ tách riêng những khác biệt thực sự.

## Slide 48: Bảng so sánh 3DGS gốc vs. SADGS (chỉ những khác biệt ĐANG CHẠY)
Đây là một trong những slide quan trọng nhất của bài. Bảng chỉ giữ lại những khác biệt đang thực sự chạy giữa 3DGS gốc và SADGS ở cấu hình mặc định. Chúng ta đi qua từng dòng.

Dòng thứ nhất là tín hiệu densify chính. 3DGS gốc chỉ dựa vào gradient vị trí: gradient lớn nghĩa là vùng đó chưa hội tụ và cần thêm Gaussian. SADGS vẫn giữ gradient nhưng thêm tín hiệu η, đo theo tần số đa tỉ lệ, và kết hợp hai tín hiệu bằng phép OR. Như đã học, gradient chỉ phản ánh trạng thái hội tụ hiện tại. Còn η đo xem kích thước Gaussian có quá lớn so với chi tiết ảnh hay không. Hai tín hiệu này bổ sung cho nhau, không cái nào thay thế cái nào.

Dòng thứ hai là split dị hướng theo trục. Ở 3DGS gốc, mỗi lần split luôn tạo N bằng 2 Gaussian con, và mọi trục đều bị chia scale cho 1,6, bất kể chi tiết nằm theo hướng nào. SADGS tính riêng cho từng trục k_axis bằng trần của căn bậc hai của max(η, 1). Trục nào vi phạm nặng thì bị chia nhiều hơn. Trục nào không vi phạm thì η nhỏ hơn hoặc bằng 1, khi đó k bằng 1 và trục đó giữ nguyên.

Dòng thứ ba là tích luỹ đa view trước khi quyết định. 3DGS gốc đọc gradient tích luỹ tại thời điểm quyết định mà không kiểm tra các view có đồng thuận hay không. SADGS yêu cầu high_ratio hoặc low_ratio lớn hơn 0,8, nghĩa là hơn 80% số view phải đồng ý. Nhờ vậy, một góc chụp bất thường đơn lẻ không thể tự mình kích hoạt split hay prune.

Dòng thứ tư là vị trí của các Gaussian con khi split. 3DGS gốc lấy mẫu ngẫu nhiên từ phân phối chuẩn N(0, diag(s)²). SADGS đặt các Gaussian con trên một lưới tất định, cách đều nhau một khoảng căn 12 nhân σ. Cách đặt này tất định và có thể tái lập.

Chú thích cuối slide cũng cần nhấn mạnh. Compact box, bộ lọc low-pass 2D và optimizer hybrid chạy giống nhau ở cả hai nhánh, như đã nói ở slide "kế thừa và mới". Vì vậy chúng không được đưa vào bảng này, dù đã có mặt ở slide trước. Tiếp theo, chúng ta sẽ xem những khác biệt này mang lại ưu điểm cốt lõi gì cho SADGS, và cũng xem những gì chưa được chứng minh.

## Slide 49: Ưu điểm cốt lõi của SADGS: hội tụ đúng chỗ, không phải "ít Gaussian hơn"
Trước khi tổng kết, tôi muốn chốt lại ưu điểm của SADGS cho thật chính xác, vì đây là chỗ rất dễ nói quá.

Điều đầu tiên: xin đừng hiểu SADGS là phương pháp "tạo ra ít Gaussian hơn". Trong repo không có bằng chứng nào cho thấy SADGS dùng ít Gaussian hơn 3DGS gốc khi đạt cùng chất lượng. Cơ chế duy nhất có thể chủ động giảm mạnh số Gaussian N ở giai đoạn cuối lại không nằm trong những gì đang chạy mặc định, như các bạn đã thấy ở slide tóm tắt. Thêm nữa, sau iteration 15.000 thì N được giữ nguyên, nên nếu có khác biệt thì nó nằm ở cách densify trong 15.000 iteration đầu.

Vậy lợi ích thật là gì? Có ba điểm, và cả ba đều suy ra trực tiếp từ các công thức đã trình bày.

Thứ nhất, ngân sách split được dùng đúng trục và đúng mức. Hệ số k_axis tính riêng cho từng trục và tăng theo bậc thang, tương ứng với mức độ vi phạm. Còn ADC gốc thì lúc nào cũng tách làm hai và chia đều mọi trục cho 1,6.

Thứ hai, quyết định dựa trên đa số view đồng thuận, tức là tỉ lệ trên 80%. Nhờ vậy, một góc chụp đơn lẻ bất thường khó mà kéo cả quá trình densify phản ứng theo nhiễu.

Thứ ba, tín hiệu η đo trực tiếp tỉ lệ giữa kích thước Gaussian và độ chi tiết của ảnh. Nó bổ sung cho gradient, chứ không thay thế. Như ta đã học, gradient chỉ phản ánh trạng thái hội tụ hiện tại, và còn có thể bị triệt tiêu giữa các pixel trong cùng một ảnh.

Vì vậy tôi gọi ưu điểm này là "hội tụ đúng chỗ": Gaussian được thêm vào nơi thật sự thiếu chi tiết và theo đúng hướng thiếu.

Cuối cùng, có một điểm còn bỏ ngỏ và phải nói thẳng: repo không có số đo PSNR, SSIM, LPIPS hay thời gian chạy thực tế nào so sánh trực tiếp hai nhánh trên cùng scene, cùng checkpoint. Nghĩa là các lợi ích trên hiện mới là lợi ích lý thuyết, có cơ sở từ công thức, và chưa được định lượng.

## Slide 50: Tổng kết: từ 3DGS gốc tới SADGS
Giờ ta nhìn lại toàn bộ hành trình qua năm ý chính.

Một, về biểu diễn và render. Ta bắt đầu với Gaussian 3D tường minh, chiếu xuống ảnh bằng EWA splatting, trộn màu bằng alpha blending và rasterize theo tile. Đây là nền tảng chung, hai nhánh giữ nguyên, SADGS không thay đổi gì ở tầng này.

Hai, về gradient. Gradient phải được tích luỹ đúng cách: lấy chuẩn trước rồi mới cộng, để các view không triệt tiêu lẫn nhau. Ta cũng thấy vấn đề triệt tiêu thật sự nằm giữa các pixel trong cùng một ảnh, và nó được xử lý bằng một kênh gradient trị tuyệt đối riêng.

Ba, về ADC gốc. Clone, split và prune đều dựa thuần trên gradient. Đây là khung sườn mà SADGS kế thừa nguyên vẹn, gồm cả lịch trình densify, reset opacity và prune.

Bốn, về SADGS. Phương pháp thêm tín hiệu η, là tỉ số giữa kích thước Gaussian và bước sóng texture cục bộ. Nó tích luỹ quyết định nhất quán qua nhiều view và split dị hướng theo đúng trục bị vi phạm. Điểm then chốt là η được kết hợp với gradient bằng phép OR chứ không thay thế gradient, và toàn bộ phần này đang chạy mặc định trong code.

Năm, và là điều tôi muốn các bạn nhớ nhất: trung thực về hiệu năng. Repo không có số đo thời gian chạy thực tế hay số đo chất lượng nào để khẳng định định lượng rằng SADGS "nhanh hơn" hay "ít Gaussian hơn". Suốt bài, tôi chỉ trình bày những gì công thức và code thể hiện, không suy diễn thêm.

Tóm lại, SADGS là một mở rộng có chủ đích của ADC: giữ nguyên nền tảng, và làm cho quyết định "thêm Gaussian ở đâu, theo hướng nào" có cơ sở hơn. Còn việc đo xem cải tiến đó đáng giá bao nhiêu bằng con số là bước tiếp theo, cần một benchmark có phương pháp rõ ràng.

## Slide 51: Cảm ơn
Cảm ơn các bạn đã theo dõi đến cuối. Toàn bộ công thức trong bài đều đã được đối chiếu trực tiếp với code: train.py, gaussian_model.py, freq_utils.py, loss_utils.py, file cấu hình tham số, hai file CUDA của rasterizer là forward.cu và auxiliary.h, cùng tài liệu trong thư mục DOCS/BOOK. Nếu có chỗ nào chưa rõ, về công thức hay về vị trí trong code, xin mời các bạn đặt câu hỏi, chúng ta cùng thảo luận.
