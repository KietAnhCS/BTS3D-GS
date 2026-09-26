# Kịch bản thuyết trình 45 phút — 3D Gaussian Splatting & SADGS (60 slide)

> Khớp với `DOCS/Slides67/Slides45min/main45.pdf` — **60 trang**, thứ tự `\input` trong `main45.tex`:
> `part45_01..09` → `part45_11` → `part45_sadgsx_01..15` → `part45_10` (slide kết nằm cuối cùng).

## Phân bổ thời gian (45 phút = 2700 giây)

| Khối | Slide | Số slide | Thời lượng | Nhịp |
|---|---|---|---|---|
| A. Nền tảng 3DGS (Gaussian, SH, chiếu, rasterizer) | 1–11 | 11 | ~6,5 phút | ~35 s/slide, đi nhanh |
| B. Loss, gradient, optimizer, ADC tổng quan | 12–28 | 17 | ~10 phút | ~35 s/slide |
| C. Tổng kết cải tiến & thực nghiệm | 29–37 | 9 | ~5,5 phút | ~35 s/slide |
| D. Kế thừa FastGS + bug opacity reset + bảng so sánh | A–C | 3 | ~2 phút | ~40 s/slide |
| **E. SADGS chuyên sâu (trọng tâm)** | **S1–S15** | **15** | **~15 phút** | **~60 s/slide** |
| F. Quan sát cuối, tổng kết, Q&A | 58–60 | 3 | ~2,5 phút | thong thả |
| Dự phòng (hụt nhịp, hỏi xen kẽ) | — | — | ~3,5 phút | — |
| **Tổng** | **60** | **60** | **45 phút** | |

**Nếu bị hụt giờ:** cắt khối A xuống ~5 phút (slide 5–6 SH nói 1 câu rồi chuyển), và trong khối E gộp
S5+S6 (clone/expand) và S9+S10 (optimizer/3D filter) thành 1 lượt nói mỗi cặp. **Tuyệt đối không cắt**
S1–S4 (structure tensor → $\eta$ → đa view → split dị hướng) vì đó là đóng góp cốt lõi của SADGS.



---
## Slide 1: Trang bìa
**Lời thuyết trình (rút gọn):**
Kính chào thầy cô và các bạn. Em xin trình bày đồ án "Quy trình Render 3D Gaussian Splatting và Cải tiến SADGS", tên gọi tắt là SADGS, với phụ đề "Từ toán học nền tảng đến Adaptive Density Control chuyên sâu". Đồ án được thực hiện bởi nhóm Digital Twin GS, dựa trên việc tái hiện và rút gọn thuật toán SADGS gốc để chạy được trên phần cứng phổ thông. Trong 45 phút tới, chúng em sẽ đi nhanh qua nền tảng toán học, sau đó tập trung vào đóng góp chính là cải tiến Adaptive Density Control, và kết quả thực nghiệm đạt được. Sau đây em xin trình bày nội dung chi tiết.

---
## Slide 2: Agenda
**Lời thuyết trình (rút gọn):**
Bài trình bày gồm bảy phần chính. Một, nền tảng 3D Gaussian và Spherical Harmonics — sẽ đi rất nhanh vì chỉ là kiến thức nền. Hai, chiếu phối cảnh và rasterizer khả vi. Ba, hàm mất mát, gradient và optimizer. Bốn — trọng tâm của đồ án — Adaptive Density Control, đặc biệt là các cải tiến nhóm em đề xuất. Năm, ba đòn bẩy tăng tốc của SADGS. Sáu, kết quả thực nghiệm khi triển khai SADGS trên Colab T4. Bảy, quan sát, giới hạn và kết luận. Vì thời lượng có hạn, phần toán nền tảng sẽ được lược gọn tối đa, dành phần lớn thời gian cho đóng góp chính và số liệu thực nghiệm. Bắt đầu với động lực ra đời của 3D Gaussian Splatting.

---
## Slide 3: Động lực: vì sao cần một biểu diễn cảnh mới?
**Lời thuyết trình (rút gọn):**
Trước 3DGS, NeRF là hướng tiếp cận thống trị. NeRF biểu diễn cảnh "implicit" — thông tin màu và mật độ mã hóa ẩn trong trọng số một mạng MLP. Chất lượng cao nhưng render phải ray-marching, bắn tia và truy vấn mạng liên tục dọc tia, nên rất chậm, có thể mất vài giây mỗi khung hình. 3D Gaussian Splatting chuyển sang biểu diễn "explicit": cảnh được mô tả tường minh bằng hàng trăm nghìn quả cầu Gaussian mờ có vị trí, hình dạng rõ ràng. Nhờ đó render chỉ cần rasterization — kỹ thuật đồ họa truyền thống nhanh hơn ray-marching rất nhiều — đạt hàng chục đến hàng trăm khung hình mỗi giây, gần thời gian thực. Đây chính là lý do 3DGS trở thành hướng đi mới. Tiếp theo, ta xem một Gaussian 3D thực chất là gì.

---
## Slide 4: 3D Gaussian là gì?
**Lời thuyết trình (rút gọn):**
Đây là công thức trung tâm của phương pháp: hàm mật độ $G(x) = \exp(-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu))$. Trực quan, điểm x càng gần tâm $\mu$ thì giá trị càng lớn, càng xa thì giảm về 0, tạo hình "quả chuông" mờ dần. Ma trận $\Sigma^{-1}$ quyết định quả chuông giãn nở nhanh hay chậm theo từng hướng — tức hình dạng và độ định hướng. Mỗi Gaussian được đặc trưng bởi bốn nhóm tham số học được: vị trí tâm $\mu$, hiệp phương sai $\Sigma$ quy định hình dạng, độ mờ opacity $\alpha$, và màu sắc qua hệ số Spherical Harmonics. Bốn tham số này là những gì mô hình tối ưu trong suốt quá trình huấn luyện. Tiếp theo, chúng ta sẽ đi nhanh qua phần chiếu phối cảnh và rasterizer trước khi vào phần trọng tâm: Adaptive Density Control.


---
## Slide 5: Tham số hoá hiệp phương sai
**Lời thuyết trình (rút gọn):**
Nếu học trực tiếp ma trận hiệp phương sai $\Sigma$ với 6 giá trị tự do, gradient có thể khiến nó mất tính positive semi-definite, làm công thức Gaussian vô nghĩa. Giải pháp: không học $\Sigma$ trực tiếp, mà học một vector scale $s$ ba chiều và một quaternion $q$ bốn chiều biểu diễn phép xoay. Từ đó dựng lại $\Sigma = R S S^T R^T$, với $S$ là ma trận đường chéo từ $s$, $R$ là ma trận xoay từ $q$. Cách phân tích kiểu SVD này đảm bảo $\Sigma$ luôn hợp lệ dù $s$, $q$ nhận giá trị gì. Scale ban đầu được khởi tạo dựa trên khoảng cách tới điểm lân cận trong point cloud COLMAP. Về mặt hình học, $\Sigma$ chính là một ellipsoid: trục và độ dài trục lần lượt là eigenvector và eigenvalue của nó — Gaussian càng dẹt thì càng bất đẳng hướng. Tiếp theo, ta xem màu sắc được biểu diễn thế nào.

---
## Slide 6: Màu sắc qua Spherical Harmonics (SH)
**Lời thuyết trình (rút gọn):**
Mỗi Gaussian không lưu một màu RGB cố định mà lưu hệ số Spherical Harmonics, vì màu quan sát thực tế thay đổi theo góc nhìn — ví dụ hiệu ứng phản xạ, độ bóng. Biểu diễn màu như hàm phụ thuộc hướng nhìn qua hệ số SH cho phép mô hình tính lại màu phù hợp với từng góc camera. Hệ số SH bậc 0, gọi là thành phần DC, được khởi tạo trực tiếp từ màu RGB quan sát qua công thức $f_{dc} = \text{RGB2SH}(\text{color})$. Bậc càng cao thì càng biểu diễn được nhiều biến thiên màu phức tạp theo góc nhìn, nhưng tốn thêm bộ nhớ — nên khi huấn luyện, bậc SH được tăng dần chứ không dùng tối đa ngay. Vì đồ án đã trình bày đầy đủ phần suy diễn toán học của SH, trong bài báo cáo hôm nay nhóm em xin phép bỏ qua phần này để dành thời gian tập trung vào trọng tâm chính: Adaptive Density Control.

---
## Slide 7: Tổng quan pipeline render
**Lời thuyết trình (rút gọn):**
Sau khi đã xong phần nền tảng 3D Gaussian, ta chuyển sang phần thứ hai: chiếu phối cảnh và rasterizer khả vi — bước biến các Gaussian 3D cùng hệ số SH vừa học thành một bức ảnh 2D cụ thể. Pipeline gồm năm bước: chuyển từ tọa độ thế giới sang tọa độ camera bằng ma trận quay R và tịnh tiến t; chiếu Gaussian 3D xuống mặt phẳng ảnh 2D; tiling để xác định Gaussian nào ảnh hưởng tile nào; sorting theo độ sâu trong từng tile; và cuối cùng alpha blending để ra màu pixel. Toàn bộ chuỗi năm bước này đều khả vi, cho phép lan truyền gradient ngược từ ảnh render về tham số Gaussian. Tiếp theo, ta đi vào chi tiết bước đầu tiên: phép chiếu camera pinhole và chiếu hiệp phương sai.

---
## Slide 8: Chiếu hiệp phương sai (EWA splatting)
**Lời thuyết trình (rút gọn):**
Phép chiếu phối cảnh phi tuyến do có phép chia cho z, nên không thể áp dụng trực tiếp cho hình dạng elip 3D của Gaussian. Giải pháp là xấp xỉ tuyến tính cục bộ quanh tâm Gaussian bằng ma trận Jacobian J. Hiệp phương sai 2D sau chiếu được tính bằng $\Sigma' = JW\Sigma W^TJ^T$, với W là phần quay của ma trận view — đây là quy tắc biến đổi hiệp phương sai qua một ánh xạ tuyến tính gần đúng. Kết quả là mỗi Gaussian 3D trở thành một ellipse 2D cụ thể trên ảnh, có tâm và hình dạng xác định rõ ràng. Kỹ thuật này gọi là Elliptical Weighted Average splatting, viết tắt EWA splatting, nền tảng kinh điển của các phương pháp splatting. Tiếp theo, ta sẽ xem cách SADGS tối ưu vùng ảnh hưởng của ellipse này để tăng tốc rasterization.


---
## Slide 9: Compact Box — đòn bẩy tăng tốc #1 của SADGS
**Lời thuyết trình (rút gọn):**
Sau khi chiếu Gaussian 3D thành ellipse 2D, hệ thống cần một bounding-box bao quanh để biết vùng ảnh hưởng trên ảnh. Cài đặt gốc dùng bounding-box 3-sigma — an toàn về thống kê nhưng khá rộng, khiến Gaussian chạm nhiều tile hơn mức cần thiết, gây lãng phí tính toán. SADGS thay bằng compact box nhỏ gọn hơn, điều chỉnh qua tham số mult, mặc định 0.5, có thể tăng lên 0.7 cho cảnh lớn để tránh cắt mất phần đuôi Gaussian quan trọng. Kết quả là giảm đáng kể số tile mỗi Gaussian phải rasterize, tức giảm tải tính toán mà không đánh đổi nhiều chất lượng hình ảnh. Đây là đòn bẩy tăng tốc đầu tiên và quan trọng nhất của SADGS. Tiếp theo, ta xem công thức alpha blending — bước kết hợp các Gaussian này thành màu pixel cuối cùng.

---
## Slide 10: Alpha blending (front-to-back)
**Lời thuyết trình (rút gọn):**
Đây là công thức quyết định màu cuối cùng của mỗi pixel: C bằng tổng của ci nhân alpha_i nhân Ti, theo đúng thứ tự độ sâu đã sắp xếp. Trong đó ci là màu Gaussian giải mã từ hệ số SH, alpha_i là độ mờ tại điểm đó, còn Ti là transmittance — độ truyền sáng còn lại, bằng tích của (1 trừ alpha_j) cho mọi Gaussian j gần camera hơn i. Nói cách khác, Ti đo "còn bao nhiêu ánh sáng lọt qua" sau khi các lớp phía trước đã hấp thụ một phần. Gaussian càng gần camera thì đóng góp càng lớn vào màu cuối cùng, vì chưa bị lớp nào che khuất. Đây là mô hình front-to-back alpha compositing kinh điển. Vì Ti giảm theo cấp số nhân khi có nhiều lớp che, khi Ti đủ nhỏ, rasterizer có thể dừng sớm (early stopping) để tiết kiệm tính toán mà không ảnh hưởng chất lượng. Slide sau sẽ tổng kết toàn bộ phương trình render.

---
## Slide 11: Tổng kết phương trình render
**Lời thuyết trình (rút gọn):**
Slide này tổng kết toàn bộ phần chiếu phối cảnh và rasterizer bằng một phương trình duy nhất: C tại p bằng tổng ci nhân alpha_i nhân tích (1 trừ alpha_j) với j nhỏ hơn i — chính là công thức blend vừa nêu, viết gọn thành phương trình render hoàn chỉnh cho cả pipeline. Điều cốt lõi là toàn bộ phương trình này khả vi theo mọi tham số: vị trí tâm mu, hiệp phương sai Sigma, độ mờ alpha, và hệ số SH quyết định màu. Nhờ tính khả vi đầy đủ này, ta huấn luyện trực tiếp bằng gradient descent: so sánh ảnh render với ảnh RGB thực tế, rồi lan truyền ngược sai số qua toàn bộ chuỗi Gaussian 3D, chiếu 2D, trường alpha, blend, để cập nhật từng tham số. Đây là điểm khép lại phần render, và cũng là lúc chuyển sang phần tiếp theo: hàm mất mát và cách tối ưu các tham số này.

---
## Slide 12: Hàm mất mát: kết hợp L1 và D-SSIM
**Lời thuyết trình (rút gọn):**
Chuyển sang phần huấn luyện: hàm mất mát dùng để tối ưu tham số Gaussian. Loss kết hợp hai thành phần bổ trợ nhau. Thứ nhất là L1, đo sai khác trung bình theo từng pixel giữa ảnh render và ảnh gốc — tín hiệu cơ bản, đơn giản nhưng nhạy với nhiễu. Thứ hai là D-SSIM, bằng 1 trừ SSIM, phản ánh sai lệch về cấu trúc chứ không chỉ độ sáng từng điểm. Công thức tổng là trung bình có trọng số: L bằng (1 trừ lambda) nhân L1 cộng lambda nhân (1 trừ SSIM). Giá trị lambda_dssim mặc định của 3DGS gốc là 0.2, ưu tiên L1 nhiều hơn; SADGS tăng lên 0.25 để nhấn mạnh cấu trúc, giữ chi tiết hình học tốt hơn khi giảm số Gaussian. Tiếp theo ta sẽ đi sâu vào cơ chế của SSIM.


---
## Slide 13: Gradient tập trung tại biên Gaussian
**Lời thuyết trình (rút gọn):**
Một quan sát quan trọng: gradient trong không gian view không phân bố đều trên Gaussian mà tập trung mạnh nhất ở vùng biên — nơi ranh giới với nền hoặc Gaussian khác, cũng là nơi ảnh render dễ lệch nhiều nhất so với ảnh thật. Ý nghĩa thực tiễn rất lớn: gradient tại biên chính là tín hiệu để hệ thống quyết định có densify hay không, tức có thêm Gaussian mới vào vùng đó không, ở bước Adaptive Density Control. Ngược lại, vùng gradient thấp nghĩa là Gaussian đã khớp tốt với dữ liệu, không cần chia tách thêm. Đây là nền tảng kết nối phần loss, gradient với phần điều chỉnh mật độ Gaussian. Tiếp theo, ta tổng kết lại toàn bộ vòng lặp huấn luyện trước khi đi sâu vào phần densify/prune.

---
## Slide 14: Tổng kết vòng lặp huấn luyện
**Lời thuyết trình (rút gọn):**
Toàn bộ vòng lặp huấn luyện gồm năm bước lặp lại liên tục: Render ảnh từ tập Gaussian hiện tại theo góc camera; Tính loss kết hợp L1 và D-SSIM; Backward — lan truyền ngược gradient qua rasterizer khả vi về từng tham số; optimizer.step() dùng sparse Adam để cập nhật tham số hiệu quả; và cuối cùng, định kỳ chứ không phải mọi iteration, là Densify/Prune — điều chỉnh mật độ Gaussian. Sau bước năm, vòng lặp quay lại bước một, lặp hàng chục nghìn iteration tới khi hội tụ. Đây là bước chuyển rất quan trọng: từ đây, em sẽ đi sâu vào chính phần trọng tâm của đồ án — Adaptive Density Control, cơ chế quyết định trực tiếp chất lượng và hiệu năng của SADGS.

---
## Slide 15: Tổng quan Adaptive Density Control (ADC)
**Lời thuyết trình (rút gọn):**
ADC giải quyết vấn đề: point cloud khởi tạo từ COLMAP thường không hoàn hảo — có vùng quá thưa, thiếu điểm ở khu vực chi tiết phức tạp, trong khi có Gaussian quá to che khuất chi tiết nhỏ. ADC chạy định kỳ sau mỗi khoảng densification_interval iteration, điều chỉnh động mật độ Gaussian theo độ phức tạp hình học thực tế của scene. ADC gồm hai nhóm thao tác: Densify, gồm clone (nhân bản) và split (tách), dùng để thêm Gaussian ở nơi thiếu hình học; và Prune, cắt tỉa Gaussian thừa, mờ hoặc không còn đóng góp giá trị. Mục tiêu là cân bằng chất lượng tái tạo với số lượng Gaussian, vì số lượng lớn ảnh hưởng trực tiếp tới bộ nhớ và tốc độ render. Tiếp theo, ta xem tín hiệu cụ thể để quyết định densify: gradient view-space tích lũy.

---
## Slide 16: Tích luỹ gradient view-space
**Lời thuyết trình (rút gọn):**
Để quyết định Gaussian nào cần densify, hệ thống cần tín hiệu định lượng: gradient view-space tích lũy qua thời gian. Với mỗi Gaussian, gradient vị trí 2D sinh ra từ mỗi lần render ở góc camera khác nhau được cộng dồn lại, đồng thời đếm denom — số lần Gaussian thực sự được nhìn thấy. Gradient trung bình g-bar bằng tổng gradient view-space chia cho denom. Nếu g-bar lớn, nghĩa là Gaussian liên tục gây sai số đáng kể qua nhiều góc nhìn, nó trở thành ứng viên clone hoặc split. Cách tích lũy qua nhiều view giúp tín hiệu ổn định hơn so với dùng một lần render đơn lẻ. Tuy nhiên, chính cách lấy trung bình vector này cũng tiềm ẩn một vấn đề quan trọng — gradient cancellation — mà slide tiếp theo sẽ phân tích.


---
## Slide 17: Vấn đề triệt tiêu gradient (gradient cancellation)
**Lời thuyết trình (rút gọn):**
Đây là một hạn chế đã biết của 3DGS gốc: khi lấy trung bình gradient từ nhiều góc nhìn, nếu các vector này ngược hướng nhau, chúng có thể triệt tiêu lẫn nhau thay vì cộng dồn. Hệ quả là dù mỗi góc nhìn riêng lẻ đều báo Gaussian đó có sai số lớn, gradient trung bình tổng hợp lại có thể rất nhỏ — Gaussian bị bỏ sót khỏi danh sách densify dù đáng lẽ cần xử lý. Đây là hạn chế nội tại của việc chỉ dùng gradient trung bình làm tín hiệu duy nhất. SADGS khắc phục bằng multi-view consistency score, tức điểm nhất quán đa góc nhìn, sẽ trình bày kỹ hơn ở phần sau. Tiếp theo, chúng ta xem cơ chế densify với hai thao tác cụ thể: clone và split.

---
## Slide 18: Clone (nhân bản)
**Lời thuyết trình (rút gọn):**
Clone áp dụng cho Gaussian nhỏ nhưng thiếu hình học — vùng đó cần thêm chi tiết, nhưng bản thân Gaussian đã đủ nhỏ nên không cần tách. Cơ chế: nhân đôi Gaussian, bản sao đặt lệch một khoảng nhỏ theo đúng hướng gradient vị trí, tức về phía sai số lớn nhất. Quan trọng là clone không đổi kích thước hay ma trận hiệp phương sai — cả bản gốc lẫn bản sao giữ nguyên hình dạng. Mục đích duy nhất là tăng mật độ điểm để lấp đầy vùng thiếu chi tiết, như thêm quân tiếp viện vào đúng khu vực thiếu người chứ không thay đổi cấu trúc đã có. Tiếp theo là thao tác còn lại trong nhóm densify: split.

---
## Slide 19: Split (tách)
**Lời thuyết trình (rút gọn):**
Ngược với clone, split áp dụng cho Gaussian quá to, thuộc trường hợp over-reconstruction — một Gaussian đang cố biểu diễn vùng quá nhiều chi tiết so với kích thước của nó. Khi split, Gaussian cha tách thành hai Gaussian con nhỏ hơn, scale giảm theo hệ số cố định. Điểm đáng chú ý: vị trí các Gaussian con không cố định mà được lấy mẫu ngẫu nhiên theo phân phối chuẩn, xây dựng dựa trên chính ma trận hiệp phương sai của Gaussian cha — nên các điểm con phân bố theo đúng hình dạng, hướng và độ trải rộng mà cha vốn biểu diễn, tránh lệch lạc ngẫu nhiên. Sau khi có clone và split để tăng mật độ, ta chuyển sang chiều ngược lại: prune, để cắt giảm Gaussian dư thừa.

---
## Slide 20: Prune theo độ quan trọng đa góc nhìn (SADGS)
**Lời thuyết trình (rút gọn):**
Đây là một đóng góp chính của SADGS. Trong 3DGS gốc, prune chỉ dựa trên ngưỡng opacity thấp hoặc kích thước quá lớn — cách này đơn giản nhưng không phản ánh đúng mức đóng góp thực tế của từng Gaussian vào chất lượng ảnh. SADGS cải tiến bằng cách tính điểm importance cho mỗi Gaussian, dựa trên đóng góp thực tế vào chất lượng render, và tổng hợp điểm này qua nhiều góc nhìn khác nhau chứ không chỉ một góc. Gaussian có importance thấp nhất bị cắt trước tiên, bất kể opacity hay kích thước — ngay cả Gaussian opacity cao nhưng đóng góp thực tế thấp vẫn có thể bị loại. Kết quả: mô hình gọn hơn đáng kể về số lượng Gaussian nhưng vẫn giữ chất lượng tái tạo — đây là một trong những đòn bẩy chính giúp SADGS tăng tốc độ render. Tiếp theo, ta xem số lượng Gaussian biến đổi ra sao theo thời gian huấn luyện.


---
## Slide 21: Reset opacity định kỳ
**Lời thuyết trình (rút gọn):**
Cơ chế cuối trong Adaptive Density Control: cứ mỗi opacity_reset_interval, mặc định 3000 iteration, toàn bộ opacity của mọi Gaussian bị đặt lại về giá trị thấp qua sigmoid-inverse. Mục đích là buộc optimizer đánh giá lại vai trò từng Gaussian, thay vì để những Gaussian đã đạt opacity cao giữ nguyên trạng thái dù không còn cần thiết. Sau reset, Gaussian thực sự quan trọng sẽ nhanh chóng tăng opacity trở lại nhờ gradient; Gaussian dư thừa không kịp hồi phục sẽ bị prune ở vòng kế tiếp. Đây là bước dọn dẹp định kỳ giúp scene không tích tụ Gaussian "chết" theo thời gian.

Đến đây, em xin kết thúc phần tổng quan Loss, Gradient, Optimizer và ADC. Bây giờ ta đi sâu vào cách tính điểm Importance đa view — đóng góp cốt lõi của SADGS.

---
## Slide 22: Tổng hợp counts qua nhiều view: đầu vào cho Importance
**Lời thuyết trình (rút gọn):**
Sau khi có counts cho từng Gaussian ở từng view riêng lẻ, bước tiếp theo là tổng hợp lại. Mỗi Gaussian có một vector counts ứng với các view nó xuất hiện. Lưu ý: tổng counts trên một view luôn ≥ số pixel lỗi thật của view đó, vì một pixel lỗi có thể nằm trong footprint của nhiều Gaussian chồng lấn nên bị đếm lặp — điều này có chủ đích, vì mục tiêu là quy trách nhiệm cho tất cả Gaussian liên quan, không chia đều lỗi. Về cài đặt, mỗi lần render một view, kernel CUDA cộng dồn atomic vào tensor metricCount, rồi lũy kế qua tất cả view thành full_metric_counts. Vector counts tổng hợp qua toàn bộ V view này chính là đầu vào trực tiếp cho bước tính Importance.

---
## Slide 23: Importance score: trung bình counts qua các view, làm tròn xuống
**Lời thuyết trình (rút gọn):**
Đây là công thức trung tâm, khớp với hàm compute_gaussian_score_structgs trong code. Importance của một Gaussian bằng trung bình cộng counts qua tất cả view, rồi làm tròn xuống (torch.div chế độ floor). Một Gaussian được coi là "quan trọng" — giữ lại hoặc ưu tiên densify — khi Importance vượt quá 5. Dùng trung bình thay vì max hay tổng có lý do: nó đảm bảo Gaussian phải nhất quán gây lỗi qua nhiều góc nhìn, chứ không chỉ nổi bật ở một view đơn lẻ, mới được coi là thực sự quan trọng — tránh việc một góc nhìn bất thường quyết định số phận Gaussian.

Từ điểm Importance này, ta chuyển sang xem nó được dùng ở đâu trong quyết định densify.

---
## Slide 24: Sơ đồ quyết định densify đầy đủ
**Lời thuyết trình (rút gọn):**
Đây là sơ đồ tổng hợp toàn bộ logic densify của SADGS, áp dụng mỗi 100-500 vòng lặp, trong khoảng iteration 500-15000. Bước một: dựa vào kích thước Gaussian so với ngưỡng để chọn nhánh. Nếu nhỏ — "under-reconstruction" — thì kiểm tra gradient có dấu vượt ngưỡng và Importance > 5; nếu đúng cả hai thì CLONE, tạo bản sao giữ nguyên tham số gốc. Nếu to — "over-reconstruction" — thì kiểm tra gradient trị tuyệt đối vượt ngưỡng cao hơn và Importance > 5; nếu đúng thì SPLIT thành hai Gaussian con, xóa Gaussian gốc. Điểm mấu chốt: nếu Importance không thỏa, 3DGS gốc vẫn densify chỉ cần gradient đạt, nhưng SADGS kiên quyết chặn lại — đây chính là đóng góp cốt lõi giúp tránh sinh thừa Gaussian không cần thiết.


---
## Slide 25: Top-k cứng vs. Multinomial: hai chiến lược pruning
**Lời thuyết trình (rút gọn):**
Em so sánh hai chiến lược pruning. Top-k cứng: sắp xếp ứng viên theo điểm số rồi xoá đúng số lượng thấp nhất — dễ đoán, luôn xoá đúng ngân sách, nhưng cắt theo ngưỡng cứng nên dễ xoá sạch cả một cụm điểm tương tự nhau cùng lúc. Multinomial: rút mẫu không hoàn lại theo trọng số tỉ lệ nghịch với (1 trừ P) — đa dạng hơn, tránh xoá đồng loạt một vùng, nhưng có nhiễu nên số lượng xoá thực tế có thể ít hơn ngân sách. Vì sao tránh "xoá sạch một cụm" lại quan trọng? Vì trên hệ Gaussian thật, một cụm điểm cao thường nằm ở cùng một vùng hình học — xoá sạch cùng lúc tạo lỗ hổng lớn khó phục hồi. Để thấy rõ cơ chế này hoạt động ra sao trên dữ liệu thật, mình chuyển sang một toy model 2D trực quan.

---
## Slide 26: Toy model 2D: mục tiêu và một Gaussian
**Lời thuyết trình (rút gọn):**
Từ đây em dùng một toy model 2D — mô hình thu nhỏ, đơn giản hoá để minh hoạ trực quan cơ chế ADC mà không cần dữ liệu 3D phức tạp. Ảnh mục tiêu chỉ 64x64 pixel: một hình tròn phẳng cộng một vùng chi tiết nhỏ (dải mảnh, chấm màu, ô vuông) để tạo cả vùng dễ và khó tái tạo. Mỗi Gaussian 2D là hàm mũ dạng toàn phương âm, tham số hoá bởi tâm mu, hai scale và một góc xoay; footprint giới hạn ở 3-sigma. Việc render dùng alpha-compositing theo thứ tự chỉ số — đúng nguyên lý blending đã học ở phần 3D, chỉ đơn giản hoá còn 2D cho dễ quan sát. Với nền tảng này, slide sau em sẽ so sánh trực tiếp kết quả có và không có ADC trên cùng toy model.

---
## Slide 27: So sánh trực tiếp: không ADC vs có ADC
**Lời thuyết trình (rút gọn):**
Đây là slide so sánh trực quan nhất của phần toy model, đặt ba kịch bản cạnh nhau: (1) không ADC, giữ N cố định bằng N0 ban đầu; (2) không ADC nhưng khởi tạo ngẫu nhiên ngay từ đầu với số lượng bằng đúng N cuối mà ADC đạt được; (3) có ADC đầy đủ, N tăng dần từ N0 lên N cuối theo đúng cơ chế densify. Điểm mấu chốt: giữa kịch bản 2 và 3, dù CÙNG số lượng Gaussian, kịch bản 2 với vị trí ngẫu nhiên vẫn không khớp tự nhiên vào vùng chi tiết cần thiết, nên chất lượng kém hơn. Kết luận cốt lõi: ADC không chỉ tăng số lượng N, mà quan trọng hơn là đặt đúng Gaussian vào đúng chỗ, có định hướng theo gradient của loss. Từ kết quả trên toy model này, slide tiếp theo sẽ cho thấy kết quả tương tự nhưng đầy đủ trên ba kịch bản thật, bao gồm cả SADGS.

---
## Slide 28: Kết quả render cuối cùng: 3 kịch bản đối chiếu
**Lời thuyết trình (rút gọn):**
Đây là slide chốt hạ của toàn bộ phần so sánh, đối chiếu ba kịch bản song song. Hàng một: ADC 3DGS gốc, N tăng dần lên mức cuối cùng, kèm loss L1. Hàng hai: ADC SADGS — đây là KẾT QUẢ QUAN TRỌNG NHẤT của toàn bộ đồ án: loss L1 cuối gần như TƯƠNG ĐƯƠNG với 3DGS gốc, nhưng số lượng Gaussian ÍT HƠN ĐÁNG KỂ. Nói cách khác, cùng chất lượng render nhưng chi phí bộ nhớ và tính toán thấp hơn hẳn, nhờ tổng hợp của Importance đa view, prune multinomial, và final prune. Hàng ba: không dùng ADC, giữ N cố định bằng N0, không clone/split/prune — chất lượng kém hơn rõ rệt, chứng minh ADC là thành phần không thể thiếu. Mỗi hàng gồm render cuối, bản đồ sai số, và ellipse 1.5-sigma. Sau đây chúng ta sẽ đi sâu vào từng cải tiến cụ thể để hiểu vì sao chúng hoạt động hiệu quả đến vậy.


---
## Slide 29: Tổng kết 5 cải tiến của SADGS trong ADC
**Lời thuyết trình (rút gọn):**
Em tổng kết nhanh 5 cải tiến SADGS đưa vào chu trình ADC so với 3DGS gốc. Một: quyết định densify không chỉ dựa gradient mà AND thêm với Importance đa view, lọc bớt ứng viên sai. Hai: split dùng tổng trị tuyệt đối gradient thay vì tổng có dấu, tránh bị triệt tiêu khi Gaussian đúng vị trí nhưng sai kích thước. Ba: bước prune không xoá cứng toàn bộ ứng viên mà rút mẫu multinomial theo trọng số, giữ lại phần còn hữu ích. Bốn: chặn trần opacity ở 0,8 để không "chôn" gradient của các lớp Gaussian phía sau. Và cuối cùng, cải tiến năm: thêm bước final prune sau khi densify kết thúc, tại t=1000 và t=1100, xoá theo alpha thấp hoặc xác suất Pruning cao — tạo ra hai bậc thang giảm N rõ rệt mà 3DGS gốc không có. Kết quả chung: loss L1 tương đương 3DGS gốc nhưng N nhỏ hơn hẳn — mô hình gọn nhất trong ba kịch bản so sánh. Tiếp theo, em chuyển sang phần ba đòn bẩy tăng tốc của SADGS.

---
## Slide 30: Vì sao 3DGS gốc chậm?
**Lời thuyết trình (rút gọn):**
Trước khi nói về tăng tốc, em phân tích 3DGS gốc chậm ở đâu. Mỗi iteration gồm bốn khâu: render, tính loss, backward, và định kỳ densify/prune. Với cảnh thực có hàng trăm nghìn đến hàng triệu Gaussian, ước tính render chiếm khoảng 45% thời gian, backward khoảng 35%, tính loss chỉ 10%, còn densify/prune tuy chỉ khoảng 10% tổng thời gian nhưng chạy định kỳ và gây đột biến chi phí mỗi lần kích hoạt. Nhìn vào tỉ trọng này, rõ ràng muốn tăng tốc hiệu quả thì phải nhắm đúng hai khâu chiếm phần lớn nhất là render và backward, thay vì tối ưu dàn trải. Đây chính là tiền đề để áp dụng định luật Amdahl: tăng tốc một phần hệ thống chỉ có lợi tương ứng với tỉ trọng thời gian phần đó chiếm trong toàn pipeline. Dựa trên phân tích này, SADGS chọn lọc đúng ba đòn bẩy tăng tốc, em sẽ giới thiệu tổng quan ngay sau đây.

---
## Slide 31: Ba đòn bẩy tăng tốc của SADGS (tổng quan)
**Lời thuyết trình (rút gọn):**
Đây là slide tổng quan quan trọng nhất của phần này: ba đòn bẩy tăng tốc chính của SADGS. Một, Multi-view score: đánh giá tầm quan trọng của Gaussian qua nhiều góc nhìn thay vì một view, giúp pruning chính xác hơn, giảm N hiệu quả hơn. Hai, Compact Box: giảm hệ số nhân bounding-box từ 3,0 mặc định xuống 0,5-0,7, khiến số tile mỗi Gaussian chạm tới giảm theo bình phương hệ số, kéo giảm trực tiếp chi phí rasterize. Ba, Sparse Adam: áp dụng lịch cập nhật phân tầng, giảm tần suất gọi optimizer.step() cho các tham số ít quan trọng như SH bậc cao. Cả ba đòn bẩy đều nhắm đúng những khâu chiếm tỉ trọng lớn theo phân tích Amdahl vừa nêu — rasterize, densify/prune, và bước cập nhật tham số — và gần như không chồng lấn nhau. Slide tiếp theo em sẽ cho thấy hiệu quả tích luỹ khi kết hợp cả ba.

---
## Slide 32: Hiệu năng tích luỹ — Biểu đồ thác nước
**Lời thuyết trình (rút gọn):**
Slide này tổng hợp hiệu quả ba đòn bẩy bằng biểu đồ thác nước, thể hiện mức tăng tốc tích luỹ khi bật lần lượt từng đòn bẩy lên nền baseline 3DGS gốc: bắt đầu từ baseline, cộng Compact Box, cộng tiếp Multi-view score, cộng tiếp Sparse Adam, và cuối cùng là SADGS đầy đủ với cả ba cùng hoạt động. Điểm mấu chốt là mỗi đòn bẩy đóng góp tăng tốc độc lập, và khi kết hợp cả ba thì mức tăng tốc tổng vượt trội hẳn so với dùng riêng lẻ. Điều này khớp hoàn toàn với dự đoán từ định luật Amdahl: vì ba đòn bẩy tác động vào ba khâu chi phí khác nhau, gần như không chồng lấn — rasterize, densify/prune, optimizer step — nên hiệu quả cộng dồn tự nhiên chứ không triệt tiêu lẫn nhau. Đây là bằng chứng thực nghiệm cho toàn bộ chiến lược thiết kế của SADGS.


---
## Slide 33: Tổng kết: từ 3DGS gốc đến SADGS
**Lời thuyết trình (rút gọn):**
Slide này tổng kết hành trình từ 3DGS gốc sang SADGS qua bảng so sánh tham số then chốt. SADGS thêm loss_thresh 0,1; đổi gradient thường sang Abs-GS với ngưỡng 0,0012 để khắc phục gradient cancellation; tham số dense dao động 0,001-0,013 theo kích thước cảnh; learning rate bậc cao giảm còn 0,005-0,02; và hệ số bounding-box mult giảm từ 3,0 xuống 0,5-0,7. Điểm mấu chốt: SADGS không đổi mô hình biểu diễn Gaussian, chỉ tối ưu tham số hóa và lịch huấn luyện dựa trên mô hình chi phí và định luật Amdahl đã phân tích, giúp tăng tốc huấn luyện đáng kể mà PSNR/SSIM gần như không đổi. Sau phần lý thuyết này, em chuyển sang kết quả thực nghiệm thực tế trên bản fork SADGS chạy trên Colab T4.

---
## Slide 34: SADGS: từ 24GB RTX 4090 xuống Colab T4 miễn phí
**Lời thuyết trình (rút gọn):**
SADGS là bản fork của nhóm, đóng gói lại SADGS gốc để chạy trên phần cứng phổ thông. Điểm quan trọng: nó không đổi bất kỳ công thức toán hay pipeline huấn luyện nào so với SADGS gốc — chỉ đổi cách đóng gói và vận hành, để bài toán vốn cần GPU 24GB như RTX 4090 chạy được trên Colab T4 miễn phí, chỉ khoảng 15GB VRAM. Cụ thể, CUDA extension được vendor sẵn trong repo thay vì submodule rời rạc, entry point bổ sung notebook có theo dõi tiến trình, và tín hiệu huấn luyện được đổi thành một Score tổng hợp log mỗi 1000 iteration. Đây là nền tảng để hiểu các bảng số liệu thực nghiệm tiếp theo, bắt đầu với chất lượng cuối cùng theo từng scene.

---
## Slide 35: Bảng 1: Chất lượng cuối cùng theo scene
**Lời thuyết trình (rút gọn):**
Đây là bảng kết quả quan trọng nhất. Score trung bình trên 4 scene đạt 0,7643, PSNR trung bình 24,46dB. Theo từng scene: playroom cao nhất với 0,8198, tiếp đến drjohnson 0,8027, rồi truck 0,7482, và thấp nhất là train chỉ 0,6867. Đáng chú ý, hai scene indoor — playroom và drjohnson — vượt trội hơn hai scene outdoor dù chỉ dùng khoảng 129-174 nghìn Gaussian, ít hơn hẳn so với 187-201 nghìn Gaussian của scene outdoor. Điều này cho thấy chất lượng không tỷ lệ thuận với số lượng Gaussian mà phụ thuộc nhiều vào đặc tính hình học và ánh sáng của cảnh. Tiếp theo em trình bày chi phí tính toán và lưu trữ để đo hiệu quả thực tế của pipeline này.

---
## Slide 36: Bảng 3: Chi phí & lưu trữ
**Lời thuyết trình (rút gọn):**
Về chi phí, kết quả khá ấn tượng cho phần cứng miễn phí: tổng thời gian train cả 4 scene chỉ 327,6 giây, khoảng 5,5 phút, tốc độ trung bình 85,5 iteration mỗi giây. VRAM đỉnh trung bình chỉ 0,88GB, cao nhất 1,16GB — rất thấp so với 14,56GB khả dụng trên T4, củng cố quan sát VRAM không phải nút thắt. Về lưu trữ, file point_cloud.ply trung bình nặng 171,6MB mỗi scene, khớp chính xác 248 byte trên mỗi Gaussian — kích thước một record 3DGS hoàn toàn chưa nén. Con số này là manh mối quan trọng cho phần giới hạn phép đo mà em trình bày ngay sau đây.

---
## Slide 37: Khả năng tái lập & giới hạn phép đo
**Lời thuyết trình (rút gọn):**
Trước khi sang phần Quan sát và Kết luận, nhóm nêu rõ giới hạn của phép đo này. Thứ nhất, đây chỉ là một lần chạy duy nhất, không cố định seed, không có error bar. Thứ hai, mẫu thử hẹp, chỉ 4 scene từ 2 dataset. Thứ ba và quan trọng nhất, ngân sách 7000 iteration chỉ bằng một phần ba so với lịch trình 30 nghìn iteration mà optimizer gốc thiết kế để chạy đủ. Thứ tư, quá trình này bỏ qua hoàn toàn bước prune cuối cùng final_prune_structgs. Vì vậy, kết luận công bằng nhất: Bảng 1 nên được xem là baseline có thể tái lập của riêng repo này, không phải benchmark chính thức để so sánh trực tiếp với số liệu công bố của SADGS hay 3DGS gốc.


---
---

# KHỐI D — Kế thừa FastGS & bug đã sửa (3 slide)

---
## Slide A: Kế thừa từ FastGS — 4 cơ chế nền luôn bật trong SADGS
**Thời lượng: ~40 giây**

**Lời thuyết trình:**
Trước khi đi vào đóng góp riêng của SADGS, em xin tách bạch rõ một điểm dễ gây nhầm. Có bốn cơ chế trong pipeline này là **kế thừa từ FastGS**, không phải đóng góp của SADGS. Thứ nhất, Fused Adam — gộp phép cập nhật Adam vào một kernel CUDA elementwise duy nhất; thuật toán không đổi, chỉ bớt overhead phóng kernel. Thứ hai, bộ lọc chống alias 3D — chặn dưới kích thước Gaussian theo khoảng cách camera. Thứ ba, Morton Z-order reordering — cứ 5000 iteration lại sắp xếp lại Gaussian trong bộ nhớ theo mã bit-interleave để tăng cache locality cho rasterizer tile-based. Thứ tư, random-init fallback — sinh point cloud ngẫu nhiên đều trong hình cầu bằng công thức r bằng R nhân u mũ một phần ba, khi COLMAP không trả về points3D. Cả bốn đều luôn bật, không còn cờ tắt mở. Riêng MCMC densification thì có sẵn code nhưng giữ tắt, vì nó xung đột kiến trúc với densify chính.

**Chuyển tiếp:** Nhưng trong quá trình chạy, nhóm em phát hiện một bug khiến điểm số sụt hẳn — xin trình bày ở slide sau.

**Nếu bị hỏi:**
- *Vì sao không dùng luôn CUDA backend của Faster-GS?* — Backend đó không trả về `radii` và `viewspace_points` mà thuật toán densify cần, nên nhóm áp dụng từng cơ chế lên code hiện có thay vì đổi backend.
- *Morton reorder có làm đổi kết quả không?* — Không; nó chỉ đổi thứ tự lưu trong bộ nhớ, kết quả render bất biến, chỉ khác tốc độ.

---
## Slide B: Bug đã sửa — `opacity_reset_interval` không co giãn theo iterations
**Thời lượng: ~40 giây**

**Lời thuyết trình:**
Đây là nguyên nhân thật của các điểm sụt Score mà nhóm em vừa trình bày. Tham số `opacity_reset_interval` bị hardcode bằng 3000 — con số này đúng cho lịch trình gốc 30 nghìn iteration, nhưng khi nhóm rút ngân sách xuống còn 3000 hoặc 7000 iteration thì nó **không co giãn theo**, nên reset opacity kích hoạt quá sớm, ngay giữa giai đoạn mô hình đang hội tụ. Bằng chứng là log thật trên scene HCM0539: PSNR đang 22,02 thì rơi thẳng xuống 5,92 đúng tại iteration 3000 — không phải nhiễu, mà là toàn bộ Gaussian bị ép độ đục về mức thấp cùng lúc. Cách sửa của nhóm là thêm một field mới `opacity_reset_frac` bằng 0,2, rồi tính `opacity_reset_interval` bằng 0,2 nhân với số iteration thực tế. Nhóm em xin nói thẳng: chưa có log "sau khi sửa" vì môi trường phát triển không có GPU để chạy lại.

**Chuyển tiếp:** Tiếp theo là bảng đối chiếu ba phương pháp để thấy rõ đâu là phần SADGS thực sự đóng góp.

**Nếu bị hỏi:**
- *Sao không đơn giản tắt opacity reset?* — Reset vẫn cần để thoát các cực tiểu cục bộ do Gaussian "mờ mà không chết"; vấn đề là thời điểm, không phải bản thân cơ chế.
- *0,2 lấy từ đâu?* — Là tỉ lệ 3000/15000 của lịch gốc trong giai đoạn densify, giữ nguyên tỉ lệ khi rút ngắn lịch.

---
## Slide C: Bảng tổng kết — 3DGS gốc vs FastGS (kế thừa) vs SADGS (mới)
**Thời lượng: ~40 giây**

**Lời thuyết trình:**
Bảng này gom lại toàn bộ sự khác biệt. Cột một, 3DGS gốc: không hề có khái niệm đo độ khớp kích thước giữa Gaussian và chi tiết ảnh; split thì luôn sinh đúng 2 con, vị trí lấy mẫu ngẫu nhiên. Cột hai, FastGS: phần densification giữ như 3DGS gốc, đóng góp nằm ở kernel, cache và khởi tạo. Cột ba, SADGS: đây mới là chỗ mới. SADGS đưa vào đại lượng eta — tỉ số giữa độ dài trục chiếu của Gaussian và bước sóng texture cục bộ; phân loại theo hai ngưỡng tau-high bằng 1,0 và tau-low bằng 0,1; và split dị hướng với số con trên mỗi trục bằng trần của căn eta của trục đó. Hai cơ chế còn lại — giãn Gaussian dưới cỡ và prune cuối — nhóm em phải nói rõ là **đã cài đặt đúng trong `gaussian_model.py` nhưng lời gọi đang bị comment trong `train.py`**, tức không chạy ở cấu hình mặc định.

**Chuyển tiếp:** Phần tiếp theo là trọng tâm của báo cáo: mười lăm slide đi sâu vào từng cơ chế của SADGS, đọc trực tiếp từ source code.

**Nếu bị hỏi:**
- *Vì sao để nguyên code đã tắt thay vì xoá?* — Để giữ khả năng bật lại khi có GPU benchmark; và để báo cáo trung thực về trạng thái repo.

---
---

# KHỐI E — SADGS CHUYÊN SÂU (15 slide, trọng tâm ~13 phút)

> Toàn văn 15 slide được gộp inline bên dưới theo đúng thứ tự trình chiếu. Bản rời của từng bot vẫn giữ ở
> `sadgsx_botNN_45min.md` cùng thư mục (tiện in riêng phần mình phụ trách).
>
> Nhãn `[S5-S6]`, `[S7-S8]`… nghĩa là một mục bao hai slide liên tiếp.


---

## [S1] Structure tensor đa tỉ lệ

### Slide: Tensor cấu trúc đa tỉ lệ — nền tảng đo cấu trúc ảnh của SADGS

**Thời lượng: ~75 giây**

**Lời thuyết trình:**

Trước khi nói về việc SADGS chia nhỏ Gaussian như thế nào, phải trả lời câu hỏi nền: làm sao máy biết chỗ nào trong ảnh là chi tiết mịn, chỗ nào là mảng phẳng. Câu trả lời của SADGS là tensor cấu trúc, công thức ở góc trên bên trái. Với mỗi pixel, ta làm mượt ảnh rồi lấy đạo hàm Sobel theo hai hướng x và y, nhân chéo hai đạo hàm đó thành một ma trận hai nhân hai, rồi làm mượt lần nữa bằng một cửa sổ Gauss bán kính rho. Có hai chi tiết đáng chú ý ở đây. Thứ nhất, đạo hàm chạy riêng trên từng kênh đỏ, lục, lam rồi mới cộng **năng lượng** lại — đây là phương pháp Di Zenzo, giúp không bỏ sót những cạnh mà hai màu khác nhau nhưng độ sáng bằng nhau. Thứ hai, bước làm mượt bằng rho là bắt buộc chứ không phải tuỳ chọn: nếu bỏ nó đi thì ma trận luôn có hạng một, định thức bằng không ở mọi điểm, và ta không còn phân biệt được cạnh thẳng với góc hay vân texture nữa. Toàn bộ nằm ở `loss_utils.py`, dòng 170 đến 230.

Ma trận hai nhân hai này đối xứng nên có hai trị riêng, tính bằng công thức đóng ở dòng thứ hai của slide. Trị riêng lớn hơn, lambda một, là năng lượng biến thiên theo hướng mạnh nhất; nghịch đảo căn bậc hai của nó cho ta một đại lượng mang đơn vị **pixel**, gọi là bước sóng nhỏ nhất w-min — hiểu nôm na là "chi tiết nhỏ nhất mà chỗ này còn nhìn thấy được". Đây chính là giới hạn tốc độ cho kích thước Gaussian: Gaussian nào chiếu lên màn hình mà dài hơn w-min thì đang làm mờ chi tiết, cần tách. Vector riêng thứ hai, vuông góc với vector riêng thứ nhất, là hướng mà cạnh chạy dọc theo — tức hướng duy nhất Gaussian được phép kéo dài mà không mất chi tiết. Hình quiver ở chương sách cho thấy rất rõ điều này quanh cung tròn: vector thứ hai luôn bám theo tiếp tuyến.

Nhưng một giá trị sigma duy nhất chỉ nhìn thấy một dải tần. Nên SADGS quét bốn octave, mỗi octave nhân sigma với một-phẩy-năm, và gán cho octave đó một tần số danh nghĩa f bằng một-phẩy-năm mũ trừ i. Ở mỗi octave, ta đo band response bằng hiệu hai ảnh mờ liên tiếp — tức Difference-of-Gaussian, đo xem làm mờ thêm thì mất bao nhiêu chi tiết — rồi luỹ thừa ba để làm trọng số. Công thức gộp ở dòng thứ ba là chỗ tinh tế nhất: mỗi octave bị **chia cho vết của chính nó** trước khi cộng vào, nên nó chỉ đóng góp **hướng**, còn **độ lớn** thì do f bình phương áp đặt. Hệ quả là vết của bản đồ cuối cùng không còn là năng lượng gradient nữa, mà chính là trung bình có trọng số của bình phương tần số trội. Thay số vào cấu hình mặc định, vết này nằm gọn trong khoảng từ không-phẩy-không-tám-tám đến một, nên w-min luôn rơi trong khoảng một đến bốn-phẩy-tám pixel. Đây là một ràng buộc thực tế đáng nhớ: ngay cả một bức tường trắng trơn cũng chỉ cho phép Gaussian dài tối đa cỡ năm pixel trên màn hình.

Cuối cùng, hai biến thể v1 và v2 khác nhau đúng một chỗ: v1 tính tensor một lần ở thang mịn nhất rồi chỉ làm mờ lại cho các thang sau, còn v2 tính lại tensor trên ảnh đã mờ ở từng thang. Hình dưới bên phải so trace của hai bản: chúng gần như trùng nhau ở vùng sọc thẳng, và chỉ lệch tại cung tròn cùng các ranh giới vùng — tức đúng những nơi hướng cấu trúc thay đổi theo thang. Mặc định huấn luyện chọn v1, bản rẻ hơn, dù chính docstring của tác giả gọi v2 là bản "đúng". Lý do hợp lý: bản đồ này chỉ dùng làm tín hiệu cổng cho quyết định tách hay cắt, không đi vào hàm mất mát. Và nó được tính **một lần duy nhất trước vòng huấn luyện** rồi cache theo tên ảnh, ở `train.py` dòng 160 đến 199.

**Chuyển tiếp:** Bản đồ cấu trúc đã sẵn sàng. Việc tiếp theo là biến nó thành một con số cho từng Gaussian ở từng góc nhìn — đó là đại lượng eta, và slide sau sẽ nói về cách chiếu ba trục của Gaussian xuống màn hình để tính nó.

---

#### Nếu bị hỏi

**Hỏi: Tính tensor cấu trúc cho toàn bộ ảnh huấn luyện thì có tốn không? Sao không tính lại mỗi vòng?**
Không tốn theo thời gian, nhưng tốn theo bộ nhớ. Nó chạy **một lần** trước vòng lặp, trong `torch.no_grad()`, gom camera theo độ phân giải rồi xử lý theo lô 100 ảnh (`train.py:168-190`). Vì ảnh GT không đổi trong suốt quá trình huấn luyện nên không có lý do tính lại. Cái phải trả là cache: mỗi view giữ thường trú một tensor `(1,3,H,W)` float32 trên GPU (`train.py:193-194`) — ở độ phân giải 1600×1200 là khoảng 12 MB mỗi view, nên 200 view tốn cỡ 4,6 GB. Đây là ràng buộc VRAM thực tế của SADGS, không phải chi tiết phụ.

**Hỏi: Nhóm chỉnh được những tham số nào của phần này?**
Thực tế chỉ hai: `st_levels` (mặc định 4) và `st_mode` (mặc định `"v1"`), khai báo ở `arguments/__init__.py:120-121`. Lý do là `train.py:188` và `:190` chỉ truyền đúng `levels`; `power_factor=3.0`, `octave_step=1.5`, `base_sigma=1.0` giữ nguyên mặc định trong chữ ký hàm, muốn đổi phải sửa code. Ngoài ra có hai tham số trông như chỉnh được nhưng không: `aggregation_mode` không được đọc ở bất kỳ dòng nào (tham số chết), còn `smoothing_factor` chỉ dùng làm **cờ bật/tắt** trong điều kiện `if i > 0 and smoothing_factor > 0` — độ mượt thật bị hard-code là `2*target_sigma` (`loss_utils.py:273-275`), nên đặt 0,1 hay 100 đều cho kết quả y hệt.

**Hỏi: Có hàm loss nào dùng tensor cấu trúc này không?**
Có code, nhưng **không chạy**. `lambda_freq` mặc định bằng 0 (`arguments/__init__.py:119`) và không được đọc ở đâu trong `SADGS/`. `frequency_loss_simple` (`loss_utils.py:389`) được `import` ở `freq_utils.py:5` nhưng không được gọi lần nào; `frequency_loss` (`loss_utils.py:80`) cũng không có người gọi; còn `estimate_required_gaussians` (`loss_utils.py:394`) thì gọi một hàm tên `get_multiscale_structure_tensor` **không tồn tại** ở dòng 411, sẽ ném `NameError` nếu ai đó thực sự chạy nó. Nói cách khác, `st_map` hiện chỉ đóng vai trò tín hiệu cổng cho densify/prune, hoàn toàn nằm ngoài đồ thị gradient.

**Hỏi: Vì sao mặc định lại chọn v1 nếu tác giả tự nhận v2 mới đúng?**
Vì đánh đổi có lợi. v2 phải gọi đầy đủ `get_structure_tensor_torch` ở từng octave (`loss_utils.py:357`), tức thêm ba lần Sobel trên ảnh RGB đầy đủ so với v1 chỉ làm mờ lại một tensor một kênh (`:282-284`). Đổi lại, hình so sánh cho thấy chênh lệch giữa hai bản **chỉ tập trung ở vùng cong và ranh giới vùng**, gần như bằng 0 ở vùng sọc thẳng. Vì kết quả chỉ dùng để quyết định tách hay không tách chứ không lan vào gradient, sai số hướng ở octave thô không tích luỹ — nên chọn bản rẻ là hợp lý.

---

## [S2] Tỉ số cấu trúc eta

### Slide: SADGS — Tỉ số cấu trúc $\eta$: Gaussian có "quá to" so với texture?

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này trả lời đúng một câu hỏi: một Gaussian có đang quá to so với chi tiết ảnh mà nó phải tái tạo hay không? Câu hỏi đó không có lời giải trong không gian ba chiều, vì cùng một quả Gaussian sẽ là thô khi camera đứng gần và là mịn khi camera lùi ra xa — nên SADGS đem cả hai vế về mặt phẳng ảnh, đơn vị pixel. Vế thứ nhất, bên trái slide: ta lấy ba trục chính của ellipsoid, xoay về camera space rồi chiếu qua Jacobian phối cảnh, được ba vector hai chiều; độ dài của chúng ở tâm ảnh rút gọn thành f nhân s chia z — tỉ lệ nghịch với khoảng cách. Vế thứ hai: từ structure tensor của ảnh gốc, ta lấy trị riêng lớn nhất lambda-một, là năng lượng tần số cực đại, rồi nghịch đảo căn của nó ra lambda-min — bước sóng nhỏ nhất mà texture ở chỗ đó chứa. Chia hai vế cho nhau ta được eta, một phân số không thứ nguyên: pixel chia pixel. Nhìn heatmap bên phải: eta lớn hơn một là Gaussian rộng hơn cả chu kỳ texture, nó sẽ xoá mất chi tiết, phải SPLIT; eta nhỏ hơn một phần mười là Gaussian nhỏ hơn chi tiết hàng chục lần, dư thừa, ứng viên PRUNE. Và vì eta đổi theo góc nhìn, SADGS không quyết định theo một view, mà đòi tám mươi phần trăm số view đồng thuận.

**Chuyển tiếp:** Có ba phiếu bầu high, mid, low như vậy cho mỗi view — slide sau sẽ xem chúng được cộng dồn và chốt thành quyết định split hay prune như thế nào.

---

#### Nếu bị hỏi

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

---

## [S3] Thống kê eta đa view

### Slide: SADGS — Tỉ lệ nhất quán đa view cho densify/prune

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide trước ta đã có đại lượng $\eta$ — mức vi phạm tần số của một Gaussian. Nhưng $\eta$ được đo trên mặt phẳng ảnh, nên nó phụ thuộc góc nhìn: cùng một Gaussian, ở view gần thì trục chiếu dài hàng chục pixel nên $\eta$ rất lớn, ở view xa thì co lại dưới một pixel nên $\eta$ gần như bằng không. Vì vậy SADGS không bao giờ quyết định trên một view. Cứ mỗi 10 iteration, hàm `update_freq_stats_online` trong `utils/freq_utils.py` dòng 181 chạy một lần, đo $\eta$ và tích luỹ bộ đếm cho từng Gaussian — nhưng chỉ với những mẫu qua được chuỗi ba bộ lọc: transmittance lớn hơn 0, opacity lớn hơn 0.05, và gradient view-space lớn hơn 2 nhân 10 mũ trừ 5. Tầng gradient là tầng quan trọng nhất: nó bảo đảm ta chỉ đếm ở những view mà hình học **đang thực sự sai**. Mỗi quan sát được phân loại theo trục xấu nhất trong ba trục: gọi là HIGH nếu $\eta$ vượt 1.0 — tức Gaussian dài hơn bước sóng kết cấu, chắc chắn gây răng cưa; gọi là LOW nếu $\eta$ dưới 0.1 — tức nhỏ hơn chi tiết ảnh mười lần, tức là dư thừa. Rồi cứ mỗi 100 iteration, `train.py` dòng 337 đến 353 mới lấy hai bộ đếm đó chia cho `accum_view_count` để ra `high_ratio` và `low_ratio`. Dùng tỉ lệ chứ không dùng số đếm thô, vì mỗi Gaussian được nhìn ở số view rất khác nhau — chia cho mẫu số sẽ chuẩn hoá tất cả về thang không đến một, nên một ngưỡng duy nhất dùng được cho cả đám mây điểm. Ngưỡng đó là 0.8, đọc ra tiếng Việt là "ít nhất tám mươi phần trăm số view phải đồng ý". Nhìn hình trên bên phải: phân bố `high_ratio` và `low_ratio` trên bốn nghìn Gaussian, và vạch đứt 0.8 chỉ cắt đúng phần đuôi — đại đa số Gaussian không bị đụng tới, đó chính là cơ chế giữ số điểm không bùng nổ. Hai chi tiết cuối cần nhấn. Thứ nhất, SPLIT còn phải kèm điều kiện gradient trung bình lớn hơn 10 mũ trừ 5; comment ngay trong `train.py` dòng 331 ghi nguyên văn "we must use this to prevent 7 million points". Thứ hai, kích thước Gaussian con được định hình bằng `max_eta_3ch`, tức cực đại qua các view, chứ không phải trung bình — vì con phải đủ nhỏ để không aliasing ở **mọi** view, chia một lần là xong. Và như hình dưới cho thấy, ngay sau mỗi lần densify, cả chín bộ đếm bị `zero_()` ở `train.py` dòng 377 đến 386, nên `accum_view_count` có dạng răng cưa — mẫu số luôn là một cửa sổ trượt khoảng mười mẫu, chứ không tích luỹ từ đầu huấn luyện.

**Chuyển tiếp:** Đã biết Gaussian nào cần chia và chia theo trục nào bị vi phạm, phần tiếp theo sẽ xem SADGS thực sự sinh Gaussian con ra sao — cơ chế anisotropic split.

---

#### Nếu bị hỏi

**H1: Tại sao lại là 0.8 mà không phải 0.5?**
Vì mẫu số rất nhỏ. Mỗi chu kỳ densify chỉ cho tối đa 100 chia 10, tức 10 mẫu cho một Gaussian (`train.py:275` và `densification_interval = 100` ở `arguments/__init__.py:87`), lại còn bị bộ lọc gradient cắt bớt. Với $M = 10$ và $p = 0.8$, nửa khoảng tin cậy 95% là $1.96\sqrt{0.8\cdot0.2/10} \approx 0.25$. Ngưỡng thấp như 0.5 sẽ cho rất nhiều split sai. Đặt cao ở 0.8, cộng thêm điều kiện gradient, là cách bù cho việc ước lượng còn nhiễu. Giá trị nằm ở `arguments/__init__.py:148-149`, cả `split_ratio_threshold` lẫn `prune_ratio_threshold` đều là 0.8.

**H2: Vì sao phải reset bộ đếm, giữ lại để có thống kê tốt hơn không được à?**
Không được, vì hai lý do. Một là sau khi split thì hình học đã đổi hẳn — Gaussian cha biến mất, con có scale mới, nên `high_ratio` cũ vô nghĩa và sẽ khiến chia tiếp ngay dù con đã đủ nhỏ. Hai là `max_eta_3ch` cập nhật bằng `torch.max` chạy dần (`freq_utils.py:391-392`), nó đơn điệu không giảm; không reset thì nó chỉ tăng và cuối cùng mọi Gaussian đều vượt ngưỡng 1.0. Ngoài ra, điểm mới sinh được nối bằng zeros (`gaussian_model.py:625-636`) nên tự động có `valid_mask` bằng False và không thể bị prune ngay ở chu kỳ kế tiếp.

**H3: `eta_mid_count` và `eta_high_sum_3ch` dùng vào việc gì?**
Nói thẳng là trong cấu hình mặc định hiện tại chúng không ảnh hưởng gì đến huấn luyện. `eta_high_sum_3ch` được dùng để tính `avg_high_eta_3ch` ở `train.py:348-350`, nhưng biến đó chỉ được truyền vào `expand_undersized_gs` — mà lời gọi này đang bị comment (`train.py:355-359`). Hàm chia nhận `max_eta_3ch` chứ không phải trung bình. Tương tự, `accum_eta` và `accum_weights_valid` được ghi mỗi chu kỳ nhưng không đọc ở nhánh quyết định; riêng `accum_weights_valid` hiện chỉ là bản sao của `accum_view_count` vì `weights_valid` được gán bằng `torch.ones_like` (`freq_utils.py:238`), tức trọng số transmittance đang bị vô hiệu hoá. Chúng là bộ đếm chẩn đoán, để sẵn cho các biến thể tương lai.

**H4: Việc lấy mẫu structure tensor có gì đặc biệt?**
Có — code không lấy mẫu tại tâm Gaussian mà tại một điểm ngẫu nhiên rút từ chính phân bố 2D của nó, bằng phân tích Cholesky của hiệp phương sai màn hình (`freq_utils.py:255-292`): $L_{11}=\sqrt{\sigma_{xx}}$, $L_{21}=\sigma_{xy}/L_{11}$, $L_{22}=\sqrt{\sigma_{yy}-L_{21}^2}$, rồi jitter bằng $L\varepsilon$ với $\varepsilon$ chuẩn. Lý do: $\eta$ phải phản ánh kết cấu trên toàn bộ vết loang của Gaussian, không phải một pixel tâm — một Gaussian dẹt vắt qua biên vật thể có tâm rơi vào vùng phẳng sẽ bị bỏ sót nếu chỉ lấy mẫu ở tâm. Cái giá là phương sai từng mẫu tăng, và đó chính là lý do thứ hai bắt buộc phải tích luỹ nhiều view trước khi quyết định.

---

## [S4] Split dị hướng

### Slide: SADGS — tách Gaussian dị hướng theo từng trục (`densify_and_split_structgs`)

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Đây là đóng góp cốt lõi của SADGS ở phía density control, và cũng là hàm dài nhất trong mã nguồn — gần hai trăm dòng, thay cho hai mươi bảy dòng của 3DGS gốc. Slide trước đã cho chúng ta đại lượng eta, tức năng lượng tần số vượt ngưỡng Nyquist, đo riêng cho từng trục của ellipsoid. Câu hỏi bây giờ là: đã biết Gaussian vi phạm, thì cắt như thế nào. Ba công thức bên trái trả lời trọn vẹn. Thứ nhất, số lát cắt trên mỗi trục bằng trần của căn bậc hai eta — suy ra thẳng từ lý thuyết lấy mẫu: muốn eta chia cho k bình phương nhỏ hơn hoặc bằng một, thì k phải lớn hơn hoặc bằng căn eta. Trục nào aliasing nhiều thì cắt nhiều; trục nào đã đủ mịn, eta nhỏ hơn một, thì k bằng một, giữ nguyên, không đụng tới. Thứ hai, tổng số con bằng k-x nhân k-y nhân k-z — thay đổi theo từng Gaussian, chứ không khóa cứng ở hai như 3DGS. Thứ ba, các con được đặt trên một lưới đều tất định trong hệ tọa độ local rồi xoay bằng ma trận R của quaternion, với bước lưới bằng căn mười hai nhân sigma mới — hệ số căn mười hai chính là để tập các con tái tạo đúng phương sai của cha. Nhìn hình bên phải: cùng một Gaussian dị hướng, 3DGS bên trái sinh đúng hai con ở vị trí lấy mẫu ngẫu nhiên và thu nhỏ cả hai trục theo cùng hệ số một phẩy sáu, nên con vẫn còn dài theo đúng cái trục đang vi phạm. SADGS bên phải cho k-x bằng bốn, k-y bằng hai, ra tám con trên lưới xác định, mỗi trục co đúng hệ số mà lý thuyết yêu cầu — một lần là xong, thay vì phải tách lại ba bốn vòng.

**Chuyển tiếp:** Tách xong thì Gaussian con được kế thừa gì từ cha, và cha bị xử lý ra sao — đó là nội dung slide tiếp theo.

---

#### Nếu bị hỏi

**Hỏi 1: Nếu eta rất lớn thì k có bị chặn trên không? Có nguy cơ bùng nổ bộ nhớ không?**
Câu trả lời thẳng là **không có trần**. Comment ở dòng 698 ghi là "clamp min=2 và max=8", nhưng code thật ở dòng 699–701 chỉ có `torch.clamp(ks, min=1)` — hoàn toàn không có giới hạn trên. Eta bằng 100 sẽ cho k bằng 10, và nếu cả ba trục cùng vi phạm thì một Gaussian sinh một nghìn con trong một lần gọi. Cái giữ cho nó không bùng nổ là ba tầng mặt nạ ở `gaussian_model.py:993-1011`: phải vừa có gradient tuyệt đối vượt `grad_abs_thresh` bằng 0.0002 hoặc được đánh dấu bởi tiêu chí multiview, vừa có scale lớn hơn `dense` nhân extent với `dense` bằng 0.001, vừa có importance score lớn hơn 0.5. Riêng tiêu chí multiview ở `train.py:345` yêu cầu hơn 80% số view đều báo eta cao (`split_ratio_threshold = 0.8`). Ba lớp lọc đó khiến tỉ lệ f trong công thức tăng trưởng n-mới bằng n-cũ nhân với (1 trừ f cộng f nhân N) rất nhỏ trên thực tế.

**Hỏi 2: Vì sao lại là hệ số căn mười hai trong khoảng cách giữa các con?**
Vì một phân bố đều trên đoạn dài L có độ lệch chuẩn bằng L chia căn mười hai. Muốn tập các Gaussian con — vốn là các tâm rời rạc cách đều nhau — tái tạo lại đúng phương sai của Gaussian cha theo trục đó, ta phải đặt bước lưới bằng căn mười hai nhân sigma mới. Đó là điều kiện bảo toàn moment bậc hai, cài ở dòng 775–776. Kèm theo, lưới được căn giữa bằng công thức i trừ (k trừ 1) chia hai ở dòng 763–765, nên trọng tâm tập con trùng khít tâm cha, không bị trôi vị trí. Đánh đổi là khoảng cách tâm–tâm bằng khoảng ba phẩy bốn sáu lần sigma con, lớn hơn bề rộng một sigma, nên ngay sau khi tách bề mặt có khe hở — optimizer sẽ hàn lại trong vài trăm bước tiếp theo.

**Hỏi 3: Tham số `scale_power` để làm gì, có nên chỉnh không?**
Nó là số mũ p trong công thức sigma con bằng sigma cha chia k mũ p, ở dòng 722, lấy từ `args.ks_scale_power`, mặc định 1.0 tại `arguments/__init__.py:138`. Khuyến nghị là **giữ nguyên 1.0**, vì dòng 722 dùng k mũ p để tính kích thước con, nhưng dòng 775 tính khoảng cách lại chỉ dùng sigma chia k, không có p. Hai công thức tách rời nhau: chỉ khi p bằng một thì chúng mới mô tả cùng một hình học. Đặt p bằng hai thì con teo thêm k lần nữa trong khi lưới giữ nguyên độ thưa, dẫn tới thủng bề mặt — hình `04_scale_power.png` trong chương sách cho thấy rõ ba panel p bằng 0.5, 1.0 và 2.0 có tâm con giống hệt nhau, chỉ khác kích thước.

**Hỏi 4 (dự phòng): SADGS có còn dùng split đẳng hướng của 3DGS nữa không?**
Hàm `densify_and_split` gốc vẫn còn nguyên trong file tại dòng 866–892 và được nhánh warmup dùng. Ngoài ra, ngay bên trong `densify_and_split_structgs` cũng có một nhánh dự phòng ở dòng 703–709: nếu `max_eta_3ch` được truyền vào là `None` thì code đặt k bằng hai trên trục dài nhất và k bằng một cho hai trục còn lại, tức N bằng hai — mô phỏng lại đúng 3DGS. Nhưng trong đường chạy thật, `train.py:373` luôn truyền `max_eta_3ch` xuống, nên nhánh đó chỉ là lưới an toàn, không bao giờ chạy.

---

## [S5-S6] Clone & expand (đang tắt)

### Slide: Clone trong SADGS và cách ghép tensor vào optimizer

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

#### Nếu bị hỏi

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

### Slide: Giãn giải tích Gaussian dưới cỡ (`expand_undersized_gs`) — và trạng thái *bị tắt*

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

#### Nếu bị hỏi

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

---

## [S7-S8] Densify+prune & final prune

### Slide SADGSX-07: `densify_and_prune_structgs` — một vòng sinh + xoá hợp nhất

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

### Slide SADGSX-08: Tỉa cuối `final_prune_structgs` — công thức có, lời gọi đang tắt

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

---

## [S9-S10] Optimizer/LR/opacity reset & bộ lọc 3D

### Slide `part45_sadgsx_09`: Optimizer, lịch learning rate & reset opacity trong SADGS

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này trả lời câu hỏi: SADGS điều khiển quá trình tối ưu khác 3DGS gốc ở chỗ nào. Có bốn điểm. Thứ nhất, lịch learning rate cho vị trí vẫn là nội suy log-tuyến tính từ 1,6 nhân 10 mũ trừ 4 xuống 1,6 nhân 10 mũ trừ 6 qua 30 nghìn bước — nhưng khi đọc `training_setup` dòng 325 đến 328, nhóm em phát hiện code chỉ truyền `lr_delay_mult` mà quên truyền `lr_delay_steps`, nên tham số này giữ mặc định bằng 0 và **nhánh warm-up không bao giờ chạy**. Thứ hai, năm nhóm tham số còn lại giữ learning rate hằng số suốt 30 nghìn bước, trong đó opacity, scaling và rotation đều được **nhân đôi** so với 3DGS — chính comment trong source cũng ghi lại giá trị cũ. Thứ ba, optimizer mặc định là chế độ `hybrid`: Adam kèm `amsgrad` cho hình học, còn Spherical Harmonics bậc cao — nhóm tham số nặng nhất với 45 hệ số mỗi Gaussian — dùng `SparseGaussianAdam`, chỉ cập nhật những Gaussian đang hiển thị. Thứ tư, và đây là điểm thú vị nhất: reset opacity. Nhìn vào hình bên phải, mỗi 3000 iteration opacity bị kéo tụt đúng 10 lần rồi hồi phục dần, tạo ra mẫu răng cưa này. Điều đáng nói là trong code, hệ số bù của bộ lọc 3D được nhân vào rồi chia ngược ra nên **triệt tiêu hoàn toàn** — kết quả cuối cùng chỉ là phép nhân opacity với 0,1. Khác hẳn 3DGS gốc vốn kẹp về hằng số 0,01, SADGS **giữ nguyên thứ tự** độ đục giữa các Gaussian, nên phép prune ở ngưỡng 0,1 chọn lọc chính xác hơn nhiều.

**Chuyển tiếp:** Đường đứt nằm ngang trong hình chính là ngưỡng prune 0,1 — và slide tiếp theo sẽ cho thấy còn một cơ chế thứ hai cũng đẩy Gaussian xuống dưới ngưỡng đó, lần này xuất phát từ lý thuyết lấy mẫu chứ không phải từ lịch huấn luyện.

**Nếu bị hỏi:**

- *"Tại sao lại tăng gấp đôi learning rate của opacity, scaling, rotation?"* — Vì SADGS reset opacity theo phép **nhân** 0,1 mỗi 3000 bước. Opacity phải kịp hồi phục trong 3000 bước, nếu không thì sau n chu kỳ nó chỉ còn 0,1 mũ n nhân giá trị ban đầu — hai chu kỳ đã giảm 100 lần. `opacity_lr = 0.05` so với 0,025 của 3DGS (`arguments/__init__.py:76`) chính là để bù cho điều đó. Scaling và rotation tăng theo để hình học hội tụ sớm, cho tín hiệu $\eta$ của densify rõ ràng từ giai đoạn đầu.

- *"Hàm `optimizer_step` giãn tần suất cập nhật 1 / 32 / 64 có thực sự chạy không?"* — Không, ở cấu hình mặc định. Hàm đó nằm ở `gaussian_model.py:357-378`, nhưng `train.py:422-424` chỉ gọi nó khi `optimizer_type == "default"`. Mặc định là `"hybrid"` (`arguments:128`), và nhánh hybrid ở `train.py:432-437` gọi `.step()` mỗi iteration bình thường.

- *"Bậc SH tăng dần như 3DGS chứ?"* — Không. `train.py:216-217` gọi `oneupSHdegree()` với điều kiện `iteration % 1 == 0`, tức **mỗi iteration**, nên bậc SH đạt tối đa 3 ngay tại iteration thứ 3, thay vì tại 3000 như 3DGS gốc. Cơ chế warm-up SH thực chất đã bị vô hiệu hoá; thay vào đó SADGS kiềm chế bằng learning rate rất nhỏ cho `f_rest`, chỉ 2,5 nhân 10 mũ trừ 4.

---

### Slide `part45_sadgsx_10`: Bộ lọc 3D chống alias — chặn dưới kích thước Gaussian theo Nyquist

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Vấn đề đặt ra rất cụ thể: một Gaussian 3D có thể co nhỏ tuỳ ý, và khi camera lùi xa, hình chiếu của nó xuống ảnh nhỏ hơn một pixel — vượt giới hạn Nyquist của lưới pixel, gây nhấp nháy và moiré. Lời giải theo lý thuyết lấy mẫu là tiền lọc, và may mắn là tích chập Gaussian với Gaussian vẫn là Gaussian nên có thể lọc giải tích, không cần mipmap. SADGS làm ba bước. Bước một, tính ngưỡng lọc: `filter_3D` bằng độ sâu z chia tiêu cự f nhân căn 0,2 — trong đó z là độ sâu tới camera **gần nhất** nhìn thấy Gaussian, còn f là tiêu cự **lớn nhất**, tức camera nét nhất. Ý nghĩa: z chia f chính là kích thước thế giới của một pixel tại độ sâu đó, nên `filter_3D` xấp xỉ 0,447 lần một pixel quy về 3D — đúng là giới hạn Nyquist chiếu ngược vào không gian. Bước hai, scale hiệu dụng bằng căn của s bình phương cộng filter bình phương, tức cộng phương sai vào mỗi trục: Gaussian nhỏ bị **kéo lên sàn** filter, Gaussian lớn không bị đụng chạm. Bước ba là điểm mà nhiều người bỏ sót: nếu chỉ phình scale mà giữ nguyên opacity thì tổng năng lượng tăng lên, vật thể xa sẽ sáng và đậm hơn thực tế. Nên phải nhân thêm hệ số `coef` bằng căn của tỉ số hai định thức, để bảo toàn tích phân. Hình bên phải chính là hệ số đó: với r bằng filter chia s, ở trường hợp đẳng hướng ta có coef bằng một cộng r bình phương, mũ trừ ba phần hai. Đọc hình: khi r bằng 1, coef còn khoảng 0,354; khi r bằng 2, chỉ còn 0,089. Gaussian nhỏ hơn pixel sẽ **mờ hẳn đi** thay vì nhấp nháy. Và đây chính là chỗ bộ lọc khớp vào cơ chế densify theo $\eta$: chia theo $\eta$ làm s nhỏ dần, nhưng khi s tụt xuống dưới filter thì opacity hiệu dụng sụp đổ và Gaussian bị prune — bộ lọc 3D đóng vai trò **phanh vật lý** cho densification.

**Chuyển tiếp:** Cần nói rõ một điều: cờ `--compute_3d_filter` mặc định là `False`, nên đây là cơ chế tuỳ chọn, không nằm trong đường chạy mặc định của SADGS — phần tiếp theo sẽ quay lại các cơ chế luôn hoạt động.

**Nếu bị hỏi:**

- *"Vì sao lấy camera gần nhất chứ không phải xa nhất?"* — Vì lấy `distance` nhỏ nhất là chọn ràng buộc **chặt nhất** qua mọi view: nếu có view nào nhìn rất gần, Gaussian được phép nhỏ tương ứng, và ở các view xa hơn thì khâu chiếu 2D tự lọc thêm. Ngược lại `focal_length` lấy **lớn nhất** (`gaussian_model.py:247-248`) để chuẩn theo camera phân giải cao nhất, tránh làm mờ quá mức ảnh nét nhất trong tập train. Gaussian không camera nào thấy thì nhận giá trị lớn nhất trong nhóm nhìn thấy (`:250`) — tức chịu mức lọc mạnh nhất, an toàn.

- *"Hệ số căn 0,2 ở đâu ra?"* — Đó là hằng số hard-code ở `gaussian_model.py:254`, ngay phía trên có hai dòng TODO của chính tác giả: `#TODO remove hard coded value` và `#TODO box to gaussian transform`. Về ý nghĩa, nó quy đổi hộp lấy mẫu rộng một pixel thành độ lệch chuẩn Gaussian tương đương. Đây là điểm thừa kế từ Mip-Splatting chứ không phải đóng góp riêng của SADGS.

- *"Bộ lọc có ảnh hưởng tới quyết định densify không?"* — Không, và đây là điểm dễ hiểu nhầm. Điều kiện split và clone dùng `get_scaling` **thô** (`gaussian_model.py:683` và `:896-897`), không dùng `get_scaling_with_3D_filter`. Bộ lọc chỉ được tiêu thụ ở khâu render — `opacity` tại `gaussian_renderer/__init__.py:63` và `scales` tại `:74`. Vòng phản hồi giữa hai cơ chế là **gián tiếp**, đi qua opacity rồi tới prune ở ngưỡng 0,1, chứ không đi qua tiêu chí densify.

- *"Trong pha densify, `filter_3D` có được cập nhật theo không?"* — Không. `train.py:405-409` chỉ chạy lại `compute_3D_filter` khi iteration chia hết 100 **và** lớn hơn `densify_until_iter` bằng 15 nghìn **và** còn cách cuối hơn 100 bước. Nghĩa là suốt toàn bộ pha densification, Gaussian con kế thừa nguyên `filter_3D` của cha (`:728, :886, :907`) dù bản thân nó đã nhỏ hơn — tỉ số r tăng, coef giảm, nên con mới sinh ra đã bị mờ sẵn.

---

## [S11-S12] Loss & lấy mẫu

### Slide — Hàm mất mát của SADGS: $L_1$ + SSIM + $L_2$ (và hai loss chưa dùng)

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Sang phần hàm mất mát. Em muốn nói thẳng một điều trước: toàn bộ tín hiệu gradient của SADGS đi qua đúng một dòng code, dòng 259 của train.py, và nó là loss photometric quen thuộc — 0,8 nhân L1, cộng 0,2 nhân một trừ SSIM, cộng 2,0 nhân L2. Ba hệ số này lấy từ arguments dòng 86 và 117, không phải em tự đặt. So với 3DGS gốc, khác biệt duy nhất ở tầng photometric là số hạng L2 với trọng số 2,0. Và trọng số 2,0 nghe thì lớn nhưng thực ra chỉ để kéo L2 về cùng bậc độ lớn với L1 thôi — vì khi sai số nhỏ hơn 1 thì bình phương luôn nhỏ hơn trị tuyệt đối. SSIM ở đây dùng kernel CUDA fused_ssim chứ không phải hàm ssim thuần PyTorch trong loss_utils. Bây giờ đến phần quan trọng: trong file loss_utils có **hai hàm loss mang tinh thần structure-aware rất rõ nhưng chưa bao giờ được gọi**. Thứ nhất là tone_curve_loss, dòng 28: nó chia sai số cho chính độ sáng dự đoán, có stop-gradient ở mẫu số, nên hệ thống học trên **sai số tương đối** thay vì tuyệt đối — hệ quả là vùng tối được khuếch đại gradient mạnh, đúng tinh thần Weber-Fechner. Thứ hai là frequency_loss, dòng 80: nó không tính trên pixel mà tính **trên từng Gaussian** — lấy mẫu structure tensor tại tâm Gaussian, suy ra bước sóng texture cho phép, rồi phạt ReLU một phía mỗi khi bán kính Gaussian vượt quá bước sóng đó. Hình bên phải minh hoạ đúng cơ chế này trên một ảnh chirp: bước sóng cho phép co lại dần khi texture rậm hơn, và vùng tô đỏ chính là nơi Gaussian bị phạt. Nhưng cả hai hàm này đều có trọng số bằng 0 và **không được tham chiếu ở bất kỳ đâu** — nghĩa là đặt tham số qua dòng lệnh cũng vô hiệu, vì dòng nối chưa tồn tại. Kết luận cần nhớ: loss của SADGS vẫn là loss 3DGS cộng một số hạng L2 nặng; toàn bộ tính structure-aware nằm ở nhánh densification, qua thống kê eta không gradient, chứ không nằm trong hàm mất mát.

**Chuyển tiếp:** Đã rõ tần số không đi vào loss, vậy nó đi vào đâu — câu trả lời là qua thống kê tích luỹ theo từng view, và điều đó đặt ra câu hỏi: chọn view nào để tích luỹ? Đó là nội dung slide tiếp theo.

#### Nếu bị hỏi

**H: Tại sao nhóm lại thêm L2 với trọng số tận 2,0? Có làm ảnh mờ đi không?**
Đ: Vì L2 nhỏ hơn L1 hẳn một bậc khi sai số dưới 1 — ví dụ sai số 0,1 thì L1 bằng 0,1 còn L2 chỉ 0,01. Hệ số 2,0 chỉ đưa hai số hạng về cùng bậc, không phải ưu tiên L2. Tác dụng thực là khuếch đại gradient ở các pixel sai nhiều, giúp hội tụ nhanh giai đoạn đầu. Còn nguy cơ mờ thì được số hạng D-SSIM giữ lại — nó phạt rất nặng việc mất tương quan cục bộ, là thứ L1 và L2 đều bỏ qua.

**H: Làm sao khẳng định tone_curve_loss và frequency_loss chưa dùng, chứ không phải nhóm đọc sót?**
Đ: Ba bằng chứng độc lập. Một, train.py dòng 17 chỉ import l1_loss, l2_loss và hai hàm structure tensor — hai hàm kia không có trong danh sách import. Hai, lambda_tone và lambda_freq đều bằng 0 ở arguments dòng 118 và 119. Ba, grep toàn repo không tìm thấy lời gọi nào. Ngoài ra còn một bằng chứng gián tiếp: hàm estimate_required_gaussians ở dòng 394 gọi get_multiscale_structure_tensor tại dòng 411, mà tên đó không tồn tại trong file — chỉ có hậu tố _v1 và _v2. Nếu hàm đó từng chạy, nó đã ném NameError rồi.

**H: Vậy ý tưởng "so cỡ Gaussian với bước sóng texture" có được dùng không?**
Đ: Có, nhưng không ở dạng loss. Nó chạy ở freq_utils dòng 181, hàm update_freq_stats_online, dưới dạng thống kê eta bằng độ dài trục chiếu chia cho bước sóng cực tiểu, tính riêng cho cả ba trục, và chạy trong torch.no_grad. Khác biệt cốt lõi: frequency_loss là đẳng hướng và khả vi, còn eta là dị hướng ba trục và không khả vi. SADGS chọn đường thứ hai — điều khiển tần số qua densification chứ không qua gradient.

---

### Slide — Lấy mẫu camera bằng FPS & lấy mẫu Gaussian dị hướng 2D

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này nói về hai bộ lấy mẫu trong SADGS. Thứ nhất là sampling_cameras ở freq_utils dòng 12, mặc định chế độ fps với 60 camera. Nó lấy vị trí camera bằng công thức trừ R chuyển vị nhân t — tức giải ngược ma trận view ra toạ độ thế giới — rồi chạy vòng lặp tham lam kinh điển: mỗi bước cập nhật khoảng cách tới điểm gần nhất đã chọn, rồi chọn camera **xa nhất** so với mọi camera đã chọn. Đây chính là thuật toán xấp xỉ 2 của bài toán k-center. Hình bên phải cho thấy hiệu quả: em cố tình dồn 70 phần trăm camera vào một phần ba quỹ đạo, vậy mà tập được chọn vẫn trải gần như đều quanh vòng tròn — vì quy tắc arg-max luôn nhảy sang vùng đang trống nhất, không quan tâm mật độ gốc. Điều này quan trọng với SADGS vì thống kê eta tích luỹ theo từng view: nếu camera dồn về một phía, các Gaussian phía đối diện gần như không bao giờ được cập nhật, và ước lượng vi phạm tần số sẽ thiên lệch. Có hai điểm em phải nói thẳng. Một, metric chỉ dùng **vị trí** camera, hoàn toàn không dùng hướng nhìn — hai camera đứng cùng chỗ nhìn ngược nhau có khoảng cách bằng 0. Hai, chi phí là N nhân num_cams, nên phải lấy tập con mới rẻ; nhưng ở train.py dòng 224 và 240, hàm lại được gọi với num_cams bằng đúng số camera trong stack, cộng với việc code sắp lại kết quả theo chỉ số gốc tăng dần, nên FPS ở đó **suy biến thành phép tắt xáo trộn** kèm chi phí N bình phương — và mặc định camera_sampling còn là "random" nên nhánh đó thậm chí không chạy. Bộ lấy mẫu thứ hai là sample_anisotropic_gaussians_2d: đặt Gaussian theo xác suất tỉ lệ với vết của structure tensor, hình dạng lấy từ hai trị riêng, trục dài nằm dọc vector riêng ứng với trị riêng nhỏ — tức bám theo cạnh. Ý tưởng rất đẹp, nhưng sự thật là hàm này không được gọi ở đâu cả; khởi tạo point cloud thật vẫn là create_from_pcd từ COLMAP, ở scene dòng 83.

**Chuyển tiếp:** Như vậy cả tầng loss lẫn tầng lấy mẫu đều cho thấy cùng một điều — hạ tầng đã có, nhưng cơ chế thực sự tạo ra chất lượng của SADGS nằm ở nhánh densification; phần tiếp theo sẽ đi vào optimizer và lịch học.

#### Nếu bị hỏi

**H: Nếu FPS ở train.py bị suy biến thì tại sao còn giữ nó?**
Đ: Vì nó vẫn đúng thuật toán, chỉ sai ở tham số gọi. Sửa num_cams từ len(stack) xuống một tập con, ví dụ 60 như mặc định của chữ ký hàm, là lấy lại được lợi ích ngay. Trên thực nghiệm mô phỏng 220 camera, bán kính phủ của FPS luôn thấp hơn trung bình random, và chênh nhiều nhất đúng ở vùng num_cams nhỏ. Random còn có phương sai lớn, nghĩa là một lần rút xấu có thể bỏ trống cả một phía cảnh.

**H: Chi phí FPS có đáng lo không khi dataset lớn?**
Đ: Chi phí là O của N nhân num_cams so với O của N cho random. Với N bằng 1000, lấy k bằng 60 rẻ hơn lấy k bằng N khoảng 17 lần. Nhưng lưu ý FPS chỉ chạy khi stack cạn, không phải mỗi iteration, nên ngay cả trường hợp N bình phương thì chi phí phân bổ trên cả một epoch cũng không phải nút cổ chai — vấn đề chính là nó không mang lại lợi ích gì khi num_cams bằng N.

**H: Vậy lấy mẫu Gaussian dị hướng 2D để làm gì nếu không dùng?**
Đ: Docstring nói nó được tách ra từ fit_2d_image.py để tái sử dụng — mà file đó không còn trong repo. Vai trò đúng của nó là tiện ích nghiên cứu cho bài toán khớp ảnh 2D, đồng thời là bản mẫu cho ý tưởng cốt lõi của SADGS: dùng structure tensor để quyết định kích thước và hướng Gaussian. Con số thuyết phục là tỉ lệ mẫu rơi vào vùng phẳng vô ích: lấy đều cho khoảng 25 phần trăm, lấy theo structure tensor chỉ 0,3 phần trăm. Ý tưởng đó đã được hiện thực hoá trong huấn luyện 3D qua eta của update_freq_stats_online, chứ không qua hàm này.

---

## [S13-S14] Siêu tham số & rasterizer riêng

### Slide: Siêu tham số & lịch trình huấn luyện của SADGS
*(part45_sadgsx_13.tex)*

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Slide này trả lời câu hỏi rất thực tế: muốn chạy SADGS thì phải vặn những nút nào. SADGS kế thừa toàn bộ tham số của 3DGS rồi mở rộng lên hơn năm mươi cờ, nhưng chỉ khoảng một nửa thực sự được nối vào code. Bảng bên trái liệt kê những cờ quan trọng nhất kèm giá trị mặc định thật, lấy từ file `arguments/__init__.py`: `mult` bằng 0.7, `eta_compute_mode` là "wavelength", hai ngưỡng đồng thuận đa view `split_ratio_threshold` và `prune_ratio_threshold` cùng bằng 0.8, `dense` bằng 0.001 — tức nhỏ hơn `percent_dense` của 3DGS gốc mười lần, `opacity_reset_decay` bằng 0.1, và optimizer mặc định là "hybrid". Nhưng điểm em muốn nhấn mạnh nhất nằm ngay dưới bảng: hai ngưỡng quyết định nhất của cả phương pháp — tau-high bằng 1.0 và tau-low bằng 0.1 — lại **không** nằm trong file tham số mà bị hard-code trong `freq_utils.py` dòng 395 đến 396. Muốn đổi chúng phải sửa mã nguồn, không có cờ dòng lệnh nào cả. Hình bên phải là sơ đồ Gantt dựng từ đúng các điều kiện `if` trong `train.py`: bậc SH được nâng mỗi iteration nên chạm bậc ba ngay ở iteration thứ ba; thống kê tần số eta cập nhật mỗi mười iteration và chỉ chạy đến mốc mười lăm nghìn; densify mỗi trăm iteration rồi lập tức xoá sạch bộ đếm eta — nghĩa là cái gọi là "đồng thuận đa view" thực chất chỉ dựa trên khoảng mười quan sát; reset opacity mỗi ba nghìn iteration bằng cách nhân alpha với 0.1; prune thô tại bốn nghìn và tám nghìn; còn mười lăm nghìn iteration cuối thì chỉ tinh chỉnh tham số chứ không thêm bớt Gaussian nữa. Cuối cùng, xin lưu ý ba nhánh đã bị comment-out trong code: `expand_undersized_gs` khiến cờ `tau_expand` hiện vô hiệu, prune trong nhánh warmup, và `final_prune_structgs` ở cuối huấn luyện.

**Chuyển tiếp:** Một trong những cờ vừa nêu — `mult` — không chạy trong Python mà được truyền thẳng xuống CUDA, nên slide tiếp theo ta sẽ mở nắp capô xem rasterizer riêng của SADGS đã sửa những gì.

#### Nếu bị hỏi

**Hỏi: Cấu hình mặc định trong `arguments` có phải là cấu hình dùng để báo cáo kết quả không?**
Không ạ. `run_train.sh` mới là cấu hình thực nghiệm thật, và nó khác khá nhiều: số iteration chỉ 3000 cho Mip-NeRF 360 và Deep Blending, 7000 cho Tanks and Temples, chứ không phải 30000; `mult` dùng 0.25 ở ba trên bốn cấu hình; `freq_transmittance_threshold` đặt 0.8 thay vì 0.0; và cả bốn cấu hình đều bật `--warmup_densification` dù mặc định trong code là `False`.

**Hỏi: Những cờ nào chỉnh cũng vô ích?**
Em đã quét toàn bộ file `.py` ngoài thư mục `arguments` và tìm được mười tám cờ có đúng không lần sử dụng: `clone_target_eta`, `lambda_freq`, `lambda_tone`, `min_weight`, `prune_from_iter`, `prune_interval`, `densify_prune_ratio`, `after_densify_prune_ratio`, `loss_thresh`, `sample_far_plane`, `far_plane_dist`, `far_plane_res`, `densification_window_width`, `max_clones_per_axis`, `expansion_speed`, `adaptive_clone`, `feature_lr`, `shfeature_lr`. Ngoài ra `prune_until_iter` bị khai báo trùng hai lần ở dòng 96 và 100, giá trị 25000 bị ghi đè thành 30000, và cũng không dùng ở đâu.

**Hỏi: Tham số nào SADGS đổi so với 3DGS gốc mà đáng chú ý nhất?**
Ba cái. Một, learning rate hình học đều cao gấp đôi: `opacity_lr` 0.05 so với 0.025, `scaling_lr` 0.01 so với 0.005, `rotation_lr` 0.002 so với 0.001. Hai, reset opacity là phép **nhân** alpha với 0.1 chứ không phải kẹp về 0.01 như 3DGS — nhẹ tay hơn nhiều. Ba, optimizer "hybrid" tách làm đôi: Adam có amsgrad cho hình học, còn `SparseGaussianAdam` fused CUDA riêng cho hệ số SH bậc cao, chỉ cập nhật Gaussian đang nhìn thấy được.

---

### Slide: Rasterizer riêng của SADGS — `mult`, `cov2D` và $T^{\max}$
*(part45_sadgsx_14.tex)*

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

SADGS không dùng được rasterizer CUDA gốc của 3DGS mà phải fork thành một submodule riêng tên `diff-gaussian-rasterization_structgs`, gọi qua hàm `render_structgs`. Lý do rất cụ thể: cơ chế densify theo tần số cần hai đại lượng mà bản gốc không hề xuất ra. Phần nền tảng thì không đổi — vẫn chia ảnh thành tile mười sáu nhân mười sáu, vẫn sort theo cặp tile-và-độ-sâu, vẫn alpha blending với alpha bằng opacity nhân hàm mũ, transmittance là tích của một trừ alpha các lớp phía trước, bỏ qua đóng góp nhỏ hơn một phần hai trăm năm mươi lăm và dừng sớm khi T xuống dưới mười mũ trừ bốn. Cái mới thứ nhất là `mult`: thay vì gán tile theo hộp bao vuông ba sigma, SADGS xác định chính xác vùng ellipse mà d chuyển vị nhân nghịch đảo sigma hai D nhân d nhỏ hơn hoặc bằng t, với t bằng `mult` nhân hai lần log của hai trăm năm mươi lăm nhân opacity — đúng hai dòng code trong `auxiliary.h`. Hình bên phải cho thấy ellipse nghiêng nằm gọn hẳn trong hình vuông ba sigma, và số tile phải xử lý mỗi splat giảm rõ rệt khi `mult` nhỏ đi. Điều quan trọng là `mult` **không** đụng vào công thức alpha — nó chỉ quyết định Gaussian nào được đưa vào danh sách tile nào, nên giảm `mult` là giảm khối lượng công việc chứ không làm sai blending; tại `mult` bằng một, phần bị cắt có alpha đúng bằng ngưỡng một phần hai trăm năm mươi lăm mà nhân render vốn đã bỏ qua. Cái mới thứ hai là tensor `cov2D` kích thước N nhân bảy: ba cột đầu là ma trận hiệp phương sai hai chiều, cột ba bốn là tâm theo pixel, cột năm là độ sâu, và cột sáu là T-max — transmittance lớn nhất mà Gaussian đó đạt được trên toàn ảnh, ghi bằng thủ thuật `atomicMax` trên bit-pattern của số thực, hợp lệ vì T luôn không âm. Python lấy tensor này ra, lọc những Gaussian có T-max vượt ngưỡng và opacity đủ lớn, rồi mới lấy mẫu Cholesky từ hiệp phương sai để tính eta. Nói ngắn gọn, SADGS đã biến rasterizer thành một cảm biến: nó vừa vẽ ảnh vừa đo đạc trạng thái từng Gaussian trong cùng một lượt forward.

**Chuyển tiếp:** Như vậy ta đã có đủ cả tham số lẫn hạ tầng CUDA; phần tiếp theo sẽ ghép chúng lại để xem kết quả thực nghiệm SADGS đạt được.

#### Nếu bị hỏi

**Hỏi: `cov2D` có tham gia vào lan truyền ngược không?**
Không. Nó được `save_for_backward` ở dòng 112 của `diff_gaussian_rasterization_structgs/__init__.py`, nhưng chữ ký hàm `backward` ở dòng 116 nhận nó dưới tên `_cov2D` và không dùng — gradient bị bỏ qua hoàn toàn. Đây là kênh thống kê một chiều, GPU đo rồi trả về Python, không can thiệp vào tối ưu.

**Hỏi: Đặt `mult` nhỏ có làm giảm chất lượng ảnh không?**
Về lý thuyết alpha bị cắt tại biên hộp bằng opacity nhân với, hai trăm năm mươi lăm nhân opacity, mũ trừ `mult`. Tại `mult` bằng 1 thì đúng bằng một phần hai trăm năm mươi lăm — không mất gì. Tại 0.7, sai số chỉ gấp vài lần ngưỡng đó, vẫn dưới một mức lượng tử tám bit, trong khi chi phí giảm khoảng hai mươi đến hai mươi lăm phần trăm. Nhưng `run_train.sh` dùng `mult` bằng 0.25 — lúc đó sai số cắt đã lên hai bậc log, nên cấu hình đó chỉ hợp lý khi đi kèm số iteration ngắn và ngân sách Gaussian bị siết.

**Hỏi: `metric_map` và `compute_extra` dùng để làm gì, có chạy trong lúc train không?**
`metric_map` là mask nhị phân người dùng đưa vào; khi bật `get_flag`, nhân render sẽ `atomicAdd` đếm số lần mỗi Gaussian đóng góp vào vùng mask đó — `forward.cu` dòng 461 đến 467. `compute_extra` thì xuất thêm depth map, opacity map và normal map trong cùng lượt render — `rasterize_points.cu` dòng 116 đến 129. Cả hai đều **không** được bật trong `train.py`: lời gọi `render_structgs` ở dòng 250 không truyền `get_flag`, nên nó về `False`; còn `compute_extra` chỉ được dùng ở `render.py` dòng 41 qua cờ `--render_extra`. Đây là hạ tầng đã có sẵn nhưng chưa dùng trong vòng lặp huấn luyện.

**Hỏi: Vì sao rasterizer trả về tới mười ba tensor?**
Ngoài bảy tensor xuất ra Python (`color`, `radii`, `accum_metric_counts`, `cov2D`, và ba bản đồ phụ), còn bốn buffer nội bộ cùng `num_rendered` và `num_buckets`. `num_buckets` và `sampleBuffer` là do SADGS lưu mẫu transmittance mỗi ba mươi hai Gaussian vào `sampled_T` — `forward.cu` dòng 412 đến 420 — để backward chạy theo bucket song song thay vì duyệt ngược tuần tự toàn tile.

---

## [S15] Tổng hợp đầu-cuối

### Slide: Tổng hợp đầu-cuối — pipeline SADGS, chi phí, và phần chưa bật

### Nhịp 1 — Sơ đồ pipeline (cột trái)
**Thời lượng: ~45 giây**

Đến đây em đã trình bày từng cơ chế riêng lẻ, nên slide cuối này ghép tất cả lại thành một sơ đồ chạy được. Hình bên trái là mười ba bước của **một** iteration SADGS, đúng thứ tự thực thi trong `train.py` từ dòng 202 đến 440. Màu xanh lá là phần lõi kế thừa từ 3DGS và FastGS: cập nhật learning rate, chọn camera, render, tính loss, backward, rồi bước optimizer. Màu cam là tầng tần số riêng của SADGS. Màu hồng là khối densify và prune. Còn dải màu xanh nhạt trên cùng là phần tiền xử lý chạy **một lần duy nhất** ngoài vòng lặp — cache structure tensor cho toàn bộ ảnh train, ở dòng 160 đến 200. Khối xám nét đứt bên phải là hai nhánh mặc định TẮT: `compute_3D_filter` và `sample_bbox_faces`. Một chi tiết nhỏ mà em muốn các thầy cô để ý: dòng 216 viết là `if iteration % 1 == 0` — điều kiện luôn đúng, nên bậc SH tăng **mỗi vòng**, chạm trần bậc ba chỉ sau ba iteration, thay vì sau ba nghìn vòng như 3DGS gốc.

**Chuyển tiếp:** Vấn đề là mười ba bước này không chạy với cùng một nhịp — và chính điều đó quyết định chi phí.

---

### Nhịp 2 — Mô hình chi phí và khác biệt cốt lõi (cột phải, phần trên)
**Thời lượng: ~45 giây**

Chi phí trung bình một vòng có năm số hạng: $a$ nhân $N$ cho forward-backward-Adam, $b$ nhân $P$ cho rasterize theo số pixel, $c$ là overhead cố định, rồi hai số hạng riêng của SADGS — $a_\eta N$ chia 10 cho phần thống kê tần số, và $a_D N$ chia 100 cho densify và prune. Hai mẫu số 10 và 100 không phải em tự đặt: số 10 đến từ điều kiện `iteration % 10 == 0` ở dòng 275, số 100 là `densification_interval` mặc định trong `arguments`. Nghĩa là hai cơ chế đắt nhất về mặt thuật toán lại chỉ chiếm phần nhỏ trong tổng chi phí, vì chúng được chia cho chu kỳ kích hoạt. Và vì **bốn trên năm** số hạng đều tuyến tính theo $N$, kết luận thực hành là: cắt được ba mươi phần trăm số Gaussian thì cắt được xấp xỉ ba mươi phần trăm chi phí, trên **mọi** iteration còn lại. Đó là lý do prune đa view — một dòng code so `low_ratio` với 0.8 ở dòng 353 — là đòn bẩy mạnh nhất, mạnh hơn nhiều so với tối ưu vi mô một CUDA kernel, vì nó tác động lên **biến** $N$ chứ không phải lên **hệ số** $a$.

So với FastGS, khác biệt cốt lõi gói trong ba điểm: tín hiệu densify là phép **AND** của gradient với `high_ratio` lớn hơn 0.8; split là **dị hướng**, mỗi trục chia $k$ bằng trần của căn $\eta$; và prune có thêm tiêu chí đa view. Nền rasterization — compact box `mult` bằng 0.7, Fused và Sparse Adam — giữ nguyên không đổi.

**Chuyển tiếp:** Nhưng nếu chỉ dừng ở đây thì bài trình bày sẽ không trung thực. Em muốn nói rõ phần còn lại.

---

### Nhịp 3 — Phần đã cài nhưng chưa bật, và rủi ro đo đạc (cột phải, phần dưới)
**Thời lượng: ~45 giây**

Hai ô cuối của slide là phần em nghĩ quan trọng nhất. Có năm lời gọi trong `train.py` đã được viết đầy đủ nhưng **bị comment**, nên không bao giờ chạy: `expand_undersized_gs` ở dòng 356, `final_prune_structgs` ở 418, prune đa view trong nhánh warmup ở 393, `training_report` ở 306 — nên không có log PSNR nào lúc train — và `scene.save` ở dòng 442. Cộng thêm mười cờ mặc định `False` và hai nhánh chết do truyền `None` vào đối số. Phần **thực sự chạy** của SADGS chỉ là một chuỗi bốn khâu: cache structure tensor, `update_freq_stats_online`, tính `high_ratio` với `low_ratio`, rồi `densify_and_prune_structgs`.

Về đo đạc, có ba rủi ro cần nói: `render.py` dòng 40 đến 44 đo FPS bằng `time.time()` mà **không** gọi `cuda.synchronize()`, nên con số FPS báo cáo có thể lạc quan; `metrics.py` dòng 74 dùng hàm SSIM khác với `fast_ssim` dùng lúc train; và `full_eval.py` ở các dòng 90, 95, 121 truyền ba tham số `--budget`, `--mode`, `--sh_lower` mà `argparse` chưa hề khai báo, nên script đó không chạy được như đang viết. Cuối cùng — và em xin nhấn mạnh — **repo SADGS hiện không chứa bất kỳ số đo PSNR, SSIM, LPIPS hay thời gian train nào**. Bảng số liệu trong `DOCS/README.md` là của pipeline FastGS-lite cũ, trước khi tích hợp structure-aware densification; đó **không** phải kết quả của SADGS, và em không trình bày nó như vậy.

**Chuyển tiếp:** Em xin kết thúc phần SADGS ở đây và chuyển sang phần hỏi đáp.

---

#### Nếu bị hỏi

**Hỏi: "Em nói SADGS train 30 nghìn iteration à?"**
Dạ không. 30 000 chỉ là mặc định trong `arguments/__init__.py` dòng 75. Script thật sinh số liệu paper là `run_train.sh`, và nó dùng `--iterations 3000` cho Mip-NeRF 360 với Deep Blending, `--iterations 7000` cho Tanks & Temples, với `densify_until_iter` tương ứng là 1900 và 4000. Thêm nữa `run_train.sh` truyền `--densification_interval 500`, nên khối densify chỉ chạy khoảng ba đến bốn lần trong toàn bộ quá trình train — toàn bộ cơ chế structure-aware phải phát huy tác dụng trong vài lần gọi đó.

**Hỏi: "Vậy các cờ mặc định TẮT nghĩa là paper không dùng chúng?"**
Không hẳn, và đây là điểm em phải nói rõ. `sample_bbox_faces` và `warmup_densification` mặc định là `False` ở dòng 129 và 130, nhưng **cả bốn cấu hình** trong `run_train.sh` đều truyền `--sample_bbox_faces --warmup_densification` để bật lại. Cấu hình 2 và 4 còn truyền `--batch_size 2`. Nên "cấu hình mặc định" và "cấu hình paper" là hai chế độ khác nhau — bảng mặc định không mô tả lần chạy sinh ra số liệu công bố.

**Hỏi: "Split dị hướng có nguy cơ bùng nổ số Gaussian không?"**
Có, và đây là một bug thật em tìm thấy. Docstring ở `gaussian_model.py` dòng 697 viết "clamp min=2 and max=8 to limit VRAM usage", nhưng dòng thực thi 701 chỉ là `torch.clamp(ks, min=1)` — **không có trần 8**. Mà số con sinh ra là tích $k_x k_y k_z$, nên một Gaussian có $\eta$ cao ở cả ba trục có thể sinh ra rất nhiều con. Chốt chặn duy nhất còn lại là ngưỡng gradient $10^{-5}$ ở `train.py` dòng 333, mà chính comment trong code gọi là `[CRITICAL] We MUST use this to prevent 7M points`.

**Hỏi: "Với `batch_size 2` thì loss có được trung bình không?"**
Không. `train.py` dòng 262 có `#/ opt.batch_size` bị comment, và `loss.backward()` ở dòng 263 nằm **trong** vòng lặp batch, nên gradient cộng dồn chứ không trung bình — learning rate hiệu dụng tăng gấp `batch_size` lần. Vì `run_train.sh` thật sự dùng `--batch_size 2`, đây là thứ cần sửa trước khi so sánh công bằng giữa các cấu hình.

**Hỏi: "File `.ply` xuất ra nặng bao nhiêu?"**
`construct_list_of_attributes` ở `gaussian_model.py` dòng 379 sinh ra 63 trường `float32`: 3 toạ độ, 3 normal luôn bằng 0, 3 hệ số SH bậc 0, 45 hệ số SH bậc cao, 1 opacity, 3 scale, 4 quaternion, và 1 trường `filter_3D`. Vậy là 252 byte mỗi Gaussian — nhiều hơn đúng 4 byte so với record 3DGS chuẩn, vì SADGS ghi thêm `filter_3D`.

---
---

# KHỐI F — Quan sát cuối & kết luận

---
## Slide 38: Quan sát 2: Nút thắt là RAM hệ thống, không phải VRAM
**Lời thuyết trình (rút gọn):**
Quan sát thứ hai: trên Colab T4, nút thắt thật sự là RAM hệ thống, không phải VRAM. VRAM chỉ dùng 8%, trong khi RAM hệ thống áp sát 91% giới hạn mềm. Hệ quả thực tiễn: khuyến nghị truyền thống "giảm resolution để tiết kiệm VRAM" thực ra không cần thiết, vì VRAM vốn dư thừa rất nhiều — muốn tối ưu nên tập trung quản lý RAM. Thú vị hơn, vì VRAM không phải nút thắt, train trực tiếp ở resolution gốc, tức resolution bằng 1, hoàn toàn khả thi — đồng thời loại bỏ luôn độ lệch resolution giữa train và eval mà nhóm gặp phải. Đây là hướng cải tiến khả thi cho các lần chạy sau. Tiếp theo, nhóm xin tổng kết lại toàn bộ pipeline render 3D Gaussian Splatting.

---
## Slide 39: Tổng kết pipeline render 3D Gaussian Splatting
**Lời thuyết trình (rút gọn):**
Nhóm xin tổng kết pipeline render bằng một sơ đồ khép kín, lặp lại qua từng iteration. Bắt đầu từ tập Gaussian 3D — vị trí, hiệp phương sai, độ mờ, hệ số spherical harmonics. Các Gaussian được chiếu lên ảnh qua phép chiếu phối cảnh xấp xỉ EWA, rasterize theo tile bằng rasterizer khả vi, rồi alpha-blend ra ảnh cuối. Ảnh này so với ảnh thật qua loss kết hợp L1 và D-SSIM, gradient lan truyền ngược để cập nhật tham số, đồng thời điều khiển Adaptive Density Control — thêm, tách, loại Gaussian — rồi vòng lặp quay lại điểm xuất phát. Toàn bộ cải tiến của SADGS chính là tăng tốc các bước trong chu trình này. Sau đây, nhóm tổng kết lại các cải tiến cụ thể và giới hạn còn lại.

---
## Slide 40: Tổng kết cải tiến SADGS & giới hạn còn lại
**Lời thuyết trình (rút gọn):**
SADGS cải tiến 3DGS gốc qua ba đòn bẩy: Multi-view score kết hợp Compact box giảm số Gaussian dư thừa mà không giảm chất lượng — lý do playroom và drjohnson đạt điểm cao dù dùng ít Gaussian hơn; Sparse Adam giảm chi phí mỗi iteration bằng cách chỉ cập nhật tham số thực sự liên quan. Bên cạnh đó, nhóm cũng nhận diện rõ giới hạn: dữ liệu Gaussian đầu ra chưa được nén; ngân sách 7000 iteration của SADGS bỏ lỡ bước final_prune quan trọng; chưa có ablation study để cô lập đóng góp từng đòn bẩy; và bốn nhánh mở rộng của SADGS gốc — dynamic scenes, sparse view, surface reconstruction, SLAM — chưa được tích hợp, đây là hướng phát triển tiềm năng cho tương lai.

---
## Slide 41: Cảm ơn đã theo dõi! / Hỏi đáp (Q&A)
**Lời thuyết trình (rút gọn):**
Như vậy, nhóm em vừa trình bày xong toàn bộ nội dung đồ án: từ nền tảng toán học của 3D Gaussian Splatting, ba đòn bẩy cải tiến tốc độ của SADGS, quá trình đóng gói thành SADGS để chạy được trên phần cứng miễn phí như Colab T4, cùng kết quả thực nghiệm và bài học rút ra — nổi bật nhất là RAM hệ thống mới là nút thắt thật sự chứ không phải VRAM, và việc thiếu bước prune cuối do giới hạn ngân sách iteration. Em xin chân thành cảm ơn quý thầy cô trong hội đồng cùng toàn thể các bạn đã dành thời gian lắng nghe phần trình bày của nhóm hôm nay. Rất mong nhận được câu hỏi, góp ý và nhận xét từ thầy cô để nhóm hoàn thiện đồ án tốt hơn. Nhóm em xin phép được lắng nghe và trả lời câu hỏi ạ. Xin cảm ơn.

