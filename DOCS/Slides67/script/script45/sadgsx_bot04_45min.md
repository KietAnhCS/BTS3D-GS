# Kịch bản 45 phút — Bot 04 (SADGS chuyên sâu, slide `part45_sadgsx_04`)

> Slide: `DOCS/Slides67/Slides45min/part45_sadgsx_04.tex` — "SADGS: tách Gaussian **dị hướng** theo từng trục (`densify_and_split_structgs`)"
> Hình trên slide: `sadgsx/04_3dgs_vs_sadgs.png`
> Code gốc: `SADGS/scene/gaussian_model.py:638-831`; đối chiếu 3DGS `:866-892`

---

## Slide: SADGS — tách Gaussian dị hướng theo từng trục (`densify_and_split_structgs`)

**Thời lượng: ~45 giây**

**Lời thuyết trình:**

Đây là đóng góp cốt lõi của SADGS ở phía density control, và cũng là hàm dài nhất trong mã nguồn — gần hai trăm dòng, thay cho hai mươi bảy dòng của 3DGS gốc. Slide trước đã cho chúng ta đại lượng eta, tức năng lượng tần số vượt ngưỡng Nyquist, đo riêng cho từng trục của ellipsoid. Câu hỏi bây giờ là: đã biết Gaussian vi phạm, thì cắt như thế nào. Ba công thức bên trái trả lời trọn vẹn. Thứ nhất, số lát cắt trên mỗi trục bằng trần của căn bậc hai eta — suy ra thẳng từ lý thuyết lấy mẫu: muốn eta chia cho k bình phương nhỏ hơn hoặc bằng một, thì k phải lớn hơn hoặc bằng căn eta. Trục nào aliasing nhiều thì cắt nhiều; trục nào đã đủ mịn, eta nhỏ hơn một, thì k bằng một, giữ nguyên, không đụng tới. Thứ hai, tổng số con bằng k-x nhân k-y nhân k-z — thay đổi theo từng Gaussian, chứ không khóa cứng ở hai như 3DGS. Thứ ba, các con được đặt trên một lưới đều tất định trong hệ tọa độ local rồi xoay bằng ma trận R của quaternion, với bước lưới bằng căn mười hai nhân sigma mới — hệ số căn mười hai chính là để tập các con tái tạo đúng phương sai của cha. Nhìn hình bên phải: cùng một Gaussian dị hướng, 3DGS bên trái sinh đúng hai con ở vị trí lấy mẫu ngẫu nhiên và thu nhỏ cả hai trục theo cùng hệ số một phẩy sáu, nên con vẫn còn dài theo đúng cái trục đang vi phạm. SADGS bên phải cho k-x bằng bốn, k-y bằng hai, ra tám con trên lưới xác định, mỗi trục co đúng hệ số mà lý thuyết yêu cầu — một lần là xong, thay vì phải tách lại ba bốn vòng.

**Chuyển tiếp:** Tách xong thì Gaussian con được kế thừa gì từ cha, và cha bị xử lý ra sao — đó là nội dung slide tiếp theo.

---

## Nếu bị hỏi

**Hỏi 1: Nếu eta rất lớn thì k có bị chặn trên không? Có nguy cơ bùng nổ bộ nhớ không?**
Câu trả lời thẳng là **không có trần**. Comment ở dòng 698 ghi là "clamp min=2 và max=8", nhưng code thật ở dòng 699–701 chỉ có `torch.clamp(ks, min=1)` — hoàn toàn không có giới hạn trên. Eta bằng 100 sẽ cho k bằng 10, và nếu cả ba trục cùng vi phạm thì một Gaussian sinh một nghìn con trong một lần gọi. Cái giữ cho nó không bùng nổ là ba tầng mặt nạ ở `gaussian_model.py:993-1011`: phải vừa có gradient tuyệt đối vượt `grad_abs_thresh` bằng 0.0002 hoặc được đánh dấu bởi tiêu chí multiview, vừa có scale lớn hơn `dense` nhân extent với `dense` bằng 0.001, vừa có importance score lớn hơn 0.5. Riêng tiêu chí multiview ở `train.py:345` yêu cầu hơn 80% số view đều báo eta cao (`split_ratio_threshold = 0.8`). Ba lớp lọc đó khiến tỉ lệ f trong công thức tăng trưởng n-mới bằng n-cũ nhân với (1 trừ f cộng f nhân N) rất nhỏ trên thực tế.

**Hỏi 2: Vì sao lại là hệ số căn mười hai trong khoảng cách giữa các con?**
Vì một phân bố đều trên đoạn dài L có độ lệch chuẩn bằng L chia căn mười hai. Muốn tập các Gaussian con — vốn là các tâm rời rạc cách đều nhau — tái tạo lại đúng phương sai của Gaussian cha theo trục đó, ta phải đặt bước lưới bằng căn mười hai nhân sigma mới. Đó là điều kiện bảo toàn moment bậc hai, cài ở dòng 775–776. Kèm theo, lưới được căn giữa bằng công thức i trừ (k trừ 1) chia hai ở dòng 763–765, nên trọng tâm tập con trùng khít tâm cha, không bị trôi vị trí. Đánh đổi là khoảng cách tâm–tâm bằng khoảng ba phẩy bốn sáu lần sigma con, lớn hơn bề rộng một sigma, nên ngay sau khi tách bề mặt có khe hở — optimizer sẽ hàn lại trong vài trăm bước tiếp theo.

**Hỏi 3: Tham số `scale_power` để làm gì, có nên chỉnh không?**
Nó là số mũ p trong công thức sigma con bằng sigma cha chia k mũ p, ở dòng 722, lấy từ `args.ks_scale_power`, mặc định 1.0 tại `arguments/__init__.py:138`. Khuyến nghị là **giữ nguyên 1.0**, vì dòng 722 dùng k mũ p để tính kích thước con, nhưng dòng 775 tính khoảng cách lại chỉ dùng sigma chia k, không có p. Hai công thức tách rời nhau: chỉ khi p bằng một thì chúng mới mô tả cùng một hình học. Đặt p bằng hai thì con teo thêm k lần nữa trong khi lưới giữ nguyên độ thưa, dẫn tới thủng bề mặt — hình `04_scale_power.png` trong chương sách cho thấy rõ ba panel p bằng 0.5, 1.0 và 2.0 có tâm con giống hệt nhau, chỉ khác kích thước.

**Hỏi 4 (dự phòng): SADGS có còn dùng split đẳng hướng của 3DGS nữa không?**
Hàm `densify_and_split` gốc vẫn còn nguyên trong file tại dòng 866–892 và được nhánh warmup dùng. Ngoài ra, ngay bên trong `densify_and_split_structgs` cũng có một nhánh dự phòng ở dòng 703–709: nếu `max_eta_3ch` được truyền vào là `None` thì code đặt k bằng hai trên trục dài nhất và k bằng một cho hai trục còn lại, tức N bằng hai — mô phỏng lại đúng 3DGS. Nhưng trong đường chạy thật, `train.py:373` luôn truyền `max_eta_3ch` xuống, nên nhánh đó chỉ là lưới an toàn, không bao giờ chạy.
