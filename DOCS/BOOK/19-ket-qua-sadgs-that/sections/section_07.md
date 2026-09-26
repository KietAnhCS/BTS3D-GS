## Quan sát định tính trên ảnh render thực tế (submission scene HCM0539)

Để đánh giá chất lượng thực sự của mô hình Gaussian Splatting sau khi train trên scene HCM0539, nhóm đã xem trực tiếp 6/60 ảnh render trong bộ `submission.zip` đã nộp, lấy dàn đều theo chỉ số khung hình: `0001.png`, `0011.png`, `0021.png`, `0031.png`, `0041.png`, `0051.png`. Đây là ảnh render thật từ model đã train (không phải minh hoạ, không phải ảnh dựng giả), độ phân giải 1320x989.

![Contact sheet 6 ảnh submission thật, scene HCM0539](figures/07_submission_montage_real.png)

### Những gì quan sát được

- **Bố cục cảnh và độ sắc nét tổng thể**: Cả 6 ảnh đều là góc nhìn chéo trên cao (góc chụp kiểu drone/oblique) xuống một khu dân cư đông đúc dọc kênh rạch (nhìn thấy rõ mặt nước và cây xanh ở góc trên bên trái của các ảnh 0021, 0041, 0051). Các mái nhà, mái tôn, bồn nước inox, tấm pin năng lượng mặt trời, sân thượng được tái tạo khá rõ nét, đường viền mái nhà, gờ tường, cửa sổ vẫn phân biệt được, màu sắc mái tôn xanh/đỏ, tường trắng/hồng/vàng trông tự nhiên, không bị ám màu hay sai lệch tông rõ rệt.

- **Artifact dạng vệt sọc dọc quanh cột/trụ ăng-ten**: Đây là lỗi rõ nhất và lặp lại ở hầu hết các ảnh (0001, 0011, 0031, 0041, 0051). Ở giữa khung hình có một cột ăng-ten viễn thông (trụ BTS) cao, và xung quanh/ngay tại vị trí cột này xuất hiện các vệt nhiễu dọc dạng "ghosting"/floater rung lắc, làm cột bị nhòe thành nhiều lớp chồng lên nhau, mất hình dạng khối trụ rõ ràng. Đây nhiều khả năng là artifact kinh điển của 3DGS với vật thể mảnh, cao, ít được quan sát từ nhiều góc (occlusion thay đổi liên tục giữa các khung hình train) khiến Gaussian không hội tụ tốt tại vị trí đó.

- **Artifact dạng vệt ngang lượn sóng ở vùng trời/xa (ảnh 0001, 0011)**: Ở rìa trên của hai ảnh này, vùng xa (bầu trời hoặc mặt nước xa) xuất hiện các vệt ngoằn ngoèo, lởm chởm giống hoa văn nhiễu tần số cao — điển hình của floater bị kéo giãn khi render từ góc nhìn nằm ngoài vùng được các camera train bao phủ tốt.

- **Vệt nứt/đường đen chéo lặp lại**: Ở các ảnh 0021, 0031, 0041, 0051 đều thấy một hoặc vài đường đen mảnh, sắc nét, chạy chéo xuyên qua khung hình (rõ nhất ở góc phải-trên đến giữa ảnh). Đường này xuất hiện ở vị trí gần như cố định qua nhiều khung hình liên tiếp, gợi ý đây là seam/khe hở do thiếu Gaussian tại ranh giới giữa các "mảnh" scene được ghép hoặc là floater phẳng nằm sai vị trí theo chiều sâu, chứ không phải nhiễu ngẫu nhiên mỗi ảnh.

- **Không thấy vùng bị thiếu hình học lớn (hole) hay biến dạng nghiêm trọng ở các vùng mái nhà/công trình chính**: phần lớn diện tích ảnh (nhà cửa, mái, sân) được tái tạo đặc, liền mạch, không có lỗ thủng lớn hay mất kết cấu; artifact chủ yếu tập trung ở vật thể mảnh (cột ăng-ten), vùng xa/trời, và một số đường seam chéo cụ thể.

### Kết luận ngắn

Chất lượng render tổng thể ở mức chấp nhận được cho một scene đô thị dày đặc, hình học chính (nhà, mái) tái tạo tốt, màu sắc tự nhiên. Tuy nhiên vẫn còn artifact rõ ràng và lặp lại có hệ thống ở: (1) cột/trụ mảnh cao (ăng-ten viễn thông), (2) vùng xa/nền trời tại một số khung hình, và (3) các đường seam/vệt chéo cố định xuất hiện xuyên suốt nhiều khung hình liên tiếp. Đây là các điểm cần cải thiện ở bước huấn luyện (thêm góc nhìn train phủ vật thể mảnh, densify/prune tốt hơn, hoặc regularize theo chiều sâu) để giảm floater và seam trong các lần huấn luyện tiếp theo.
