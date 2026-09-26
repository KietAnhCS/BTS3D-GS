## Đối chiếu quy trình chạy thật trong `bts_digital_twin.ipynb` với dữ liệu output

Notebook `bts_digital_twin.ipynb` được thiết kế theo triết lý "glue only": mỗi cell chỉ gọi một hàm trong package `pipeline/`, toàn bộ logic (train, score, render, đóng gói) nằm ngoài notebook. Notebook gồm 19 cell (9 markdown giải thích + 10 cell code chạy pipeline), tương ứng 7 bước chính đánh số 0→7. Dưới đây là đối chiếu từng bước với bằng chứng thật lấy từ thư mục `output/26sepoutput/`.

### Bước 0 — Get the code (cell 2)

Notebook clone repo `https://github.com/KietAnhCS/BTS3D-GS.git` vào `/content/BTS3D-GS`, hoặc dùng thư mục hiện tại nếu đã có `pipeline/` (trường hợp chạy local). Đây là bước hạ tầng, không sinh ra dữ liệu số liệu riêng để đối chiếu.

- **Bằng chứng**: không có file log riêng của bước này. Suy luận gián tiếp: toàn bộ các bước sau (import `pipeline.env`, `pipeline.data`, `pipeline import Config, run, report` ở cell 4 và cell 6) chạy được và sinh output hợp lệ, nghĩa là mã nguồn `pipeline/` đã được nạp đúng.

### Bước 1 — Check GPU / mount Drive (cell 4)

Gọi `install_dependencies()` (pip package + build 3 submodule CUDA), `mount_drive()` (xin quyền Google Drive một lần), `check_gpu(require=True)` (kiểm tra GPU/VRAM/torch/CUDA/RAM, bắt buộc phải có GPU nếu không sẽ raise lỗi).

- **Bằng chứng gián tiếp**: `check_gpu(require=True)` chặn cứng nếu không có GPU — vì bước Train (30.000 iteration, cell 13) đã chạy xong và tạo ra 2.968.520 Gaussian, chắc chắn GPU đã được cấp phát thành công. Cột `vram_gb` trong `history.csv` (dao động 2.6 → 7.2 GB) là dấu vết trực tiếp cho thấy quá trình theo dõi VRAM của `check_gpu`/`env.py` đang hoạt động xuyên suốt training, không chỉ ở bước khởi tạo.
- Không có file log riêng ghi lại kết quả `check_gpu`/`mount_drive` (in ra console, không lưu file), nên không thể xác nhận trực tiếp thông số GPU (model, dung lượng VRAM tối đa) từ artefact — chỉ suy ra pipeline đã chạy trên GPU thật vì các bước tốn tài nguyên compute sau đó hoàn tất.

### Bước 2 — Configuration (cell 6)

Cell này định nghĩa `Config(...)` với các tham số quan trọng:
`scene_root="/content/drive/MyDrive/VAI_NVS_DATA_ROUND2"`, `drive_subdir="HCM0539"`, `resolution=2`, `iterations=30000`, `score_every=1000`, `submission_resolution=1`, `autosave_to_drive=True`, `save_to_drive=True`, `submission_order="csv"`.

- **Bằng chứng trực tiếp và khớp 100% với dữ liệu thật**:
  - `history.csv` (`output/26sepoutput/sadgs_models/history.csv`) có đúng **30 dòng dữ liệu** (không tính header), với cột `iter` chạy đều 1000, 2000, ..., 30000 — khớp chính xác `score_every=1000` và `iterations=30000`.
  - `submission/HCM0539/` chứa đúng scene `HCM0539` — khớp `drive_subdir="HCM0539"`.
  - Ảnh submission có kích thước **1320×989 px** (đo trực tiếp trên `0001.png`), tức là kích thước ảnh gốc test (không bị chia đôi như ảnh train), khớp `submission_resolution=1` (render đúng độ phân giải gốc, khác với `resolution=2` chỉ áp dụng cho ảnh train).
  - Có đúng **60 ảnh** trong `submission/HCM0539/` — khớp mô tả trong cell 5 (mỗi scene test có 60 ảnh) và với `test_poses.csv`.
  - Sự tồn tại của `sadgs_models.zip` và `submission.zip` ở cùng thư mục output là bằng chứng gián tiếp cho `save_to_drive=True`/`autosave_to_drive=True` — logic zip/lưu Drive được kích hoạt và không bị tắt.

### Bước 3 — Load & profile dataset (cell 8: `run.load_data(cfg)`)

Hàm này lấy dữ liệu (ưu tiên gắn Drive), thu hẹp vào `drive_subdir`, tìm scene, đối chiếu số ảnh train/test với `README.txt`, in ra bảng `profile`.

- **Bằng chứng gián tiếp**: không có file profile riêng được lưu ra đĩa (chỉ `display(profile)` trong notebook, không ghi file). Tuy nhiên vì `run.run_all` ở cell 13 chạy được trên scene `HCM0539` và sinh ra đúng số liệu (2.968.520 Gaussian sau 30k iteration, 60 ảnh test render đúng kích thước), suy ra bước load/profile đã tìm đúng scene, đối chiếu đủ 240 ảnh train + 60 ảnh test mà không bị cảnh báo `[THIẾU]` (nếu thiếu ảnh, README-check sẽ cảnh báo và có khả năng làm sai lệch số liệu ở các bước sau).

### Bước 4 — Quick smoke test (cell 10: `run.smoke_test(cfg, scenes[0])`)

Chạy vài trăm iteration trên một scene để kiểm tra dữ liệu + CUDA + scoring hoạt động trước khi tốn hàng giờ train thật.

- **Bằng chứng**: không có file log riêng để xác nhận (smoke test không ghi vào `history.csv`, chỉ in tiến trình ra console và không lưu artefact). Suy luận gián tiếp: vì bước Train (cell 13, tốn hàng giờ) chạy tới hết 30.000 iteration mà không crash, và vì logic pipeline được mô tả là "test data + CUDA + scoring hoạt động trước khi train thật", nhiều khả năng smoke test đã pass — nếu nó fail thì thông thường notebook sẽ dừng ở đây trước khi tới bước train tốn tài nguyên.

### Bước 5 — Train (cell 13: `run.run_all(cfg, scenes)`)

Đây là bước có bằng chứng mạnh nhất. Theo mô tả ở cell 12: sau mỗi scene, `.ply` được chép lên Drive ngay (`autosave_to_drive`), giải phóng RAM/VRAM, render test pose, giải phóng lần nữa rồi ghi `results.json`.

- **Bằng chứng trực tiếp**:
  - `history.csv` có 30 dòng, mỗi dòng ứng với một mốc `score_every=1000`, cột `elapsed_s` tăng đơn điệu từ 58s (iter 1000) đến 4190.6s (iter 30000, tức tổng thời gian train ~70 phút cho scene này).
  - `n_gauss` tăng dần từ 1.072.940 (iter 1000) lên tới 2.968.520 và giữ nguyên từ iter ~19000 trở đi — hành vi đặc trưng của 3D Gaussian Splatting (số Gaussian tăng trong giai đoạn densification rồi ổn định).
  - Các chỉ số chất lượng tăng và hội tụ: `psnr` từ 17.58 → 23.78, `ssim` từ 0.670 → 0.890, `lpips` giảm từ 0.586 → 0.0915, `score` (công thức `0.4(1−LPIPS)+0.3·SSIM+0.3·PSNR_norm`) tăng từ 0.649 → 0.868 rồi bão hòa (`d_score` ở các dòng cuối chỉ còn ±0.0005) — đúng dạng đường cong hội tụ của training thật, không phải số giả lập tuyến tính.
  - `leaderboard.csv` tổng hợp đúng scene `HCM0539` với `iters=30000`, `n_gauss=2968520`, `train_s=4234.8`, `peak_vram_gb=7.21`, `images=60`, `size=1320x989` — khớp khít với dòng cuối của `history.csv` (chỉ số `train_s` lớn hơn `elapsed_s` một chút do gồm cả overhead render/lưu sau khi train xong).

### Bước 6 — Data analytics (cell 15–16: `run.analytics(cfg, results, submissions)` và `report.show_samples`)

Sinh ra `history`, `board` (bảng so sánh các scene) và vẽ ảnh render cạnh ground truth.

- **Bằng chứng trực tiếp**: file `sadgs_models/history.csv` và `sadgs_models/leaderboard.csv` chính là kết quả xuất ra từ bước này (tên biến `history`/`board` khớp tên file). Ngoài ra còn có `sadgs_models/leaderboard.png` (33.7 KB) và `sadgs_models/training.png` (146 KB) — hai biểu đồ trực quan hoá do `report`/`analytics` sinh ra, xác nhận bước visualize đã chạy và ghi file thành công.
- `report.show_samples(cfg, scenes[0], n=3)` (cell 16) chỉ hiển thị inline trong notebook, không lưu file riêng, nên không có bằng chứng file cho riêng cell này — suy luận gián tiếp từ việc `submission/HCM0539/` có đủ 60 ảnh render hợp lệ (đọc được bằng PIL, kích thước đúng 1320×989) rằng cơ chế render dùng chung đã hoạt động đúng.

### Bước 7 — Build submission.zip (cell 18: `run.finish(cfg, scenes)`)

`run.finish` nén ảnh thành `submission.zip`, đối chiếu số lượng pose/kích thước ảnh/tên file liên tục với `test_poses.csv`, trả về `check, problems`.

- **Bằng chứng trực tiếp**: `submission.zip` (132.438.322 bytes ≈ 126 MB) và `sadgs_models.zip` (748.359.637 bytes ≈ 713 MB) tồn tại thật trong `output/26sepoutput/`, với timestamp tạo lần lượt 16:23 và 16:27 (submission.zip xong trước, models.zip xong sau — khớp với thứ tự code: `finish` xử lý ảnh/nén trước khi model lớn hơn được đóng gói/chép xong). Thư mục `submission/HCM0539/` chứa đúng 60 file PNG đọc được, khớp yêu cầu đối chiếu với `test_poses.csv` (60 pose).
- Biến `problems` (danh sách lỗi nếu số lượng/kích thước không khớp CSV) không được ghi ra file, nên không thể xác nhận trực tiếp nó rỗng — nhưng vì `submission.zip` tồn tại và chứa đủ 60 ảnh đúng kích thước gốc, khả năng cao bước đối chiếu đã pass mà không có lỗi nghiêm trọng.

### Nhận định

Chuỗi bằng chứng khớp nhau xuyên suốt 7 bước — từ cấu hình khai báo (`scene_root`, `iterations=30000`, `score_every=1000`, `submission_resolution=1`) đến số liệu thật (30 dòng `history.csv` đúng mốc, 2.968.520 Gaussian hội tụ, 60 ảnh submission đúng độ phân giải gốc, hai file zip cuối cùng) — cho thấy đây không phải log giả lập mà là một lần chạy thật, đủ nhất quán nội bộ để tin cậy. Điểm yếu duy nhất là các bước không ghi file (mount Drive, check GPU, smoke test, load/profile, hiển thị mẫu render) chỉ có thể xác nhận gián tiếp qua việc các bước sau chạy trót lọt, chứ không có log độc lập.

Về khả năng tái lập: thiết kế `autosave_to_drive=True` (chép `.ply` lên Drive ngay sau mỗi scene, trước khi giải phóng bộ nhớ) và `save_to_drive=True` (đẩy cả `submission.zip`/`sadgs_models.zip` lên Drive trước khi tải về máy) là cơ chế chống mất dữ liệu hợp lý cho môi trường Colab vốn hay bị ngắt phiên giữa chừng — nếu runtime chết ở iteration 25000 chẳng hạn, model của các scene đã hoàn tất trước đó vẫn an toàn trên Drive. Kết hợp với việc pipeline tách hoàn toàn logic ra khỏi notebook (chỉ gọi hàm `pipeline.*`), quy trình này có thể coi là dễ tái lập: chạy lại notebook với cùng `Config` sẽ đi qua đúng các bước 0-7 như trên, miễn dữ liệu nguồn trên Drive (`VAI_NVS_DATA_ROUND2/HCM0539`) không đổi.
