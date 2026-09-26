[← Mục lục](00-muc-luc.md) · Chương 14/17

# Chương 14 — Triển khai thực tế: cài đặt & chạy SAD-GS

> Nguồn: `SADGS/README.md`, `SADGS/run_train.sh`, `SADGS/environment.yml`.
>
> **Đổi so với bản trước.** Chương 14 trước đây (khi tài liệu còn viết cho pipeline cũ, trước khi chuyển sang SAD-GS) mô tả một
> notebook Google Colab (`fastgs-acceleration-method.ipynb`) và gói `pipeline/` (`Config`, `pipeline.run.run_all`,
> chống tràn RAM tự động, `submission.zip`...), cùng một "nhật ký huấn luyện" ghi lại một phiên chạy thật trên
> Colab T4. **Không cái nào trong số đó tồn tại trong `SADGS/`.** Không có notebook, không có thư mục
> `pipeline/`, không có cơ chế nộp bài, và không có log/CSV của bất kỳ phiên train thật nào trong repo này.
> Chương này viết lại theo đúng những gì `SADGS/` thật sự cung cấp: cài đặt bằng conda, chạy bằng
> `run_train.sh` trên máy/cluster có GPU CUDA, và một mục cuối nói rõ đây là phần **chưa có dữ liệu đo được**.

## 14.1 — Hai bộ hướng dẫn cài đặt không khớp nhau trong repo

`SADGS/` có **hai nguồn khai báo môi trường khác nhau**, và chúng mâu thuẫn nhau — đáng ghi nhận trước khi
làm theo bất kỳ nguồn nào:

| | `SADGS/environment.yml` | Comment đầu `SADGS/run_train.sh` (dòng 20-27) và `SADGS/README.md` |
|---|---|---|
| Python | `3.7.13` | `3.12` |
| PyTorch | `1.12.1` | cài qua `--index-url .../cu121` (không ghim version) |
| CUDA toolkit | `11.6` (`cudatoolkit=11.6`, kênh conda) | ngụ ý CUDA 12.1 (từ index url `cu121`) |
| Cách cài extension | `pip:` trong cùng file `environment.yml` (`submodules/diff-gaussian-rasterization_structgs`, `submodules/simple-knn`, `submodules/fused-ssim`) | `pip install` riêng từng dòng sau khi tạo env bằng `conda create -n sadgs python=3.12` |

Hai bộ số này **không tương thích**: một CUDA extension biên dịch với `cudatoolkit=11.6`/PyTorch 1.12 (theo
`environment.yml`) không chạy được với runtime PyTorch build cho `cu121` (theo `README.md`/`run_train.sh`), vì
CUDA extension của PyTorch được biên dịch gắn chặt với ABI của bản PyTorch dùng lúc build. Đây gần như chắc
chắn là dấu vết của hai giai đoạn phát triển khác nhau (một bản môi trường cũ hơn để lại trong
`environment.yml`, một bản hướng dẫn mới hơn trong README/`run_train.sh`) — không suy diễn thêm bản nào
"đúng hơn"; người dùng cần **chọn một trong hai và làm nhất quán từ đầu đến cuối**, không trộn.

**Khuyến nghị thực dụng:** làm theo `README.md`/`run_train.sh` (Python 3.12, `cu121`) vì đó là bộ hướng dẫn
mới hơn và là bộ duy nhất có script chạy thật (`run_train.sh`) đi kèm; `environment.yml` chỉ nên dùng để tham
khảo tên các submodule cần cài (ba dòng `pip:` giống hệt nhau ở cả hai nguồn), không nên dùng để tạo `conda
env create -f environment.yml` nếu đã cài theo README.

## 14.2 — Cài đặt theo README (khuyến nghị)

```bash
conda create -n sadgs python=3.12 -y
conda activate sadgs

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install plyfile tqdm websockets

pip install submodules/diff-gaussian-rasterization_structgs
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

Nếu CUDA toolkit/driver của máy khác 12.1, README nói rõ: cài bản PyTorch khớp CUDA thật của máy **trước**
khi build ba submodule — build CUDA extension sai version driver là nguyên nhân lỗi phổ biến nhất với loại
repo này (không riêng SAD-GS).

`diff-gaussian-rasterization_structgs` là fork riêng của SAD-GS (khác `diff-gaussian-rasterization_fastgs`
dùng ở các chương trước) — build nó, **không** dùng lại wheel/build của rasterizer cũ dù hai tên gần
giống nhau.

## 14.3 — Bố cục dataset bắt buộc

`run_train.sh` đọc ba biến môi trường trỏ tới ba gốc dataset khác nhau, đúng layout COLMAP/benchmark chuẩn
của 3DGS:

```text
$MIPNERF360_DATASET/          # 360_v2: bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill
$TANDT_DB_DATASET/            # tandt_db/db: drjohnson, playroom
$TANDT_DATASET/               # tandt_db/tandt: train, truck
```

Mỗi scene cần thư mục `images/` cùng dữ liệu COLMAP chuẩn (`sparse/0/{cameras,images,points3D}.{bin,txt}`).
MipNeRF360 lấy từ trang gốc của tác giả; Tanks&Temples + Deep Blending lấy từ gói `tandt_db.zip` chính thức
của 3DGS (cả hai đường link đều nằm trong `SADGS/README.md`, mục Datasets).

## 14.4 — Chạy `run_train.sh`

```bash
export PROJECT_ROOT=/path/to/SADGS
export MIPNERF360_DATASET=/path/to/datasets/360_v2
export TANDT_DB_DATASET=/path/to/datasets/tandt_db/db
export TANDT_DATASET=/path/to/datasets/tandt_db/tandt
export CUDA_VISIBLE_DEVICES=0

bash run_train.sh
```

Script tự kiểm tra từng đường dẫn tồn tại trước khi chạy (`if [ ! -d ... ]; then echo ERROR; exit 1; fi`) —
lỗi sớm và rõ ràng thay vì chạy nửa chừng rồi crash ở bước load dữ liệu. Nó lần lượt: train → render
(`--skip_train`, cùng `--mult` đã dùng lúc train) → `metrics.py`, cho **12 scene cố định** chia theo 4 cấu
hình đã liệt kê ở Chương 13, mục 13.2. Không có tham số dòng lệnh nào của `run_train.sh` để chọn chạy một
scene lẻ — muốn train một scene riêng, gọi thẳng lệnh `python train.py -s ... --eval ...` tương ứng, chép ra
từ đúng khối `switch` của scene đó trong `run_train.sh` (từng dòng đã liệt kê nguyên văn ở mục 13.2).

Kết quả nằm ở `output/<scene>/` — `point_cloud/iteration_<n>/point_cloud.ply` (định dạng chuẩn 3DGS, xem
Chương 4), ảnh render trong `test/ours_<n>/renders`, và `results.json` do `metrics.py` ghi ra sau bước cuối
cùng của mỗi scene (PSNR/SSIM/LPIPS trung bình, không có metric Score tổng hợp như pipeline cũ).

## 14.5 — Yêu cầu tài nguyên (theo README, chưa đo lại)

README khuyến nghị **24 GB VRAM / RTX 4090** — không có phép đo VRAM đỉnh thật nào trong repo `SADGS/` để
đối chiếu (khác chương 14 cũ, nơi từng có log Colab T4 đo được 0.68–1.16 GB cho pipeline cũ; log đó
thuộc về một dataset và một cơ chế densify khác, không áp dụng được cho SAD-GS). Vì `run_train.sh` chạy 4
cấu hình với `--iterations` chỉ 3000–7000 (thấp hơn nhiều so với 30000 mặc định), khả năng cao VRAM/thời gian
thực tế thấp hơn khuyến nghị 24 GB — nhưng đây là suy luận từ so sánh ngân sách vòng lặp, **không phải số đo**.

Muốn có số thật: chạy `run_train.sh` (hoặc một lệnh `train.py` đơn lẻ) trên máy có GPU, dùng `nvidia-smi
--query-gpu=memory.used --loop=5` hoặc tương đương song song với quá trình train, rồi ghi lại. Không có hook
đo tài nguyên tích hợp sẵn trong `train.py`.

## 14.6 — Không có nhật ký huấn luyện thật trong repo này

Phần trước của chương này (khi còn viết cho pipeline cũ) có một "Phiên 1" đầy đủ: môi trường Colab T4 thật,
bảng Score/PSNR/SSIM/LPIPS theo từng scene, biểu đồ `training.png`/`leaderboard.png`, và một phân tích chi
tiết hai "hố sụt" điểm số tại vòng 3000/6000 do `opacity_reset_interval` không co giãn. Toàn bộ nội dung đó
dựa trên `DOCS/assets/history.csv` và `DOCS/assets/leaderboard.csv` — hai file **không tồn tại và không có
tương đương trong `SADGS/`**. `SADGS/` không có thư mục `output/` sẵn có với kết quả train, không có
`history.csv`, không có log nào đã chạy.

**Vì vậy chương này không có mục "nhật ký huấn luyện thật".** Ghi lại đúng trạng thái: chưa có phiên
`run_train.sh` nào được chạy và lưu kết quả trong phạm vi tài liệu này. Người chạy `run_train.sh` trên máy
của mình nên tự lưu `output/*/results.json` của từng scene lại làm bằng chứng — và nếu muốn nối tiếp mạch
"nhật ký" như tài liệu cũ từng làm cho pipeline trước đây, nên tạo một mục mới ở đây ghi đúng khuôn: môi trường →
cấu hình (switch nào trong `run_train.sh`) → bảng PSNR/SSIM/LPIPS từ `results.json` → thời gian/VRAM đo được
→ những gì học được — chứ không sao chép số liệu của pipeline khác sang.

## 14.7 — Bẫy cấu hình cần biết trước khi chạy thật

Tổng hợp từ Chương 13 và các phần trên, để không mất thời gian debug lại từ đầu:

| Bẫy | Vì sao |
|---|---|
| Trộn `environment.yml` với hướng dẫn README | hai bản PyTorch/CUDA khác nhau, extension build lẫn sẽ lỗi ABI hoặc crash lúc import (mục 14.1) |
| Chạy `full_eval.py` | dùng cờ `--budget`/`--mode` mà `train.py` hiện tại không định nghĩa — báo lỗi tham số ngay lập tức (Chương 13, mục 13.3) |
| Đặt `--iterations` thấp mà kỳ vọng `prune_from_iter=6000` chạy | với `iterations=3000` (config 2/3/4 của `run_train.sh`), nhánh prune theo lịch không bao giờ kích hoạt (Chương 13, mục 13.4) |
| Copy tham số `--mult` giữa các dataset | Tanks&Temples dùng `0.7`, ba nhóm còn lại dùng `0.25` — không phải giá trị mặc định `0.7` của `arguments/__init__.py` cho mọi scene |
| Kỳ vọng có Score tổng hợp trong `results.json` | `metrics.py` ở `SADGS/` là bản gốc 3DGS, chỉ có PSNR/SSIM/LPIPS thô, không có công thức Score như pipeline cũ |

---

[← Chương 13](13-tong-hop-chi-phi-sadgs.md) | [Mục lục](00-muc-luc.md) | [Chương 15 →](15-phu-luc-kiem-dinh-tai-nguyen.md)
