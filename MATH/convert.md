# Công thức toán học trong `convert.py`

## Nhận định chung

`convert.py` là một wrapper shell gọi các lệnh **COLMAP** (`feature_extractor`, `exhaustive_matcher`, `mapper`, `image_undistorter`) và **ImageMagick** (`mogrify -resize`) qua `os.system(...)`. File **không tự cài đặt bất kỳ phép biến đổi hình học/camera nào bằng Python/PyTorch** — toàn bộ toán học (SfM: trích đặc trưng SIFT, ghép cặp đặc trưng, bundle adjustment, hiệu chỉnh méo ảnh/undistortion) được thực hiện **bên trong binary COLMAP**, không xuất hiện dưới dạng công thức tường minh trong file Python này.

Các phép toán duy nhất có thể biểu diễn trực tiếp từ nội dung file là các tỉ lệ resize ảnh (số học đơn giản, không phải hình học camera) và một tham số dung sai hội tụ truyền cho bundle adjustment của COLMAP.

---

## 1. Tỉ lệ thu nhỏ ảnh (`--resize`)

Trích nguyên văn (`convert.py`, dòng 100–122):

```python
for file in files:
    source_file = os.path.join(args.source_path, "images", file)

    destination_file = os.path.join(args.source_path, "images_2", file)
    shutil.copy2(source_file, destination_file)
    exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
    if exit_code != 0:
        logging.error(f"50% resize failed with code {exit_code}. Exiting.")
        exit(exit_code)

    destination_file = os.path.join(args.source_path, "images_4", file)
    shutil.copy2(source_file, destination_file)
    exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
    if exit_code != 0:
        logging.error(f"25% resize failed with code {exit_code}. Exiting.")
        exit(exit_code)

    destination_file = os.path.join(args.source_path, "images_8", file)
    shutil.copy2(source_file, destination_file)
    exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
    if exit_code != 0:
        logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
        exit(exit_code)
```

Với ảnh gốc kích thước $(W,H)$, ImageMagick `mogrify -resize p%` tạo ảnh mới với tỉ lệ:

$$
(W', H') = (p\% \cdot W,\; p\% \cdot H)
$$

Ba cấp áp dụng trong code:

$$
p \in \{50\%,\ 25\%,\ 12.5\%\} \;\Longleftrightarrow\; \text{thư mục } \{\texttt{images\_2}, \texttt{images\_4}, \texttt{images\_8}\}
$$

tức tương ứng hệ số chia $\{1/2,\ 1/4,\ 1/8\}$ theo đúng tên thư mục.

## 2. Tham số dung sai Bundle Adjustment (truyền cho COLMAP, không tính trong Python)

Trích nguyên văn (`convert.py`, dòng 58–66):

```python
mapper_cmd = (colmap_command + " mapper \
    --database_path " + args.source_path + "/distorted/database.db \
    --image_path "  + args.source_path + "/input \
    --output_path "  + args.source_path + "/distorted/sparse \
    --Mapper.ba_global_function_tolerance=0.000001")
exit_code = os.system(mapper_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)
```

$$
\text{Mapper.ba\_global\_function\_tolerance} = 10^{-6}
$$

Đây là ngưỡng hội tụ của hàm mục tiêu (reprojection error) trong bundle adjustment toàn cục của COLMAP — bản thân phép tối ưu (tối thiểu hoá sai số tái chiếu $\sum \lVert \pi(P,X)-x\rVert^2$) được thực hiện trong COLMAP (binary bên ngoài), không phải trong `convert.py`.

## 3. Cờ GPU nhị phân

Trích nguyên văn (`convert.py`, dòng 29):

```python
use_gpu = 1 if not args.no_gpu else 0
```

$$
\text{use\_gpu} = \begin{cases} 1 & \text{args.no\_gpu}=\text{False}\\ 0 & \text{args.no\_gpu}=\text{True}\end{cases}
$$

Đây chỉ là một cờ cấu hình, không mang ý nghĩa toán học sâu hơn. Nó được dùng lại ở dòng 40 (`--SiftExtraction.use_gpu " + str(use_gpu)`) và dòng 49 (`--SiftMatching.use_gpu " + str(use_gpu)`).

---

## Bảng hằng số/ngưỡng

| Ký hiệu | Giá trị | Vị trí |
|---|---|---|
| `Mapper.ba_global_function_tolerance` | $10^{-6}$ | dung sai bundle adjustment COLMAP |
| resize ratios | $50\%,\ 25\%,\ 12.5\%$ | tạo `images_2`, `images_4`, `images_8` |
| `camera` model mặc định | `OPENCV` | mô hình camera COLMAP (không phải công thức trong file này) |
| `use_gpu` | $\{0,1\}$ | cờ bật/tắt GPU cho SIFT |

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức |
|---|---|
| `use_gpu = 1 if not args.no_gpu else 0` | $\text{use\_gpu} = \mathbb{1}[\neg \text{no\_gpu}]$ |
| `--Mapper.ba_global_function_tolerance=0.000001` | dung sai hội tụ bundle adjustment $=10^{-6}$ (xử lý nội bộ trong COLMAP) |
| `magick_command + " mogrify -resize 50% " + destination_file` | $(W',H') = (0.5W,\ 0.5H)$ → `images_2` |
| `magick_command + " mogrify -resize 25% " + destination_file` | $(W',H') = (0.25W,\ 0.25H)$ → `images_4` |
| `magick_command + " mogrify -resize 12.5% " + destination_file` | $(W',H') = (0.125W,\ 0.125H)$ → `images_8` |
