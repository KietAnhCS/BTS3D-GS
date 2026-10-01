# Công thức toán học trong `utils/system_utils.py`

**Nhận định**: file này **không chứa công thức toán học**. Toàn bộ nội dung là các hàm tiện ích thao tác hệ thống tập tin (I/O), không liên quan đến tính toán số học/hình học của 3D Gaussian Splatting.

---

## Tóm tắt chức năng

### `mkdir_p(folder_path)`

Trích nguyên văn (`system_utils.py`, dòng 16–24):

```python
def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
```

Tạo thư mục đệ quy, tương đương lệnh shell `mkdir -p`:

- Gọi `os.makedirs(folder_path)`.
- Nếu gặp `OSError` với `errno == EEXIST` và đường dẫn đã là thư mục tồn tại → bỏ qua lỗi (coi như thành công).
- Nếu là lỗi khác → raise lại ngoại lệ.

Mục đích: đảm bảo thư mục output/checkpoint tồn tại trước khi ghi file, không gây lỗi nếu thư mục đã có sẵn.

### `searchForMaxIteration(folder)`

Trích nguyên văn (`system_utils.py`, dòng 26–28):

```python
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
```

Tìm iteration (bước huấn luyện) lớn nhất đã được lưu checkpoint trong một thư mục:

- Liệt kê toàn bộ tên file/thư mục con trong `folder` (`os.listdir`).
- Với mỗi tên file, tách theo dấu `_` và lấy phần cuối cùng, ép kiểu `int` — quy ước đặt tên dạng `..._<iteration>` (ví dụ `iteration_7000`, `point_cloud_30000`).
- Trả về giá trị lớn nhất trong danh sách các số nguyên thu được (`max(saved_iters)`).

Mục đích: khi resume training hoặc load checkpoint ở chế độ "mới nhất", hàm này xác định checkpoint ứng với iteration cao nhất đã lưu, không thực hiện phép toán nào ngoài so sánh số nguyên (`max`).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `makedirs(folder_path)` | Không có công thức toán học (thao tác I/O: tạo thư mục) |
| `exc.errno == EEXIST and path.isdir(folder_path)` | Không có công thức toán học (kiểm tra điều kiện tồn tại) |
| `[int(fname.split("_")[-1]) for fname in os.listdir(folder)]` | $S = \{ \text{int}(\text{split}(f,\texttt{"\_"})[-1]) : f \in \text{listdir(folder)} \}$ |
| `max(saved_iters)` | $\text{iter}_{max} = \max(S)$ |
