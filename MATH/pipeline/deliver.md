# Công thức toán học trong `pipeline/deliver.py`

**Nhận xét:** File đóng gói mô hình (`.ply`) và báo cáo thành ZIP, tự động sao lưu sang Google Drive, và tải về máy. Hầu như không có công thức toán học — chỉ có phép chuyển đổi đơn vị dung lượng (byte → MB) và quy tắc chọn checkpoint mới nhất theo số vòng lặp lớn nhất.

## 1. Chọn checkpoint mới nhất theo số vòng lặp (iteration) lớn nhất

Trích nguyên văn (`pipeline/deliver.py`, dòng 17–21, hàm `pack_models`):

```python
plys = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*",
                              "point_cloud.ply"))
plys.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[-1]))
for path in plys[-1:]:                     # chỉ giữ mốc mới nhất
    relative = os.path.relpath(path, model_path).replace(os.sep, "/")
```

Tên thư mục checkpoint có dạng `iteration_<N>`. Danh sách đường dẫn `.ply` được sắp xếp theo khoá là số nguyên `N` trích từ tên thư mục:

$$
\text{key}(p) = N(p), \quad \text{với } N(p) = \text{phần số sau dấu "\_" cuối trong tên thư mục chứa } p
$$

$$
\text{plys\_sorted} = \operatorname*{sort}_{p}\ \text{key}(p) \ \text{(tăng dần)}, \qquad \text{latest} = \text{plys\_sorted}[-1] = \operatorname*{argmax}_p N(p)
$$

Trong `pack_models`, chỉ checkpoint có $N$ lớn nhất được giữ lại (`plys[-1:]`, dòng 20).

Tương tự trong `autosave_scene` (`pipeline/deliver.py`, dòng 62–67):

```python
plys = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*", "point_cloud.ply"))
if not plys:
    print(f"[{scene}] chưa có .ply để lưu")
    return None
plys.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[-1]))
latest = plys[-1]
```

$$
\text{latest} = \text{plys\_sorted}[-1] = \operatorname*{argmax}_p N(p)
$$

đúng cùng công thức, dùng lại để chọn mốc huấn luyện mới nhất mang sao lưu sang Drive.

## 2. Chuyển đổi dung lượng byte sang megabyte

Dùng nhất quán ở cả 3 hàm. Trích nguyên văn:

`pack_models` (dòng 31):

```python
size_mb = os.path.getsize(cfg.model_zip) / 1024 ** 2
```

`autosave_scene` (dòng 78):

```python
print(f"[{scene}] đã lưu Drive: {target} ({os.path.getsize(target) / 1024 ** 2:.1f} MB)")
```

`download` (dòng 106, 109):

```python
print("file sẵn sàng:", path, f"({os.path.getsize(path) / 1024 ** 2:.1f} MB)")
...
print("đang tải về:", path, f"({os.path.getsize(path) / 1024 ** 2:.1f} MB)")
```

$$
\text{size\_mb} = \frac{\text{filesize (byte)}}{1024^2}
$$

($1024^2 = 1\,048\,576$ byte/MB — quy ước mebibyte nhị phân, không phải $10^6$ của megabyte thập phân SI.)

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc | Công thức tương ứng |
|---|---|
| `plys.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("_")[-1]))` | $\text{key}(p) = N(p)$, sắp xếp tăng dần theo số vòng lặp |
| `plys[-1:]` (trong `pack_models`) | giữ lại phần tử có $N(p)$ lớn nhất: $\operatorname*{argmax}_p N(p)$ |
| `latest = plys[-1]` (trong `autosave_scene`) | $\text{latest} = \operatorname*{argmax}_p N(p)$ |
| `size_mb = os.path.getsize(cfg.model_zip) / 1024 ** 2` | $\text{size\_mb} = \text{bytes}/1024^2$ |
| `os.path.getsize(target) / 1024 ** 2` | $\text{size\_mb} = \text{bytes}/1024^2$ |
| `os.path.getsize(path) / 1024 ** 2` | $\text{size\_mb} = \text{bytes}/1024^2$ |
| `os.path.relpath(path, model_path).replace(os.sep, "/")` | — (chuẩn hoá đường dẫn, không phải công thức toán) |
