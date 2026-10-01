# Công thức toán học trong `pipeline/data.py`

**Nhận xét:** File xử lý tải dữ liệu, dò tìm scene, và lập hồ sơ (profile) dữ liệu. Phần lớn là I/O (tải file, duyệt thư mục). Công thức số học chỉ xuất hiện trong `verify_scene` (đối chiếu số lượng ảnh) và `profile_scenes` (thống kê train/test, kích thước, dung lượng).

## 1. Đối chiếu số ảnh thực tế với README (`verify_scene`)

Trích nguyên văn (`pipeline/data.py`, dòng 119–130) — trích `expected[k]` từ README bằng regex, và đếm `actual[k]` bằng liệt kê thư mục:

```python
    expected = {key: int(value) for key, value in
                re.findall(r"(Train|Test) images:\s*(\d+)", text)}
    if not expected:
        return None

    def _count():
        counts = {}
        for split, key in (("train", "Train"), ("test", "Test")):
            images = os.path.join(root, split, cfg.images_dir)
            counts[key] = (len([f for f in os.listdir(images) if f.endswith(IMAGE_EXT)])
                           if os.path.isdir(images) else 0)
        return counts
```

Với mỗi khoá (`Train`/`Test`) được trích từ README bằng regex `(Train|Test) images:\s*(\d+)`, hàm đếm số ảnh thực tế có đuôi hợp lệ trong thư mục tương ứng:

$$
\text{actual}[k] = \big|\{\, f \in \text{listdir(images}_k) : f \text{ kết thúc bằng đuôi ảnh hợp lệ} \,\}\big|
$$

Trích nguyên văn phần retry + kiểm định (dòng 136–150):

```python
    actual = _count()
    if getattr(cfg, "drive_mount", False) and any(actual.get(k, 0) != v for k, v in expected.items()):
        for _ in range(4):
            time.sleep(2)
            actual = _count()
            if all(actual.get(k, 0) == v for k, v in expected.items()):
                break

    ok = True
    for key, want in expected.items():
        got = actual.get(key, 0)
        mark = "OK" if got == want else "THIẾU"
        if got != want:
            ok = False
        print(f"  [{mark}] {key} images: {got}/{want}")
```

Kiểm định bằng so sánh trực tiếp (dòng 144–150, dùng cho vòng lặp in kết quả; điều kiện dừng sớm trong retry ở dòng 141 dùng `all(...)` tương đương):

$$
\text{ok} = \bigwedge_{k \in \{\text{Train}, \text{Test}\}} \big(\text{actual}[k] = \text{expected}[k]\big)
$$

Khi `drive_mount=True` và điều kiện trên sai (dòng 137), cơ chế thử lại (retry) được áp dụng tối đa $4$ lần với độ trễ $2$ giây mỗi lần (dòng 138–142; do FUSE của Google Drive có thể liệt kê thư mục chưa đầy đủ trong vài giây đầu):

$$
\text{retry}_j,\quad j = 1,\dots,4,\quad \text{độ trễ} = 2\,\text{s/lần}, \quad \text{dừng sớm nếu ok} = \text{True}
$$

## 2. Suy luận số ảnh train/test khi không có `test_poses.csv` (`profile_scenes`)

Trích nguyên văn (`pipeline/data.py`, dòng 272–278):

```python
        csv_path = testposes.poses_csv(cfg, scene)
        if csv_path:
            n_test = len(testposes.read_rows(csv_path))
            n_train, split = len(files), "test_poses.csv"
        else:
            n_test = len(files) // cfg.llffhold if cfg.llffhold else 0
            n_train, split = len(files) - n_test, f"llffhold={cfg.llffhold}"
```

Khi scene không thuộc dạng cuộc thi (không có `test/test_poses.csv`), số ảnh test được suy ra theo chu kỳ lấy mẫu `llffhold` (chia lấy phần nguyên, floor division, dòng 277–278):

$$
n_{test} = \begin{cases} \left\lfloor \dfrac{n_{files}}{\text{llffhold}} \right\rfloor & \text{llffhold} \ne 0 \\ 0 & \text{llffhold} = 0 \end{cases}, \qquad n_{train} = n_{files} - n_{test}
$$

Nếu có `test_poses.csv` (layout cuộc thi, dòng 273–275): toàn bộ ảnh hiện có đều là train, và $n_{test}$ = số dòng trong CSV:

$$
n_{train} = n_{files}, \qquad n_{test} = |\text{rows}(\text{test\_poses.csv})|
$$

## 3. Kích thước ảnh sau downscale khi train (`train_px`)

Trích nguyên văn (`pipeline/data.py`, dòng 288):

```python
            train_px=f"{width // cfg.resolution}x{height // cfg.resolution}",
```

$$
W_{train} = \left\lfloor \frac{W}{\text{resolution}} \right\rfloor, \qquad H_{train} = \left\lfloor \frac{H}{\text{resolution}} \right\rfloor
$$

(phép chia nguyên `//` trong Python, tương đương `floor`)

## 4. Tổng dung lượng dữ liệu (byte → megabyte)

Trích nguyên văn (`pipeline/data.py`, dòng 267 và 289):

```python
        n_bytes = sum(os.path.getsize(os.path.join(image_root, f)) for f in files)
```

```python
            size_mb=round(n_bytes / 1024 ** 2, 1),
```

$$
\text{size\_mb} = \frac{\sum_f \text{filesize}(f)}{1024^2}, \quad \text{làm tròn 1 chữ số thập phân}
$$

## 5. Độ sâu thư mục khi duyệt scene (`find_scenes`)

Trích nguyên văn (`pipeline/data.py`, dòng 212–216):

```python
    for current, dirs, _ in os.walk(root):
        depth = current[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
            continue
```

Độ sâu thư mục hiện tại so với `root` được tính bằng số dấu phân cách đường dẫn (`os.sep`) còn lại sau khi cắt bỏ tiền tố `root` khỏi `current`:

$$
\text{depth}(\text{current}) = \big|\{\, c \in \text{current}[\,|root|:\,] : c = \text{os.sep} \,\}\big|
$$

Việc duyệt bị cắt (`dirs[:] = []`) ngay khi $\text{depth}\ge\text{max\_depth}$, giới hạn độ sâu tìm kiếm scene.

---

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc | Công thức tương ứng |
|---|---|
| `expected = {key: int(value) for key, value in re.findall(...)}` | $\text{expected}[k]$ trích từ README |
| `counts[key] = len([...])` trong `_count()` | $\text{actual}[k] = \lvert \{f : \dots\} \rvert$ |
| `all(actual.get(k, 0) == v for k, v in expected.items())` | $\bigwedge_k \text{actual}[k] = \text{expected}[k]$ |
| `for _ in range(4): time.sleep(2); actual = _count()` | retry tối đa $4$ lần, $2$s/lần |
| `n_test = len(files) // cfg.llffhold if cfg.llffhold else 0` | $n_{test} = \lfloor n_{files}/\text{llffhold} \rfloor$ nếu llffhold≠0, else $0$ |
| `n_train, split = len(files) - n_test, ...` | $n_{train} = n_{files} - n_{test}$ |
| `n_train, split = len(files), "test_poses.csv"` | $n_{train} = n_{files}$ (khi có CSV test) |
| `n_test = len(testposes.read_rows(csv_path))` | $n_{test} = \lvert \text{rows(csv)} \rvert$ |
| `f"{width // cfg.resolution}x{height // cfg.resolution}"` | $W_{train} = \lfloor W/r \rfloor,\ H_{train} = \lfloor H/r \rfloor$ |
| `n_bytes = sum(os.path.getsize(...) for f in files)` | $\sum_f \text{filesize}(f)$ |
| `round(n_bytes / 1024 ** 2, 1)` | $\text{size\_mb} = n_{bytes}/1024^2$, làm tròn $1$ chữ số |
| `depth = current[len(root):].count(os.sep)` (trong `find_scenes`) | đếm độ sâu thư mục $= $ số dấu phân cách đường dẫn kể từ `root` |
