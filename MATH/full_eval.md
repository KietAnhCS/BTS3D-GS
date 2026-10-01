# Công thức toán học trong `full_eval.py`

## Nhận định chung

`full_eval.py` **không chứa công thức toán học nào của riêng nó**. Đây thuần tuý là một script điều phối pipeline (orchestration): nó lặp qua danh sách scene của 3 bộ dữ liệu chuẩn (MipNeRF360, Tanks&Temples, Deep Blending), rồi với mỗi scene phát sinh một chuỗi lệnh `os.system(...)` gọi lần lượt `train.py` → `render.py` → `metrics.py` dưới dạng subprocess. Mọi phép toán (loss, densify, PSNR/SSIM/LPIPS...) nằm trong các script được gọi, không nằm trong `full_eval.py`.

Phần duy nhất đáng ghi lại là các **bộ tham số/hyperparameter theo từng scene** (budget ngân sách số điểm Gaussian) và logic rẽ nhánh chế độ `big` / `budget`.

---

## 1. Logic điều phối

Với mỗi scene $s$ thuộc 4 nhóm:

- `mipnerf360_outdoor_scenes` = {bicycle, flowers, garden, stump, treehill}
- `mipnerf360_indoor_scenes` = {room, counter, kitchen, bonsai}
- `tanks_and_temples_scenes` = {truck, train}
- `deep_blending_scenes` = {drjohnson, playroom}

Trích nguyên văn cách dựng lệnh train cho nhóm outdoor, chế độ "big" (`full_eval.py`, dòng 89–96):

```python
if args.mode == "big":
    mode_param = " --densification_interval 100 --mode final_count"
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        budget_param = " --budget {} ".format(big_budgets[scene])
        CMD = "python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
        run_cmd(CMD, args)
```

rồi tương tự `render.py -m <output_path>` (dòng 157–160) và `metrics.py -m <output_path>` (dòng 170–173) cho từng scene, có `--dry_run` để chỉ in lệnh mà không thực thi (`run_cmd`, dòng 77–80):

```python
def run_cmd(CMD, args):
    print(CMD)
    if not args.dry_run:
        os.system(CMD)
```

Thời gian mỗi nhóm dataset được đo bằng `time.time()` trước/sau vòng lặp train và ghi ra `timing.txt`, ví dụ cho nhóm MipNeRF360 (`full_eval.py`, dòng 91 và 102):

```python
start_time = time.time()
# ... vòng lặp train tất cả scene outdoor + indoor của MipNeRF360 ...
m360_timing = (time.time() - start_time)/60.0
```

(cùng khuôn mẫu lặp lại cho `tandt_timing`, dòng 104 và 110, và `db_timing`, dòng 112 và 118):

$$
t_{group} = \frac{t_{end} - t_{start}}{60}\ \text{(phút)}
$$

---

## 2. Bộ tham số (hyperparameter sets) theo từng scene

### 2.1. Chế độ "big" — `densification_interval=100`, `mode=final_count`, dùng `big_budgets[scene]` làm `--budget` (số điểm Gaussian mục tiêu cuối cùng):

Trích nguyên văn dict `big_budgets` (`full_eval.py`, dòng 21–35):

```python
big_budgets = {
    "bicycle": 5987095,
    "flowers": 3618411,
    "garden": 5728191,
    "stump": 4867429,
    "treehill": 3770257,
    "room": 1548960,
    "counter": 1190919,
    "kitchen": 1803735,
    "bonsai": 1252367,
    "truck": 2584171,
    "train": 1085480,
    "playroom": 2326100,
    "drjohnson": 3273600
}
```

| Scene | Budget (số điểm) |
|---|---|
| bicycle | 5,987,095 |
| flowers | 3,618,411 |
| garden | 5,728,191 |
| stump | 4,867,429 |
| treehill | 3,770,257 |
| room | 1,548,960 |
| counter | 1,190,919 |
| kitchen | 1,803,735 |
| bonsai | 1,252,367 |
| truck | 2,584,171 |
| train | 1,085,480 |
| playroom | 2,326,100 |
| drjohnson | 3,273,600 |

### 2.2. Chế độ "budget" — `densification_interval=500`, `mode=multiplier`, dùng `budget_multipliers[scene]` làm hệ số nhân:

Trích nguyên văn dict `budget_multipliers` (`full_eval.py`, dòng 38–52) và cách dựng lệnh chế độ budget cho nhóm outdoor (dòng 120–127):

```python
budget_multipliers = {
    "bicycle": 15,
    "flowers": 15,
    "garden": 15,
    "stump": 15,
    "treehill": 15,
    "room": 2,
    "counter": 2,
    "kitchen": 2,
    "bonsai": 2,
    "truck": 2,
    "train": 2,
    "playroom": 5,
    "drjohnson": 5
}
```

```python
elif args.mode == "budget":
    mode_param = " --densification_interval 500 --mode multiplier"
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        budget_param = " --budget {} ".format(budget_multipliers[scene])
        CMD = "python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + f"{scene}_budget" + common_args + budget_param + mode_param
        run_cmd(CMD, args)
```

| Scene | Multiplier |
|---|---|
| bicycle, flowers, garden, stump, treehill | 15 |
| room, counter, kitchen, bonsai, truck, train | 2 |
| playroom, drjohnson | 5 |

(ý nghĩa toán học của "multiplier" — ví dụ nhân với số điểm khởi tạo ban đầu để ra ngân sách cuối — được quyết định bên trong `train.py`/`scene` khi xử lý `--mode multiplier`, không nằm trong chính `full_eval.py`.)

### 2.3. Độ phân giải ảnh theo nhóm

Trích nguyên văn: nhóm outdoor dùng `images_4` (`full_eval.py`, dòng 95), nhóm indoor dùng `images_2` (dòng 100), Tanks&Temples / Deep Blending không truyền cờ `-i` nên dùng ảnh gốc (dòng 108, 116):

```python
CMD = "python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
```
```python
CMD = "python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
```
```python
CMD = "python train.py -s " + source + " -m " + args.output_path + "/" + f"{scene}_big" + common_args + budget_param + mode_param
```

$$
\text{images\_4 cho outdoor},\qquad \text{images\_2 cho indoor (MipNeRF360)},\qquad \text{gốc cho Tanks\&Temples / Deep Blending}
$$

tương ứng hệ số downsample $1/4$ và $1/2$ (xem thêm `convert.md` §1 cho ý nghĩa resize).

---

## Bảng hằng số/ngưỡng

| Ký hiệu | Giá trị | Vị trí |
|---|---|---|
| `densification_interval` (big) | 100 | tham số truyền cho `train.py` |
| `densification_interval` (budget) | 500 | tham số truyền cho `train.py` |
| `big_budgets[scene]` | xem bảng 2.1 | ngân sách số điểm Gaussian cuối cùng mỗi scene |
| `budget_multipliers[scene]` | xem bảng 2.2 | hệ số nhân ngân sách mỗi scene |
| downsample ratio outdoor (M360) | $1/4$ (`images_4`) | độ phân giải train |
| downsample ratio indoor (M360) | $1/2$ (`images_2`) | độ phân giải train |

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức / ý nghĩa |
|---|---|
| `budget_param = " --budget {} ".format(big_budgets[scene])` | $B(s) = \texttt{big\_budgets}[s]$, truyền làm ngân sách số điểm Gaussian cuối cùng |
| `budget_param = " --budget {} ".format(budget_multipliers[scene])` | $m(s) = \texttt{budget\_multipliers}[s]$, hệ số nhân ngân sách |
| `m360_timing = (time.time() - start_time)/60.0` | $t_{m360} = (t_{end}-t_{start})/60$ (phút) |
| `tandt_timing = (time.time() - start_time)/60.0` | $t_{tandt} = (t_{end}-t_{start})/60$ |
| `db_timing = (time.time() - start_time)/60.0` | $t_{db} = (t_{end}-t_{start})/60$ |
| `mode_param = " --densification_interval 100 --mode final_count"` | cấu hình chế độ "big": chu kỳ densify $=100$, dừng theo số điểm cuối |
| `mode_param = " --densification_interval 500 --mode multiplier"` | cấu hình chế độ "budget": chu kỳ densify $=500$, ngân sách theo hệ số nhân |
