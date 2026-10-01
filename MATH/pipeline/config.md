# Công thức toán học trong `pipeline/config.py`

**Nhận xét:** Đây là file cấu hình (`dataclass Config`) cho pipeline huấn luyện trên Colab. Phần lớn là tham số mặc định (đường dẫn, cờ bật/tắt); công thức toán học chỉ xuất hiện ở chỗ các tham số huấn luyện 3DGS được **co giãn theo số vòng lặp thực tế** (`iterations`) để tương thích với lịch densify gốc vốn thiết kế cho 30.000 vòng.

**Sửa so với bản trước:** `config.py` tự nó chỉ lưu các hệ số tỉ lệ ($f_{densify}, f_{reset}$) dưới dạng field của dataclass — phép nhân/round/floor thật sự nằm ở `pipeline/trainer.py::build_args`. Bản cũ viết công thức là phép nhân đơn thuần ($f\times T$), thiếu $\max(1,\cdot)$ và $\mathrm{round}(\cdot)$ mà code thật sự dùng; đã sửa lại ở mục 1 và 2 bên dưới cho khớp chính xác với `trainer.py` dòng 29 và 33.

## 1. Co giãn mốc dừng densify theo số vòng huấn luyện

3DGS gốc đặt mốc dừng densification cố định cho lịch 30k iterations. `pipeline/config.py` chỉ khai báo **hệ số tỉ lệ** `densify_until_frac` (mặc định $0.5$) cùng dòng comment giải thích (dòng 43–47):

```python
# Lịch densify của 3DGS gốc được đặt cho 30k vòng. Chạy ngắn hơn mà giữ
# nguyên nó thì lần reset opacity cuối rơi quá sát vòng kết thúc và model
# không kịp hồi phục. Co lịch theo số vòng thực tế:
#   densify_until_iter = densify_until_frac * iterations
densify_until_frac: float = 0.5
```

Công thức này **không được tính trong `config.py`** — nó chỉ lưu hằng số $f_{densify}=0.5$; phép tính thật nằm ở nơi tiêu thụ `cfg`, cụ thể `pipeline/trainer.py` hàm `build_args`, dòng 26–29:

```python
n_iter = int(iterations or cfg.iterations)
# Densify (và các lần reset opacity bên trong nó) phải kết thúc sớm hơn vòng
# cuối, nếu không model dừng ngay sau một lần reset và PSNR sụp.
densify_until = max(1, int(round(getattr(cfg, "densify_until_frac", 0.5) * n_iter)))
```

Tức là, khớp chính xác với code (có `round` và sàn tối thiểu $1$, điều bản phân tích trước đây bỏ sót):

$$
\text{densify\_until\_iter} = \max\Big(1,\ \mathrm{round}\big(\text{densify\_until\_frac} \times \text{iterations}\big)\Big)
$$

### Tại sao không giữ nguyên mốc gốc?

Nếu giữ nguyên số vòng tuyệt đối của lịch 30k mà chạy ngắn hơn, lần reset opacity cuối rơi quá sát vòng kết thúc, model không kịp hồi phục trước khi dừng.

## 2. Co giãn chu kỳ reset opacity theo số vòng huấn luyện

`opacity_reset_interval` gốc là 3000, tương ứng $3000/30000 = 10\%$ của lịch 30k. `config.py` khai báo hệ số `opacity_reset_frac` (mặc định $0.2$, tức 20%) kèm comment (dòng 48–54):

```python
# opacity_reset_interval gốc (3000) được đặt cho lịch 30k iterations (= 10%).
# Trước đây KHÔNG co giãn theo iterations như densify_until_iter -> ở lịch
# ngắn (vd 7000 iter) reset rơi vào 43% tiến trình thay vì rải đều, gây sụp
# PSNR giữa chừng (quan sát thật trong output/sadgs_models/history.csv,
# scene HCM0539: PSNR 22->5.9 tại iter 3000/7000). Co theo cùng nguyên tắc:
#   opacity_reset_interval = opacity_reset_frac * iterations
opacity_reset_frac: float = 0.2
```

và được tính thật trong `pipeline/trainer.py`, `build_args`, dòng 30–33:

```python
# opacity_reset_interval gốc (3000, mặc định của OptimizationParams) được đặt cho
# lịch 30k -- co giãn theo cùng nguyên tắc với densify_until_iter ở trên, nếu không
# reset rơi lệch tỷ lệ và gây sụp PSNR giữa chừng (xem pipeline/config.py).
opacity_reset = max(1, int(round(getattr(cfg, "opacity_reset_frac", 0.2) * n_iter)))
```

$$
\text{opacity\_reset\_interval} = \max\Big(1,\ \mathrm{round}\big(\text{opacity\_reset\_frac} \times \text{iterations}\big)\Big)
$$

### Lý do thay đổi (ghi trong comment của code)

Trước đây hằng số 3000 không co giãn theo `iterations`: ở lịch ngắn (ví dụ 7000 iter), lần reset rơi vào $3000/7000 \approx 43\%$ tiến trình thay vì rải đều, gây sụp PSNR giữa chừng (quan sát thực tế: PSNR giảm từ $22 \to 5.9$ tại iter 3000/7000 trên scene HCM0539). Công thức co giãn theo tỉ lệ `opacity_reset_frac` khắc phục việc này bằng cách giữ **vị trí tương đối** (phần trăm tiến trình) của lần reset cố định bất kể tổng số vòng lặp.

**Lưu ý thêm (suy ra từ `pipeline/trainer.py` dòng 134–136, không nằm trong `config.py` nhưng cùng nguyên tắc co giãn):** các mốc prune (vốn là $4000/8000$ trên lịch gốc 30k) cũng được co theo tỉ lệ của `densify_until_iter` đã tính ở trên, chứ không theo `iterations` trực tiếp:

```python
# 3DGS gốc dùng 4000/8000 trên lịch 30k -> co theo cùng tỷ lệ với densify_until_iter.
prune_iterations = {max(1, round(0.267 * opt.densify_until_iter)),
                    max(1, round(0.533 * opt.densify_until_iter))}
```

$$
\text{prune\_iterations} = \Big\{\max(1,\mathrm{round}(0.267\cdot\text{densify\_until\_iter})),\ \max(1,\mathrm{round}(0.533\cdot\text{densify\_until\_iter}))\Big\}
$$

(hệ số $0.267\approx4000/15000$ và $0.533\approx8000/15000$ — tỉ lệ so với $\text{densify\_until\_iter}$ gốc $=15000$ của 3DGS, chứ không phải so với $30000$.)

## 3. Lấy mẫu con hold-out (gián tiếp, dùng ở nơi khác qua `llffhold`)

Tham số `llffhold = 8` không tính trực tiếp trong file này nhưng định nghĩa quy tắc lấy mẫu test set theo chu kỳ, thực thi tại `scene/dataset_readers.py`, hàm `readColmapSceneInfo`, dòng 198–200:

```python
if eval:
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
```

$$
\text{is\_test\_view}(i) = \big(i \bmod \text{llffhold} = 0\big)
$$

## 4. Các hàm tiện ích đường dẫn (không mang ý nghĩa toán học)

`model_path`, `scene_dir`, `resolved_scene_root`, `submission_scene_dir` chỉ là nối chuỗi đường dẫn (`os.path.join`) — không phải công thức số học.

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc | Công thức tương ứng |
|---|---|
| `densify_until_frac: float = 0.5` (`config.py` dòng 47) | $f_{densify} = 0.5$ — hệ số tỉ lệ |
| `densify_until = max(1, int(round(cfg.densify_until_frac * n_iter)))` (`trainer.py` dòng 29) | $\text{densify\_until\_iter} = \max(1,\mathrm{round}(f_{densify}\cdot T))$, với $T=$ `iterations` |
| `opacity_reset_frac: float = 0.2` (`config.py` dòng 54) | $f_{reset} = 0.2$ — hệ số tỉ lệ |
| `opacity_reset = max(1, int(round(cfg.opacity_reset_frac * n_iter)))` (`trainer.py` dòng 33) | $\text{opacity\_reset\_interval} = \max(1,\mathrm{round}(f_{reset}\cdot T))$ |
| `llffhold: int = 8` (áp dụng ở `dataset_readers.py`) | chu kỳ lấy mẫu: $i \bmod 8 = 0 \Rightarrow$ ảnh test |
| `mult: float = 0.5` | hệ số nhân dùng trong chấm điểm (`pipeline/score.py`), không tính ở file này |
| `psnr_max: float = 30.0` | ngưỡng chuẩn hoá PSNR dùng trong chấm điểm (`pipeline/score.py`), không tính ở file này |
| `resolution: int = 2` | hệ số downscale ảnh: $W' = W/2,\ H' = H/2$ (áp dụng khi train, không tính ở file này) |
| `score_every`, `save_every`, `iterations` | các mốc nguyên số vòng lặp, không phải công thức mà là tham số chu kỳ |
| `as_dict()` → `asdict(self)` | — (chuyển dataclass → dict, không phải công thức) |
