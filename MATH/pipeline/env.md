# Công thức toán học trong `pipeline/env.py`

**Nhận xét:** File kiểm tra môi trường (GPU/CUDA/RAM), cài đặt phụ thuộc, và quản lý bộ nhớ. Công thức toán học duy nhất là chuyển đổi đơn vị dung lượng bộ nhớ (byte → gigabyte); phần còn lại (cài đặt, build submodule, dọn bộ nhớ) là logic hệ thống, không phải công thức số học.

## 1. Chuyển đổi byte sang gigabyte (`_gb`)

Trích nguyên văn (`pipeline/env.py`, dòng 19–20):

```python
def _gb(value):
    return value / (1024 ** 3)
```

$$
\text{GB}(x) = \frac{x}{1024^3}
$$

Áp dụng cho mọi số đo bộ nhớ, trích từ hàm `mem()` (dòng 27–32):

```python
    vm = psutil.virtual_memory()
    usage = dict(ram_used_gb=_gb(vm.used), ram_total_gb=_gb(vm.total), ram_pct=vm.percent)
    if torch.cuda.is_available():
        usage["vram_alloc_gb"] = _gb(torch.cuda.memory_allocated())
        usage["vram_reserved_gb"] = _gb(torch.cuda.memory_reserved())
        usage["vram_total_gb"] = _gb(torch.cuda.get_device_properties(0).total_memory)
```

tức RAM đã dùng/tổng (`psutil.virtual_memory`), VRAM đã cấp phát/giữ trước/tổng (`torch.cuda.memory_allocated/reserved/get_device_properties`) đều đi qua cùng công thức $\text{GB}(x)$ ở trên. Cùng công thức này được dùng lại trong `check_gpu` (dòng 68):

```python
        info["vram_total_gb"] = round(_gb(torch.cuda.get_device_properties(0).total_memory), 2)
```

## 2. Phần trăm RAM đã dùng

Lấy trực tiếp từ `psutil.virtual_memory().percent` (dòng 28, trường `ram_pct=vm.percent` ở trích dẫn trên — không tính lại trong file này), nhưng về bản chất `psutil` tính:

$$
\text{ram\_pct} = \frac{\text{ram\_used}}{\text{ram\_total}} \times 100\%
$$

## 3. Giới hạn song song khi build CUDA (`MAX_JOBS`)

Trích nguyên văn (`pipeline/env.py`, dòng 119):

```python
    os.environ.setdefault("MAX_JOBS", "2")
```

Không phải công thức toán mà là một hằng số cấu hình hệ thống: `MAX_JOBS = 2` giới hạn số tiến trình biên dịch `.cu` song song để tránh nvcc/cc1plus bị OOM-killed trên RAM giới hạn của Colab. Không có biểu thức tính toán.

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc | Công thức tương ứng |
|---|---|
| `def _gb(value): return value / (1024 ** 3)` | $\text{GB}(x) = x/1024^3$ |
| `usage = dict(ram_used_gb=_gb(vm.used), ram_total_gb=_gb(vm.total), ram_pct=vm.percent)` | $\text{ram\_used\_gb}=\text{GB}(\text{used})$, $\text{ram\_total\_gb}=\text{GB}(\text{total})$, $\text{ram\_pct}=\dfrac{\text{used}}{\text{total}}\times100\%$ |
| `usage["vram_alloc_gb"] = _gb(torch.cuda.memory_allocated())` | $\text{vram\_alloc\_gb} = \text{GB}(\text{bytes\_allocated})$ |
| `usage["vram_reserved_gb"] = _gb(torch.cuda.memory_reserved())` | $\text{vram\_reserved\_gb} = \text{GB}(\text{bytes\_reserved})$ |
| `usage["vram_total_gb"] = _gb(torch.cuda.get_device_properties(0).total_memory)` | $\text{vram\_total\_gb} = \text{GB}(\text{bytes\_total})$ |
| `round(_gb(torch.cuda.get_device_properties(0).total_memory), 2)` (trong `check_gpu`) | $\text{vram\_total\_gb} = \text{GB}(\text{bytes\_total})$, làm tròn $2$ chữ số |
| `os.environ.setdefault("MAX_JOBS", "2")` | hằng số cấu hình (không phải công thức toán) |
| `open(flag_path, "w").close()` | — (ghi cờ đánh dấu, không phải công thức) |
