# Công thức toán học trong `pipeline/__init__.py`

**Nhận xét:** File này chỉ là lớp re-export (gom `Config` và các hàm của `pipeline.env` thành API công khai của package `pipeline`). Không có công thức toán học nào — đây thuần tuý là định tuyến import. Toàn văn file (9 dòng), trích để đối chiếu đúng số dòng với bảng bên dưới:

```python
"""Pipeline SADGS: kiểm tra máy -> dữ liệu -> train -> submission -> báo cáo -> tải về.

Notebook chỉ gọi các hàm ở đây; mọi logic nằm trong các module .py này.
"""

from pipeline.config import Config
from pipeline.env import check_gpu, free_memory, install_dependencies, mem, show_mem

__all__ = ["Config", "check_gpu", "free_memory", "install_dependencies", "mem", "show_mem"]
```

Không có bước tính toán, điều kiện, vòng lặp hay biến đổi dữ liệu nào trong file — chỉ 2 câu lệnh `import` (dòng 6–7) và 1 khai báo `__all__` (dòng 9). Do đó không có mục nào cần "kiểm chứng tính đúng sai" hay "ví dụ số" theo nghĩa toán học.

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc (số dòng) | Công thức tương ứng |
|---|---|
| `from pipeline.config import Config` (dòng 6) | — (không có công thức, import mô-đun) |
| `from pipeline.env import check_gpu, free_memory, install_dependencies, mem, show_mem` (dòng 7) | — (không có công thức, import mô-đun) |
| `__all__ = [...]` (dòng 9) | — (khai báo API công khai, không phải công thức) |
