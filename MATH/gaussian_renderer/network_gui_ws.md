# Công thức toán học trong `gaussian_renderer/network_gui_ws.py`

**Nhận định chung**: file này là một máy chủ WebSocket (dùng `websockets` + `asyncio`, chạy trên thread riêng) để stream kết quả render mới nhất (`latest_result`) tới client GUI theo yêu cầu (nhận một ID dạng số nguyên, trả về header + dữ liệu ảnh). Toàn bộ nội dung là hạ tầng mạng/bất đồng bộ (event loop, encode/decode nhị phân) — **không có bất kỳ phép toán hình học, chiếu phối cảnh hay alpha compositing nào** liên quan đến Gaussian Splatting trong file này. Phần duy nhất gần với "công thức" là cách mã hoá số nguyên và đóng gói header nhị phân.

---

## 1. Giải mã ID yêu cầu từ client (`echo`)

Trích nguyên văn (`gaussian_renderer/network_gui_ws.py`, dòng 35–36):

```python
                value = int.from_bytes(message, byteorder='big', signed=True)
                curr_id = value
```

Chuyển chuỗi byte nhận được (big-endian, có dấu, bù hai) thành số nguyên:

$$
\text{value} = \left(\sum_{k=0}^{n-1} b_k \cdot 256^{\,n-1-k}\right) - 256^{n}\cdot[\text{bit dấu}=1]
$$

tức là: cộng dồn các byte theo trọng số $256^{n-1-k}$ (big-endian, không dấu) như bình thường, rồi nếu bit dấu (bit cao nhất của $b_0$) bằng 1 thì trừ đi $256^n$ (chứ không phải trừ riêng số hạng $b_0\cdot256^{n-1}$) — đây chính là định nghĩa bù hai (two's complement) nhiều byte, với $n = $ `len(message)` byte.

**Kiểm chứng bằng phản ví dụ** ($n=1$, $b_0=0xFF=255$): công thức đúng cho $255 - 256 = -1$, khớp với `int.from_bytes(b'\xff', 'big', signed=True) == -1` trong Python thật. (Công thức cũ $-b_0\cdot256^{n-1}\cdot[\text{dấu}=1]+\sum b_k 256^{n-1-k}$ cho $-255\cdot1\cdot1+255\cdot1=0$, sai.)

## 2. Đóng gói header nhị phân (`struct.pack`)

Trích nguyên văn (`gaussian_renderer/network_gui_ws.py`, dòng 38–41):

```python
                header = struct.pack('ii', latest_width, latest_height)  # Pack the two integers (height and width)

                # Send the entire tensor as one WebSocket message
                await websocket.send(header + latest_result)
```

Định dạng `'ii'` đóng gói 2 số nguyên 32-bit (little-endian theo mặc định hệ thống) liên tiếp thành $8$ byte:

$$
\text{header} = \mathrm{enc}_{32}(\text{latest\_width}) \,\Vert\, \mathrm{enc}_{32}(\text{latest\_height})
$$

với $\Vert$ là phép nối byte, $\mathrm{enc}_{32}$ là hàm mã hoá số nguyên 32-bit. Dữ liệu ảnh (`latest_result`, một buffer byte được gán từ bên ngoài file này — không khởi tạo giá trị trong module) được nối ngay sau header.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `int.from_bytes(message, byteorder='big', signed=True)` | giải mã bù hai big-endian: $\text{value} = \sum_{k} b_k\cdot 256^{n-1-k}$ (có bù dấu) |
| `struct.pack('ii', latest_width, latest_height)` | $\text{header} = \mathrm{enc}_{32}(\text{width}) \Vert \mathrm{enc}_{32}(\text{height})$ |
| `header + latest_result` | $\text{payload} = \text{header} \Vert \text{latest\_result}$ |
