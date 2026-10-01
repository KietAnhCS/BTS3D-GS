# Công thức toán học trong `gaussian_renderer/network_gui.py`

**Nhận định chung**: file này là lớp giao tiếp mạng (TCP socket, blocking-free) phục vụ GUI debug tương tác (kiểu SIBR viewer) — gửi/nhận khung hình render qua socket, không chứa bất kỳ phép toán hình học/densification nào của Gaussian Splatting. Phần lớn nội dung là I/O: bind/listen/accept socket, đọc độ dài message rồi giải mã JSON. Điểm toán học duy nhất đáng ghi nhận là **phép đảo dấu 2 cột của ma trận camera** nhận từ client GUI, dùng để đổi quy ước hệ toạ độ.

---

## 1. Giao thức độ dài message (`read`)

Trích nguyên văn (`gaussian_renderer/network_gui.py`, dòng 43–48):

```python
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))
```

Không phải công thức toán học mà là một giao thức khung dữ liệu (length-prefixed framing): 4 byte đầu (little-endian unsigned int) cho biết độ dài $L$ (byte) của payload JSON theo sau:

$$
L = \sum_{k=0}^{3} b_k \cdot 256^{k}
$$

với $b_k$ là byte thứ $k$ nhận được từ `conn.recv(4)`.

---

## 2. Đảo dấu cột của ma trận view/projection (`receive`)

Trích nguyên văn (`gaussian_renderer/network_gui.py`, dòng 74–78):

```python
world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
world_view_transform[:,1] = -world_view_transform[:,1]
world_view_transform[:,2] = -world_view_transform[:,2]
full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
full_proj_transform[:,1] = -full_proj_transform[:,1]
```

Ma trận $4\times4$ nhận từ GUI (quy ước toạ độ của engine hiển thị, ví dụ Unity/trục trái) được chuyển đổi sang quy ước hệ toạ độ dùng trong rasterizer (trục phải, kiểu OpenGL) bằng cách đảo dấu cột thứ 2 và cột thứ 3 (chỉ số 0-based: cột $y$ và cột $z$):

$$
V' = V \cdot D, \qquad D = \mathrm{diag}(1,\,-1,\,-1,\,1)
$$

(áp dụng cho `world_view_transform`, nhân cột tương đương phép đổi dấu trục $y,z$ của hệ camera)

$$
P' = P \cdot D_{xy}, \qquad D_{xy} = \mathrm{diag}(1,\,-1,\,1,\,1)
$$

(áp dụng cho `full_proj_transform`, chỉ đảo dấu cột $y$ — tương ứng lật trục tung của không gian clip, thường do khác biệt quy ước $v$ hướng lên/xuống giữa GUI và rasterizer).

Lưu ý: code không nhân ma trận tường minh $V\cdot D$ mà gán trực tiếp `[:,1] = -[:,1]`, về mặt đại số tương đương với nhân phải ma trận đường chéo $D$ ở trên vì $D$ chỉ đổi dấu cột tương ứng.

---

## 3. Các tham số camera khác (không biến đổi số học)

Trích nguyên văn (`gaussian_renderer/network_gui.py`, dòng 66–73, 79):

```python
fovy = message["fov_y"]
fovx = message["fov_x"]
znear = message["z_near"]
zfar = message["z_far"]
do_shs_python = bool(message["shs_python"])
do_rot_scale_python = bool(message["rot_scale_python"])
keep_alive = bool(message["keep_alive"])
scaling_modifier = message["scaling_modifier"]
...
custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
```

Được truyền thẳng (không qua biến đổi công thức nào trong file này) vào `MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)` — các phép dựng ma trận chiếu từ `fovy, fovx, znear, zfar` (nếu có) nằm trong `scene/cameras.py`, không thuộc phạm vi file này.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `messageLength = int.from_bytes(conn.recv(4), 'little')` | $L = \sum_{k=0}^{3} b_k \cdot 256^k$ |
| `world_view_transform[:,1] = -world_view_transform[:,1]` | $V'=V\cdot D,\ D=\mathrm{diag}(1,-1,-1,1)$ (cột $y$) |
| `world_view_transform[:,2] = -world_view_transform[:,2]` | $V'=V\cdot D,\ D=\mathrm{diag}(1,-1,-1,1)$ (cột $z$) |
| `full_proj_transform[:,1] = -full_proj_transform[:,1]` | $P'=P\cdot D_{xy},\ D_{xy}=\mathrm{diag}(1,-1,1,1)$ |
| `fovy, fovx, znear, zfar, scaling_modifier` (truyền thẳng) | không biến đổi — tham số đầu vào thô cho `MiniCam` |
