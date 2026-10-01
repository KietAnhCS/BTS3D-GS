# Công thức toán học trong `scene/colmap_loader.py`

File này đọc dữ liệu COLMAP (camera, pose ảnh, point cloud thưa) ở hai định dạng text và binary. Phần lớn là parsing I/O thuần tuý (không có công thức toán); hai công thức toán học thật sự nằm ở `qvec2rotmat` (quaternion → ma trận xoay) và `rotmat2qvec` (ma trận xoay → quaternion, qua eigen-decomposition).

---

## 1. Ký hiệu

- $q=(q_0,q_1,q_2,q_3)=(w,x,y,z)$: quaternion đơn vị biểu diễn phép xoay (`qvec`, quy ước COLMAP: phần tử đầu là phần thực $w$).
- $R\in SO(3)$: ma trận xoay $3\times3$ tương ứng.
- $K$: ma trận đối xứng $4\times4$ dùng trong phương pháp Shepperd/Markley để tìm quaternion từ $R$ qua eigenvector.

## 2. Quaternion → ma trận xoay (`qvec2rotmat`)

Trích nguyên văn (`scene/colmap_loader.py`, dòng 43–53):

```python
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
```

Với $q=(q_0,q_1,q_2,q_3)=(w,x,y,z)$, công thức tiêu chuẩn chuyển quaternion đơn vị thành ma trận xoay:

$$
R(q) =
\begin{bmatrix}
1-2q_2^2-2q_3^2 & 2q_1q_2-2q_0q_3 & 2q_3q_1+2q_0q_2 \\
2q_1q_2+2q_0q_3 & 1-2q_1^2-2q_3^2 & 2q_2q_3-2q_0q_1 \\
2q_3q_1-2q_0q_2 & 2q_2q_3+2q_0q_1 & 1-2q_1^2-2q_2^2
\end{bmatrix}
$$

Đây đúng là công thức Rodrigues dạng quaternion chuẩn $R=I+2w[\mathbf v]_\times+2[\mathbf v]_\times^2$ với $\mathbf v=(q_1,q_2,q_3)$, viết tường minh theo từng phần tử ma trận — khớp quy ước hàng/cột phổ biến dùng trong COLMAP's `read_write_model.py` gốc (COLMAP lưu `qvec` dạng $(w,x,y,z)$).

## 3. Ma trận xoay → quaternion (`rotmat2qvec`)

Trích nguyên văn (`scene/colmap_loader.py`, dòng 55–66):

```python
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
```

Đây là **phương pháp Bar-Itzhack / Shepperd-Markley**: xây ma trận đối xứng $4\times4$

$$
K = \frac{1}{3}
\begin{bmatrix}
R_{xx}-R_{yy}-R_{zz} & R_{yx}+R_{xy} & R_{zx}+R_{xz} & R_{yz}-R_{zy} \\
R_{yx}+R_{xy} & R_{yy}-R_{xx}-R_{zz} & R_{zy}+R_{yz} & R_{zx}-R_{xz} \\
R_{zx}+R_{xz} & R_{zy}+R_{yz} & R_{zz}-R_{xx}-R_{yy} & R_{xy}-R_{yx} \\
R_{yz}-R_{zy} & R_{zx}-R_{xz} & R_{xy}-R_{yx} & R_{xx}+R_{yy}+R_{zz}
\end{bmatrix}
$$

(lưu ý: dòng `K = np.array([...]) / 3.0` chỉ chia toàn ma trận cho 3 — nửa tam giác trên của mảng Python thực chất bằng 0 theo code, nhưng `np.linalg.eigh` chỉ đọc nửa tam giác dưới (hoặc trên, tuỳ `UPLO`, mặc định `'L'` — tam giác dưới) của ma trận đối xứng đầu vào nên việc nửa trên bằng 0 không ảnh hưởng kết quả, vì $K$ vốn đối xứng về mặt lý thuyết).

Quaternion ứng với eigenvector có trị riêng lớn nhất của $K$:

$$
(\lambda^*, \mathbf{e}^*) = \arg\max_{\lambda_i} \lambda_i,\qquad q = \big(e^*_{3},\,e^*_{0},\,e^*_{1},\,e^*_{2}\big)
$$

tức lấy eigenvector ứng với trị riêng lớn nhất (`np.argmax(eigvals)`, `eigh` trả trị riêng tăng dần nên đây luôn là cột cuối), rồi **hoán vị chỉ số** `[3, 0, 1, 2]` để đưa thành phần thứ 4 của eigenvector (ứng với "phần thực" trong xây dựng $K$ theo quy ước này) lên vị trí đầu — khớp quy ước `qvec=(w,x,y,z)` dùng trong `qvec2rotmat`. Cuối cùng chuẩn hoá dấu để phần thực không âm:

$$
q \leftarrow \begin{cases} -q & \text{nếu } q_0<0 \\ q & \text{ngược lại}\end{cases}
$$

(do $q$ và $-q$ biểu diễn cùng một phép xoay, quy ước chọn $q_0\ge0$ để có biểu diễn duy nhất).

**Kiểm chứng tính đúng sai:** đây là thuật toán kinh điển để trích quaternion ổn định số học từ ma trận xoay (tránh chia cho 0 khi dùng công thức trực tiếp $q_0=\frac12\sqrt{1+\mathrm{tr}(R)}$ lúc $\mathrm{tr}(R)\approx-1$). Về mặt toán học, `rotmat2qvec` là nghịch đảo (gần đúng, tới dấu) của `qvec2rotmat`: với $q$ chuẩn từ `rotmat2qvec(qvec2rotmat(q))`, kết quả phải bằng $\pm q$.

## 4. Đọc nhị phân (`read_next_bytes`) — không phải công thức số học

Trích nguyên văn (`scene/colmap_loader.py`, dòng 72–81):

```python
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
```

Hàm tiện ích đọc và giải mã (`struct.unpack`) một số byte cố định theo định dạng `format_char_sequence` và thứ tự byte `endian_character` (mặc định little-endian `<`, khớp định dạng nhị phân của COLMAP). Không có công thức toán — đây là ánh xạ byte → số theo chuẩn IEEE-754/số nguyên bù hai của `struct`, dùng làm khối xây dựng cho mọi hàm `read_*_binary` bên dưới.

Các hàm `read_points3D_text/binary`, `read_intrinsics_text/binary`, `read_extrinsics_text/binary`, `read_colmap_bin_array` chỉ parse cấu trúc dữ liệu COLMAP (toạ độ $xyz$, màu $rgb$, sai số tái chiếu `error`, tham số nội tại camera `params`, quaternion/tịnh tiến `qvec`/`tvec` của từng ảnh) thành mảng `numpy`, không áp dụng biến đổi toán học nào lên giá trị đọc được — các giá trị $xyz, qvec, tvec,$ `params` được dùng nguyên văn bởi các module khác trong codebase (ví dụ `scene/dataset_readers.py`) để dựng camera pose và point cloud khởi tạo.

## 5. Kiểm chứng tính đúng sai (tổng hợp)

| Hàm | Công thức | Đối chiếu | Khớp? |
|---|---|---|---|
| `qvec2rotmat` | $R(q)$ theo quaternion chuẩn $(w,x,y,z)$ | Đúng công thức Rodrigues-quaternion tiêu chuẩn, khớp quy ước COLMAP gốc (`read_write_model.py`) | Khớp |
| `rotmat2qvec` | Eigenvector lớn nhất của ma trận $K$ Shepperd-Markley, hoán vị `[3,0,1,2]`, chuẩn hoá dấu $q_0\ge0$ | Khớp thuật toán kinh điển trích quaternion ổn định số học; là nghịch đảo (tới dấu) của `qvec2rotmat` | Khớp |
| `read_next_bytes` | Giải mã nhị phân `struct.unpack`, không phải công thức toán | Đúng vai trò hạ tầng I/O | Không áp dụng (N/A) |

Không phát hiện sai sót toán học trong file `scene/colmap_loader.py` so với công thức COLMAP gốc.

## 6. Ví dụ số minh hoạ `qvec2rotmat` ↔ `rotmat2qvec`

Lấy $q=(w,x,y,z)=(0.9239,\,0.3827,\,0,\,0)$ — phép xoay $45^\circ$ quanh trục $x$ (vì $w=\cos(\theta/2)=\cos(22.5^\circ)\approx0.9239$, $x=\sin(22.5^\circ)\approx0.3827$).

Áp dụng `qvec2rotmat`:

$$
R = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1-2(0.3827)^2 & -2(0.9239)(0.3827) \\
0 & 2(0.9239)(0.3827) & 1-2(0.3827)^2
\end{bmatrix}
\approx
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0.7071 & -0.7071 \\
0 & 0.7071 & 0.7071
\end{bmatrix}
$$

đúng ma trận xoay $45^\circ$ quanh trục $x$ ($\cos45^\circ=\sin45^\circ\approx0.7071$). Áp dụng ngược `rotmat2qvec(R)` sẽ cho lại $q\approx(0.9239,0.3827,0,0)$ (hoặc dấu đối $(-0.9239,-0.3827,0,0)$ trước bước chuẩn hoá dấu, sau đó được lật về $q_0\ge0$) — xác nhận hai hàm là nghịch đảo của nhau.
