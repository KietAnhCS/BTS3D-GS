# Công thức toán học trong `utils/general_utils.py`

File này là nơi **định nghĩa gốc** của các hàm toán học được `scene/gaussian_model.py` import trực tiếp (`inverse_sigmoid`, `get_expon_lr_func`, `build_rotation`, `identity_gate`, `strip_symmetric`, `build_scaling_rotation`) cũng như hàm tiện ích `PILtoTorch` dùng trong `camera_utils.py` và `matrix_to_quaternion` (chiều ngược của `build_rotation`). Tài liệu trích chi tiết công thức thực thi, đối chiếu với mục 1, 4 của `gaussian_model.md` để đảm bảo nhất quán.

---

## 1. `identity_gate(x)`

Hàm cổng đồng nhất (identity), dùng làm hàm kích hoạt thay thế cho opacity ở "chế độ tuyến tính" (`modify_functions`, mục 1.2 của `gaussian_model.md`). Trích nguyên văn (`utils/general_utils.py`, dòng 18–19):

```python
def identity_gate(x):
    return x
```

$$
\text{identity\_gate}(x) = x
$$

Khi dùng làm `opacity_activation`, kết hợp với $\alpha_i=|o_i^{raw}|$ ở nơi gọi (not trong file này — `gaussian_model.py` áp `torch.abs` trước khi hoặc sau khi gọi gate tuỳ cấu hình).

---

## 2. `inverse_sigmoid(x)`

Trích nguyên văn (`utils/general_utils.py`, dòng 21–22):

```python
def inverse_sigmoid(x):
    return torch.log(x/(1-x))
```

$$
\mathrm{inverse\_sigmoid}(x) = \log\!\frac{x}{1-x}
$$

Đây chính là hàm logit — nghịch đảo chính xác của $\sigma(y)=\dfrac{1}{1+e^{-y}}$: nếu $x=\sigma(y)$ thì $y=\mathrm{inverse\_sigmoid}(x)$. Dùng trong `gaussian_model.py` để chuyển opacity mong muốn $\alpha\in(0,1)$ về tham số thô `_opacity` trước sigmoid (mục 1.1, 2.4, 3, 6.5 của `gaussian_model.md`).

---

## 3. `PILtoTorch(pil_image, resolution)`

Trích nguyên văn (`utils/general_utils.py`, dòng 24–30):

```python
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
```

$$
I_{resized} = \mathrm{resize}(I_{PIL}, \text{resolution}), \qquad T = \frac{\mathrm{asarray}(I_{resized})}{255}
$$

Nếu ảnh có 3 chiều (có kênh màu), hoán vị $(H,W,C)\to(C,H,W)$ (dòng 27–28):

$$
T' = \mathrm{permute}(T, (2,0,1))
$$

Nếu ảnh xám (2 chiều $(H,W)$), thêm một chiều kênh ở cuối rồi hoán vị tương tự (dòng 29–30):

$$
T' = \mathrm{permute}(\mathrm{unsqueeze}(T,-1), (2,0,1)) \in \mathbb{R}^{1\times H\times W}
$$

---

## 4. `get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps)`

Trích nguyên văn toàn bộ `helper(step)` (`utils/general_utils.py`, dòng 50–63):

```python
def helper(step):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp
```

Trả về hàm `helper(step)` tính tốc độ học tại bước `step`. Đặt $t=\mathrm{clip}(\text{step}/\text{max\_steps},\,0,\,1)$ (dòng 61).

**Nội suy log-tuyến tính** (tương đương suy giảm theo hàm mũ giữa $\mathrm{lr}_{init}$ và $\mathrm{lr}_{final}$, dòng 62):

$$
\mathrm{log\_lerp}(t) = \exp\!\Big(\log(\mathrm{lr}_{init})\,(1-t) + \log(\mathrm{lr}_{final})\,t\Big)
$$

**Hệ số trễ khởi động** (delay, dạng cosine ngược — nếu `lr_delay_steps > 0`, dòng 54–58):

$$
\delta(\text{step}) = \mathrm{lr}_{delay\_mult} + (1-\mathrm{lr}_{delay\_mult})\cdot \sin\!\Big(\frac{\pi}{2}\,\mathrm{clip}\big(\tfrac{\text{step}}{\mathrm{lr\_delay\_steps}},0,1\big)\Big)
$$

Nếu `lr_delay_steps <= 0` thì $\delta \equiv 1$ (không trễ, dòng 60).

**Tốc độ học cuối cùng** (dòng 63):

$$
\mathrm{lr}(\text{step}) = \delta(\text{step})\cdot \mathrm{log\_lerp}(t)
$$

**Trường hợp đặc biệt** (dòng 51–53): nếu `step < 0` hoặc ($\mathrm{lr}_{init}=0$ và $\mathrm{lr}_{final}=0$) thì trả về $0$ (tham số bị vô hiệu hoá, không tối ưu).

Công thức này **khớp hoàn toàn** với mục 4 của `gaussian_model.md` (ký hiệu $\delta(t)$, $\mathrm{lr}(t)$ ở đó dùng $t$ đã chuẩn hoá $[0,1]$ và $t_{delay}=\mathrm{lr\_delay\_steps}/\mathrm{max\_steps}$ một cách tương đương).

---

## 5. `strip_lowerdiag(L)` / `strip_symmetric(sym)`

Trích nguyên văn (`utils/general_utils.py`, dòng 67–79):

```python
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)
```

Với ma trận đối xứng $3\times3$ $\Sigma=\begin{pmatrix}\Sigma_{00}&\Sigma_{01}&\Sigma_{02}\\ \Sigma_{01}&\Sigma_{11}&\Sigma_{12}\\ \Sigma_{02}&\Sigma_{12}&\Sigma_{22}\end{pmatrix}$, chỉ 6 phần tử độc lập (tam giác trên, bao gồm đường chéo) được giữ lại theo thứ tự:

$$
\mathrm{uncertainty} = (\Sigma_{00},\ \Sigma_{01},\ \Sigma_{02},\ \Sigma_{11},\ \Sigma_{12},\ \Sigma_{22}) \in \mathbb{R}^6
$$

`strip_symmetric` chỉ là lớp gọi lại `strip_lowerdiag` (bí danh). Đây là cơ chế lưu nén $\Sigma_i$ được nói tới ở mục 1.3 của `gaussian_model.md`.

---

## 6. `build_rotation(r)` — quaternion → ma trận xoay 3×3

Trích nguyên văn (`utils/general_utils.py`, dòng 81–102):

```python
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
```

**Thứ tự thành phần quaternion trong code: $(w, x, y, z)$**, cụ thể `r = q[:,0]` (phần thực $w$), `x = q[:,1]`, `y = q[:,2]`, `z = q[:,3]` (dòng 88–91).

Chuẩn hoá trước (dòng 82–84):

$$
\lVert q\rVert = \sqrt{w^2+x^2+y^2+z^2}, \qquad (w,x,y,z) \leftarrow \frac{(w,x,y,z)}{\lVert q\rVert}
$$

Ma trận xoay chuẩn (quaternion đơn vị, quy ước hàng/cột khớp chính xác với code, dòng 93–101):

$$
R(q) = \begin{pmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{pmatrix}
$$

Đối chiếu từng phần tử với code (biến `r` trong code = $w$ ở đây):

- $R_{00}=1-2(y^2+z^2)$, $R_{01}=2(xy-wz)$, $R_{02}=2(xz+wy)$
- $R_{10}=2(xy+wz)$, $R_{11}=1-2(x^2+z^2)$, $R_{12}=2(yz-wx)$
- $R_{20}=2(xz-wy)$, $R_{21}=2(yz+wx)$, $R_{22}=1-2(x^2+y^2)$

Đây chính là $R_i=R(q_i)$ dùng xuyên suốt `gaussian_model.py` (mục 1.3, 5.3, 6.3) và trong phép chiếu trục ở `freq_utils.py` (`compute_projected_axes_subset`, mục 3 của `freq_utils.md`).

---

## 7. `matrix_to_quaternion(R)` — chiều ngược lại (ma trận xoay → quaternion)

Trích nguyên văn toàn bộ (`utils/general_utils.py`, dòng 104–144):

```python
def matrix_to_quaternion(R):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros((R.shape[0], 4), device=R.device)

    # Case tr > 0
    mask0 = tr > 0
    if mask0.any():
        s = torch.sqrt(tr[mask0] + 1.0) * 2
        q[mask0, 0] = 0.25 * s
        q[mask0, 1] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / s
        q[mask0, 2] = (R[mask0, 0, 2] - R[mask0, 2, 0]) / s
        q[mask0, 3] = (R[mask0, 1, 0] - R[mask0, 0, 1]) / s

    # Case R00 is max
    mask1 = (~mask0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask1.any():
        s = torch.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 1] = 0.25 * s
        q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s
        q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / s

    # Case R11 is max
    mask2 = (~mask0) & (~mask1) & (R[:, 1, 1] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / s
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 2] = 0.25 * s
        q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s

    # Case R22 is max
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2
        q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / s
        q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / s
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s
        q[mask3, 3] = 0.25 * s

    return q / torch.norm(q, dim=1, keepdim=True)
```

Thuật toán chuẩn (Shepperd's method) chọn nhánh theo vết ma trận $\mathrm{tr}=R_{00}+R_{11}+R_{22}$ để tránh chia cho số gần 0 (dòng 105, 109). Kết quả trả về theo **cùng thứ tự $(w,x,y,z)$** như `build_rotation` nhận vào (`q[:,0]` là $w$).

**Nhánh 1** ($\mathrm{tr}>0$, dòng 110–115):

$$
s = 2\sqrt{\mathrm{tr}+1}, \qquad w=\frac{s}{4},\quad x=\frac{R_{21}-R_{12}}{s},\quad y=\frac{R_{02}-R_{20}}{s},\quad z=\frac{R_{10}-R_{01}}{s}
$$

**Nhánh 2** ($R_{00}$ lớn nhất trong đường chéo, dòng 118–124):

$$
s = 2\sqrt{1+R_{00}-R_{11}-R_{22}}, \qquad w=\frac{R_{21}-R_{12}}{s},\quad x=\frac{s}{4},\quad y=\frac{R_{01}+R_{10}}{s},\quad z=\frac{R_{02}+R_{20}}{s}
$$

**Nhánh 3** ($R_{11}$ lớn nhất, dòng 127–133):

$$
s = 2\sqrt{1+R_{11}-R_{00}-R_{22}}, \qquad w=\frac{R_{02}-R_{20}}{s},\quad x=\frac{R_{01}+R_{10}}{s},\quad y=\frac{s}{4},\quad z=\frac{R_{12}+R_{21}}{s}
$$

**Nhánh 4** ($R_{22}$ lớn nhất, dòng 136–142):

$$
s = 2\sqrt{1+R_{22}-R_{00}-R_{11}}, \qquad w=\frac{R_{10}-R_{01}}{s},\quad x=\frac{R_{02}+R_{20}}{s},\quad y=\frac{R_{12}+R_{21}}{s},\quad z=\frac{s}{4}
$$

Kết quả cuối cùng chuẩn hoá lại: $q \leftarrow q/\lVert q\rVert_2$ (dòng 144).

(Hàm này không được `gaussian_model.py` dùng trực tiếp trong các đoạn đã khảo sát, nhưng là nghịch đảo toán học chính xác của `build_rotation` ở mục 6, hữu ích khi cần khôi phục quaternion từ một phép xoay hợp thành, ví dụ $R_i^{total}=R_{view}R_i$ trong `freq_utils.py`.)

---

## 8. `build_scaling_rotation(s, r)`

Trích nguyên văn (`utils/general_utils.py`, dòng 146–155):

```python
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L
```

Với $s\in\mathbb{R}^{N\times3}$ (tỉ lệ theo 3 trục, **không phải log**, giá trị đã $\exp$) và $r$ là quaternion thô (sẽ được `build_rotation` chuẩn hoá nội bộ):

$$
S = \mathrm{diag}(s_0,s_1,s_2) \quad (\text{ma trận đường chéo } 3\times3), \qquad R=\mathrm{build\_rotation}(r)
$$

$$
L = R\,S
$$

Đây chính là ma trận $L_i$ trong công thức hiệp phương sai $\Sigma_i=L_iL_i^\top=R_iS_iS_i^\top R_i^\top$ ở mục 1.3 của `gaussian_model.md` — `build_scaling_rotation` tạo ra đúng $L_i=R_iS_i$ được dùng làm đầu vào cho `strip_symmetric(L @ L.transpose(1,2))` trong `build_covariance_from_scaling_rotation`.

---

## 9. `safe_state(silent)`

Không chứa công thức toán học; chỉ thiết lập seed cố định cho tái lập kết quả. Trích nguyên văn phần seed (`utils/general_utils.py`, dòng 175–180):

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # For multi-GPU
torch.cuda.set_device(torch.device("cuda:0"))
```

$$
\text{seed}(\mathrm{random}) = \text{seed}(\mathrm{numpy}) = \text{seed}(\mathrm{torch}) = \text{seed}(\mathrm{torch.cuda}) = 0
$$

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `identity_gate(x): return x` | $\mathrm{identity\_gate}(x) = x$ |
| `inverse_sigmoid(x): return log(x/(1-x))` | $\mathrm{inverse\_sigmoid}(x) = \log\dfrac{x}{1-x}$ |
| `resized_image = np.array(pil_image.resize(resolution))/255.0` | $T = \mathrm{asarray}(\mathrm{resize}(I,\text{res}))/255$ |
| `resized_image.permute(2,0,1)` | $(H,W,C)\to(C,H,W)$ |
| `resized_image.unsqueeze(-1).permute(2,0,1)` | ảnh xám: thêm kênh rồi hoán vị |
| `delay_rate = lr_delay_mult + (1-lr_delay_mult)*sin(0.5*pi*clip(step/lr_delay_steps,0,1))` | $\delta=\mathrm{lr}_{delay\_mult}+(1-\mathrm{lr}_{delay\_mult})\sin\!\big(\tfrac{\pi}{2}\mathrm{clip}(\cdot,0,1)\big)$ |
| `t = clip(step/max_steps, 0, 1)` | $t=\mathrm{clip}(\text{step}/\text{max\_steps},0,1)$ |
| `log_lerp = exp(log(lr_init)*(1-t) + log(lr_final)*t)` | $\mathrm{log\_lerp}(t)=\exp(\log(\mathrm{lr}_{init})(1-t)+\log(\mathrm{lr}_{final})t)$ |
| `return delay_rate * log_lerp` | $\mathrm{lr}(\text{step})=\delta\cdot\mathrm{log\_lerp}(t)$ |
| `uncertainty[:,0]=L[:,0,0]`...`[:,5]=L[:,2,2]` | $(\Sigma_{00},\Sigma_{01},\Sigma_{02},\Sigma_{11},\Sigma_{12},\Sigma_{22})$ |
| `norm = sqrt(r0^2+r1^2+r2^2+r3^2)` | $\lVert q\rVert=\sqrt{w^2+x^2+y^2+z^2}$ |
| `q = r / norm[:, None]` | $(w,x,y,z)\leftarrow(w,x,y,z)/\lVert q\rVert$ |
| `r=q[:,0]; x=q[:,1]; y=q[:,2]; z=q[:,3]` | thứ tự quaternion $(w,x,y,z)$ |
| `R[:,0,0] = 1 - 2*(y*y+z*z)` | $R_{00}=1-2(y^2+z^2)$ |
| `R[:,0,1] = 2*(x*y - r*z)` | $R_{01}=2(xy-wz)$ |
| `R[:,0,2] = 2*(x*z + r*y)` | $R_{02}=2(xz+wy)$ |
| `R[:,1,0] = 2*(x*y + r*z)` | $R_{10}=2(xy+wz)$ |
| `R[:,1,1] = 1 - 2*(x*x+z*z)` | $R_{11}=1-2(x^2+z^2)$ |
| `R[:,1,2] = 2*(y*z - r*x)` | $R_{12}=2(yz-wx)$ |
| `R[:,2,0] = 2*(x*z - r*y)` | $R_{20}=2(xz-wy)$ |
| `R[:,2,1] = 2*(y*z + r*x)` | $R_{21}=2(yz+wx)$ |
| `R[:,2,2] = 1 - 2*(x*x+y*y)` | $R_{22}=1-2(x^2+y^2)$ |
| `s = sqrt(tr+1)*2` (nhánh `tr>0`) | $s=2\sqrt{\mathrm{tr}+1}$ |
| `q[mask0,0] = 0.25*s` | $w=s/4$ |
| `q[mask0,1] = (R21-R12)/s` | $x=(R_{21}-R_{12})/s$ |
| `L[:,0,0]=s[:,0]; L[:,1,1]=s[:,1]; L[:,2,2]=s[:,2]` | $S=\mathrm{diag}(s_0,s_1,s_2)$ |
| `L = R @ L` | $L = R\,S$ |
| `random.seed(0); np.random.seed(0); torch.manual_seed(0)` | seed cố định $=0$ cho mọi RNG |
