[← Phần 7](07-density-control.md) · [Mục lục lời giải](00-muc-luc-loi-giai.md) · [Đề bài](../16-bai-toan-lon-de-bai.md)

# Phần 8 — Mô hình Chi phí & Xuất `point_cloud.ply` (SADGS thật)

> **Đầu vào nhận từ Phần 7:** $\mathcal G_1$ — quần thể Gaussian sau `densify_and_prune_structgs` tại
> $t=5000$ (18 Gaussian: 8 con của $G_1$, 1 bản của $G_2$, 8 con của $G_3$, $G_4$ giữ nguyên) — xem
> [Phần 7](07-density-control.md).

> **Đính chính so với bản nháp trước:** bản trước của Phần 8 dựa trên một cơ chế densify/prune giả định
> và layout PLY **62 cột** (không có `filter_3D`). Codebase thật `SADGS/scene/gaussian_model.py` dùng
> `construct_list_of_attributes`/`save_ply` với **63 cột** (thêm cột `filter_3D`, luôn tồn tại vì
> `train.py:158` khởi tạo `gaussians.filter_3D = torch.zeros((N,1))` ngay sau `create_from_pcd`, bất kể
> `compute_3d_filter` có bật hay không). Phần 8 này viết lại theo đúng mã nguồn và đúng quần thể $\mathcal
> G_1$ (18 Gaussian) đã tính ở Phần 7 mới.

---

## Mục lục Phần 8

- [8.A — Mô hình chi phí $T_{\text{iter}}$ trước/sau bước densify](#8a)
  - [8.A.1 — Công thức và ba đòn bẩy thật của SADGS](#8a1)
  - [8.A.2 — $K$ trước densify: 4 Gaussian gốc × 3 camera](#8a2)
  - [8.A.3 — $K$ sau densify: minh hoạ trên các họ con tiêu biểu](#8a3)
  - [8.A.4 — Đòn bẩy 1 (compact box `mult=0.7`) và hiện tượng bão hoà lưới tile](#8a4)
  - [8.A.5 — Đòn bẩy 2 ($N$): $4\to18$, không rút gọn bằng multinomial](#8a5)
  - [8.A.6 — Đòn bẩy 3 (nhịp Adam dùng chung `optimizer`+`shoptimizer`)](#8a6)
  - [8.A.7 — Overhead ADC thật: `update_freq_stats_online` mỗi 10 vòng, không phải một pha chấm điểm riêng](#8a7)
  - [8.A.8 — Tổng hợp](#8a8)
- [8.B — Xuất `point_cloud.ply` (63 cột, đúng `construct_list_of_attributes`/`save_ply` thật)](#8b)
  - [8.B.1 — Từ $\theta$ nội bộ sang 63 cột PLY](#8b1)
  - [8.B.2 — Bảng thay số cho 4 Gaussian đại diện](#8b2)
  - [8.B.3 — Script sinh file (mã tham khảo, KHÔNG ghi đè assets đã có)](#8b3)
  - [8.B.4 — File `.ply` mẫu đã có sẵn trong `assets/`: phạm vi và giới hạn](#8b4)
  - [8.B.5 — Cách render ra 3D](#8b5)
  - [8.B.6 — Ghi chú đối chiếu với code (`save_ply`/`load_ply` thật)](#8b6)
  - [8.B.7 — Bài tập](#8b7)
- [Kết thúc lời giải](#ket-thuc)

---

<a id="8a"></a>
## 8.A — Mô hình chi phí $T_{\text{iter}}$ trước/sau bước densify

<a id="8a1"></a>
### 8.A.1 — Công thức và ba đòn bẩy thật của SADGS

$$
\boxed{\ T_{\text{iter}}=\underbrace{a\,N}_{\text{preprocess}}+\underbrace{b\,NK}_{\text{dup+sort+blend+backward}}+\underbrace{c\,N\cdot\mathbb 1[\text{Adam step}]}_{\text{optimizer}}+\underbrace{F}_{\text{loss, SSIM, IO}}\ }
$$

Ba đòn bẩy **thật** của SADGS (đối chiếu `SADGS/arguments/__init__.py`, `SADGS/scene/gaussian_model.py`,
`SADGS/train.py`), áp lên cảnh đồ chơi 4 Gaussian tại $t=5000$:

| Đòn bẩy | Cơ chế thật | Khác gì so với bản nháp trước đã đính chính |
|---|---|---|
| Giảm $K$ | Compact box `mult=0.7` (không phải `0.5`) — `self.mult` trong `arguments/__init__.py:114` | hệ số nhân lớn hơn giả định trước đây, nhưng xem §8.A.4: trên lưới tile $3\times2$ thô của cảnh đồ chơi, hiệu ứng gần như **bão hoà** |
| Giảm/tăng $N$ | `densify_and_prune_structgs`: $N:4\to18$ ở $t=5000$ (Phần 7) — **không có** bước rút gọn bằng `multinomial`/`pruning_score` trong đường chạy thật | bản nháp trước giả định $N:4\to8\to4$ (rút bớt bằng multinomial) — sai với `train.py` thật, nơi `pruning_score=None` luôn |
| Nhịp Adam | `optimizer_step`: `self.optimizer` **và** `self.shoptimizer` dùng **chung một lịch** ($1\times$ mọi vòng $\le15000$, $1/32$ trong $(15000,20000]$, $1/64$ sau đó) | bản nháp trước giả định SH có lịch riêng $1/16$ — không có trong `optimizer_step` thật của SADGS |

<a id="8a2"></a>
### 8.A.2 — $K$ trước densify: 4 Gaussian gốc × 3 camera

Dùng lại phép chiếu xấp xỉ đã thiết lập ở các phần trước (bỏ số hạng ngoài-chéo của Jacobian $J$,
$\Sigma'_{11}\approx(f_x/t_z)^2s^2+0.3$, đẳng hướng nên $\Sigma'_{11}=\Sigma'_{22}$), nhưng với
**`mult` thật $=0.7$**:

$$
t_{\text{box}}=\texttt{mult}\cdot2\ln(255\alpha)=0.7\times2\ln(25.5)=1.4\times3.23868=4.53415
$$

Bốn Gaussian gốc $\mathcal G_0$ (từ §6.2), ba camera $c_1=(0,0,-4)$, $c_2=(1.5,0,-4)$, $c_3=(-1.5,0.5,-4)$:

| Gaussian | cam | $\mu'$ (px) | $\Sigma'_{11}$ | half $=\sqrt{t_{\text{box}}\Sigma'_{11}}$ | rect (clamp lưới $3\times2$, tile 16px) | $K$ |
|---|---|---|---|---|---|---|
| $G_1$ | 1 | $(23.50,15.50)$ | 72.64 | 18.15 | $[0,3)\times[0,2)$ | 6 |
| $G_1$ | 2 | $(8.50,15.50)$ | 72.64 | 18.15 | $[0,2)\times[0,2)$ | 4 |
| $G_1$ | 3 | $(38.50,10.50)$ | 72.64 | 18.15 | $[1,3)\times[0,2)$ | 4 |
| $G_2$ | 1 | $(27.94,18.17)$ | 70.62 | 17.89 | $[0,3)\times[0,2)$ | 6 |
| $G_2$ | 2 | $(14.61,18.17)$ | 70.62 | 17.89 | $[0,2)\times[0,2)$ | 4 |
| $G_2$ | 3 | $(41.28,13.72)$ | 70.62 | 17.89 | $[1,3)\times[0,2)$ | 4 |
| $G_3$ | 1 | $(20.30,13.90)$ | 79.87 | 19.02 | $[0,3)\times[0,2)$ | 6 |
| $G_3$ | 2 | $(8.30,13.90)$ | 79.87 | 19.02 | $[0,2)\times[0,2)$ | 4 |
| $G_3$ | 3 | $(32.30,9.90)$ | 79.87 | 19.02 | $[1,3)\times[0,2)$ | 4 |
| $G_4$ | 1 | $(26.36,10.74)$ | 71.95 | 18.06 | $[0,3)\times[0,2)$ | 6 |
| $G_4$ | 2 | $(12.07,10.74)$ | 71.95 | 18.06 | $[0,2)\times[0,2)$ | 4 |
| $G_4$ | 3 | $(40.64,5.98)$ | 71.95 | 18.06 | $[1,3)\times[0,2)$ | 4 |

$$
K_{\text{before}}=\sum_{i=1}^4\sum_{v=1}^3K_{i,v}=(6+4+4)\times4=\mathbf{56}
$$

**Đúng bằng** con số tính với `mult=0.5` ở phiên bản trước — xem giải thích bão hoà ở §8.A.4.

<a id="8a3"></a>
### 8.A.3 — $K$ sau densify: minh hoạ trên các họ con tiêu biểu

$\mathcal G_1$ (Phần 7) có 18 Gaussian không đồng nhất về scale/vị trí; tính đủ $18\times3=54$ cặp vượt
phạm vi hợp lý của một ví dụ tay. Ta tính **đại diện một con của mỗi họ** (đủ để thấy xu hướng, không
phải tổng chính xác toàn bộ $\mathcal G_1$):

**Con $G_1c_{(0,0,0)}$** ($\mu=(-0.7366,-0.7366,-0.7366)$, $s=0.42525$, §7.8.4):

$$
\Sigma'_{11}=(f_x/t_z)^2s^2+0.3,\quad t_z=\mu_z-c_{v,z}
$$

Với camera 1 ($c_1=(0,0,-4)$): $t_z=-0.7366-(-4)=3.2634$; $\mu'_x=24+40\cdot(-0.7366)/3.2634-0.5=24-9.028-0.5=14.47$;
$\mu'_y=16+40\cdot(-0.7366)/3.2634-0.5=16-9.028-0.5=6.47$; $\Sigma'_{11}=(40/3.2634)^2\times0.42525^2+0.3=150.15\times0.18084+0.3=27.15+0.3=27.45$;
half$=\sqrt{4.53415\times27.45}=\sqrt{124.44}=11.16$. Rect $x$: $\lfloor(14.47-11.16)/16\rfloor=0$;
$\lfloor(14.47+11.16)/16\rfloor+1=1+1=2$, clamp $\min(3,2)=2$. Rect $y$: $\lfloor(6.47-11.16)/16\rfloor=-1\to0$;
$\lfloor(6.47+11.16)/16\rfloor+1=1+1=2$, clamp $\min(2,2)=2$. $K=2\times2=4$.

**Con $G_3c_{(0,0,0)}$** ($s=0.5575$, §7.8.4), camera 1, $\mu\approx(-2.9925,-0.4440,-0.3893)$
(cùng công thức lưới như $G_1$ nhưng bán kính lớn hơn $1.9313$ so với $1.4732$):
$t_z=-0.3893+4=3.6107$; $\mu'_x=24+40\times(-2.9925)/3.6107-0.5=24-33.15-0.5=-9.65$ (ra ngoài khung hình!);
$\Sigma'_{11}=(40/3.6107)^2\times0.5575^2+0.3=122.7\times0.3108+0.3=38.14+0.3=38.44$; half$=\sqrt{4.53415\times38.44}=13.20$.
Rect $x$: $\lfloor(-9.65-13.20)/16\rfloor=-2\to$ clamp $\max(0,-2)=0$; $\lfloor(-9.65+13.20)/16\rfloor+1=0+1=1$,
clamp $\min(3,1)=1$. $K_x=1$ tile — con này **gần như ra khỏi khung hình** ở camera 1 (hệ quả trực tiếp của
việc rải lưới lập phương cạnh $2\times1.9313=3.86$, lớn hơn khoảng cách $G_3$ tới biên khung hình).

**Bản duy nhất của $G_2$** (không đổi vị trí/scale so với gốc, §7.8.4): $K$ giống hệt hàng "$G_2$" của
bảng §8.A.2 ($K=6,4,4$ cho 3 camera).

**$G_4$** (không đổi qua vòng densify này): $K$ giống hệt hàng "$G_4$" của bảng §8.A.2.

Xu hướng chung: con của Gaussian càng bị split mạnh (nhiều trục $k=2$) càng có scale nhỏ hơn hẳn cha
($s/2$ thay vì $s/1.6$ của bản nháp trước) ⇒ $K$ mỗi con **giảm rõ rệt** so với cha (ví dụ $G_1c$: $K=4$
so với $K=6$ của $G_1$ ở cùng camera 1) — nhưng vì **số con tăng gấp 8** (không phải gấp đôi), **tổng**
$K$ của cả họ con thường **lớn hơn** tổng $K$ của một Gaussian cha, dù mỗi con rẻ hơn. Đây là điểm khác
biệt cơ bản so với 3DGS gốc ($N\to2N$): SADGS đánh đổi "mỗi điểm rẻ hơn nhiều" lấy "nhiều điểm hơn
nhiều" khi cả ba trục cùng vi phạm Nyquist — xem Bài tập 7.4 (Phần 7).

<a id="8a4"></a>
### 8.A.4 — Đòn bẩy 1 (compact box `mult=0.7`) và hiện tượng bão hoà lưới tile

So $K_{\text{compact}}$ (`mult=0.7`, giá trị CLI thật) với $K_{\text{compact}}$ (`mult=0.5`, giả định cũ)
trên **cùng** $\mathcal G_0$ (4 Gaussian gốc): cả hai đều cho $K_{\text{before}}=56$ (bảng §8.A.2 và bản
trước). Lý do: ảnh $48\times32$ với tile $16$px chỉ có lưới $3\times2=6$ tile — với `half`$\approx15$–$19$
px (cả hai `mult`), rect theo trục $x$ đã bị **clamp** vào đúng $\{[0,2),[0,3),[1,3)\}$ ở cả hai giá trị
`mult` (nửa-nửa băng ảnh $48$px chỉ rộng $24$px, còn `half`$>15$px đã đủ phủ quá nửa ảnh). Vậy trên **cảnh
đồ chơi cực nhỏ** này, đòn bẩy "giảm K bằng compact box" **không thể hiện tác dụng phân biệt** giữa
`mult=0.5` và `mult=0.7` (cả hai đều bão hoà lưới) — tác dụng thật của tham số `mult` chỉ rõ ràng trên
ảnh độ phân giải cao (nhiều tile hơn nhiều so với kích thước Gaussian), như các scene thật trong
`full_eval.py`. Ở các Gaussian con **nhỏ hơn** (sau split, §8.A.3), `half` giảm hẳn xuống $\sim11$–$13$px,
không còn bão hoà lưới — ở quy mô này `mult` mới thực sự ảnh hưởng tuyến tính tới diện tích rect.

<a id="8a5"></a>
### 8.A.5 — Đòn bẩy 2 ($N$): $4\to18$, không rút gọn bằng multinomial

$$
N:\ 4\ \xrightarrow{\text{split }G_1(\times8),\,G_2(\times1),\,G_3(\times8),\,G_4\text{ giữ nguyên}}\ 18
$$

Không có bước "prune một nửa ứng viên bằng `multinomial`" trong đường chạy thật (`train.py` luôn gọi
`densify_and_prune_structgs` với `pruning_score=None` — xem Phần 7 §7.3/§7.6). Việc $N$ **tăng** ròng
$4\to18$ ở $t=5000$ (thay vì giữ nguyên hoặc giảm như bản nháp trước giả định) phản ánh đúng thiết kế
thật: SADGS ưu tiên **độ trung thực cục bộ theo tần số** hơn kiểm soát ngân sách điểm tại một vòng cụ thể
— việc kiểm soát ngân sách tổng thể nằm ở `min_opacity`/`prune_ratio_threshold`/`final_prune_structgs` ở
các vòng/giai đoạn khác, không phải bằng lấy mẫu ngẫu nhiên ngay sau mỗi lần split.

<a id="8a6"></a>
### 8.A.6 — Đòn bẩy 3 (nhịp Adam dùng chung `optimizer` + `shoptimizer`)

Trích `SADGS/scene/gaussian_model.py:357-377` (`optimizer_step`, đã đọc ở Phần 6):

```python
def optimizer_step(self, iteration):
    if iteration <= 15000:
        self.optimizer.step();  self.optimizer.zero_grad(set_to_none=True)
        self.shoptimizer.step(); self.shoptimizer.zero_grad(set_to_none=True)
    elif iteration <= 20000:
        if iteration % 32 == 0: ...step cả hai...
    else:
        if iteration % 64 == 0: ...step cả hai...
```

Tại $t=5000\le15000$: **cả `optimizer` (14 tham số/Gaussian: $\mu,\tilde q,\tilde s,\tilde\alpha,k_{00}$)
lẫn `shoptimizer` (45 tham số/Gaussian: $f_{\text{rest}}$) đều step, không có chênh lệch nhịp** — khác
hẳn giả định trước đây (SH riêng $1/16$). Đòn bẩy "giảm nhịp Adam" của SADGS chỉ phát huy tác dụng
**sau** $t=15000$ (khi cả hai optimizer cùng chuyển sang $1/32$ rồi $1/64$) — tại $t=5000$, chi phí
optimizer của SADGS **bằng đúng** chi phí optimizer của 3DGS gốc (không tiết kiệm gì ở vòng này).

<a id="8a7"></a>
### 8.A.7 — Overhead ADC thật: `update_freq_stats_online` mỗi 10 vòng, không phải một pha chấm điểm riêng

Khác biệt lớn nhất so với giả định của bản nháp trước: SADGS **không** chạy một forward-pass phụ tại đúng vòng
densify để "chấm điểm". Chi phí tính $\eta_{3ch}$ được **rải đều** vào các vòng lặp bình thường, mỗi 10
vòng (`iteration % 10 == 0`, `train.py:275-276`) — dùng **đúng** ảnh/camera của vòng lặp đó, không render
thêm lần nào. Chi phí thêm mỗi lần: lấy mẫu song tuyến `grid_sample` trên bản đồ structure tensor đã
cache theo tên ảnh (`structure_tensor_cache`, tính một lần đầu mỗi ảnh, không tính lại mỗi lần dùng) cùng
vài phép toán ma trận $2\times2$ (trace/det/eigenvalue) trên $N$ điểm khả kiến — chi phí $O(N)$, không
$O(NK)$, và **không nhân thêm hệ số camera** như $\times7$ mà bản nháp trước giả định.

$$
\boxed{\text{Overhead ADC của SADGS tại chính vòng }t=5000\text{: không có forward phụ; chỉ thêm }O(N)\text{ mỗi 10 vòng, không riêng gì vòng densify}}
$$

Đây là một khác biệt thiết kế quan trọng so với cơ chế "chấm điểm tập trung" mà bản nháp trước giả định
(một cú tăng vọt tại đúng vòng densify, do phải render lại $V$ camera để chấm điểm): SADGS trả overhead
**rải mỏng** (một phép tính rẻ mỗi 10 vòng, tận dụng lại đúng ảnh/camera đang huấn luyện).

<a id="8a8"></a>
### 8.A.8 — Tổng hợp

| Đòn bẩy | Tại $t=5000$ (cụ thể) | Ghi chú |
|---|---|---|
| Compact box `mult=0.7` | $K_{\text{before}}=56$ (bão hoà lưới, §8.A.4); $K$ mỗi con sau split giảm rõ (ví dụ $G_1c$: $K=4$ so với cha $K=6$ ở cam 1) | tác dụng thật chỉ rõ trên ảnh phân giải cao |
| $N$ | $4\to18$ | tăng ròng, không rút gọn multinomial |
| Nhịp Adam | không tiết kiệm ở $t=5000$ (cả hai optimizer đều step, giống 3DGS gốc) | tiết kiệm chỉ bắt đầu sau $t=15000$ |
| Overhead ADC | $O(N)$ rải mỗi 10 vòng, không có pha chấm điểm riêng tại $t=5000$ | khác hẳn overhead tập trung mà bản nháp trước giả định |

**Kết luận:** trên đúng cảnh đồ chơi 4-Gaussian/3-camera, tại chính vòng $t=5000$, SADGS **tốn nhiều
$N$ hơn** (do split đẳng hướng $\times8$) nhưng **không** phải trả thêm chi phí Adam hay chi phí "chấm
điểm" tập trung nào — bù lại, mỗi Gaussian con rẻ hơn đáng kể về $K$. Việc $N$ tăng mạnh ở vòng densify
đầu tiên là đặc trưng cấu trúc của thuật toán split đẳng hướng $k^3$, sẽ được cân bằng dần bởi các cơ chế
prune (`custom_prune_mask` theo `low_ratio`, `final_prune_structgs`) ở các vòng/giai đoạn sau — không
nằm trong phạm vi số liệu của $t=5000$.

---

<a id="8b"></a>
## 8.B — Xuất `point_cloud.ply` (63 cột, đúng `construct_list_of_attributes`/`save_ply` thật)

<a id="8b1"></a>
### 8.B.1 — Từ $\theta$ nội bộ sang 63 cột PLY

Đọc trực tiếp từ `SADGS/scene/gaussian_model.py:379-413`:

```python
def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):    # 1*3 = 3
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]): # 15*3 = 45
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):     # 3
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):    # 4
        l.append('rot_{}'.format(i))
    l.append('filter_3D')                       # [KHÁC 3DGS gốc] cột thêm
    return l

def save_ply(self, path):
    mkdir_p(os.path.dirname(path))
    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()
    filter_3D = self.filter_3D.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
```

$$
6\,(xyz,n)+3\,(f_{dc})+45\,(f_{rest})+1\,(\text{opacity})+3\,(\text{scale})+4\,(\text{rot})+1\,(\text{filter\_3D})=\boxed{63\text{ cột}}
$$

**Điểm mấu chốt — không có activation nào được áp dụng lúc ghi (giống 3DGS gốc):**

| Cột PLY | Tensor nội bộ | Activation lúc ghi? | Activation thật (chỉ dùng khi render/`get_*`) |
|---|---|---|---|
| `x,y,z` | `self._xyz` | Không (đã là toạ độ world) | — |
| `nx,ny,nz` | `np.zeros_like(xyz)` | luôn $=0$, không liên quan hình học | — |
| `f_dc_0..2` | `self._features_dc` | Không | — (SH không qua activation) |
| `f_rest_0..44` | `self._features_rest` | Không | — |
| `opacity` | `self._opacity` | **KHÔNG** — ghi $\tilde\alpha$ thô | `get_opacity`$=\sigma(\tilde\alpha)$ |
| `scale_0..2` | `self._scaling` | **KHÔNG** — ghi $\tilde s$ thô | `get_scaling`$=\exp(\tilde s)$ |
| `rot_0..3` | `self._rotation` | **KHÔNG** — quaternion thô, chưa chuẩn hoá | `get_rotation`$=$`normalize`$(\tilde q)$ |
| `filter_3D` | `self.filter_3D` | Không — giá trị **đã là** bán kính lọc 3D vật lý (không qua activation nào cả, kể cả lúc dùng — dùng trực tiếp trong `get_opacity_with_3D_filter`/`get_scaling_with_3D_filter`) | — |

`filter_3D` mặc định $=0$ cho toàn bộ điểm trong cảnh đồ chơi này vì `opt.compute_3d_filter=False`
(mặc định, `arguments/__init__.py:132`) — `train.py:158` chỉ khởi tạo `torch.zeros((N,1))` một lần ngay
sau `create_from_pcd` và không bao giờ gọi `compute_3D_filter` trong vòng lặp huấn luyện mặc định; giá
trị này được `densification_postfix` nối thêm (giữ $0$ cho con mới) và `prune_points` cắt bớt tương ứng
mỗi lần densify/prune — xem `scene/gaussian_model.py:619-620,547-548`.

<a id="8b2"></a>
### 8.B.2 — Bảng thay số cho 4 Gaussian đại diện

$\mathcal G_1$ có 18 Gaussian (Phần 7 §7.8) — để bảng còn đọc được, ta liệt kê đủ 63 cột cho **4 Gaussian
đại diện** (một con tiêu biểu của mỗi họ: $G_1c_{(0,0,0)}$, bản duy nhất của $G_2$, $G_3c_{(0,0,0)}$,
$G_4$ không đổi):

| # | property | $G_1c_{(000)}$ | $G_2$ (bản duy nhất) | $G_3c_{(000)}$ | $G_4$ (không đổi) |
|---|---|---|---|---|---|
| 1–3 | `x,y,z` | −0.7366, −0.7366, −0.7366 | 0.5, 0.3, 0.5 | −2.9925, −0.4440, −0.3893 | 0.3, −0.5, 0.2 |
| 4–6 | `nx,ny,nz` | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| 7–9 | `f_dc_0..2` | 1.063472, −1.063472, −1.063472 | −1.063472, 0.708982, −0.708982 | −1.417963, −0.708982, 1.417963 | 0, 0, 0 |
| 10–54 | `f_rest_0..44` | $0\times45$ | $0\times45$ | $0\times45$ | $0\times45$ |
| 55 | `opacity` ($\tilde\alpha$) | −2.197225 | −2.197225 | −2.197225 | −2.197225 |
| 56–58 | `scale_0..2` ($\tilde s$) | −0.855321 (×3) | −0.058267 (×3, không đổi) | −0.584249 (×3) | −0.117861 (×3, không đổi) |
| 59–62 | `rot_0..3` | 1, 0, 0, 0 | 1, 0, 0, 0 | 1, 0, 0, 0 | 1, 0, 0, 0 |
| 63 | `filter_3D` | 0 | 0 | 0 | 0 |

($\tilde s(G_1c)=\log(0.8505/2)=-0.855321$, $\tilde s(G_3c)=\log(1.1150/2)=-0.584249$ — khớp §7.8.4 Phần 7;
$G_2,G_4$ giữ nguyên $\tilde s$ gốc vì không thực sự bị chia nhỏ, xem §7.8.4.)

<a id="8b3"></a>
### 8.B.3 — Script sinh file (mã tham khảo, KHÔNG ghi đè assets đã có)

Theo yêu cầu của nhiệm vụ này, thư mục `assets/` của lời giải **không được sửa**. Script dưới đây là mã
tham khảo (đúng theo `construct_list_of_attributes`/`save_ply` thật, 63 cột) để người đọc tự chạy nếu
muốn tái tạo file `.ply` khớp với $\mathcal G_1$ đầy đủ 18 Gaussian — nó **không** được thực thi trong
lời giải này và không ghi vào `assets/`:

```python
import numpy as np, math

C0 = 1.0 / (2.0 * math.sqrt(math.pi))
def rgb2sh(c):
    return (np.array(c, dtype=np.float64) - 0.5) / C0

ALPHA_TILDE = math.log(0.1 / 0.9)
QUAT = (1.0, 0.0, 0.0, 0.0)
SH_REST = 45

def make_row(mu, s_tilde_xyz, color):
    f_dc = tuple(rgb2sh(color).tolist())
    row = tuple(mu) + (0.0, 0.0, 0.0) + f_dc + tuple([0.0]*SH_REST) \
          + (ALPHA_TILDE,) + tuple(s_tilde_xyz) + QUAT + (0.0,)   # cột cuối: filter_3D = 0
    assert len(row) == 63, len(row)
    return row

# POP: danh sách 18 Gaussian của G1 (Phan 7, S7.8.4), moi phan tu la (mu, s_tilde x3, color ke thua tu cha)
attr_names = ['x','y','z','nx','ny','nz'] + [f'f_dc_{i}' for i in range(3)] \
    + [f'f_rest_{i}' for i in range(SH_REST)] + ['opacity'] \
    + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)] + ['filter_3D']
assert len(attr_names) == 63
```

<a id="8b4"></a>
### 8.B.4 — File `.ply` mẫu đã có sẵn trong `assets/`: phạm vi và giới hạn

Thư mục `DOCS/BOOK/16-loi-giai/assets/` chứa các file `point_cloud_iter5000.ply` (62 cột) và
`point_cloud_iter5000_simple_pointcloud.ply` (point cloud RGB suy biến) **được sinh ra cho bản nháp
trước đây** (quần thể 4 Gaussian, layout 62 cột không có `filter_3D`). Nhiệm vụ này chỉ sửa các
file `.md`, không được sửa nội dung `assets/`, nên hai file đó **vẫn giữ nguyên trên đĩa** và không còn
là đại diện chính xác cho quần thể $\mathcal G_1$ 18-Gaussian/63-cột của SADGS thật đã suy ra ở Phần 7.
Người đọc muốn có file `.ply` khớp đúng SADGS cần tự chạy script tham khảo ở §8.B.3 (hoặc chạy thật
`Scene.save`/`gaussians.save_ply` trong `SADGS/train.py` trên một scene COLMAP thật) — lời giải này chỉ
đảm bảo **công thức và layout cột là chính xác**, không tái sinh file nhị phân.

<a id="8b5"></a>
### 8.B.5 — Cách render ra 3D

File `.ply` 63 cột theo đúng schema trên tương thích với các công cụ hiểu 3D Gaussian Splatting chuẩn
(SIBR viewer của INRIA, các web viewer Gaussian Splat mã nguồn mở, `SADGS/render.py` của chính repo) —
**miễn** công cụ đó bỏ qua an toàn các cột lạ ngoài 3DGS gốc (`filter_3D`) hoặc hiểu đúng `save_ply` của
SADGS; một số viewer 3DGS "chuẩn" (chỉ biết 62 cột gốc, đọc theo tên `property`, không theo vị trí cột)
vẫn đọc đúng vì `plyfile`/hầu hết parser PLY đọc theo **tên** property, không theo thứ tự — cột thừa
`filter_3D` chỉ đơn giản bị bỏ qua nếu viewer không cần tới nó.

<a id="8b6"></a>
### 8.B.6 — Ghi chú đối chiếu với code (`save_ply`/`load_ply` thật)

| Hàm | Vị trí (`SADGS/scene/gaussian_model.py`) | Vai trò |
|---|---|---|
| `construct_list_of_attributes` | dòng 379–392 | sinh danh sách **63** tên cột, kết thúc bằng `filter_3D` |
| `save_ply` | dòng 394–413 | ghi 63 cột thô (không activation) — `attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)` |
| `load_ply` | dòng 435–483 | đọc ngược; có `try/except ValueError` quanh việc đọc cột `filter_3D` — nếu file **không có** cột này (ví dụ file `.ply` cũ từ 3DGS gốc), `filter_3D` được điền $0$ thay vì lỗi: `filter_3D = np.zeros((xyz.shape[0], 1))` |
| `reset_opacity` | dòng 415–433 | dùng `inverse_opacity_activation`, có tính tới `get_opacity_with_3D_filter` — xác nhận `filter_3D` **có** ảnh hưởng tới opacity hiệu dụng lúc render, dù bản thân cột này không qua activation nào khi lưu/đọc |

Trích `load_ply` (xác nhận cơ chế tương thích ngược):

```python
try:
    filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
except ValueError:
    filter_3D = np.zeros((xyz.shape[0], 1))
```

Đây là lý do file `.ply` **62 cột cũ** trong `assets/` (§8.B.4) vẫn nạp được vào `load_ply` của SADGS mà
không lỗi — chỉ khác là `filter_3D` sẽ được điền $0$ tự động, tương đương đúng giá trị mặc định của cảnh
đồ chơi này (vì `compute_3d_filter=False` trong toàn bộ pipeline đang xét).

<a id="8b7"></a>
### 8.B.7 — Bài tập

1. Viết 3 dòng Python dùng `1/(1+exp(-x))` để suy ra opacity "vật lý" từ cột `opacity`$=-2.197225$ trong
   bảng §8.B.2, xác nhận kết quả $=0.1$.
2. Vì sao `load_ply` cần khối `try/except ValueError` riêng cho `filter_3D` nhưng không cần cho các cột
   khác như `opacity`/`scale`/`rot`? (Gợi ý: `filter_3D` là tính năng bổ sung sau này của SADGS, các file
   `.ply` từ 3DGS gốc không có cột này.)
3. Với quần thể $\mathcal G_1$ đầy đủ 18 Gaussian (Phần 7 §7.8), tính tổng số tham số float ghi vào
   `.ply`: $18\times63=?$ So với $4\times62=248$ của quần thể $\mathcal G_0$ ban đầu — kích thước file
   nhị phân (body, không tính header) tăng bao nhiêu byte (mỗi float32 = 4 byte)?
4. `filter_3D` mặc định $0$ khi `compute_3d_filter=False`. Đọc lại `get_opacity_with_3D_filter` (Phần 2 /
   `scene/gaussian_model.py:189-201`): khi `filter_3D=0`, hệ số `coef` trong công thức có bằng $1$ không?
   Chứng minh bằng đại số, từ đó giải thích vì sao cột `filter_3D=0` là "vô hại" (không làm lệch opacity
   hiệu dụng) trong toàn bộ cảnh đồ chơi của sách.

---

<a id="ket-thuc"></a>
## Kết thúc lời giải

Tám phần của lời giải khép kín đúng vòng lặp huấn luyện **SADGS thật** trên cảnh đồ chơi chung: từ ba
file text COLMAP thô (`cameras.txt`, `images.txt`, `points3D.txt`, §16.1) qua khởi tạo 4 Gaussian × 59
tham số (Phần 1), dựng hiệp phương sai và giải mã màu SH (Phần 2), chiếu EWA lên cả 3 camera (Phần 3),
rasterize vi phân ra ảnh $48\times32$ (Phần 4), so khớp với ảnh ground-truth bằng $L_1$/SSIM/PSNR (Phần
5), lan truyền ngược và cập nhật một bước Adam tại $t=5000$ (Phần 6), ra quyết định densify/clone/prune
theo cơ chế **structure-aware** thật của SADGS (`densify_and_prune_structgs`, `densify_and_split_structgs`
với $k=\lceil\sqrt{\max(\eta,1)}\rceil$ đẳng hướng $\Rightarrow N:4\to18$, Phần 7), rồi cuối cùng — Phần 8
này — định lượng chi phí $T_{\text{iter}}$ trước/sau bước densify đó và mô tả đúng cách ghi ra
`point_cloud.ply` **63 cột thật** (`construct_list_of_attributes`/`save_ply` của
`SADGS/scene/gaussian_model.py`, bao gồm cột `filter_3D` mà 3DGS gốc không có).

---

[← Mục lục lời giải](00-muc-luc-loi-giai.md) | [Đề bài](../16-bai-toan-lon-de-bai.md) | [Mục lục sách](../00-muc-luc.md)
