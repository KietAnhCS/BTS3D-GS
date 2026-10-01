# Công thức toán học của `rasterizer_impl.cu` / `rasterizer_impl.h` / `rasterizer.h` / `config.h`

**Lưu ý đồng bộ:** File này đã được đồng bộ lại với bản đối chiếu/kiểm chứng kỹ hơn tại
`MATH/submodules/diff-gaussian-rasterization_structgs/rasterizer_impl.md` (đọc lại source thật
gần đây nhất, 599 dòng). Khi có mâu thuẫn với các phiên bản cũ, bản đó được ưu tiên — nội dung
dưới đây là bản rút gọn nhưng giữ đúng toàn bộ công thức, trích dẫn code và số dòng đã kiểm
chứng ở đó. Đọc file kia để xem phần dẫn giải đầy đủ (chứng minh đại số, ví dụ số chi tiết).

Phạm vi: toàn bộ giai đoạn **tile-based binning** của rasterizer — chia màn hình thành lưới tile,
với mỗi Gaussian xác định *chính xác* tập tile mà nó phủ tới (không phải bbox vuông xấp xỉ), sinh
khoá sort 64-bit `(tile_id, depth)`, radix-sort, rồi dò biên mỗi tile trong danh sách đã sort.

Nguồn đã đọc: `cuda_rasterizer/rasterizer_impl.cu`, `rasterizer_impl.h`, `rasterizer.h`, `config.h`,
và các hàm lõi `getRect` / `duplicateToTilesTouched` / `computeEllipseIntersection` / `processTiles`
trong `auxiliary.h` (được gọi trực tiếp từ `rasterizer_impl.cu`, nên phải đọc cùng nhau mới ra
đúng công thức).

---

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $W,H$ | chiều rộng/cao ảnh (pixel) |
| $\text{BLOCK\_X}=16,\ \text{BLOCK\_Y}=16$ | kích thước 1 tile (`config.h` dòng 16–17) |
| $\text{grid}_x,\text{grid}_y$ | số tile theo trục x, y |
| $n=\text{grid}_x\text{grid}_y$ | tổng số tile |
| $P$ | số Gaussian |
| $\mathbf p_i=(p_{x,i},p_{y,i})$ | tâm Gaussian $i$ đã chiếu lên ảnh (pixel), `points_xy_image` |
| $\Sigma_i\in\mathbb R^{2\times2}$ | hiệp phương sai 2D chiếu của Gaussian $i$ |
| $\text{con\_o}_i=(a_i,b_i,c_i,\alpha_i)$ | $(\Sigma_i^{-1})_{11},(\Sigma_i^{-1})_{12},(\Sigma_i^{-1})_{22}$ và opacity; gọi là "conic + opacity" |
| $d_i=\text{depths}[i]$ | độ sâu camera-space ($p_{\text{view}}.z$), luôn $>0.2$ do near-culling |
| $t_i$ | ngưỡng mức-đồng-mức (level set) của Gaussian ứng với ngưỡng đóng góp $1/255$ |
| $\text{mult}$ | hệ số nong/co ellipse (tham số `mult` truyền từ Python) |
| $\text{tiles\_touched}_i$ | số tile Gaussian $i$ phủ tới (đúng theo hình elip, không phải bbox vuông) |
| $\text{offset}_i$ | tổng tiền tố bao hàm (inclusive prefix sum) của `tiles_touched` |
| $L=\text{num\_rendered}$ | tổng số cặp (Gaussian, tile) = $\text{offset}_{P-1}$ |
| $\text{key}_j\in\{0,1\}^{64}$ | khoá sort của bản sao thứ $j$ |
| $\text{ranges}[\tau]=[\text{ranges}[\tau].x,\ \text{ranges}[\tau].y)$ | đoạn chỉ số trong danh sách đã sort ứng với tile $\tau$ |

---

## 2. Lưới tile (tile grid)

`config.h`:

```c
#define BLOCK_X 16
#define BLOCK_Y 16
```

`rasterizer_impl.cu`, dòng 412 (và lặp lại ở `backward`, dòng 610):

```cpp
dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
```

$$
\boxed{\ \text{grid}_x=\Big\lceil \frac{W}{\text{BLOCK\_X}}\Big\rceil,\qquad \text{grid}_y=\Big\lceil \frac{H}{\text{BLOCK\_Y}}\Big\rceil\ }
\tag{dòng 412}
$$

Đây là **chia nguyên làm tròn lên** (ceiling division) viết bằng số nguyên: với $W,\text{BLOCK\_X}>0$,

$$
\Big\lfloor\frac{W+\text{BLOCK\_X}-1}{\text{BLOCK\_X}}\Big\rfloor=\Big\lceil\frac{W}{\text{BLOCK\_X}}\Big\rceil
$$

(đẳng thức số học chuẩn, đúng với mọi $W\ge0$).

---

## 3. `tiles_touched` và prefix sum — `GeometryState`, `forward()`

`tiles_touched[i]` được ghi trong `FORWARD::preprocess` (`forward.cu`, không nằm trong các file được giao ở đây, nhưng là input bắt buộc để hiểu bước tiếp theo — xem mục 4). `rasterizer_impl.cu` dùng CUB để tính **tổng tiền tố bao hàm (inclusive scan)**:

```cpp
// rasterizer_impl.cu:293 (trong GeometryState::fromChunk, cấp phát scratch) và :459 (chạy thật)
CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
    geomState.tiles_touched, geomState.point_offsets, P), debug)
```

$$
\text{offset}_i=\sum_{j=0}^{i}\text{tiles\_touched}_j,\qquad i=0,\dots,P-1
\tag{dòng 459}
$$

Tổng số bản sao cần render:

```cpp
// :463
cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);
```

$$
L=\text{num\_rendered}=\text{offset}_{P-1}=\sum_{j=0}^{P-1}\text{tiles\_touched}_j
\tag{dòng 463}
$$

$L$ dùng để cấp phát `BinningState` (mảng khoá/giá trị kích thước $L$, dòng 465–467).

---

## 4. Số tile một Gaussian phủ tới — ellipse mức-đồng-mức (SnugBox), **không phải bbox vuông bán kính**

### 4.1 Có một phiên bản cũ (đã bị comment, KHÔNG chạy)

`rasterizer_impl.cu` dòng 152–222 còn một `duplicateWithKeys` bị comment, dùng `getRect` (`auxiliary.h:47-57`) — bbox hình **vuông bán kính cố định** $r=$ `radii[idx]` (bán kính = $\lceil 3\sqrt{\max(\lambda_1,\lambda_2)}\rceil$, dòng 261–263 `forward.cu`):

$$
\text{rect}_{min}=\Big(\big\lfloor\tfrac{x-r}{\text{BLOCK\_X}}\big\rfloor,\ \big\lfloor\tfrac{y-r}{\text{BLOCK\_Y}}\big\rfloor\Big),\quad
\text{rect}_{max}=\Big(\big\lceil\tfrac{x+r}{\text{BLOCK\_X}}\big\rceil,\ \big\lceil\tfrac{y+r}{\text{BLOCK\_Y}}\big\rceil\Big)
$$

(kẹp vào $[0,\text{grid}]$). Đây **chỉ còn là tham chiếu lịch sử**, mọi tile trong hình vuông này đều bị tính kể cả 4 góc không giao với ellipse thật — gây dư cặp (Gaussian, tile) giả.

### 4.2 Code thực sự chạy: `duplicateToTilesTouched` + `processTiles` (`auxiliary.h`)

Gọi tại `forward.cu:268` (chỉ đếm, truyền `nullptr` cho hai mảng khoá/giá trị) và `rasterizer_impl.cu:471-480` (`duplicateWithKeys`, ghi thật).

**Bước 4.2.1 — tính hợp lệ của conic.**

```cpp
float disc = con_o.y * con_o.y - con_o.x * con_o.z;      // auxiliary.h:324
if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) return 0;  // :327
```

$$
\text{disc}=b^2-ac
\tag{dòng 324}
$$

Ma trận $\Sigma^{-1}=\begin{pmatrix}a&b\\b&c\end{pmatrix}$ xác định dương $\iff$ $a>0$ và $\det\Sigma^{-1}=ac-b^2>0\iff \text{disc}=b^2-ac<0$. Code kiểm tra đúng cả 2 điều kiện (minor chính 1×1 và định thức) — dòng 327.

**Bước 4.2.2 — ngưỡng mức-đồng-mức $t$.**

```cpp
float t = 2.0f * log(con_o.w * 255.0f);   // :332
t = mult * t;                              // :333
```

$$
t=\text{mult}\cdot 2\ln(255\,\alpha_i)
\tag{dòng 332–333}
$$

*Dẫn giải:* điều kiện Gaussian còn đóng góp $\ge 1/255$ là $\alpha\, G(\mathbf x)\ge\tfrac1{255}$ với $G(\mathbf x)=\exp\!\big(-\tfrac12(\mathbf x-\mathbf p)^\top\Sigma^{-1}(\mathbf x-\mathbf p)\big)$. Lấy $\ln$:

$$
-\tfrac12(\mathbf x-\mathbf p)^\top\Sigma^{-1}(\mathbf x-\mathbf p)\ge \ln\tfrac1{255\alpha}
\iff (\mathbf x-\mathbf p)^\top\Sigma^{-1}(\mathbf x-\mathbf p)\le 2\ln(255\alpha)=t
$$

(khi $\text{mult}=1$). Vậy $t$ chính xác là bán kính bình phương (theo metric $\Sigma^{-1}$) của ellipse mức-đồng-mức ứng ngưỡng $1/255$; `mult` nong/co ellipse này (↑mult → ellipse lớn hơn → nhiều tile hơn).

**Bước 4.2.3 — vị trí (không phải độ dài!) điểm cực trị theo mỗi trục.**

```cpp
float x_term = sqrt(-(con_o.y*con_o.y*t)/(disc*con_o.x));
x_term = (con_o.y < 0) ? x_term : -x_term;
float y_term = sqrt(-(con_o.y*con_o.y*t)/(disc*con_o.z));
y_term = (con_o.y < 0) ? y_term : -y_term;

float2 bbox_argmin = { p.y - y_term, p.x - x_term };
float2 bbox_argmax = { p.y + y_term, p.x + x_term };
```

*Dẫn giải:* trên ellipse $a\,dx^2+2b\,dx\,dy+c\,dy^2=t$ ($dx=x-p_x,dy=y-p_y$), coi là phương trình bậc 2 theo $dx$ với $dy$ cố định:

$$
a\,dx^2+2b\,dy\,dx+(c\,dy^2-t)=0
$$

Có nghiệm thực $\iff$ biệt thức $\ge0$: $4b^2dy^2-4a(cdy^2-t)\ge0\iff dy^2(b^2-ac)\ge -at\iff dy^2\le\dfrac{-at}{\text{disc}}$ (chia cho $\text{disc}<0$ nên đổi chiều). Vậy **độ dài cực trị theo $y$** là

$$
dy_{\max}=\sqrt{\frac{-at}{\text{disc}}}
$$

Điểm đạt cực trị đó có nghiệm kép $dx^\*=-\dfrac{b\,dy_{\max}}{a}=-\,\text{sign}(b)\sqrt{\dfrac{-b^2t}{\text{disc}\cdot a}}$. Đặt $x_{term}:=dx^\*$:

$$
\boxed{\ x_{term}=-\,\text{sign}(b)\sqrt{\dfrac{-b^2 t}{\text{disc}\cdot a}}\ }
$$

Code: `x_term = (b<0) ? +\sqrt{\cdot} : -\sqrt{\cdot}` $\equiv -\text{sign}(b)\sqrt{\cdot}$ (quy ước $\text{sign}(0)=+1$ vô hại vì khi đó tử số $=0$) — **khớp chính xác**. Tương tự bằng vai trò $a\leftrightarrow c$ cho $y_{term}$. Vậy `bbox_argmin.y` $=p_x-x_{term}$ là **hoành độ** của điểm thấp nhất ($y$ nhỏ nhất) trên ellipse — không phải nửa chiều rộng — đúng như code đặt tên "argmin/argmax" (vị trí đạt cực trị), **khác bbox vuông đồng trục vì $x_{term}\ne0$ khi $b\ne0$** (Gaussian nghiêng thì đỉnh/đáy không nằm thẳng ngay dưới tâm).

**Bước 4.2.4 — giao điểm ellipse với một đường thẳng** (`computeEllipseIntersection`, dòng 179–194):

```cpp
float h = coord - p_u;
float sqrt_term = sqrt(disc*h*h + t*coeff);
return { (-con_o.y*h - sqrt_term)/coeff + p_v, (-con_o.y*h + sqrt_term)/coeff + p_v };
```

Với đường $u=\text{coord}$ ($u\in\{x,y\}$ tuỳ `isY`, $\text{coeff}=a$ nếu `isY` else $c$), đặt $h=\text{coord}-p_u$:

$$
v_{\pm}(h)=\frac{-b\,h\pm\sqrt{\text{disc}\cdot h^2+t\cdot\text{coeff}}}{\text{coeff}}+p_v
\tag{dòng 179–194}
$$

*Kiểm tra nhanh:* thế $u=p_u+h,\ v=v_\pm$ vào $a\,dx^2+2bdxdy+cdy^2=t$ với $(dx,dy)=(h,v_\pm-p_v)$ khi `isY=false` ($\text{coeff}=c$): phương trình bậc 2 theo $dy$ là $c\,dy^2+2bh\,dy+(ah^2-t)=0\Rightarrow dy=\dfrac{-bh\pm\sqrt{b^2h^2-c(ah^2-t)}}{c}=\dfrac{-bh\pm\sqrt{(b^2-ac)h^2+ct}}{c}=\dfrac{-bh\pm\sqrt{\text{disc}\,h^2+t\,c}}{c}$ — đúng khớp công thức code (coeff=c khi isY=false). ✓.

**Bước 4.2.5 — bbox ôm khít ellipse:**

```cpp
float2 bbox_min = { computeEllipseIntersection(..., true,  bbox_argmin.x).x,
                     computeEllipseIntersection(..., false, bbox_argmin.y).x };
float2 bbox_max = { computeEllipseIntersection(..., true,  bbox_argmax.x).y,
                     computeEllipseIntersection(..., false, bbox_argmax.y).y };
```

Tại $y=\text{bbox\_argmin}.x=p_y-y_{term}$ (mức $y$ nơi $x$ đạt cực tiểu toàn cục), nghiệm kép của giao điểm theo $x$ (`isY=true`) chính là $x_{\min}$ toàn cục; tương tự cho $x_{\max},y_{\min},y_{\max}$. Đây là bbox trục-song-song **ôm khít** ellipse (tight axis-aligned bounding box), không phải bbox vuông bán kính $r$.

**Bước 4.2.6 — quy đổi ra chỉ số tile (rect hình chữ nhật bao ngoài, chỉ dùng để giới hạn vòng lặp quét):**

```cpp
int2 rect_min = { max(0, min((int)grid.x, (int)(bbox_min.x / BLOCK_X))),
                   max(0, min((int)grid.y, (int)(bbox_min.y / BLOCK_Y))) };
int2 rect_max = { max(0, min((int)grid.x, (int)(bbox_max.x / BLOCK_X + 1))),
                   max(0, min((int)grid.y, (int)(bbox_max.y / BLOCK_Y + 1))) };
int y_span = rect_max.y - rect_min.y;
int x_span = rect_max.x - rect_min.x;

// If no tiles are touched, return 0
if (y_span * x_span == 0) {
    return 0;
}
```

$$
\text{rect}_{min}=\Big(\big\lfloor\tfrac{\text{bbox}_{\min,x}}{\text{BLOCK\_X}}\big\rfloor,\big\lfloor\tfrac{\text{bbox}_{\min,y}}{\text{BLOCK\_Y}}\big\rfloor\Big)_{+},\quad
\text{rect}_{max}=\Big(\big\lfloor\tfrac{\text{bbox}_{\max,x}}{\text{BLOCK\_X}}\big\rfloor+1,\big\lfloor\tfrac{\text{bbox}_{\max,y}}{\text{BLOCK\_Y}}\big\rfloor+1\Big)_{+}
$$

(ký hiệu $(\cdot)_+$ = kẹp vào $[0,\text{grid}]$). *Lưu ý:* code dùng `(int)(...)` — **truncation về 0**, không phải `floor` — nhưng vì kết quả luôn được `max(0,\cdot)` kẹp ngay sau, với mọi giá trị âm cả hai cách cho cùng kết quả sau khi kẹp; với giá trị không âm trunc≡floor. Vậy **không phải bug**, chỉ là điểm cần làm rõ khi đối chiếu ký hiệu toán học.

**Bước 4.2.7 — quét theo dải mỏng hơn, đếm tile thật sự giao ellipse (`processTiles`, dòng 196–310):**

Chọn trục quét có ít dải hơn để giảm số lần gọi `computeEllipseIntersection`:

$$
\text{isY}=\big(\text{grid}_y\text{-span} < \text{grid}_x\text{-span}\big),\qquad \text{span}=\text{rect}_{max}-\text{rect}_{min}
$$

Nếu `isY`, hoán đổi vai trò trục $x\leftrightarrow y$ trong toàn bộ dữ liệu (dòng 214–223). Sau đó, với mỗi dải $u\in[\text{rect}_{min,x},\text{rect}_{max,x})$ (bề rộng $\text{BLOCK\_U}$):

1. Tính 2 đường biên dải $[\text{min\_line},\text{max\_line}]=[u\cdot\text{BLOCK\_U},(u+1)\text{BLOCK\_U}]$.
2. Lấy giao điểm ellipse tại 2 biên đó (nếu còn trong bbox) $\to$ cho khoảng $v$ mà ellipse chiếm trong dải này: $(\text{ellipse}_{min},\text{ellipse}_{max})$. Nếu điểm cực trị toàn cục ($\text{bbox}_{argmin/argmax}$) rơi đúng vào dải này thì dùng trực tiếp $\text{bbox}_{min/max}$ (vì đó là điểm xa nhất, giao tuyến hai bên không đạt tới).
3. Quy đổi sang chỉ số tile theo trục kia và cộng dồn:

$$
\text{min\_tile}_v=\Big(\big\lfloor\tfrac{\text{ellipse}_{\min}}{\text{BLOCK\_V}}\big\rfloor\Big)_{\text{kẹp}\,[\text{rect}_{min,y},\text{rect}_{max,y}]},\quad
\text{max\_tile}_v=\Big(\big\lfloor\tfrac{\text{ellipse}_{\max}}{\text{BLOCK\_V}}\big\rfloor+1\Big)_{\text{kẹp}\,[\text{rect}_{min,y},\text{rect}_{max,y}]}
$$

$$
\boxed{\ \text{tiles\_touched}=\sum_{u=\text{rect}_{min,x}}^{\text{rect}_{max,x}-1}\big(\text{max\_tile}_v(u)-\text{min\_tile}_v(u)\big)\ }
$$

tức **tổng số tile thật sự cắt ellipse trên từng dải** — luôn $\le$ số tile trong $\text{rect}_{min}..\text{rect}_{max}$ (bbox vuông), bằng khi ellipse gần tròn và lấp đầy bbox, nhỏ hơn hẳn khi ellipse dẹt/nghiêng (4 góc bbox không có phần ellipse nào).

Mỗi tile $(u,v)$ được giữ (nếu `gaussian_keys_unsorted != nullptr`) sinh ra đúng 1 cặp khoá/giá trị — khớp đúng 1:1 với `tiles_touched` đã đếm ở `forward.cu:268` (vì cùng một hàm `duplicateToTilesTouched`/`processTiles`, chỉ khác việc có ghi ra mảng hay không) $\Rightarrow$ không có sai lệch giữa bước đếm và bước ghi.

---

## 5. Sinh khoá sắp xếp `(tile_id, depth)` — bit-packing

`duplicateWithKeys` (`rasterizer_impl.cu:121-150`), chạy 1 thread/Gaussian:

```cpp
if (tiles_touched[idx] > 0) {
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    duplicateToTilesTouched(points_xy[idx], con_o[idx], grid, mult, idx, off, depths[idx],
        gaussian_keys_unsorted, gaussian_values_unsorted);
}
```

$$
\text{off}_i=\begin{cases}0 & i=0\\ \text{offset}_{i-1} & i>0\end{cases}
\tag{dòng 121–150}
$$

(đây chính là **tổng tiền tố loại trừ — exclusive prefix sum**, suy ra từ mảng inclusive-sum bằng cách dịch chỉ số 1, không cần chạy scan lần hai.)

Bên trong `processTiles` (`auxiliary.h:297-299`), với mỗi tile $(u,v)$ (tile id hàng-chính $\tau=\text{row}\cdot\text{grid}_x+\text{col}$, xem mục 5.1):

```cpp
uint64_t key = isY ? (u * grid.x + v) : (v * grid.x + u);
key <<= 32;
key |= *((uint32_t*)&depth);
gaussian_keys_unsorted[off] = key;
gaussian_values_unsorted[off] = idx;
off++;
```

$$
\boxed{\ \text{key}=(\tau\ll 32)\ \big|\ \text{bitcast}_{32\to u32}(\text{depth})\ }
\tag{dòng 297–302}
$$

**Bố cục 64 bit** (bit 63 ở trái, bit 0 ở phải):

```
bit 63 ───────────────────── 32 31 ───────────────────── 0
│          tile_id (32 bit)         │   bitcast(depth) (32 bit)   │
```

### 5.1 Vì sao $\tau=u\cdot\text{grid}_x+v$ luôn là `row*grid_x+col` dù có hoán trục `isY`

Khi `isY=true`, dòng 214–223 đã hoán `rect/bbox/bbox_argmin/argmax` theo $x\leftrightarrow y$ **trước khi** vào vòng lặp, nên biến `u` trong vòng lặp chính là **chỉ số tile theo hàng** (gốc là $y$), còn `v` (biến nội suy từ `ellipse_min/max`) là **chỉ số theo cột** (gốc $x$). Vậy `key = u*grid.x + v = row*grid_x + col`.
Khi `isY=false`, `u` là cột, `v` là hàng, code viết `key = v*grid.x + u = row*grid_x + col`. Cả 2 nhánh cho **cùng một công thức** $\tau=\text{row}\cdot\text{grid}_x+\text{col}$ — nhất quán, không có lệch tile-id giữa 2 nhánh quét.

### 5.2 Không tràn / không chồng lấp giữa 2 phần của khoá

- Dịch `<<32` rồi `|=` với giá trị 32-bit $\Rightarrow$ 32 bit cao và 32 bit thấp **tách biệt tuyệt đối theo cấu trúc bit**, không thể chồng lấp bất kể giá trị cụ thể.
- **Phần tile-id (32 bit cao):** giá trị tối đa thực tế $\tau_{\max}=n-1=\text{grid}_x\text{grid}_y-1$. Với $W,H$ hợp lý (vài chục nghìn pixel mỗi chiều) thì $n\ll 2^{32}$ — không tràn trong thực tế.
- **Phần depth (32 bit thấp):** $\text{bitcast}_{32}(\text{depth})$ là *toàn bộ* biểu diễn IEEE-754 của `float`, nên **không có bit nào bị cắt** — khớp 1-1 giữa 32 bit và giá trị depth.

### 5.3 Depth bitcast có bảo toàn thứ tự không? — điểm cần kiểm chứng, không phải hiển nhiên

So sánh 2 số `uint32_t` tái diễn giải từ bit pattern của `float` **chỉ** bảo toàn đúng thứ tự số học khi **cả hai float đều $\ge 0$** (với IEEE-754: với số không âm, thứ tự bit-pattern-as-unsigned-int trùng thứ tự số thực). Nếu depth âm, bit dấu = 1 khiến bit-pattern unsigned rất lớn dù giá trị số âm rất nhỏ $\Rightarrow$ thứ tự sẽ **sai hoàn toàn**.

Code đảm bảo bất biến $\text{depth}>0$ bằng near-plane culling trong `auxiliary.h::in_frustum`:

```cpp
if (p_view.z <= 0.2f) { ... return false; }   // Gaussian bị loại, không tới duplicateWithKeys
```

và `depths[idx] = p_view.z` (`forward.cu:283`). Do đó mọi Gaussian còn sống đến `duplicateWithKeys` có $d_i>0.2>0$, nên phép bit-cast **hợp lệ và bảo toàn thứ tự tăng dần theo depth** trong mỗi tile sau khi sort. Đây là một bất biến ngầm (implicit invariant) giữa `forward.cu` và `rasterizer_impl.cu` — nếu sau này ai đó bỏ near-culling hoặc dùng depth có thể âm, bit-packing này sẽ **âm thầm sai thứ tự** mà không có lỗi runtime nào báo.

---

## 6. Radix sort toàn cục theo khoá

```cpp
// BinningState::fromChunk, :335-338 (chỉ cấp phát scratch dò kích thước)
cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size,
    binning.point_list_keys_unsorted, binning.point_list_keys,
    binning.point_list_unsorted, binning.point_list, P);
...
// forward(), :495-503 (chạy thật)
int bit = getHigherMsb(tile_grid.x * tile_grid.y);
cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space, binningState.sorting_size,
    binningState.point_list_keys_unsorted, binningState.point_list_keys,
    binningState.point_list_unsorted, binningState.point_list,
    num_rendered, 0, 32 + bit);
```

### 6.1 `getHigherMsb` tính gì?

```cpp
uint32_t getHigherMsb(uint32_t n) {
    uint32_t msb = sizeof(n) * 4;   // = 16
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (n >> msb) msb += step; else msb -= step;
    }
    if (n >> msb) msb++;
    return msb;
}
```

Đây là **tìm nhị phân (binary search) vị trí bit** nhỏ nhất $m$ sao cho $n\gg m=0$, tức $n<2^m$. Kết quả chính là **số bit cần để biểu diễn $n$ ở hệ nhị phân** (bit-length):

$$
\text{getHigherMsb}(n)=\lfloor\log_2 n\rfloor+1=\lceil\log_2(n+1)\rceil,\qquad n\ge1
\tag{dòng 1–11 của hàm}
$$

**Lưu ý quan trọng — khác với công thức `bit=⌈log₂(grid_x·grid_y)⌉` hay bị viết nhầm:** công thức đó **sai khi $n$ là luỹ thừa của 2**. Ví dụ $n=4$: $\lceil\log_2 4\rceil=2$ nhưng chạy tay thuật toán trên cho $n=4$:

```
msb=16,step=16 → step=8,  4>>16=0 → msb=8
                 step=4,  4>>8=0  → msb=4
                 step=2,  4>>4=0  → msb=2
                 step=1,  4>>2=1  → msb=3   (loop dừng vì step=1)
check 4>>3=0 → không +1. Trả về 3.
```

$\text{getHigherMsb}(4)=3=\lceil\log_2 5\rceil$, không phải $2$. Công thức đúng là $\boxed{\text{bit}=\lceil\log_2(n+1)\rceil}$ (tương đương bit-length của $n$). Đây **không phải bug trong code** (chỉ cần bit đủ lớn để phủ mọi tile id $0..n-1$, dư 1 bit chỉ tốn thêm 1 "pass" radix không đáng kể).

### 6.2 Vì sao chỉ sort trên $32+\text{bit}$ bit thấp là đủ và đúng

`SortPairs(..., num_rendered, 0, 32+bit)` sort trên các bit $[0, 32+\text{bit})$ của khoá 64-bit (dòng 495–503). Phần depth chiếm trọn 32 bit thấp $[0,32)$ — luôn cần sort đủ. Phần tile-id nằm ở $[32,64)$; vì $\tau<n\le 2^{\text{bit}}$ nên **mọi bit của $\tau$ đều nằm trong $[32,32+\text{bit})$, các bit $[32+\text{bit},64)$ của mọi khoá đều bằng 0** $\Rightarrow$ bỏ qua chúng không ảnh hưởng kết quả sort. Kết quả: toàn bộ $L$ cặp được sắp theo đúng thứ tự $(\tau,\text{depth})$ tăng dần.

---

## 7. Xác định biên mỗi tile trong danh sách đã sort — `identifyTileRanges`

```cpp
// rasterizer_impl.cu:227-252
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L) return;
    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    bool valid_tile = currtile != (uint32_t)-1;
    if (idx == 0) ranges[currtile].x = 0;
    else {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile) {
            ranges[prevtile].y = idx;
            if (valid_tile) ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1 && valid_tile) ranges[currtile].y = L;
}
```

$$
\tau(idx)=\text{key}_{idx}\gg 32
\tag{dòng 227–252}
$$

$$
\text{ranges}[\tau(idx-1)].y=idx,\qquad \text{ranges}[\tau(idx)].x=idx \quad\text{khi } \tau(idx)\ne\tau(idx-1)
$$

biên đặc biệt $\text{ranges}[\tau(0)].x=0$, $\text{ranges}[\tau(L-1)].y=L$.

### 7.1 Đây **không phải binary search**

Mỗi thread $idx$ chỉ so sánh **một cặp lân cận** $(\tau(idx-1),\tau(idx))$ — thao tác $O(1)$ cho mỗi thread, chạy song song trên cả $L$ thread cùng lúc (`<<<(num_rendered+255)/256,256>>>`, dòng 509-512). Đây là thuật toán **"dò biên song song" (parallel boundary detection / stream compaction kiểu "mark changes")**: tổng công việc $O(L)$ nhưng độ sâu song song (parallel depth) $O(1)$ vì mọi thread chạy đồng thời — **không có bước tìm kiếm nhị phân $O(\log L)$ nào cho từng tile**.

Tính đúng đắn: vì mảng đã sort tăng dần theo $\tau$, với mỗi tile $\tau$ giá trị này chỉ "xuất hiện liên tục" (một đoạn liền), nên chỉ cần phát hiện đúng 2 điểm chuyển tiếp (nơi $\tau$ thay đổi) là đủ xác định trọn vẹn $[\text{start},\text{end})$ của mọi tile.

`cudaMemset(imgState.ranges, 0, ...)` (dòng 505) trước khi gọi kernel đảm bảo tile không có Gaussian nào phủ tới có $\text{ranges}=[0,0)$ (đoạn rỗng) thay vì rác bộ nhớ.

---

## 8. Bucket hoá cho backward (`perTileBucketCount`)

```cpp
// :255-264
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount) {
    ...
    uint2 range = ranges[idx];
    int num_splats = range.y - range.x;
    int num_buckets = (num_splats + 31) / 32;
    bucketCount[idx] = (uint32_t) num_buckets;
}
```

$$
\text{num\_buckets}(\tau)=\Big\lceil\frac{\text{ranges}[\tau].y-\text{ranges}[\tau].x}{32}\Big\rceil
\tag{dòng 255–264}
$$

(`(x+31)/32` là công thức chia-làm-tròn-lên nguyên chuẩn, giống mục 2, với "kích thước khối" = 1 warp = 32 lane). Sau đó `bucket_offsets` = inclusive-scan của `bucket_count` theo tile (dòng 518), dùng cấp phát `SampleState` (lưu $T,ar$ theo bucket phục vụ gradient checkpointing trong `BACKWARD::render`).

---

## 9. `rasterizer.h` / `rasterizer_impl.h` — không có công thức mới

- `rasterizer.h`: khai báo interface tĩnh `markVisible / forward / backward` của `CudaRasterizer::Rasterizer`, chỉ là chữ ký hàm, không chứa phép toán.
- `rasterizer_impl.h`: khai báo 4 struct `GeometryState/ImageState/BinningState/SampleState` (layout bộ nhớ) và hàm phụ trợ `obtain` (căn chỉnh địa chỉ bộ nhớ theo `alignment`, dòng 23–28):

```cpp
// rasterizer_impl.h:23-28
std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
```

$$
\text{offset}=(\text{addr}(chunk)+\text{align}-1)\ \&\ \sim(\text{align}-1)
\tag{dòng 23–28}
$$

là kỹ thuật **căn lề bit (bit-masking để làm tròn lên bội số của luỹ thừa 2)** chuẩn trong cấp phát bộ nhớ thủ công.

---

## 10. Kiến thức toán nền tảng

**(a) Chia hết làm tròn lên (ceiling integer division).**
$\lceil a/b\rceil=\lfloor (a+b-1)/b\rfloor$ với $a\ge0,b>0$: dùng ở mục 2 (số tile), mục 8 (số bucket).

**(b) Thao tác bit (bitwise packing / bit-level encoding).**
Ghép 2 trường 32-bit vào 1 khoá 64-bit bằng `<<` và `|` biến bài toán "sort theo (tile, depth) từ điển" thành "sort theo 1 số nguyên 64-bit", vì thứ tự từ điển trên $(\tau,d)$ trùng thứ tự số học trên $\tau\cdot2^{32}+d$ khi $0\le d<2^{32}$.

**(c) Radix sort (LSD) — `cub::DeviceRadixSort`.**
Sort tuần tự theo từng "chữ số" (digit) từ thấp lên cao bằng counting sort ổn định; vì ổn định qua mọi vòng, kết quả cuối đúng thứ tự trên toàn bộ các bit đã chọn.

**(d) Prefix sum / scan song song — `cub::DeviceScan::InclusiveSum`.**
Scan song song tính $\text{offset}_i=\sum_{j\le i}x_j$ trong $O(\log P)$ bước song song, cho mỗi Gaussian "biết" vị trí ghi của riêng mình mà không cần đồng bộ hoá tuần tự.

**(e) Tìm biên đoạn trong mảng đã sort (run-length boundary detection), không phải binary search.**
So sánh phần tử $idx$ với $idx-1$ để phát hiện "điểm đổi giá trị" là một dạng *segmented array / run detection*: độ phức tạp song song $O(1)$/thread, $O(L)$ tổng.

**(f) Hình học ellipse mức-đồng-mức & dạng toàn phương.**
$\{\mathbf x:(\mathbf x-\mathbf p)^\top\Sigma^{-1}(\mathbf x-\mathbf p)=t\}$ là ellipse khi $\Sigma^{-1}$ xác định dương. Tìm bbox trục-song-song ôm khít một ellipse nghiêng là bài toán cực trị có ràng buộc, giải bằng điều kiện biệt thức $=0$.

---

## 11. Kiểm chứng tính đúng sai — tổng hợp

| # | Nội dung | Kết luận |
|---|---|---|
| 1 | Số tile lưới $\lceil W/\text{BLOCK\_X}\rceil\times\lceil H/\text{BLOCK\_Y}\rceil$ | **Đúng**, suy từ `(W+BLOCK_X-1)/BLOCK_X` (dòng 412) |
| 2 | Bit-packing khoá $\text{key}=(\tau\ll32)\,|\,\text{bitcast}_{32}(d)$ | **Đúng cấu trúc bit**; chỉ đúng thứ tự khi $d>0$ (đảm bảo bởi near-culling $z>0.2$) |
| 3 | Thuật toán đếm/ghi tile Gaussian phủ (SnugBox/AccuTile, không phải bbox vuông) | **Đúng**, dẫn giải lại bằng đại số (biệt thức phương trình bậc 2), khớp từng dòng code |
| 4 | Số bit `bit` cho radix sort | $\text{bit}=\lceil\log_2(\text{grid}_x\text{grid}_y)\rceil$ **SAI khi $n$ là luỹ thừa của 2**; công thức đúng $\text{bit}=\lceil\log_2(n+1)\rceil=$ bit-length của $n$ (mục 6.1) |
| 5 | `identifyTileRanges` | **Không phải binary search** — dò biên song song $O(1)$/thread (mục 7.1) |
| 6 | `(int)(bbox/BLOCK)` | Truncation-về-0, không phải `floor`; nhưng vì bị `max(0,\cdot)` kẹp ngay sau nên kết quả cuối không đổi — không phải bug |
| 7 | Bug chức năng? | Không phát hiện trong các file này (ngoài đoạn code cũ bị comment, không chạy). Rủi ro refactor cần lưu ý: (i) depth âm làm sai thứ tự bit-packing; (ii) $n>2^{32}$ tràn tile-id (không khả thi với ảnh thực tế) |

---

## 12. Ví dụ số

**Thiết lập ảnh:** $W=H=32$, $\text{BLOCK\_X}=\text{BLOCK\_Y}=16\Rightarrow \text{grid}_x=\text{grid}_y=\lceil32/16\rceil=2$, $n=4$ tile. Quy ước tile id $\tau=\text{row}\cdot\text{grid}_x+\text{col}$:

$$
\tau=0:(row0,col0)=[0,16)\times[0,16),\quad \tau=1:(row0,col1)=[16,32)\times[0,16)
$$
$$
\tau=2:(row1,col0)=[0,16)\times[16,32),\quad \tau=3:(row1,col1)=[16,32)\times[16,32)
$$

Chọn **2 Gaussian đẳng hướng (isotropic, $b=0$)** — ellipse mức-đồng-mức là **hình tròn bán kính** $R=\sqrt{t/a}$ (vì $a=c$), và $x_{term}=y_{term}=0$.

### Gaussian A — chạm cả 4 tile

$\mathbf p_A=(15,15)$, $\text{con\_o}_A=(a,b,c,\alpha)=(1,0,1,1)$ ($\Sigma_A=I_2$), $\text{mult}=1$, $d_A=2.0$.

- $t_A=2\ln(255\cdot1)=2\ln255\approx11.0821$.
- $R_A=\sqrt{t_A/1}\approx3.3290$.
- bbox: $x,y\in[11.671,\,18.329]$ — vượt cả 2 biên tile $x=16$ và $y=16$.
- $\text{rect}_{min}=(0,0)$, $\text{rect}_{max}=(2,2)$ $\Rightarrow$ $x$-span $=y$-span$=2\Rightarrow\text{isY}=\text{false}$ (quét theo $x$).

Quét dải $u=0$ và $u=1$ (chi tiết tính giao điểm ellipse xem mục 4.2.4), mỗi dải chạm 2 tile theo $v$:

$$
\text{tiles\_touched}_A=2+2=4\quad(\tau=0,1,2,3\text{ — chạm cả 4 tile})
$$

Thứ tự ghi khoá theo code (`key=v*grid_x+u` khi `isY=false`): dải $u=0$: $v=0\to\tau=0$, $v=1\to\tau=2$; dải $u=1$: $v=0\to\tau=1$, $v=1\to\tau=3$. Vậy 4 khoá ghi theo thứ tự $\tau=0,2,1,3$.

### Gaussian B — chỉ chạm 1 tile

$\mathbf p_B=(4,4)$, $\text{con\_o}_B=(a,b,c,\alpha)=(4,0,4,0.5)$ ($\Sigma_B=0.25I_2$), $\text{mult}=1$, $d_B=1.0$.

- $t_B=2\ln(255\cdot0.5)=2\ln127.5\approx9.6977$.
- $R_B=\sqrt{t_B/4}=\sqrt{2.4244}\approx1.5571$.
- bbox: $x,y\in[2.443,5.557]\subset[0,16)$ cả hai trục $\Rightarrow$ chỉ 1 tile.

$$
\text{tiles\_touched}_B=1\quad(\tau=0)
$$

### Prefix sum, offset, khoá 64-bit

Thứ tự Gaussian trong batch: $i=0$ là A, $i=1$ là B $\Rightarrow\text{tiles\_touched}=[4,1]$.

$$
\text{offset}=[4,\,5]\ (\text{inclusive scan}),\qquad L=\text{num\_rendered}=5
$$

$$
\text{off}_0=0\ (i=0),\qquad \text{off}_1=\text{offset}_0=4
$$

Mảng chưa sort (vị trí : (tile, depth, gaussian)) theo đúng thứ tự ghi ở trên:

| vị trí | $\tau$ | depth | Gaussian |
|---|---|---|---|
| 0 | 0 | 2.0 | A |
| 1 | 2 | 2.0 | A |
| 2 | 1 | 2.0 | A |
| 3 | 3 | 2.0 | A |
| 4 | 0 | 1.0 | B |

**Bitcast IEEE-754** của `float`: $2.0_{10}=$ `0x40000000` $=1073741824_{10}$; $1.0_{10}=$ `0x3F800000` $=1065353216_{10}$.

Khoá 64-bit mẫu (hex), $\text{key}=(\tau\ll32)\,|\,\text{bitcast}(d)$:

$$
\text{key}(\tau{=}0,d{=}2.0)=\texttt{0x0000000040000000}=1073741824
$$
$$
\text{key}(\tau{=}2,d{=}2.0)=\texttt{0x0000000240000000}=9663676416
$$
$$
\text{key}(\tau{=}1,d{=}2.0)=\texttt{0x0000000140000000}=5368709120
$$
$$
\text{key}(\tau{=}3,d{=}2.0)=\texttt{0x0000000340000000}=13958643712
$$
$$
\text{key}(\tau{=}0,d{=}1.0)=\texttt{0x000000003F800000}=1065353216
$$

(so 2 khoá cùng $\tau=0$: $1065353216<1073741824$ đúng theo $d_B=1.0<d_A=2.0$; so 2 khoá khác $\tau$: luôn cách nhau $\ge2^{32}$ $\Rightarrow$ sort theo khoá 64-bit **luôn ưu tiên tile trước, depth sau**.)

### Sort và ranges

Sort tăng dần theo khoá $\Rightarrow$ nhóm theo $\tau$ rồi theo $d$:

| vị trí sau sort | $\tau$ | $d$ | Gaussian |
|---|---|---|---|
| 0 | 0 | 1.0 | B |
| 1 | 0 | 2.0 | A |
| 2 | 1 | 2.0 | A |
| 3 | 2 | 2.0 | A |
| 4 | 3 | 2.0 | A |

Chạy tay `identifyTileRanges`:

- $idx=0$: $\tau(0)=0$, biên trái $\Rightarrow\text{ranges}[0].x=0$.
- $idx=1$: $\tau(1)=0=\tau(0)$, không đổi.
- $idx=2$: $\tau(2)=1\ne\tau(1)=0\Rightarrow\text{ranges}[0].y=2,\ \text{ranges}[1].x=2$.
- $idx=3$: $\tau(3)=2\ne\tau(2)=1\Rightarrow\text{ranges}[1].y=3,\ \text{ranges}[2].x=3$.
- $idx=4=L-1$: $\tau(4)=3\ne\tau(3)=2\Rightarrow\text{ranges}[2].y=4,\ \text{ranges}[3].x=4$; đồng thời do $idx=L-1\Rightarrow\text{ranges}[3].y=5$.

$$
\text{ranges}[0]=[0,2),\quad \text{ranges}[1]=[2,3),\quad \text{ranges}[2]=[3,4),\quad \text{ranges}[3]=[4,5)
$$

Khi tile $\tau=0$ được blend (alpha-compositing), nó xử lý 2 Gaussian theo đúng thứ tự gần→xa: **B ($d=1.0$) trước, A ($d=2.0$) sau** — đúng ngữ nghĩa "front-to-back" cần cho alpha blending.

### Số bit sort

$n=\text{grid}_x\text{grid}_y=4\Rightarrow\text{getHigherMsb}(4)=3$ (chạy tay ở mục 6.1) $\Rightarrow$ sort trên $32+3=35$ bit thấp nhất.

### Bucket hoá (minh hoạ mục 8)

$\text{ranges}[0].y-\text{ranges}[0].x=2\Rightarrow\text{num\_buckets}(0)=\lceil2/32\rceil=1$; tương tự $\text{num\_buckets}(1)=\text{num\_buckets}(2)=\text{num\_buckets}(3)=\lceil1/32\rceil=1$. Tổng $\text{bucket\_offsets}=[1,2,3,4]$, `bucket_sum=4` $\Rightarrow$ `SampleState` cấp phát cho $4\times\text{BLOCK\_SIZE}=4\times256=1024$ phần tử.

---

## Bảng tương ứng cú pháp ↔ công thức (tra nhanh)

| Code | Công thức LaTeX |
|---|---|
| `(width + BLOCK_X - 1) / BLOCK_X` | $\text{grid}_x = \lceil W/\text{BLOCK\_X}\rceil$ |
| `cub::DeviceScan::InclusiveSum(..., tiles_touched, point_offsets, P)` | $\text{offset}_i = \sum_{j\le i}\text{tiles\_touched}_j$ |
| `off = (idx == 0) ? 0 : offsets[idx - 1]` | $\text{off}_i = \text{offset}_{i-1}$ (hoặc 0 nếu $i=0$) |
| `key = tile_id; key <<= 32; key |= *(uint32_t*)&depth` | $\text{key}_i = (\text{tile\_id}_i\ll 32) \mathbin\| \text{bitcast}_{32}(\text{depth}_i)$ |
| `t = 2.0f*log(con_o.w*255.0f); t = mult*t` | $t = \text{mult}\cdot 2\ln(255\alpha_i)$ |
| `disc = con_o.y*con_o.y - con_o.x*con_o.z` | $\text{disc}=b^2-ac$ |
| `x_term = sqrt(-(b*b*t)/(disc*a))` (có dấu) | $x_{term}=-\text{sign}(b)\sqrt{-b^2t/(\text{disc}\cdot a)}$ |
| `computeEllipseIntersection(...)` trả `{v_min, v_max}` | $v_\pm = \dfrac{-bh\pm\sqrt{\text{disc}\,h^2+t\cdot\text{coeff}}}{\text{coeff}}+p_v$ |
| `rect_min = (int)(bbox_min.x/BLOCK_X)`, `rect_max = (int)(bbox_max.x/BLOCK_X + 1)` | $\text{rect}_{min}=\lfloor \text{bbox}_{min}/\text{BLOCK}\rfloor,\ \text{rect}_{max}=\lfloor\text{bbox}_{max}/\text{BLOCK}\rfloor+1$ |
| `tiles_count += max_tile_v - min_tile_v` (trong `processTiles`) | số tile thực giao với elip trong dải $u$ |
| `currtile = key >> 32` | $\text{tile}(idx)=\text{key}_{idx}\gg 32$ |
| `ranges[prevtile].y = idx; ranges[currtile].x = idx` | biên $[\text{ranges}[\tau].x,\text{ranges}[\tau].y)$ |
| `getHigherMsb(tile_grid.x * tile_grid.y)` | $\text{bit}=\lceil\log_2(n+1)\rceil$ (bit-length của $n$ — **đã sửa**, không phải $\lceil\log_2 n\rceil$) |
| `SortPairs(..., 0, 32 + bit)` | sort trên $32+\text{bit}$ bit thấp của khoá 64-bit |
| `num_buckets = (num_splats + 31) / 32` | $\lceil (\text{range}.y-\text{range}.x)/32\rceil$ |
| `getRect(...)` (code cũ, đã comment, KHÔNG dùng) | bbox hình vuông bán kính $r$ — chỉ để tham chiếu lịch sử |
