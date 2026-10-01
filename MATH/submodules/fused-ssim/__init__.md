# Công thức toán học của `fused_ssim/__init__.py`

File này là lớp wrapper Python (custom `torch.autograd.Function`) bọc quanh extension CUDA `fused_ssim_cuda` (biên dịch từ `ext.cpp` + `ssim.cu`). Nó cung cấp một phép tính SSIM "fused" (gộp forward+backward trong kernel CUDA) nhanh hơn cách tính SSIM bằng chuỗi `conv2d` thông thường (xem `tests/test.py`, hàm `_ssim` tham chiếu).

---

## Ký hiệu

- $x_1=\text{img1}\in\mathbb R^{B\times C\times H\times W}$: ảnh render (có gradient, "ground-truth dự đoán").
- $x_2=\text{img2}\in\mathbb R^{B\times C\times H\times W}$: ảnh ground-truth cố định (không cần gradient).
- $C_1=0.01^2,\ C_2=0.03^2$: hằng số ổn định SSIM chuẩn (Wang et al. 2004).
- $w$: cửa sổ Gaussian $11\times11$ (tham số cố định trong kernel CUDA, bán kính $r=5$).
- $\mu_1,\mu_2$: trung bình cục bộ của $x_1,x_2$ qua cửa sổ $w$ (tính trong kernel, không lộ ra Python).
- $\sigma_1^2,\sigma_2^2,\sigma_{12}$: phương sai/hiệp phương sai cục bộ.
- $y=\text{ssim\_map}\in\mathbb R^{B\times C\times H\times W}$ (hoặc đã cắt biên nếu `padding="valid"`): bản đồ SSIM điểm-ảnh, $y=\text{SSIM}(x_1,x_2)$.
- $\dfrac{\partial\mathcal L}{\partial y}=\text{opt\_grad}$ (còn gọi `dL_dmap`): gradient loss ngoài truyền vào qua `backward`.
- $\dfrac{\partial m}{\partial \mu_1},\dfrac{\partial m}{\partial \sigma_1^2},\dfrac{\partial m}{\partial \sigma_{12}}$ (biến Python: `dm_dmu1`, `dm_dsigma1_sq`, `dm_dsigma12`): ba đạo hàm riêng của bản đồ SSIM $m$ theo ba thống kê cục bộ của $x_1$ — đây là các tensor trung gian do forward CUDA tính sẵn, lưu lại để backward dùng theo quy tắc dây chuyền (chain rule), tránh phải tính lại conv2d.
- $\mathbb{1}[\cdot]$: hàm chỉ thị (indicator), dùng khi mô tả phép cắt biên (`padding="valid"`).

---

## 1. Lớp `FusedSSIMMap(torch.autograd.Function)`

### 1.1. `forward` (dòng 9–21)

```python
@staticmethod
def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
    ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

    if padding == "valid":
        ssim_map = ssim_map[:, :, 5:-5, 5:-5]

    ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
    ctx.C1 = C1
    ctx.C2 = C2
    ctx.padding = padding

    return ssim_map
```

**Diễn giải toán học.** Hàm gọi thẳng vào extension CUDA `fusedssim` (dòng 11), là hàm `fusedssim` khai báo trong `ssim.h` (dòng 7–14) và bind sang Python qua `ext.cpp` dòng 5 (`m.def("fusedssim", &fusedssim)`). Về công thức, kernel này tính SSIM chuẩn theo cửa sổ trượt Gaussian:

$$
\mu_1=w*x_1,\qquad \mu_2=w*x_2,\qquad
\sigma_1^2=w*x_1^2-\mu_1^2,\qquad
\sigma_2^2=w*x_2^2-\mu_2^2,\qquad
\sigma_{12}=w*(x_1x_2)-\mu_1\mu_2
$$

$$
y=\text{SSIM}(x_1,x_2)=\frac{(2\mu_1\mu_2+C_1)(2\sigma_{12}+C_2)}{(\mu_1^2+\mu_2^2+C_1)(\sigma_1^2+\sigma_2^2+C_2)}
$$

(công thức này suy từ chính đối chứng `_ssim` ở `tests/test.py` dòng 34–49, được assert là khớp bit-với-bit tại dòng 82 `assert torch.isclose(og_ssim_val, mine_ssim_val_same)`). Ở mức giao diện (interface-level), có thể viết gọn:

$$
y=\text{forward}(x_1,x_2)=\text{SSIM}(x_1,x_2)\in\mathbb R^{B\times C\times H\times W}
$$

Ngoài $y$, kernel CUDA còn trả thêm ba tensor đạo hàm riêng một phần đã tính sẵn (để tiết kiệm công tính lại ở backward):

$$
\frac{\partial y}{\partial\mu_1},\qquad \frac{\partial y}{\partial\sigma_1^2},\qquad \frac{\partial y}{\partial\sigma_{12}}
$$

tức `dm_dmu1, dm_dsigma1_sq, dm_dsigma12` ở dòng 11.

**Cắt biên (`padding="valid"`), dòng 13–14.** Cửa sổ $11\times11$ có bán kính $5$ px; ở chế độ `"valid"` các pixel biên (không đủ 11×11 hàng xóm thật, bị ảnh hưởng bởi padding ngầm của conv2d) bị loại:

$$
y_{\text{valid}} = y\big[:,:,5:H-5,\,5:W-5\big]
$$

tương đương $y_{\text{valid},i,j}=y_{i+5,j+5}$ với $i\in[0,H-11],\,j\in[0,W-11]$.

**Lưu trạng thái cho backward (dòng 16–19).**

```python
ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
ctx.C1 = C1
ctx.C2 = C2
ctx.padding = padding
```

Chú ý: `img1` được `.detach()` trước khi lưu — chỉ lưu **giá trị số** của $x_1$, không lưu đồ thị tính toán của nó (vì nếu không detach, autograd engine có thể giữ một tham chiếu vòng/graph thừa; bản thân gradient của $x_1$ sẽ được autograd gắn vào output của `backward`, không phải thông qua graph lưu trong `ctx`). `img2` lưu nguyên (không detach), nhưng vì `img2` không bao giờ được gán gradient (xem 1.2), việc không detach không gây vấn đề.

### 1.2. `backward` (dòng 23–32)

```python
@staticmethod
def backward(ctx, opt_grad):
    img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
    C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
    dL_dmap = opt_grad
    if padding == "valid":
        dL_dmap = torch.zeros_like(img1)
        dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
    grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
    return None, None, grad, None, None, None
```

**Diễn giải toán học.** `opt_grad` chính là $\dfrac{\partial\mathcal L}{\partial y}$ do autograd truyền ngược từ phía sau (ví dụ từ `.mean()` ở `fused_ssim`). Nếu `padding="valid"`, gradient chỉ tồn tại ở vùng đã giữ lại ở forward; phần biên bị cắt được gán gradient $0$ (vì chúng không đóng góp vào $y_{\text{valid}}$, nên không đóng góp vào $\mathcal L$):

$$
\frac{\partial\mathcal L}{\partial y}\Big|_{i,j}=
\begin{cases}
\text{opt\_grad}_{i-5,j-5}, & 5\le i<H-5,\ 5\le j<W-5\\[4pt]
0, & \text{ngược lại}
\end{cases}
$$

Sau đó gọi kernel backward CUDA `fusedssim_backward` (dòng 31), tức hàm khai báo trong `ssim.h` dòng 16–26, bind qua `ext.cpp` dòng 6 (`m.def("fusedssim_backward", &fusedssim_backward)`). Về mặt công thức, kernel dùng quy tắc dây chuyền qua ba thống kê trung gian đã lưu:

$$
\frac{\partial\mathcal L}{\partial x_1}
=\frac{\partial\mathcal L}{\partial y}\cdot\frac{\partial y}{\partial x_1}
=\frac{\partial\mathcal L}{\partial y}\cdot\left(
\frac{\partial y}{\partial\mu_1}\frac{\partial\mu_1}{\partial x_1}
+\frac{\partial y}{\partial\sigma_1^2}\frac{\partial\sigma_1^2}{\partial x_1}
+\frac{\partial y}{\partial\sigma_{12}}\frac{\partial\sigma_{12}}{\partial x_1}
\right)
$$

trong đó $\dfrac{\partial\mu_1}{\partial x_1},\dfrac{\partial\sigma_1^2}{\partial x_1},\dfrac{\partial\sigma_{12}}{\partial x_1}$ là các toán tử convolution tuyến tính với cùng cửa sổ $w$ (vì $\mu_1=w*x_1$, $\sigma_1^2=w*x_1^2-\mu_1^2$, $\sigma_{12}=w*(x_1x_2)-\mu_1\mu_2$ — đều tuyến tính/đa thức bậc thấp theo $x_1$ dưới một convolution cố định), và được tính lại bằng một convolution "ngược" ngay trong kernel CUDA (đây chính là lý do ba tensor `dm_dmu1, dm_dsigma1_sq, dm_dsigma12` cần được lưu từ forward: chúng đóng vai trò $\partial y/\partial(\cdot)$ trong công thức trên). Ở mức giao diện:

$$
\boxed{\ \frac{\partial\mathcal L}{\partial x_1}=\text{backward}\!\left(\frac{\partial\mathcal L}{\partial y}\right)=\text{fusedssim\_backward}(C_1,C_2,x_1,x_2,\,dL\_dmap,\,dm\_d\mu_1,\,dm\_d\sigma_1^2,\,dm\_d\sigma_{12})\ }
$$

**Gradient trả về (dòng 32).** `backward` trả 6 giá trị tương ứng 6 tham số của `forward` theo đúng thứ tự `(ctx, C1, C2, img1, img2, padding, train)` (bỏ `ctx`):

| Tham số forward | `C1` | `C2` | `img1` | `img2` | `padding` | `train` |
|---|---|---|---|---|---|---|
| Gradient trả về | `None` | `None` | `grad` | `None` | `None` | `None` |

Nghĩa là chỉ $x_1$ (`img1`) nhận gradient; $x_2$ (`img2`, ground-truth) nhận `None` — xác nhận đúng như kỳ vọng: ground-truth cố định không cần (và không có) gradient. $C_1,C_2,\text{padding},\text{train}$ không phải tensor cần gradient nên cũng `None`.

### 1.3. Hàm tiện ích mức cao `fused_ssim` (dòng 34–41)

```python
def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    assert padding in allowed_padding
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()
```

Hàm này **không** tính loss $1-\text{SSIM}$ — nó chỉ trả về SSIM trung bình trên toàn bộ bản đồ:

$$
\overline{\text{SSIM}}=\frac{1}{BCHW'}\sum_{b,c,i,j} y_{b,c,i,j}
$$

(với $H',W'$ là kích thước sau cắt biên nếu `padding="valid"`). Muốn dùng làm loss SSIM kiểu chuẩn (như trong 3D Gaussian Splatting gốc) thì nơi gọi phải tự lấy $\mathcal L_{ssim}=1-\overline{\text{SSIM}}$; công thức này **không nằm trong file `__init__.py`** — cần xác nhận lại ở nơi gọi (ví dụ `scene/` hoặc `utils/loss_utils.py` nếu có) khi ghép vào tổng loss huấn luyện.

### 1.4. Hàm tiện ích `fused_ssim_` (dòng 43–50)

```python
def fused_ssim_(img1, img2, padding="same", train=True):
    ...
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.squeeze(0).mean(0)
```

Giống hệt `fused_ssim` nhưng không rút gọn về một số vô hướng: bỏ chiều batch (giả định $B=1$) rồi lấy trung bình theo chiều kênh $C$, trả về bản đồ SSIM 2 chiều $H'\times W'$:

$$
\text{map}_{-}=\frac{1}{C}\sum_{c=1}^{C} y_{0,c,:,:}
$$

Dùng để trực quan hoá SSIM theo từng pixel, không phải một giá trị loss duy nhất.

---

## 2. Kiến thức toán nền tảng

- **Automatic differentiation (vi phân tự động) kiểu "reverse-mode"**: PyTorch xây đồ thị tính toán động (dynamic computation graph) khi forward chạy, rồi duyệt ngược để tính gradient. Với các phép toán built-in (ví dụ `conv2d`), PyTorch tự sinh backward. Nhưng ở đây forward được cài đặt thủ công bằng kernel CUDA bên ngoài đồ thị autograd chuẩn của PyTorch (không gồm toàn `nn.functional` ops), nên **không thể** để PyTorch tự suy backward — phải khai báo tường minh.
- **Custom `torch.autograd.Function`**: là cơ chế PyTorch cho phép định nghĩa một "node" tùy ý trong đồ thị autograd bằng cách cài đặt cặp `forward`/`backward` thủ công. Hợp đồng bắt buộc:
  - `forward(ctx, *inputs) -> outputs`: tính giá trị tiến, và dùng `ctx.save_for_backward(...)` để lưu đúng những tensor cần cho đạo hàm (không lưu thừa để tiết kiệm bộ nhớ, không lưu thiếu vì sẽ gây lỗi/NaN khi backward).
  - `backward(ctx, *grad_outputs) -> grad_inputs`: nhận gradient của loss theo **output**, trả về gradient của loss theo **từng input theo đúng thứ tự và số lượng** của `forward` (input nào không cần gradient thì trả `None`). Đây chính là áp dụng quy tắc dây chuyền ở mức "một khối" (block-wise chain rule) thay vì để autograd tự ghép từng phép toán nguyên tố.
  - `ctx` là nơi duy nhất mang trạng thái từ forward sang backward (tensor qua `save_for_backward`, scalar/metadata qua gán thuộc tính trực tiếp như `ctx.C1`).
- **PyTorch C++/CUDA extension dispatch**: `ext.cpp` dùng `PYBIND11_MODULE` để export hai hàm C++ (`fusedssim`, `fusedssim_backward`, cài đặt thật trong `ssim.cu`) thành một Python module gọi được (`fused_ssim_cuda`), được `__init__.py` `import` trực tiếp ở dòng 4. Khi gọi `fusedssim(...)` từ Python, lệnh gọi đi qua pybind11 binding sang hàm C++ đã biên dịch, hàm này launch CUDA kernel (song song hoá theo pixel/cửa sổ trượt) rồi trả tensor kết quả ngược lại Python — không đi qua bất kỳ autograd op nguyên tố nào của PyTorch, nên **bắt buộc** phải bọc trong `torch.autograd.Function` để có gradient.

---

## 3. Kiểm chứng tính đúng sai

### 3.1. `ctx.save_for_backward` có lưu đủ tensor cho backward không?

So khớp trực tiếp giữa chữ ký `backward` cần gì và `forward` lưu gì:

| Backward (dòng 25, `ctx.saved_tensors`) cần | Có được `save_for_backward` ở dòng 16 lưu không? |
|---|---|
| `img1` | ✓ (lưu dạng `img1.detach()`) |
| `img2` | ✓ |
| `dm_dmu1` | ✓ |
| `dm_dsigma1_sq` | ✓ |
| `dm_dsigma12` | ✓ |

Và các giá trị không-tensor `C1, C2, padding` được lưu qua `ctx.C1/ctx.C2/ctx.padding` (dòng 17–19) — không cần `save_for_backward` vì không phải tensor tham gia graph.

**Kết luận: ĐÚNG.** `save_for_backward` lưu chính xác và đầy đủ 5 tensor mà `backward` cần — không thừa, không thiếu. Đây là một ví dụ cài đặt custom autograd Function không mắc lỗi phổ biến "quên lưu tensor cần cho backward".

### 3.2. `padding` có ảnh hưởng đúng đến công thức không?

- Forward: cắt $y\to y[:,:,5:-5,5:-5]$ khi `padding="valid"` (dòng 13–14).
- Backward: phải "đệm ngược" gradient về đúng kích thước gốc trước khi đưa vào kernel backward (vốn luôn làm việc trên kích thước đầy đủ $H\times W$, vì nó cần tích chập lại trên toàn ảnh để tính $\partial\mu_1/\partial x_1$ v.v.):

```python
if padding == "valid":
    dL_dmap = torch.zeros_like(img1)
    dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
```

Kiểm tra kích thước: `img1` có shape gốc $(B,C,H,W)$ (chưa cắt, vì `img1` lưu từ trước lúc cắt — dòng 16 lưu `img1.detach()` là ảnh gốc, không phải `ssim_map` đã cắt). `torch.zeros_like(img1)` do đó đúng shape $H\times W$, và gán `opt_grad` (shape $(H-10)\times(W-10)$) vào đúng vùng `[5:-5, 5:-5]` — khớp chính xác với vùng đã giữ lại ở forward (`[:, :, 5:-5, 5:-5]`, dòng 14). Hai chỗ cắt/đệm dùng cùng một chỉ số `5:-5` nên khớp nhau tuyệt đối, không lệch biên.

**Kết luận: ĐÚNG.** Biến `padding` được xử lý nhất quán giữa forward (cắt output) và backward (đệm gradient về đúng vị trí đã cắt, phần biên gradient $=0$ đúng về mặt toán học vì $\partial(\text{vùng bị cắt bỏ})/\partial\mathcal L=0$ do vùng đó không còn đóng góp vào loss).

### 3.3. `train` có ảnh hưởng gì, và có nguy cơ mismatch không?

Nhìn vào `ssim.cu` (không phải file được giao đọc sâu, nhưng cần đối chiếu để xác minh điểm này):

```cpp
// ssim.cu, hàm fusedssim
torch::Tensor dm_dmu1 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
torch::Tensor dm_dsigma1_sq = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
torch::Tensor dm_dsigma12 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
```

Khi `train=False`, ba tensor trung gian `dm_dmu1, dm_dsigma1_sq, dm_dsigma12` là **tensor rỗng** (`torch::empty(0)`), không phải tensor rỗng-nhưng-đúng-shape — tức kernel **không tính** các đạo hàm riêng này khi `train=False` (tối ưu tốc độ cho suy luận thuần, đúng như cách `test.py` dòng 153–157 dùng `train=False` chỉ để đo thời gian inference, không gọi `.backward()`).

**Đây là điểm cần lưu ý (mismatch tiềm ẩn):** `FusedSSIMMap.forward` **không kiểm tra** giá trị `train` trước khi gọi `ctx.save_for_backward(..., dm_dmu1, dm_dsigma1_sq, dm_dsigma12)` ở dòng 16. Nếu người dùng gọi `fused_ssim(img1, img2, train=False)` rồi vẫn (vô tình) gọi `.backward()` trên kết quả, `ctx.saved_tensors` sẽ chứa các tensor rỗng (`numel()=0`), và `fusedssim_backward` ở dòng 31 sẽ nhận các tensor rỗng này làm đối số `dm_dmu1, dm_dsigma1_sq, dm_dsigma12` — kernel CUDA bên trong (dòng 322, 335, 348 của `ssim.cu`, hàm `load_into_shared`) sẽ đọc ra ngoài vùng cấp phát (vì nó indexing theo `H,W,CH` thật nhưng tensor có 0 phần tử), dẫn tới lỗi runtime CUDA hoặc đọc bộ nhớ rác, **không có gradient chính xác**.

**Kết luận: SAI (có lỗ hổng tiềm ẩn) nếu dùng sai cách.** Bản thân cặp `forward`/`backward` *đúng* trong điều kiện sử dụng chuẩn (`train=True` khi cần gradient, đúng như giá trị mặc định của tham số `train=True` ở dòng 10 và dòng 34). Nhưng API không có `assert`/kiểm tra bảo vệ cho trường hợp người dùng set `train=False` mà vẫn gọi `.backward()` — đây là trách nhiệm được đẩy ngầm cho người gọi (convention: `train=False` ⇒ chỉ dùng cho inference, không bao giờ backward), không phải lỗi logic trong *luồng đúng*, nhưng là một API thiếu guard rõ ràng. Không có mismatch nào giữa input mà forward cần và input mà backward dùng trong *trường hợp bình thường* (`train=True`), chỉ có rủi ro khi người gọi vi phạm hợp đồng ngầm định của tham số `train`.

### 3.4. Tóm tắt kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| `save_for_backward` đủ tensor cho backward (trường hợp `train=True`) | Đúng |
| Không lưu thừa tensor không cần thiết | Đúng |
| `padding="valid"` nhất quán giữa cắt (forward) và đệm (backward) | Đúng |
| Gradient chỉ chảy về `img1`, không về `img2` | Đúng, xác nhận bằng dòng 32: `return None, None, grad, None, None, None` |
| An toàn khi `train=False` + gọi `.backward()` | **Không an toàn** — tensor trung gian rỗng, cần người dùng tự đảm bảo không backward khi `train=False` |

---

## 4. Ví dụ số (minh hoạ luồng gọi forward → backward)

Dùng tensor cực nhỏ, tượng trưng, **không chạy kernel CUDA thật** (vì công thức SSIM đầy đủ cần cửa sổ $11\times11$ và không viết lại trong tài liệu này) — mục tiêu là minh hoạ đúng **luồng dữ liệu qua các lệnh gọi hàm**, không phải tái tạo số học bên trong kernel.

Giả sử $B=1, C=1$, ảnh nhỏ $H=W=16$ (đủ lớn hơn cửa sổ $11\times11$ để có vùng "valid" không rỗng), `padding="valid"`, `train=True`.

**Bước 1 — gọi `fused_ssim(img1, img2, padding="valid")`:**

$$
C_1=0.0001,\quad C_2=0.0009
$$

gọi `FusedSSIMMap.apply(C1, C2, img1, img2, "valid", True)`.

**Bước 2 — bên trong `forward`:**

$$
(y,\ dm\_d\mu_1,\ dm\_d\sigma_1^2,\ dm\_d\sigma_{12}) = \text{fusedssim}(C_1,C_2,x_1,x_2,\text{train=True})
$$

với $y\in\mathbb R^{1\times1\times16\times16}$, ba tensor đạo hàm riêng cùng shape $1\times1\times16\times16$.

Cắt biên do `padding="valid"`:

$$
y_{\text{valid}}=y[:,:,5:11,5:11]\in\mathbb R^{1\times1\times6\times6}
$$

Lưu `ctx`:

$$
\text{ctx.saved\_tensors}=(x_1,\ x_2,\ dm\_d\mu_1,\ dm\_d\sigma_1^2,\ dm\_d\sigma_{12})
$$
$$
\text{ctx.C1}=0.0001,\quad\text{ctx.C2}=0.0009,\quad\text{ctx.padding}="valid"
$$

Trả về $y_{\text{valid}}$ (shape $6\times6$).

**Bước 3 — `fused_ssim` lấy trung bình:**

$$
\overline{\text{SSIM}}=\frac{1}{36}\sum_{i=1}^{6}\sum_{j=1}^{6} y_{\text{valid},i,j}
$$

Giả sử kết quả số là $\overline{\text{SSIM}}=0.8421$ (một scalar, `requires_grad=True` qua $x_1$).

**Bước 4 — gọi `.backward()` (ví dụ dùng trực tiếp làm loss $\mathcal L=1-\overline{\text{SSIM}}$ ở nơi gọi ngoài file này):**

$$
\frac{\partial\mathcal L}{\partial\overline{\text{SSIM}}}=-1
$$

Autograd lan gradient qua `.mean()`:

$$
\frac{\partial\mathcal L}{\partial y_{\text{valid},i,j}}=\frac{\partial\mathcal L}{\partial\overline{\text{SSIM}}}\cdot\frac{1}{36}=-\frac{1}{36}\approx-0.02778\quad\forall i,j
$$

Đây chính là `opt_grad` truyền vào `FusedSSIMMap.backward(ctx, opt_grad)`, shape $1\times1\times6\times6$, mọi phần tử $=-0.02778$.

**Bước 5 — bên trong `backward`:**

Vì `padding="valid"`, đệm gradient về kích thước gốc $16\times16$:

$$
dL\_dmap = \mathbf 0_{1\times1\times16\times16},\qquad
dL\_dmap[:,:,5{:}11,5{:}11] = -0.02778
$$

(36 phần tử ở giữa bằng $-0.02778$, 220 phần tử biên còn lại bằng $0$).

Gọi kernel backward:

$$
\text{grad} = \text{fusedssim\_backward}(C_1,C_2,x_1,x_2,\ dL\_dmap,\ dm\_d\mu_1,\ dm\_d\sigma_1^2,\ dm\_d\sigma_{12})
$$

`grad` có shape giống $x_1$: $1\times1\times16\times16$ — đây chính là $\dfrac{\partial\mathcal L}{\partial x_1}$.

**Bước 6 — `backward` trả về:**

$$
(\text{None},\ \text{None},\ \text{grad},\ \text{None},\ \text{None},\ \text{None})
$$

PyTorch gán `grad` vào `img1.grad` (nếu `img1.requires_grad=True`); `img2.grad` không đổi (luôn `None` từ hàm này) vì vị trí thứ 4 (ứng với `img2`) nhận `None`.

**Tổng kết luồng số liệu:**

$$
x_1,x_2
\ \xrightarrow{\text{fusedssim (CUDA)}}\
y,\,dm\_d\mu_1,\,dm\_d\sigma_1^2,\,dm\_d\sigma_{12}
\ \xrightarrow{\text{cắt biên nếu valid}}\
y_{\text{valid}}
\ \xrightarrow{.\text{mean()}}\
\overline{\text{SSIM}}
$$

$$
\frac{\partial\mathcal L}{\partial\overline{\text{SSIM}}}
\ \xrightarrow{\text{lan qua mean}}\
\frac{\partial\mathcal L}{\partial y_{\text{valid}}}=\text{opt\_grad}
\ \xrightarrow{\text{đệm 0 nếu valid}}\
dL\_dmap
\ \xrightarrow{\text{fusedssim\_backward (CUDA)}}\
\frac{\partial\mathcal L}{\partial x_1}=\text{grad}
$$

---

## Tham chiếu dòng code

| Nội dung | File | Dòng |
|---|---|---|
| `forward` | `fused_ssim/__init__.py` | 9–21 |
| cắt biên `padding="valid"` (forward) | `fused_ssim/__init__.py` | 13–14 |
| `ctx.save_for_backward` | `fused_ssim/__init__.py` | 16 |
| `backward` | `fused_ssim/__init__.py` | 23–32 |
| đệm gradient `padding="valid"` (backward) | `fused_ssim/__init__.py` | 28–30 |
| gradient trả về (chỉ `img1`) | `fused_ssim/__init__.py` | 32 |
| `fused_ssim()` (SSIM trung bình, không phải loss) | `fused_ssim/__init__.py` | 34–41 |
| `fused_ssim_()` (bản đồ SSIM trung bình theo kênh) | `fused_ssim/__init__.py` | 43–50 |
| bind `fusedssim`, `fusedssim_backward` sang Python | `ext.cpp` | 4–7 |
| khai báo chữ ký C++ của hai hàm trên | `ssim.h` | 7–26 |
| `train` quyết định tensor trung gian rỗng hay không | `ssim.cu` | 384–386 |
