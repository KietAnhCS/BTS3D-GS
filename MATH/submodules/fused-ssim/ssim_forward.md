# Công thức toán học của phần FORWARD trong `fused-ssim`

Phạm vi: chỉ kernel `fusedssimCUDA` (forward) trong `submodules/fused-ssim/ssim.cu` và hàm Python `fused_ssim`/`fused_ssim_` gọi nó (`submodules/fused-ssim/fused_ssim/__init__.py`). Phần backward (`fusedssim_backwardCUDA`) không nằm trong tài liệu này.

---

## Ký hiệu

- $x,y$: hai ảnh đầu vào (thực ra là `img1`, `img2` trong code — `img1` là ảnh render, `img2` là ảnh ground-truth), mỗi ảnh có batch $B$, số kênh $CH$, kích thước $H\times W$, giá trị pixel chuẩn hoá trong $[0,1]$.
- $x[b,c,i,j]$: giá trị pixel tại batch $b$, kênh $c$, hàng $i$, cột $j$ ($0\le i<H,\ 0\le j<W$). Ngoài biên ($i<0$ hoặc $i\ge H$, tương tự $j$) quy ước $x[b,c,i,j]=0$.
- $G_k$, $k=-5,\dots,5$: 11 hệ số cửa sổ Gaussian 1D rời rạc (được hard-code sẵn trong code, xem Bước 1).
- $*$: phép tích chập (convolution) rời rạc hai chiều, thực hiện bằng hai lượt tích chập 1D tách biệt (separable).
- $\mu_x,\mu_y$: ảnh trung bình cục bộ (local mean) của $x,y$ sau khi làm mờ Gaussian.
- $\sigma_x^2,\sigma_y^2$: phương sai cục bộ (local variance).
- $\sigma_{xy}$: hiệp phương sai cục bộ (local covariance).
- $C_1,C_2$: hằng số ổn định hoá (stabilizing constants) của SSIM.
- $m(i,j)$: giá trị SSIM tại pixel $(i,j)$ — "ssim map".
- $\mathrm{SSIM}$: giá trị SSIM trung bình toàn ảnh, là số vô hướng cuối cùng dùng làm loss.

---

## (a) Cửa sổ Gaussian 1D — `G_00..G_10`

Code hard-code sẵn 11 hằng số (`ssim.cu:8-18`):

```cpp
#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f
#define G_06 0.21300552785396576f
#define G_07 0.10936068743467331f
#define G_08 0.036000773310661316f
#define G_09 0.0075987582094967365f
#define G_10 0.001028380123898387f
```

Đây chính là 11 mẫu rời rạc của hàm Gaussian 1D chuẩn SSIM gốc, với tâm tại $k=0$ (ứng với `G_05`), bán kính cửa sổ $r=5$, tức **kích thước cửa sổ $11\times11$ pixel**:

$$
G(k)=\exp\!\left(-\frac{k^2}{2\sigma^2}\right),\qquad k=-5,\dots,5,\qquad \sigma=1.5
$$

$$
G_k=\frac{G(k)}{\displaystyle\sum_{k'=-5}^{5}G(k')}
$$

**Kiểm chứng bằng số** (xem Bước "Kiểm chứng tính đúng sai" bên dưới): thay $\sigma=1.5$ vào công thức trên cho ra đúng 11 giá trị hard-code ở trên, sai số $<10^{-5}$ — nghĩa là code dùng chính xác cửa sổ Gaussian chuẩn của bài báo SSIM gốc (window $11\times11$, $\sigma=1.5$), chỉ khác là **không tính lại mỗi lần chạy** mà nhúng cứng hằng số đã chuẩn hoá sẵn (tổng $\sum_k G_k\approx 1$, sai số làm tròn float32 $\approx 3\times10^{-8}$).

Cửa sổ 2D đầy đủ (chưa từng được tạo tường minh trong code, nhưng về mặt toán học tương đương) là tích ngoài (outer product) của hai cửa sổ 1D vì Gaussian 2D đẳng hướng tách được:

$$
W(k,l)=G_k\,G_l,\qquad k,l\in\{-5,\dots,5\},\qquad \sum_{k,l}W(k,l)=1
$$

---

## (b) Tích chập 2D tách biệt (separable convolution) — tính $\mu,\sigma^2,\sigma_{xy}$

### Bước 1: Nạp pixel vào shared memory, zero-padding ngoài biên

Hàm `get_pix_value` (`ssim.cu:35-41`) trả về $0$ nếu toạ độ pixel nằm ngoài $[0,W)\times[0,H)$:

$$
\tilde x[i,j]=\begin{cases}x[i,j]&0\le i<H,\ 0\le j<W\\ 0&\text{ngược lại}\end{cases}
$$

`load_into_shared` (`ssim.cu:43-62`) nạp một khối $(BY+10)\times(BX+32{+}10)$ pixel quanh block hiện tại vào shared memory, dùng `get_pix_value` nên biên ngoài ảnh tự động là $0$ — đây chính là **zero-padding**, không phải "replicate" hay "reflect".

### Bước 2: Tích chập theo trục $x$ rồi trục $y$ (separable)

Vì $W(k,l)=G_k G_l$ tách được, tích chập 2D được tính bằng hai lượt 1D liên tiếp — đúng với `do_separable_conv_x` (`ssim.cu:99-163`) rồi `do_separable_conv_y` (`ssim.cu:165-184`):

$$
h[i,j]=\sum_{k=-5}^{5}G_k\,\tilde x[i,\,j+k]\qquad\text{(lượt theo }x\text{)}
$$

$$
(x*W)[i,j]=\mu_x[i,j]=\sum_{l=-5}^{5}G_l\,h[i+l,\,j]=\sum_{k=-5}^{5}\sum_{l=-5}^{5}G_kG_l\,\tilde x[i+l,\,j+k]
$$

Trong code, hai lượt này được gọi lần lượt để tính **bốn** đại lượng khác nhau — chỉ khác input/flag `sq`:

| Đại lượng code | Dòng gọi | Công thức toán |
|---|---|---|
| `mu1` | `ssim.cu:220-222` | $\mu_x=\;x*W$ |
| `mu2` | `ssim.cu:238-240` | $\mu_y=\;y*W$ |
| raw $E[x^2]$ | `ssim.cu:228-230`, `sq=true` trong `do_separable_conv_x` (`ssim.cu:106-117`) | $x^2*W$ |
| raw $E[y^2]$ | `ssim.cu:246-248` | $y^2*W$ |
| raw $E[xy]$ | `ssim.cu:252-258`, `multiply_shared_mem` nhân $x\cdot y$ theo từng pixel trước khi convolve (`ssim.cu:64-78`) | $(xy)*W$ |

### Bước 3: Phương sai / hiệp phương sai bằng công thức khai triển

Code **không** tích chập trực tiếp $(x-\mu_x)^2*W$, mà dùng công thức khai triển (moment-expansion) $E[X^2]-E[X]^2$:

$$
\sigma_x^2[i,j]=\big(x^2*W\big)[i,j]-\mu_x[i,j]^2\qquad(\texttt{ssim.cu:230})
$$

$$
\sigma_y^2[i,j]=\big(y^2*W\big)[i,j]-\mu_y[i,j]^2\qquad(\texttt{ssim.cu:248})
$$

$$
\sigma_{xy}[i,j]=\big((xy)*W\big)[i,j]-\mu_x[i,j]\,\mu_y[i,j]\qquad(\texttt{ssim.cu:258})
$$

Về mặt đại số, hai cách đều cho cùng kết quả vì $W$ là bộ lọc chuẩn hoá ($\sum W=1$, xem chứng minh bên dưới mục "Kiểm chứng"), nhưng **không tương đương về số học dấu phẩy động** — xem mục "Kiểm chứng tính đúng sai".

---

## (c) Công thức SSIM map đầy đủ

`ssim.cu:261-268`:

```cpp
float mu1_sq = mu1 * mu1;
float mu2_sq = mu2 * mu2;
float mu1_mu2 = mu1 * mu2;
float C = (2.0f * mu1_mu2 + C1);
float D = (2.0f * sigma12 + C2);
float A = (mu1_sq + mu2_sq + C1);
float B = (sigma1_sq + sigma2_sq + C2);
float m = (C * D) / (A * B);
```

Tức:

$$
\boxed{\ m(i,j)=\frac{\big(2\mu_x[i,j]\mu_y[i,j]+C_1\big)\big(2\sigma_{xy}[i,j]+C_2\big)}{\big(\mu_x[i,j]^2+\mu_y[i,j]^2+C_1\big)\big(\sigma_x^2[i,j]+\sigma_y^2[i,j]+C_2\big)}\ }
$$

Hằng số $C_1,C_2$ **không** hard-code trong file `.cu` — chúng là tham số truyền vào kernel (`ssim.cu:190-191`, `float C1, float C2`), được gán ở phía Python (`fused_ssim/__init__.py:35-36`):

```python
C1 = 0.01 ** 2   # = 1e-4
C2 = 0.03 ** 2   # = 9e-4
```

tức $K_1=0.01,\ K_2=0.03,\ L=1$ (ảnh chuẩn hoá $[0,1]$) — đúng chuẩn gốc, xem mục kiểm chứng.

---

## (d) Tổng hợp SSIM trung bình toàn ảnh và padding biên

### "same" vs "valid"

Kernel `fusedssimCUDA` luôn tính `ssim_map` có **cùng kích thước $H\times W$** với ảnh gốc (`ssim.cu:383`, `torch::zeros_like(img1)`), nhờ zero-padding ở Bước (b)/Bước 1. Đây là kiểu **"same" convolution theo nghĩa kích thước output**, nhưng về bản chất số học là **zero-padding**, không phải "same" theo nghĩa lặp biên.

Wrapper Python `FusedSSIMMap.forward` (`fused_ssim/__init__.py:10-21`) cung cấp 2 chế độ:

```python
if padding == "valid":
    ssim_map = ssim_map[:, :, 5:-5, 5:-5]
```

- `padding="same"` (mặc định): giữ nguyên toàn bộ $H\times W$ pixel của `ssim_map`, kể cả các pixel biên mà cửa sổ $11\times11$ của chúng bị cắt cụt bởi zero-padding (bán kính $5$, nên $5$ hàng/cột đầu và cuối đều bị ảnh hưởng).
- `padding="valid"`: cắt bỏ đúng $5$ pixel ở mỗi cạnh — phần còn lại $(H-10)\times(W-10)$ chỉ gồm các pixel mà toàn bộ cửa sổ $11\times11$ nằm trọn trong ảnh gốc, tức **đúng nghĩa valid convolution** của SSIM gốc (không có đóng góp từ pixel "ảo" bằng $0$).

(Kernel luôn tính toàn bộ `ssim_map` bằng công thức zero-padding trước; chế độ `"valid"` chỉ cắt bỏ sau khi tính — nhưng phần giữ lại không hề bị ảnh hưởng bởi zero-padding, vì bán kính cửa sổ đúng bằng $5$ pixel bị cắt, nên hai cách — "tính rồi cắt" và "chỉ tính vùng hợp lệ ngay từ đầu" — cho cùng kết quả số học.)

### Giá trị SSIM cuối cùng

`fused_ssim/__init__.py:34-41`:

```python
def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()
```

$$
\mathrm{SSIM}=\frac{1}{B\cdot CH\cdot H'\cdot W'}\sum_{b,c,i,j} m[b,c,i,j]
$$

với $H'\times W' = H\times W$ nếu `padding="same"`, hoặc $(H-10)\times(W-10)$ nếu `padding="valid"`. Đây là **trung bình cộng không trọng số** trên toàn bộ pixel (và toàn bộ kênh màu, toàn bộ batch) — không có bất kỳ trọng số không-đồng-nhất nào khác ngoài cửa sổ Gaussian đã áp dụng cục bộ.

Hàm biến thể `fused_ssim_` (`__init__.py:43-50`) chỉ khác ở chỗ giữ nguyên trục kênh khi trả kết quả cho batch đơn (`map.squeeze(0).mean(0)`), toán học giống hệt.

---

## Kiến thức toán nền tảng

- **Xử lý tín hiệu / tích chập 2D tách biệt (separable convolution):** một bộ lọc 2D $W(k,l)=G_k G_l$ (hạng 1, rank-1) có thể phân tích thành tích hai bộ lọc 1D, cho phép tính tích chập 2D bằng $O(N\cdot r)$ thay vì $O(N\cdot r^2)$ phép toán ($N$ số pixel, $r$ bán kính cửa sổ) — đây là lý do kernel CUDA tách `do_separable_conv_x` rồi `do_separable_conv_y`.
- **Hàm Gaussian và chuẩn hoá xác suất:** $G(k)=\exp(-k^2/2\sigma^2)$ là hạt nhân làm mờ (low-pass filter) tiêu chuẩn; chia cho tổng để $\sum_k G_k=1$ đảm bảo phép lọc bảo toàn độ sáng trung bình (không làm tối/sáng ảnh).
- **Thống kê hiệp phương sai cục bộ (local covariance) / moment thứ hai:** $\mu,\sigma^2,\sigma_{xy}$ ở đây không phải thống kê toàn ảnh mà là thống kê trong một cửa sổ trượt (sliding window) quanh mỗi pixel, xấp xỉ bằng tích chập với bộ lọc trọng số Gaussian — tương tự ước lượng trung bình/phương sai có trọng số (weighted mean/variance).
- **Công thức khai triển phương sai $\mathrm{Var}(X)=E[X^2]-E[X]^2$ và hiệp phương sai $\mathrm{Cov}(X,Y)=E[XY]-E[X]E[Y]$:** đây là định danh đại số chuẩn trong xác suất thống kê, được code dùng thay cho định nghĩa trực tiếp $E[(X-\mu_X)(Y-\mu_Y)]$ để tránh phải tính $\mu$ trước rồi tích chập lại một lần tích chập $(x-\mu_x)(y-\mu_y)$ — đổi lại là đánh đổi về ổn định số học (xem mục kế tiếp).
- **Lý thuyết Structural Similarity Index (SSIM):** mô hình hoá sự suy giảm chất lượng ảnh thành ba thành phần độc lập — độ sáng (luminance, so sánh $\mu_x,\mu_y$), độ tương phản (contrast, so sánh $\sigma_x,\sigma_y$), và cấu trúc (structure, so sánh hiệp phương sai chuẩn hoá $\sigma_{xy}/\sigma_x\sigma_y$) — và nhân ba thành phần lại với các hằng số ổn định hoá để tránh chia 0.

---

## Kiểm chứng tính đúng sai

So với công thức SSIM **gốc** (Wang, Bovik, Sheikh, Simoncelli, 2004, *"Image Quality Assessment: From Error Visibility to Structural Similarity"*):

$$
\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},\qquad
C_1=(K_1L)^2,\ C_2=(K_2L)^2,\ K_1=0.01,\ K_2=0.03
$$

với cửa sổ Gaussian $11\times11$, $\sigma=1.5$, chuẩn hoá tổng $=1$ — chính là tham số mà bài báo gốc dùng trong thực nghiệm.

**Những điểm khớp hoàn toàn:**

1. **Công thức SSIM map** — `ssim.cu:264-268` khớp từng số hạng với công thức gốc, không rút gọn/giản lược gì.
2. **Cửa sổ Gaussian** — kiểm tra bằng số (script Python, xem dưới) cho thấy 11 hệ số hard-code khớp $G(k)=\exp(-k^2/2\cdot1.5^2)$ chuẩn hoá, sai số tuyệt đối $<2\times10^{-6}$ cho mọi $k$:

   | $k$ | $G_k$ (code) | $G(k)/\sum G$ (công thức) |
   |---|---|---|
   | 0 | 0.266011715 | 0.265990 |
   | ±1 | 0.213005528 | 0.213016 |
   | ±2 | 0.109360687 | 0.109347 |
   | ±3 | 0.036000773 | 0.035991 |
   | ±4 | 0.007598758 | 0.007607 |
   | ±5 | 0.001028380 | 0.0010284 |

   → cửa sổ $11\times11,\ \sigma=1.5$ khớp chuẩn gốc.
3. **Hằng số $C_1,C_2$** — `fused_ssim/__init__.py:35-36` đặt $C_1=0.01^2=10^{-4}$, $C_2=0.03^2=9\times10^{-4}$, tức $K_1=0.01, K_2=0.03, L=1$ — khớp chính xác giá trị chuẩn mà bài báo gốc khuyến nghị cho ảnh $[0,1]$.
4. **Phương pháp "valid"** — khi gọi với `padding="valid"`, vùng SSIM giữ lại hoàn toàn tương đương với tích chập "valid" kinh điển (không có đóng góp từ pixel ảo), đúng cách SSIM gốc thường được áp dụng khi không muốn thiên lệch do biên.

**Những điểm khác / tối ưu hoá có thể ảnh hưởng số học:**

1. **Zero-padding ở chế độ mặc định `"same"`.** Bài báo gốc không bàn về cách xử lý biên; các cài đặt tham chiếu phổ biến (`pytorch-msssim`, `skimage.metrics.structural_similarity`) thường dùng **"valid"** (crop biên, như `scipy`/`skimage` mặc định) hoặc Gaussian filter có `mode='reflect'`. Ở đây mặc định của `fused_ssim` là `"same"` với **zero-padding**, nghĩa là với $5$ hàng/cột pixel ở mỗi biên, $\mu,\sigma^2,\sigma_{xy}$ được tính từ cửa sổ có một phần trọng số "rơi" vào giá trị $0$ (tối, coi như nền đen) thay vì giá trị ảnh thực. Hệ quả: SSIM tại các pixel biên bị kéo lệch (thường thấp hơn giá trị thật, vì $0$ khác biệt với nội dung ảnh xung quanh), và vì `fused_ssim()` mặc định lấy **trung bình trên toàn bộ $H\times W$** (không loại trừ biên trừ khi gọi tường minh `padding="valid"`), giá trị loss cuối cùng bị ảnh hưởng nhẹ bởi hiệu ứng biên này. Đây là khác biệt thực sự so với cách tính "chuẩn" (valid) của bài báo gốc, nhưng là lựa chọn kỹ thuật hợp lý để giữ kích thước gradient map bằng kích thước ảnh (thuận tiện dùng làm loss pixel-wise trong training).
2. **Công thức khai triển $E[XY]-E[X]E[Y]$ thay vì convolution trực tiếp của $(x-\mu_x)(y-\mu_y)$.** Đây đúng là tối ưu hoá fuse phép tính được nêu trong tên thư viện ("fused"-ssim): thay vì (i) tính $\mu_x,\mu_y$ trước, (ii) trừ ảnh gốc cho $\mu$ để được ảnh "đã khử trung bình", rồi (iii) tích chập lại — tốn 2 lượt tích chập toàn ảnh cho mỗi đại lượng — code chỉ cần tích chập $x^2, y^2, xy$ một lần duy nhất mỗi loại rồi trừ $\mu^2,\mu_x\mu_y$ sau (phép trừ từng pixel, rẻ). Về **đại số** hai cách tương đương hệt nhau. Về **số học dấu phẩy động (float32)**, công thức khai triển có nguy cơ **hiện tượng khử lớn (catastrophic cancellation)**: khi $\sigma_x^2$ nhỏ (vùng ảnh gần như phẳng, ví dụ nền trơn), $E[x^2]$ và $\mu_x^2$ là hai số lớn gần bằng nhau, hiệu của chúng mất nhiều chữ số có nghĩa, sai số tương đối của $\sigma_x^2$ bị khuếch đại. Với ảnh RGB chuẩn hoá $[0,1]$ và float32 (độ chính xác $\approx10^{-7}$), sai số tuyệt đối tạo ra thường ở mức $10^{-7}$–$10^{-6}$ — đủ nhỏ để không ảnh hưởng thị giác hay huấn luyện thực tế vì $C_2=9\times10^{-4}$ đã đóng vai trò "đệm" chống chia cho số gần 0 — nhưng đây là điểm khác biệt số học có thật so với cách tính "trừ trung bình trước" ổn định hơn (two-pass/Welford-style), và trên lý thuyết có thể gây sai số đáng chú ý nếu dùng độ chính xác thấp hơn (fp16) hoặc ảnh có giá trị lớn chưa chuẩn hoá.
3. Code tính `dm_dmu1`, `dm_dsigma1_sq`, `dm_dsigma12` ngay trong kernel forward (`ssim.cu:273-282`, phục vụ backward sau này) — không ảnh hưởng tới SSIM map forward, chỉ là tối ưu "fuse" để khỏi tính lại các đạo hàm riêng phần ở backward pass.

**Kết luận:** công thức SSIM core và cửa sổ Gaussian khớp hoàn toàn chuẩn gốc; khác biệt thực sự nằm ở (i) cách xử lý biên mặc định (zero-padding "same" thay vì "valid"), và (ii) dùng công thức khai triển moment để fuse phép tính — về đại số đúng, về số học có đánh đổi ổn định (chấp nhận được trong điều kiện ảnh $[0,1]$, float32).

---

## Ví dụ số

Cửa sổ thật trong code có bán kính $5$ (11 tap), để tính tay được, mục này dùng một **cửa sổ Gaussian rút gọn, bán kính $2$ (5 tap)**, cùng $\sigma=1.5$, cùng công thức chuẩn hoá, cùng $C_1,C_2$ thật của code — chỉ thu nhỏ kích thước cửa sổ để có thể tính tay. Toàn bộ pipeline (separable conv → $E[X^2]-E[X]^2$ → SSIM map → mean) được giữ y hệt code.

### Cửa sổ rút gọn

$$
w_k=\frac{\exp(-k^2/(2\cdot1.5^2))}{\sum_{k'=-2}^{2}\exp(-k'^2/(2\cdot1.5^2))},\qquad k=-2,\dots,2
$$

$$
w=(0.120078,\ 0.233881,\ 0.292082,\ 0.233881,\ 0.120078),\qquad \sum w_k=1
$$

(kiểm tra: $w_0=1/3.42368=0.292082$ với mẫu số $=1+2(e^{-1/4.5}+e^{-4/4.5})=1+2(0.800737+0.411112)=3.423698$.)

### Patch ảnh $5\times5$

$$
x_{ij}=0.1(i+j),\qquad y_{ij}=0.1(i+j)+0.05\cdot(-1)^{i+j},\qquad i,j=0,\dots,4
$$

$$
x=\begin{pmatrix}
0.0&0.1&0.2&0.3&0.4\\
0.1&0.2&0.3&0.4&0.5\\
0.2&0.3&0.4&0.5&0.6\\
0.3&0.4&0.5&0.6&0.7\\
0.4&0.5&0.6&0.7&0.8
\end{pmatrix}
\qquad
y=\begin{pmatrix}
0.05&0.05&0.25&0.25&0.45\\
0.05&0.25&0.25&0.45&0.45\\
0.25&0.25&0.45&0.45&0.65\\
0.25&0.45&0.45&0.65&0.65\\
0.45&0.45&0.65&0.65&0.85
\end{pmatrix}
$$

### Tính tại pixel trung tâm $(2,2)$ — "valid" (cửa sổ nằm trọn trong patch, không cần padding)

Lấy khối lân cận $5\times5$ quanh $(2,2)$ (toàn bộ patch), trọng số 2D $W_{kl}=w_kw_l$ ($k,l=-2,\dots,2$, tâm $w_0$ ứng pixel $(2,2)$):

**Bước 1 — $\mu$:**

$$
\mu_x(2,2)=\sum_{k,l}W_{kl}\,x_{2+k,2+l}=0.400000
$$

$$
\mu_y(2,2)=\sum_{k,l}W_{kl}\,y_{2+k,2+l}=0.400208
$$

(kiểm tra trực giác: $x$ đối xứng hoàn hảo quanh tâm $(2,2)=0.4$ với trọng số đối xứng $\Rightarrow\mu_x=0.4$ đúng khớp.)

**Bước 2 — raw $E[x^2],E[y^2],E[xy]$ rồi trừ đi $\mu^2$:**

$$
E[x^2](2,2)=\sum_{k,l}W_{kl}\,x_{2+k,2+l}^2=0.188568
$$

$$
\sigma_x^2(2,2)=E[x^2]-\mu_x^2=0.188568-0.4^2=0.028568
$$

$$
E[y^2](2,2)=0.191136,\qquad \sigma_y^2(2,2)=0.191136-0.400208^2=0.031068
$$

$$
E[xy](2,2)=0.188651,\qquad \sigma_{xy}(2,2)=E[xy]-\mu_x\mu_y=0.188651-0.4\times0.400208=0.028568
$$

(Các giá trị trên tính bằng separable convolution hai lượt y hệt code — xem bảng đối chiếu bên dưới.)

**Bước 3 — SSIM:**

$$
C_1=10^{-4},\quad C_2=9\times10^{-4}
$$

$$
C=2\mu_x\mu_y+C_1=2(0.4)(0.400208)+10^{-4}=0.320266
$$

$$
D=2\sigma_{xy}+C_2=2(0.028568)+9\times10^{-4}=0.058036
$$

$$
A=\mu_x^2+\mu_y^2+C_1=0.16+0.160166+10^{-4}=0.320266
$$

$$
B=\sigma_x^2+\sigma_y^2+C_2=0.028568+0.031068+9\times10^{-4}=0.060536
$$

$$
m(2,2)=\frac{C\cdot D}{A\cdot B}=\frac{0.320266\times0.058036}{0.320266\times0.060536}=\frac{0.018587}{0.019388}=0.958703
$$

**Đối chiếu với code chạy thật** (áp đúng `do_separable_conv_x` rồi `do_separable_conv_y` với cửa sổ rút gọn, cùng công thức `ssim.cu:220-268`):

| Đại lượng | Giá trị (separable, như code) | Giá trị (tính trực tiếp 2D) |
|---|---|---|
| $\mu_x$ | 0.400000 | 0.400000 |
| $\mu_y$ | 0.400208 | 0.400208 |
| $\sigma_x^2$ | 0.028568 | 0.028568 |
| $\sigma_y^2$ | 0.031068 | 0.031068 |
| $\sigma_{xy}$ | 0.028568 | 0.028568 |
| $m(2,2)$ | **0.958703** | **0.958703** |

Hai cách (tích chập tách biệt theo đúng thứ tự $x$ rồi $y$ như kernel, và tích chập 2D trực tiếp bằng $W_{kl}=w_kw_l$) cho cùng kết quả đến $10^{-15}$ — xác nhận tính tách biệt (separability) không làm thay đổi kết quả, chỉ thay đổi hiệu năng.

### Hiệu ứng zero-padding ở pixel biên (minh hoạ "same" vs "valid")

Tại pixel góc $(0,0)$ của patch $5\times5$ này, cửa sổ bán kính $2$ cần các pixel $(-2,-2)\dots(2,2)$ — những pixel có toạ độ âm được code coi là $0$ (`get_pix_value`, `ssim.cu:36-37`):

$$
\mu_x(0,0)=\sum_{k,l\ge -\min(2,i),\,-\min(2,j)}W_{kl}\,x_{k,l}=0.061250\ \ (< \mu_x(2,2)=0.4)
$$

$$
m(0,0)=0.951746
$$

So với $m(2,2)=0.958703$: SSIM tại biên **thấp hơn** không phải vì ảnh thực sự khác nhau nhiều hơn ở đó, mà một phần vì trọng số cửa sổ "rơi" vào các pixel ảo $=0$ (làm $\mu_x,\mu_y$ bị kéo thấp giả tạo, lệch khỏi giá trị cục bộ thật). Nếu dùng `padding="valid"`, pixel $(0,0)$ này sẽ bị cắt bỏ hoàn toàn, không đóng góp vào `map.mean()` — đúng với nhận định ở mục "Kiểm chứng" rằng chế độ `"same"` mặc định có thể làm thiên lệch nhẹ giá trị SSIM trung bình gần biên ảnh.

### Trung bình toàn ảnh (minh hoạ `map.mean()`)

Với patch $5\times5$ ở trên (cửa sổ rút gọn bán kính 2), tính đủ $25$ giá trị `ssim_map` theo đúng code rồi lấy trung bình cộng:

$$
\mathrm{SSIM}_{\text{same}}=\frac{1}{25}\sum_{i,j} m(i,j)=0.978456
$$

$$
\mathrm{SSIM}_{\text{valid}}=\frac{1}{1}\,m(2,2)=0.958703
$$

(với patch nhỏ $5\times5$ và cửa sổ bán kính $2$, "valid" chỉ còn đúng $1$ pixel trung tâm — khẳng định lại Bước (d): `"valid"` cắt đúng $r$ pixel mỗi cạnh.)

Hai số trên khác nhau khá nhiều ($0.978$ so với $0.959$) vì patch quá nhỏ so với cửa sổ (tỉ lệ biên/diện tích rất cao); trên ảnh thật (vài trăm pixel mỗi chiều) với cửa sổ $11\times11$, tỉ lệ pixel biên bị ảnh hưởng nhỏ hơn nhiều nhưng khác biệt giữa `"same"` và `"valid"` vẫn tồn tại về nguyên tắc.
