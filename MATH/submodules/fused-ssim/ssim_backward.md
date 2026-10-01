# Công thức toán học của backward (gradient) trong `fused-ssim`

File nguồn: `submodules/fused-ssim/ssim.cu` (kernel `fusedssimCUDA` — forward, dòng 186–285;
kernel `fusedssim_backwardCUDA` — backward, dòng 287–365) và `submodules/fused-ssim/ssim.h`
(khai báo hai hàm C++ `fusedssim`, `fusedssim_backward`, dòng 7–26).

Mục tiêu: tính $\dfrac{\partial \mathcal L}{\partial x}$ với $x=\text{img1}$ (ảnh render, cần
gradient) và $y=\text{img2}$ (ảnh ground-truth, không cần gradient), trong đó
$\mathcal L_{ssim}=1-\overline{\text{SSIM}}$ (hoặc tổng/trung bình của `ssim_map`) là một phần
của loss huấn luyện. Toàn bộ phép tính ở đây là **lan truyền ngược của giá trị SSIM theo từng
pixel**, chưa nhân với trọng số $\lambda_{ssim}$ hay $(1-\lambda)$ trong `loss.py` — việc đó nằm
ngoài phạm vi file này.

---

## 1. Ký hiệu và tóm tắt phần forward cần dùng

Gọi $G$ là kernel Gaussian rời rạc 1 chiều, 11 tap, đối xứng (dòng 8–18):

$$
G=(G_0,\dots,G_{10}),\qquad G_k=G_{10-k}\ \ \forall k,\qquad \sum_{k=0}^{10}G_k=1
$$

(cụ thể $G_0=0.0010284,\ G_1=0.0075988,\ G_2=0.0360008,\ G_3=0.1093607,\ G_4=0.2130055,\
G_5=0.2660117,\ G_6=G_4,\dots,G_{10}=G_0$ — một Gaussian $\sigma\approx1.5$ px đã rời rạc hoá).

Kernel 2D tách được (separable) là $G\otimes G$, tức với mọi ảnh $f$:

$$
(G*f)(i,j)=\sum_{p=-5}^{5}\sum_{q=-5}^{5} G_{p+5}\,G_{q+5}\; f(i+p,\,j+q)
$$

và trong code việc này được thực hiện bằng hai bước 1D: `do_separable_conv_x` rồi
`do_separable_conv_y` (dòng 99–184), ngoài biên ảnh thì `get_pix_value` trả về $0$ (zero-padding,
dòng 35–41).

Forward (dòng 212–283), với từng pixel $i$ (viết gọn index 2D thành chỉ số $i$) và từng kênh màu:

$$
\mu_x=G*x,\qquad \mu_y=G*y
$$

$$
\sigma_x^2=G*x^2-\mu_x^2,\qquad \sigma_y^2=G*y^2-\mu_y^2,\qquad
\sigma_{xy}=G*(xy)-\mu_x\mu_y
$$

SSIM map tại từng pixel (dòng 261–268), với $C_1,C_2$ là hằng số ổn định SSIM chuẩn:

$$
A=\mu_x^2+\mu_y^2+C_1,\qquad B=\sigma_x^2+\sigma_y^2+C_2,\qquad
C=2\mu_x\mu_y+C_1,\qquad D=2\sigma_{xy}+C_2
$$

$$
\boxed{\ m \;=\; \text{SSIM}(i) \;=\; \dfrac{C\cdot D}{A\cdot B} \;=\;
\dfrac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}\ }
$$

Đây đúng là công thức SSIM chuẩn (Wang et al. 2004). Forward kernel **đồng thời** tính luôn ba
đạo hàm riêng cục bộ (dòng 273–281), lưu lại để backward dùng — đây là điểm mấu chốt của thiết kế
"fused": forward đã tính sẵn phần không phụ thuộc vị trí pixel lân cận, backward chỉ cần convolve
lại.

$$
\texttt{dm\_dmu1} = \frac{\partial m}{\partial \mu_x}\Big|_{\text{code}},\qquad
\texttt{dm\_dsigma1\_sq} = \frac{\partial m}{\partial \sigma_x^2},\qquad
\texttt{dm\_dsigma12} = \frac{\partial m}{\partial \sigma_{xy}}
$$

Phần forward (mu, sigma, SSIM map) không trình bày lại chi tiết — chỉ dùng làm input cho mục dưới.

---

## 2. Suy đạo hàm chain rule từng bước (bám đúng thứ tự code)

Ta cần $\dfrac{\partial \mathcal L}{\partial x_j}$ tại mỗi pixel $j$ của ảnh $x$. Đồ thị tính toán:

$$
x \ \xrightarrow{G*}\ \mu_x \ \longrightarrow\ m
\qquad
x \ \xrightarrow{G*(\cdot)^2,\ -\mu_x^2}\ \sigma_x^2 \ \longrightarrow\ m
\qquad
x \ \xrightarrow{G*(x y),\ -\mu_x\mu_y}\ \sigma_{xy} \ \longrightarrow\ m
$$

tức $x$ ảnh hưởng tới $m$ qua **ba nhánh song song**: $\mu_x$, $\sigma_x^2$, $\sigma_{xy}$. Theo
quy tắc chuỗi nhiều biến (multivariable chain rule), gradient tổng là tổng ba nhánh:

$$
\frac{\partial \mathcal L}{\partial x_j}
=\underbrace{\sum_i \frac{\partial \mathcal L}{\partial m_i}\frac{\partial m_i}{\partial \mu_{x,i}}\frac{\partial \mu_{x,i}}{\partial x_j}}_{\text{nhánh }\mu_x}
+\underbrace{\sum_i \frac{\partial \mathcal L}{\partial m_i}\frac{\partial m_i}{\partial \sigma_{x,i}^2}\frac{\partial \sigma_{x,i}^2}{\partial x_j}}_{\text{nhánh }\sigma_x^2}
+\underbrace{\sum_i \frac{\partial \mathcal L}{\partial m_i}\frac{\partial m_i}{\partial \sigma_{xy,i}}\frac{\partial \sigma_{xy,i}}{\partial x_j}}_{\text{nhánh }\sigma_{xy}}
$$

trong đó tổng chạy trên **mọi pixel $i$** trong vùng ảnh hưởng tới $j$ (vì $\mu_{x,i}$ phụ thuộc
vào $x_j$ với mọi $j$ trong bán kính 5 quanh $i$, không chỉ $i=j$) — đây chính là lý do backward
cần một phép **convolution khác**, không phải chỉ nhân element-wise.

Đặt $g_i=\dfrac{\partial \mathcal L}{\partial m_i}=\texttt{dL\_dmap}[i]$ (gradient từ loss phía
trên truyền vào, input của kernel backward, dòng 295).

### Bước 1 — Đạo hàm của phép convolution là convolution với kernel lật

Vì $\mu_{x,i}=(G*x)_i=\sum_j G_{i-j}\,x_j$ là **tuyến tính** theo $x$, nên

$$
\frac{\partial \mu_{x,i}}{\partial x_j}=G_{i-j}
$$

Thay vào nhánh $\mu_x$:

$$
\sum_i g_i\cdot\frac{\partial m_i}{\partial \mu_{x,i}}\cdot G_{i-j}
=\sum_i \big(g_i\cdot \texttt{dm\_dmu1}_i\big)\;G_{i-j}
$$

Đây là **tương quan chéo** (cross-correlation) của trường $h_i=g_i\cdot\texttt{dm\_dmu1}_i$ với
kernel $G$. Nhưng $G$ đối xứng ($G_{k}=G_{-k}$ vì $G_k=G_{10-k}$ quanh tâm), nên với kernel đối
xứng, **tương quan chéo $=$ convolution**:

$$
\sum_i h_i\,G_{i-j}=\sum_i h_i\,G_{j-i}=(G*h)_j
$$

Đây đúng là điều code khai thác: không cần "lật kernel" tường minh khi backward vì $G$ tự đối
xứng — `do_separable_conv_x`/`do_separable_conv_y` dùng **chính kernel $G$** cho cả forward lẫn
backward (dòng 328, 341, 354 gọi đúng hai hàm giống hệt forward). Nếu kernel không đối xứng thì
backward bắt buộc phải lật kernel ($G_{-k}$ thay vì $G_k$) — đây là tính chất tổng quát của đạo
hàm convolution: $\dfrac{\partial (K*x)}{\partial x}$ áp vào một gradient ngược dòng tương đương
với convolution của gradient đó với kernel lật $\tilde K(k)=K(-k)$.

Kết quả nhánh $\mu_x$ (khớp code dòng 319–332):

$$
\Big(\frac{\partial \mathcal L}{\partial x}\Big)_{\mu_x\text{-branch}, j}
= \big(G*(\texttt{dm\_dmu1}\odot\texttt{dL\_dmap})\big)_j
$$

với $\odot$ là nhân element-wise (hàm `multiply_shared_mem`, dòng 64–78, 324).

### Bước 2 — Nhánh $\sigma_x^2$

$\sigma_{x,i}^2=(G*x^2)_i-\mu_{x,i}^2$. Đạo hàm theo $x_j$ (dùng đạo hàm riêng phần cho từng số
hạng, lưu ý $x^2$ cũng convolve tuyến tính theo biến mới $x^2$ nhưng ta cần đạo hàm theo $x$ gốc,
nên áp dụng chain rule thêm một lớp):

$$
\frac{\partial \sigma_{x,i}^2}{\partial x_j}
=\frac{\partial (G*x^2)_i}{\partial x_j}-2\mu_{x,i}\frac{\partial \mu_{x,i}}{\partial x_j}
=2x_j\,G_{i-j}-2\mu_{x,i}\,G_{i-j}
=2G_{i-j}\,(x_j-\mu_{x,i})
$$

(dùng $\partial(x_j^2)/\partial x_j=2x_j$ và kết quả Bước 1 cho $\partial\mu_{x,i}/\partial x_j$).

Nhánh này đóng góp vào $\partial\mathcal L/\partial x_j$:

$$
\sum_i g_i\,\texttt{dm\_dsigma1\_sq}_i\cdot 2G_{i-j}(x_j-\mu_{x,i})
=2x_j\underbrace{\sum_i G_{j-i}\,\big(g_i\,\texttt{dm\_dsigma1\_sq}_i\big)}_{(G*(\texttt{dm\_dsigma1\_sq}\odot\texttt{dL\_dmap}))_j}
\;-\;2\underbrace{\sum_i G_{j-i}\,\big(g_i\,\texttt{dm\_dsigma1\_sq}_i\,\mu_{x,i}\big)}_{(G*(\mu_x\odot\texttt{dm\_dsigma1\_sq}\odot\texttt{dL\_dmap}))_j}
$$

(lại dùng tính đối xứng $G_{i-j}=G_{j-i}$ ở Bước 1). Gọi số hạng thứ nhất là $T_{\sigma}^{(1)}$,
số hạng thứ hai (có dấu trừ) là $T_\sigma^{(2)}$:

$$
\Big(\frac{\partial \mathcal L}{\partial x}\Big)_{\sigma_x^2\text{-branch}, j}
= 2x_j\, T_\sigma^{(1)}{}_j \;-\; 2\,T_\sigma^{(2)}{}_j
$$

Code (dòng 335–345) **chỉ tính $2x_j\,T_\sigma^{(1)}{}_j$**:

```cpp
load_into_shared(buf2, dm_dsigma1_sq, CH, H, W, i);
...
multiply_shared_mem(buf2, buf1);      // buf2 = dm_dsigma1_sq ⊙ dL_dmap
...
do_separable_conv_x(buf2, buf3, H, W);
...
tmp = pix1 * 2.0f * do_separable_conv_y(buf3, H, W);   // = 2*x_j * T_sigma^(1)_j
dL_dpix += tmp;
```

Số hạng $-2\,T_\sigma^{(2)}{}_j$ **không xuất hiện ở đây** — xem Bước 4 để thấy nó được gộp vào
nhánh $\mu_x$ thay vì tính riêng.

### Bước 3 — Nhánh $\sigma_{xy}$

$\sigma_{xy,i}=(G*(xy))_i-\mu_{x,i}\mu_{y,i}$. Vì $y$ không phụ thuộc $x$:

$$
\frac{\partial \sigma_{xy,i}}{\partial x_j}
=\frac{\partial (G*(xy))_i}{\partial x_j}-\mu_{y,i}\frac{\partial \mu_{x,i}}{\partial x_j}
=y_j\,G_{i-j}-\mu_{y,i}\,G_{i-j}
=G_{i-j}\,(y_j-\mu_{y,i})
$$

Đóng góp vào gradient:

$$
\sum_i g_i\,\texttt{dm\_dsigma12}_i\,G_{i-j}(y_j-\mu_{y,i})
=y_j\underbrace{\sum_i G_{j-i}(g_i\,\texttt{dm\_dsigma12}_i)}_{T_{12}^{(1)}{}_j}
-\underbrace{\sum_i G_{j-i}(g_i\,\texttt{dm\_dsigma12}_i\,\mu_{y,i})}_{T_{12}^{(2)}{}_j}
$$

Code (dòng 348–358) **chỉ tính $y_j\,T_{12}^{(1)}{}_j$**:

```cpp
load_into_shared(buf2, dm_dsigma12, CH, H, W, i);
...
multiply_shared_mem(buf2, buf1);     // buf2 = dm_dsigma12 ⊙ dL_dmap
...
tmp = pix2 * do_separable_conv_y(buf3, H, W);   // = y_j * T_12^(1)_j
dL_dpix += tmp;
```

Số hạng $-T_{12}^{(2)}{}_j$ cũng **không xuất hiện riêng** — cũng gộp vào nhánh $\mu_x$ (Bước 4).

### Bước 4 — Vì sao `dm_dmu1` trong code có 4 số hạng thay vì 2: gộp (fuse) ba convolution thành một

Nếu chỉ coi $m=CD/(AB)$ như hàm của 5 biến độc lập $(\mu_x,\mu_y,\sigma_x^2,\sigma_y^2,\sigma_{xy})$
(giữ 4 biến còn lại cố định khi lấy đạo hàm riêng theo một biến — đây là **định nghĩa đạo hàm
riêng phần nhiều biến**), áp dụng quy tắc thương (quotient rule) với $f=CD$ (tử), $g=AB$ (mẫu):

$$
\frac{\partial m}{\partial \mu_x}\Big|_{\text{"naive"}}
=\frac{\partial C/\partial\mu_x\cdot D}{AB}-\frac{CD\cdot \partial A/\partial \mu_x\cdot B}{A^2B^2}
=\frac{2\mu_y D}{AB}-\frac{2\mu_x CD}{A^2B}
$$

(vì $\partial C/\partial\mu_x=2\mu_y$, $\partial A/\partial\mu_x=2\mu_x$, còn $D,B$ không phụ
thuộc $\mu_x$). Đây chỉ là **2 số hạng**. Nhưng code (dòng 274–279) viết:

```cpp
dm_dmu1[global_idx] = (
  (mu2 * 2.0f * D) / (A * B)
  -(mu2 * 2.0f * C) / (A * B)
  -(mu1 * 2.0f * C * D) / ( A * A * B)
  +(mu1 * 2.0f * C * D) / (A * B * B)
);
```

tức 4 số hạng. Hai số hạng đầu+cuối ($\frac{2\mu_y D}{AB}$ và $-\frac{2\mu_x CD}{A^2B}$) khớp
đúng "naive" ở trên. **Hai số hạng còn lại chính là các số hạng $T_\sigma^{(2)}$ và $T_{12}^{(2)}$
bị thiếu ở Bước 2, 3** — được gấp (fold) vào đây dưới dạng hệ số nhân với $\texttt{dm\_dmu1}$ **trước
khi** convolve, thay vì convolve riêng rồi trừ sau. Cụ thể, so khớp:

$$
-\frac{2\mu_x CD}{AB^2}\ \big(\text{số hạng 4, dấu }+\big) \;=\; -2\mu_x\cdot\underbrace{\Big(\!-\frac{CD}{AB^2}\Big)}_{\texttt{dm\_dsigma1\_sq}}
$$

$$
-\frac{2\mu_y C}{AB}\ \big(\text{số hạng 2}\big) \;=\; -\mu_y\cdot\underbrace{\Big(\frac{2C}{AB}\Big)}_{\texttt{dm\_dsigma12}}
$$

Vậy, đúng bằng đại số (đã kiểm chứng bằng `sympy`, xem mục 4):

$$
\boxed{\ \texttt{dm\_dmu1}=\underbrace{\frac{2\mu_y D}{AB}-\frac{2\mu_x CD}{A^2B}}_{\partial m/\partial\mu_x\ \text{(naive, 2 số hạng)}}
\;-\;2\mu_x\cdot\texttt{dm\_dsigma1\_sq}\;-\;\mu_y\cdot\texttt{dm\_dsigma12}\ }
$$

Nhờ **tính tuyến tính của convolution** ($G*(a+b)=G*a+G*b$), khi nhân $\texttt{dL\_dmap}$ rồi
convolve với $G$ (Bước 1), số hạng $-2\mu_x\cdot\texttt{dm\_dsigma1\_sq}\cdot\texttt{dL\_dmap}$
sau khi convolve đúng bằng $-2\,T_\sigma^{(2)}$, và $-\mu_y\cdot\texttt{dm\_dsigma12}\cdot
\texttt{dL\_dmap}$ sau khi convolve đúng bằng $-T_{12}^{(2)}$ — tức **một convolution duy nhất
của nhánh $\mu_x$ gánh luôn phần "thiếu" của hai nhánh kia**. Đây là một tối ưu hoá hiệu năng rất
tinh tế (giảm từ lẽ ra cần 5 convolution xuống còn 3), **không phải lỗi dấu hay hệ số** — xem
kiểm chứng số học ở mục 4.

### Bước 5 — Tổng hợp

$$
\frac{\partial \mathcal L}{\partial x_j}
= \big(G*(\texttt{dm\_dmu1}\odot\texttt{dL\_dmap})\big)_j
+ 2x_j\big(G*(\texttt{dm\_dsigma1\_sq}\odot\texttt{dL\_dmap})\big)_j
+ y_j\big(G*(\texttt{dm\_dsigma12}\odot\texttt{dL\_dmap})\big)_j
$$

khớp chính xác dòng `dL_dimg1[global_idx] = dL_dpix;` (dòng 362), với `dL_dpix` là tổng 3 `tmp`
tích lũy ở dòng 332, 345, 358.

---

## 3. Kiến thức toán nền tảng

- **Đạo hàm riêng phần (partial derivative)**: đạo hàm của hàm nhiều biến theo một biến, giữ các
  biến còn lại cố định — dùng để định nghĩa $\texttt{dm\_dmu1}$, $\texttt{dm\_dsigma1\_sq}$,
  $\texttt{dm\_dsigma12}$ như đạo hàm của $m(\mu_x,\mu_y,\sigma_x^2,\sigma_y^2,\sigma_{xy})$ theo
  từng biến trung gian.
- **Quy tắc chuỗi nhiều biến (multivariable chain rule)**: khi $x$ ảnh hưởng tới $m$ qua nhiều
  đường (ở đây là 3 đường: $\mu_x,\sigma_x^2,\sigma_{xy}$), tổng đạo hàm là tổng đóng góp của từng
  đường — $\dfrac{d\mathcal L}{dx}=\sum_{\text{path}} \dfrac{\partial \mathcal L}{\partial
  (\cdot)}\dfrac{\partial(\cdot)}{\partial x}$.
- **Quy tắc thương (quotient rule)** và **quy tắc tích**: dùng để đạo hàm $m=CD/(AB)$.
- **Tính chất đạo hàm của convolution/tương quan chéo**: convolution là ánh xạ tuyến tính theo
  input, nên đạo hàm (Jacobian) của nó theo input chính là ma trận convolution (Toeplitz), và khi
  lan truyền ngược một gradient qua convolution ta lại thu được một convolution — với kernel **lật**
  nếu kernel gốc không đối xứng, hoặc **chính kernel đó** nếu kernel đối xứng (trường hợp Gaussian
  $G$ ở đây). Đây là nguyên lý vận hành của "transposed convolution"/backward-conv trong CNN nói
  chung.
- **Tính tuyến tính (linearity)**: $G*(af+bh)=a(G*f)+b(G*h)$ — nền tảng cho việc gộp 3 nhánh
  gradient $\mu_x,\sigma_x^2,\sigma_{xy}$ thành một convolution duy nhất ở Bước 4.
- **Vi tích phân hàm nhiều biến / Jacobian**: toàn bộ $m$ là hàm $\mathbb R^{5}\to\mathbb R$ của
  $(\mu_x,\mu_y,\sigma_x^2,\sigma_y^2,\sigma_{xy})$, còn các biến này lại là hàm (tuyến tính +
  bậc hai) của $x\in\mathbb R^{HW}$; gradient $\partial\mathcal L/\partial x$ là tích các Jacobian
  theo chain rule, nhưng nhờ cấu trúc convolution nên không cần lập ma trận Jacobian tường minh.

---

## 4. Kiểm chứng tính đúng sai

### 4.1. Tự đạo hàm bằng tay

Với SSIM chuẩn $\text{SSIM}=\dfrac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}=\dfrac{CD}{AB}$,
coi $y$ (do đó $\mu_y,\sigma_y^2$) cố định, các đạo hàm riêng "sạch" (mỗi biến trung gian độc lập,
đúng định nghĩa toán học của $\partial m/\partial(\cdot)$) là:

$$
\frac{\partial m}{\partial \mu_x}=\frac{2\mu_y D}{AB}-\frac{2\mu_x CD}{A^2B},\qquad
\frac{\partial m}{\partial \sigma_x^2}=-\frac{CD}{AB^2},\qquad
\frac{\partial m}{\partial \sigma_{xy}}=\frac{2C}{AB}
$$

Áp dụng chain rule qua $\mu_x=G*x,\ \sigma_x^2=G*x^2-\mu_x^2,\ \sigma_{xy}=G*(xy)-\mu_x\mu_y$ (đã
suy ở mục 2, Bước 1–3):

$$
\frac{\partial \mathcal L}{\partial x_j}=
G*\!\Big(\frac{\partial m}{\partial\mu_x}\,g\Big)_j
+2x_j\,G*\!\Big(\frac{\partial m}{\partial\sigma_x^2}\,g\Big)_j
-2\,G*\!\Big(\mu_x\frac{\partial m}{\partial\sigma_x^2}\,g\Big)_j
+y_j\,G*\!\Big(\frac{\partial m}{\partial\sigma_{xy}}\,g\Big)_j
-G*\!\Big(\mu_y\frac{\partial m}{\partial\sigma_{xy}}\,g\Big)_j
$$

với $g=\partial\mathcal L/\partial m=\texttt{dL\_dmap}$. Đây là công thức "đúng" gồm **5 số hạng,
5 convolution**.

### 4.2. So sánh với code

Code chỉ dùng **3 convolution**. Kiểm tra bằng đại số (xác nhận lại bằng `sympy`, xem script bên
dưới) cho thấy:

$$
\texttt{dm\_dmu1}_{\text{code}}=\frac{\partial m}{\partial\mu_x}-2\mu_x\,\frac{\partial m}{\partial\sigma_x^2}-\mu_y\,\frac{\partial m}{\partial\sigma_{xy}}
$$

tức số hạng "naive" cộng thêm đúng hai phần hiệu chỉnh mà lẽ ra phải tính rời (số hạng 3 và 5 ở
trên). Vì convolution tuyến tính, nhân hệ số này **trước** khi convolve với $G$ cho kết quả giống
hệt convolve riêng từng phần rồi cộng — nên:

$$
G*(\texttt{dm\_dmu1}_{\text{code}}\cdot g)
= G*\Big(\frac{\partial m}{\partial\mu_x}g\Big) - 2\,G*\Big(\mu_x\frac{\partial m}{\partial\sigma_x^2}g\Big) - G*\Big(\mu_y\frac{\partial m}{\partial\sigma_{xy}}g\Big)
$$

Cộng thêm hai số hạng còn lại của công thức "đúng" ($2x_j\,G*(\partial m/\partial\sigma_x^2\,g)$
và $y_j\,G*(\partial m/\partial\sigma_{xy}\,g)$, chính là 2 convolution còn lại trong code) ta thu
được **chính xác công thức 5 số hạng ở mục 4.1**. Vậy công thức trong code **khớp hoàn toàn** với
đạo hàm giải tích — không có bug dấu hay hệ số, không có số hạng nào bị bỏ qua; sự khác biệt chỉ
là code **gộp đại số** 2 trong 5 convolution vào nhánh $\mu_x$ để giảm từ 5 xuống 3 phép convolution
(tối ưu hiệu năng CUDA, không đánh đổi độ chính xác vì phép gộp là một đẳng thức đại số đúng
tuyệt đối, không phải một xấp xỉ).

Đã verify bằng `sympy.simplify` (xem `/tmp`-style script dùng trong quá trình viết tài liệu này):

```python
dm_dmu1_true = sp.diff(m, mu1)                       # 2 số hạng "naive"
dm_dmu1_code = (mu2*2*D)/(A*B) - (mu2*2*C)/(A*B) \
             - (mu1*2*C*D)/(A*A*B) + (mu1*2*C*D)/(A*B*B)
sp.simplify(dm_dmu1_true - (dm_dmu1_code
            + 2*mu1*((-C*D)/(A*B*B))            # +2*mu1*dm_dsigma1_sq
            + mu2*((2*C)/(A*B))))                # +mu2*dm_dsigma12
# => 0   (đẳng thức đúng tuyệt đối)
```

đồng thời `dm_dsigma1_sq` và `dm_dsigma12` trong code khớp **chính xác từng chữ** (`sympy`
simplify ra 0) với $\partial m/\partial\sigma_x^2$ và $\partial m/\partial\sigma_{xy}$ ở trên —
không cần hiệu chỉnh gì thêm cho hai đại lượng này.

**Kết luận mục kiểm chứng**: công thức backward trong `fused-ssim` đúng 100% về mặt toán học so
với đạo hàm SSIM chuẩn; "4 số hạng" trong `dm_dmu1` không phải lỗi mà là kết quả gộp đại số hợp lệ
của 3 trong 5 số hạng gradient, tận dụng tính tuyến tính của convolution để giảm số lần convolve
từ 5 xuống 3 (tiết kiệm ~40% chi phí convolution trong backward).

---

## 5. Ví dụ số (kiểm chứng bằng finite difference)

### 5.1. Thiết lập

Dùng đúng kernel $G$ (11-tap) và đúng 3 công thức `dm_dmu1`, `dm_dsigma1_sq`, `dm_dsigma12` của
code, cài lại bằng NumPy (zero-padding ngoài biên, giống `get_pix_value`). Ảnh thử: $16\times16$,
1 kênh, giá trị ngẫu nhiên (seed cố định) trong $[0.2,0.8]$, $C_1=0.01^2=10^{-4}$,
$C_2=0.03^2=9\times10^{-4}$. Loss thử: $\mathcal L=\sum_{i} m_i$ (tức $g_i\equiv1$ — tương đương
`dL_dmap` toàn số 1, trường hợp tổng quát khi `dL_dmap` khác 1 hoàn toàn tương tự vì nó chỉ nhân
element-wise trước convolve).

### 5.2. Gradient giải tích (theo đúng công thức code, mục 2 Bước 5)

Tại pixel $(y,x)=(8,8)$:

| Đại lượng | Giá trị |
|---|---|
| $x_{8,8}$ (img1) | $0.544595$ |
| $\mu_x$ | $0.537337$ |
| $\mu_y$ | $0.474365$ |
| $\sigma_x^2$ | $0.021531$ |
| $\sigma_{xy}$ | $0.002550$ |
| $A$ | $0.513853$ |
| $B$ | $0.056427$ |
| $C$ | $0.509887$ |
| $D$ | $0.005999$ |
| $m_{8,8}=\text{SSIM}$ | $0.105496$ |
| $\partial m/\partial\mu_x$ (naive, 2 số hạng) | $-0.024342$ |
| $-2\mu_x\cdot\texttt{dm\_dsigma1\_sq}$ | $+2.009206$ |
| $-\mu_y\cdot\texttt{dm\_dsigma12}$ | $-16.683607$ |
| $\texttt{dm\_dmu1}$ (tổng, khớp công thức code) | $-14.698743$ |

(3 dòng cuối cộng lại đúng bằng $-0.024342+2.009206-16.683607=-14.698743$ — xác nhận đẳng thức
đại số ở mục 4.2 bằng số cụ thể, không chỉ ký hiệu.)

Sau khi convolve 3 trường $(\texttt{dm\_dmu1}\odot g)$, $(\texttt{dm\_dsigma1\_sq}\odot g)$,
$(\texttt{dm\_dsigma12}\odot g)$ với $G$ và tổng hợp theo công thức Bước 5:

$$
\frac{\partial \mathcal L}{\partial x_{8,8}}\Big|_{\text{code, giải tích}} = -4.328802910
$$

### 5.3. Gradient số (finite difference)

$$
\frac{\partial \mathcal L}{\partial x_{j}}\Big|_{\text{numeric}}\approx
\frac{\mathcal L(x+\epsilon\,e_j)-\mathcal L(x-\epsilon\,e_j)}{2\epsilon},\qquad \epsilon=10^{-4}
$$

với $e_j$ là vector đơn vị tại pixel $j$ — tức nhiễu **chỉ một pixel** $x_{8,8}$ lên/xuống
$\epsilon$, chạy lại **toàn bộ forward** (mu, sigma, SSIM map trên cả ảnh, vì pixel $(8,8)$ ảnh
hưởng tới $\mu,\sigma$ của các pixel lân cận trong bán kính 5), rồi lấy $\mathcal L=\sum m$.

$$
\frac{\partial \mathcal L}{\partial x_{8,8}}\Big|_{\text{numeric}} = -4.328802885
$$

Sai khác tuyệt đối: $|{-4.328802910}-({-4.328802885})|\approx 2.5\times10^{-8}$ — khớp đến sai số
làm tròn của finite difference bậc hai ($O(\epsilon^2)\sim10^{-8}$ với $\epsilon=10^{-4}$).

Kiểm tra thêm tại 4 pixel khác (kể cả 2 pixel sát biên ảnh $(0,0)$ và $(15,15)$, nơi zero-padding
có hiệu lực):

| Pixel $(y,x)$ | Gradient giải tích (code) | Gradient numeric (finite diff) | $\lvert$sai khác$\rvert$ |
|---|---|---|---|
| $(8,8)$ | $-4.328802910$ | $-4.328802885$ | $2.5\times10^{-8}$ |
| $(5,10)$ | $0.354197101$ | $0.354197104$ | $2.5\times10^{-9}$ |
| $(3,3)$ | $-4.737435312$ | $-4.737435290$ | $2.2\times10^{-8}$ |
| $(0,0)$ | $0.218087477$ | $0.218087480$ | $2.9\times10^{-9}$ |
| $(15,15)$ | $0.691928418$ | $0.691928415$ | $2.4\times10^{-9}$ |

Tất cả khớp nhau tới $\sim10^{-8}$–$10^{-9}$, kể cả tại biên ảnh (xác nhận zero-padding trong
`get_pix_value` được xử lý nhất quán giữa forward và backward). Điều này xác nhận bằng số: công
thức backward trong `fused-ssim` **tính đúng** $\partial \mathcal L/\partial x$.

### 5.4. Cách đọc

- Sai khác cỡ $10^{-8}$ là nhiễu số học của finite difference (không phải bug) — nếu công thức
  giải tích sai thực sự (ví dụ thiếu hẳn một trong hai số hạng hiệu chỉnh ở mục 4), sai khác sẽ
  lớn cỡ $O(1)$ chứ không phải $O(10^{-8})$. Để minh hoạ, nếu dùng "naive" 2-số-hạng
  ($\texttt{dm\_dmu1}=-0.024342$ thay vì $-14.698743$) thay cho công thức code tại pixel $(8,8)$,
  gradient tính ra sẽ lệch khỏi $-4.3288$ một lượng cỡ hàng chục — chứng tỏ hai số hạng hiệu chỉnh
  (mục 4.2) thực sự cần thiết, không phải tùy chọn.
- Trong thực tế huấn luyện, `dL_dmap` không phải toàn số 1 mà là $\partial \mathcal L_{ssim}/
  \partial(\text{ssim\_map})$ từ hàm loss phía Python (ví dụ $\mathcal L_{ssim}=1-\text{mean(ssim\_
  map)}\Rightarrow \texttt{dL\_dmap}=-1/(H\!\cdot\!W\!\cdot\!CH)$ hằng số, hoặc trọng số khác nếu
  dùng biến thể per-pixel). Vì `dL_dmap` chỉ nhân element-wise trước khi convolve (dòng 324, 337,
  350), toàn bộ suy luận ở mục 2–4 không đổi với bất kỳ `dL_dmap` nào.
