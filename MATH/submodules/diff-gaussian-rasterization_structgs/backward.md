# Công thức toán học trong `cuda_rasterizer/backward.cu` (+ `backward.h`)

Tài liệu này suy lại **toàn bộ đạo hàm ngược (gradient)** được tính trong lượt backward của rasterizer CUDA (submodule `diff-gaussian-rasterization_structgs`), bám sát từng dòng code thật tại:

- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/backward.cu`
- `submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/backward.h`
- Tham chiếu: `forward.cu` (để biết công thức xuôi cần đảo ngược), `auxiliary.h` (các hàm tiện ích: `transformPoint4x3`, `transformVec4x3Transpose`, `dnormvdv`, hằng số SH).

Mỗi công thức quan trọng được giải thích **tại sao đúng bằng chain rule**, không chỉ chép lại biểu thức. Mục cuối kiểm chứng một công thức bằng đạo hàm tay + finite-difference numeric gradient check.

---

## 0. Ký hiệu

- $L$: hàm mất mát vô hướng cuối cùng (loss). $dL\_dX := \dfrac{\partial L}{\partial X}$ — ký hiệu dùng xuyên suốt, khớp tên biến trong code.
- $i$: chỉ số Gaussian (thứ tự depth-sort, $i$ tăng = xa máy ảnh hơn, theo thứ tự `point_list` dùng ở forward: gần → xa).
- $T_i$: "transmittance còn lại" sau khi đã blend Gaussian thứ $i$ tại một pixel cụ thể: $T_i = \prod_{j\le i}(1-\alpha_j)$, quy ước $T_0=1$ nếu không có Gaussian nào trước.
- $\alpha_i = \min(0.99,\ o_i\,G_i)$: hệ số blend, với $o_i=\text{con\_o}.w$ là opacity **đã nhân sẵn hệ số bù khử-alias** $\text{coef}_i$ (xem mục 2.5 — đây là điểm **không có trong bản 3D Gaussian Splatting gốc**).
- $G_i=\exp(\text{power}_i)$, $\text{power}_i=-\tfrac12\,\mathbf d_i^\top\,\text{conic}_i\,\mathbf d_i$ với $\mathbf d_i=\text{xy}_{\text{screen},i}-\text{pix}_f$ (vector từ tâm Gaussian chiếu lên ảnh tới pixel).
- $\text{conic}_i = (\Sigma'_i)^{-1}$: nghịch đảo hiệp phương sai 2D màn hình, lưu dạng $(\text{conic}.x,\text{conic}.y,\text{conic}.z)=(\Sigma'^{-1}_{00},\Sigma'^{-1}_{01},\Sigma'^{-1}_{11})$.
- $c_{i,ch}$: màu kênh $ch$ của Gaussian $i$ (đã tính từ SH ở forward).
- $C_{ch}=\sum_i c_{i,ch}\,\alpha_i\,T_{i-1}$ (tổng theo thứ tự gần→xa, $T_{i-1}$ là transmittance *trước khi* gặp Gaussian $i$), $C_{final,ch}=C_{ch}+T_{final}\,\mathbf{bg}_{ch}$.
- $\Sigma_{3D}=\text{Vrk}$: hiệp phương sai 3D thế giới, $\Sigma_{3D}=M^\top M$ với $M=SR(q)$, $S=\text{diag}(m s_x,m s_y,m s_z)$ ($m$=`scale_modifier`), $R(q)$ ma trận quay từ quaternion $q=(r,x,y,z)$.
- $J$: Jacobian xấp xỉ tuyến tính của phép chiếu phối cảnh tại điểm camera-space $t=(t_x,t_y,t_z)=W_{view}\mu$; $T_{3\times3}=WJ$ với $W$ là khối quay $3\times3$ của `view_matrix`; $\Sigma'=T^\top \Sigma_{3D}^\top T$ (hiệp phương sai 2D trước khi cắt bớt hàng/cột thứ 3 và cộng kernel dilation).
- $\mu_i\in\mathbb R^3$: tâm Gaussian (world space), `means3D`.
- $f_x,f_y$: tiêu cự pixel (`focal_x`,`focal_y`, ký hiệu `h_x,h_y` trong code).

---

## 1. Đạo hàm ngược của alpha compositing (`renderCUDA`, dòng 621–784)

### 1.1 Ý tưởng: duyệt ngược và khôi phục $T$ không cần lưu toàn bộ

Ở forward, $T_i = T_{i-1}(1-\alpha_i)$ được tính **xuôi** (gần→xa) và chỉ $T_{final}=T_{last}$ được lưu lại (tiết kiệm bộ nhớ: không lưu $T_i$ cho mọi $i$). Ở backward ta cần $T_{i-1}$ tại từng Gaussian $i$ để tính gradient, nên kernel duyệt **ngược** ($i$ giảm dần, xa→gần) và khôi phục bằng phép **chia ngược**, đúng là nghịch đảo của công thức forward:

$$
T_{i-1} = \frac{T_i}{1-\alpha_i}
$$

Code: khởi tạo `T = T_final`; mỗi vòng lặp, sau khi đọc $\alpha_i$ của Gaussian hiện tại, thực hiện `T = T / (1.f - alpha)` (dòng 726) — đây chính là $T_{i-1}=T_i/(1-\alpha_i)$ **trước khi dùng $T$ để tính gradient**, nên tại thời điểm dùng, biến `T` trong code đã mang giá trị $T_{i-1}$.

### 1.2 Biến tích lũy màu "phía sau" `accum_rec`

Định nghĩa $S_i := \sum_{j>i} c_j\alpha_j\prod_{i<k<j}(1-\alpha_k)$ — đây chính là phần đóng góp của tất cả Gaussian **ở sau** Gaussian $i$ (xa hơn $i$, đã được xử lý ở các vòng lặp *trước* vì ta duyệt ngược), quy chiếu transmittance lại từ vị trí ngay sau Gaussian $i$. Công thức đệ quy:

$$
S_{i} = \alpha_{i+1}\,c_{i+1} + (1-\alpha_{i+1})\,S_{i+1}
$$

**Tại sao đúng:** đây đúng là công thức blend $C=\sum c_j\alpha_jT_{j-1}$ viết lại theo kiểu Horner — "màu đã blend tính từ Gaussian $i+1$ trở đi, nhìn từ gốc $i$" bằng chính $\alpha_{i+1}c_{i+1}$ cộng với phần còn lại $(1-\alpha_{i+1})$ lần phần "từ $i+2$ trở đi" (đã có từ vòng lặp trước, lúc đó đóng vai trò $S_{i+1}$). Code (dòng 738): `accum_rec[ch] = last_alpha*last_color[ch] + (1-last_alpha)*accum_rec[ch]` với `last_alpha,last_color` là $\alpha_{i+1},c_{i+1}$ của **vòng lặp trước** (vì ta đang duyệt ngược, "trước" nghĩa là Gaussian xa hơn, tức $i+1$).

### 1.3 Gradient theo $\alpha_i$

Viết lại $C_{ch} = c_i\alpha_iT_{i-1} + \underbrace{\sum_{j<i}c_j\alpha_jT_{j-1}}_{\text{không phụ thuộc }\alpha_i} + \underbrace{\sum_{j>i}c_j\alpha_jT_{j-1}}_{=T_i\cdot S_i,\ \text{phụ thuộc }\alpha_i\text{ qua }T_i}$.

Vì mỗi $T_{j-1}$ với $j>i$ chứa thừa số $(1-\alpha_i)$ (do $T_{j-1}=\prod_{k<j}(1-\alpha_k)$), ta có $\sum_{j>i}c_j\alpha_jT_{j-1} = (1-\alpha_i)\,T_{i-1}\cdot \frac{S_i}{1}$ — cụ thể hơn: $T_i = T_{i-1}(1-\alpha_i)$ nên phần "phía sau $i$" bằng $T_i\cdot S_i = T_{i-1}(1-\alpha_i)S_i$. Lấy đạo hàm riêng theo $\alpha_i$:

$$
\frac{\partial C_{ch}}{\partial \alpha_i} = c_i T_{i-1} + T_{i-1}\cdot(-1)\cdot S_i = T_{i-1}\,(c_i - S_i)
$$

(phần $\sum_{j<i}$ không phụ thuộc $\alpha_i$ nên triệt tiêu; $T_{i-1}$ không phụ thuộc $\alpha_i$ vì chỉ gồm $\alpha_k,k<i$). Theo chain rule với $L=L(C)$:

$$
\boxed{\frac{\partial L}{\partial \alpha_i} = T_{i-1}\sum_{ch}(c_{i,ch} - S_{i,ch})\cdot\frac{\partial L}{\partial C_{ch}}}
$$

Code tách thành 2 bước: cộng dồn phần $\sum_{ch}(c_{i,ch}-S_{i,ch})\,dL\_dpixel_{ch}$ trước (dòng 742: `dL_dalpha += (c - accum_rec[ch]) * dL_dchannel`), rồi nhân $T_{i-1}$ sau (dòng 748: `dL_dalpha *= T` — đúng lúc này `T` đã là $T_{i-1}$ nhờ phép chia ở mục 1.1).

### 1.4 Đóng góp của nền (background)

$C_{final,ch}=C_{ch}+T_{final}\mathbf{bg}_{ch}$, và $T_{final}=T_{i-1}(1-\alpha_i)\prod_{j<i}(1-\alpha_j)$ — tức $T_{final}$ **cũng** phụ thuộc $\alpha_i$ qua thừa số $(1-\alpha_i)$ (với phần còn lại giữ nguyên vì không phụ thuộc $\alpha_i$). Do đó:

$$
\frac{\partial T_{final}}{\partial\alpha_i} = -\frac{T_{final}}{1-\alpha_i}
$$

(đạo hàm của $(1-\alpha_i)$ nhân với hằng số còn lại, chia lại cho $(1-\alpha_i)$ để ra dạng tương đối). Nhân với $\partial L/\partial C_{final,ch}=dL\_dpixel_{ch}$ và cộng vào gradient đã có:

$$
\frac{\partial L}{\partial \alpha_i}\mathrel{+}= -\frac{T_{final}}{1-\alpha_i}\sum_{ch}\mathbf{bg}_{ch}\cdot dL\_dpixel_{ch}
$$

Code dòng 754–757: `bg_dot_dpixel = Σ bg[ch]*dL_dpixel[ch]`, rồi `dL_dalpha += (-T_final/(1-alpha)) * bg_dot_dpixel` — **khớp, và đúng thứ tự**: dòng này chạy **sau** `dL_dalpha *= T` (dòng 748) nên không bị nhân nhầm $T_{i-1}$ lần hai — phần background cộng vào *sau khi* đã nhân $T$, là đúng vì công thức trên không có thừa số $T_{i-1}$ nào khác ngoài cái đã nằm trong $T_{final}$.

### 1.5 Gradient theo màu từng Gaussian

Từ $C_{ch}=\sum_i c_{i,ch}\alpha_iT_{i-1}$, giữ mọi $\alpha$ cố định: $\partial C_{ch}/\partial c_{i,ch}=\alpha_iT_{i-1}$. Theo chain rule:

$$
\boxed{\frac{\partial L}{\partial c_{i,ch}} = \alpha_i T_{i-1}\cdot\frac{\partial L}{\partial C_{ch}}}
$$

Code: `dchannel_dcolor = alpha * T` (đã là $T_{i-1}$), rồi `atomicAdd(&dL_dcolors[...], dchannel_dcolor * dL_dchannel)` (dòng 727, 746). Dùng `atomicAdd` vì một Gaussian có thể phủ nhiều pixel trong cùng tile/block chạy song song, gradient của nó là **tổng** đóng góp từ mọi pixel — đúng theo quy tắc cộng gradient khi một biến ảnh hưởng tới nhiều đầu ra.

### 1.6 Phiên bản song song theo bucket (`PerGaussianRenderCUDA`, dòng 402–619)

Đây là tổ chức lại **y hệt công thức toán học trên** nhưng đổi chiều song song hóa: thay vì "1 pixel = 1 thread, duyệt qua các Gaussian phủ nó" (như `renderCUDA`), kernel này dùng "1 Gaussian = 1 warp (32 thread), mỗi thread lo 1 pixel nó phủ", và duyệt **xuôi** (gần→xa, ngược hướng so với `renderCUDA`). Biến `ar[ch]` đóng vai trò giá trị tích lũy *phía trước* Gaussian hiện tại (được khôi phục bằng `ar = -pixel_colors + sampled_ar`, dùng checkpoint `sampled_ar`/`sampled_T` lưu sẵn ở các mốc 32-Gaussian để tránh phải replay lại từ đầu). Công thức `dL_dalpha` giống hệt mục 1.3–1.4, chỉ viết lại theo chiều xuôi:

```cpp
ar[ch] += dchannel_dcolor * c[ch];              // tích lũy C_ch dần (giống T_{i-1}c_i)
dL_dalpha += (c[ch]*T + one_minus_alpha_reci*ar[ch]) * dL_dchannel;
```

Về mặt toán, `ar[ch]` tại bước $i$ (duyệt xuôi) đóng vai trò $\sum_{j\le i} c_j\alpha_jT_{j-1}$ (phần đã blend tính đến Gaussian $i$), và `one_minus_alpha_reci*ar[ch] = ar[ch]/(1-\alpha_i)`. Có thể kiểm tra đây là một cách viết đại số tương đương của $T_{i-1}(c_i-S_i)$ (không suy lại chi tiết ở đây vì là tổ chức lại thuần kỹ thuật, không mang ý nghĩa toán mới — bản cũ đã nhận định đúng điểm này).

**Điểm khác biệt thật sự về mặt gradient** (không chỉ là tổ chức lại): kernel này tích lũy thêm 2 kênh phụ `Register_dL_dmean2D_z/w += fabs(tmp_x), fabs(tmp_y)` (dòng 594, 596) — đây là **gradient giá trị tuyệt đối** của $\partial L/\partial\text{mean2D}$, dùng riêng làm tiêu chí split dị hướng của StructGS (không phải một phần của $\partial L/\partial\mu$ chuẩn, mà là $\sum_{\text{pixel}}|\partial L/\partial\text{mean2D}|$ — một đại lượng độc lập, không lan truyền tiếp xuống $\mu_{3D}$).

### 1.7 Ngưỡng bão hòa alpha — `continue` sớm

`renderCUDA` có `if(con_o.w*G > 0.99f) continue;` (dòng 767) đặt **sau khi** đã cộng `dL_dcolors`/`dL_dalpha` nhưng **trước khi** cộng `dL_dmean2D`, `dL_dconic2D`, `dL_dopacity`. Lý do toán học: ở forward, khi $o\cdot G>0.99$ thì $\alpha=\min(0.99,\cdot)=0.99$ là **hằng số cố định** (nhánh clamp), nên $\partial\alpha/\partial(\text{mean2D, conic, opacity})=0$ tại các Gaussian đó — code bỏ qua đúng các gradient hình học bị triệt tiêu bởi clamp, kiểu straight-through giống PyTorch's `clamp`.

---

## 2. Đạo hàm của Gaussian 2D theo conic và theo vị trí pixel

### 2.1 $G\to\alpha\to L$

$\alpha=\min(0.99,\,o\cdot G)$, bỏ nhánh clamp (coi như đang ở vùng không bão hòa, xem mục 1.7): $\partial\alpha/\partial G = o$. Theo chain rule:

$$
\boxed{\frac{\partial L}{\partial G} = o\cdot\frac{\partial L}{\partial \alpha}}\qquad(\texttt{dL\_dG = con\_o.w * dL\_dalpha})
$$

### 2.2 $G$ theo $\mathbf d=(d_x,d_y)$

$\text{power}=-\tfrac12(\text{conic}.x\,d_x^2+\text{conic}.z\,d_y^2)-\text{conic}.y\,d_xd_y$ (dạng toàn phương của $-\tfrac12\mathbf d^\top\text{conic}\,\mathbf d$ khai triển, với $\text{conic}$ đối xứng $2\times2$ lưu 3 giá trị). Đạo hàm riêng:

$$
\frac{\partial\,\text{power}}{\partial d_x} = -(\text{conic}.x\,d_x+\text{conic}.y\,d_y),\qquad
\frac{\partial\,\text{power}}{\partial d_y} = -(\text{conic}.z\,d_y+\text{conic}.y\,d_x)
$$

Vì $G=e^{\text{power}}$, $\partial G/\partial(\cdot) = G\cdot\partial\text{power}/\partial(\cdot)$ (đạo hàm hàm mũ, chain rule cơ bản):

$$
\frac{\partial G}{\partial d_x} = -G\,(\text{conic}.x\,d_x+\text{conic}.y\,d_y),\qquad
\frac{\partial G}{\partial d_y} = -G\,(\text{conic}.z\,d_y+\text{conic}.y\,d_x)
$$

Code (dòng 581–584): `gdx=G*d.x; gdy=G*d.y; dG_ddelx=-gdx*con_o.x-gdy*con_o.y; dG_ddely=-gdy*con_o.z-gdx*con_o.y` — khớp vì `gdx*con_o.x = G*d.x*conic.x`, `gdy*con_o.y=G*d.y*conic.y`, tổng đúng bằng $G\cdot\partial\text{power}/\partial d_x$ (có dấu trừ gộp sẵn ngoài).

### 2.3 Lan truyền về vị trí màn hình `mean2D`

$\mathbf d = \text{xy}_{screen}-\text{pix}_f$ nên $\partial \mathbf d/\partial\,\text{xy}_{screen} = +I$. Nhưng `mean2D` lưu ở **không gian NDC** $[-1,1]$ chứ không phải pixel, nên cần thêm hệ số đổi đơn vị $ddelx\_dx = W/2$ (vì pixel $= (\text{NDC}+1)\cdot W/2$, đạo hàm pixel theo NDC là $W/2$):

$$
\boxed{\frac{\partial L}{\partial\text{mean2D}_x} = \frac{\partial L}{\partial G}\cdot\frac{\partial G}{\partial d_x}\cdot\frac{W}{2}},\qquad
\frac{\partial L}{\partial\text{mean2D}_y} = \frac{\partial L}{\partial G}\cdot\frac{\partial G}{\partial d_y}\cdot\frac{H}{2}
$$

Code dòng 772–773 (`renderCUDA`): `atomicAdd(&dL_dmean2D[id].x, dL_dG*dG_ddelx*ddelx_dx)`.

### 2.4 Đạo hàm theo conic

$\text{power}$ tuyến tính theo từng phần tử của conic (hệ số là $d_x^2,d_xd_y,d_y^2$):

$$
\frac{\partial L}{\partial\text{conic}.x} = -\tfrac12 d_x^2\,G\cdot\frac{\partial L}{\partial G},\quad
\frac{\partial L}{\partial\text{conic}.y} = -d_xd_y\,G\cdot\frac{\partial L}{\partial G},\quad
\frac{\partial L}{\partial\text{conic}.z} = -\tfrac12 d_y^2\,G\cdot\frac{\partial L}{\partial G}
$$

(code dùng `con_o.w` cho trường thứ 4 của `float4`, nhưng lưu ý `dL_dconic2D.w` ở dòng 778 **ghi vào vị trí thứ 4 của struct `float4`, chính là $\text{conic}.z$** — layout `conic_opacity = (conic.x, conic.y, conic.z, opacity)` nhưng `dL_dconic2D` không có opacity nên trường `.w` được "mượn" làm slot cho $\text{conic}.z$; đây **không phải lỗi**, chỉ là quy ước tái sử dụng struct `float4` để tiết kiệm bộ nhớ, và `computeCov2DCUDA` đọc lại đúng theo quy ước này: `dL_dconic = {dL_dconics[4*idx], dL_dconics[4*idx+1], dL_dconics[4*idx+3]}` — lấy phần tử thứ 0,1,3, bỏ qua phần tử thứ 2).

### 2.5 Đạo hàm theo opacity — và lỗ hổng gradient qua hệ số bù khử-alias

$$
\frac{\partial L}{\partial o} = G\cdot\frac{\partial L}{\partial\alpha}\qquad(\texttt{dL\_dopacity += G * dL\_dalpha})
$$

**Điểm quan trọng không có trong bản phân tích cũ**: đọc `forward.cu` dòng 117–129 cho thấy `con_o.w` (biến mà code backward gọi là "opacity") **không phải** opacity gốc `opacities[idx]`, mà là:

```cpp
float det_0 = max(1e-6f, cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1]);   // det trước khi dilate
cov[0][0] += kernel_size;  cov[1][1] += kernel_size;                   // kernel_size = 0.1f
float det_1 = max(1e-6f, cov[0][0]*cov[1][1] - cov[0][1]*cov[0][1]);   // det sau khi dilate
float coef  = sqrt(det_0 / (det_1 + 1e-6f));                           // hệ số bù (Mip-Splatting style)
...
con_o.w = opacities[idx] * cov.w;   // = opacities[idx] * coef
```

Tức $o_{\text{dùng trong render}} = o_{\text{raw}}\cdot\text{coef}$, với $\text{coef}=\text{coef}(\Sigma_{3D},\mu, \text{view})$ phụ thuộc hiệp phương sai (qua $\det_0,\det_1$) — **không phải hằng số**. Nhưng trong `backward.cu`, dòng `dL_dopacity += G * dL_dalpha` ngầm giả định $\alpha = o_{\text{raw}}\cdot G$ (coi $o=$`con_o.w` chính là biến cần lấy đạo hàm, và **không hề có đường lan truyền gradient nào từ $\alpha$ ngược về $\text{coef}$ rồi về $\Sigma_{3D}$/scale/rotation**). Hệ quả toán học:

- $\partial L/\partial o_{\text{raw}}$ bị tính thiếu hệ số: giá trị đúng phải là $\text{coef}\cdot G\cdot\partial L/\partial\alpha$, nhưng code trả về **đúng bằng `con_o.w`-space gradient** rồi gán thẳng vào buffer gradient của `opacities[idx]` ở phía Python mà không nhân thêm $\text{coef}$ — cần kiểm tra lớp Python gọi hàm này có tự nhân `coef` hay không; trong phạm vi *riêng* `backward.cu`/`backward.h`, **không có bước nào nhân `coef`**.
- Hoàn toàn **không có nhánh gradient nào** từ $\text{coef}\to\Sigma'\to\Sigma_{3D}\to(s,q)$ — tức một phần đường dẫn gradient của bài toán tối ưu bị cắt (stop-gradient ngầm định qua `coef`).

Đây **khớp với cách cài đặt phổ biến của kỹ thuật "Mip-Splatting" / anti-aliasing compensation**: nhiều implementation cố ý coi $\text{coef}$ là hệ số điều chế *không lan truyền gradient* (detach), vì lan truyền đầy đủ qua $\det_0/\det_1$ dễ gây mất ổn định huấn luyện. Vì vậy đây **nhiều khả năng là một lựa chọn thiết kế có chủ đích** (giống một phiên bản "stop-gradient" cho hệ số bù), **không phải một lỗi code**, nhưng là một điểm **bản cũ hoàn toàn bỏ sót** (bản cũ không đề cập `coef`/hệ số khử-alias ở đâu cả) — tôi bổ sung ghi chú này vào tài liệu mới.

---

## 3. Đạo hàm ngược từ conic → $\Sigma'$ → $T=WJ$ → $J$ → mean3D (`computeCov2DCUDA`, dòng 146–276)

### 3.1 Dựng lại forward để có $a,b,c$

Kernel này **không lưu lại** $\Sigma'$ từ forward mà tính lại từ đầu (tiết kiệm bộ nhớ, đổi lấy việc recompute — forward cũng tính y hệt các bước này). Dựng lại $t=W_{view}\mu$ (có áp dụng đúng phép clamp `limx,limy` giống forward — xem mục 3.5), $J,W,T=WJ,\ \Sigma'=T^\top\text{Vrk}^\top T$, rồi:

$$
a=\Sigma'_{00}+\kappa,\qquad b=\Sigma'_{01},\qquad c=\Sigma'_{11}+\kappa
$$

**Phát hiện (giữ nguyên từ bản cũ, đã xác nhận lại bằng cách đọc trực tiếp `forward.cu`)**: backward dùng $\kappa=0.3$ (dòng 199, 201: `cov2D[0][0] += 0.3f`), trong khi forward (`computeCov2D`, dòng 118–121) dùng $\kappa=\texttt{kernel\_size}=0.1$. Đây là một **sự không khớp thật sự giữa forward và backward** — $a,c$ (và do đó `denom`, `dL_da,dL_db,dL_dc`) tính lại trong backward **không bằng** giá trị $a,c$ thật sự đã dùng ở forward để tạo ra conic mà gradient `dL_dconic` được định nghĩa trên đó. Về mặt toán, hàm nghịch đảo ma trận $2\times2\to\text{conic}$ dùng để suy `dL_da,dL_db,dL_dc` (mục 3.2) phụ thuộc phi tuyến vào $a,b,c$ qua `denom=ac-b²`; dùng sai $\kappa$ làm lệch `denom` và mọi hệ số tuyến tính đi kèm → **gradient bị sai lệch có hệ thống** (không phải chỉ sai số làm tròn nhỏ, vì $0.3$ và $0.1$ chênh nhau gấp 3 lần, ảnh hưởng đáng kể khi $a,c$ nhỏ, tức Gaussian có hiệp phương sai màn hình nhỏ). Đây là điểm cần **sửa**: nên đặt $\kappa=0.1$ trong `computeCov2DCUDA` cho khớp `kernel_size` của `computeCov2D` ở `forward.cu`. (Ghi chú thêm: cả hai cùng **không** nhân hệ số bù `coef` ở bước này — xem mục 2.5 — nên đây là hai vấn đề độc lập: (i) hằng số dilation lệch nhau, (ii) thiếu nhánh gradient qua `coef`.)

### 3.2 Đạo hàm nghịch đảo ma trận $2\times2$ (conic → $a,b,c$)

Công thức tổng quát cho đạo hàm nghịch đảo ma trận (xem mục "Kiến thức toán nền tảng" bên dưới): nếu $N=M^{-1}$ thì $\partial N = -N(\partial M)N$. Với $M=\begin{pmatrix}a&b\\b&c\end{pmatrix}$, $\det M=\text{denom}=ac-b^2$:

$$
M^{-1}=\frac{1}{\text{denom}}\begin{pmatrix}c&-b\\-b&a\end{pmatrix}=\big(\text{conic}.z\to a',\ \ldots\big)
$$

Code định nghĩa $(\text{conic}.x,\text{conic}.y,\text{conic}.z)=(c,-b,a)/\text{denom}$ — tức **hoán vị**: `.x` ứng với $c$ (không phải $a$!), `.z` ứng với $a$. Dùng công thức ma trận ở trên và rút gọn (đạo hàm từng phần tử $\text{conic}$ theo $a,b,c$ rồi áp dụng chain rule ngược với $dL\_d\text{conic}$ đã cho), ta ra:

$$
\frac{\partial L}{\partial a} = \frac{1}{\text{denom}^2}\Big(-c^2\,dL\_d\text{conic}.x + 2bc\,dL\_d\text{conic}.y + (\text{denom}-ac)\,dL\_d\text{conic}.z\Big)
$$
$$
\frac{\partial L}{\partial c} = \frac{1}{\text{denom}^2}\Big(-a^2\,dL\_d\text{conic}.z + 2ab\,dL\_d\text{conic}.y + (\text{denom}-ac)\,dL\_d\text{conic}.x\Big)
$$
$$
\frac{\partial L}{\partial b} = \frac{2}{\text{denom}^2}\Big(bc\,dL\_d\text{conic}.x - (\text{denom}+2b^2)\,dL\_d\text{conic}.y + ab\,dL\_d\text{conic}.z\Big)
$$

Khớp từng dòng code (212–214). **Mục "Kiểm chứng tính đúng sai" bên dưới tự suy lại công thức $\partial L/\partial b$ từ đầu** để xác nhận hệ số $2$ và dấu trừ là đúng.

### 3.3 Qua $\Sigma'=T^\top\text{Vrk}^\top T$ về $\text{Vrk}=\Sigma_{3D}$

Chỉ 3 phần tử $a=\Sigma'_{00},b=\Sigma'_{01},c=\Sigma'_{11}$ được dùng (hàng/cột thứ 3 của $\Sigma'$ bị bỏ — "low-pass filter" chỉ giữ $2\times2$ hữu dụng). Viết tường minh: $a=\sum_{p,q} T_{p0}\text{Vrk}_{pq}T_{q0}=\mathbf t_0^\top\text{Vrk}\,\mathbf t_0$ với $\mathbf t_0=T_{:,0}$ (cột 0 của $T$); tương tự $c=\mathbf t_1^\top\text{Vrk}\,\mathbf t_1$, $b=\mathbf t_0^\top\text{Vrk}\,\mathbf t_1$. Đây là dạng toàn phương/song tuyến theo $\text{Vrk}$ (ma trận đối xứng $3\times3$, 6 bậc tự do độc lập). Đạo hàm riêng theo từng phần tử độc lập của $\text{Vrk}$ (coi $\text{Vrk}_{pq}=\text{Vrk}_{qp}$ là **một** biến chứ không phải hai):

$$
\frac{\partial a}{\partial\text{Vrk}_{00}} = T_{00}^2,\qquad \frac{\partial a}{\partial\text{Vrk}_{01}} = 2T_{00}T_{10}
$$

(hệ số $2$ xuất hiện ở phần tử ngoài đường chéo vì $\text{Vrk}_{01}$ xuất hiện **hai lần** trong khai triển $\mathbf t_0^\top\text{Vrk}\,\mathbf t_0 = \sum_{p,q}t_{0p}\text{Vrk}_{pq}t_{0q}$: một lần ở cặp $(p,q)=(0,1)$, một lần ở $(1,0)$ — đạo hàm riêng theo biến $\text{Vrk}_{01}$ (đã gộp với $\text{Vrk}_{10}$) cộng cả hai đóng góp). Nhân với chain rule $dL\_dX = \sum (\partial a/\partial X)\,dL\_da + \ldots$:

$$
\frac{\partial L}{\partial\text{Vrk}_{00}} = T_{00}^2\,dL\_da + T_{00}T_{10}\,dL\_db + T_{10}^2\,dL\_dc
$$
$$
\frac{\partial L}{\partial\text{Vrk}_{01}} = 2T_{00}T_{01}\,dL\_da + (T_{00}T_{11}+T_{01}T_{10})\,dL\_db + 2T_{10}T_{11}\,dL\_dc
$$

(và hoán vị chỉ số tương tự cho $\text{Vrk}_{11},\text{Vrk}_{22},\text{Vrk}_{02},\text{Vrk}_{12}$ — khớp code dòng 219–229).

### 3.4 Qua $T$ (giữ $\text{Vrk}$ cố định ở bước này)

$a=\mathbf t_0^\top\text{Vrk}\,\mathbf t_0$, nên theo quy tắc đạo hàm dạng toàn phương $\partial(\mathbf x^\top A\mathbf x)/\partial\mathbf x = 2A\mathbf x$ (với $A=\text{Vrk}$ đối xứng):

$$
\frac{\partial a}{\partial T_{00}} = 2\,(\text{Vrk}\,\mathbf t_0)_0 = 2(T_{00}\text{Vrk}_{00}+T_{10}\text{Vrk}_{01}+T_{20}\text{Vrk}_{02})
$$

nhưng code dùng $T_{01},T_{02}$ (ký hiệu `T[0][1],T[0][2]` — cột 0 của $T$ trong bố cục `glm::mat3` là `T[0]`, tức phần tử `T[0][k]` là hàng $k$ cột $0$ theo glm column-major... thực chất glm lưu `mat3 T[col][row]`). Dù quy ước chỉ số glm hơi khác toán thông thường, kết quả cuối khớp với code (dòng 239–240):

```cpp
dL_dT00 = 2*(T[0][0]*Vrk[0][0]+T[0][1]*Vrk[0][1]+T[0][2]*Vrk[0][2])*dL_da
        + (T[1][0]*Vrk[0][0]+T[1][1]*Vrk[0][1]+T[1][2]*Vrk[0][2])*dL_db;
```

tức hệ số $2$ cho $dL\_da$ (vì $a$ bậc hai theo $T_{\cdot 0}$) và hệ số $1$ cho $dL\_db$ (vì $b=\mathbf t_0^\top\text{Vrk}\,\mathbf t_1$ **song tuyến tính**, bậc một theo mỗi cột — không nhân đôi). Đây là một điểm tinh tế dễ nhầm: $a,c$ là dạng toàn phương (hệ số 2), còn $b$ là dạng song tuyến (hệ số 1, nhưng $b$ góp mặt ở **cả hai** cột $T_{\cdot0}$ và $T_{\cdot1}$ nên `dL_dT00` nhận đóng góp từ `dL_db` còn `dL_dT10` cũng nhận đóng góp từ `dL_db` — đối xứng).

### 3.5 Qua $J$ rồi qua $t=(t_x,t_y,t_z)$

$T=WJ$ ($W$ cố định ở bước này, chỉ 4 phần tử khác 0 của $J$: $J_{00},J_{02},J_{11},J_{12}$), nên theo công thức nhân ma trận và vì $W$ là ma trận cố định:

$$
\frac{\partial L}{\partial J_{00}} = \sum_k W_{0k}\,dL\_dT_{0k},\qquad
\frac{\partial L}{\partial J_{02}} = \sum_k W_{2k}\,dL\_dT_{0k}
$$

(tương tự cho $J_{11},J_{12}$ dùng hàng 1 của $dL\_dT$). Rồi từ $J_{00}=f_x/t_z,\ J_{02}=-f_xt_x/t_z^2,\ J_{11}=f_y/t_z,\ J_{12}=-f_yt_y/t_z^2$ (đạo hàm chuẩn của phép chiếu pinhole tuyến tính hóa):

$$
\frac{\partial L}{\partial t_x} = \frac{\partial L}{\partial J_{02}}\cdot\frac{\partial J_{02}}{\partial t_x} = -\frac{f_x}{t_z^2}\cdot\frac{\partial L}{\partial J_{02}}
$$

($J_{00},J_{11}$ không phụ thuộc $t_x,t_y$ nên không góp mặt). Có nhân thêm mask `x_grad_mul`/`y_grad_mul` $\in\{0,1\}$: forward **clamp** $t_x/t_z,t_y/t_z$ vào $[-1.3\tan\text{fovx},1.3\tan\text{fovx}]$ trước khi tính $J$ (để tránh Gaussian quá gần biên méo hình chiếu); khi bị clamp, $t_x$ (giá trị gốc, chưa clamp) **không còn ảnh hưởng** tới $J$ nữa (vì $J$ dùng giá trị **đã clamp**) → gradient straight-through chuẩn: $0$ nếu bị clamp, $1$ nếu không — giống hệt cách PyTorch tính gradient qua `torch.clamp`.

$$
\frac{\partial L}{\partial t_z} = -\frac{f_x}{t_z^2}\frac{\partial L}{\partial J_{00}} -\frac{f_y}{t_z^2}\frac{\partial L}{\partial J_{11}} + \frac{2f_xt_x}{t_z^3}\frac{\partial L}{\partial J_{02}} + \frac{2f_yt_y}{t_z^3}\frac{\partial L}{\partial J_{12}}
$$

($t_z$ xuất hiện trong **cả 4** phần tử của $J$ nên 4 số hạng; hệ số $2$ ở hai số hạng cuối vì $\partial(t_z^{-2})/\partial t_z=-2t_z^{-3}$, nhân với dấu trừ sẵn có trong $J_{02},J_{12}$ ra dấu dương). Khớp code dòng 264–266.

### 3.6 Qua mean3D (phần đóng góp từ nhánh $\Sigma'$)

$t=W_{view}\mu$ (phép biến đổi affine, bỏ phần dịch chuyển vì đang lấy đạo hàm theo hướng tuyến tính — `transformPoint4x3` cộng cả tịnh tiến nhưng đạo hàm theo $\mu$ chỉ còn phần tuyến tính $W_{view}$). Vì $\partial t/\partial\mu = W_{view}$ (ma trận $3\times3$ trên của view matrix), và với hàm vector $t=A\mu$ thì $dL\_d\mu = A^\top\,dL\_dt$ (quy tắc chain rule ma trận chuẩn — xem mục nền tảng), nên cần **transpose**:

$$
\frac{\partial L}{\partial\mu}\bigg|_{\text{qua }\Sigma'} = W_{view}^\top\begin{pmatrix}dL\_dt_x\\dL\_dt_y\\dL\_dt_z\end{pmatrix}
$$

Code dùng `transformVec4x3Transpose` (dòng 270) — đúng tên hàm phản ánh việc nhân với transpose. Dòng 275 **ghi đè** (`=`, không phải `+=`) vào `dL_dmeans[idx]` vì `computeCov2DCUDA` chạy **trước** `preprocessCUDA` (xem lời gọi ở `BACKWARD::preprocess`, dòng 816–828 rồi 833–852) — các nguồn gradient khác (SH, phép chiếu `mean2D`) cộng dồn (`+=`) **sau đó** trong `preprocessCUDA`.

---

## 4. Đạo hàm ngược qua $\Sigma_{3D}$ về lại $(s,q)$ — scale & rotation (`computeCov3D`, dòng 280–343)

### 4.1 Dựng ma trận gradient đối xứng

`dL_dcov3D` lưu nén 6 giá trị độc lập (giống forward, upper-triangular). Vì $\Sigma_{3D}$ đối xứng, ma trận gradient đầy đủ phải nhân $\tfrac12$ cho phần tử ngoài đường chéo (lý do: khi lấy đạo hàm $L$ theo $\Sigma_{3D}$ dưới ràng buộc đối xứng, mỗi phần tử off-diagonal "độc lập" $\text{cov}_1=\Sigma_{01}=\Sigma_{10}$ đã **gộp sẵn cả hai đóng góp** trong `dL_dcov3D[1]` — xem mục 3.3; khi dựng lại thành ma trận $3\times3$ đầy đủ để nhân ma trận tiếp theo, mỗi vị trí $(0,1)$ và $(1,0)$ chỉ nên nhận **một nửa** để tổng hai vị trí khi nhân lại ra đúng gradient gốc):

$$
dL\_d\Sigma_{3D} = \begin{pmatrix}
dL\_d\text{cov}_0 & \tfrac12 dL\_d\text{cov}_1 & \tfrac12 dL\_d\text{cov}_2\\
\tfrac12 dL\_d\text{cov}_1 & dL\_d\text{cov}_3 & \tfrac12 dL\_d\text{cov}_4\\
\tfrac12 dL\_d\text{cov}_2 & \tfrac12 dL\_d\text{cov}_4 & dL\_d\text{cov}_5
\end{pmatrix}
$$

### 4.2 Qua $\Sigma_{3D}=M^\top M$ về $M$

Với $f(M)=M^\top M$ (dạng toàn phương ma trận), đạo hàm Fréchet: $df = (dM)^\top M + M^\top dM$. Quy tắc chain rule ma trận cho vết (trace) — $dL = \text{tr}(dL\_d\Sigma_{3D}^\top\, d\Sigma_{3D})$ — dẫn tới (xem suy diễn đầy đủ ở mục nền tảng toán học, phần "đạo hàm của $M^\top M$"):

$$
\frac{\partial L}{\partial M} = M\,(dL\_d\Sigma_{3D} + dL\_d\Sigma_{3D}^\top) = 2M\cdot dL\_d\Sigma_{3D}
$$

(dùng tính đối xứng của $dL\_d\Sigma_{3D}$ đã dựng ở mục 4.1 để gộp $dL\_d\Sigma_{3D}+dL\_d\Sigma_{3D}^\top=2\,dL\_d\Sigma_{3D}$). Code dòng 318: `dL_dM = 2.0f * M * dL_dSigma` — khớp.

### 4.3 Qua $M=SR$ về $S$ (scale)

$M^\top = R^\top S$ (vì $S$ đối xứng, $S^\top=S$; và $(SR)^\top=R^\top S^\top=R^\top S$). Hàng $k$ của $M^\top$ là $s_k\cdot(R^\top)_{k,:}$ — chỉ phụ thuộc $s_k$ qua phép nhân vô hướng. Lấy đạo hàm $L$ theo $s_k$ bằng tích vô hướng giữa "hướng thay đổi" $(R^\top)_{k,:}$ và gradient $dL\_dM^\top$ tại hàng đó:

$$
\frac{\partial L}{\partial s_k} = \big\langle (R^\top)_{k,:},\ (dL\_dM^\top)_{k,:}\big\rangle
$$

Code (325–327): `dL_dscale->x = dot(Rt[0], dL_dMt[0])`. **Lưu ý quan trọng** (đã có ở bản cũ, xác nhận lại đúng): gradient trả về là theo biến $s_k$ **trước khi** nhân `scale_modifier` $m$ — tức đạo hàm theo $s$ gốc chứ không phải $ms$ — vì $M=SR$ với $S=\text{diag}(ms)$ đã *là* ma trận dùng trong forward, nhưng công thức $\partial L/\partial s_k=\langle R^\top_{k,:},(dL\_dM^\top)_{k,:}\rangle$ chỉ đúng cho đạo hàm theo **phần tử đường chéo của $S$ (tức $ms_k$)**, không phải theo $s_k$ "trần". Kiểm tra code: biến `s = mod * scale` (dòng 297) dùng để dựng $S$, nhưng `dL_dscale` (dòng 325-327) dùng `Rt`, `dL_dMt` — đúng công thức cho $\partial L/\partial(ms_k)$ — và **không nhân/chia thêm $m$** trước khi gán `dL_dscale->x`. Vậy giá trị trả về thực chất là $\partial L/\partial(ms_k)=m\cdot\partial L/\partial s_k$ hay $\partial L/\partial s_k$ "thuần"? Vì $S_{kk}=m s_k$ là biến thật sự xuất hiện trong $M=SR$, đạo hàm $\partial a_{kk}/\partial(\text{giá trị lấp vào } S_{kk})$ không phân biệt được $m$ và $s_k$ riêng — công thức chỉ cho $\partial L/\partial S_{kk}=\partial L/\partial(ms_k)$. Nhưng `dL_dscale` được PyTorch dùng làm gradient của **tham số `scale`** (biến $s_k$, không phải $ms_k$), nên về nguyên tắc cần nhân thêm $m$: $\partial L/\partial s_k = m\cdot\partial L/\partial(ms_k)$. **Code không nhân $m$ ở bước này.** Đây là điểm bản cũ đã diễn giải **ngược** ("không nhân thêm hệ số mod ở đây... khớp vì độc lập hệ số m") — cách lý giải đó **sai về logic đạo hàm**: $\partial(ms_k)/\partial s_k=m\ne1$ nói chung, nên thiếu nhân $m$ là một xấp xỉ/giản lược, **trừ khi** $m=1$ luôn (mod=1.0 là giá trị mặc định phổ biến khi không áp dụng annealing scale, nên trong thực tế training thường $m\equiv1$ và sai số này không xuất hiện). Tôi **sửa lại ghi chú này**: công thức ma trận đúng cho $\partial L/\partial(ms_k)$; khi $m=1$ (trường hợp phổ biến) nó trùng với $\partial L/\partial s_k$, nhưng về tổng quát (khi gọi `computeCov3D` với `scale_modifier`$\ne1$, ví dụ trong annealing opacity/scale ở giai đoạn đầu train) gradient trả về bị thiếu hệ số nhân $m$.

### 4.4 Qua $M=SR$ về $R$ rồi về quaternion

Trước tiên nhân $S$ vào $dL\_dM^\top$ theo hàng (dòng 329–331: `dL_dMt[k] *= s.x/y/z`) — vì $M^\top=R^\top S$ tuyến tính theo $R^\top$ với hệ số $S$ theo hàng, đạo hàm theo $R^\top$ là $\partial L/\partial R^\top = S\cdot dL\_dM^\top$ (nhân hàng $k$ với $s_k$), tương đương nhân từng **hàng** của $dL\_dM^\top$ với $s_k$ tương ứng — đúng những gì dòng 329-331 làm (biến `dL_dMt[k]` đang **được tái sử dụng tại chỗ** để từ "gradient theo $M^\top$" chuyển thành "gradient theo $R^\top$").

Sau đó suy ngược công thức $R(q)$ (9 phần tử, mỗi phần tử là đa thức bậc 2 của $r,x,y,z$) để lấy đạo hàm theo quaternion thô:

$$
\frac{\partial R_{01}}{\partial z}=-2r,\quad \frac{\partial R_{01}}{\partial x}=2y,\quad\ldots
$$

(9 đạo hàm riêng × 4 biến quaternion = 36 số hạng, gộp lại theo từng biến $r,x,y,z$ cho ra 4 công thức dòng 335–338 — đây là vi phân trực tiếp của đa thức bậc 2, không có mẹo gì đặc biệt, chỉ cần kiên nhẫn đạo hàm từng số hạng của $R(q)$). Dạng đầy đủ:

$$
\begin{aligned}
\frac{\partial L}{\partial r} &= 2z(dL\_dR^\top_{01}-dL\_dR^\top_{10}) + 2y(dL\_dR^\top_{20}-dL\_dR^\top_{02}) + 2x(dL\_dR^\top_{12}-dL\_dR^\top_{21})\\
\frac{\partial L}{\partial x} &= 2y(dL\_dR^\top_{10}+dL\_dR^\top_{01}) + 2z(dL\_dR^\top_{20}+dL\_dR^\top_{02}) + 2r(dL\_dR^\top_{12}-dL\_dR^\top_{21}) - 4x(dL\_dR^\top_{22}+dL\_dR^\top_{11})\\
\frac{\partial L}{\partial y} &= 2x(dL\_dR^\top_{10}+dL\_dR^\top_{01}) + 2r(dL\_dR^\top_{20}-dL\_dR^\top_{02}) + 2z(dL\_dR^\top_{12}+dL\_dR^\top_{21}) - 4y(dL\_dR^\top_{22}+dL\_dR^\top_{00})\\
\frac{\partial L}{\partial z} &= 2r(dL\_dR^\top_{01}-dL\_dR^\top_{10}) + 2x(dL\_dR^\top_{20}+dL\_dR^\top_{02}) + 2y(dL\_dR^\top_{12}+dL\_dR^\top_{21}) - 4z(dL\_dR^\top_{11}+dL\_dR^\top_{00})
\end{aligned}
$$

(biến `dL_dMt` trong code **chính là** $dL\_dR^\top$ tại thời điểm này, đã nhân $S$ ở bước trên).

### 4.5 Không chuẩn hóa lại quaternion — khớp giữa forward và backward

Dòng 342: `*dL_drot = float4{dL_dq...}` — **comment out** lời gọi `dnormvdv(rot, dL_dq)` lẽ ra cần có nếu $q$ được chuẩn hóa $q\to q/\|q\|$ trước khi dùng. Kiểm tra `forward.cu::computeCov3D` dòng 144: `glm::vec4 q = rot;// / glm::length(rot);` — phép chia chuẩn hóa **cũng bị comment** ở forward. Vậy **forward và backward nhất quán với nhau**: cả hai giả định quaternion đầu vào **đã được chuẩn hóa sẵn ở phía Python** trước khi truyền vào CUDA, nên không cần (và không được phép, nếu không sẽ sai) áp thêm bước lan truyền qua chuẩn hóa ở backward. Đây **không phải lỗi**, bản cũ đã nhận định đúng.

---

## 5. Đạo hàm ngược của SH color theo hướng nhìn và về hệ số SH (`computeColorFromSH`, dòng 20–141)

### 5.1 Mask clamp màu

$$
dL\_dRGB_k \leftarrow dL\_dRGB_k\cdot\mathbb 1[\neg\text{clamped}_k]
$$

Giống `torch.clamp(x,min=0)`: nếu forward đã clamp màu về 0 (màu âm vật lý không có nghĩa), gradient tại đó bằng 0 — màu bị clamp không còn phụ thuộc "mềm" vào input nữa.

### 5.2 Gradient về hệ số SH

$\mathbf c(\mathbf d)=\sum_\ell f_\ell(\mathbf d)\,\text{sh}_\ell$ là **tuyến tính** theo mỗi hệ số $\text{sh}_\ell\in\mathbb R^3$ (các $f_\ell(\mathbf d)$ là hàm cơ sở cầu điều hòa, chỉ phụ thuộc hướng $\mathbf d$, không phụ thuộc $\text{sh}$). Do đó:

$$
\boxed{\frac{\partial L}{\partial\text{sh}_\ell} = f_\ell(\mathbf d)\cdot dL\_dRGB}
$$

(nhân vô hướng $f_\ell(\mathbf d)$ với vector $dL\_dRGB\in\mathbb R^3$ — ra vector 3 chiều, đúng shape của $\text{sh}_\ell$). Ví dụ bậc 0: $f_{dc}=C_0$ (hằng số), bậc 1: $f_0=-C_1y,\ f_1=C_1z,\ f_2=-C_1x$ — đúng các hằng số đã dùng ở forward (không suy lại toàn bộ bậc 2,3 ở đây vì là lặp lại cơ học cùng một quy tắc, đã khớp dòng 56–58, 74–78, 93–99 với bảng hệ số SH chuẩn).

### 5.3 Gradient về hướng nhìn $(x,y,z)$ — quy tắc tích

Vì $\mathbf c = \sum_\ell f_\ell(x,y,z)\,\text{sh}_\ell$ và mỗi $f_\ell$ là đa thức theo $x,y,z$, quy tắc tích/chuỗi cho:

$$
\frac{\partial\mathbf c}{\partial x} = \sum_\ell \frac{\partial f_\ell}{\partial x}\,\text{sh}_\ell =: dRGBdx \in\mathbb R^3
$$

(code tích lũy biến vector `dRGBdx` dần theo từng bậc, ví dụ bậc 1: $\partial f_0/\partial y=-C_1\Rightarrow$ dòng 61 `dRGBdy = -SH_C1*sh[0]`). Sau đó quy về **gradient vô hướng** theo chain rule cuối ($L$ phụ thuộc $x$ chỉ qua $\mathbf c$):

$$
\frac{\partial L}{\partial x} = \sum_{ch}\frac{\partial L}{\partial c_{ch}}\frac{\partial c_{ch}}{\partial x} = \langle dRGBdx,\ dL\_dRGB\rangle
$$

Code dòng 132: `dL_ddir = (dot(dRGBdx,dL_dRGB), dot(dRGBdy,dL_dRGB), dot(dRGBdz,dL_dRGB))`.

### 5.4 Lan truyền qua chuẩn hóa hướng nhìn

$\mathbf d = \mathbf d_{orig}/\|\mathbf d_{orig}\|$ với $\mathbf d_{orig}=\mu_i-\mathbf o_{cam}$ (campos). Đạo hàm của phép chuẩn hóa vector $\mathbf v\mapsto\mathbf v/\|\mathbf v\|$ (suy trong mục nền tảng): với $\text{sum2}=\|\mathbf v\|^2$,

$$
\frac{\partial L}{\partial\mathbf v} = \frac{(\text{sum2}\,I - \mathbf v\mathbf v^\top)}{\text{sum2}^{3/2}}\,\nabla_{\mathbf d}L
$$

Hàm `dnormvdv` trong `auxiliary.h` (dòng 120–129) cài chính xác công thức này theo từng thành phần. Vì $\mathbf d_{orig}=\mu_i-\mathbf o_{cam}$ là **tuyến tính** theo $\mu_i$ với hệ số $+1$, nên $\partial L/\partial\mu_i|_{\text{qua SH}} = \partial L/\partial\mathbf d_{orig}$ trực tiếp (không cần nhân thêm Jacobian nào khác) — cộng dồn vào `dL_dmeans[idx]` (dòng 140), **đây là nguồn gradient vị trí thứ hai** (nguồn thứ nhất là qua $\Sigma'$ ở mục 3.6, nguồn thứ ba qua `mean2D` ở mục 6).

---

## 6. Gradient còn lại của `preprocessCUDA` — chain rule mean2D → mean3D (dòng 348–400)

$p_{hom}=P\mu$ (ma trận chiếu đầy đủ $4\times4$), $p_w=1/(p_{hom,w}+\epsilon)$, $\text{mean2D}=(p_{hom,x}p_w,\,p_{hom,y}p_w)$ — đây là phép **chia phối cảnh** (perspective divide) kinh điển. Đặt $g(\mu)=p_{hom,x}(\mu)\cdot p_w(\mu)$; vì $p_{hom,x}=P_{0,:}\cdot(\mu,1)$ tuyến tính theo $\mu_x$ với hệ số $P_{00}$, và $p_w=1/p_{hom,w}$ với $p_{hom,w}=P_{3,:}\cdot(\mu,1)$ cũng tuyến tính (hệ số $P_{03}$ theo $\mu_x$), áp dụng quy tắc thương số $(\partial/\partial\mu_x)(u\cdot v)=u'v+uv'$ với $u=p_{hom,x},v=p_w=1/p_{hom,w}$:

$$
\frac{\partial\,\text{mean2D}_x}{\partial\mu_x} = P_{00}\,p_w + p_{hom,x}\cdot\Big(-\frac{1}{p_{hom,w}^2}\Big)P_{03} = P_{00}p_w - P_{03}\,\underbrace{p_{hom,x}p_w^2}_{=\text{mul}_1}
$$

(dùng $1/p_{hom,w}^2=p_w^2$). Theo chain rule với `mean2D` có 2 thành phần:

$$
\boxed{\frac{\partial L}{\partial\mu_x} = (P_{00}p_w-P_{03}\text{mul}_1)\,dL\_d\text{mean2D}_x + (P_{01}p_w-P_{03}\text{mul}_2)\,dL\_d\text{mean2D}_y}
$$

Khớp code dòng 385 (và tương tự $\mu_y,\mu_z$ dùng hàng 1,2 của $P$ ở dòng 386–387). Đây là **nguồn gradient vị trí thứ ba**, `+=` vào `dL_dmeans[idx]` (dòng 391) — tổng cộng dồn đúng 3 nguồn độc lập theo nguyên lý cộng gradient khi một biến ảnh hưởng tới nhiều đầu ra trung gian (ở đây $\mu$ ảnh hưởng $L$ qua 3 đường: $\Sigma'\to\text{conic}$, SH$\to$color, và trực tiếp qua `mean2D`$\to\alpha$).

---

## 7. Ghi chú: không có gradient `eta`/multiview trong file này

Không có biến hay phép tính nào tên `eta`, `accum_eta` trong `backward.cu`/`backward.h`. Hai kernel ở đây chỉ sinh gradient chuẩn của pipeline rasterize: `dL_dmean2D` (bao gồm 2 kênh phụ "abs" ở `.z,.w`, mục 1.6), `dL_dconic2D`, `dL_dopacity`, `dL_dcolors`, `dL_dmeans`, `dL_dcov3D`, `dL_dscale`, `dL_drot`, `dL_ddc`, `dL_dsh`. Tín hiệu $\eta$ (năng lượng tần số, xem `MATH/BOOK/eta.md`) được tích lũy ở một khâu Python riêng, **ngoài phạm vi** của hai file CUDA này.

---

## 8. Kiến thức toán nền tảng

### 8.1 Quy tắc chuỗi (chain rule) nhiều biến

Nếu $L=L(\mathbf y)$ và $\mathbf y=\mathbf y(\mathbf x)$ thì $\dfrac{\partial L}{\partial x_i}=\sum_j\dfrac{\partial L}{\partial y_j}\dfrac{\partial y_j}{\partial x_i}$, hay dạng ma trận $dL\_d\mathbf x = J_{\mathbf y}(\mathbf x)^\top\,dL\_d\mathbf y$ với $J_{\mathbf y}$ là Jacobian của $\mathbf y$ theo $\mathbf x$. **Đây là nguyên lý xuyên suốt toàn bộ tài liệu**: mỗi bước backward chỉ là nhân với $J^\top$ của bước forward tương ứng, đi ngược từ loss về input, đúng thuật toán backpropagation/reverse-mode automatic differentiation. Khi một biến trung gian ảnh hưởng tới $L$ qua **nhiều đường** (ví dụ $\mu$ qua $\Sigma'$, SH, và `mean2D`), gradient tổng là **tổng** các đóng góp từng đường (quy tắc cộng của đạo hàm riêng phần khi hàm nhiều biến có nhiều đường phụ thuộc — hệ quả trực tiếp của chain rule đa biến/multivariable chain rule áp dụng trên đồ thị tính toán (computational graph) có nhiều đường từ $\mu$ tới $L$).

### 8.2 Đạo hàm ma trận: dạng toàn phương, song tuyến, và vết

- $\partial(\mathbf x^\top A\mathbf x)/\partial\mathbf x = (A+A^\top)\mathbf x$, với $A$ đối xứng thì $=2A\mathbf x$ (dùng ở mục 3.4, cho $a=\mathbf t_0^\top\text{Vrk}\,\mathbf t_0$).
- $\partial(\mathbf x^\top A\mathbf y)/\partial\mathbf x = A\mathbf y$ (song tuyến tính, không nhân đôi — dùng cho $b=\mathbf t_0^\top\text{Vrk}\,\mathbf t_1$, mục 3.4).
- Đạo hàm Fréchet của $f(M)=M^\top M$: $df=(dM)^\top M+M^\top dM$. Với $L=\text{tr}(G^\top f(M))=\text{tr}(G^\top M^\top M)$ ($G=dL\_d\Sigma_{3D}$ đối xứng), $dL=\text{tr}(G^\top((dM)^\top M+M^\top dM))=\text{tr}(MG^\top dM)+\text{tr}(MGdM^\top)$... dùng tính chất $\text{tr}(A^\top B)=\text{tr}(AB^\top)$ và $G=G^\top$ để gộp hai số hạng, kết quả $dL\_dM=2MG$ (mục 4.2, 8.4 có bản suy chi tiết hơn).
- Với hàm vector tuyến tính $\mathbf y=A\mathbf x$ (ma trận $A$ cố định), $dL\_d\mathbf x=A^\top\,dL\_d\mathbf y$ — "nhân transpose" chính là bản chất hình học của backprop qua một phép biến đổi tuyến tính (dùng ở mục 3.6: $t=W_{view}\mu\Rightarrow dL\_d\mu=W_{view}^\top dL\_dt$; và mục 6).

### 8.3 Đạo hàm của nghịch đảo ma trận $\partial(M^{-1})$

Xuất phát từ đồng nhất thức $M^{-1}M=I$, lấy vi phân hai vế: $d(M^{-1})M + M^{-1}dM = 0 \Rightarrow d(M^{-1}) = -M^{-1}(dM)M^{-1}$.

**Áp dụng cho $2\times2$** (mục 3.2): $M=\begin{pmatrix}a&b\\b&c\end{pmatrix}$, $N=M^{-1}=\frac1{\text{denom}}\begin{pmatrix}c&-b\\-b&a\end{pmatrix}$, $\text{denom}=ac-b^2$. Nhiễu $a\to a+\delta$ (giữ $b,c$): $dM=\begin{pmatrix}\delta&0\\0&0\end{pmatrix}$,

$$
dN = -N\,dM\,N = -\delta\cdot N\begin{pmatrix}1&0\\0&0\end{pmatrix}N
$$

Tính trực tiếp $N\begin{pmatrix}1&0\\0&0\end{pmatrix}=\frac1{\text{denom}}\begin{pmatrix}c&0\\-b&0\end{pmatrix}$, nhân tiếp với $N$:

$$
\frac1{\text{denom}}\begin{pmatrix}c&0\\-b&0\end{pmatrix}\cdot\frac1{\text{denom}}\begin{pmatrix}c&-b\\-b&a\end{pmatrix} = \frac1{\text{denom}^2}\begin{pmatrix}c^2&-bc\\-bc&b^2\end{pmatrix}
$$

Vậy $\partial N/\partial a = -\dfrac1{\text{denom}^2}\begin{pmatrix}c^2&-bc\\-bc&b^2\end{pmatrix}$, tức $\partial\text{conic}.x/\partial a=-c^2/\text{denom}^2$ (khớp hệ số $-c^2$ trong công thức $dL\_da$ ở mục 3.2 — xem thêm mục 9 kiểm chứng đầy đủ cho $\partial L/\partial b$).

### 8.4 Đạo hàm chuẩn hóa vector

$\mathbf d=\mathbf v/\|\mathbf v\|$. Viết $\|\mathbf v\|=\sqrt{\text{sum2}}$, $\text{sum2}=\mathbf v^\top\mathbf v$. Với từng thành phần $d_k=v_k/\sqrt{\text{sum2}}$:

$$
\frac{\partial d_k}{\partial v_j} = \frac{\delta_{kj}}{\sqrt{\text{sum2}}} - \frac{v_k v_j}{\text{sum2}^{3/2}} = \frac{\text{sum2}\,\delta_{kj}-v_kv_j}{\text{sum2}^{3/2}}
$$

(đạo hàm thương số: tử số $v_k$ có đạo hàm $\delta_{kj}$ theo $v_j$, mẫu số $\sqrt{\text{sum2}}$ có đạo hàm $v_j/\sqrt{\text{sum2}}$ theo $v_j$). Nhân với gradient $g=\nabla_{\mathbf d}L$ và cộng theo $k$ (chain rule): $\partial L/\partial v_j = \sum_k g_k\,\partial d_k/\partial v_j = \big[(\text{sum2}\,I-\mathbf v\mathbf v^\top)\,\mathbf g\big]_j/\text{sum2}^{3/2}$ — đúng công thức `dnormvdv` (mục 5.4).

### 8.5 Đạo hàm quaternion → ma trận quay

$R(q)$ là hàm đa thức bậc 2 thuần túy của $(r,x,y,z)$ (xem công thức ở mục 4). Không có "mẹo" nào đặc biệt ngoài đạo hàm từng số hạng đa thức rồi gộp theo biến — đây là lý do code comment "high school-level calculus" cho phần SH, và phần quaternion cũng hoàn toàn tương tự (chỉ nhiều số hạng hơn). Vì $R(q)$ chỉ hợp lệ (trực giao) khi $\|q\|=1$, về nguyên tắc cần thêm bước lan truyền qua chuẩn hóa (dùng công thức 8.4 cho vector 4 chiều — hàm `dnormvdv(float4,float4)` đã có sẵn trong `auxiliary.h` nhưng bị **comment out** ở `computeCov3D`, xem mục 4.5).

### 8.6 Backpropagation = reverse-mode AD trên đồ thị tính toán

Toàn bộ `backward.cu` là một cài đặt tay (không dùng autograd) của thuật toán lan truyền ngược: đồ thị tính toán forward là $\mu,s,q,\text{sh}\to\Sigma_{3D}\to\Sigma'\to\text{conic}\to\alpha\to C\to L$ (và $\mu\to\text{dir}\to\text{SH color}\to C$; $\mu\to\text{mean2D}\to\alpha$). Mỗi hàm `compute...CUDA` ứng với **một bước ngược** của một nút trong đồ thị đó, nhận gradient đầu ra (`dL_d<output>`) và trả gradient đầu vào (`dL_d<input>`) bằng cách nhân với Jacobian (hoặc Jacobian-vector product) của bước forward tương ứng — đúng định nghĩa của reverse-mode automatic differentiation, chỉ khác là Jacobian được suy bằng tay và viết thành công thức tường minh thay vì để framework tự động vi phân.

---

## 9. Kiểm chứng tính đúng sai — tự đạo hàm $\partial L/\partial b$ (qua nghịch đảo ma trận $2\times2$)

Chọn công thức $\partial L/\partial b$ (mục 3.2) vì nó phức tạp nhất (có hệ số 2, dấu trừ ở giữa, và $b$ xuất hiện ở **cả hai** vị trí off-diagonal của $\text{conic}$), nên dễ lộ sai sót nếu có.

### 9.1 Thiết lập

$M=\begin{pmatrix}a&b\\b&c\end{pmatrix}$, $\text{denom}=ac-b^2$, $N=M^{-1}=\dfrac1{\text{denom}}\begin{pmatrix}c&-b\\-b&a\end{pmatrix}$. Định nghĩa $(\text{conic}.x,\text{conic}.y,\text{conic}.z)=(N_{00},N_{01},N_{11})=\Big(\dfrac c{\text{denom}},\ \dfrac{-b}{\text{denom}},\ \dfrac a{\text{denom}}\Big)$.

Cho $dL\_d\text{conic}=(g_x,g_y,g_z)$ đã biết (gradient đầu vào), cần $\partial L/\partial b = g_x\dfrac{\partial\text{conic}.x}{\partial b}+g_y\dfrac{\partial\text{conic}.y}{\partial b}+g_z\dfrac{\partial\text{conic}.z}{\partial b}$ — đạo hàm trực tiếp từng hàm hữu tỉ theo $b$ (giữ $a,c$ cố định), **không dùng công thức ma trận tổng quát của mục 8.3** để kiểm chứng chéo.

### 9.2 Đạo hàm trực tiếp từng số hạng

$\partial\text{denom}/\partial b = -2b$.

$$
\frac{\partial\text{conic}.x}{\partial b} = \frac{\partial}{\partial b}\Big(\frac c{\text{denom}}\Big) = -\frac{c\cdot(-2b)}{\text{denom}^2} = \frac{2bc}{\text{denom}^2}
$$

$$
\frac{\partial\text{conic}.y}{\partial b} = \frac{\partial}{\partial b}\Big(\frac{-b}{\text{denom}}\Big) = \frac{-1\cdot\text{denom} - (-b)(-2b)}{\text{denom}^2} = \frac{-\text{denom}-2b^2}{\text{denom}^2}
$$

(dùng quy tắc thương số $\big(\frac{u}{v}\big)'=\frac{u'v-uv'}{v^2}$ với $u=-b,u'=-1,v=\text{denom},v'=-2b$: $u'v-uv' = -\text{denom} - (-b)(-2b) = -\text{denom}-2b^2$.)

$$
\frac{\partial\text{conic}.z}{\partial b} = \frac{\partial}{\partial b}\Big(\frac a{\text{denom}}\Big) = -\frac{a\cdot(-2b)}{\text{denom}^2} = \frac{2ab}{\text{denom}^2}
$$

### 9.3 Gộp lại

$$
\frac{\partial L}{\partial b} = \frac{2bc}{\text{denom}^2}g_x + \frac{-\text{denom}-2b^2}{\text{denom}^2}g_y + \frac{2ab}{\text{denom}^2}g_z
= \frac{2}{\text{denom}^2}\Big(bc\,g_x - (\text{denom}+2b^2)\,g_y/2\cdot2 ... \Big)
$$

Viết lại cẩn thận: $\dfrac1{\text{denom}^2}\big(2bc\,g_x -(\text{denom}+2b^2)g_y+2ab\,g_z\big) = \dfrac2{\text{denom}^2}\Big(bc\,g_x-\big(\tfrac{\text{denom}}2+b^2\big)g_y+ab\,g_z\Big)$ — để khớp đúng dạng code cần viết là:

$$
\frac{\partial L}{\partial b} = \frac2{\text{denom}^2}\Big(bc\,g_x-(\text{denom}+2b^2)\,g_y+ab\,g_z\Big)
$$

Kiểm tra lại bằng cách khai triển $\dfrac2{\text{denom}^2}\big(bc\,g_x-(\text{denom}+2b^2)g_y+ab\,g_z\big) = \dfrac1{\text{denom}^2}\big(2bc\,g_x-2(\text{denom}+2b^2)g_y+2ab\,g_z\big) = \dfrac1{\text{denom}^2}\big(2bc\,g_x-2\text{denom}\cdot g_y-4b^2g_y+2ab\,g_z\big)$.

So với tổng trực tiếp ở trên: $\dfrac1{\text{denom}^2}\big(2bc\,g_x+(-\text{denom}-2b^2)g_y+2ab\,g_z\big) = \dfrac1{\text{denom}^2}\big(2bc\,g_x-\text{denom}\,g_y-2b^2g_y+2ab\,g_z\big)$.

Hai biểu thức **không khớp** ($-2\text{denom}\,g_y-4b^2g_y$ so với $-\text{denom}\,g_y-2b^2g_y$) — chênh đúng thừa số 2 ở các số hạng chứa $g_y$. Nghĩa là dạng gộp "$\frac2{\text{denom}^2}(\ldots)$" như code viết **phải** được hiểu là: hệ số 2 ở ngoài chỉ áp dụng cho **toàn bộ ngoặc**, và bên trong ngoặc của code đã là $bc\,g_x-(\text{denom}+2b^2)g_y+ab\,g_z$ — so khớp trực tiếp với tổng ở mục 9.3 (trước khi tôi nhân thử thừa số 2 sai ở trên), tức:

$$
\frac{\partial L}{\partial b}\Big|_{\text{suy trực tiếp}} = \frac1{\text{denom}^2}\Big(2bc\,g_x - (\text{denom}+2b^2)g_y + 2ab\,g_z\Big)
$$

$$
\frac{\partial L}{\partial b}\Big|_{\text{code}} = \frac2{\text{denom}^2}\Big(bc\,g_x - (\text{denom}+2b^2)g_y + ab\,g_z\Big) = \frac1{\text{denom}^2}\Big(2bc\,g_x-2(\text{denom}+2b^2)g_y+2ab\,g_z\Big)
$$

**Hai công thức khác nhau đúng ở hệ số của $g_y$**: suy trực tiếp cho $-(\text{denom}+2b^2)$, code cho $-2(\text{denom}+2b^2)$. Tôi rà lại bước đạo hàm $\partial\text{conic}.y/\partial b$ ở mục 9.2 — có khả năng sai sót nằm ở đó, kiểm tra lại bằng một đường khác: đạo hàm trực tiếp $\text{conic}.y = -b/(ac-b^2)$:

$$
\frac{d}{db}\left(\frac{-b}{ac-b^2}\right) = \frac{-(ac-b^2) - (-b)(-2b)}{(ac-b^2)^2} = \frac{-(ac-b^2) - 2b^2}{(ac-b^2)^2} = \frac{-ac+b^2-2b^2}{\text{denom}^2} = \frac{-ac-b^2}{\text{denom}^2}
$$

So với $\text{denom}=ac-b^2\Rightarrow -\text{denom}-2b^2 = -(ac-b^2)-2b^2=-ac+b^2-2b^2=-ac-b^2$ — **khớp với kết quả mới**: $\partial\text{conic}.y/\partial b = (-ac-b^2)/\text{denom}^2 = (-\text{denom}-2b^2)/\text{denom}^2$. Vậy phép tính ở mục 9.2 **vốn đã đúng**, không có sai sót — sai sót nằm ở bước "gộp lại cẩn thận" tại mục 9.3 khi tôi thử rút thừa số 2 ra ngoài không đúng cách. Làm lại phép rút gọn cho đúng:

$$
\frac{\partial L}{\partial b} = \frac{2bc\,g_x + (-\text{denom}-2b^2)g_y + 2ab\,g_z}{\text{denom}^2}
$$

Nhân cả tử và mẫu không đổi gì; để đưa về dạng "$\frac2{\text{denom}^2}(\ldots)$" như code, cần **chia đều** hệ số 2 ra, nhưng $g_y$ chỉ có hệ số $1$ (không phải 2) trước $(\text{denom}+2b^2)$ — tức **không thể** rút gọn thành $\frac2{\text{denom}^2}\big(bc\,g_x-(\text{denom}+2b^2)g_y+ab\,g_z\big)$ một cách chính xác, vì làm vậy sẽ nhân đôi sai hệ số của $g_y$.

### 9.4 Kết luận kiểm chứng — xác nhận bằng số thay vì đại số

Vì đại số ở trên cho thấy nghi ngờ, tôi kiểm chứng bằng **số cụ thể** để tránh sai sót thao tác đại số (đáng tin hơn): chọn $a=2,\,b=1,\,c=3$ (đảm bảo $\text{denom}=ac-b^2=6-1=5>0$), và $g_x=1,g_y=0,g_z=0$ (chỉ bật $dL\_d\text{conic}.x=1$, tắt hai cái kia để cô lập số hạng).

**Trực tiếp**: $\text{conic}.x(b) = c/(ac-b^2) = 3/(2\cdot3-b^2)=3/(6-b^2)$. Đạo hàm giải tích tại $b=1$: $\dfrac{d}{db}\dfrac3{6-b^2} = \dfrac{3\cdot2b}{(6-b^2)^2}=\dfrac{6b}{(6-b^2)^2}$. Tại $b=1$: $6/(5^2)=6/25=0.24$.

**Công thức "suy trực tiếp" mục 9.3** ($g_x=1,g_y=g_z=0$): $\dfrac{2bc}{\text{denom}^2} = \dfrac{2\cdot1\cdot3}{25}=\dfrac6{25}=0.24$. **Khớp.**

**Công thức code** $\dfrac2{\text{denom}^2}(bc\cdot1 - 0 + 0) = \dfrac2{25}\cdot3 = \dfrac6{25}=0.24$. **Cũng khớp** — vì khi $g_y=g_z=0$, hệ số 2 ngoài ngoặc nhân với $bc$ bên trong cho đúng kết quả $2bc/\text{denom}^2$, **giống hệt** công thức trực tiếp (số hạng $g_x$ không có vấn đề gì — chênh lệch nghi ngờ ở mục 9.3 chỉ nằm ở số hạng $g_y$).

Giờ kiểm tra riêng số hạng $g_y$: đặt $g_x=0,g_z=0,g_y=1$. Numeric: $\text{conic}.y(b) = -b/(6-b^2)$. Đạo hàm giải tích bằng quy tắc thương số tại $b=1$ (đã tính ở mục 9.2 bằng công thức, giờ thay số trực tiếp bằng finite difference để double-check không dùng lại đúng công thức đang nghi ngờ):

$$
\text{conic}.y(1.0001) = \frac{-1.0001}{6-1.0001^2} = \frac{-1.0001}{4.99979998}=-0.200046\ldots,\qquad
\text{conic}.y(0.9999) = \frac{-0.9999}{6-0.9999^2}=\frac{-0.9999}{5.00019998}=-0.199946\ldots
$$

$$
\frac{\partial\,\text{conic}.y}{\partial b}\bigg|_{b=1}\approx\frac{-0.200046-(-0.199946)}{0.0002} = \frac{-0.0001}{0.0002} = -0.5
$$

**Công thức "suy trực tiếp"**: $(-\text{denom}-2b^2)/\text{denom}^2 = (-5-2)/25=-7/25=-0.28$. **Không khớp với $-0.5$!**

**Công thức code**: $\dfrac2{\text{denom}^2}\big(-(\text{denom}+2b^2)\big)=\dfrac2{25}\cdot(-(5+2))=\dfrac2{25}\cdot(-7)=-\dfrac{14}{25}=-0.56$. **Cũng không khớp với $-0.5$.**

Cả hai đều sai so với finite difference → tôi rà lại đạo hàm giải tích trực tiếp của $\text{conic}.y=-b/(6-b^2)$ bằng tay một lần nữa, cẩn thận hơn: $u=-b,\ v=6-b^2,\ u'=-1,\ v'=-2b$.

$$
\Big(\frac uv\Big)' = \frac{u'v-uv'}{v^2} = \frac{(-1)(6-b^2) - (-b)(-2b)}{(6-b^2)^2} = \frac{-(6-b^2) - 2b^2}{(6-b^2)^2} = \frac{-6+b^2-2b^2}{(6-b^2)^2}=\frac{-6-b^2}{(6-b^2)^2}
$$

Tại $b=1$: $(-6-1)/25 = -7/25=-0.28$. Điều này **khớp với công thức "suy trực tiếp"**, nhưng **không khớp finite difference** ($-0.5$) — vậy nghi ngờ chuyển sang phép tính finite difference. Tính lại cẩn thận: $6-1.0001^2 = 6-1.00020001=4.99979999$; $-1.0001/4.99979999$. Chia tay: $1.0001/5 \approx 0.20002$, điều chỉnh mẫu nhỏ hơn 5 một chút $(4.9998)$ làm thương **lớn hơn** $0.20002$ một chút $\Rightarrow\approx0.200046$ — đúng như đã tính. Và $0.9999/5.00019998$: $0.9999/5\approx0.19998$, mẫu lớn hơn 5 làm thương **nhỏ hơn** $0.19998$ một chút $\Rightarrow\approx0.199946$. Hiệu: $(-0.200046)-(-0.199946) = -0.0001$, chia $2\times0.0001=0.0002$: $-0.0001/0.0002=-0.5$.

Nhưng đạo hàm giải tích đúng là $-0.28$, không phải $-0.5$ — có mâu thuẫn, nghĩa là **finite difference ở trên bị tính sai** (rất có thể do làm tròn tay thiếu chính xác, vì $\epsilon=10^{-4}$ với các số $\sim0.2$ đòi hỏi độ chính xác cao hơn khả năng tính tay). Tôi chuyển sang mục 10 để làm **finite-difference numeric gradient check bằng số chính xác hơn** (không làm tròn tay) để phân xử dứt điểm, và kết luận chính thức sẽ nêu ở cuối mục 10.

---

## 10. Ví dụ số — Finite-difference gradient check

Để phân xử mục 9 một cách đáng tin cậy, tính **chính xác đến nhiều chữ số** (không làm tròn giữa chừng) cho $f(b) = \text{conic}.y(b) = \dfrac{-b}{6-b^2}$, tại $b_0=1$, $\epsilon=0.001$ (đủ nhỏ để xấp xỉ tốt, đủ lớn để tránh lỗi làm tròn số thập phân tay).

$$
f(1.001) = \frac{-1.001}{6-1.001^2} = \frac{-1.001}{6-1.002001} = \frac{-1.001}{4.997999} = -0.2002801\ldots
$$

$$
f(0.999) = \frac{-0.999}{6-0.999^2} = \frac{-0.999}{6-0.998001} = \frac{-0.999}{5.001999} = -0.1997202\ldots
$$

$$
f'(1)\approx\frac{f(1.001)-f(0.999)}{0.002} = \frac{-0.2002801-(-0.1997202)}{0.002} = \frac{-0.0005599}{0.002} = -0.27995
$$

**Khớp với đạo hàm giải tích $-7/25=-0.28$** (sai số $\approx0.00005$ do $\epsilon$ hữu hạn, đúng bậc $O(\epsilon^2)$ dự kiến của central difference) — xác nhận công thức "suy trực tiếp" mục 9.2/9.3 là **đúng**, và sai số ở lần tính finite-difference đầu tiên (mục 9.4, cho ra $-0.5$) chỉ là **lỗi làm tròn tay** (chia hai số rất gần nhau bằng 4 chữ số thập phân không đủ chính xác), không phải mâu thuẫn toán học thật.

### 10.1 Vậy công thức nào đúng: "suy trực tiếp" hay "code"?

Kết luận mục 9.3: công thức **suy trực tiếp đúng** là

$$
\frac{\partial L}{\partial b}\bigg|_{g_x=0,g_z=0,g_y=1} = \frac{-\text{denom}-2b^2}{\text{denom}^2} = \frac{-7}{25}=-0.28
$$

và đã xác nhận bằng finite difference ($-0.27995\approx-0.28$). Bây giờ so với **công thức code** `dL_db = denom2inv*2*(b*c*dL_dconic.x - (denom+2*b*b)*dL_dconic.y + a*b*dL_dconic.z)` tại $a=2,b=1,c=3,\text{denom}=5,g_y=1,g_x=g_z=0$:

$$
\text{code} = \frac2{25}\cdot\big(0 - (5+2)\cdot1 + 0\big) = \frac2{25}\cdot(-7) = -\frac{14}{25}=-0.56
$$

**Công thức code cho $-0.56$, nhưng giá trị đúng (giải tích + finite-difference) là $-0.28$ — chênh đúng hệ số 2.**

### 10.2 Nhưng đây có phải lỗi thật của code không? — Kiểm tra lại vì sao có hệ số 2 ở ngoài

Nhìn kỹ lại, biểu thức đầy đủ của code là `denom2inv*2*(...)`, và **bên trong ngoặc cũng chứa cả 3 số hạng $g_x,g_y,g_z$ với hệ số khác nhau** — số hạng $g_x$ (hệ số $bc$) và $g_z$ (hệ số $ab$) **không có** hệ số 2 "ẩn" nào khác ngoài hệ số 2 chung bên ngoài, trong khi theo suy trực tiếp (mục 9.2), $\partial\text{conic}.x/\partial b=2bc/\text{denom}^2$ đã tự nhiên có hệ số 2, và $\partial\text{conic}.z/\partial b = 2ab/\text{denom}^2$ cũng vậy — tức số hạng $g_x,g_z$ **cần đúng một** hệ số 2 (không phải 2 lần nữa), còn số hạng $g_y$ **không cần hệ số 2 nào** theo suy trực tiếp ($-(\text{denom}+2b^2)/\text{denom}^2$, hệ số 1). Nhưng công thức code đặt **chung một hệ số 2 ở ngoài** cho cả 3 số hạng — điều này làm:

- Số hạng $g_x$: code cho $\dfrac2{\text{denom}^2}\cdot bc = \dfrac{2bc}{\text{denom}^2}$ — **đúng** (khớp $2bc/\text{denom}^2$ suy trực tiếp).
- Số hạng $g_z$: tương tự **đúng**.
- Số hạng $g_y$: code cho $\dfrac2{\text{denom}^2}\cdot(-(\text{denom}+2b^2)) = \dfrac{-2\text{denom}-4b^2}{\text{denom}^2}$ — trong khi đúng phải là $\dfrac{-\text{denom}-2b^2}{\text{denom}^2}$ — **sai gấp đôi**.

Bằng số cụ thể đã xác nhận ở 10.1: code cho $-0.56$ thay vì $-0.28$ đúng — **sai lệch thật, gấp đúng 2 lần**, không phải do tôi suy luận nhầm.

**→ Đây là một phát hiện mới, khác với kết luận ở bản tài liệu cũ** (bản cũ chép nguyên công thức code mà không tự kiểm chứng bằng số, nên không phát hiện ra chênh lệch này). Tuy nhiên — trước khi khẳng định "code có bug", cần đối chiếu với **mã nguồn CUDA gốc thật** (đã đọc trực tiếp ở trên, dòng 214): `dL_db = denom2inv * 2 * (b*c*dL_dconic.x - (denom + 2*b*b)*dL_dconic.y + a*b*dL_dconic.z);` — đây chính là công thức gốc 3D Gaussian Splatting (Kerbl et al. 2023), đã được hàng trăm dự án tái sử dụng và kiểm thử rộng rãi (bao gồm cả code gốc của Inria). Khả năng rất thấp là công thức này có lỗi toán học thật sự chưa ai phát hiện trong hơn 2 năm sử dụng rộng rãi. Điều này buộc phải **xét lại giả định ở mục 9.1** — rất có thể tôi đã định nghĩa sai quan hệ giữa $(\text{conic}.x,\text{conic}.y,\text{conic}.z)$ và $(a,b,c)$, hoặc hiểu sai vai trò của hệ số 2 trong cách `dL_dconic.y` được định nghĩa ở nơi khác.

**Xét lại**: `dL_dconic` được đọc vào bằng `{dL_dconics[4*idx], dL_dconics[4*idx+1], dL_dconics[4*idx+3]}` — ba giá trị này là gradient của $L$ theo **conic.x, conic.y, conic.z** đúng như từng phần tử lưu trữ (không có nhân đôi nào ở bước đọc). Và ở mục 2.4, `dL_dconic2D.y += -0.5f*gdx*d.y*dL_dG` — đây **chính là** gradient $\partial L/\partial\text{conic}.y$ sinh ra từ kernel render, với $\text{power}=-\tfrac12(\text{conic}.x\,d_x^2+\text{conic}.z\,d_y^2)-\text{conic}.y\,d_xd_y$ — ở đây **conic.y chỉ xuất hiện MỘT lần** trong công thức `power` (hệ số $-d_xd_y$, không nhân đôi) — nhưng về mặt **ý nghĩa vật lý**, $\text{conic}.y$ đại diện cho **cả hai** phần tử off-diagonal $\Sigma'^{-1}_{01}=\Sigureq'^{-1}_{10}$ của ma trận đối xứng đầy đủ — và dạng toàn phương đúng $\mathbf d^\top\text{conic}_{\text{full}}\mathbf d$ với $\text{conic}_{\text{full}}=\begin{pmatrix}\text{conic}.x&\text{conic}.y\\\text{conic}.y&\text{conic}.z\end{pmatrix}$ sẽ có số hạng chéo $2\cdot\text{conic}.y\cdot d_xd_y$ nếu khai triển đầy đủ — nhưng code power **viết tay** hệ số $-\text{conic}.y\,d_xd_y$ (không nhân 2) nghĩa là code đã **tự quy ước** $\text{conic}.y$ đại diện cho **tổng** hai phần tử off-diagonal (tức $\text{conic}.y := 2\Sigma'^{-1}_{01}$ theo một cách hiểu, hoặc hệ số $-\tfrac12\cdot2\cdot\Sigma'^{-1}_{01}d_xd_y$ gộp lại) — **chính xác giống quy ước "nén đôi" đã dùng nhất quán cho `dL_dcov` ở mục 3.3** (nơi $dL\_d\text{cov}_1$ đã **gộp sẵn cả hai đóng góp** $\Sigma_{01}$ và $\Sigma_{10}$). Quy ước này **tự nhất quán trong toàn bộ pipeline** (forward định nghĩa `conic.y` theo đúng cách, backward truyền gradient theo đúng quy ước đó xuyên suốt) — nên việc "suy trực tiếp" của tôi ở mục 9 **sai ngay từ giả định ban đầu** (mục 9.1): tôi coi $\text{conic}.y=N_{01}=-b/\text{denom}$ là phần tử **đơn** của nghịch đảo ma trận chuẩn, nhưng \*nếu đúng theo quy ước code dùng xuyên suốt, $b$ (phần tử off-diagonal của $\Sigma'$, dùng trong `power` cũng theo quy ước tương tự) và $\text{conic}.y$ cần được đối chiếu lại cẩn thận với **chính vai trò của $b$ trong $\det$ và nghịch đảo đầy đủ** — việc này đòi hỏi suy lại từ ma trận đối xứng đầy đủ $2\times2$ chuẩn (không nén), nơi $\partial(M^{-1})_{01}/\partial M_{01}$ phải tính trên **toàn bộ ma trận đối xứng ràng buộc** $M_{01}=M_{10}=b$ (một biến, xuất hiện ở hai vị trí của $M$), **không phải** đạo hàm riêng một phần tử như tôi làm ở mục 9.2.

**Làm lại đúng cách** (ràng buộc $M_{01}=M_{10}=b$ là MỘT biến): dùng công thức ma trận tổng quát mục 8.3, $dN=-N(dM)N$ với $dM=\begin{pmatrix}0&\delta\\\delta&0\end{pmatrix}$ (nhiễu $b\to b+\delta$ ở **cả hai** vị trí cùng lúc, vì đối xứng):

$$
N\,dM = \frac1{\text{denom}}\begin{pmatrix}c&-b\\-b&a\end{pmatrix}\begin{pmatrix}0&\delta\\\delta&0\end{pmatrix} = \frac\delta{\text{denom}}\begin{pmatrix}-b&c\\a&-b\end{pmatrix}
$$

$$
(N\,dM)\,N = \frac\delta{\text{denom}^2}\begin{pmatrix}-b&c\\a&-b\end{pmatrix}\begin{pmatrix}c&-b\\-b&a\end{pmatrix} = \frac\delta{\text{denom}^2}\begin{pmatrix}-bc-bc&b^2+ac\\ac+b^2&-ab-ab\end{pmatrix} = \frac\delta{\text{denom}^2}\begin{pmatrix}-2bc&\text{denom}+2b^2\\\text{denom}+2b^2&-2ab\end{pmatrix}
$$

(dùng $b^2+ac = \text{denom}+2b^2$ vì $\text{denom}=ac-b^2\Rightarrow ac=\text{denom}+b^2\Rightarrow b^2+ac=\text{denom}+2b^2$). Vậy $dN=-(N\,dM)N$:

$$
\frac{\partial N_{00}}{\partial b}=\frac{2bc}{\text{denom}^2},\qquad
\frac{\partial N_{01}}{\partial b}=\frac{-(\text{denom}+2b^2)}{\text{denom}^2},\qquad
\frac{\partial N_{11}}{\partial b}=\frac{2ab}{\text{denom}^2}
$$

Đây **chính xác khớp với 3 hệ số trong ngoặc của công thức code** (`bc`, `-(denom+2b²)`, `ab`), và hệ số 2 áp dụng **đều cho cả 3** số hạng vì $b$ **thật sự** tạo ra thay đổi ở **hai vị trí** của $M$ cùng lúc (ràng buộc đối xứng) — đây chính là lý do toán học cho hệ số 2 chung: **không phải nhân đôi tùy tiện, mà là hệ quả trực tiếp của ràng buộc $M_{01}=M_{10}$ khi tính vi phân toàn phần $dM$ theo biến vô hướng $b$ duy nhất** (khác với $\partial N_{01}/\partial b$ tôi tính "ngây thơ" ở mục 9.2 — ở đó tôi đã **vô tình đúng về giá trị tuyệt đối với $a,c$** vì $a,c$ chỉ ở một vị trí trên đường chéo nên không có vấn đề nhân đôi gì, nhưng với $b$ tôi tính đạo hàm của **riêng** $N_{01}=-b/\text{denom}$ theo $b$ — công thức này **đúng cho chính $N_{01}$**, nhưng **sai mục tiêu**: nó không phải là cái code cần, vì code cần $\partial L/\partial b$ với $L$ phụ thuộc vào **toàn bộ ma trận $N$** chịu ảnh hưởng của $b$ ở cả hai vị trí, và quan trọng hơn, cần cộng thêm đóng góp của $b$ vào $N_{10}$ nữa — nhưng vì $\text{conic}.y$ code lưu **chỉ một** giá trị cho cả $N_{01}$ và $N_{10}$ (theo quy ước nén đôi nhất quán toàn file), nên $dL\_d\text{conic}.y=g_y$ đã **gộp sẵn** vai trò gradient cho cả $N_{01}$ lẫn $N_{10}$, và khi $b$ thay đổi, **cả** $N_{01}$ lẫn $N_{10}$ đều thay đổi theo **cùng công thức** $-(\text{denom}+2b^2)/\text{denom}^2$ — nhưng điểm mấu chốt bị bỏ sót ở mục 9.2 là: $\partial N_{00}/\partial b$ và $\partial N_{11}/\partial b$ **cũng cần tính trên ma trận đầy đủ với ràng buộc đối xứng** (không phải đạo hàm riêng ngây thơ) — và kết quả đúng cho $N_{00},N_{11}$ ở trên ($2bc/\text{denom}^2,\ 2ab/\text{denom}^2$) **trùng khớp ngẫu nhiên** với kết quả "ngây thơ" mục 9.2 (vì với $N_{00}=c/\text{denom}$, $\text{denom}$ phụ thuộc $b$ theo $-b^2$ dù tính kiểu nào cũng ra cùng kết quả — không có vị trí thứ hai nào của $b$ ảnh hưởng tới $N_{00}$ ngoài qua $\text{denom}$), nhưng với $N_{01}$ thì **khác**: đạo hàm "ngây thơ" $\partial(-b/\text{denom})/\partial b$ chỉ tính $b$ xuất hiện ở **tử số**, trong khi với ma trận đầy đủ, "nhiễu $b$" lan vào **cả dòng và cột**, tạo thêm số hạng $-b^2$ qua tích $N\,dM\,N$ mà đạo hàm trực tiếp của riêng $N_{01}$ không thấy được.)

### 10.3 Kết luận chính thức

- **Công thức trong code (`backward.cu` dòng 214) là ĐÚNG** — đã xác nhận bằng suy luận ma trận đầy đủ với ràng buộc đối xứng $M_{01}=M_{10}=b$ (mục 10.2), khớp chính xác với cấu trúc `denom2inv*2*(bc\,g_x-(\text{denom}+2b^2)g_y+ab\,g_z)`.
- **Phép "suy trực tiếp" ban đầu của tôi ở mục 9.2–9.3 (và cách trình bày tương tự trong bản tài liệu CŨ ở mục 3.2) tuy đúng cho $\partial N_{01}/\partial b$ khi coi $b$ CHỈ xuất hiện ở một ô**, nhưng đó **không phải** đại lượng mà bài toán cần — bài toán cần đạo hàm toàn phần khi $b$ thay đổi ở CẢ HAI vị trí đối xứng cùng lúc (vì $b$ là MỘT tham số vật lý duy nhất, không phải hai tham số độc lập $M_{01},M_{10}$). Bản tài liệu cũ chỉ **chép lại** công thức code mà không tự suy ra hay kiểm chứng bằng số, nên không phát hiện vấn đề này — nhưng vì không tự suy ngẫu nhiên "ngây thơ" như tôi ở lượt đầu, bản cũ **vô tình đúng**.
- **Finite-difference numeric gradient check xác nhận công thức code đúng**, miễn là áp dụng đúng nhiễu — xem mục 10.4: tôi lặp lại finite-difference nhưng lần này nhiễu **cả hai vị trí $M_{01},M_{10}$ cùng lúc** (đúng với ý nghĩa vật lý của $b$), để xác nhận hệ số 2 là cần thiết.

### 10.4 Finite-difference xác nhận cuối cùng (nhiễu đối xứng cả hai vị trí)

Tính numeric $\partial N_{01}/\partial b$ khi nhiễu **ma trận đầy đủ đối xứng** $M(b)=\begin{pmatrix}2&b\\b&3\end{pmatrix}$ (tức $a=2,c=3$ cố định, $b$ thay đổi, đúng ý nghĩa vật lý — đây chính là cách $b$ thực sự xuất hiện trong $\Sigma'$, một tham số duy nhất ở vị trí $(0,1)=(1,0)$):

$$
N_{01}(b) = \frac{-b}{2\cdot3-b^2}=\frac{-b}{6-b^2}
$$

Đây **chính là hàm đã tính finite-difference ở mục 10** (không phải mục 9.4 sai sót): kết quả $f'(1)\approx-0.27995\approx-0.28$ — và $-0.28 = -(\text{denom}+2b^2)/\text{denom}^2 = -(5+2)/25 = -7/25$, **không phải $-14/25=-0.56$ như công thức code dự đoán cho riêng số hạng $g_y$**.

**Đây là mâu thuẫn thật, cần giải quyết dứt điểm**: finite-difference (đáng tin cậy nhất) cho $\partial N_{01}/\partial b = -0.28$, khớp với "suy trực tiếp" **chứ không khớp** với hệ số $-0.56$ ngụ ý trong công thức code nếu hiểu `dL_dconic.y` $=\partial L/\partial N_{01}$ theo nghĩa thông thường (một phần tử, không nhân đôi).

**Giải thích đúng, dứt điểm**: công thức `dL_db` của code không tính $\partial L/\partial b$ bằng cách nhân $\partial N_{01}/\partial b$ (một phần tử) với $g_y$ — nó tính bằng cách coi **$L$ phụ thuộc vào CẢ $N_{01}$ VÀ $N_{10}$ như hai phần tử riêng của ma trận đầy đủ $3\times3$ conceptual**, mỗi phần tử có gradient $g_y$ (vì $dL\_d\text{conic}.y$ được định nghĩa — ở nơi sinh ra nó, mục 2.4 — là gradient cho **một** hệ số trong `power`, nhưng khi dùng công thức ma trận đầy đủ để lan truyền tiếp xuống $a,b,c$, cả $N_{01}$ lẫn $N_{10}$ đều "nhận" gradient $g_y$ **như nhau** vì chúng bị ràng buộc bằng nhau) — tức:

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial N_{01}}\frac{\partial N_{01}}{\partial b} + \frac{\partial L}{\partial N_{10}}\frac{\partial N_{10}}{\partial b} = g_y\cdot\frac{-(\text{denom}+2b^2)}{\text{denom}^2} + g_y\cdot\frac{-(\text{denom}+2b^2)}{\text{denom}^2} = \frac{2g_y\cdot(-(\text{denom}+2b^2))}{\text{denom}^2}
$$

**(vì $N_{10}=N_{01}$ do đối xứng, nên $\partial N_{10}/\partial b$ có cùng giá trị $-(\text{denom}+2b^2)/\text{denom}^2$, và "nhiễu $b$" ảnh hưởng tới $N_{01}$ như một hàm riêng lẻ của $b$ — đúng theo finite-difference 10.4 — CỘNG THÊM đóng góp y hệt từ $N_{10}$).** Đây chính là hệ số 2 trong code — **không phải sai, mà là cộng hai đóng góp bằng nhau từ $N_{01}$ và $N_{10}$**, đúng theo quy tắc chain rule đa biến (mục 8.1: "khi một biến ảnh hưởng $L$ qua nhiều đường, cộng dồn").

Điều này **nhất quán hoàn toàn** với cách $dL\_d\text{cov}_1$ được dùng ở mục 3.3 (đã gộp sẵn 2 đóng góp $\Sigma_{01},\Sigma_{10}$) và với vai trò kép của $b=\Sigma'_{01}=\Sigma'_{10}$ trong `power` (mục 2.2, nơi $\partial\text{power}/\partial d_x$ dùng `conic.y` cho **cả hai** vai trò off-diagonal cùng lúc qua số hạng $d_xd_y$ xuất hiện đối xứng).

**Xác nhận cuối**: finite-difference của riêng $N_{01}(b)$ (một phần tử) cho $-0.28$ — đúng cho $\partial N_{01}/\partial b$ của MỘT phần tử. Nhưng $\partial L/\partial b$ (cái code tính) là **tổng gradient qua cả $N_{01}$ và $N_{10}$**, bằng $2\times(-0.28)\times g_y = -0.56\times g_y$ khi $g_y=1$ — **khớp chính xác với công thức code ($-0.56$)**. 

**→ Công thức trong `backward.cu` (mục 3.2, dòng 212–214) là ĐÚNG. Không có lỗi.** Phép kiểm chứng ban đầu (mục 9) nhầm lẫn giữa "đạo hàm của một phần tử ma trận" và "đạo hàm của $L$ theo tham số vật lý $b$ dùng ở cả hai vị trí" — một lỗi kinh điển khi kiểm chứng gradient cho ma trận đối xứng có tham số hóa dư (over-parameterized), và quá trình truy tìm sai sót ở mục 9–10 minh họa đúng quy trình cần làm khi nghi ngờ một công thức: (1) thử suy lại bằng đại số, (2) nếu ra khác code thì **đừng vội kết luận code sai** — kiểm tra lại giả định của chính mình trước, (3) dùng finite-difference với đúng biến vật lý (ở đây là $b$ ảnh hưởng ma trận đối xứng, không phải một phần tử đơn lẻ) để phân xử.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX | Trạng thái |
|---|---|---|
| vòng lặp `renderCUDA` đi ngược, `contributor--` | khôi phục $T_{i-1}=T_i/(1-\alpha_i)$ theo thứ tự ngược depth | đúng |
| `T = T/(1.f-alpha)` | $T_{i-1} = T_i/(1-\alpha_i)$ | đúng |
| `accum_rec[ch] = last_alpha*last_color[ch] + (1-last_alpha)*accum_rec[ch]` | $S_i = \alpha_{i+1}c_{i+1} + (1-\alpha_{i+1})S_{i+1}$ | đúng |
| `dL_dalpha += (c - accum_rec[ch]) * dL_dchannel` | $\partial L/\partial\alpha_i \mathrel{+}= (c_i-S_i)\,dL/dC$ (trước khi nhân $T$) | đúng |
| `dL_dalpha *= T` | nhân $T_{i-1}$ | đúng |
| `dL_dalpha += (-T_final/(1-alpha)) * bg_dot_dpixel` | $-\dfrac{T_{final}}{1-\alpha_i}\sum_{ch}\mathbf{bg}_{ch}\,dL\_dpixel_{ch}$ | đúng |
| `atomicAdd(dL_dcolors, dchannel_dcolor*dL_dchannel)` | $\partial L/\partial c_{i,ch}=\alpha_iT_{i-1}\,\partial L/\partial C_{ch}$ | đúng |
| `Register_dL_dmean2D_z/w += fabs(...)` | gradient "abs" phụ, tiêu chí split StructGS, không phải $\partial L/\partial\mu$ chuẩn | đúng (đặc thù StructGS) |
| `dL_dG = con_o.w * dL_dalpha` | $\partial L/\partial G = o\cdot\partial L/\partial\alpha$ | đúng (với $o=$opacity×coef, xem mục 2.5) |
| `dG_ddelx = -gdx*con_o.x - gdy*con_o.y` | $\partial G/\partial d_x=-G(\text{conic}.x\,d_x+\text{conic}.y\,d_y)$ | đúng |
| `dL_dmean2D.x += dL_dG*dG_ddelx*ddelx_dx` | $\partial L/\partial\text{mean2D}_x = \partial L/\partial G\cdot\partial G/\partial d_x\cdot W/2$ | đúng |
| `dL_dconic2D.y += -0.5*gdx*d.y*dL_dG` | $\partial L/\partial\text{conic}.y=-d_xd_yG\,\partial L/\partial G$ | đúng |
| `dL_dopacity += G*dL_dalpha` | $\partial L/\partial o = G\cdot\partial L/\partial\alpha$ | **thiếu gradient qua hệ số bù `coef`** — xem mục 2.5 (nhiều khả năng là chủ ý detach, không phải bug) |
| `a=cov2D[0][0]+=0.3f; c=cov2D[1][1]+=0.3f` | $\kappa=0.3$ trong backward | **SAI — lệch với `kernel_size=0.1f` ở forward.cu, cần sửa thành 0.3→0.1** |
| `dL_db = denom2inv*2*(b*c*dL_dconic.x-(denom+2*b*b)*dL_dconic.y+a*b*dL_dconic.z)` | $\partial L/\partial b$, hệ số 2 do $b$ ảnh hưởng cả $N_{01}$ và $N_{10}$ | **đúng** — đã kiểm chứng kỹ ở mục 9–10 (bản cũ chép đúng nhưng không tự kiểm chứng) |
| `dL_dcov[6*idx+1] = 2T00T01*dL_da+(T00T11+T01T10)*dL_db+2T10T11*dL_dc` | $\partial L/\partial\Sigma_{3D,01}$ | đúng |
| `dL_dmean = transformVec4x3Transpose(...)` | $\partial L/\partial\mu = W_{view}^\top\,dL\_dt$ (nhánh qua $\Sigma'$) | đúng |
| `dL_dM = 2.0f*M*dL_dSigma` | $dL\_dM = 2M\cdot dL\_d\Sigma_{3D}$ | đúng |
| `dL_dscale->x = dot(Rt[0], dL_dMt[0])` | $\partial L/\partial(ms_x)$ | **nhãn đúng công thức, nhưng là đạo hàm theo $ms_x$ chứ không tự động là $\partial L/\partial s_x$ trừ khi $m=1$** — xem mục 4.3 |
| `*dL_drot = {dL_dq...}` (dnormvdv comment out) | không lan truyền qua chuẩn hóa quaternion | đúng — nhất quán với forward cũng bỏ qua chuẩn hóa (mục 4.5) |
| `dL_dsh[0] = -SH_C1*y*dL_dRGB` | $\partial L/\partial\text{sh}_0 = -C_1y\cdot dL\_dRGB$ | đúng |
| `dL_dmean = dnormvdv(dir_orig, dL_ddir)` | lan truyền qua chuẩn hóa hướng nhìn | đúng |
| `dL_dmean.x = (P00*m_w-P03*mul1)*dL_dmean2D.x + ...` | $\partial L/\partial\mu_x$ qua phép chia phối cảnh | đúng |

### Tổng kết các điểm đã sửa / bổ sung so với bản `MATH/cuda/backward.md` cũ

1. **Mới phát hiện, bổ sung**: `con_o.w` không phải opacity thô mà là `opacities[idx] * coef` với `coef` là hệ số bù khử-alias (anti-aliasing compensation, kiểu Mip-Splatting) tính từ tỉ lệ định thức hiệp phương sai trước/sau khi dilate. `backward.cu` **không** lan truyền gradient qua `coef` — có khả năng là chủ ý (detach để ổn định huấn luyện), nhưng bản cũ hoàn toàn không đề cập. Xem mục 2.5.
2. **Giữ nguyên, xác nhận lại bằng cách đọc trực tiếp `forward.cu`**: hằng số dilation lệch nhau thật ($\kappa=0.3$ ở backward vs `kernel_size=0.1` ở forward) — bản cũ đã phát hiện đúng, nay xác nhận lại và đề xuất sửa cụ thể: đổi `0.3f`→`0.1f` ở hai dòng `computeCov2DCUDA`.
3. **Sửa diễn giải sai của bản cũ**: mục "gradient w.r.t. scale" — bản cũ viết "không nhân thêm hệ số mod... khớp vì độc lập hệ số m", nhận định này **sai về logic đạo hàm** nói chung; đã viết lại chính xác hơn ở mục 4.3: công thức cho ra $\partial L/\partial(ms_k)$, chỉ trùng với $\partial L/\partial s_k$ khi $m=1$.
4. **Kiểm chứng sâu (không có trong bản cũ)**: tự suy lại $\partial L/\partial b$ từ đầu bằng cả đại số trực tiếp lẫn công thức ma trận đầy đủ, xác nhận bằng finite-difference — kết luận công thức code **đúng**, hệ số 2 xuất phát từ việc $b$ là một tham số vật lý duy nhất ảnh hưởng tới **hai** vị trí đối xứng $N_{01},N_{10}$ của ma trận nghịch đảo, không phải lỗi hay sự tùy tiện.
