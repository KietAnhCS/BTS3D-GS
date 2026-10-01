# Công thức toán của `get_multiscale_structure_tensor_v1`

**Ký hiệu:** $L$ = `levels`, $\sigma_0$ = `base_sigma`, $s$ = `octave_step`, $p$ = `power_factor`. $G_\sigma$ là Gaussian, $*$ là tích chập. $I=(I_1,\dots,I_C)$ là ảnh vào.

Ý tưởng chung: hàm lấy **hướng** từ structure tensor ở mức nhiều scale, lấy **tần số** từ năng lượng DoG, rồi trộn hai thứ đó lại. Kết quả là một tensor có trace bằng "tần số bình phương hiệu dụng".

---

## 1. Tensor gốc (scale mịn nhất)

$$\hat S^{0}=\begin{bmatrix}\hat S^0_{xx}&\hat S^0_{xy}\\ \hat S^0_{xy}&\hat S^0_{yy}\end{bmatrix}=\text{StructureTensor}(I;\ \sigma=\sigma_0,\ \rho=1)$$

Đây chính là hàm ở câu trước (Di Zenzo, Sobel, đã chia $\max\operatorname{tr}$). `rho` không được truyền vào nên lấy mặc định $\rho=1$.

---

## 2. Thang scale của từng level

Với $i=0,\dots,L-1$:

$$f_i=\frac{1}{s^{\,i}},\qquad \sigma_i=\sigma_0\,s^{\,i}$$

$f_i$ là tần số quy ước của band $i$. Chú ý $\sigma_i f_i=\sigma_0$ luôn không đổi, nên $f_i$ tỉ lệ nghịch với scale.

Đặt $\sigma_{-1}=0$. Mỗi vòng lặp blur thêm một lượng vừa đủ để tổng sigma đạt $\sigma_i$:

$$\sigma^{\text{inc}}_i=\sqrt{\max\!\left(10^{-6},\ \sigma_i^2-\sigma_{i-1}^2\right)}$$

Vì phương sai Gaussian cộng dồn nên ảnh làm mượt lũy tiến chính là ảnh làm mượt trực tiếp với $\sigma_i$:

$$A_0=I,\qquad A_{i+1}=G_{\sigma^{\text{inc}}_i}*A_i\;=\;G_{\sigma_i}*I$$

Do đó $A_i=G_{\sigma_{i-1}}*I$ (với $A_0=I$).

---

## 3. Năng lượng băng tần (Difference of Gaussians)

$$D_{i,c}=A_i-A_{i+1}=G_{\sigma_{i-1}}*I_c-G_{\sigma_i}*I_c$$

Gộp các kênh bằng chuẩn L2:

$$r_i=\sqrt{\sum_{c=1}^{C}D_{i,c}^{\,2}}$$

$r_i(x,y)$ cho biết pixel đó có bao nhiêu chi tiết nằm trong dải scale $[\sigma_{i-1},\sigma_i]$.

Làm mượt không gian (chỉ với $i>0$ và `smoothing_factor` $>0$):

$$r_i\leftarrow G_{2\sigma_i}*r_i$$

`smoothing_factor` chỉ đóng vai trò công tắc bật/tắt. Giá trị của nó không đi vào công thức.

---

## 4. Tensor hướng ở scale $i$

Làm mượt tensor gốc với cửa sổ lớn hơn nhiều:

$$\rho_i=3\sigma_i,\qquad S^{i}_{\bullet}=G_{\rho_i}*\hat S^{0}_{\bullet},\quad \bullet\in\{xx,xy,yy\}$$

Chuẩn hoá về trace $\approx 1$ (giữ lại hướng và độ đồng nhất, bỏ độ lớn):

$$N^{i}_{\bullet}=\frac{S^{i}_{\bullet}}{S^i_{xx}+S^i_{yy}+10^{-6}}$$

Khi $\rho_i$ càng lớn, tensor càng thể hiện hướng của cấu trúc lớn (đường cong, viền dài), không phải hướng cục bộ của từng pixel.

---

## 5. Trọng số và cộng dồn

Trọng số của scale $i$ tại mỗi pixel:

$$w_i=r_i^{\,p}$$

Với $p=3$, scale nào có năng lượng trội sẽ áp đảo rất mạnh. Ví dụ $r$ gấp đôi thì $w$ gấp $8$ lần.

$$\text{Acc}_\bullet=\sum_{i=0}^{L-1}w_i\,f_i^{2}\,N^{i}_\bullet,\qquad W=\sum_{i=0}^{L-1}w_i$$

---

## 6. Kết quả cuối

$$S^{\text{final}}_\bullet=\frac{\text{Acc}_\bullet}{W+10^{-6}}$$

Đặt trọng số chuẩn hoá $\omega_i=\dfrac{w_i}{\sum_j w_j}$ (tổng bằng 1) thì:

$$\boxed{S^{\text{final}}=\sum_{i=0}^{L-1}\omega_i\ f_i^{2}\ N^{i}}$$

Đây là **trung bình có trọng số** của các tensor hướng đơn vị $N^i$, mỗi cái nhân $f_i^2$.

**Output:** $(B,3,H,W)$ gồm $[S^{\text{final}}_{xx},S^{\text{final}}_{xy},S^{\text{final}}_{yy}]$.

---

## 7. Ý nghĩa của trace

Vì $\operatorname{tr}(N^i)\approx 1$:

$$\operatorname{tr}\!\left(S^{\text{final}}\right)\approx\sum_i\omega_i f_i^2=f_{\text{eff}}^2$$

Tức là:

$$f_{\text{eff}}=\sqrt{\operatorname{tr}(S^{\text{final}})},\qquad \text{wavelength}\approx\frac{1}{f_{\text{eff}}}$$

Hướng nằm ở phần còn lại:

$$\theta=\tfrac12\operatorname{atan2}\!\left(2S_{xy},\,S_{xx}-S_{yy}\right),\qquad \mathcal C=\left(\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}\right)^2$$

Với mặc định $s=1.5$, $L=3$: $f=(1,\,0.667,\,0.444)$ nên $f^2=(1,\,0.444,\,0.198)$. Vùng nào có chi tiết mịn thì trace gần $1$. Vùng nào chỉ có cấu trúc thô thì trace gần $0.2$.

---

## Tóm tắt luồng

$$I\to\hat S^0\ \ \Big\|\ \ I\to\{A_i\}\to D_i\to r_i\to w_i=r_i^p$$

$$\hat S^0\xrightarrow{G_{3\sigma_i}}S^i\xrightarrow{/\operatorname{tr}}N^i\ \Longrightarrow\ S^{\text{final}}=\frac{\sum_i w_i f_i^2N^i}{\sum_i w_i}$$

---

## Vài điểm cần để ý trong code

1. **Vùng phẳng:** khi mọi $r_i\approx0$ thì $W\approx0$, và $\varepsilon=10^{-6}$ trong mẫu số lấn át, nên output tiến về $0$ (không phải giá trị trung tính). Với $p=3$ hiện tượng này xảy ra dễ hơn ($r=0.01\Rightarrow w=10^{-6}$).
2. **$N^i$ có thể nhỏ hơn 1 trace ở vùng phẳng:** vì $\hat S^0$ được chuẩn hoá theo max toàn ảnh, nên chỗ trace $\ll10^{-6}$ thì $\varepsilon$ cũng làm $N^i$ bị co lại.
3. **Band 0 khác các band còn lại:** $A_0=I$ chưa blur, nên $D_0$ chứa cả nhiễu tần số cao nhất.
4. **Tham số `aggregation_mode` không được dùng** trong hàm.
5. **Chi phí:** $\rho_i=3\sigma_i$ làm kernel size $\approx 24\sigma_i$, nên level lớn tốn tính toán nếu `fast_gaussian_blur` không tách separable.