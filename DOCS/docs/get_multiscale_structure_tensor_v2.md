# Công thức toán của `get_multiscale_structure_tensor_v2`

**Ký hiệu:** giống v1. $L$ = `levels`, $\sigma_0$ = `base_sigma`, $s$ = `octave_step`, $p$ = `power_factor`, $G_\sigma$ là Gaussian, $*$ là tích chập, $\star$ là cross-correlation.

v2 giữ nguyên phần **tần số và trọng số** của v1. Chỉ có **cách lấy tensor hướng** ở mỗi level là khác.

---

## 1. Phần giống v1 (nhắc lại ngắn)

Thang scale, với $i=0,\dots,L-1$:

$$f_i=s^{-i},\qquad \sigma_i=\sigma_0 s^{\,i}$$

Chuỗi ảnh làm mượt lũy tiến, với $A_0=I$:

$$A_{i+1}=G_{\sigma^{\text{inc}}_i}*A_i=G_{\sigma_i}*I,\qquad \sigma^{\text{inc}}_i=\sqrt{\max(10^{-6},\sigma_i^2-\sigma_{i-1}^2)}$$

Năng lượng băng tần (DoG, gộp kênh bằng chuẩn L2):

$$r_i=\sqrt{\sum_{c}\big(A_i-A_{i+1}\big)_c^2},\qquad r_i\leftarrow G_{2\sigma_i}*r_i\ \ (i>0)$$

Trọng số:

$$w_i=r_i^{\,p}$$

---

## 2. Phần khác: tensor tính trực tiếp trên ảnh đã blur

Định nghĩa toán tử structure tensor (chính là hàm `get_structure_tensor_torch`):

$$\mathcal T(X;\sigma,\rho):\quad
\tilde X=G_\sigma*X,\quad
X_{x,c}=K_x\star\tilde X_c,\quad X_{y,c}=K_y\star\tilde X_c$$

$$J_{xx}=\sum_c X_{x,c}^2,\quad J_{xy}=\sum_c X_{x,c}X_{y,c},\quad J_{yy}=\sum_c X_{y,c}^2$$

$$S_\bullet=G_\rho*J_\bullet,\qquad \hat S_\bullet=\frac{S_\bullet}{\max_{x,y}\operatorname{tr}(S)+10^{-6}}$$

Ở level $i$, v2 gọi:

$$\hat S^{\,i}=\mathcal T\!\left(A_{i+1};\ \sigma=\sigma_i,\ \rho=3\sigma_i\right)$$

**Trong v1** thì khác: $\hat S^{\,i}_{v1}=\dfrac{G_{3\sigma_i}*\hat S^{0}}{\dots}$, trong đó $\hat S^0$ luôn tính từ gradient ở scale $\sigma_0$.

---

## 3. Scale hiệu dụng thực sự là $\sqrt2\,\sigma_i$

$A_{i+1}$ đã được blur bằng $\sigma_i$, rồi bên trong $\mathcal T$ lại blur thêm $\sigma_i$ nữa. Vì phương sai Gaussian cộng dồn:

$$\tilde A_{i+1}=G_{\sigma_i}*G_{\sigma_i}*I=G_{\sqrt2\,\sigma_i}*I$$

Vậy gradient thực tế là:

$$X_{x,c}=K_x\star\big(G_{\sqrt2\sigma_i}*I_c\big),\qquad X_{y,c}=K_y\star\big(G_{\sqrt2\sigma_i}*I_c\big)$$

Nếu chỉ muốn scale $\sigma_i$ thì phải truyền `sigma` rất nhỏ (hoặc bỏ bước pre-smooth) khi gọi trên `next_smooth`. Còn nếu muốn scale $\sigma_i$ mà dùng `sigma=scale_sigma` thì phải đưa vào `image_tensor` gốc thay vì `next_smooth`.

---

## 4. Chuẩn hoá về hướng thuần

$$N^{i}_\bullet=\frac{\hat S^{\,i}_\bullet}{\hat S^{\,i}_{xx}+\hat S^{\,i}_{yy}+10^{-6}}$$

Vì $\hat S^i$ đã chia $\max\operatorname{tr}$ ở bước 2, phép chia trace theo pixel ở đây gần như triệt tiêu hệ số đó. Chỉ còn khác nhau ở $\varepsilon$. Cụ thể: vùng có trace nhỏ hơn $10^{-6}$ (sau khi chia max) sẽ bị co về $0$, và mức độ co phụ thuộc vào $\max\operatorname{tr}$ của từng level.

---

## 5. Cộng dồn và kết quả (y hệt v1)

$$\boxed{S^{\text{final}}=\frac{\sum_{i=0}^{L-1}w_i\,f_i^{2}\,N^{i}}{\sum_{i=0}^{L-1}w_i+10^{-6}}=\sum_i\omega_i f_i^2N^i}$$

$$\operatorname{tr}(S^{\text{final}})\approx f_{\text{eff}}^2=\sum_i\omega_i f_i^2$$

Hướng và độ đồng nhất lấy như trước:

$$\theta=\tfrac12\operatorname{atan2}(2S_{xy},S_{xx}-S_{yy}),\qquad \mathcal C=\left(\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}\right)^2$$

---

## 6. Tại sao v2 khác v1 về bản chất

Structure tensor có **phép bình phương** ($\nabla I\nabla I^\top$), nên blur và bình phương **không giao hoán**:

$$G_\rho*\big[(\nabla G_{\sigma_0}I)(\nabla G_{\sigma_0}I)^\top\big]\ \neq\ (\nabla G_{\sigma}I)(\nabla G_{\sigma}I)^\top\ \text{blur bởi}\ G_\rho$$

| | v1 | v2 |
|---|---|---|
| Scale gradient | cố định $\sigma_0$ | tăng theo level, hiệu dụng $\sqrt2\sigma_i$ |
| Scale tích phân | $\rho_i=3\sigma_i$ | $\rho_i=3\sigma_i$ |
| Nhiễu / texture mịn ở level thô | vẫn nằm trong $\hat S^0$, chỉ bị làm trung bình | bị xoá **trước** khi bình phương |
| Đường cong, cấu trúc lớn | thấy qua cửa sổ rộng | thấy qua cả gradient thô lẫn cửa sổ rộng |
| Số lần gọi structure tensor | 1 | $L$ |

Nói cách khác, v1 trả lời "hướng của gradient mịn, lấy trung bình trên vùng lớn". v2 trả lời "hướng của gradient ở đúng scale đó, lấy trung bình trên vùng lớn". Với texture nhiễu thì v2 ổn định hơn nhiều, vì các gradient mịn triệt nhau trong $J$ của v1 nhưng vẫn để lại nhiễu hướng.

---

## Điểm cần lưu ý

1. **Blur hai lần** như mục 3, scale hiệu dụng là $\sqrt2\sigma_i$ chứ không phải $\sigma_i$.
2. **Lệch band:** trọng số $r_i$ đo băng $[\sigma_{i-1},\sigma_i]$, nhưng tensor lại đo ở $\sqrt2\sigma_i$, hơi lệch về phía thô hơn so với band.
3. **Vùng phẳng** vẫn về $0$ như v1, vì $\sum w_i\approx0$ và $\varepsilon$ lấn át.
4. `aggregation_mode` vẫn không được dùng, `smoothing_factor` vẫn chỉ là công tắc.
5. **Chi phí:** mỗi level gọi thêm 1 lần structure tensor (pre-blur, 2 conv Sobel, 3 blur $\rho_i$ với kernel $\approx24\sigma_i+1$), nên nặng hơn v1 rõ rệt ở level lớn.
6. **Zero-padding** của Sobel (đã nói ở câu đầu) bị lặp lại ở mỗi level nên lỗi viền cũng được cộng dồn qua các scale.