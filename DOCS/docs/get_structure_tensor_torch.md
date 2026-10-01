# Công thức toán của hàm `get_structure_tensor_torch`

Ký hiệu: ảnh vào là $I_c(x,y)$ với $c \in \{R,G,B\}$ (tức $C$ kênh). Dấu $*$ là tích chập, $\star$ là cross-correlation (`F.conv2d` của PyTorch thực chất là cross-correlation).

---

## 1. Làm mượt trước (pre-smoothing) bằng Gaussian

Kernel Gaussian 2D:

$$G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}\exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

Kích thước kernel: $k = \lfloor 8\sigma+1 \rfloor$, cộng thêm 1 nếu chẵn (bán kính $\approx 4\sigma$, phủ gần hết năng lượng Gaussian).

Áp dụng độc lập cho từng kênh:

$$\tilde I_c = G_\sigma * I_c$$

---

## 2. Đạo hàm bằng Sobel

$$K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\qquad K_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$$

Mỗi kênh có đạo hàm riêng (`groups=C` nghĩa là kênh nào tính kênh đó, không trộn):

$$I_{x,c}=K_x \star \tilde I_c,\qquad I_{y,c}=K_y \star \tilde I_c$$

Gradient của kênh $c$: $\nabla I_c=(I_{x,c},\,I_{y,c})^\top$.

> Lưu ý: Sobel chưa chia 8 nên giá trị lớn gấp khoảng 8 lần đạo hàm thật. Bước normalize ở cuối sẽ triệt tiêu hệ số này.

---

## 3. Các tích theo từng kênh

$$I_{xx,c}=I_{x,c}^2,\qquad I_{xy,c}=I_{x,c}\,I_{y,c},\qquad I_{yy,c}=I_{y,c}^2$$

Đây chính là ma trận $\nabla I_c\,\nabla I_c^\top$ của từng kênh:

$$\nabla I_c\nabla I_c^\top=\begin{bmatrix}I_{xx,c}&I_{xy,c}\\ I_{xy,c}&I_{yy,c}\end{bmatrix}$$

---

## 4. Cộng các kênh (Di Zenzo)

$$J_{xx}=\sum_{c=1}^{C} I_{x,c}^2,\qquad J_{xy}=\sum_{c=1}^{C} I_{x,c}I_{y,c},\qquad J_{yy}=\sum_{c=1}^{C} I_{y,c}^2$$

Viết gọn dạng ma trận:

$$J=\sum_{c}\nabla I_c\nabla I_c^\top=\begin{bmatrix}J_{xx}&J_{xy}\\J_{xy}&J_{yy}\end{bmatrix}$$

Vì sao phải cộng **tích** chứ không cộng gradient trước: gradient của các kênh có thể ngược dấu và triệt tiêu nhau. Cộng $\nabla I_c\nabla I_c^\top$ (bán xác định dương) thì không bao giờ triệt tiêu, nên bắt được cả biên iso-luminant (đổi màu nhưng độ sáng không đổi).

---

## 5. Tích phân trong cửa sổ (window integration)

Làm mượt từng phần tử của $J$ bằng Gaussian scale $\rho$ (kernel size $\lfloor 8\rho+1\rfloor$, lẻ):

$$S_{xx}=G_\rho * J_{xx},\qquad S_{xy}=G_\rho * J_{xy},\qquad S_{yy}=G_\rho * J_{yy}$$

$$S=\begin{bmatrix}S_{xx}&S_{xy}\\S_{xy}&S_{yy}\end{bmatrix}$$

Bước này làm tensor phản ánh hướng chủ đạo của **vùng lân cận**, không chỉ một pixel. Không có nó thì $J$ luôn có hạng 1 (chỉ 1 hướng) tại mỗi pixel.

---

## 6. Chuẩn hoá

Vết (trace) của $S$ chính là tổng năng lượng gradient:

$$M(x,y)=S_{xx}+S_{yy}=\operatorname{tr}(S)$$

Với mỗi ảnh $b$ trong batch:

$$m_b=\max_{x,y}M_b(x,y)+10^{-6}$$

$$\hat S_{xx}=\frac{S_{xx}}{m_b},\qquad \hat S_{xy}=\frac{S_{xy}}{m_b},\qquad \hat S_{yy}=\frac{S_{yy}}{m_b}$$

Sau bước này $\max \operatorname{tr}(\hat S)\approx 1$. Vì chia cùng một hằng số nên **hướng và độ đồng nhất (coherence) không đổi**, chỉ có độ lớn được đưa về thang $[0,1]$.

**Output:** tensor $(B,3,H,W)$ gồm $[\hat S_{xx},\hat S_{xy},\hat S_{yy}]$.

---

## Phần mở rộng (code chưa tính, nhưng thường dùng ở bước sau)

Trị riêng của $S$:

$$\lambda_{1,2}=\frac{S_{xx}+S_{yy}\pm\sqrt{(S_{xx}-S_{yy})^2+4S_{xy}^2}}{2}$$

Hướng gradient chủ đạo:

$$\theta=\frac12\operatorname{atan2}\!\left(2S_{xy},\;S_{xx}-S_{yy}\right)$$

Độ đồng nhất (coherence):

$$\mathcal C=\left(\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}\right)^2\in[0,1]$$

- $\lambda_1\approx\lambda_2\approx0$: vùng phẳng
- $\lambda_1\gg\lambda_2$: biên, $\mathcal C\to1$
- $\lambda_1\approx\lambda_2\gg0$: góc / vùng texture nhiễu

---

## Tóm tắt luồng

$$I_c \xrightarrow{G_\sigma} \tilde I_c \xrightarrow{\text{Sobel}} (I_{x,c},I_{y,c}) \xrightarrow{\text{bình phương, nhân}} \xrightarrow{\sum_c} J \xrightarrow{G_\rho} S \xrightarrow{/\max\operatorname{tr}} \hat S$$

Một điểm cần để ý trong code: `padding=1` là zero-padding nên viền ảnh sẽ có gradient giả. Nếu quan trọng thì nên dùng `padding_mode='reflect'` (pad thủ công bằng `F.pad`) trước khi conv.