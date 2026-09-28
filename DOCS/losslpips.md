# Toàn bộ công thức toán trong "Drop-In Perceptual Optimization for 3DGS"

Bài báo có ít công thức, vì đóng góp chính là đổi hàm loss. Mình xếp theo thứ tự xuất hiện. Phần **⚙** là do mình bổ sung (không có trong bài) để bạn hiểu, và tách riêng ở cuối. Bản văn bản bạn gửi bị lặp và lộn xộn ở vài chỗ, nên mình lấy các phiên bản cuối cùng của công thức.

## 1. Bài toán tối ưu (mục 2.2)

**Biểu diễn 3DGS (Eq. 1):**

$$\min_{\mathcal G}\ \gamma\, D(x,\hat x)$$

- 𝒢: toàn bộ tham số Gaussian (μᵢ, Σᵢ, cᵢ, αᵢ).
- D(·): độ méo (distortion) tính trên ảnh 2D render.
- x̂: ảnh render, x: ảnh GT, γ: hệ số vô hướng chung.
- Bản nháp cũ viết tổng trên ảnh huấn luyện $\sum_{x\in D_{\text{train}}}$ và dùng ε thay γ.

**Nén cảnh (Eq. 2, rate-distortion):**

$$\min_{\mathcal G,\theta}\ D_\theta(x,\hat x)+\lambda\,R_\theta(\mathcal G),\qquad R_\theta(\mathcal G)=\mathbb E\big[-\log_2 p_{\hat y}(\hat y)\big]$$

- R_θ: chi phí lưu trữ, là cross-entropy Shannon giữa mã ẩn lượng tử ŷ của 𝒢 và mô hình entropy p_ŷ.
- λ: cân bằng chất lượng và dung lượng, quét trong [3⁻², 3⁻⁵].

## 2. Ba họ loss (mục 3.1)

**(i) Loss gốc 3DGS:** tổ hợp tuyến tính của L1 và SSIM, ký hiệu $\mathcal L_{\text{orig}}$. Bài không viết trọng số tường minh trong văn bản.

**(ii) Loss tổng hợp (Eq. 3):**

$$\mathcal L_{\text{composite}}=\omega_1\mathcal L_{L1}+\omega_2\mathcal L_{L2}+\omega_3\mathcal L_{\text{MS-SSIM}}+\omega_4\mathcal L_{\text{LPIPS}}$$

Trọng số chọn: (ω₁, ω₂, ω₃, ω₄) = (0.05, 0.30, 0.60, 0.10). LPIPS dùng mạng AlexNet khi train, VGG khi đánh giá.

**(iii) Wasserstein Distortion (Eq. 4).** Tại mỗi feature map và vị trí:

$$d_{\text{WD}}=\sqrt{(\mu-\hat\mu)^2+(\nu-\hat\nu)^2}$$

- μ, ν: trung bình và độ lệch chuẩn cục bộ của đặc trưng VGG trên ảnh GT.
- μ̂, ν̂: cùng đại lượng trên ảnh render.
- Sau đó cộng gộp qua các feature map và vị trí pixel.
- Kernel gộp có kích thước σ, chọn σ = 4 cố định. σ → 0 thì WD trở thành khoảng cách theo điểm (pointwise).

Trọng số mỗi mức pyramid (Phụ lục A.1):

$$w_i=\mathrm{ReLU}\big(1-|\log_2\sigma-i|\big)$$

Với σ = 4, log₂σ = 2. Theo bài, chỉ 20 trong 96 mức pyramid của toàn bộ chồng VGG có trọng số khác 0.

**WD chỉ:**

$$\mathcal L_{\text{WD}}=\gamma\, d_{\text{WD}}$$

**WD-R (bản chuẩn hoá, loss chính của bài):**

$$\mathcal L_{\text{WD-R}}=\gamma\big(d_{\text{WD}}+\beta\,\mathcal L_{\text{orig}}\big),\qquad \beta=\frac{1}{0.09}\approx 11.1$$

Dù β trông lớn, WD vẫn chiếm ưu thế: tỉ số gradient trên cảnh Bicycle có trung bình khoảng 1.6:

$$\frac{\|\nabla d_{\text{WD}}\|}{\|\nabla(\beta\mathcal L_{\text{orig}})\|}\approx 1.6$$

## 3. Phân tích hình dạng Gaussian: effective rank (mục 4.1)

Với Gaussian G_k có hiệp phương sai Σ_k, các giá trị suy biến s₁² ≥ s₂² ≥ s₃² > 0:

$$\mathrm{erank}(G_k)=\exp\Big\{-\sum_{i=1}^{3}q_i\log q_i\Big\},\qquad q_i=\frac{s_i^2}{\sum_{j=1}^{3}s_j^2}$$

- erank ≈ 1: Gaussian dạng kim (rất dị hướng).
- erank ≈ 3: Gaussian tròn (đẳng hướng).
- Kết quả: trung vị erank cảnh Barcelona là 1.55 (loss gốc), 1.36 (composite), 1.12 (WD-R), 1.11 (WD).

## 4. Heatmap mật độ splat (Phụ lục H)

Với mỗi Gaussian nhìn thấy trong một view, chiếu tâm μᵢ lên ảnh và cộng độ mờ αᵢ vào ô 2×2 pixel chứa tâm đó. Đây là mô tả bằng lời, bài không viết thành công thức.

## 5. Các siêu tham số (bảng tổng hợp)

| Đại lượng | Giá trị |
|---|---|
| Số iteration | 30 000 |
| Warm-up bằng loss gốc | 3k iter (5k với BungeeNeRF) |
| σ của WD | 4 |
| β | 1/0.09 |
| γ (WD / WD-R) Deep Blending | 0.028 / 0.025 |
| γ Mip-NeRF 360 trong nhà | 0.029 / 0.025 |
| γ Mip-NeRF 360 ngoài trời | 0.035 / 0.028 |
| γ Tanks & Temples | 0.038 / 0.032 |
| γ BungeeNeRF | 0.030 / 0.025 |
| Thời gian mỗi iteration (Bicycle, 4.79M Gaussian) | 61.1 ms (gốc) so với 273.9 ms (WD), tức khoảng 4.5 lần |

## ⚙ Phần mình bổ sung (không có trong bài)

**Quan hệ Elo và tỉ lệ ưa thích.** Bài nêu kết quả mà không viết công thức. Các con số của họ khớp với:

$$\frac{P(A)}{P(B)}=10^{\Delta\text{Elo}/400}$$

Kiểm tra: ΔElo = 150 cho 10^0.375 ≈ 2.4 (bài ghi hơn 2.3 lần), 72 cho 1.5, 105.7 cho 1.8, 223.2 cho 3.6, đều khớp các số trong bài.

**Vì sao γ điều khiển số Gaussian.** Densification dựa trên độ lớn gradient vị trí, mà gradient của loss nhân hệ số tỉ lệ thuận:

$$\nabla(\gamma\mathcal L)=\gamma\,\nabla\mathcal L$$

Nên γ lớn thì gradient lớn, vượt ngưỡng nhiều hơn, sinh nhiều Gaussian hơn. Đó là lý do bài tinh chỉnh γ từng bộ dữ liệu để cân số splat.

**Các metric bài dùng nhưng không viết công thức:**
- **FID:** $\|\mu_1-\mu_2\|^2+\mathrm{Tr}\big(\Sigma_1+\Sigma_2-2(\Sigma_1\Sigma_2)^{1/2}\big)$ trên đặc trưng Inception-v3.
- **CMMD:** MMD giữa hai phân bố trong không gian đặc trưng CLIP (ViT-L/14-336).
- **LPIPS, DISTS:** khoảng cách đặc trưng sâu tính theo từng ảnh rồi lấy trung bình.

## Ví dụ số cho WD-R

Giả sử tại một vị trí của một feature map: μ = 0.50, ν = 0.20, μ̂ = 0.45, ν̂ = 0.25.

$$d_{\text{WD}}=\sqrt{0.05^2+0.05^2}=\sqrt{0.005}\approx 0.0707$$

Giả sử sau khi gộp toàn ảnh d_WD = 0.0707 và L_orig = 0.05, với γ = 0.025, β = 11.1:

$$\mathcal L_{\text{WD-R}}=0.025\,(0.0707+11.1\times0.05)=0.025\times0.6257\approx 0.0156$$

Ở ví dụ này phần WD (0.0707) nhỏ hơn phần βL_orig (0.556), trái với tỉ số gradient khoảng 1.6 mà bài đo. Lý do là độ lớn loss khác với độ lớn gradient, và các số giả định của mình không phản ánh giá trị thật. Nó chỉ minh hoạ cách ráp công thức.

Nếu bạn muốn, mình viết code PyTorch cho $\mathcal L_{\text{WD-R}}$ theo đúng các công thức trên (gồm cache đặc trưng VGG của ảnh GT).