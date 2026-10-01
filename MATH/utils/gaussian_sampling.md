# Công thức toán học trong `utils/gaussian_sampling.py`

File chỉ có một hàm: `sample_anisotropic_gaussians_2d`, dùng để khởi tạo Gaussian 2D dị hướng từ cấu trúc tensor của một ảnh (ứng dụng fitting ảnh 2D độc lập, **không phải** lưới offset dùng trong `densify_and_split_structgs` của `gaussian_model.py` — hai logic khác nhau, xem mục 5 để đối chiếu).

---

## 1. Cấu trúc tensor của ảnh (`get_structure_tensor_torch`, tái dùng từ `loss_utils.py`)

Hàm được gọi nguyên văn tại `utils/gaussian_sampling.py`, dòng 35 (công thức nội bộ của `get_structure_tensor_torch` nằm trong `utils/loss_utils.py`, không thuộc file này):

```python
st_map = get_structure_tensor_torch(image_tensor, sigma=sigma, rho=rho)
```

và tách 3 kênh (dòng 38–40):

```python
Sxx_torch = st_map[0, 0, :, :]
Sxy_torch = st_map[0, 1, :, :]
Syy_torch = st_map[0, 2, :, :]
```

Với ảnh $I\in\mathbb{R}^{B\times C\times H\times W}$, sau làm mờ Gauss ($\sigma$) và đạo hàm Sobel $(I_x,I_y)$, cấu trúc tensor Di Zenzo đa kênh (tổng năng lượng trên các kênh màu) và tích chập cửa sổ Gauss $\rho$:

$$
S = G_\rho * \begin{pmatrix} \sum_c I_{x,c}^2 & \sum_c I_{x,c} I_{y,c} \\ \sum_c I_{x,c} I_{y,c} & \sum_c I_{y,c}^2 \end{pmatrix}
= \begin{pmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{pmatrix}
$$

(trả về bản đồ `st_map` $\in\mathbb{R}^{B\times3\times H\times W}$, 3 kênh là $S_{xx},S_{xy},S_{yy}$).

---

## 2. Bản đồ năng lượng và lấy mẫu vị trí theo tầm quan trọng

Năng lượng cục bộ = vết (trace) của cấu trúc tensor.

Trích nguyên văn (`utils/gaussian_sampling.py`, dòng 43):

```python
energy_map_torch = Sxx_torch + Syy_torch
```

$$
E(x,y) = S_{xx}(x,y) + S_{yy}(x,y)
\tag{dòng 43}
$$

Xác suất lấy mẫu từng pixel tỉ lệ thuận với năng lượng (chuẩn hoá thành phân phối rời rạc).

Trích nguyên văn (dòng 48–53):

```python
energy_flat = energy_map.flatten()
sum_energy = energy_flat.sum()
if sum_energy > 0:
    prob = energy_flat / sum_energy
else:
    prob = np.ones_like(energy_flat) / len(energy_flat)
```

$$
p(x,y) = \frac{E(x,y)}{\sum_{(x',y')} E(x',y')}
\tag{dòng 48–51}
$$

(nếu tổng năng lượng bằng 0 — ảnh phẳng tuyệt đối — dùng phân phối đều $p(x,y)=1/(HW)$ — dòng 53.)

Trích nguyên văn phần lấy mẫu (dòng 56–58):

```python
rng = np.random.default_rng(seed)
coords_idx = rng.choice(H * W, size=num_samples, p=prob, replace=False)
y_coords, x_coords = np.unravel_index(coords_idx, (H, W))
```

$N_{samples}$ toạ độ $(x_i,y_i)$ được lấy mẫu **không hoàn lại** (without replacement) theo $p(x,y)$ bằng `numpy.random.Generator.choice` (dòng 57).

---

## 3. Chuẩn hoá số (regularization) cấu trúc tensor tại điểm lấy mẫu

Giá trị $S_{xx},S_{xy},S_{yy}$ tại từng điểm được khử NaN/Inf rồi cộng một hệ số nhỏ vào đường chéo để đảm bảo xác định dương (positive semi-definite).

Trích nguyên văn (dòng 71–78):

```python
sxx = torch.nan_to_num(sxx, nan=0.0, posinf=1e6, neginf=-1e6)
sxy = torch.nan_to_num(sxy, nan=0.0, posinf=1e6, neginf=-1e6)
syy = torch.nan_to_num(syy, nan=0.0, posinf=1e6, neginf=-1e6)

# Ensure positive semi-definite by adding small epsilon to diagonal
epsilon = 1e-6
sxx = sxx + epsilon
syy = syy + epsilon
```

$$
\epsilon = 10^{-6}, \qquad S_{xx} \leftarrow S_{xx}+\epsilon, \qquad S_{yy}\leftarrow S_{yy}+\epsilon
\tag{dòng 76–78}
$$

(Lưu ý bổ sung so với bản cũ: trước khi cộng $\epsilon$, cả ba thành phần còn bị *clamp* về $[-10^6,10^6]$ qua `torch.nan_to_num(..., posinf=1e6, neginf=-1e6)` — dòng 71–73 — không chỉ khử NaN/Inf đơn thuần như câu chữ "khử NaN/Inf" gợi ý.)

Ma trận hiệp phương sai cục bộ (dùng làm "structure tensor" tại điểm mẫu).

Trích nguyên văn (dòng 82–85):

```python
S = torch.stack([
    torch.stack([sxx, sxy], dim=-1),
    torch.stack([sxy, syy], dim=-1)
], dim=-2)
```

$$
S_i = \begin{pmatrix} S_{xx,i} & S_{xy,i} \\ S_{xy,i} & S_{yy,i}\end{pmatrix}
\tag{dòng 82–85}
$$

---

## 4. Phân rã trị riêng và hướng dị hướng (anisotropy)

Phân rã trị riêng đối xứng (`torch.linalg.eigh`) cho từng điểm.

Trích nguyên văn (dòng 89–95, kèm nhánh fallback):

```python
try:
    eigvals, eigvecs = torch.linalg.eigh(S)
except RuntimeError as e:
    print(f"Warning: Eigendecomposition failed, using fallback. Error: {e}")
    # Fallback: use identity matrices
    eigvals = torch.ones((S.shape[0], 2), device=device)
    eigvecs = torch.eye(2, device=device).unsqueeze(0).expand(S.shape[0], -1, -1)
```

$$
S_i = V_i \Lambda_i V_i^\top, \qquad \Lambda_i = \mathrm{diag}(\lambda_1^{raw}, \lambda_2^{raw})
\tag{dòng 90}
$$

Sắp xếp giảm dần để có trị riêng lớn nhất $\lambda_1$ (hướng biến thiên gradient mạnh nhất — "hướng pháp tuyến cạnh") và nhỏ nhất $\lambda_2$ (hướng dọc theo cạnh — "hướng edge").

Trích nguyên văn (dòng 99, 103–104):

```python
sorted_indices = torch.argsort(eigvals, dim=-1, descending=True)
```

```python
l1 = torch.gather(eigvals, -1, sorted_indices[:, 0:1]).squeeze(-1)
l2 = torch.gather(eigvals, -1, sorted_indices[:, 1:2]).squeeze(-1)
```

$$
\lambda_1 = \max(\lambda_1^{raw},\lambda_2^{raw}), \qquad \lambda_2 = \min(\lambda_1^{raw},\lambda_2^{raw})
\tag{dòng 99, 103–104}
$$

Vector riêng ứng với $\lambda_2$ (hướng cạnh $v_2$) được trích ra để xác định **hướng dài (major axis)** mà Gaussian nên hướng theo (dọc theo cạnh, không cắt ngang cạnh).

Trích nguyên văn (dòng 108–109, 113):

```python
v2_indices = sorted_indices[:, 1:2].unsqueeze(-1).expand(-1, -1, 2)  # (num_samples, 1, 2)
v2 = torch.gather(eigvecs, 1, v2_indices).squeeze(1)  # (num_samples, 2)
```

```python
angles = torch.rad2deg(torch.atan2(v2[:, 1], v2[:, 0]))
```

$$
\theta_i = \mathrm{atan2}(v_{2,i,y},\, v_{2,i,x}) \quad(\text{radian}\to\text{độ, qua } \texttt{torch.rad2deg})
\tag{dòng 108–109, 113}
$$

Nếu phân rã trị riêng thất bại (lỗi số), dùng fallback đẳng hướng: $\lambda_1=\lambda_2=1$, $V=I_2$ (ma trận đơn vị) — dòng 94–95.

---

## 5. Đối chiếu với `densify_and_split_structgs` (`gaussian_model.py`, mục 6.3)

Hai hàm có logic **khác nhau về bản chất**, không trùng công thức:

- `gaussian_sampling.py`: lấy mẫu **vị trí pixel rời rạc** trong một ảnh 2D theo xác suất tỉ lệ năng lượng gradient (importance sampling theo phân phối rời rạc trên lưới pixel), sau đó gán cho mỗi điểm được chọn một **hướng dị hướng** (từ trị riêng/vector riêng của structure tensor cục bộ) — dùng để **khởi tạo** tham số ban đầu (vị trí + góc xoay) của các Gaussian 2D mới trong bài toán fit ảnh.
- `densify_and_split_structgs` trong `gaussian_model.py`: xuất phát từ một Gaussian 3D **đã tồn tại**, sinh lưới con đều $(k_x,k_y,k_z)$ quanh tâm cha theo chỉ số $\eta$ (Nyquist), dùng khoảng cách $\Delta=\sigma^{child}\sqrt{12}$ (từ phương sai phân phối đều) rồi xoay bằng $R_i$ của chính Gaussian cha — không lấy mẫu xác suất rời rạc, không dùng trị riêng/vector riêng của structure tensor.

Điểm chung duy nhất: cả hai đều dùng **cấu trúc tensor ảnh** ($S_{xx},S_{xy},S_{yy}$ từ `get_structure_tensor_torch`) làm tín hiệu dẫn hướng — nhưng `gaussian_sampling.py` dùng nó để **lấy mẫu vị trí + hướng ban đầu**, còn pipeline StructGS trong `gaussian_model.py`/`freq_utils.py` dùng nó để tính **$\eta$** (tỉ số kích thước/bước sóng, mục 5 của `freq_utils.md`) nhằm quyết định **số lượng** điểm con cần tách, không quyết định hướng tách (hướng tách luôn theo $R_i$ sẵn có của Gaussian).

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `energy_map_torch = Sxx_torch + Syy_torch` | $E(x,y) = S_{xx}+S_{yy}$ |
| `prob = energy_flat / sum_energy` | $p(x,y) = E(x,y)/\sum E$ |
| `prob = np.ones_like(...) / len(...)` | $p(x,y)=1/(HW)$ (fallback khi $\sum E=0$) |
| `rng.choice(H*W, size=num_samples, p=prob, replace=False)` | lấy mẫu $N_{samples}$ toạ độ theo $p(x,y)$, không hoàn lại |
| `sxx = sxx + epsilon` (epsilon=1e-6) | $S_{xx}\leftarrow S_{xx}+\epsilon$ |
| `S = [[sxx, sxy],[sxy, syy]]` | $S_i=\begin{pmatrix}S_{xx,i}&S_{xy,i}\\S_{xy,i}&S_{yy,i}\end{pmatrix}$ |
| `eigvals, eigvecs = torch.linalg.eigh(S)` | $S_i = V_i\Lambda_i V_i^\top$ |
| `sorted_indices = torch.argsort(eigvals, descending=True)` | sắp xếp $\lambda_1\ge\lambda_2$ |
| `l1 = ...sorted_indices[:,0:1]` | $\lambda_1=\max(\lambda_1^{raw},\lambda_2^{raw})$ |
| `l2 = ...sorted_indices[:,1:2]` | $\lambda_2=\min(\lambda_1^{raw},\lambda_2^{raw})$ |
| `v2 = torch.gather(eigvecs, 1, v2_indices)` | vector riêng $v_2$ ứng với $\lambda_2$ |
| `angles = torch.rad2deg(torch.atan2(v2[:,1], v2[:,0]))` | $\theta_i = \mathrm{atan2}(v_{2,y}, v_{2,x})$ (độ) |
| `eigvals = torch.ones((S.shape[0], 2))` (fallback) | $\lambda_1=\lambda_2=1$ khi `eigh` lỗi |
