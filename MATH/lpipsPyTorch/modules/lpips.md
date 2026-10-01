# Công thức toán học trong `lpipsPyTorch/modules/lpips.py`

Tài liệu mô tả chính xác công thức LPIPS (Learned Perceptual Image Patch Similarity) **như được cài đặt trong code**, bám sát từng dòng của `forward`.

---

## 1. Khởi tạo (`__init__`)

- `self.net = get_network(net_type)`: mạng backbone trích đặc trưng (AlexNet/SqueezeNet/VGG16), xem `networks.md`. Mạng này đã bao gồm bước chuẩn hoá đơn vị theo kênh (unit-normalize) ở từng layer đích — xem mục 2 của `networks.md`.
- `self.lin = LinLayers(self.net.n_channels_list)`: với mỗi layer đích $l$, một lớp tuyến tính không có bias, hiện thực bằng tích chập $1\times1$. Trích nguyên văn (`lpips.py`, dòng 24–28):

```python
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))
```

$$
\mathrm{conv}_l(\cdot) : \mathbb{R}^{C_l} \to \mathbb{R}, \qquad \mathrm{conv}_l(z)_{hw} = \sum_{c=1}^{C_l} w_{l,c}\, z_{c,hw}
$$

trong đó $w_{l,c} \in \mathbb{R}$ là trọng số đã huấn luyện sẵn (nạp từ `get_state_dict(net_type, version)`, không nhân thêm bias).

---

## 2. Lan truyền xuôi (`forward`)

### 2.1. Trích đặc trưng đã chuẩn hoá

Trích nguyên văn (`lpips.py`, dòng 30–31):

```python
def forward(self, x: torch.Tensor, y: torch.Tensor):
    feat_x, feat_y = self.net(x), self.net(y)
```

$$
\hat{y}^{(l)} = \mathrm{net}(x)^{(l)}, \qquad \hat{y}_0^{(l)} = \mathrm{net}(x_0)^{(l)}, \qquad l = 1,\dots,L
$$

(mỗi $\hat{y}^{(l)}$ đã được chuẩn hoá đơn vị theo kênh tại từng vị trí không gian — xem `normalize_activation` trong `utils.md`.)

### 2.2. Bình phương sai khác theo từng kênh, từng vị trí không gian

Trích nguyên văn (`lpips.py`, dòng 33):

```python
diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
```

$$
D^{(l)}_{c,hw} = \big(\hat{y}^{(l)}_{c,hw} - \hat{y}^{(l)}_{0,c,hw}\big)^2, \qquad c = 1,\dots,C_l
$$

### 2.3. Trọng số hoá theo kênh + trung bình không gian (`l(d).mean((2,3), True)`)

Trích nguyên văn (`lpips.py`, dòng 34):

```python
res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
```

Trọng số $w_{l,c}$ được áp dụng **trực tiếp lên $D^{(l)}$ đã bình phương** (tích chập $1\times1$, không mixing không gian — xem `conv_l` định nghĩa ở mục 1, nạp qua `self.lin.load_state_dict(...)`, `lpips.py` dòng 27–28), sau đó lấy trung bình trên chiều không gian $H_l\times W_l$ (giữ chiều, `keepdim=True`):

$$
\mathrm{res}^{(l)} = \frac{1}{H_l W_l}\sum_{h=1}^{H_l}\sum_{w=1}^{W_l}\ \sum_{c=1}^{C_l} w_{l,c}\, D^{(l)}_{c,hw}
$$

> **Lưu ý bám sát code:** công thức lý thuyết chuẩn của LPIPS viết là $\lVert w_l \odot (\hat{y}^l_{hw}-\hat{y}^l_{0,hw})\rVert_2^2 = \sum_c w_{l,c}^2 D^{(l)}_{c,hw}$ (trọng số bị bình phương do nằm trong chuẩn $\ell_2$). Tuy nhiên trong cài đặt thực tế ở đây, `nn.Conv2d(nc, 1, 1, 1, 0, bias=False)` nhân **trực tiếp** $w_{l,c}$ (không bình phương) vào $D^{(l)}_{c,hw}$ đã được bình phương sẵn từ bước 2.2. Về mặt đại số hai cách viết tương đương nhau nếu ta coi $w_{l,c}$ trong công thức code đã "gộp" vai trò bình phương của trọng số gốc; tài liệu này mô tả đúng phép toán mà code thực hiện, không phải công thức lý thuyết gốc có bình phương tường minh trên $w_l$.

### 2.4. Cộng dồn qua tất cả các layer (`torch.sum(torch.cat(res, 0), 0, True)`)

Trích nguyên văn (`lpips.py`, dòng 36):

```python
return torch.sum(torch.cat(res, 0), 0, True)
```

Các $\mathrm{res}^{(l)}$ (mỗi cái có shape $(B,1,1,1)$) được nối theo chiều 0 rồi cộng lại theo đúng chiều đó, giữ chiều (`keepdim=True`):

$$
d(x,x_0) = \sum_{l=1}^{L} \mathrm{res}^{(l)} = \sum_{l=1}^{L} \frac{1}{H_l W_l}\sum_{h,w}\sum_c w_{l,c}\,\big(\hat{y}^{(l)}_{c,hw}-\hat{y}^{(l)}_{0,c,hw}\big)^2
$$

Đây chính là giá trị LPIPS cuối cùng trả về bởi `forward`.

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `self.net = get_network(net_type)` | định nghĩa $\mathrm{net}(\cdot)^{(l)}$, $l=1,\dots,L$ |
| `self.lin = LinLayers(...)` | định nghĩa $\mathrm{conv}_l(z)_{hw} = \sum_c w_{l,c} z_{c,hw}$ |
| `feat_x, feat_y = self.net(x), self.net(y)` | $\hat{y}^{(l)} = \mathrm{net}(x)^{(l)},\ \hat{y}_0^{(l)} = \mathrm{net}(x_0)^{(l)}$ |
| `diff = [(fx - fy) ** 2 ...]` | $D^{(l)}_{c,hw} = (\hat{y}^{(l)}_{c,hw}-\hat{y}^{(l)}_{0,c,hw})^2$ |
| `res = [l(d).mean((2,3), True) for d, l in zip(diff, self.lin)]` | $\mathrm{res}^{(l)} = \frac{1}{H_lW_l}\sum_{h,w}\sum_c w_{l,c} D^{(l)}_{c,hw}$ |
| `torch.sum(torch.cat(res, 0), 0, True)` | $d(x,x_0) = \sum_{l=1}^{L} \mathrm{res}^{(l)}$ |
