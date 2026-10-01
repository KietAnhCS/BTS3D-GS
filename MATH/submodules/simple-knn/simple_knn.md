# Công thức toán học của submodule `simple-knn`

Submodule này tính, cho mỗi điểm 3D $i$ trong đám mây điểm khởi tạo, **trung bình bình phương khoảng cách Euclid tới $K=3$ hàng xóm gần nhất**. Giá trị này (`dist2`) được `scene/gaussian_model.py` dùng làm scale khởi tạo của Gaussian:

```python
# scene/gaussian_model.py, dòng 271-272
dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
```

tức mỗi Gaussian ban đầu là một **hình cầu đẳng hướng** (isotropic) với bán kính $\approx\sqrt{\overline{d^2}}$ — xấp xỉ khoảng cách trung bình tới 3 điểm lân cận, để các Gaussian lấp đầy không gian điểm mà không chồng lấn quá mức.

---

## Ký hiệu

- $P$: số điểm trong đám mây. $\mathbf x_i=(x_i,y_i,z_i)\in\mathbb R^3$, $i=0,\dots,P-1$: toạ độ điểm thứ $i$ (kiểu `float3`).
- $\texttt{minn}=(x^{\min},y^{\min},z^{\min})$, $\texttt{maxx}=(x^{\max},y^{\max},z^{\max})$: bounding box (AABB) của toàn bộ đám mây.
- $b=10$: số bit lượng tử hoá mỗi trục (code dùng `(1<<10)-1 = 1023`).
- $\mathrm{dilate}_3(v)$: hàm "dãn" (bit-dilation) — chèn 2 bit 0 sau mỗi bit của số 10-bit $v$, để bit thứ $k$ của $v$ rơi vào vị trí $3k$ của kết quả. Đây chính là hàm `prepMorton`.
- $M(\mathbf x)\in\{0,\dots,2^{30}-1\}$: mã Morton (Z-order code) 30-bit của điểm đã lượng tử hoá.
- $\pi$: hoán vị của $\{0,\dots,P-1\}$ sinh ra bởi sắp xếp các điểm theo $M(\mathbf x_i)$ tăng dần (biến `indices_sorted`); $\pi(j)$ là chỉ số điểm gốc đứng ở vị trí $j$ trong thứ tự Morton.
- `BOX_SIZE` $=1024$: kích thước mỗi "hộp" (box) — một đoạn liên tiếp 1024 điểm trong thứ tự Morton, đóng vai trò như lá của một BVH ngầm một tầng.
- $\mathrm{Box}_b=\{\pi(j): b\cdot 1024\le j<(b+1)\cdot 1024\}$, với AABB riêng $[\mathbf m_b,\mathbf M_b]$ (`MinMax`).
- $d^2(\mathbf p,\mathbf q)=\lVert\mathbf p-\mathbf q\rVert_2^2$: bình phương khoảng cách Euclid.
- $d^2_{\text{box}}(\mathbf p,\mathrm{Box}_b)$: cận dưới của $d^2(\mathbf p,\mathbf q)$ với mọi $\mathbf q\in\mathrm{Box}_b$ (khoảng cách điểm–AABB).
- $\mathrm{knn}_i=(\kappa_{i,0}\le\kappa_{i,1}\le\kappa_{i,2})$: 3 giá trị $d^2$ nhỏ nhất tìm được cho điểm $i$ (mảng `best`).
- $\overline{d^2_i}$: output cuối (biến `meanDists[i]` / `dist2` trong Python).

---

## Bước 1: Hàm dãn bit `prepMorton` (dòng 47-54)

```cpp
__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8))  & 0x0300F00F;
	x = (x | (x << 4))  & 0x030C30C3;
	x = (x | (x << 2))  & 0x09249249;
	return x;
}
```

Đây là kỹ thuật kinh điển "magic numbers" để dãn 10 bit thấp của $x$ ra 30 bit, chèn đúng 2 bit 0 giữa hai bit liên tiếp. Viết $x=\sum_{k=0}^{9}x_k 2^k$ với $x_k\in\{0,1\}$. Mỗi bước AND với mask chỉ giữ lại đúng những bit đã nằm đúng vị trí đích sau khi OR-shift; sau 4 bước:

$$
\mathrm{dilate}_3(x)=\mathrm{prepMorton}(x)=\sum_{k=0}^{9}x_k\,2^{3k}
$$

Kiểm tra: mask cuối $\texttt{0x09249249}=\texttt{0b}\underbrace{001\,001\,001\,001\,001\,001\,001\,001\,001}_{\text{bit }3k\text{ được giữ, }k=0..9}\texttt{001}$ — đúng là tập các vị trí $\{0,3,6,\dots,27\}$, xác nhận công thức trên.

## Bước 2: Mã Morton 3D `coord2Morton` (dòng 56-63)

```cpp
__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));
	return x | (y << 1) | (z << 2);
}
```

**2.1. Lượng tử hoá tuyến tính về lưới $[0,1023]$:**

$$
q_x=\left\lfloor \frac{x-x^{\min}}{x^{\max}-x^{\min}}\cdot(2^{10}-1)\right\rfloor,\qquad
q_y,\ q_z\ \text{tương tự}
$$

(dấu $\lfloor\cdot\rfloor$ do ép kiểu `float`→`uint32_t` trong tham số của `prepMorton`).

**2.2. Dãn bit từng trục rồi đan xen (bit-interleaving):**

$$
X=\mathrm{dilate}_3(q_x)=\sum_k (q_x)_k 2^{3k},\quad
Y=\mathrm{dilate}_3(q_y),\quad Z=\mathrm{dilate}_3(q_z)
$$

$$
\boxed{\ M(\mathbf x)=X\mid (Y\ll1)\mid(Z\ll2)=\sum_{k=0}^{9}\Big[(q_x)_k\,2^{3k}+(q_y)_k\,2^{3k+1}+(q_z)_k\,2^{3k+2}\Big]\ }
$$

Đây đúng là định nghĩa chuẩn của **Morton code / Z-order curve** 3 chiều: bit thứ $3k,3k+1,3k+2$ của mã lần lượt là bit thứ $k$ của $q_x,q_y,q_z$.

### Kernel `coord2Morton` (dòng 65-72)

```cpp
__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) return;
	codes[idx] = coord2Morton(points[idx], minn, maxx);
}
```

Chỉ đơn thuần song song hoá công thức Bước 2 trên $P$ điểm, mỗi thread một điểm: $\texttt{codes}[i]=M(\mathbf x_i)$, $i=0,\dots,P-1$, độc lập — không có phụ thuộc dữ liệu giữa các thread (embarrassingly parallel).

## Bước 3: Bounding box toàn cục (trong `SimpleKNN::knn`, dòng 193-202)

```cpp
cub::DeviceReduce::Reduce(..., points, result, P, CustomMin(), init);
...
cub::DeviceReduce::Reduce(..., points, result, P, CustomMax(), init);
```

với `CustomMin`/`CustomMax` (dòng 31-45) là phép toán element-wise trên `float3`. Đây là một **parallel reduction** chuẩn:

$$
\texttt{minn}=\Big(\min_i x_i,\ \min_i y_i,\ \min_i z_i\Big),\qquad
\texttt{maxx}=\Big(\max_i x_i,\ \max_i y_i,\ \max_i z_i\Big)
$$

tính độc lập theo từng trục bằng cây giảm (tree reduction) $O(\log P)$ bước trên GPU (bên trong `cub::DeviceReduce`).

## Bước 4: Sắp xếp theo Morton code (dòng 204-215)

```cpp
coord2Morton <<<...>>> (P, points, minn, maxx, morton.data().get());
thrust::sequence(indices.begin(), indices.end());           // indices[i] = i
cub::DeviceRadixSort::SortPairs(..., morton, morton_sorted, indices, indices_sorted, P);
```

Sắp xếp cặp khoá-giá trị $(M(\mathbf x_i),\,i)$ theo $M$ tăng dần bằng **radix sort song song** (so sánh theo từng nhóm bit của khoá 30-bit, ổn định, $O(P)$ amortized với số bit cố định). Kết quả là hoán vị $\pi$:

$$
M(\mathbf x_{\pi(0)})\le M(\mathbf x_{\pi(1)})\le\cdots\le M(\mathbf x_{\pi(P-1)})
$$

Vì Morton code bảo toàn cục bộ không gian (hai điểm gần nhau trong $\mathbb R^3$ *thường* có mã gần nhau — xem phần "Kiểm chứng" về trường hợp ngoại lệ), thứ tự $\pi$ gần đúng một phép duyệt không gian liên tục ("space-filling curve"), cho phép dùng chỉ số mảng làm proxy cho khoảng cách không gian ở Bước 6.

## Bước 5: Bounding box theo từng hộp — `boxMinMax` (dòng 80-119)

```cpp
__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	...
	me.minn = me.maxx = points[indices[idx]];     // idx < P, else ±FLT_MAX (phần tử trung lập)
	__shared__ MinMax redResult[BOX_SIZE];
	for (int off = BOX_SIZE/2; off >= 1; off /= 2) {
		if (threadIdx.x < 2*off) redResult[threadIdx.x] = me;
		__syncthreads();
		if (threadIdx.x < off) {
			MinMax other = redResult[threadIdx.x + off];
			me.minn = min(me.minn, other.minn);  me.maxx = max(me.maxx, other.maxx);
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) boxes[blockIdx.x] = me;
}
```

Mỗi **block** CUDA gồm `BOX_SIZE`=1024 thread xử lý đúng 1024 phần tử liên tiếp của $\pi$ (một block = một box $\mathrm{Box}_b$). Đây là **parallel tree reduction trong shared memory**, $\log_2(1024)=10$ bước giảm đôi:

$$
\mathbf m_b=\min_{j\in\mathrm{Box}_b}\mathbf x_{\pi(j)},\qquad
\mathbf M_b=\max_{j\in\mathrm{Box}_b}\mathbf x_{\pi(j)}
$$

(phần tử trung lập $(+\infty,-\infty)$ cho các thread "thừa" khi $P$ không chia hết 1024, không ảnh hưởng kết quả min/max thật). Kết quả: `boxes[b]` = AABB bao các điểm có chỉ số Morton liền kề $b\cdot1024,\dots,(b+1)\cdot1024-1$ — đây chính là **"BVH ngầm 1 tầng"**: lá = từng box 1024 điểm, không có cây phân cấp thật sự (không chia nhỏ tiếp, không merge box).

## Bước 6: Khoảng cách điểm–hộp — `distBoxPoint` (dòng 121-131)

```cpp
__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = {0,0,0};
	if (p.x < box.minn.x || p.x > box.maxx.x) diff.x = min(abs(p.x-box.minn.x), abs(p.x-box.maxx.x));
	// tương tự diff.y, diff.z
	return diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
}
```

Đây là công thức chuẩn của **khoảng cách bình phương từ điểm tới AABB** (clamping từng trục):

$$
\delta_a(p_a)=\max\big(0,\ m_a-p_a,\ p_a-M_a\big),\qquad a\in\{x,y,z\}
$$

$$
\boxed{\ d^2_{\text{box}}(\mathbf p,\mathrm{Box}_b)=\sum_{a\in\{x,y,z\}}\delta_a(p_a)^2\ }
$$

(Khi $p_a\in[m_a,M_a]$ code gán $\mathrm{diff}_a=0$ đúng bằng $\delta_a=0$; khi $p_a<m_a$, $\min(|p_a-m_a|,|p_a-M_a|)=m_a-p_a=\delta_a$ vì $|p_a-m_a|<|p_a-M_a|$ trong trường hợp này — tương tự phía $p_a>M_a$. Vậy công thức trong code **tương đương toán học** với dạng clamp chuẩn, chỉ viết theo cách khác.) Đây là cận dưới "admissible": với mọi $\mathbf q\in\mathrm{Box}_b$, $d^2(\mathbf p,\mathbf q)\ge d^2_{\text{box}}(\mathbf p,\mathrm{Box}_b)$ — tính chất này là nền tảng cho phép cắt tỉa (pruning) branch-and-bound ở Bước 8.

## Bước 7: Duy trì 3-NN — `updateKBest<K>` (dòng 133-147)

```cpp
template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = {point.x-ref.x, point.y-ref.y, point.z-ref.z};
	float dist = d.x*d.x + d.y*d.y + d.z*d.z;
	for (int j = 0; j < K; j++)
		if (knn[j] > dist) { float t = knn[j]; knn[j] = dist; dist = t; }
}
```

Tính $d^2(\mathbf{ref},\mathbf{point})$ rồi **chèn có thứ tự** (insertion) vào mảng `knn[0..K-1]` được duy trì **tăng dần**: nếu giá trị mới nhỏ hơn `knn[j]`, hoán đổi và tiếp tục "đẩy" giá trị cũ (lớn hơn) xuống các vị trí sau. Đây tương đương một **bounded min-structure kích thước $K$** (hàng đợi ưu tiên giới hạn, ở đây $K=3$ nên cài bằng vòng lặp tuyến tính thay vì heap nhị phân — hợp lý vì $K$ nhỏ, $O(K)$ mỗi lần chèn, rẻ hơn overhead của heap thật). Bất biến sau mỗi lần gọi:

$$
\mathrm{knn}[0]\le \mathrm{knn}[1]\le\cdots\le \mathrm{knn}[K-1]=\text{$K$ giá trị } d^2 \text{ nhỏ nhất đã thấy}
$$

## Bước 8: Kernel chính — `boxMeanDist` (dòng 149-185)

```cpp
__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P) return;
	float3 point = points[indices[idx]];
	float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

	for (int i = max(0, idx-3); i <= min(P-1, idx+3); i++) {
		if (i == idx) continue;
		updateKBest<3>(point, points[indices[i]], best);
	}
	float reject = best[2];
	best[0] = best[1] = best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1)/BOX_SIZE; b++) {
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2]) continue;
		for (int i = b*BOX_SIZE; i < min(P, (b+1)*BOX_SIZE); i++) {
			if (i == idx) continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}
```

Thuật toán cho mỗi điểm $\mathbf x_{\pi(\mathrm{idx})}$ (viết tắt $\mathbf p=\mathbf x_{\pi(\mathrm{idx})}$):

**8.1. Cận trên khởi tạo (pha "sơ bộ").** Quét cửa sổ $\pm3$ vị trí lân cận trong mảng đã sắp Morton:

$$
\mathrm{reject}=\kappa^{(0)}_2=\text{giá trị lớn thứ 3 (tức phần tử thứ 3) trong }
\{d^2(\mathbf p,\mathbf x_{\pi(j)}):\, j\in[\mathrm{idx}-3,\mathrm{idx}+3]\setminus\{\mathrm{idx}\}\}
$$

Đây **không phải** kết quả cuối (mảng `best` bị reset ngay sau đó) — nó chỉ là một **cận trên hợp lệ** của $\kappa_{i,2}$ thật (vì các điểm được so sánh là điểm thật), dùng để cắt tỉa ở pha sau.

**8.2. Pha box (branch-and-bound).** Với mỗi box $\mathrm{Box}_b$ ($b=0,\dots,\lceil P/1024\rceil-1$):

$$
\text{nếu } d^2_{\text{box}}(\mathbf p,\mathrm{Box}_b) > \min(\mathrm{reject},\,\kappa_{i,2}^{\text{hiện tại}}) \ \Rightarrow\ \text{bỏ qua toàn bộ box}
$$

ngược lại quét tuần tự mọi điểm trong box và cập nhật `best` bằng `updateKBest<3>`. Vì $d^2_{\text{box}}$ là cận dưới hợp lệ (Bước 6), việc bỏ qua box không bao giờ bỏ sót một hàng xóm thật sự gần hơn $\kappa_{i,2}$ hiện tại — **mọi** box được xét (không box nào bị bỏ qua mà không kiểm tra điều kiện), nên kết quả cuối là **chính xác** (không xấp xỉ) so với brute-force trên toàn bộ $P$ điểm, miễn $K=3$ cố định và không giới hạn gì khác (xem thêm mục "Kiểm chứng").

**8.3. Output.**

$$
\boxed{\ \overline{d^2_i}=\frac{\kappa_{i,0}+\kappa_{i,1}+\kappa_{i,2}}{3}\ }
$$

tức **trung bình cộng của 3 giá trị bình phương khoảng cách nhỏ nhất** — không phải trung bình của khoảng cách (không có $\sqrt{\cdot}$ ở đây).

## Bước 9: Điều phối — `SimpleKNN::knn` (dòng 187-223)

Hàm host gom toàn bộ pipeline, theo đúng thứ tự Bước 3 → 4 → 5 → 8:

```cpp
void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
    // (3) bbox toàn cục bằng cub::DeviceReduce (Min rồi Max)
    // (4) coord2Morton kernel -> radix sort (morton, index) theo khoá morton
    // (5) boxMinMax kernel trên indices_sorted -> boxes[]
    // (8) boxMeanDist kernel -> meanDists[]
}
```

Không có phép toán mới ở đây — thuần tuý là lắp ráp (orchestration) các kernel đã suy ở trên theo đúng thứ tự dữ liệu phụ thuộc: cần `minn,maxx` trước khi tính Morton; cần Morton sort trước khi tính box; cần box trước khi tính khoảng cách.

## Bước 10: Wrapper PyTorch — `distCUDA2` (`spatial.cu`, dòng 15-26)

```cpp
torch::Tensor distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);
  torch::Tensor means = torch::full({P}, 0.0, points.options().dtype(torch::kFloat32));
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());
  return means;
}
```

Chỉ là **ép kiểu** con trỏ dữ liệu tensor $(P,3)$ liên tục trong bộ nhớ sang mảng `float3*` (giả định layout hàng-chính của PyTorch khớp với struct `{x,y,z}` — đúng vì tensor `(P,3)` contiguous có đúng 3 số `float` liên tiếp cho mỗi điểm), gọi `SimpleKNN::knn`, trả về tensor `means` $\in\mathbb R^P$ với

$$
\texttt{means}_i=\overline{d^2_i}\quad(\text{gọi là "dist2" ở phía Python})
$$

`ext.cpp` chỉ bind tên `distCUDA2` vào module Python `simple_knn._C` (không có phép toán).

## Bước 11: Sử dụng trong `scene/gaussian_model.py` (dòng 271-272)

$$
d2_i=\max(\overline{d^2_i},\,10^{-7}),\qquad
s_i=\log\sqrt{d2_i}=\tfrac12\log d2_i
$$

$s_i$ được gán cho cả 3 trục scale (`repeat(1,3)`) — Gaussian khởi tạo **đẳng hướng**; do activation của scale trong 3DGS là $\exp(\cdot)$, scale thật sau activation là $\exp(s_i)=\sqrt{d2_i}\approx\sqrt{\overline{d^2_i}}$ — đúng bằng căn bậc hai của trung bình bình phương khoảng cách tới 3 hàng xóm gần nhất, một ước lượng mật độ điểm cục bộ (local point spacing).

---

## Kiến thức toán nền tảng

- **Lý thuyết số / số học bit**: phép dãn bit (bit dilation/interleaving) dùng magic-number bitmask — liên quan lý thuyết biểu diễn nhị phân và các phép toán bit song song (bit tricks, "Morton code" cổ điển trong đồ hoạ máy tính và cơ sở dữ liệu không gian).
- **Lý thuyết đường cong lấp đầy không gian (space-filling curve)**: Z-order curve/Morton order là một ánh xạ $\mathbb Z_{\ge0}^3\to\mathbb Z_{\ge0}$ bảo toàn cục bộ (locality-preserving) nhưng không hoàn hảo — nền tảng lý thuyết cho việc dùng thứ tự 1D làm proxy của khoảng cách 3D.
- **Không gian metric Euclid $(\mathbb R^3,\lVert\cdot\rVert_2)$**: định nghĩa khoảng cách, bất đẳng thức tam giác là cơ sở cho tính hợp lệ (admissibility) của cận dưới $d^2_{\text{box}}$.
- **Hình học tính toán — khoảng cách điểm tới AABB**: công thức clamp từng trục, dùng rộng rãi trong các cấu trúc tăng tốc không gian (BVH, kd-tree, octree, R-tree) cho truy vấn phạm vi/láng giềng gần nhất.
- **Bài toán $k$-nearest-neighbor (KNN)**: định nghĩa chuẩn, so sánh brute-force $O(P^2)$ với các cấu trúc tăng tốc (kd-tree $O(P\log P)$ trung bình, octree, BVH); kỹ thuật branch-and-bound / pruning dùng cận dưới để loại nhánh an toàn.
- **Cấu trúc dữ liệu top-$K$ / hàng đợi ưu tiên giới hạn**: duy trì $K$ phần tử nhỏ nhất bằng chèn tuyến tính (ở đây) hoặc bounded max-heap (tổng quát) — độ phức tạp $O(K)$ mỗi lần chèn.
- **Giải thuật song song GPU**: parallel reduction (min/max cây nhị phân, $O(\log n)$ bước), parallel radix sort (sắp xếp theo khoá nguyên cố định số bit, $O(n)$ số bước hằng), mô hình lập trình SIMT (một thread/điểm, một block/box).
- **Sai số lượng tử hoá (quantization error)**: rời rạc hoá toạ độ liên tục thành lưới $2^{10}$ mức mỗi trục trước khi tính Morton code — sai số tối đa mỗi trục $\approx \tfrac{1}{1023}(\text{maxx}-\text{minn})$.

---

## Kiểm chứng tính đúng sai

**1. Morton code — khớp lý thuyết chuẩn.** Công thức suy ra ở Bước 2,
$M(\mathbf x)=\sum_k[(q_x)_k2^{3k}+(q_y)_k2^{3k+1}+(q_z)_k2^{3k+2}]$, đúng là định nghĩa tiêu chuẩn của Morton/Z-order code 3D (đan xen bit của 3 toạ độ nguyên hoá). Việc kiểm tra mask `0x09249249` ở Bước 1 xác nhận `prepMorton` thực hiện đúng phép dãn-bit-mỗi-3. **Đúng lý thuyết.**

**2. Bug tiềm ẩn ở Morton: chia 0 và tràn do làm tròn dấu phẩy động.**
- Nếu bbox suy biến theo một trục (tất cả điểm có cùng $x$, ví dụ đám mây phẳng), `maxx.x - minn.x = 0` → chia 0 → `inf`/`nan`. Ép kiểu `nan`/`inf` sang `uint32_t` là **undefined behavior** trong C++ — đây là lỗi toán học thật sự (không được code xử lý, không có `+eps` ở mẫu số).
- Dù bbox không suy biến, sai số làm tròn có thể khiến $(\text{coord}-\text{minn})$ âm rất nhỏ cho chính điểm đạt `minn` (hiếm nhưng có thể xảy ra do thứ tự rút gọn song song của `cub::DeviceReduce` không nhất thiết khớp bit-for-bit với giá trị gốc) → tỉ lệ âm nhỏ → ép kiểu `float`→`uint32_t` của số âm là **undefined behavior**, có thể "wrap" thành số rất lớn, làm `prepMorton` nhận input vượt quá 10 bit → các bit cao "tràn" chồng lấn sang vùng bit của trục khác khi OR lại ở dòng 62, làm hỏng mã Morton của điểm đó (không gây crash, nhưng làm sai thứ tự sắp xếp cục bộ — ảnh hưởng hiệu năng, không ảnh hưởng *tính đúng* của kết quả KNN cuối vì box vẫn được quét đủ, chỉ pruning kém hiệu quả hơn).

**3. Thuật toán 3-NN — chính xác (exact), không phải xấp xỉ, nhờ pruning dùng cận dưới hợp lệ.** Mặc dù tên thư viện là "simple-knn" và cách làm (chia box theo Morton order) gợi ý một heuristic, về mặt toán học thuật toán **không bỏ sót hàng xóm thật**: mọi điểm thuộc đúng một box, với mỗi box hoặc (a) bị chứng minh an toàn là không chứa điểm nào gần hơn ngưỡng hiện tại (vì $d^2_{\text{box}}$ là cận dưới thật của khoảng cách tới *bất kỳ* điểm nào trong box — Bước 6), hoặc (b) được quét toàn bộ. Do đó kết quả `best[0..2]` sau vòng lặp box đúng bằng 3-NN thật theo khoảng cách Euclid bình phương trên toàn bộ $P$ điểm — **tương đương brute-force có tăng tốc**, không phải approximate-KNN theo nghĩa "có thể sai". Điểm "xấp xỉ" duy nhất nằm ở hiệu năng: nếu Morton order cục bộ tệ (xem mục 4), nhiều box không bị cắt tỉa được, thuật toán chậm lại nhưng **vẫn đúng**.

**4. Hạn chế của Morton order làm proxy khoảng cách — ảnh hưởng hiệu năng, không ảnh hưởng tính đúng.** Z-order curve có các "bước nhảy" (discontinuity) nổi tiếng: hai điểm kề nhau trong không gian có thể có mã Morton rất khác nhau (ví dụ $(0,1,1)$ và $(1,0,0)$ ở ranh giới một khối lớn). Vì vậy cửa sổ $\pm3$ ở Bước 8.1 (pha "sơ bộ") không đảm bảo tìm hàng xóm thật gần — nhưng nó chỉ dùng để tạo `reject`, một **cận trên hợp lệ** (so sánh với điểm thật), nên không gây sai; nó chỉ có thể làm `reject` lớn hơn mức tối ưu, khiến ít box bị cắt tỉa hơn → chậm hơn, không sai hơn.

**5. Dấu — không có bug.** Mọi phép trừ dùng để tính $d^2$ đều bị bình phương ngay (`d.x*d.x`), nên không có vấn đề về dấu.

**6. Tràn số (overflow) khi $P$ rất nhỏ.** Nếu $P\le3$ (hoặc một điểm không có đủ 3 hàng xóm nào gần — không thể xảy ra nếu $P\ge4$, nhưng $P<4$ thì có), một số slot của `best[]` giữ nguyên `FLT_MAX`. Khi đó `best[0]+best[1]+best[2]` có thể là tổng của 2–3 lần `FLT_MAX` $\approx3.4\times10^{38}$, **vượt quá** giá trị biểu diễn được của `float` ($\approx3.4\times10^{38}$) → tràn thành `+Inf`. Chia cho 3 vẫn là `Inf`. Đây là **bug số học có thật** (dù hiếm gặp trong thực tế vì số điểm khởi tạo của 3DGS luôn $\gg4$), không được code phòng vệ (không có `isfinite` check hay giới hạn dưới của $P$).

**7. `clamp_min(dist2, 1e-7)` phía Python — hợp lý.** Vì $\overline{d^2_i}\ge0$ luôn đúng (tổng bình phương), chỉ có nguy cơ $\overline{d^2_i}=0$ khi có $\ge3$ điểm trùng toạ độ chính xác — khi đó $\log\sqrt{0}=-\infty$. Việc `clamp_min` ở `gaussian_model.py` dòng 271 xử lý đúng trường hợp biên này.

**Kết luận:** phần Morton code khớp hoàn toàn lý thuyết chuẩn; phần 3-NN tuy trông như heuristic nhưng thực chất là brute-force-có-tỉa-nhánh **chính xác** (không mất hàng xóm), chỉ có hiệu năng phụ thuộc chất lượng locality của Morton order. Bug thật sự nằm ở các trường hợp biên số học: chia 0 khi bbox suy biến, ép kiểu số âm/NaN sang `uint32_t` (UB), và tràn số khi $P$ cực nhỏ.

---

## Ví dụ số

### A. Minh hoạ bit-interleaving (dùng $b=3$ bit/trục thay vì 10, để tính tay được — cơ chế giống hệt, chỉ ít bit đệm hơn)

Chọn 6 điểm, bbox $\texttt{minn}=(0,0,0)$, $\texttt{maxx}=(5,5,5)$:

| Điểm | $(x,y,z)$ |
|---|---|
| $P_0$ | $(0,0,0)$ |
| $P_1$ | $(1,0,0)$ |
| $P_2$ | $(0,1,0)$ |
| $P_3$ | $(0,0,1)$ |
| $P_4$ | $(2,2,2)$ |
| $P_5$ | $(5,5,5)$ |

Lượng tử hoá $q=\lfloor \frac{x-0}{5}\cdot7\rfloor$ (dùng $7=2^3-1$ thay cho $1023=2^{10}-1$):

| Điểm | $q=(q_x,q_y,q_z)$ |
|---|---|
| $P_0$ | $(0,0,0)$ |
| $P_1$ | $(1,0,0)$ — vì $\lfloor1/5\cdot7\rfloor=\lfloor1.4\rfloor=1$ |
| $P_2$ | $(0,1,0)$ |
| $P_3$ | $(0,0,1)$ |
| $P_4$ | $(2,2,2)$ — vì $\lfloor2/5\cdot7\rfloor=\lfloor2.8\rfloor=2$ |
| $P_5$ | $(7,7,7)$ — vì $5/5\cdot7=7$ đúng |

Hàm dãn 3-bit $\mathrm{dilate}_3(v)$, viết $v=v_2v_1v_0$ (nhị phân), đặt bit $k$ vào vị trí $3k$:

$$
\mathrm{dilate}_3(0){=}0,\quad
\mathrm{dilate}_3(1){=}\texttt{0b}1{=}1,\quad
\mathrm{dilate}_3(2){=}\texttt{0b}1000{=}8,\quad
\mathrm{dilate}_3(7){=}\texttt{0b}001001001{=}73
$$

(kiểm tra $\mathrm{dilate}_3(7)$: $v=111_2$ có cả 3 bit $=1$ → đặt tại vị trí $0,3,6$: $2^0+2^3+2^6=1+8+64=73$ ✓.)

Mã Morton $M=X\mid(Y\ll1)\mid(Z\ll2)$:

| Điểm | $X=\mathrm{dil}(q_x)$ | $Y\ll1$ | $Z\ll2$ | $M$ |
|---|---|---|---|---|
| $P_0$ | $0$ | $0$ | $0$ | $\mathbf{0}$ |
| $P_1$ | $1$ | $0$ | $0$ | $\mathbf{1}$ |
| $P_2$ | $0$ | $1\ll1{=}2$ | $0$ | $\mathbf{2}$ |
| $P_3$ | $0$ | $0$ | $1\ll2{=}4$ | $\mathbf{4}$ |
| $P_4$ | $8$ | $8\ll1{=}16$ | $8\ll2{=}32$ | $8{\mid}16{\mid}32=\mathbf{56}$ |
| $P_5$ | $73$ | $146$ | $292$ | $73{\mid}146{\mid}292=\mathbf{511}$ |

($73=\texttt{0b}001001001$, $146{=}\texttt{0b}010010010$, $292{=}\texttt{0b}100100100$; OR theo bit cho $\texttt{0b}111111111{=}511=2^9{-}1$ — đúng giá trị lớn nhất có thể với 3 trục × 3 bit, khớp kỳ vọng vì $P_5$ ở góc xa nhất `maxx`.)

Thứ tự sắp theo $M$ tăng dần: $P_0(0),P_1(1),P_2(2),P_3(4),P_4(56),P_5(511)$ — trùng khớp trực giác khoảng cách tới gốc toạ độ, minh hoạ tính bảo toàn cục bộ của Morton order.

### B. Tính tay $\overline{d^2}$ cho $P_0$ và $P_4$ ($K=3$, brute-force, xác nhận công thức Bước 8)

Dùng toạ độ **thực** (không lượng tử hoá — `boxMeanDist` dùng `float3` gốc):

Khoảng cách bình phương từ $P_0=(0,0,0)$:

$$
d^2(P_0,P_1)=1,\quad d^2(P_0,P_2)=1,\quad d^2(P_0,P_3)=1,\quad d^2(P_0,P_4)=4{+}4{+}4=12,\quad d^2(P_0,P_5)=75
$$

3 nhỏ nhất: $(1,1,1)$ →

$$
\overline{d^2_{P_0}}=\frac{1+1+1}{3}=1.0
$$

**Mô phỏng `updateKBest<3>` bằng tay** (thứ tự xử lý $P_1,P_2,P_3,P_4,P_5$, bắt đầu `best=[∞,∞,∞]`):

- Xử lý $P_1$ ($dist=1$): $j{=}0$: $\infty>1$ → hoán đổi, `best[0]=1`, `dist=∞`; $j{=}1,2$: không đổi. `best=[1,∞,∞]`
- Xử lý $P_2$ ($dist=1$): $j{=}0$: $1>1$? sai; $j{=}1$: $\infty>1$ → `best[1]=1`. `best=[1,1,∞]`
- Xử lý $P_3$ ($dist=1$): $j{=}0,1$: sai; $j{=}2$: $\infty>1$ → `best[2]=1`. `best=[1,1,1]`
- Xử lý $P_4$ ($dist=12$): mọi $j$: $1>12$ sai → không đổi.
- Xử lý $P_5$ ($dist=75$): tương tự không đổi.

Kết quả cuối `best=[1,1,1]` → $\overline{d^2_{P_0}}=1.0$ — **khớp chính xác** tính tay brute-force ở trên.

Khoảng cách bình phương từ $P_4=(2,2,2)$:

$$
d^2(P_4,P_0)=12,\ d^2(P_4,P_1)=1{+}4{+}4=9,\ d^2(P_4,P_2)=4{+}1{+}4=9,\ d^2(P_4,P_3)=4{+}4{+}1=9,\ d^2(P_4,P_5)=27
$$

3 nhỏ nhất: $(9,9,9)$ →

$$
\overline{d^2_{P_4}}=\frac{9+9+9}{3}=9.0\ \Rightarrow\ \text{scale khởi tạo } =\sqrt{9.0}=3.0
$$

### C. Minh hoạ `distBoxPoint` và tính hợp lệ của pruning

Giả sử (để minh hoạ, `BOX_SIZE` nhỏ hơn thực tế) một box gồm $\{P_0,P_1,P_2,P_3\}$ có AABB $\texttt{minn}=(0,0,0)$, $\texttt{maxx}=(1,1,1)$. Tính $d^2_{\text{box}}$ từ $P_4=(2,2,2)$:

$$
\delta_x=\max(0,\,0-2,\,2-1)=1,\quad \delta_y=1,\quad \delta_z=1
\ \Rightarrow\ d^2_{\text{box}}(P_4,\mathrm{Box})=1+1+1=3
$$

So với khoảng cách thật gần nhất trong box, $d^2(P_4,P_1)=9$ (đã tính ở trên): $3\le9$ — cận dưới hợp lệ, **không bị vi phạm** (nếu ngưỡng cắt tỉa hiện tại $<3$ thì box này bị bỏ qua an toàn vì chắc chắn không có điểm nào trong box gần hơn ngưỡng; ở đây $3<9$ nên nếu ngưỡng đang là $9$, box vẫn phải được quét vì $3\le9$ — đúng logic dòng `if (dist > reject || dist > best[2]) continue;`).

### D. Kiểm tra output cuối cùng khớp `gaussian_model.py`

Với $P_0$: `dist2` $=\max(1.0,10^{-7})=1.0$ → `scales` $=\log\sqrt{1.0}=\log(1.0)=0$ cho cả 3 trục → sau activation $\exp(0)=1.0$ — đúng bằng khoảng cách (không bình phương) tới hàng xóm gần nhất thật ($P_1,P_2,P_3$ đều cách 1 đơn vị) — xác nhận trực giác hình học: Gaussian tại $P_0$ được khởi tạo với bán kính $\approx1$, vừa đủ phủ tới các hàng xóm gần nhất mà không chồng lấn quá mức.
