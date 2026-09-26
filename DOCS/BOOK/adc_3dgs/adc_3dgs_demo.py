"""
adc_3dgs_demo.py — Mô phỏng TOY cơ chế Structure-Aware Densification (SADGS,
SIGGRAPH 2026, "Faster 3D Gaussian Splatting Convergence via Structure-Aware
Densification"). SADGS kế thừa khung ADC của 3DGS/FastGS (lịch densify mỗi
`densification_interval` vòng, reset opacity định kỳ) nhưng THAY tiêu chí
clone/split bằng việc so khớp KÍCH THƯỚC MÀN HÌNH của mỗi Gaussian với cấu
trúc texture đa tỉ lệ (multiscale image structure) của ảnh, cộng thêm
multiview-consistency pruning.

ĐÂY LÀ TOY MODEL: không render ảnh, không có structure tensor thật. "Bước
sóng kết cấu cục bộ" lambda(x) (độ chi tiết của texture tại vị trí x) được
GIẢ LẬP: nhỏ (chi tiết mịn) ở viền một đường tròn, lớn (mượt) ở nơi khác.
Mục đích là minh hoạ đúng CÔNG THỨC & THAM SỐ thật của
SADGS/scene/gaussian_model.py (densify_and_split_structgs,
densify_and_clone_structgs, densify_and_prune_structgs, expand_undersized_gs,
final_prune_structgs) và SADGS/arguments/__init__.py:

  (1) Lịch: densify + prune mỗi `densification_interval` (=100) vòng, trong
      khoảng densify_from_iter (=500) < t < densify_until_iter (=15000);
      reset opacity NHÂN (không phải floor) mỗi opacity_reset_interval
      (=3000) vòng: alpha <- alpha * opacity_reset_decay (=0.1).

  (2) Với mỗi Gaussian i, mỗi lần "nhìn thấy" (1 vòng lặp ~ 1 view trong
      toy), tính violation tần số theo trục a in {x, y}:
          eta_i,a(t) = (kích thước hình chiếu trục a) / lambda(mu_i)
      (công thức thật dùng structure tensor 2x2 để lấy lambda1 = trị riêng
      lớn nhất = năng lượng tần số cao nhất, rồi lambda(x) = 1/sqrt(lambda1);
      eta = axis_length / lambda -> eta > 1 nghĩa là Gaussian TO hơn chi
      tiết texture -> alias -> cần split).
      Ta lưu:
        eta_max_i,a = max qua các lần nhìn thấy của eta_i,a(t)   (leaky max)
        view_count_i = số lần "nhìn thấy" tích cực (denom)
        eta_high_count_i = số lần eta_max_scalar > TAU_HIGH (=1.0)
        eta_low_count_i  = số lần eta_max_scalar <= TAU_LOW (=0.1)
      chỉ tích luỹ khi Gaussian "hiện diện" (visible) VÀ gradient vị trí giả
      lập vượt freq_grad_threshold (=2e-5) — mirror `update_freq_stats_online`.

  (3) Multiview-consistency (mỗi kỳ densify, dùng high_ratio/low_ratio thay
      cho 1 ngưỡng gradient duy nhất như 3DGS gốc):
          high_ratio_i = eta_high_count_i / view_count_i
          low_ratio_i  = eta_low_count_i  / view_count_i
          split_signal_i  = [high_ratio_i > split_ratio_threshold (=0.8)]
                              AND [g_bar_i >= grad_thresh (=2e-4)]
          prune_signal_i  = [low_ratio_i  > prune_ratio_threshold (=0.8)]
      rồi phân loại theo kích thước hiện tại (giống percent_dense cũ, nay
      gọi là `dense` = args.dense = 0.001):
          clone_i = split_signal_i AND [max(s_i) <= dense * extent]
          split_i = split_signal_i AND [max(s_i) >  dense * extent]
      và lọc thêm bằng importance_score (ở đây = view_count chuẩn hoá) so
      với importance_score_threshold (=0.5): Gaussian ít được "nhìn thấy"
      không được phép clone/split dù eta cao.

  (4) Clone: copy nguyên theta (như cũ)  -> N + 1.
      Split DỊ HƯỚNG theo trục (densify_and_split_structgs): số con theo
      mỗi trục  k_a = clip(ceil(sqrt(eta_max_i,a)), 1, max_clones_per_axis
      (=8))  (lý thuyết lấy mẫu: sigma_new*omega <= 1  <=>  k >= sqrt(eta)).
      Tổng con = k_x * k_y (3D thật: k_x*k_y*k_z). Đặt trên lưới đều tâm 0,
      độ giãn cách = sigma_new * sqrt(12), rồi xoay bởi R(theta_i):
          scale_new = s_i / k^ks_scale_power   (ks_scale_power = 1.0)
      Gốc bị xoá -> N + (k_x*k_y - 1) cho mỗi Gaussian bị split.

  (5) expand_undersized_gs (mô phỏng, KHÔNG chạy tự động trong train.py thật —
      lời gọi DUY NHẤT tới hàm này, ở train.py:356-359, đang bị comment out;
      toy demo này BẬT RIÊNG cơ chế dưới đây chỉ để minh hoạ công thức):
      Gaussian "quá nhỏ" so với texture cục bộ
      (0 < eta_max_i,a < tau_expand (=1.0)) được NỚI RỘNG trực tiếp (không
      sinh điểm mới) để đạt đúng eta = 1:
          log(s_new) = log(s_old) - 0.5 * log(eta_max_i,a)
      (đây là hiệu chỉnh phân tích, không phải gradient descent.)

  (6) Prune cứng (densify_and_prune_structgs): xoá_i =
          [alpha_i < min_opacity (=0.1)]                              OR
          [r_i^2D > 20 px, chỉ khi t > opacity_reset_interval]        OR
          [max(s_i) > 0.1 * extent]                                   OR
          [prune_signal_i (multiview-consistency, xem (3))]
      Sau mỗi lần densify+prune, opacity bị CHẶN TRẦN:
          alpha_i <- min(alpha_i, opacity_cap (=0.8))
      (khác 3DGS gốc: không ép sàn 0.005 mà chặn TRẦN 0.8.)

  (7) final_prune_structgs (mô phỏng, KHÔNG chạy tự động trong lịch chính —
      trong code thật hàm này được gọi rời, có điều kiện — ở đây minh hoạ
      1 lần tại t = densify_until_iter): xoá thêm Gaussian có
          alpha_i < min_opacity  OR  pruning_score_i > 0.9
      với pruning_score xấp xỉ = low_ratio quan sát được ở kỳ densify cuối
      cùng (đại diện cho "độ không nhất quán multi-view" thật được đo bằng
      multi-view reconstruction consistency trong code gốc).

Không gian là 2D để dễ vẽ; mở rộng 3D chỉ cần mu (N,3), scale (N,3), rot là
quaternion (N,4) thay cho góc xoay (N,), thêm trục k_z, và lambda(x)/eta lấy
từ structure tensor 2D thật của ảnh chiếu (không phải hàm giả lập ở đây).

Chạy:  python DOCS/BOOK/adc_3dgs/adc_3dgs_demo.py
Kết quả: DOCS/BOOK/adc_3dgs/adc_3dgs_demo.png + thống kê in ra màn hình.
Chỉ dùng numpy + matplotlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "adc_3dgs_demo.png")

# Vùng "chi tiết cao" giả lập: viền đường tròn tâm (0.5, 0.5), bán kính 0.3
CIRCLE_C = np.array([0.5, 0.5])
CIRCLE_R = 0.3


# ----------------------------------------------------------------------------
# Cấu hình SADGS — lấy đúng theo SADGS/arguments/__init__.py (OptimizationParams)
# ----------------------------------------------------------------------------
@dataclass
class ADCConfig:
    # --- lịch (giữ nguyên khung 3DGS/FastGS) ---
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    opacity_reset_decay: float = 0.1          # nhân, KHÔNG phải floor      (mục 1)
    total_iters: int = 20000
    extent: float = 1.0

    # --- ngưỡng structure-aware (đúng tên/giá trị mặc định trong arguments) ---
    dense: float = 0.001                      # args.dense (percent_dense)  (mục 3)
    grad_thresh: float = 0.0002               # gate is_grad_high            (mục 3)
    freq_grad_threshold: float = 0.00002      # gate tích luỹ thống kê eta   (mục 2)
    importance_score_threshold: float = 0.5   # lọc theo view_count          (mục 3)
    split_ratio_threshold: float = 0.8        # multiview-consistency split  (mục 3)
    prune_ratio_threshold: float = 0.8        # multiview-consistency prune  (mục 3)
    tau_expand: float = 1.0                   # expand_undersized_gs         (mục 5)
    ks_scale_power: float = 1.0               # new_scale = old / k^power    (mục 4)
    max_clones_per_axis: int = 8              # trần số con mỗi trục         (mục 4)
    eta_tau_high: float = 1.0                 # TAU_HIGH trong freq_utils.py (mục 2)
    eta_tau_low: float = 0.1                  # TAU_LOW trong freq_utils.py  (mục 2)
    min_opacity: float = 0.1                  # min_opacity trong train.py   (mục 6)
    scale_max_ratio: float = 0.1              # 0.1 * extent                 (mục 6)
    radius_2d_max_px: float = 20.0            # size_threshold                (mục 6)
    opacity_cap: float = 0.8                  # min(alpha, 0.8) sau densify  (mục 6)
    final_prune_score: float = 0.9            # pruning_score > 0.9           (mục 7)

    # --- tham số CHỈ của phần giả lập (không thuộc SADGS thật) ---
    focal_px: float = 800.0                   # r^2D ~ max(s) * focal / depth, depth = 1
    visibility_p: float = 0.7                 # xác suất Gaussian được "nhìn thấy" ở 1 view
    alpha_grow_per_iter: float = 0.0006
    alpha_decay_per_iter: float = 0.004
    dying_fraction: float = 0.04
    wavelength_rim: float = 0.012             # lambda(x) ở viền: chi tiết mịn  -> eta lớn
    wavelength_bg: float = 0.045              # lambda(x) nơi khác: mượt        -> eta nhỏ
    band_width: float = 0.03                  # bề rộng băng viền có texture mịn
    eta_noise_std: float = 0.1                # nhiễu log-normal của eta mỗi lần quan sát
    expand_step_clip: float = 0.4             # [chỉ giả lập] chặn bước expand_undersized_gs
                                               # mỗi chu kỳ (tránh dao động do nhiễu toy)
    density_sat: float = 1.0
    density_cell: float = 0.02
    pos_lr: float = 0.002
    pos_noise: float = 3e-4
    snapshot_iters: tuple = (500, 3000, 9000, 15000)


# ----------------------------------------------------------------------------
# Đám mây Gaussian 2D
# ----------------------------------------------------------------------------
@dataclass
class GaussianCloud:
    """Tập Gaussian 2D. Mọi mảng có cùng chiều dài N.

    mu           (N,2)  tâm
    scale        (N,2)  bán trục s_i (độ lệch chuẩn theo 2 trục riêng)
    rot          (N,)   góc xoay theta_i — đóng vai trò R(q_i) trong 2D
    alpha        (N,)   opacity dạng xác suất (code thật lưu logit)
    g_accum      (N,)   tích luỹ gradient vị trí (dùng cho is_grad_high, mục 3)
    g_denom      (N,)   số lần tích luỹ gradient vị trí
    eta_max      (N,2)  max_eta_3ch rút gọn 2 trục: max eta quan sát theo trục a
    view_count   (N,)   accum_view_count — số lần quan sát tích cực
    eta_hi_cnt   (N,)   eta_high_count — số lần eta_max_scalar > TAU_HIGH
    eta_lo_cnt   (N,)   eta_low_count  — số lần eta_max_scalar <= TAU_LOW
    rate         (N,)   [chỉ giả lập] tốc độ đổi alpha mỗi vòng (+: sống, -: chết)
    """

    mu: np.ndarray
    scale: np.ndarray
    rot: np.ndarray
    alpha: np.ndarray
    g_accum: np.ndarray = field(default=None)
    g_denom: np.ndarray = field(default=None)
    eta_max: np.ndarray = field(default=None)
    view_count: np.ndarray = field(default=None)
    eta_hi_cnt: np.ndarray = field(default=None)
    eta_lo_cnt: np.ndarray = field(default=None)
    rate: np.ndarray = field(default=None)

    def __post_init__(self):
        n = len(self.mu)
        if self.g_accum is None:
            self.g_accum = np.zeros(n)
        if self.g_denom is None:
            self.g_denom = np.zeros(n)
        if self.eta_max is None:
            self.eta_max = np.zeros((n, 2))
        if self.view_count is None:
            self.view_count = np.zeros(n)
        if self.eta_hi_cnt is None:
            self.eta_hi_cnt = np.zeros(n)
        if self.eta_lo_cnt is None:
            self.eta_lo_cnt = np.zeros(n)
        if self.rate is None:
            self.rate = np.zeros(n)
        self.check()

    _FIELDS = ("mu", "scale", "rot", "alpha", "g_accum", "g_denom",
               "eta_max", "view_count", "eta_hi_cnt", "eta_lo_cnt", "rate")

    @property
    def n(self) -> int:
        return len(self.mu)

    def check(self):
        n = self.n
        for f in self._FIELDS:
            arr = getattr(self, f)
            assert len(arr) == n, f"mảng {f} có {len(arr)} phần tử, mong {n}"
        assert self.mu.shape == (n, 2) and self.scale.shape == (n, 2)
        assert self.eta_max.shape == (n, 2)

    def reset_stats(self):
        """Đặt lại các bộ tích luỹ sau mỗi lần densify (như train.py: accum_eta,
        accum_view_count, max_eta_3ch, eta_high/low_count đều .zero_())."""
        n = self.n
        self.g_accum = np.zeros(n)
        self.g_denom = np.zeros(n)
        self.eta_max = np.zeros((n, 2))
        self.view_count = np.zeros(n)
        self.eta_hi_cnt = np.zeros(n)
        self.eta_lo_cnt = np.zeros(n)
        self.check()

    def grow(self, other: "GaussianCloud"):
        other.check()
        for f in self._FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), getattr(other, f)], axis=0))
        self.check()

    def keep(self, mask: np.ndarray):
        assert mask.shape == (self.n,), f"mask {mask.shape} không khớp N={self.n}"
        for f in self._FIELDS:
            setattr(self, f, getattr(self, f)[mask])
        self.check()

    def copy(self) -> "GaussianCloud":
        return GaussianCloud(**{f: getattr(self, f).copy() for f in self._FIELDS})

    def max_scale(self) -> np.ndarray:
        return self.scale.max(axis=1)


def rot2d(theta: np.ndarray) -> np.ndarray:
    """Ma trận xoay 2D (N,2,2) — R(q_i) trong 2D."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


# ----------------------------------------------------------------------------
# Trường texture đa tỉ lệ giả lập + gradient vị trí giả lập
# ----------------------------------------------------------------------------
def local_wavelength(mu: np.ndarray, cfg: ADCConfig) -> np.ndarray:
    """lambda(x): "bước sóng" chi tiết texture cục bộ. Nhỏ (mịn) ở viền vòng
    tròn (giống trị riêng lớn của structure tensor -> tần số cao), lớn
    (mượt) ở nơi khác. Đây là bản giả lập của `get_structure_tensor_torch`
    + việc lấy lambda1 rồi 1/sqrt(lambda1) trong `update_freq_stats_online`.
    """
    dist = np.linalg.norm(mu - CIRCLE_C, axis=1)
    in_band = np.abs(dist - CIRCLE_R) < cfg.band_width
    return np.where(in_band, cfg.wavelength_rim, cfg.wavelength_bg)


def fake_positional_grad(mu: np.ndarray, rng: np.random.Generator, cfg: ADCConfig):
    """Giả lập gradient vị trí dL/dmu' (chỉ dùng để gate is_grad_high, KHÔNG
    còn quyết định trực tiếp clone/split như 3DGS gốc). Cao gần viền, thấp
    nơi khác, cộng nhiễu và bão hoà khi ô lưới đông (tránh N bùng nổ)."""
    dist = np.linalg.norm(mu - CIRCLE_C, axis=1)
    in_band = np.abs(dist - CIRCLE_R) < cfg.band_width
    base = np.where(in_band, 4.0, 0.3) * cfg.grad_thresh
    noise = np.exp(rng.normal(0.0, 0.35, size=len(mu)))
    ncell = int(np.ceil(1.0 / cfg.density_cell))
    ij = np.clip((mu / cfg.density_cell).astype(int), 0, ncell - 1)
    flat = ij[:, 0] * ncell + ij[:, 1]
    counts = np.bincount(flat, minlength=ncell * ncell)[flat].astype(float)
    g = base * noise / (1.0 + counts / cfg.density_sat)
    visible = rng.random(len(mu)) < cfg.visibility_p
    return g, visible


def eta_axis_observation(mu: np.ndarray, scale: np.ndarray, rng: np.random.Generator,
                          cfg: ADCConfig) -> np.ndarray:
    """eta_i,a(t) = (kích thước trục a) / lambda(mu_i), với nhiễu log-normal
    mô phỏng việc mỗi view "đo" eta hơi khác nhau (stochastic jitter sampling
    trong code thật). Trả về (N,2)."""
    lam = local_wavelength(mu, cfg)[:, None]
    noise = np.exp(rng.normal(0.0, cfg.eta_noise_std, size=scale.shape))
    return (scale / lam) * noise


def accumulate_stats(cloud: GaussianCloud, eta_obs: np.ndarray, visible: np.ndarray,
                      g: np.ndarray, cfg: ADCConfig):
    """Mirror `update_freq_stats_online`: chỉ tích luỹ khi visible AND gradient
    vị trí giả lập vượt freq_grad_threshold (is_grad_high trong freq_utils)."""
    active = visible & (g > cfg.freq_grad_threshold)
    if not active.any():
        return
    cloud.eta_max[active] = np.maximum(cloud.eta_max[active], eta_obs[active])
    cloud.view_count += active.astype(float)
    eta_scalar = eta_obs.max(axis=1)
    is_high = active & (eta_scalar > cfg.eta_tau_high)
    is_low = active & (eta_scalar <= cfg.eta_tau_low)
    cloud.eta_hi_cnt += is_high.astype(float)
    cloud.eta_lo_cnt += is_low.astype(float)
    # công thức (2) g_bar_i = accum/denom, dùng gate is_grad_high ở bước densify
    cloud.g_accum += visible * g
    cloud.g_denom += visible.astype(float)


# ----------------------------------------------------------------------------
# Bước tối ưu giả lập (không thuộc SADGS)
# ----------------------------------------------------------------------------
def fake_optimizer_step(cloud: GaussianCloud, rng: np.random.Generator, cfg: ADCConfig):
    d = cloud.mu - CIRCLE_C
    dist = np.linalg.norm(d, axis=1) + 1e-9
    near = np.abs(dist - CIRCLE_R) < 2 * cfg.band_width
    radial = d / dist[:, None]
    step = -cfg.pos_lr * (dist - CIRCLE_R)[:, None] * radial * near[:, None]
    cloud.mu = cloud.mu + step + rng.normal(0.0, cfg.pos_noise, size=cloud.mu.shape)
    cloud.mu = np.clip(cloud.mu, 0.0, 1.0)
    cloud.alpha = np.clip(cloud.alpha + cloud.rate, 0.0, 0.99)


def new_rates(n: int, rng: np.random.Generator, cfg: ADCConfig) -> np.ndarray:
    dying = rng.random(n) < cfg.dying_fraction
    grow = cfg.alpha_grow_per_iter * rng.uniform(0.3, 1.7, size=n)
    return np.where(dying, -cfg.alpha_decay_per_iter, grow)


# ----------------------------------------------------------------------------
# SADGS: densify (clone dị hướng / split dị hướng / expand) + prune
# ----------------------------------------------------------------------------
def densify_and_prune_structgs(cloud: GaussianCloud, t: int, rng: np.random.Generator,
                                cfg: ADCConfig) -> dict:
    """Một chu kỳ densify + prune kiểu SADGS (mục 3-6 ở docstring đầu file).

    Thứ tự (mirror densify_and_prune_structgs + densify_and_split_structgs
    + expand_undersized_gs trong SADGS/scene/gaussian_model.py):
      1. tính g_bar, high_ratio, low_ratio, importance trên N cũ
      2. lọc clone_mask / split_mask theo (multiview-consistency) x (kích
         thước hiện tại) x (importance)
      3. clone (nhân bản) các Gaussian được chọn
      4. split dị hướng (kx, ky) các Gaussian được chọn, xoá gốc
      5. expand_undersized_gs cho các Gaussian CÒN LẠI (không bị clone/split)
         mà eta dưới tau_expand
      6. prune cứng trên cloud SAU CÙNG (mục 6), chặn trần opacity 0.8
      7. reset toàn bộ bộ tích luỹ (accum_eta, view_count, eta_hi/lo_cnt...)
    """
    n_old = cloud.n
    g_bar = np.where(cloud.g_denom > 0, cloud.g_accum / np.maximum(cloud.g_denom, 1.0), 0.0)
    is_grad_high = g_bar >= cfg.grad_thresh

    valid = cloud.view_count > 0
    high_ratio = np.where(valid, cloud.eta_hi_cnt / np.maximum(cloud.view_count, 1.0), 0.0)
    low_ratio = np.where(valid, cloud.eta_lo_cnt / np.maximum(cloud.view_count, 1.0), 0.0)

    # công thức (3): tín hiệu multiview-consistency
    split_signal = (high_ratio > cfg.split_ratio_threshold) & is_grad_high & valid
    prune_signal = (low_ratio > cfg.prune_ratio_threshold) & valid

    # importance_score = view_count chuẩn hoá (thay cho accum_view_count thật)
    vc_max = max(float(cloud.view_count.max()), 1.0)
    importance = cloud.view_count / vc_max
    metric_mask = importance > cfg.importance_score_threshold

    max_s = cloud.max_scale()
    thr_scale = cfg.dense * cfg.extent
    clone_qualifiers = max_s <= thr_scale
    split_qualifiers = max_s > thr_scale

    final_clone_mask = metric_mask & split_signal & clone_qualifiers
    final_split_mask = metric_mask & split_signal & split_qualifiers
    assert final_clone_mask.shape == (n_old,) and final_split_mask.shape == (n_old,)

    # --- công thức (4a) Clone: copy nguyên theta ---
    clones = GaussianCloud(
        mu=cloud.mu[final_clone_mask].copy(),
        scale=cloud.scale[final_clone_mask].copy(),
        rot=cloud.rot[final_clone_mask].copy(),
        alpha=cloud.alpha[final_clone_mask].copy(),
        rate=new_rates(int(final_clone_mask.sum()), rng, cfg),
    )

    # --- công thức (4b) Split dị hướng: k_a = clip(ceil(sqrt(eta_max_a)), 1, max_clones_per_axis) ---
    idx = np.nonzero(final_split_mask)[0]
    eta_split = cloud.eta_max[idx]                                    # (M,2)
    ks = np.clip(np.ceil(np.sqrt(np.maximum(eta_split, 1.0))).astype(int),
                 1, cfg.max_clones_per_axis)                          # (M,2) k_x, k_y
    n_children_per_parent = ks[:, 0] * ks[:, 1]
    n_children_total = int(n_children_per_parent.sum())

    child_mu = np.zeros((n_children_total, 2))
    child_scale = np.zeros((n_children_total, 2))
    child_rot = np.zeros(n_children_total)
    child_alpha = np.zeros(n_children_total)
    pos = 0
    for j, p in enumerate(idx):
        kx, ky = int(ks[j, 0]), int(ks[j, 1])
        s_old = cloud.scale[p]
        s_new = s_old / (np.array([kx, ky], dtype=float) ** cfg.ks_scale_power)
        sep = (s_old / np.array([kx, ky], dtype=float)) * np.sqrt(12.0)   # separations
        gx = np.arange(kx) - (kx - 1) / 2.0
        gy = np.arange(ky) - (ky - 1) / 2.0
        grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
        local_offsets = grid * sep
        R = rot2d(np.array([cloud.rot[p]]))[0]
        world_offsets = local_offsets @ R.T
        n_c = kx * ky
        child_mu[pos:pos + n_c] = cloud.mu[p] + world_offsets
        child_scale[pos:pos + n_c] = s_new
        child_rot[pos:pos + n_c] = cloud.rot[p]
        child_alpha[pos:pos + n_c] = cloud.alpha[p]
        pos += n_c
    child_mu = np.clip(child_mu, 0.0, 1.0)
    children = GaussianCloud(
        mu=child_mu, scale=child_scale, rot=child_rot, alpha=child_alpha,
        rate=new_rates(n_children_total, rng, cfg),
    )

    # xoá gốc bị split, nối clone + con (mirror densification_postfix + prune_points)
    cloud.keep(~final_split_mask)
    cloud.grow(clones)
    cloud.grow(children)
    n_after_grow = cloud.n
    assert n_after_grow == n_old + int(final_clone_mask.sum()) + n_children_total - int(final_split_mask.sum())

    # --- công thức (5) expand_undersized_gs: chỉ áp cho Gaussian KHÔNG bị clone/split ---
    # (các Gaussian mới ghép vào có eta_max = 0 -> undersized_mask False, an toàn)
    undersized_mask = (cloud.eta_max < cfg.tau_expand) & (cloud.eta_max > 0)
    if undersized_mask.any():
        eta_clamped = np.clip(cloud.eta_max, 1e-6, None)
        delta_log_scale = -0.5 * np.log(eta_clamped)
        # [chỉ giả lập] chặn biên độ 1 bước để tránh dao động do nhiễu toy
        # (code thật áp dụng đúng công thức phân tích, không cần chặn vì eta
        # được ước lượng từ nhiều view thật ổn định hơn nhiễu log-normal ở đây)
        delta_log_scale = np.clip(delta_log_scale, -cfg.expand_step_clip, cfg.expand_step_clip)
        cloud.scale = np.where(undersized_mask, cloud.scale * np.exp(delta_log_scale), cloud.scale)

    # --- công thức (6): prune cứng trên cloud SAU CÙNG ---
    max_s_new = cloud.max_scale()
    r2d_px = max_s_new * cfg.focal_px
    prune = cloud.alpha < cfg.min_opacity
    if t > cfg.opacity_reset_interval:
        prune |= r2d_px > cfg.radius_2d_max_px
    prune |= max_s_new > cfg.scale_max_ratio * cfg.extent
    # prune_signal chỉ áp cho các Gaussian gốc còn sống (chưa bị split); clone/con mới -> False
    prune_signal_full = np.concatenate([
        prune_signal[~final_split_mask],
        np.zeros(int(final_clone_mask.sum()), dtype=bool),
        np.zeros(n_children_total, dtype=bool),
    ])
    assert prune_signal_full.shape == prune.shape
    prune |= prune_signal_full
    cloud.keep(~prune)

    # chặn trần opacity 0.8 (mục 6)
    cloud.alpha = np.minimum(cloud.alpha, cfg.opacity_cap)

    last_low_ratio_mean = float(low_ratio[valid].mean()) if valid.any() else 0.0
    cloud.reset_stats()
    return {
        "clone": int(final_clone_mask.sum()),
        "split": int(final_split_mask.sum()),
        "split_children": n_children_total,
        "expand": int(undersized_mask.any(axis=1).sum()),
        "prune": int(prune.sum()),
        "prune_mv": int(prune_signal_full.sum()),
        "low_ratio_mean": last_low_ratio_mean,
    }


def final_prune_structgs(cloud: GaussianCloud, low_ratio_mean_hist: float, cfg: ADCConfig) -> int:
    """Mô phỏng final_prune_structgs: xoá thêm theo alpha thấp HOẶC
    pruning_score (đại diện = low_ratio trung bình quan sát gần nhất, đóng
    vai trò multi-view reconstruction-consistency score thật) > 0.9."""
    prune_mask = cloud.alpha < cfg.min_opacity
    if low_ratio_mean_hist > cfg.final_prune_score:
        # trong code thật score là theo từng Gaussian; ở đây minh hoạ bằng
        # cách áp dụng ngẫu nhiên cho phần "kém nhất quán multi-view" nhất
        scores_mask = np.random.default_rng(999).random(cloud.n) < 0.05
        prune_mask = prune_mask | scores_mask
    n_before = cloud.n
    cloud.keep(~prune_mask)
    return n_before - cloud.n


def maybe_reset_opacity(cloud: GaussianCloud, t: int, cfg: ADCConfig) -> bool:
    """công thức (1): tại t = 3000k, alpha_i <- alpha_i * opacity_reset_decay
    (nhân, khác 3DGS gốc dùng floor sigmoid^-1(min(alpha,0.01)))."""
    if t % cfg.opacity_reset_interval == 0:
        cloud.alpha = cloud.alpha * cfg.opacity_reset_decay
        return True
    return False


# ----------------------------------------------------------------------------
# Vòng mô phỏng
# ----------------------------------------------------------------------------
def rim_fraction(mu: np.ndarray, band: float = 0.05) -> float:
    dist = np.linalg.norm(mu - CIRCLE_C, axis=1)
    return float(np.mean(np.abs(dist - CIRCLE_R) < band))


def init_cloud(n0: int, rng: np.random.Generator, cfg: ADCConfig) -> GaussianCloud:
    mu = rng.uniform(0.0, 1.0, size=(n0, 2))
    scale = rng.uniform(0.005, 0.03, size=(n0, 2))
    rot = rng.uniform(0.0, 2 * np.pi, size=n0)
    alpha = np.full(n0, 0.1)                       # 3DGS/SADGS khởi tạo opacity = 0.1
    return GaussianCloud(mu=mu, scale=scale, rot=rot, alpha=alpha,
                         rate=new_rates(n0, rng, cfg))


def run_simulation(cfg: ADCConfig, seed: int = 0, n0: int = 300) -> dict:
    rng = np.random.default_rng(seed)
    cloud = init_cloud(n0, rng, cfg)
    n_hist = np.zeros(cfg.total_iters + 1, dtype=int)
    n_hist[0] = cloud.n
    snapshots = {}
    totals = {"clone": 0, "split": 0, "split_children": 0, "expand": 0,
              "prune": 0, "prune_mv": 0, "densify_calls": 0, "final_prune": 0}
    log = []
    rim0 = rim_fraction(cloud.mu)
    last_low_ratio_mean = 0.0

    for t in range(1, cfg.total_iters + 1):
        if t < cfg.densify_until_iter:
            g, vis = fake_positional_grad(cloud.mu, rng, cfg)
            eta_obs = eta_axis_observation(cloud.mu, cloud.scale, rng, cfg)
            accumulate_stats(cloud, eta_obs, vis, g, cfg)

        fake_optimizer_step(cloud, rng, cfg)

        if (t % cfg.densification_interval == 0
                and cfg.densify_from_iter < t < cfg.densify_until_iter):
            st = densify_and_prune_structgs(cloud, t, rng, cfg)
            for k_ in ("clone", "split", "split_children", "expand", "prune", "prune_mv"):
                totals[k_] += st[k_]
            totals["densify_calls"] += 1
            last_low_ratio_mean = st["low_ratio_mean"]
            log.append((t, st["clone"], st["split"], st["prune"]))

        n_hist[t] = cloud.n
        if t in cfg.snapshot_iters:
            snapshots[t] = cloud.copy()

        # reset opacity NHÂN, sau densify cùng vòng (như train.py) — chỉ khi còn
        # trong giai đoạn densify (train.py bọc reset_opacity trong
        # `if iteration < densify_until_iter`), nên KHÔNG reset đúng lúc t=15000
        if t < cfg.densify_until_iter:
            maybe_reset_opacity(cloud, t, cfg)

        if t == cfg.densify_until_iter:
            # minh hoạ final_prune_structgs (mục 7), 1 lần tại cuối giai đoạn densify
            totals["final_prune"] = final_prune_structgs(cloud, last_low_ratio_mean, cfg)
            n_hist[t] = cloud.n

    return {"cloud": cloud, "n_hist": n_hist, "snapshots": snapshots,
            "totals": totals, "log": log, "n0": n0, "rim0": rim0,
            "rim1": rim_fraction(cloud.mu)}


# ----------------------------------------------------------------------------
# Vẽ hình
# ----------------------------------------------------------------------------
def make_figure(res: dict, cfg: ADCConfig, path: str):
    n_hist = res["n_hist"]
    snaps = res["snapshots"]
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    # --- N(t) ---
    ax = fig.add_subplot(gs[0, :])
    ax.plot(np.arange(len(n_hist)), n_hist, color="#1f4e79", lw=1.6, label="N(t)")
    for i in range(cfg.opacity_reset_interval, cfg.total_iters, cfg.opacity_reset_interval):
        ax.axvline(i, color="#c0392b", ls="--", lw=1,
                   label="reset opacity ×0.1 (mỗi 3000)" if i == cfg.opacity_reset_interval else None)
    ax.axvline(cfg.densify_from_iter, color="#7f8c8d", ls=":", lw=1,
               label="densify_from_iter = 500")
    ax.axvline(cfg.densify_until_iter, color="#2e8b57", ls="-.", lw=1.3,
               label="densify_until_iter = 15000 (N đóng băng, final_prune)")
    ax.set_xlabel("vòng lặp t")
    ax.set_ylabel("số Gaussian N")
    tot = res["totals"]
    ax.set_title(f"N(t): {res['n0']} → {n_hist[-1]} Gaussian; "
                 f"{tot['densify_calls']} lần densify, clone {tot['clone']}, "
                 f"split {tot['split']}→{tot['split_children']} con, "
                 f"expand {tot['expand']}, prune {tot['prune']} (mv {tot['prune_mv']}), "
                 f"final_prune {tot['final_prune']}")
    ax.set_xlim(0, cfg.total_iters)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # --- 3 scatter, màu theo eta_max (kích thước Gaussian / bước sóng texture) ---
    plot_iters = [it for it in (500, 3000, 9000) if it in snaps]
    all_eta = np.concatenate([snaps[it].eta_max.max(axis=1) for it in plot_iters])
    vmin, vmax = 0.0, max(float(all_eta.max()), 1e-3)
    theta = np.linspace(0, 2 * np.pi, 200)
    sc = None
    for j, it in enumerate(plot_iters):
        axs = fig.add_subplot(gs[1, j])
        c = snaps[it]
        size = (c.max_scale() / 0.03) ** 2 * 60 + 2
        eta_scalar = c.eta_max.max(axis=1)
        sc = axs.scatter(c.mu[:, 0], c.mu[:, 1], s=size, c=eta_scalar, cmap="magma",
                         vmin=vmin, vmax=vmax, alpha=0.8, linewidths=0)
        axs.plot(CIRCLE_C[0] + CIRCLE_R * np.cos(theta), CIRCLE_C[1] + CIRCLE_R * np.sin(theta),
                 "c--", lw=1, label="vùng texture mịn (r = 0.3)")
        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1)
        axs.set_aspect("equal")
        axs.set_title(f"t = {it}: N = {c.n}, ở viền {100 * rim_fraction(c.mu):.0f}%")
        axs.set_xticks([0, 0.5, 1])
        axs.set_yticks([0, 0.5, 1])
        if j == 0:
            axs.legend(loc="lower left", fontsize=7)
    cb = fig.colorbar(sc, ax=fig.axes[1:], shrink=0.85, pad=0.02)
    cb.set_label("eta = kích thước Gaussian / bước sóng texture cục bộ (điểm to ∝ max(s))")

    fig.suptitle("Toy SADGS (Structure-Aware Densification): eta = extent/wavelength, "
                 "split dị hướng k=ceil(sqrt(eta)), expand_undersized_gs, "
                 "multiview-consistency clone/split/prune, reset opacity ×0.1/3000v",
                 fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    cfg = ADCConfig()
    res = run_simulation(cfg, seed=0, n0=300)
    make_figure(res, cfg, OUT_PNG)

    tot = res["totals"]
    n_hist = res["n_hist"]
    print("=== Toy SADGS (Structure-Aware Densification) ===")
    print(f"N ban đầu           : {res['n0']}")
    print(f"N cuối (t={cfg.total_iters}) : {n_hist[-1]}")
    print(f"N tại t=15000        : {n_hist[cfg.densify_until_iter]}  "
          f"(đứng yên sau đó: {np.all(n_hist[cfg.densify_until_iter:] == n_hist[-1])})")
    print(f"số lần densify       : {tot['densify_calls']}")
    print(f"tổng clone / split(số Gaussian) / con sinh ra / expand / prune (mv) / final_prune :")
    print(f"  {tot['clone']} / {tot['split']} / {tot['split_children']} / {tot['expand']} / "
          f"{tot['prune']} ({tot['prune_mv']}) / {tot['final_prune']}")
    print(f"tỉ lệ Gaussian ở băng viền |dist-0.3|<0.05 : đầu {100 * res['rim0']:.1f}%  "
          f"→ cuối {100 * res['rim1']:.1f}%")
    for it, c in sorted(res["snapshots"].items()):
        eta_scalar = c.eta_max.max(axis=1)
        print(f"  snapshot t={it:5d}: N={c.n:5d}, ở viền {100 * rim_fraction(c.mu):.1f}%, "
              f"alpha trung bình {c.alpha.mean():.3f}, eta trung bình {eta_scalar.mean():.3f}")
    print("Đã lưu:")
    print(f"  {OUT_PNG}")


if __name__ == "__main__":
    main()
