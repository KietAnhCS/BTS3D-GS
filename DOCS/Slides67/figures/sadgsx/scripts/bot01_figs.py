# -*- coding: utf-8 -*-
"""
BOT 01 — Structure tensor & multiscale image structure (SADGS).
Cài đặt lại bằng numpy/scipy ĐÚNG theo SADGS/utils/loss_utils.py:
  fast_gaussian_blur            (loss_utils.py:113)
  get_structure_tensor_torch    (loss_utils.py:170)
  get_multiscale_structure_tensor_v1 (loss_utils.py:232)
  get_multiscale_structure_tensor_v2 (loss_utils.py:315)
Output: DOCS/Slides67/figures/sadgsx/01_*.png
"""
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, convolve

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------
# 0. Ảnh test: texture nhiều tần số + cạnh cong + vùng phẳng
# ----------------------------------------------------------------------
def make_test_image(N=256):
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    img = np.full((N, N), 0.5)
    h = N // 2
    # Góc trên-trái: sọc dọc tần số CAO (chu kỳ 4 px)
    m = (x < h) & (y < h)
    img[m] = 0.5 + 0.35 * np.sin(2 * np.pi * x[m] / 4.0)
    # Góc trên-phải: sọc ngang tần số THẤP (chu kỳ 24 px)
    m = (x >= h) & (y < h)
    img[m] = 0.5 + 0.35 * np.sin(2 * np.pi * y[m] / 24.0)
    # Góc dưới-trái: sọc chéo 45 độ, chu kỳ 10 px
    m = (x < h) & (y >= h)
    img[m] = 0.5 + 0.35 * np.sin(2 * np.pi * (x[m] + y[m]) / 10.0)
    # Góc dưới-phải: vùng phẳng + 1 cung tròn (cấu trúc cong)
    r = np.sqrt((x - 0.78 * N) ** 2 + (y - 0.78 * N) ** 2)
    m = (x >= h) & (y >= h)
    img[m] = 0.55
    ring = m & (np.abs(r - 34) < 3)
    img[ring] = 0.05
    rng = np.random.default_rng(0)
    img = img + 0.01 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


IMG = make_test_image()
# (C, H, W) — Di Zenzo cộng năng lượng qua các kênh; ở đây dùng ảnh xám lặp 3 kênh
IMG3 = np.stack([IMG, IMG, IMG], axis=0)

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def fast_blur(img, sigma):
    """Tương đương fast_gaussian_blur (loss_utils.py:113) — bỏ nhánh downsample
    tăng tốc vì ở đây không cần tối ưu GPU; kết quả toán học như nhau."""
    if sigma < 0.01:
        return img
    if img.ndim == 3:
        return np.stack([gaussian_filter(c, sigma, mode="reflect") for c in img], 0)
    return gaussian_filter(img, sigma, mode="reflect")


def structure_tensor(img3, sigma=1.0, rho=1.0, normalize=True):
    """get_structure_tensor_torch (loss_utils.py:170)."""
    sm = fast_blur(img3, sigma)                       # :183
    Ixx = Ixy = Iyy = 0.0
    Ixs, Iys = [], []
    for c in range(sm.shape[0]):
        Ix = convolve(sm[c], SOBEL_X, mode="nearest")  # :195
        Iy = convolve(sm[c], SOBEL_Y, mode="nearest")  # :196
        Ixs.append(Ix); Iys.append(Iy)
        Ixx = Ixx + Ix * Ix                            # :199-:208 (sum qua kênh)
        Ixy = Ixy + Ix * Iy
        Iyy = Iyy + Iy * Iy
    Sxx = fast_blur(Ixx, rho)                          # :215
    Sxy = fast_blur(Ixy, rho)                          # :216
    Syy = fast_blur(Iyy, rho)                          # :217
    if normalize:                                      # :222-:227
        mx = (Sxx + Syy).max() + 1e-6
        Sxx, Sxy, Syy = Sxx / mx, Sxy / mx, Syy / mx
    return Sxx, Sxy, Syy, np.mean(Ixs, 0), np.mean(Iys, 0)


def eig2x2(Sxx, Sxy, Syy):
    tr = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    d = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0.0))
    l1 = tr / 2 + d           # freq_utils.py:329
    l2 = tr / 2 - d
    # vector riêng ứng với l1
    v1x = Sxy
    v1y = l1 - Sxx
    n = np.sqrt(v1x ** 2 + v1y ** 2) + 1e-12
    deg = np.abs(v1y) + np.abs(v1x) < 1e-12
    v1x = np.where(deg, 1.0, v1x / n)
    v1y = np.where(deg, 0.0, v1y / n)
    return l1, l2, v1x, v1y


def multiscale(img3, version, levels=4, base_sigma=1.0, octave_step=1.5,
               smoothing_factor=1.0, power_factor=3.0, keep_levels=False):
    """v1: loss_utils.py:232 / v2: loss_utils.py:315."""
    H, W = img3.shape[1:]
    if version == 1:
        bSxx, bSxy, bSyy, _, _ = structure_tensor(img3, sigma=base_sigma, rho=1.0)  # :241
    cur = img3
    aS = [np.zeros((H, W)) for _ in range(3)]
    aw = np.zeros((H, W))
    sigma_accum = 0.0
    per_level = []
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)          # :263 / :336
        target_sigma = base_sigma * (octave_step ** i)
        sigma_inc = math.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
        nxt = fast_blur(cur, sigma_inc)
        band = np.sqrt(((cur - nxt) ** 2).sum(0))      # DoG :270 / :344
        if i > 0 and smoothing_factor > 0:             # :273 / :347
            band = fast_blur(band, target_sigma * 2.0)
        rho_i = target_sigma * 3.0                     # :279 / :354
        if version == 1:
            Sxx_i, Sxy_i, Syy_i = (fast_blur(bSxx, rho_i),
                                   fast_blur(bSxy, rho_i),
                                   fast_blur(bSyy, rho_i))   # :282-:284
        else:
            Sxx_i, Sxy_i, Syy_i, _, _ = structure_tensor(nxt, sigma=target_sigma,
                                                         rho=rho_i)  # :357
        tr = Sxx_i + Syy_i + 1e-6
        Sn = [Sxx_i / tr, Sxy_i / tr, Syy_i / tr]      # :289-:291 / :364-:366
        w = band ** power_factor                       # :294 / :369
        f2 = band_freq ** 2                            # :298 / :372
        for k in range(3):
            aS[k] += Sn[k] * w * f2
        aw += w
        if keep_levels:
            per_level.append(dict(i=i, sigma=target_sigma, rho=rho_i,
                                  band_freq=band_freq, band=band.copy(),
                                  w=w.copy(), trace=(Sn[0] + Sn[2]) * w * f2))
        cur = nxt
        sigma_accum = target_sigma
    out = [a / (aw + 1e-6) for a in aS]                # :309-:311 / :383-:385
    return (out, per_level) if keep_levels else out


# ----------------------------------------------------------------------
# HÌNH 1 — ảnh test + Ix, Iy
# ----------------------------------------------------------------------
Sxx, Sxy, Syy, Ix, Iy = structure_tensor(IMG3, sigma=1.0, rho=1.0)
fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
ax[0].imshow(IMG, cmap="gray"); ax[0].set_title("Ảnh test I (4 vùng tần số)")
m = np.abs(Ix).max()
ax[1].imshow(Ix, cmap="coolwarm", vmin=-m, vmax=m)
ax[1].set_title(r"$I_x = S_x * (G_\sigma * I)$,  $\sigma=1$")
m = np.abs(Iy).max()
ax[2].imshow(Iy, cmap="coolwarm", vmin=-m, vmax=m)
ax[2].set_title(r"$I_y = S_y * (G_\sigma * I)$")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Bước 1: làm mượt trước rồi lấy đạo hàm Sobel (loss_utils.py:183-196)", y=1.02)
save(fig, "01_gradient_ix_iy.png")

# ----------------------------------------------------------------------
# HÌNH 2 — 3 kênh tensor cấu trúc Sxx, Sxy, Syy
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
for a, d, t in zip(ax, [Sxx, Sxy, Syy],
                   [r"$S_{xx}=G_\rho * I_x^2$", r"$S_{xy}=G_\rho * I_xI_y$",
                    r"$S_{yy}=G_\rho * I_y^2$"]):
    v = np.abs(d).max()
    im = a.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
    a.set_title(t); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(r"$J_\rho = G_\rho * \nabla I\,\nabla I^{\top}$ — chuẩn hoá bởi $\max(S_{xx}+S_{yy})$ (loss_utils.py:222-227)", y=1.03)
save(fig, "01_structure_tensor_channels.png")

# ----------------------------------------------------------------------
# HÌNH 3 — lambda1, lambda2, coherence
# ----------------------------------------------------------------------
l1, l2, vx, vy = eig2x2(Sxx, Sxy, Syy)
coh = ((l1 - l2) / (l1 + l2 + 1e-9)) ** 2
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2))
for a, d, t, cm in zip(ax, [l1, l2, coh],
                       [r"$\lambda_1$ (năng lượng tần số cao)",
                        r"$\lambda_2$",
                        r"coherence $\left(\frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2}\right)^2$"],
                       ["magma", "magma", "viridis"]):
    im = a.imshow(d, cmap=cm)
    a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(r"$\lambda_{1,2}=\frac{tr}{2}\pm\sqrt{(\frac{tr}{2})^2-\det}$ — SADGS dùng $\lambda_1$ làm ngưỡng bước sóng $w_{\min}=1/\sqrt{\lambda_1}$ (freq_utils.py:325-334)", y=1.04, fontsize=9)
save(fig, "01_eigen_coherence.png")

# ----------------------------------------------------------------------
# HÌNH 4 — quiver hướng vector riêng
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.2))
ax[0].imshow(IMG, cmap="gray")
s = 12
ys, xs = np.mgrid[6:256:s, 6:256:s]
L = np.sqrt(l1[ys, xs]); L = L / (L.max() + 1e-9)
# v1 = hướng gradient trội; vuông góc = hướng texture/cạnh
ax[0].quiver(xs, ys, vx[ys, xs] * L, vy[ys, xs] * L, color="red",
             scale=18, width=0.004)
ax[0].set_title(r"$v_1$ — hướng biến thiên mạnh nhất (pháp tuyến cạnh)", fontsize=9)
ax[1].imshow(IMG, cmap="gray")
ax[1].quiver(xs, ys, -vy[ys, xs] * L, vx[ys, xs] * L, color="lime",
             scale=18, width=0.004)
ax[1].set_title(r"$v_2\perp v_1$ — hướng texture (cạnh chạy dọc theo)", fontsize=9)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Vector riêng của J: độ dài mũi tên ~ $\\sqrt{\\lambda_1}$", y=0.99, fontsize=10)
save(fig, "01_eigenvector_quiver.png")

# ----------------------------------------------------------------------
# HÌNH 5 — kim tự tháp đa tỉ lệ (v2), từng level
# ----------------------------------------------------------------------
LV = 4
outv2, per = multiscale(IMG3, 2, levels=LV, keep_levels=True)
fig, ax = plt.subplots(3, LV, figsize=(10.5, 7.2))
for i, d in enumerate(per):
    ax[0, i].imshow(d["band"], cmap="inferno")
    ax[0, i].set_title(f"i={i}: DoG band\n$\\sigma$={d['sigma']:.2f}, $\\rho$={d['rho']:.2f}",
                       fontsize=8)
    ax[1, i].imshow(d["w"], cmap="inferno")
    ax[1, i].set_title(r"$w_i=\mathrm{band}^{p}$, $p$=3", fontsize=8)
    ax[2, i].imshow(d["trace"], cmap="magma")
    ax[2, i].set_title(f"đóng góp $\\cdot f_i^2$, $f_i$={d['band_freq']:.3f}", fontsize=8)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Kim tự tháp v2: levels=4, base_sigma=1, octave_step=1.5, power_factor=3, smoothing_factor=1\n"
             r"$\sigma_i=\sigma_0 s^i$,  $f_i=s^{-i}$,  $\rho_i=3\sigma_i$", y=1.0, fontsize=10)
save(fig, "01_pyramid_levels.png")

# ----------------------------------------------------------------------
# HÌNH 6 — v1 vs v2
# ----------------------------------------------------------------------
o1 = multiscale(IMG3, 1, levels=LV)
o2 = outv2
t1, t2 = o1[0] + o1[2], o2[0] + o2[2]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
vmax = max(t1.max(), t2.max())
im = ax[0].imshow(t1, cmap="magma", vmin=0, vmax=vmax)
ax[0].set_title("v1: J tính 1 lần ở base_sigma\nrồi blur theo $\\rho_i$ (loss_utils.py:241,282)", fontsize=8)
fig.colorbar(im, ax=ax[0], fraction=0.046)
im = ax[1].imshow(t2, cmap="magma", vmin=0, vmax=vmax)
ax[1].set_title("v2: J tính LẠI trên ảnh đã blur\nở từng level (loss_utils.py:357)", fontsize=8)
fig.colorbar(im, ax=ax[1], fraction=0.046)
im = ax[2].imshow(t2 - t1, cmap="coolwarm",
                  vmin=-np.abs(t2 - t1).max(), vmax=np.abs(t2 - t1).max())
ax[2].set_title("v2 $-$ v1", fontsize=9)
fig.colorbar(im, ax=ax[2], fraction=0.046)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle(r"So sánh trace $S_{xx}+S_{yy}$ của st_map (levels=4). Mặc định train: st_mode='v1' (arguments/__init__.py:121)",
             y=1.04, fontsize=9)
save(fig, "01_v1_vs_v2.png")

# ----------------------------------------------------------------------
# HÌNH 7 — ảnh hưởng power_factor & octave_step
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
b = np.linspace(0, 1, 300)
for p in [1.0, 2.0, 3.0, 4.0, 6.0]:
    ax[0].plot(b, b ** p, label=f"p={p:g}")
ax[0].set_xlabel("band response $b_i$"); ax[0].set_ylabel(r"trọng số $w_i=b_i^{\,p}$")
ax[0].set_title("power_factor: p càng lớn càng\n'winner-take-all' theo thang trội", fontsize=9)
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

ii = np.arange(0, 6)
for s in [1.2, 1.5, 2.0, 2.5]:
    ax[1].plot(ii, s ** (-2.0 * ii), "o-", label=f"octave_step={s:g}")
ax[1].set_xlabel("level i"); ax[1].set_ylabel(r"$f_i^2=s^{-2i}$")
ax[1].set_yscale("log"); ax[1].set_title("octave_step: hệ số tần số$^2$ nhân vào\nmỗi level (loss_utils.py:263,298)", fontsize=9)
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

for s in [1.2, 1.5, 2.0, 2.5]:
    ax[2].plot(ii, 1.0 * s ** ii, "s-", label=f"s={s:g}")
ax[2].plot(ii, 3.0 * 1.5 ** ii, "k--", label=r"$\rho_i=3\sigma_i$ (s=1.5)")
ax[2].set_xlabel("level i"); ax[2].set_ylabel(r"$\sigma_i=\sigma_0 s^i$ (px)")
ax[2].set_title("base_sigma & octave_step: dải $\\sigma$ quét\n$\\rho_i=3\\sigma_i$ để 'thấy' độ cong", fontsize=9)
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
save(fig, "01_power_octave_effect.png")

# ----------------------------------------------------------------------
# HÌNH 8 — power_factor tác động lên st_map thật
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.1))
for a, p in zip(ax, [1.0, 2.0, 3.0, 6.0]):
    o = multiscale(IMG3, 2, levels=LV, power_factor=p)
    tr = o[0] + o[2]
    im = a.imshow(tr, cmap="magma")
    a.set_title(f"power_factor = {p:g}", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(r"trace của st_map (v2, levels=4) khi đổi power_factor — vùng tần số cao được ưu tiên mạnh dần",
             y=1.05, fontsize=9)
save(fig, "01_power_factor_stmap.png")

print("DONE")
