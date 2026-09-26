   # -*- coding: utf-8 -*-
"""
BOT 12 — Chon mau camera (FPS sampling) & lay mau Gaussian di huong 2D.
Nguon code that:
  SADGS/utils/freq_utils.py : sampling_cameras (12-91)
  SADGS/utils/gaussian_sampling.py : sample_anisotropic_gaussians_2d (11-128)
  SADGS/utils/loss_utils.py : get_structure_tensor_torch (170-...)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.abspath(OUT)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


# ---------------------------------------------------------------- data camera
def make_cameras(n=220, seed=7):
    """Quy dao camera kieu 'vong quanh doi tuong' + cum day dac (nhu COLMAP)."""
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    # 70% cameras don vao cung 1/3 quy dao -> mo phong lay mau khong deu
    t = np.where(rng.random(n) < 0.7, 0.15 + 0.30 * rng.random(n), t)
    ang = 2 * np.pi * t
    r = 4.0 + 0.25 * rng.standard_normal(n)
    x = r * np.cos(ang)
    y = r * np.sin(ang)
    return np.stack([x, y], axis=1)


def fps(P, k, start=0):
    """Farthest Point Sampling — ban sao 1:1 vong lap freq_utils.py:52-72.
    Metric = chuan L2 tren VI TRI camera (khong dung huong nhin)."""
    N = len(P)
    k = min(k, N)
    sel = [start]
    d = np.full(N, np.inf)
    cur = start
    for _ in range(k - 1):
        d = np.minimum(d, np.linalg.norm(P - P[cur], axis=1))
        d[sel] = -1.0
        cur = int(np.argmax(d))
        sel.append(cur)
    return np.array(sel)


def coverage(P, sel):
    """Khoang cach phu: max_i min_{j in sel} ||p_i - p_j|| (cang nho cang phu tot)."""
    D = np.linalg.norm(P[:, None, :] - P[None, sel, :], axis=2)
    return D.min(axis=1).max()


# ---------------------------------------------------------------- FIG 1
def fig1(P):
    k = 24
    sel = fps(P, k, start=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(P[:, 0], P[:, 1], s=14, c="#c9ccd1", edgecolors="none",
               label="Camera chua chon (N=%d)" % len(P), zorder=1)
    order = np.arange(len(sel))
    sc = ax.scatter(P[sel, 0], P[sel, 1], c=order, cmap="viridis", s=90,
                    edgecolors="k", linewidths=0.6, zorder=3)
    for i, idx in enumerate(sel):
        ax.annotate(str(i + 1), (P[idx, 0], P[idx, 1]), fontsize=6.5,
                    color="white", ha="center", va="center", zorder=4)
    ax.scatter([P[sel[0], 0]], [P[sel[0], 1]], marker="*", s=320,
               facecolor="none", edgecolor="crimson", linewidths=1.6, zorder=5,
               label="Hat giong ngau nhien (buoc 1)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Thu tu duoc chon (1..%d)" % k, fontsize=9)
    ax.set_title(u"FPS tren vi tri camera $c_i=-R^{\\top}t$\n"
                 u"$i_{k}=\\arg\\max_i\\ \\min_{j<k}\\ \\|c_i-c_{i_j}\\|_2$",
                 fontsize=10)
    ax.set_xlabel("x (world)"); ax.set_ylabel("y (world)")
    ax.legend(fontsize=7.5, loc="center", framealpha=0.95)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    save(fig, "12_fps_selection.png")


# ---------------------------------------------------------------- FIG 2
def fig2(P):
    ks = np.arange(2, 81, 2)
    cov_fps, cov_rnd_m, cov_rnd_lo, cov_rnd_hi = [], [], [], []
    rng = np.random.default_rng(1)
    for k in ks:
        cov_fps.append(coverage(P, fps(P, k, start=0)))
        vals = [coverage(P, rng.choice(len(P), size=k, replace=False))
                for _ in range(40)]
        cov_rnd_m.append(np.mean(vals))
        cov_rnd_lo.append(np.percentile(vals, 10))
        cov_rnd_hi.append(np.percentile(vals, 90))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ks, cov_fps, "-o", ms=3, color="#1f77b4", label='mode="fps"')
    ax.plot(ks, cov_rnd_m, "-s", ms=3, color="#d62728", label='mode="random" (trung binh 40 lan)')
    ax.fill_between(ks, cov_rnd_lo, cov_rnd_hi, color="#d62728", alpha=0.18,
                    label="random: bach phan vi 10–90%")
    ax.set_xlabel("num_cams (so camera duoc chon)")
    ax.set_ylabel(r"Ban kinh phu $\max_i \min_{j\in S}\|c_i-c_j\|$")
    ax.set_title(u"FPS phu khong gian camera tot hon random\n(cang thap cang tot)",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, "12_fps_vs_random_coverage.png")


# ---------------------------------------------------------------- FIG 3
def fig3():
    N = np.arange(20, 1001, 20)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for k_lab, kfun, c in [("num_cams = 60 (mac dinh)", lambda n: np.minimum(n, 60), "#2ca02c"),
                           ("num_cams = N (train.py:224,240)", lambda n: n, "#d62728")]:
        k = kfun(N).astype(float)
        ax.plot(N, N * k, label=k_lab + r"  $\Rightarrow O(Nk)$", color=c)
    ax.plot(N, N.astype(float), "--", color="#7f7f7f", label=r"random: $O(N)$")
    ax.set_yscale("log")
    ax.set_xlabel("N = so camera train trong stack")
    ax.set_ylabel(u"So phep tinh khoang cach (log)")
    ax.set_title(u"Chi phi FPS $= O(N\\cdot num\\_cams)$ phep tinh $\\|c_i-c_j\\|$\n"
                 u"lay tap con (k=60) re hon nhieu so voi k=N", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    save(fig, "12_fps_cost.png")


# ------------------------------------------------- structure tensor (numpy)
def gauss_kernel1d(sigma):
    ks = int(2 * 4 * sigma + 1)
    if ks % 2 == 0:
        ks += 1
    x = np.arange(ks) - ks // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def blur(img, sigma):
    g = gauss_kernel1d(sigma)
    p = len(g) // 2
    a = np.pad(img, p, mode="reflect")
    a = np.apply_along_axis(lambda m: np.convolve(m, g, mode="valid"), 1, a)
    a = np.apply_along_axis(lambda m: np.convolve(m, g, mode="valid"), 0, a)
    return a


def structure_tensor(img, sigma=1.0, rho=1.5):
    """Mo phong get_structure_tensor_torch: blur sigma -> Sobel -> tich -> blur rho."""
    sm = blur(img, sigma)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    sy = sx.T
    def conv(a, k):
        a = np.pad(a, 1, mode="reflect")
        o = np.zeros_like(img)
        for i in range(3):
            for j in range(3):
                o += k[i, j] * a[i:i + img.shape[0], j:j + img.shape[1]]
        return o
    Ix, Iy = conv(sm, sx), conv(sm, sy)
    return blur(Ix * Ix, rho), blur(Ix * Iy, rho), blur(Iy * Iy, rho)


def test_image(H=200, W=200):
    """Anh test co texture DI HUONG: soc doc, soc ngang, soc xien, vung phang."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 0.5)
    img[:H // 2, :W // 2] = 0.5 + 0.45 * np.sin(2 * np.pi * x[:H // 2, :W // 2] / 9.0)   # soc doc
    img[:H // 2, W // 2:] = 0.5 + 0.45 * np.sin(2 * np.pi * y[:H // 2, W // 2:] / 9.0)   # soc ngang
    d = (x[H // 2:, :W // 2] + y[H // 2:, :W // 2])
    img[H // 2:, :W // 2] = 0.5 + 0.45 * np.sin(2 * np.pi * d / 13.0)                    # soc xien
    img[H // 2:, W // 2:] = 0.5                                                          # phang
    return np.clip(img, 0, 1)


def sample_st(img, n, seed, sigma=1.0, rho=1.5):
    """Ban sao logic sample_anisotropic_gaussians_2d (gaussian_sampling.py:34-119)."""
    H, W = img.shape
    Sxx, Sxy, Syy = structure_tensor(img, sigma, rho)
    E = Sxx + Syy                     # energy_map = trace(S)  (dong 43)
    f = E.flatten()
    s = f.sum()
    p = f / s if s > 0 else np.ones_like(f) / f.size
    rng = np.random.default_rng(seed)
    idx = rng.choice(H * W, size=n, p=p, replace=False)   # dong 57
    yy, xx = np.unravel_index(idx, (H, W))
    eps = 1e-6
    a = Sxx[yy, xx] + eps; b = Sxy[yy, xx]; c = Syy[yy, xx] + eps
    tr = a + c
    det = a * c - b * b
    dl = np.sqrt(np.maximum((tr / 2) ** 2 - det, 0))
    l1 = tr / 2 + dl          # lon nhat -> huong GRADIENT
    l2 = tr / 2 - dl          # nho nhat -> huong CANH (edge)
    # vector rieng ung voi l2 (v2), goc canh = atan2(v2y, v2x)  (dong 108-113)
    vx, vy = b, l2 - a
    nz = np.hypot(vx, vy) < 1e-12
    vx = np.where(nz, 1.0, vx); vy = np.where(nz, 0.0, vy)
    nrm = np.hypot(vx, vy)
    v2 = np.stack([vx / nrm, vy / nrm], axis=1)
    ang = np.degrees(np.arctan2(v2[:, 1], v2[:, 0]))
    return np.stack([xx, yy], 1).astype(float), l1, l2, v2, ang, E


# ---------------------------------------------------------------- FIG 4
def fig4():
    img = test_image()
    coords, l1, l2, v2, ang, E = sample_st(img, 260, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    axes[0].imshow(img, cmap="gray", origin="upper")
    axes[0].set_title(u"Anh test di huong (soc doc / ngang / xien / phang)", fontsize=9.5)
    axes[1].imshow(img, cmap="gray", origin="upper", alpha=0.55)
    # Ellipse: truc dai doc theo v2 (canh), ti le 1/sqrt(l) -> texture manh => ellipse mong
    s1 = 1.0 / np.sqrt(l1 + 1e-6)
    s2 = 1.0 / np.sqrt(l2 + 1e-6)
    k = 22.0 / (np.median(s2) + 1e-9)
    for (cx, cy), a1, a2, th in zip(coords, s1, s2, ang):
        w = np.clip(k * a2, 2, 26)   # doc theo canh (l2 nho -> dai)
        h = np.clip(k * a1, 1, 26)   # doc theo gradient (l1 lon -> ngan)
        axes[1].add_patch(Ellipse((cx, cy), w, h, angle=th, facecolor="none",
                                  edgecolor="#ff7f0e", lw=0.7, alpha=0.9))
    axes[1].set_title(u"Ellipse $1\\sigma$: truc $\\propto 1/\\sqrt{\\lambda_{1,2}}$, "
                      u"xoay theo $v_2$\n(sigma=1.0, rho=1.5, seed=42)", fontsize=9.5)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(u"sample_anisotropic_gaussians_2d:  $S=[[S_{xx},S_{xy}],[S_{xy},S_{yy}]]$,"
                 u"  $\\lambda_{1,2}=\\frac{tr}{2}\\pm\\sqrt{(\\frac{tr}{2})^2-\\det S}$",
                 fontsize=10)
    save(fig, "12_aniso_ellipses.png")


# ---------------------------------------------------------------- FIG 5
def fig5():
    img = test_image()
    n = 900
    coords, l1, l2, v2, ang, E = sample_st(img, n, seed=42)
    rng = np.random.default_rng(0)
    H, W = img.shape
    uni = np.stack([rng.integers(0, W, n), rng.integers(0, H, n)], 1).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    axes[0].imshow(img, cmap="gray"); axes[0].scatter(uni[:, 0], uni[:, 1], s=2,
                                                      c="#d62728")
    axes[0].set_title(u"Lay mau DEU (uniform)\n%d diem" % n, fontsize=9.5)
    axes[1].imshow(img, cmap="gray"); axes[1].scatter(coords[:, 0], coords[:, 1],
                                                      s=2, c="#1f77b4")
    axes[1].set_title(u"Theo structure tensor\n$p(x,y)=\\frac{S_{xx}+S_{yy}}"
                      u"{\\sum(S_{xx}+S_{yy})}$", fontsize=9.5)
    im = axes[2].imshow(E, cmap="magma")
    axes[2].set_title(u"Ban do nang luong $\\mathrm{tr}(S)=S_{xx}+S_{yy}$", fontsize=9.5)
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])

    # ti le diem roi vao vung phang (goc duoi-phai)
    flat = lambda C: np.mean((C[:, 0] >= W / 2) & (C[:, 1] >= H / 2)) * 100
    fig.suptitle(u"Ty le diem roi vao vung PHANG (vo ich): deu = %.1f%%  vs  "
                 u"structure tensor = %.1f%%" % (flat(uni), flat(coords)), fontsize=10)
    save(fig, "12_uniform_vs_st.png")


if __name__ == "__main__":
    P = make_cameras()
    fig1(P); fig2(P); fig3(); fig4(); fig5()
    print("DONE")
