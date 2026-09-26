# -*- coding: utf-8 -*-
# BOT 14: Rasterizer rieng cua SADGS (diff-gaussian-rasterization_structgs & render_structgs)
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.abspath(OUT)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------
# 1) Alpha blending tich luy + duong cong transmittance theo depth
# ---------------------------------------------------------------
def fig_alpha_blending():
    rng = np.random.default_rng(7)
    N = 24
    depth = np.arange(1, N + 1)
    alpha = np.clip(rng.uniform(0.05, 0.45, N), 1.0 / 255.0, 0.99)

    T = np.ones(N + 1)
    for i in range(N):
        T[i + 1] = T[i] * (1.0 - alpha[i])
    w = alpha * T[:N]          # trong so dong gop alpha_i * T_i
    Cacc = np.cumsum(w)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))

    a = ax[0]
    a.bar(depth, w, color="#4C78A8", label=r"$w_i=\alpha_i T_i$ (dong gop)")
    a.plot(depth, Cacc, "o-", color="#E45756", lw=1.8, ms=3.5,
           label=r"$\sum_{j\leq i}\alpha_j T_j$ (mau tich luy)")
    a.axhline(1.0, color="gray", ls=":", lw=1)
    a.set_xlabel("thu tu Gaussian theo depth (da sort trong tile)")
    a.set_ylabel("gia tri")
    a.set_title(r"Alpha blending: $C=\sum_i c_i\,\alpha_i\,T_i$", fontsize=10)
    a.legend(fontsize=7.5, loc="center right")
    a.grid(alpha=0.25)

    b = ax[1]
    b.step(np.arange(N + 1), T, where="post", color="#54A24B", lw=2,
           label=r"$T_i=\prod_{j<i}(1-\alpha_j)$")
    b.axhline(1e-4, color="#E45756", ls="--", lw=1.4,
              label=r"early stop: $T<10^{-4}$")
    b.axhline(0.0, color="gray", lw=0.6)
    # diem dung som
    idx = np.argmax(T < 1e-4) if (T < 1e-4).any() else N
    Tmax = T.max()
    b.annotate(r"$T^{\max}$ ghi vao cov2D[:,6]" + "\n(atomicMax, forward.cu:470)",
               xy=(0.3, Tmax), xytext=(6.0, 0.35), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", color="#333333", lw=1))
    b.set_yscale("log")
    b.set_ylim(1e-5, 2)
    b.set_xlabel("thu tu Gaussian theo depth")
    b.set_ylabel("transmittance T (log)")
    b.set_title("Duong cong transmittance tich luy", fontsize=10)
    b.legend(fontsize=7.5, loc="lower left")
    b.grid(alpha=0.25, which="both")

    fig.tight_layout()
    save(fig, "14_alpha_blending_transmittance.png")


# ---------------------------------------------------------------
# 2) Vung anh huong: 3-sigma (3DGS goc) vs compact box mult*t (SADGS)
# ---------------------------------------------------------------
def make_conic(sx, sy, rho):
    cov = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])
    inv = np.linalg.inv(cov)
    return cov, inv


def ellipse_from_t(inv, t):
    """Tap {d : d^T inv d = t}. Tra ve (width, height, angle_deg) cho Ellipse."""
    evals, evecs = np.linalg.eigh(inv)
    # ban truc = sqrt(t / lambda)
    ax_len = np.sqrt(t / evals)
    ang = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    return 2 * ax_len[1], 2 * ax_len[0], ang


def fig_mult_region():
    sx, sy, rho, op = 16.0, 8.0, 0.55, 0.9
    cov, inv = make_conic(sx, sy, rho)
    t0 = 2.0 * np.log(op * 255.0)          # auxiliary.h:332
    mults = [0.4, 0.7, 1.0, 1.5]
    colors = ["#54A24B", "#4C78A8", "#F58518", "#B279A2"]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))

    a = ax[0]
    # luoi tile 16x16
    B = 16
    for k in range(-5, 6):
        a.axhline(k * B, color="#dddddd", lw=0.7, zorder=0)
        a.axvline(k * B, color="#dddddd", lw=0.7, zorder=0)
    # 3-sigma cua 3DGS goc (radius = 3*sqrt(lambda_max))
    lam_max = np.linalg.eigvalsh(cov).max()
    r3 = 3.0 * np.sqrt(lam_max)
    a.add_patch(Rectangle((-r3, -r3), 2 * r3, 2 * r3, fill=False, ec="#E45756",
                          lw=2.0, ls="--",
                          label=r"3DGS goc: AABB $r=3\sqrt{\lambda_{\max}}$"))
    for m, c in zip(mults, colors):
        w, h, an = ellipse_from_t(inv, m * t0)
        a.add_patch(Ellipse((0, 0), w, h, angle=an, fill=False, ec=c, lw=1.8,
                            label=f"SADGS mult={m}"))
    a.set_xlim(-100, 100); a.set_ylim(-72, 72)
    a.set_aspect("equal")
    a.set_xlabel("pixel x"); a.set_ylabel("pixel y")
    a.set_title(r"Vung anh huong: $d^\top\Sigma_{2D}^{-1}d \leq \mathrm{mult}\cdot 2\ln(255\sigma_o)$",
                fontsize=9)
    a.legend(fontsize=7, loc="upper right")

    # so tile phai xu ly theo mult (dem that tren luoi 16x16)
    b = ax[1]
    mrange = np.linspace(0.2, 2.0, 40)
    ntiles, ncircle = [], []
    gx = np.arange(-10, 10)
    for m in mrange:
        t = m * t0
        cnt = 0
        for i in gx:
            for j in gx:
                # tam tile
                px = np.linspace(i * B, (i + 1) * B, 6)
                py = np.linspace(j * B, (j + 1) * B, 6)
                X, Y = np.meshgrid(px, py)
                D = inv[0, 0] * X * X + 2 * inv[0, 1] * X * Y + inv[1, 1] * Y * Y
                if (D <= t).any():
                    cnt += 1
        ntiles.append(cnt)
        ncircle.append(((2 * r3) // B + 2) ** 2)
    b.plot(mrange, ntiles, "-", color="#4C78A8", lw=2, label="SADGS: compact box (ellipse)")
    b.plot(mrange, ncircle, "--", color="#E45756", lw=1.8, label=r"3DGS goc: AABB $3\sigma$")
    b.axvline(0.7, color="#333333", ls=":", lw=1.5)
    b.text(0.73, max(ncircle) * 0.55, "mac dinh\nopt.mult = 0.7", fontsize=8)
    b.set_xlabel("mult (box multiplier)")
    b.set_ylabel("so tile 16x16 phai xu ly / splat")
    b.set_title("mult nho -> it tile -> nhanh hon", fontsize=10)
    b.legend(fontsize=7.5)
    b.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "14_mult_vs_3sigma.png")


# ---------------------------------------------------------------
# 3) Chi phi render mo phong theo mult (tradeoff toc do / chat luong)
# ---------------------------------------------------------------
def fig_cost_vs_mult():
    m = np.linspace(0.2, 2.0, 120)
    # so tile ~ dien tich ellipse ~ t = mult*t0  => A ~ mult  (cong them vien tile)
    A = m                                  # dien tich chuan hoa
    ntile = (np.sqrt(A) * 3.2 + 1.0) ** 2  # (span + vien)^2
    cost = ntile / ((np.sqrt(1.0) * 3.2 + 1.0) ** 2)   # chuan hoa tai mult=1

    # sai so cat duoi: alpha bo qua tai bien = exp(-t/2) * sigma_o
    op = 0.9
    t0 = 2.0 * np.log(op * 255.0)
    alpha_cut = op * np.exp(-m * t0 / 2.0)     # alpha tai bien box
    rel = alpha_cut / (1.0 / 255.0)            # so voi nguong 1/255 cua rasterizer

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(m, cost, color="#4C78A8", lw=2.2, label="Chi phi render (chuan hoa, mult=1)")
    ax1.set_xlabel("mult (box multiplier)")
    ax1.set_ylabel("Chi phi tuong doi (so tile/splat)", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(m, rel, color="#E45756", lw=2.2, ls="--",
             label=r"$\alpha$ bi cat tai bien / (1/255)")
    ax2.axhline(1.0, color="#54A24B", ls=":", lw=1.6)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"sai so cat $\alpha_{bien}$ so voi nguong $1/255$", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")

    ax1.axvline(0.7, color="#333333", ls=":", lw=1.5)
    ax1.annotate("opt.mult = 0.7\n(arguments/__init__.py:114)", xy=(0.7, cost[np.argmin(abs(m - 0.7))]),
                 xytext=(1.05, 0.62), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#333333"))
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")
    ax1.set_title(r"Danh doi toc do / chat luong: $t=\mathrm{mult}\cdot 2\ln(255\,\sigma_o)$", fontsize=10)
    fig.tight_layout()
    save(fig, "14_cost_vs_mult.png")


# ---------------------------------------------------------------
# 4) So do tile 16x16 voi footprint Gaussian (AccuTile per-slice)
# ---------------------------------------------------------------
def fig_tile_grid():
    B = 16
    GX, GY = 13, 9
    sx, sy, rho, op = 22.0, 9.0, 0.6, 0.9
    cov, inv = make_conic(sx, sy, rho)
    t0 = 2.0 * np.log(op * 255.0)
    t = 0.7 * t0
    cx, cy = GX * B / 2, GY * B / 2

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    lam_max = np.linalg.eigvalsh(cov).max()
    r3 = 3.0 * np.sqrt(lam_max)

    touched_box = 0
    touched_aabb = 0
    for i in range(GX):
        for j in range(GY):
            x0, y0 = i * B, j * B
            px = np.linspace(x0, x0 + B, 8) - cx
            py = np.linspace(y0, y0 + B, 8) - cy
            X, Y = np.meshgrid(px, py)
            D = inv[0, 0] * X * X + 2 * inv[0, 1] * X * Y + inv[1, 1] * Y * Y
            inb = (D <= t).any()
            # tile giao voi AABB [-r3,r3]^2 quanh tam
            inaabb = (X.min() <= r3 and X.max() >= -r3 and
                      Y.min() <= r3 and Y.max() >= -r3)
            fc = "none"
            if inaabb:
                touched_aabb += 1
                fc = "#FFE9E6"
            if inb:
                touched_box += 1
                fc = "#CFE3F5"
            ax.add_patch(Rectangle((x0, y0), B, B, facecolor=fc,
                                   edgecolor="#999999", lw=0.8))

    w, h, an = ellipse_from_t(inv, t)
    ax.add_patch(Ellipse((cx, cy), w, h, angle=an, fill=False, ec="#4C78A8", lw=2.4))
    ax.add_patch(Rectangle((cx - r3, cy - r3), 2 * r3, 2 * r3, fill=False,
                           ec="#E45756", lw=1.8, ls="--"))
    ax.set_xlim(0, GX * B); ax.set_ylim(-34, GY * B)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, GX * B + 1, B))
    ax.set_yticks(np.arange(0, GY * B + 1, B))
    ax.set_xlabel("pixel x  (BLOCK_X = 16)")
    ax.set_ylabel("pixel y  (BLOCK_Y = 16)")
    ax.set_title("Tile 16x16 (config.h:16-17): xanh = %d tile SADGS thuc su xu ly,\n"
                 "hong = %d tile thua do AABB 3-sigma cua 3DGS goc" %
                 (touched_box, max(0, touched_aabb - touched_box)),
                 fontsize=9.5)
    # legend thu cong
    ax.add_patch(Rectangle((4, -16), 11, 8, facecolor="#CFE3F5", edgecolor="#999",
                           clip_on=False))
    ax.text(18, -14, "duplicateToTilesTouched (compact box, mult=0.7)", fontsize=7.5)
    ax.add_patch(Rectangle((4, -30), 11, 8, facecolor="#FFE9E6", edgecolor="#999",
                           clip_on=False))
    ax.text(18, -28, "tile thua cua AABB 3-sigma (3DGS goc)", fontsize=7.5)
    fig.tight_layout()
    save(fig, "14_tile_grid_footprint.png")


# ---------------------------------------------------------------
# 5) Luong du lieu GPU -> CPU cua cov2D
# ---------------------------------------------------------------
def fig_cov2d_flow():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=8):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#555555",
                               lw=1.1, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, zorder=3)

    def arrow(x1, y1, x2, y2, txt="", fs=7.5, dy=0.12):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color="#333333", lw=1.3,
                                     zorder=1))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, txt, ha="center",
                    fontsize=fs, color="#333333")

    ax.text(0.5, 3.62, "GPU (CUDA)", fontsize=10, weight="bold", color="#4C78A8")
    ax.text(6.55, 3.62, "CPU / PyTorch", fontsize=10, weight="bold", color="#54A24B")
    ax.plot([6.35, 6.35], [-0.1, 3.5], color="#bbbbbb", ls="--", lw=1.2)

    box(0.2, 2.85, 2.5, 0.6, "preprocessCUDA\ncomputeCov2D (EWA)", "#DCE9F7")
    box(0.2, 1.95, 2.5, 0.6, "cov2Ds[7i+0..2] = $\\Sigma_{2D}$\n[7i+3,4] = mean pixel\n[7i+5] = depth", "#DCE9F7", 7.5)
    box(0.2, 0.85, 2.5, 0.75, "renderCUDA:\natomicMax(cov2Ds[7i+6], T)\n-> $T^{\\max}$ tren moi pixel", "#DCE9F7", 7.5)
    box(3.35, 1.85, 2.6, 0.8, "torch::Tensor cov2D\n{P, 7} (float32, cuda)\nrasterize_points.cu:92", "#EAF4E4", 7.5)
    box(6.7, 2.7, 3.0, 0.75, "render_pkg[\"cov2D\"]\ngaussian_renderer/__init__.py:124", "#E6F2DF", 7.5)
    box(6.7, 1.5, 3.0, 0.9, "update_freq_stats_online()\nfreq_utils.py:181\nloc: $T^{\\max}>\\tau_T$ & $\\sigma_o>\\tau_o$", "#E6F2DF", 7.5)
    box(6.7, 0.25, 3.0, 0.85, "Cholesky jitter tu $\\Sigma_{2D}$\n-> tra cuu structure tensor\n-> $\\eta$ (freq statistics)", "#E6F2DF", 7.5)

    arrow(1.45, 2.85, 1.45, 2.58)
    arrow(1.45, 1.95, 1.45, 1.62)
    arrow(2.75, 2.25, 3.35, 2.25)
    arrow(2.75, 1.22, 3.35, 2.05)
    arrow(5.95, 2.25, 6.7, 3.05)
    ax.text(6.32, 1.65, "tensor tra ve\nqua autograd\n(GPU -> .cpu() khi can)",
            ha="center", fontsize=7.2, color="#333333")
    arrow(8.2, 2.7, 8.2, 2.42)
    arrow(8.2, 1.5, 8.2, 1.12)

    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 3.9)
    ax.set_title("Luong du lieu cov2D: GPU rasterizer -> CPU-side thong ke tan so (eta)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "14_cov2d_dataflow.png")


# ---------------------------------------------------------------
# 6) Nguong loc thong ke theo transmittance & opacity
# ---------------------------------------------------------------
def fig_transmittance_filter():
    rng = np.random.default_rng(3)
    n = 900
    Tmax = np.clip(rng.beta(2.0, 2.0, n), 0, 1)
    opac = np.clip(rng.beta(2.5, 3.0, n), 0, 1)
    tauT, tauO = 0.0, 0.05      # arguments/__init__.py:145-146

    keep = (Tmax > tauT) & (opac > tauO)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))

    a = ax[0]
    a.scatter(Tmax[~keep], opac[~keep], s=9, c="#cccccc", label="bi loai")
    a.scatter(Tmax[keep], opac[keep], s=9, c="#4C78A8", label="dung cho thong ke $\\eta$")
    a.axhline(tauO, color="#E45756", ls="--", lw=1.6,
              label="tau_o = 0.05 (freq_opacity_threshold)")
    a.axvline(tauT, color="#54A24B", ls="--", lw=1.8,
              label=r"$\tau_T=0.0$ (mac dinh)")
    a.axvline(0.3, color="#333333", ls=":", lw=1.6,
              label=r"vi du $\tau_T=0.3$: loai Gaussian bi che")
    a.axvspan(0, 0.3, color="#333333", alpha=0.07)
    a.set_xlabel(r"$T^{\max}$ = cov2D[:,6]")
    a.set_ylabel(r"opacity $\sigma_o$")
    a.set_ylim(-0.02, 1.18)
    a.set_title("Mask: $(T^{\\max}>\\tau_T)\\ \\wedge\\ (\\sigma_o>\\tau_o)$", fontsize=10)
    a.legend(fontsize=6.5, loc="upper center", ncol=2)
    a.grid(alpha=0.25)

    b = ax[1]
    taus = np.linspace(0.0, 0.9, 60)
    frac = [(( Tmax > t) & (opac > tauO)).mean() for t in taus]
    b.plot(taus, frac, color="#F58518", lw=2.2)
    b.axvline(0.0, color="#54A24B", ls="--", lw=1.6,
              label=r"mac dinh $\tau_T=0.0$")
    b.set_xlabel(r"nguong transmittance $\tau_T$")
    b.set_ylabel("ti le Gaussian duoc dung")
    b.set_title(r"$\tau_T$ lon -> chi giu Gaussian o be mat truoc", fontsize=10)
    b.legend(fontsize=8)
    b.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "14_transmittance_filter.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_alpha_blending()
    fig_mult_region()
    fig_cost_vs_mult()
    fig_tile_grid()
    fig_cov2d_flow()
    fig_transmittance_filter()
    print("DONE")
