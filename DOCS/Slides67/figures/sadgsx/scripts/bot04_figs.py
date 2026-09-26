# -*- coding: utf-8 -*-
"""
BOT 04 — Hinh minh hoa cho densify_and_split_structgs (SADGS/scene/gaussian_model.py:638-832)
Moi cong thuc <=> 1 hinh. Tat ca tham so lay tu code that.
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
    print("saved", p)


# ---------------------------------------------------------------- formula 1
# gaussian_model.py:699-701
#   ks = torch.sqrt(torch.clamp(etavals, min=1.0)).ceil().int()
#   ks = torch.clamp(ks, min=1)
def k_of_eta(eta):
    return np.ceil(np.sqrt(np.clip(eta, 1.0, None))).astype(int)


def fig1():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    eta = np.linspace(0.0, 40.0, 4001)
    k = k_of_eta(eta)
    ax.step(eta, k, where="post", color="#1f4e79", lw=2.0,
            label=r"$k=\lceil\sqrt{\max(\eta,1)}\;\rceil$  (dòng 699)")
    ax.plot(eta, np.sqrt(np.clip(eta, 1e-9, None)), "--", color="#c0392b", lw=1.3,
            label=r"$\sqrt{\eta}$ (lý thuyết lấy mẫu)")
    for kk in range(2, 7):
        e = (kk - 1) ** 2
        ax.axvline(e, color="0.8", lw=0.8, zorder=0)
        ax.annotate(r"$\eta=%d$" % e, (e, 0.25), fontsize=7, rotation=90,
                    color="0.35", ha="right", va="bottom")
    ax.axvspan(0, 1, color="#2e7d32", alpha=0.12)
    ax.annotate("η ≤ 1 → k = 1\n(không tách trục này)", (0.6, 5.2), fontsize=8,
                color="#2e7d32", ha="left")
    ax.set_xlabel(r"$\eta$ (năng lượng tần số vượt Nyquist, theo từng kênh/trục)")
    ax.set_ylabel(r"$k$ — số lát cắt trên trục")
    ax.set_title("Bậc thang k = f(η) — không có clamp max trong code (chỉ clamp min=1)", fontsize=10)
    ax.set_yticks(range(0, 8))
    ax.set_xlim(0, 40); ax.set_ylim(0, 7.2)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    save(fig, "04_k_eta_stair.png")


# ---------------------------------------------------------------- formula 2
# gaussian_model.py:714  N_per_point = ks.prod(dim=1)
def fig2():
    K = 6
    kx = np.arange(1, K + 1)
    ky = np.arange(1, K + 1)
    N = np.outer(kx, ky)  # kz = 1
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(N, origin="lower", cmap="YlOrRd",
                   extent=[0.5, K + 0.5, 0.5, K + 0.5], vmin=1, vmax=K * K)
    for i in range(K):
        for j in range(K):
            ax.text(j + 1, i + 1, str(N[i, j]), ha="center", va="center",
                    fontsize=9, color="black" if N[i, j] < 0.6 * K * K else "white")
    ax.set_xticks(kx); ax.set_yticks(ky)
    ax.set_xlabel(r"$k_y$"); ax.set_ylabel(r"$k_x$")
    ax.set_title(r"$N = k_x k_y k_z$ (ở đây $k_z=1$) — số con sinh ra mỗi Gaussian"
                 "\n(gaussian_model.py:714)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("N con")
    ax.annotate("3DGS gốc: luôn N = 2", xy=(2, 1), xytext=(3.6, 1.6), fontsize=8,
                color="#1f4e79",
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0))
    save(fig, "04_N_heatmap.png")


# ---------------------------------------------------------------- formula 3
# gaussian_model.py:763-788
#   grid = i - (k-1)/2 ; sep = (sigma/k)*sqrt(12) ; off = R @ (sep*grid)
#   scale_child = sigma / k^p
SQRT12 = 12.0 ** 0.5


def children(sigma, ks, theta, p=1.0, center=(0.0, 0.0)):
    """Tra ve list (cx, cy, sx_child, sy_child) dung theo code."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sig = np.asarray(sigma, float)
    k = np.asarray(ks, float)
    child_scale = sig / k ** p                 # dòng 722
    sep = (sig / k) * SQRT12                   # dòng 775-776
    out = []
    for ix in range(int(ks[0])):
        for iy in range(int(ks[1])):
            g = np.array([ix - (ks[0] - 1) / 2.0, iy - (ks[1] - 1) / 2.0])
            local = sep * g                    # dòng 778
            world = R @ local                  # dòng 786
            out.append((center[0] + world[0], center[1] + world[1],
                        child_scale[0], child_scale[1]))
    return out, child_scale


def draw_ell(ax, cx, cy, sx, sy, theta, **kw):
    ax.add_patch(Ellipse((cx, cy), 2 * sx, 2 * sy,
                         angle=np.degrees(theta), **kw))


def fig3():
    sigma = np.array([1.0, 0.32])   # Gaussian dị hướng
    theta = np.radians(28.0)
    ks = np.array([4, 2])           # eta_x ~ 10 -> kx=4 ; eta_y ~ 3 -> ky=2
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    draw_ell(ax, 0, 0, sigma[0], sigma[1], theta, facecolor="none",
             edgecolor="#1f4e79", lw=2.0, ls="--", label="Gaussian cha (1σ)")
    ch, cs = children(sigma, ks, theta, p=1.0)
    for i, (cx, cy, sx, sy) in enumerate(ch):
        draw_ell(ax, cx, cy, sx, sy, theta, facecolor="#e67e22", alpha=0.45,
                 edgecolor="#a04000", lw=1.2,
                 label="Gaussian con" if i == 0 else None)
        ax.plot(cx, cy, ".", color="#7b241c", ms=4)
    # truc local
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for v, c, lab in ((np.array([1, 0]), "#c0392b", "trục local x"),
                      (np.array([0, 1]), "#27ae60", "trục local y")):
        d = R @ (v * sigma * 2.3)
        ax.annotate("", xy=d, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        ax.annotate(lab, xy=d * 1.08, color=c, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.0, 2.0)
    ax.grid(alpha=0.2)
    ax.set_title(r"Tách dị hướng $k_x{=}4,\;k_y{=}2 \Rightarrow N=8$ con;  "
                 r"$\sigma'=\sigma/k$,  khoảng cách $=\sqrt{12}\,\sigma'$",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("σ' = (%.3f, %.3f) — lưới đều trong hệ toạ độ local rồi xoay bằng R(q)"
                  % (cs[0], cs[1]), fontsize=8)
    save(fig, "04_split_grid_2d.png")


# ---------------------------------------------------------------- formula 4
def fig4():
    rng = np.random.default_rng(7)
    sigma = np.array([1.0, 0.32]); theta = np.radians(28.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)

    # --- 3DGS goc: densify_and_split, dong 875-880
    ax = axes[0]
    draw_ell(ax, 0, 0, sigma[0], sigma[1], theta, facecolor="none",
             edgecolor="#1f4e79", lw=2.0, ls="--")
    N = 2
    s3 = sigma / (0.8 * N)              # dòng 880: / (0.8*N)
    for i in range(N):
        smp = rng.normal(0.0, sigma)    # dòng 877: torch.normal(0, stds)
        c = R @ smp
        draw_ell(ax, c[0], c[1], s3[0], s3[1], theta, facecolor="#5dade2",
                 alpha=0.55, edgecolor="#1b4f72", lw=1.2)
        ax.plot(*c, ".", color="#154360", ms=5)
    ax.set_title("3DGS gốc: N=2, vị trí LẤY MẪU NGẪU NHIÊN\n"
                 r"$\sigma'=\sigma/(0.8N)=\sigma/1.6$ (đẳng hướng theo hệ số)",
                 fontsize=9)

    # --- SADGS
    ax = axes[1]
    draw_ell(ax, 0, 0, sigma[0], sigma[1], theta, facecolor="none",
             edgecolor="#1f4e79", lw=2.0, ls="--")
    ks = np.array([4, 2])
    ch, cs = children(sigma, ks, theta, 1.0)
    for cx, cy, sx, sy in ch:
        draw_ell(ax, cx, cy, sx, sy, theta, facecolor="#e67e22", alpha=0.5,
                 edgecolor="#a04000", lw=1.2)
        ax.plot(cx, cy, ".", color="#7b241c", ms=4)
    ax.set_title(r"SADGS: $k_x{=}4,k_y{=}2\Rightarrow N{=}8$, lưới TẤT ĐỊNH"
                 "\n" r"$\sigma'=\sigma/k$ — thu nhỏ khác nhau theo từng trục",
                 fontsize=9)

    for ax in axes:
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.9, 1.9)
    fig.suptitle("Cùng một Gaussian dị hướng — hai cách tách", fontsize=11)
    save(fig, "04_3dgs_vs_sadgs.png")


# ---------------------------------------------------------------- formula 5
# scale_power p:  sigma' = sigma / k^p  (dòng 722) — nhung khoang cach van dung sigma/k (dòng 775)
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharex=True, sharey=True)
    sigma = np.array([1.0, 0.32]); theta = np.radians(28.0)
    ks = np.array([4, 2])
    for ax, p in zip(axes, (0.5, 1.0, 2.0)):
        draw_ell(ax, 0, 0, sigma[0], sigma[1], theta, facecolor="none",
                 edgecolor="#1f4e79", lw=1.6, ls="--")
        ch, cs = children(sigma, ks, theta, p)
        for cx, cy, sx, sy in ch:
            draw_ell(ax, cx, cy, sx, sy, theta, facecolor="#8e44ad", alpha=0.45,
                     edgecolor="#4a235a", lw=1.0)
        ax.set_title(r"scale_power $p=%.1f$" % p + "\n" +
                     r"$\sigma'=(%.3f,\,%.3f)$" % (cs[0], cs[1]), fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.9, 1.9)
    fig.suptitle(r"Ảnh hưởng của scale_power: $\sigma'=\sigma/k^{p}$ (dòng 722), "
                 r"nhưng vị trí vẫn dùng $\sqrt{12}\,\sigma/k$ (dòng 775) "
                 "→ p lớn = con nhỏ, hở khe", fontsize=10)
    save(fig, "04_scale_power.png")


# ---------------------------------------------------------------- formula 6
def fig6():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    rounds = np.arange(0, 11)
    n0 = 1e5
    for eta, col in ((1.0, "#2e7d32"), (2.5, "#1f4e79"),
                     (5.0, "#e67e22"), (10.0, "#c0392b")):
        k = int(k_of_eta(eta))
        # gia dinh 20% Gaussian duoc chon tach moi vong, N = k^3 (ca 3 truc vi pham)
        frac = 0.2
        N = k ** 3
        n = [n0]
        for _ in rounds[1:]:
            n.append(n[-1] * (1 - frac) + n[-1] * frac * N)
        ax.semilogy(rounds, n, "-o", ms=3.5, color=col,
                    label=r"$\eta=%.1f \Rightarrow k=%d,\;N=k^3=%d$" % (eta, k, N))
    # 3DGS
    n = [n0]
    for _ in rounds[1:]:
        n.append(n[-1] * (1 - 0.2) + n[-1] * 0.2 * 2)
    ax.semilogy(rounds, n, "--s", ms=3.5, color="black", label="3DGS gốc: N=2")
    ax.set_xlabel("số vòng densification (20% Gaussian được tách mỗi vòng)")
    ax.set_ylabel("số Gaussian (log)")
    ax.set_title(r"Tăng trưởng: $n_{t+1}=n_t\,[\,(1-f)+f\cdot N\,]$, $N=k_xk_yk_z$",
                 fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    save(fig, "04_growth.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
