# -*- coding: utf-8 -*-
"""
BOT 06 - expand_undersized_gs (SADGS)
Nguon code that:
  scene/gaussian_model.py:833-864  (expand_undersized_gs)
  train.py:355-359                 (loi goi BI COMMENT OUT)
  arguments/__init__.py:135        (tau_expand = 1.0)
  utils/freq_utils.py:348          (eta_3ch = axis_lengths / wavelength_min)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("OK", p)


# ---------------------------------------------------------------- FIG 1
# delta_log_scale = -0.5*log(eta)  =>  s_new/s_old = eta^(-1/2)
def fig1():
    eta = np.logspace(-6, 0.3, 500)
    ratio = np.exp(-0.5 * np.log(np.clip(eta, 1e-6, None)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(eta, ratio, lw=2.4, color="#1f77b4",
              label=r"$s_{moi}/s_{cu}=\exp(-0.5\ln\eta)=\eta^{-1/2}$")
    ax.axvline(1.0, color="#d62728", ls="--", lw=1.6,
               label=r"$\tau_{expand}=1.0$ (arguments/__init__.py:135)")
    ax.axvline(1e-6, color="#7f7f7f", ls=":", lw=1.6,
               label=r"clamp $\eta_{min}=10^{-6}$ (gaussian\_model.py:854)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.fill_between(eta, 1.0, ratio, where=(eta < 1.0), color="#1f77b4", alpha=0.12)

    for e in [1e-4, 1e-2, 0.25]:
        r = e ** -0.5
        ax.plot([e], [r], "o", color="#ff7f0e", ms=6)
        ax.annotate(r"$\eta=%g \Rightarrow \times%.0f$" % (e, r), (e, r),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)

    ax.set_xlabel(r"$\eta$ = (do dai truc chieu) / (buoc song texture)")
    ax.set_ylabel(r"He so gian scale  $s_{moi}/s_{cu}$")
    ax.set_title("He so gian giai tich theo $\\eta$ (thang log-log)\n"
                 "delta_log_scale = -0.5 * torch.log(eta)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "06_scale_ratio_eta.png")


# ---------------------------------------------------------------- FIG 2
# vung eta bi tac dong theo tau_expand
def fig2():
    eta = np.logspace(-4, 0.6, 600)
    taus = [0.25, 0.5, 1.0, 2.0]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    for t, c in zip(taus, colors):
        gain = np.where((eta < t) & (eta > 0), eta ** -0.5, 1.0)
        lw = 2.6 if t == 1.0 else 1.6
        ax1.loglog(eta, gain, color=c, lw=lw,
                   label=r"$\tau=%.2f$%s" % (t, " (mac dinh)" if t == 1.0 else ""))
    ax1.axhline(1.0, color="k", lw=0.8)
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$s_{moi}/s_{cu}$")
    ax1.set_title("Mask: (eta < tau_expand) & (eta > 0)\ngaussian_model.py:846", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    # bang 2: be rong vung tac dong (thanh ngang tren truc log eta)
    for i, (t, c) in enumerate(zip(taus, colors)):
        ax2.barh(i, np.log10(t) - (-4), left=-4, color=c, alpha=0.55, height=0.55)
        ax2.text(np.log10(t) + 0.05, i, r"$\tau=%.2f$" % t, va="center", fontsize=9)
    ax2.axvline(0.0, color="k", ls="--", lw=1.0)
    ax2.text(0.05, -0.75, r"$\eta=1$ (Nyquist)", fontsize=8)
    ax2.set_yticks(range(len(taus)))
    ax2.set_yticklabels([r"$\tau$=%.2f" % t for t in taus], fontsize=8)
    ax2.set_xlabel(r"$\log_{10}\eta$  (vung to mau = Gaussian bi GIAN)")
    ax2.set_xlim(-4.3, 0.8)
    ax2.set_ylim(-1.1, len(taus) - 0.3)
    ax2.set_title("Vung eta bi tac dong theo tau_expand", fontsize=9)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    save(fig, "06_tau_expand_region.png")


# ---------------------------------------------------------------- FIG 3
# Bat nhat quan: docstring gia dinh eta ~ (sigma*omega)^2, code thuc te eta ~ sigma*omega (tuyen tinh)
def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    eta0 = np.logspace(-4, 0, 300)
    ax1.loglog(eta0, np.ones_like(eta0), "--", color="#7f7f7f", lw=1.6,
               label=r"Gia dinh docstring: $\eta\propto(\sigma\omega)^2\Rightarrow\eta_{moi}=1$")
    ax1.loglog(eta0, np.sqrt(eta0), color="#d62728", lw=2.4,
               label=r"Code that: $\eta=\ell_{truc}/\lambda\Rightarrow\eta_{moi}=\sqrt{\eta}$")
    ax1.axhline(1.0, color="k", lw=0.8)
    ax1.set_xlabel(r"$\eta$ truoc khi gian")
    ax1.set_ylabel(r"$\eta$ sau 1 lan goi")
    ax1.set_title("1 lan goi KHONG dua eta ve 1 (voi eta tuyen tinh)\n"
                  "freq_utils.py:348 vs docstring gaussian_model.py:838", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=7.5, loc="lower right")

    steps = np.arange(0, 9)
    for e0, c in zip([1e-4, 1e-2, 0.2], ["#1f77b4", "#2ca02c", "#ff7f0e"]):
        vals = [e0 ** (0.5 ** k) for k in steps]
        ax2.semilogy(steps, vals, "o-", color=c, lw=1.8, ms=5,
                     label=r"$\eta_0=%g$" % e0)
    ax2.axhline(1.0, color="#d62728", ls="--", lw=1.4, label=r"$\eta=1$")
    ax2.set_xlabel("So lan goi expand_undersized_gs (moi densify_interval)")
    ax2.set_ylabel(r"$\eta$")
    ax2.set_title(r"Hoi tu hinh hoc: $\eta_k=\eta_0^{(1/2)^k}$", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "06_eta_convergence.png")


# ---------------------------------------------------------------- FIG 4
# Ellipse 2D truoc/sau khi gian phu dung buoc song texture
def fig4():
    lam = 2.0          # buoc song texture (pixel)
    s_old = np.array([0.35, 0.9])   # nua truc (pixel) truoc khi gian
    eta = s_old / lam               # eta theo tung truc (freq_utils.py:348)
    gain = np.where(eta < 1.0, eta ** -0.5, 1.0)
    s_new = s_old * gain

    x = np.linspace(-4, 4, 600)
    y = np.linspace(-4, 4, 600)
    X, Y = np.meshgrid(x, y)
    tex = np.sin(2 * np.pi * X / lam)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, s, tag in ((axes[0], s_old, "TRUOC"), (axes[1], s_new, "SAU")):
        ax.imshow(tex, extent=[-4, 4, -4, 4], origin="lower",
                  cmap="Greys", alpha=0.35, vmin=-1.6, vmax=1.6)
        ax.add_patch(Ellipse((0, 0), 2 * s[0], 2 * s[1], facecolor="#1f77b4",
                             alpha=0.45, edgecolor="#08306b", lw=2))
        ax.annotate("", xy=(s[0], 0), xytext=(-s[0], 0),
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.8))
        ax.text(0, 0.18, r"$2s_x=%.2f$ px" % (2 * s[0]), color="#d62728",
                ha="center", fontsize=9)
        ax.annotate("", xy=(2.6, 0), xytext=(2.6 + lam, 0),
                    arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.8))
        ax.text(2.6 + lam / 2, 0.22, r"$\lambda=%.1f$" % lam, color="#2ca02c",
                ha="center", fontsize=9)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s: $\\eta_x=%.2f$, $\\eta_y=%.2f$" %
                     (tag, s[0] / lam, s[1] / lam), fontsize=10)

    fig.suptitle("Gian giai tich: chi SUA _scaling (khong sinh Gaussian moi)\n"
                 r"$s_x:%.2f\to%.2f$ px ($\times%.2f=\eta_x^{-1/2}$); ca 2 truc co $\eta<\tau_{expand}=1$ nen deu duoc gian"
                 % (s_old[0], s_new[0], gain[0]), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "06_ellipse_expand.png")


# ---------------------------------------------------------------- FIG 5
# Chi phi bo nho: expand (0 GS moi) vs clone/split k^3 con
def fig5():
    # 1 Gaussian = 59 float32 = 236 B
    # (xyz 3 + scaling 3 + rotation 4 + opacity 1 + f_dc 3 + f_rest 45)
    # LUU Y: day la UOC LUONG BAC DO LON minh hoa (khong phai code that).
    # densify_and_split_structgs chi chay khi eta>1 (Gaussian qua to), KHONG
    # chay cho Gaussian duoi co (eta<1). "Tile bang ban sao cu" o day nghia la:
    # neu KHONG co expand, phai xep nhieu ban sao GIU NGUYEN kich thuoc cu de
    # lap day cung mot the tich ma expand lam duoc bang cach gian mot Gaussian.
    # eta la ti so TUYEN TINH theo scale (Frame 5 cua slide), nen he so can
    # nhan la 1/eta (khong phai eta^-0.5), va luy thua 3 vi lap mot THE TICH 3D.
    B = 59 * 4
    N0 = 1_000_000
    eta = np.logspace(-4, 0, 200)
    gain = 1.0 / eta                         # he so can nhan moi truc (tuyen tinh)
    k = np.ceil(gain)                        # so ban sao cu moi truc de lap day
    n_tile = k ** 3                          # lap day the tich 3D

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    ax1.loglog(eta, N0 * n_tile * B / 1e9, color="#d62728", lw=2.4,
               label=r"Tile bằng bản sao cũ: $N\cdot k^3$, $k=\lceil 1/\eta\rceil$")
    ax1.loglog(eta, np.full_like(eta, N0 * B / 1e9), color="#2ca02c", lw=2.4,
               label="expand_undersized_gs: $N$ không đổi")
    ax1.set_xlabel(r"$\eta$ của Gaussian dưới cỡ")
    ax1.set_ylabel("Bộ nhớ tham số (GB), $N=10^6$")
    ax1.set_title("Ước lượng bậc độ lớn để lấp cùng thể tích", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    etas = [0.5, 0.25, 0.1, 0.01]
    xs = np.arange(len(etas))
    kk = np.ceil(1.0 / np.array(etas))
    mem_tile = N0 * (kk ** 3) * B / 1e9
    mem_exp = np.full(len(etas), N0 * B / 1e9)
    ax2.bar(xs - 0.19, mem_tile, 0.38, color="#d62728", label="tile bằng bản sao cũ")
    ax2.bar(xs + 0.19, mem_exp, 0.38, color="#2ca02c", label="expand")
    for i, (mt, kv) in enumerate(zip(mem_tile, kk)):
        ax2.text(i - 0.19, mt * 1.1, r"$\times%d$" % (kv ** 3), ha="center", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([r"$\eta=%g$" % e for e in etas], fontsize=9)
    ax2.set_ylabel("GB")
    ax2.set_title("236 B/Gaussian (59 float32)", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("Ước lượng bậc độ lớn (minh hoạ) — không phải hành vi thật của\n"
                 "densify_and_split_structgs, hàm này không chạy khi $\\eta<1$",
                 fontsize=9, y=1.05)
    fig.tight_layout()
    save(fig, "06_memory_expand_vs_clone.png")


# ---------------------------------------------------------------- FIG 6
# Timeline train.py: nhanh expand bi comment out
def fig6():
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.axis("off")

    steps = [
        ("train.py:320  is_normal_densification\niteration > densify_from_iter", "#dbe9f6", "-"),
        ("train.py:326-331  grads, is_grad_high\n||grad|| >= 1e-5", "#dbe9f6", "-"),
        ("train.py:337-345  high/low ratio -> split_mask\nsplit_ratio_threshold = 0.8", "#dbe9f6", "-"),
        ("train.py:348-350  avg_high_eta_3ch\n= eta_high_sum_3ch / eta_high_count", "#dbe9f6", "-"),
        ("train.py:355-359  expand_undersized_gs(tau_expand,\navg_high_eta_3ch)   ### BI COMMENT OUT ###", "#f2f2f2", "--"),
        ("train.py:361-374  densify_and_prune_structgs(\nmax_eta_3ch = max_high_eta)", "#dbe9f6", "-"),
        ("train.py:377-383  reset accumulators  .zero_()", "#dbe9f6", "-"),
    ]

    x0, w, h = 0.16, 0.66, 0.085
    gap = 0.038
    y = 0.90
    ys = []
    for i, (txt, col, ls) in enumerate(steps):
        edge = "#999999" if ls == "--" else "#1f4e79"
        ax.add_patch(FancyBboxPatch((x0, y - h), w, h,
                                    boxstyle="round,pad=0.006",
                                    facecolor=col, edgecolor=edge, lw=1.8, linestyle=ls))
        ax.text(x0 + w / 2, y - h / 2, txt, ha="center", va="center", fontsize=7.4,
                color="#8c8c8c" if ls == "--" else "black")
        ys.append((y, y - h))
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y - h), (x0 + w / 2, y - h - gap),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#555555", lw=1.4))
        y -= (h + gap)

    # duong by-pass quanh buoc bi comment out
    y_top = ys[3][1]
    y_bot = ys[5][0]
    ax.add_patch(FancyArrowPatch((x0, y_top - 0.008), (x0, y_bot + 0.008),
                                 connectionstyle="arc3,rad=0.75", arrowstyle="-|>",
                                 mutation_scale=14, color="#2ca02c", lw=2.2))
    ax.text(0.015, (y_top + y_bot) / 2, "LUONG\nTHUC TE\n(nhay qua)",
            ha="left", va="center", fontsize=8, color="#2ca02c")

    ax.text(0.5, 0.01,
            "expand_undersized_gs (gaussian_model.py:833-864) TON TAI nhung KHONG duoc goi\n"
            "=> mac dinh SADGS chi SPLIT / PRUNE, khong co nhanh gian giai tich",
            ha="center", fontsize=8.5, color="#b22222")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.06, 1.0)
    ax.set_title("Timeline khoi densification trong train.py - vi tri nhanh bi vo hieu hoa",
                 fontsize=11)
    save(fig, "06_timeline_commented.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("DONE")
