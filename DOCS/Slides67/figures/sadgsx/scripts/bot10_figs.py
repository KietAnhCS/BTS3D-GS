# -*- coding: utf-8 -*-
"""
BOT 10 - Bo loc 3D chong alias (3D filter / Mip-splatting style) trong SADGS.
Moi hinh minh hoa dung mot cong thuc trich tu code that:
  - SADGS/scene/gaussian_model.py:254  filter_3D = distance / focal_length * (0.2 ** 0.5)
  - SADGS/scene/gaussian_model.py:156-161  scale hieu dung = sqrt(s^2 + filter^2)
  - SADGS/scene/gaussian_model.py:190-201  coef = sqrt(det1/det2), opacity *= coef
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.abspath(OUT)
C = 0.2 ** 0.5  # hang so hard-coded trong gaussian_model.py:254


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------- Fig 1
# filter_3D = z / f * sqrt(0.2)
def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    z = np.linspace(0.5, 20.0, 400)
    for f in [400.0, 800.0, 1600.0]:
        ax[0].plot(z, z / f * C, label=r"$f_x=%d$ px" % f)
    ax[0].set_xlabel("z = do sau toi camera gan nhat (don vi canh)")
    ax[0].set_ylabel(r"$filter_{3D}$")
    ax[0].set_title(r"$filter_{3D}=\frac{z}{f}\sqrt{0.2}$   (gaussian_model.py:254)",
                    fontsize=9)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    zz = np.linspace(0.5, 20.0, 300)
    ff = np.linspace(200.0, 2000.0, 300)
    Z, F = np.meshgrid(zz, ff)
    V = Z / F * C
    im = ax[1].pcolormesh(Z, F, V, shading="auto", cmap="viridis")
    cs = ax[1].contour(Z, F, V, levels=[0.002, 0.005, 0.01, 0.02],
                       colors="w", linewidths=0.8)
    ax[1].clabel(cs, fmt="%.3f", fontsize=7)
    fig.colorbar(im, ax=ax[1], label=r"$filter_{3D}$")
    ax[1].set_xlabel("z (do sau)")
    ax[1].set_ylabel(r"focal_length $=\max_i f_x^{(i)}$ (px)")
    ax[1].set_title("Ban do gia tri: xa camera / focal nho $\\Rightarrow$ loc manh", fontsize=9)
    fig.tight_layout()
    save(fig, "10_filter_vs_dist_focal.png")


# ---------------------------------------------------------------- Fig 2
# Gaussian 1D truoc / sau loc (convolution voi kernel sigma = filter_3D)
def fig2():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    x = np.linspace(-0.06, 0.06, 800)

    def g(x, s, a=1.0):
        return a * np.exp(-0.5 * (x / s) ** 2)

    s = 0.008
    for filt, col in zip([0.0, 0.006, 0.012], ["k", "tab:blue", "tab:red"]):
        se = np.sqrt(s ** 2 + filt ** 2)
        coef = (s ** 2 / se ** 2) ** 0.5  # 1D: sqrt(det1/det2)
        ax[0].plot(x, g(x, se, 1.0), col, ls="--", lw=1.2,
                   label=r"$filter=%.3f$, chua bu $\alpha$" % filt)
        ax[1].plot(x, g(x, se, coef), col, lw=1.6,
                   label=r"$filter=%.3f$, $\alpha\cdot$coef" % filt)
    for a, t in zip(ax, ["Chi mo rong scale: dinh giu nguyen\n$\\Rightarrow$ nang luong PHONG LEN",
                         "Co bu opacity: dien tich duoi duong cong bao toan"]):
        a.set_xlabel("x (khong gian the gioi)")
        a.set_ylabel(r"$\alpha\,G(x)$")
        a.set_title(t, fontsize=9)
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.suptitle(r"Profile Gaussian 1D: $s_{eff}=\sqrt{s^2+filter_{3D}^2}$ (gaussian_model.py:159-160)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "10_gauss_profile.png")


# ---------------------------------------------------------------- Fig 3
# scale hieu dung vs scale goc
def fig3():
    fig, ax = plt.subplots(figsize=(6, 4))
    s = np.linspace(0, 0.05, 500)
    ax.plot(s, s, "k--", lw=1.2, label=r"$s_{eff}=s$ (khong loc, $filter=0$)")
    for filt, col in zip([0.005, 0.010, 0.020], ["tab:blue", "tab:orange", "tab:red"]):
        ax.plot(s, np.sqrt(s ** 2 + filt ** 2), col, lw=1.8,
                label=r"$filter_{3D}=%.3f$" % filt)
        ax.axhline(filt, color=col, ls=":", lw=0.9)
    ax.set_xlabel(r"$s$ = get_scaling = $\exp(\_scaling)$")
    ax.set_ylabel(r"$s_{eff}=\sqrt{s^2+filter_{3D}^2}$")
    ax.set_title("Scale hieu dung: san nen kich thuoc toi thieu = $filter_{3D}$\n"
                 "(Gaussian qua nho bi keo len nguong Nyquist)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "10_scale_effective.png")


# ---------------------------------------------------------------- Fig 4
# he so bu opacity coef = sqrt(det1/det2)
def fig4():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    r = np.linspace(0, 4, 500)  # r = filter / s (dang dang huong)
    coef_iso = (1.0 + r ** 2) ** (-1.5)
    ax[0].plot(r, coef_iso, "tab:red", lw=2, label=r"dang huong: $(1+r^2)^{-3/2}$")
    ax[0].plot(r, (1.0 + r ** 2) ** (-1.0), "tab:blue", lw=1.4, ls="--",
               label=r"2 truc bi loc: $(1+r^2)^{-1}$")
    ax[0].plot(r, (1.0 + r ** 2) ** (-0.5), "tab:green", lw=1.4, ls=":",
               label=r"1 truc bi loc: $(1+r^2)^{-1/2}$")
    ax[0].set_xlabel(r"$r = filter_{3D}/s$")
    ax[0].set_ylabel("coef")
    ax[0].set_title(r"$coef=\sqrt{\dfrac{\prod_i s_i^2}{\prod_i (s_i^2+filter^2)}}$"
                    "\n(gaussian_model.py:196-200)", fontsize=9)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].axvline(1.0, color="gray", lw=0.8)
    ax[0].annotate(r"$r=1\Rightarrow coef=2^{-3/2}\approx0.354$", xy=(1.0, 2 ** -1.5),
                   xytext=(1.6, 0.55), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.8))

    # opacity hieu dung cho vai muc alpha
    for a0, col in zip([0.9, 0.5, 0.2], ["tab:purple", "tab:orange", "tab:cyan"]):
        ax[1].plot(r, a0 * coef_iso, col, lw=1.8, label=r"$\alpha=%.1f$" % a0)
    ax[1].set_xlabel(r"$r = filter_{3D}/s$")
    ax[1].set_ylabel(r"$\alpha\cdot coef$ = get_opacity_with_3D_filter")
    ax[1].set_title("Opacity sau bu: Gaussian bi lam mo nhieu\n"
                    "thi mo di, tranh 'day' mau", fontsize=9)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "10_opacity_coef.png")


# ---------------------------------------------------------------- Fig 5
# Minh hoa alias khi zoom xa neu khong co filter
def fig5():
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=True)
    x = np.linspace(0, 1, 2000)
    f0 = 23.0
    sig = np.sin(2 * np.pi * f0 * x)

    # lay mau thua (camera o xa -> it pixel tren cung be mat)
    N = 18
    xs = np.linspace(0, 1, N, endpoint=False)
    ys = np.sin(2 * np.pi * f0 * xs)

    ax[0].plot(x, sig, color="0.6", lw=0.8, label="tin hieu that (23 chu ky)")
    ax[0].plot(xs, ys, "o-", color="tab:red", ms=4, lw=1.4,
               label="mau (18 diem) $\\Rightarrow$ alias")
    ax[0].set_title("KHONG loc: lay mau thua\n$\\Rightarrow$ tan so gia (moire)", fontsize=9)

    # Loc truoc bang Gaussian sigma ~ filter_3D roi moi lay mau
    sigma = 0.02
    atten = np.exp(-0.5 * (2 * np.pi * f0 * sigma) ** 2)
    sig_f = atten * np.sin(2 * np.pi * f0 * x)
    ax[1].plot(x, sig, color="0.85", lw=0.8)
    ax[1].plot(x, sig_f, color="tab:blue", lw=1.2,
               label=r"sau loc $\sigma=filter_{3D}$")
    ax[1].plot(xs, atten * np.sin(2 * np.pi * f0 * xs), "o", color="tab:blue", ms=4,
               label="mau: bien do $\\approx$ 0")
    ax[1].set_title(r"CO loc: $\hat G(\omega)=e^{-\omega^2 filter^2/2}$"
                    "\ndap tat tan so > Nyquist", fontsize=9)

    # do loi bien do theo tan so
    w = np.linspace(0, 60, 400)
    ax[2].plot(w, np.ones_like(w), "k--", lw=1.0, label="khong loc")
    for sg, col in zip([0.005, 0.01, 0.02], ["tab:green", "tab:orange", "tab:blue"]):
        ax[2].plot(w, np.exp(-0.5 * (2 * np.pi * w * sg) ** 2), col, lw=1.6,
                   label=r"$filter=%.3f$" % sg)
    ax[2].axvline(N / 2.0, color="tab:red", lw=1.0, ls=":")
    ax[2].text(N / 2.0 + 1, 0.8, "Nyquist", color="tab:red", fontsize=8)
    ax[2].set_xlabel("tan so khong gian (chu ky/don vi)")
    ax[2].set_title("Dap ung tan so cua bo loc", fontsize=9)
    ax[2].set_ylim(-0.05, 1.15)

    for a in ax[:2]:
        a.set_xlabel("x")
        a.set_ylim(-1.3, 1.3)
    ax[0].set_ylabel("cuong do")
    for a in ax:
        a.legend(fontsize=7, loc="lower right")
        a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "10_alias_demo.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("DONE")
