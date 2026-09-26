"""BOT 16 - Duong day SADGS tu STRUCTURE TENSOR den ADC (4 hinh tong quan).

Muc dich: mot bo hinh "nhin la hieu" noi lien mach 5 tang:
  anh -> structure tensor -> lambda_min -> eta 3 truc -> bo dem da view -> ADC.

Nguon doi chieu (doc thang tu source):
  SADGS/utils/loss_utils.py:170-230      get_structure_tensor_torch (Di Zenzo)
  SADGS/utils/loss_utils.py:232-313      get_multiscale_structure_tensor_v1
  SADGS/train.py:160-200                 cache structure tensor truoc vong lap
  SADGS/utils/freq_utils.py:255-292      jitter Cholesky lay mau tensor
  SADGS/utils/freq_utils.py:313-348      lambda_1 -> lambda_min -> eta = ||a||/lambda_min
  SADGS/utils/freq_utils.py:380-415      9 bo dem online, TAU_HIGH=1.0 TAU_LOW=0.1
  SADGS/train.py:337-353                 high_ratio/low_ratio -> split_mask/prune_mask
  SADGS/scene/gaussian_model.py:698-701  k = ceil(sqrt(eta)), scale /= k^p

Cac gia tri so trong hinh la MINH HOA (illustrative) de thay dang, khong phai so do.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

C_IMG = "#dfe9f5"   # tang anh / structure tensor
C_GEO = "#d8e8d8"   # tang hinh hoc Gaussian
C_ETA = "#ffe2b8"   # tang tan so / eta
C_STAT = "#e8ddf2"  # tang thong ke da view
C_ADC = "#f6cccc"   # tang ADC
C_OFF = "#e6e6e6"


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


def box(ax, x, y, w, h, text, color, fs=7.2, ec="#333333", ls="-", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.1",
                                fc=color, ec=ec, lw=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.3)


def arrow(ax, p1, p2, color="#444444", ls="-", lw=1.3, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 lw=lw, color=color, linestyle=ls, zorder=1,
                                 connectionstyle="arc3,rad=%.2f" % rad))


# ------------------------------------------------------------------ FIG 1
def fig_flow():
    """Duong day day du: tu anh GT + Gaussian den 3 hanh dong ADC."""
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 100)
    ax.axis("off")

    ax.text(50, 97.5, "SADGS: tu STRUCTURE TENSOR den ADC", ha="center",
            fontsize=13.5, fontweight="bold")
    ax.text(50, 94.4,
            "trai = nhanh ANH (1 lan, truoc vong lap)   |   phai = nhanh GAUSSIAN (moi view)",
            ha="center", fontsize=8.2, color="#555555")

    # --- Cot trai: nhanh anh ----------------------------------------------
    box(ax, 2, 84, 30, 7.5,
        "[A1] Anh train I (HxWx3)\nScene / dataloader", C_IMG, fs=7.6)
    box(ax, 2, 72.5, 30, 9,
        "[A2] Kim tu thap L = st_levels = 4\nblur + downsample 2^l\nloss_utils.py:232-313", C_IMG)
    box(ax, 2, 59.5, 30, 10.5,
        "[A3] Structure tensor Di Zenzo\nS = sum_c [Ix^2, IxIy; IxIy, Iy^2]\n"
        "-> st_map (Sxx,Sxy,Syy), cache 1 lan\nloss_utils.py:170-230 | train.py:160-200",
        C_IMG, fs=6.9)
    box(ax, 2, 47.5, 30, 9.5,
        "[A4] Tri rieng lon nhat\nlam1 = tr/2 + sqrt((tr/2)^2 - det)\nfreq_utils.py:325-334", C_IMG)
    box(ax, 2, 35.5, 30, 9.5,
        "[A5] Buoc song cuc bo\nlambda_min = 1/(sqrt(lam1)+1e-5)  [px]\n"
        "vung phang -> lambda_min ~ 1e5", C_IMG)

    for y in (84, 72.5, 59.5, 47.5):
        arrow(ax, (17, y), (17, y - 3.0))

    # --- Cot phai: nhanh Gaussian -----------------------------------------
    box(ax, 68, 84, 30, 7.5,
        "[G1] N Gaussian (mu, R, s, alpha, SH)\nrender_structgs -> cov2D [N,7]", C_GEO, fs=7.4)
    box(ax, 68, 72.5, 30, 9,
        "[G2] Chieu 3 truc chinh qua Jacobian\na_j = J R diag(s) e_j   (j = 1,2,3)\n"
        "freq_utils.py:116-178", C_GEO)
    box(ax, 68, 59.5, 30, 10.5,
        "[G3] Do dai truc tren anh\n||a_j|| = sqrt(u_j^2 + v_j^2 + 1e-8)  [px]\n"
        "~ f*s_j/z  ->  ti le nghich depth", C_GEO)
    box(ax, 68, 47.5, 30, 9.5,
        "[G4] Jitter Cholesky trong ellipse 1-sigma\n(du,dv) = L*eps -> grid_sample st_map\n"
        "freq_utils.py:255-292", C_GEO, fs=6.9)

    for y in (84, 72.5, 59.5):
        arrow(ax, (83, y), (83, y - 3.0))

    # --- Hop nhat -> eta ---------------------------------------------------
    box(ax, 30, 22.5, 40, 9.5,
        "[E] TI SO VI PHAM TAN SO (khong thu nguyen)\n"
        r"$\eta_j = \|a_j\| \,/\, \lambda_{\min}$" + "   cho tung truc j = 1,2,3\n"
        "freq_utils.py:348   (eta_compute_mode = 'wavelength')", C_ETA, fs=8.0)
    arrow(ax, (17, 35.5), (40, 32.0), rad=-0.15, color="#2b6cb0")
    arrow(ax, (83, 47.5), (60, 32.0), rad=0.15, color="#2f6f3f")
    arrow(ax, (68, 52.0), (32.5, 41.5), rad=0.22, color="#999999", ls="--", lw=1.0)
    ax.text(50, 43.5, "G4 lay mau st_map ngay trong vung Gaussian phu",
            ha="center", fontsize=6.6, color="#777777")

    # --- Phan loai + bo dem ------------------------------------------------
    box(ax, 2, 9.5, 27, 9.5,
        "[S1] Phan loai moi view\neta_max = max_j eta_j\n"
        "HIGH > 1.0 | LOW < 0.1 | MID\nfreq_utils.py:394-415", C_STAT, fs=7.0)
    box(ax, 32, 9.5, 27, 9.5,
        "[S2] 9 bo dem online (moi 10 iter)\neta_high_count, eta_low_count,\n"
        "accum_view_count, max_eta_3ch...\ntrain.py:275-276", C_STAT, fs=7.0)
    box(ax, 62, 9.5, 36, 9.5,
        "[S3] Ti le nhat quan da view (moi 100 iter)\n"
        "high_ratio = eta_high_count / accum_view_count\n"
        "low_ratio  = eta_low_count  / accum_view_count\ntrain.py:337-343", C_STAT, fs=7.0)
    arrow(ax, (50, 22.5), (15.5, 19.0), rad=0.12)
    arrow(ax, (29, 14.2), (32, 14.2))
    arrow(ax, (59, 14.2), (62, 14.2))

    # --- ADC ---------------------------------------------------------------
    box(ax, 2, -4.6, 30, 7.0,
        "SPLIT di huong\nhigh_ratio > 0.8 AND ||grad|| >= 1e-5\n"
        "k_j = ceil(sqrt(eta_j)),  s/k^p", C_ADC, fs=7.0)
    box(ax, 35, -4.6, 28, 7.0,
        "PRUNE\nlow_ratio > 0.8 AND view_count > 0\n+ alpha < 0.1, r2D > 20", C_ADC, fs=7.0)
    box(ax, 66, -4.6, 32, 7.0,
        "EXPAND (tau_expand) - MAC DINH TAT\ntrain.py:355-359 bi comment", C_OFF, fs=7.0, ls="--")
    arrow(ax, (68, 9.5), (17, 2.6), rad=0.10, color="#a03333")
    arrow(ax, (78, 9.5), (49, 2.6), rad=0.10, color="#a03333")
    arrow(ax, (90, 9.5), (84, 2.6), color="#999999", ls="--")

    ax.text(99.3, 30, "moi 10 iter", rotation=90, fontsize=6.8, color="#666666", va="center")
    ax.text(99.3, 13, "moi 100 iter", rotation=90, fontsize=6.8, color="#666666", va="center")
    save(fig, "16_flow_tensor_to_adc.png")


# ------------------------------------------------------------------ FIG 2
def fig_eta_geometry():
    """Vi sao eta la mot ti so: ve that anh soc + ellipse Gaussian chieu len no."""
    fig = plt.figure(figsize=(13.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.28)

    # (a) anh + ellipse
    ax = fig.add_subplot(gs[0, 0])
    H = W = 220
    yy, xx = np.mgrid[0:H, 0:W]
    img = 0.5 + 0.5 * np.sin(2 * np.pi * xx / 9.0)
    img[:, :90] = 0.5 + 0.5 * np.sin(2 * np.pi * xx[:, :90] / 40.0)
    ax.imshow(img, cmap="gray", origin="upper")
    ax.add_patch(Ellipse((150, 110), width=70, height=26, angle=25,
                         fc="none", ec="#d94a4a", lw=2.2))
    th = np.deg2rad(25)
    bb = dict(fc="white", ec="none", alpha=0.85, pad=0.15)
    ax.annotate("", xy=(150 + 35 * np.cos(th), 110 + 35 * np.sin(th)), xytext=(150, 110),
                arrowprops=dict(arrowstyle="-|>", color="#d94a4a", lw=2.2))
    ax.text(150 + 34 * np.cos(th), 110 + 34 * np.sin(th) - 30, r"$\|a_1\|=35$px",
            color="#d94a4a", fontsize=9, fontweight="bold", ha="center", bbox=bb)
    ax.annotate("", xy=(150 - 13 * np.sin(th), 110 + 13 * np.cos(th)), xytext=(150, 110),
                arrowprops=dict(arrowstyle="-|>", color="#1a7a3a", lw=2.2))
    ax.text(150 - 62 * np.sin(th), 110 + 40 * np.cos(th), r"$\|a_2\|=13$px",
            color="#1a7a3a", fontsize=9, fontweight="bold", ha="center", bbox=bb)
    ax.plot([176, 185], [178, 178], color="#1f6fd0", lw=5, solid_capstyle="butt")
    ax.text(180, 170, r"$\lambda_{\min}=9$px", color="#1f6fd0", fontsize=9,
            ha="center", fontweight="bold", bbox=bb)
    ax.set_title("(a) Gaussian chieu len vung texture\ntu so = truc chieu, mau so = buoc song",
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    # (b) bar eta 3 truc
    ax = fig.add_subplot(gs[0, 1])
    lam = 9.0
    eta = np.array([35.0, 13.0, 0.6]) / lam
    cols = ["#d94a4a" if e > 1.0 else ("#cccc55" if e > 0.1 else "#2f6f3f") for e in eta]
    ax.bar(["truc 1", "truc 2", "truc 3"], eta, color=cols, edgecolor="#333333")
    for i, e in enumerate(eta):
        ax.text(i, e * 1.12, "%.2f" % e, ha="center", fontsize=9)
    ax.axhline(1.0, color="#d94a4a", ls="--", lw=1.2)
    ax.axhline(0.1, color="#2f6f3f", ls="--", lw=1.2)
    ax.text(2.45, 1.08, r"$\tau_{high}=1.0$", color="#d94a4a", fontsize=8, ha="right")
    ax.text(2.45, 0.108, r"$\tau_{low}=0.1$", color="#2f6f3f", fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 12)
    ax.set_ylabel(r"$\eta_j=\|a_j\|/\lambda_{\min}$")
    ax.set_title("(b) 3 truc = 3 con so khac nhau\n-> biet vi pham theo HUONG nao", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (c) eta theo depth va texture
    ax = fig.add_subplot(gs[0, 2])
    z = np.linspace(1.0, 25.0, 300)
    f, s = 1000.0, 0.12
    for lm, c in ((3.0, "#d94a4a"), (9.0, "#e08a1e"), (30.0, "#2f6f3f")):
        ax.plot(z, (f * s / z) / lm, color=c, lw=1.8, label=r"$\lambda_{\min}=%g$px" % lm)
    ax.axhline(1.0, color="#333333", ls="--", lw=1.0)
    ax.text(24.5, 1.2, r"$\tau_{high}$", fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("depth z (m)")
    ax.set_ylabel(r"$\eta$")
    ax.set_title("(c) cung 1 Gaussian (s=0.12m): eta doi theo view\n-> phai bo phieu da view",
                 fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    save(fig, "16_eta_geometry.png")


# ------------------------------------------------------------------ FIG 3
def fig_multiview_vote():
    """10 view bo phieu -> high_ratio / low_ratio -> mask. Hai Gaussian doi lap."""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 0.95, 1.25], "wspace": 0.32})

    # gia tri CHON TAY de thay ro hai ben nguong 0.8
    etaA = np.array([2.20, 1.35, 3.10, 1.18, 0.42, 1.72, 2.65, 1.44, 1.09, 4.60])  # 9/10 HIGH
    etaB = np.array([0.06, 0.03, 0.31, 0.05, 0.02, 0.07, 0.04, 0.08, 0.03, 0.05])  # 9/10 LOW

    # (a) bang bo phieu
    ax = axes[0]
    M = 10
    for row, (eta, name) in enumerate([(etaA, "GS A"), (etaB, "GS B")]):
        for k, e in enumerate(eta):
            c = "#d94a4a" if e > 1.0 else ("#e8d36a" if e > 0.1 else "#2f6f3f")
            ax.add_patch(Rectangle((k, -row), 0.9, 0.9, fc=c, ec="#333333", lw=0.7))
            ax.text(k + 0.45, -row + 0.45, "%.2f" % e, ha="center", va="center",
                    fontsize=6.4, color="#333333" if c == "#e8d36a" else "white")
        ax.text(-0.25, -row + 0.45, name, ha="right", va="center", fontsize=9,
                fontweight="bold")
        ax.text(M + 0.25, -row + 0.45,
                "high=%d/10\nlow=%d/10" % ((eta > 1.0).sum(), (eta < 0.1).sum()),
                ha="left", va="center", fontsize=7.2)
    ax.set_xlim(-2.4, M + 2.6)
    ax.set_ylim(-2.9, 1.9)
    ax.axis("off")
    ax.text(M / 2, 1.35, "eta_max do o 10 view  (1 chu ky densify = 100 iter / 10)",
            ha="center", fontsize=8.2)
    for lab, c, x in (("HIGH > 1.0", "#d94a4a", -1.0),
                      ("MID", "#e8d36a", 3.2),
                      ("LOW < 0.1", "#2f6f3f", 6.2)):
        ax.add_patch(Rectangle((x, -2.2), 0.9, 0.9, fc=c, ec="#333333", lw=0.6))
        ax.text(x + 1.1, -1.75, lab, fontsize=7.6, va="center")
    ax.set_title("(a) pha DO: moi 10 iter cong 1 phieu", fontsize=9.5)

    # (b) ratio
    ax = axes[1]
    hr = [(etaA > 1.0).mean(), (etaB > 1.0).mean()]
    lr = [(etaA < 0.1).mean(), (etaB < 0.1).mean()]
    x = np.arange(2)
    b1 = ax.bar(x - 0.18, hr, 0.34, label="high_ratio", color="#d94a4a", ec="#333333")
    b2 = ax.bar(x + 0.18, lr, 0.34, label="low_ratio", color="#2f6f3f", ec="#333333")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02,
                    "%.1f" % r.get_height(), ha="center", fontsize=7.5)
    ax.axhline(0.8, color="#333333", ls="--", lw=1.3)
    ax.text(1.48, 0.83, "nguong 0.8", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["GS A\n-> SPLIT", "GS B\n-> PRUNE"], fontsize=8.5)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=7.2, loc="upper center", ncol=2)
    ax.set_title("(b) pha QUYET DINH: moi 100 iter\ntrain.py:337-353", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)

    # (c) so do AND
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    box(ax, 0.1, 8.3, 4.5, 1.3, "high_ratio > 0.8", C_STAT, fs=7.4)
    box(ax, 5.2, 8.3, 4.7, 1.3, r"$\|\nabla_{x_{2D}}L\| \geq 10^{-5}$", C_GEO, fs=7.4)
    box(ax, 3.3, 5.9, 3.4, 1.1, "AND", "#ffffff", fs=9)
    box(ax, 1.4, 3.4, 7.2, 1.4, "SPLIT di huong\ngaussian_model.py:698", C_ADC, fs=7.4)
    arrow(ax, (2.3, 8.3), (4.3, 7.1))
    arrow(ax, (7.5, 8.3), (5.7, 7.1))
    arrow(ax, (5.0, 5.9), (5.0, 4.9))
    box(ax, 0.1, 0.9, 9.8, 1.4,
        "low_ratio > 0.8 AND accum_view_count > 0\n->  PRUNE", C_ADC, fs=7.4)
    ax.text(5, 0.1, "AND (khong phai OR) = chot chan chong bung no N",
            ha="center", fontsize=7.4, color="#a03333")
    ax.set_title("(c) hai cong quyet dinh", fontsize=9.5)
    save(fig, "16_multiview_vote.png")


# ------------------------------------------------------------------ FIG 4
def fig_adc_actions():
    """Hanh dong ADC nhin thay duoc: split dang huong 3DGS vs di huong SADGS."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 2.9), gridspec_kw={"wspace": 0.18})
    ang = 25.0
    th = np.deg2rad(ang)

    ax = axes[0]
    ax.add_patch(Ellipse((0, 0), 3.2, 1.1, angle=ang, fc="#e7eef7",
                         ec="#1f6fd0", lw=1.8, alpha=0.85))
    ax.set_title("(a) TRUOC: eta = (4.0, 1.3, 0.2)\n1 Gaussian det, vi pham manh theo truc 1",
                 fontsize=9)

    ax = axes[1]
    for t in (-0.62, 0.62):
        ax.add_patch(Ellipse((t * 1.6 * np.cos(th), t * 1.6 * np.sin(th)),
                             3.2 / 1.6, 1.1 / 1.6, angle=ang,
                             fc="#fde8e8", ec="#d94a4a", lw=1.6))
    ax.set_title("(b) 3DGS: split dang huong\nN = 2, moi truc deu chia 1.6", fontsize=9)

    ax = axes[2]
    k = (2, 2, 1)
    ux, uy = np.cos(th), np.sin(th)
    vx, vy = -np.sin(th), np.cos(th)
    for i in range(k[0]):
        for j in range(k[1]):
            a = (i - (k[0] - 1) / 2) * (3.2 / k[0]) * 0.95
            b = (j - (k[1] - 1) / 2) * (1.1 / k[1]) * 0.95
            ax.add_patch(Ellipse((a * ux + b * vx, a * uy + b * vy),
                                 3.2 / k[0], 1.1 / k[1], angle=ang,
                                 fc="#e3f2e3", ec="#2f6f3f", lw=1.6))
    ax.set_title("(c) SADGS: k_j = ceil(sqrt(eta_j)) = (2,2,1)\n"
                 "N_con = kx*ky*kz = 4,   s_j -> s_j / k_j^p", fontsize=9)

    for ax in axes:
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#bbbbbb")
    axes[1].text(0.5, -0.13,
                 "Doc tu code: gaussian_model.py:701 chi clamp(ks, min=1) - KHONG co tran 8 nhu "
                 "comment hua; chot chan duy nhat la nguong gradient 1e-5 (train.py:333).",
                 transform=axes[1].transAxes, ha="center", va="top",
                 fontsize=7.6, color="#a03333")
    save(fig, "16_adc_actions.png")


if __name__ == "__main__":
    fig_flow()
    fig_eta_geometry()
    fig_multiview_vote()
    fig_adc_actions()
