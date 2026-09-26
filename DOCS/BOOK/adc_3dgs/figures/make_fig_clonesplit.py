"""
Hình minh hoạ Clone / Split dị hướng / kế toán N trong ADC của SADGS
(Structure-Aware Densification, SIGGRAPH 2026) — xem DOCS/BOOK/adc_3dgs/adc_3dgs_explained.md mục 4.
Chạy:  python DOCS/BOOK/adc_3dgs/figures/make_fig_clonesplit.py
Xuất:  fig_08_clone_before_after.png, fig_09_split_geometry.png,
       fig_10_split_samples.png, fig_11_N_accounting.png (cùng thư mục script)
Chỉ dùng numpy + matplotlib, seed cố định.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 3
rng = np.random.default_rng(SEED)

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["savefig.dpi"] = 170

C_OLD = "#1f77b4"     # Gaussian gốc
C_NEW = "#d62728"     # con 1
C_NEW2 = "#ff7f0e"    # con 2
C_OBJ = "#444444"     # viền vật thể
C_GREY = "#999999"

KS_SCALE_POWER = 1.0   # SADGS args.ks_scale_power: s_new = s_old / k^power
saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    saved.append(path)


def rot2d(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def gauss_ellipse(ax, mu, s, deg, k=2.0, **kw):
    """Ellipse mức k-sigma của Gaussian 2D scale s=(sx,sy), xoay deg độ."""
    e = Ellipse(mu, width=2 * k * s[0], height=2 * k * s[1], angle=deg, **kw)
    ax.add_patch(e)
    return e


# ---------------------------------------------------------------------------
# Hình 8: Clone trước / sau
# ---------------------------------------------------------------------------
def fig08():
    s = (0.006, 0.003)
    deg = 30.0
    alpha = 0.6
    mu = np.array([0.100, 0.050])

    fig = plt.figure(figsize=(10.2, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.7])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_t = fig.add_subplot(gs[0, 2])
    ax_t.axis("off")

    # đường viền vật thể: cạnh chi tiết (đường cong nhỏ + góc nhọn)
    xs = np.linspace(0.075, 0.130, 200)
    edge = 0.043 + 0.10 * (xs - 0.075) + 0.004 * np.sin((xs - 0.075) * 250)

    for ax, title in zip(axes, ["Trước clone: N = 1", "Sau clone: N = 2"]):
        ax.plot(xs, edge, color=C_OBJ, lw=1.6, zorder=1)
        ax.fill_between(xs, edge, 0.030, color="#e8e8e8", zorder=0)
        ax.text(0.129, 0.0312, "vật thể (cạnh chi tiết)", ha="right", va="bottom",
                fontsize=7.5, color=C_OBJ)
        ax.set_xlim(0.075, 0.130)
        ax.set_ylim(0.030, 0.070)
        ax.set_aspect("equal")
        ax.set_xlabel("x (đơn vị cảnh)")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("y (đơn vị cảnh)")

    # panel trái: 1 Gaussian
    gauss_ellipse(axes[0], mu, s, deg, fc=C_OLD, ec=C_OLD, alpha=alpha, lw=1.2, zorder=3)
    axes[0].plot(*mu, "o", color="k", ms=3, zorder=4)
    axes[0].annotate(r"$\theta_i$: $\mu_i$, $s_i$=(0.006, 0.003)," "\n" r"$q_i$=30°, $\alpha_i$=0.6, SH$_i$",
                     xy=mu, xytext=(0.079, 0.063), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8), va="top")
    axes[0].text(0.079, 0.0335,
                 "split_signal (multiview-consistency η + is_grad_high)\n"
                 "và importance $>$ threshold và max($s_i$) $\\leq$ dense·extent"
                 "\n→ Gaussian NHỎ dưới tái tạo → CLONE",
                 fontsize=7.5, va="bottom", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    # panel phải: 2 Gaussian trùng vị trí (vẽ lệch 0.0008 để nhìn thấy)
    off = np.array([0.0006, -0.0006])
    gauss_ellipse(axes[1], mu, s, deg, fc=C_OLD, ec=C_OLD, alpha=alpha, lw=1.2, zorder=3)
    gauss_ellipse(axes[1], mu + off, s, deg, fc=C_NEW, ec=C_NEW, alpha=alpha, lw=1.2,
                  ls="--", zorder=4)
    axes[1].plot(*mu, "o", color="k", ms=3, zorder=5)
    axes[1].annotate("gốc $\\theta_i$ (giữ nguyên)", xy=mu, xytext=(0.079, 0.066),
                     fontsize=8, color=C_OLD, arrowprops=dict(arrowstyle="->", lw=0.8, color=C_OLD))
    axes[1].annotate("bản sao $\\theta_{new}=\\theta_i$", xy=mu + off, xytext=(0.108, 0.066),
                     fontsize=8, color=C_NEW, arrowprops=dict(arrowstyle="->", lw=0.8, color=C_NEW))
    axes[1].text(0.079, 0.0335,
                 "Hai Gaussian TRÙNG vị trí\n(vẽ lệch một chút chỉ để nhìn thấy).\n"
                 "Tách ra nhờ gradient ở các\nvòng tối ưu sau.",
                 fontsize=7.5, va="bottom", color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    # bảng tham số
    rows = [("μ", "(0.100, 0.050)", "(0.100, 0.050)"),
            ("s", "(0.006, 0.003)", "(0.006, 0.003)"),
            ("q (góc)", "30°", "30°"),
            ("α", "0.6", "0.6"),
            ("SH", "c_i", "c_i")]
    ax_t.text(0.5, 0.93, "Bảng tham số: θ_new = θ_i", ha="center", fontsize=9, transform=ax_t.transAxes)
    tbl = ax_t.table(cellText=[[a, b, c] for a, b, c in rows],
                     colLabels=["", "gốc θ_i", "mới θ_new"], colWidths=[0.22, 0.39, 0.39],
                     cellLoc="center", colLoc="center",
                     bbox=[0.0, 0.30, 1.0, 0.55], zorder=6)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    ax_t.text(0.5, 0.20, "Mọi trường sao chép y nguyên:\nμ, s, q, α, hệ số SH.\n"
              "Không đổi scale, không xoá gốc.\n\nN: 1 → 2",
              ha="center", va="top", fontsize=8, transform=ax_t.transAxes)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#f0f0f0")
        elif c == 2:
            cell.set_text_props(color=C_NEW)
    fig.suptitle("Clone: sao chép y nguyên θ, scale giữ, gốc giữ  →  N: 1 → 2", fontsize=10)
    save(fig, "fig_08_clone_before_after.png")


# ---------------------------------------------------------------------------
# Hình 9: hình học của split DỊ HƯỚNG (densify_and_split_structgs)
# ---------------------------------------------------------------------------
def grid_children(s, deg, kx, ky, power=KS_SCALE_POWER):
    """Mirror densify_and_split_structgs: lưới đều k_x*k_y con, KHÔNG lấy mẫu
    ngẫu nhiên. separation_a = (s_a/k_a)*sqrt(12), scale_new = s/k^power."""
    R = rot2d(deg)
    s_child = s / np.array([kx, ky], dtype=float) ** power
    sep = (s / np.array([kx, ky], dtype=float)) * np.sqrt(12.0)
    gx = np.arange(kx) - (kx - 1) / 2.0
    gy = np.arange(ky) - (ky - 1) / 2.0
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    local_offsets = grid * sep
    world_offsets = (R @ local_offsets.T).T
    return world_offsets, s_child, R


def fig09():
    s = np.array([0.06, 0.02])       # s_x, s_y của Gaussian gốc
    deg = 40.0
    mu = np.array([0.0, 0.0])
    # eta_max quan sát được: trục x aliasing mạnh hơn trục y -> kx > ky
    eta_x, eta_y = 7.0, 2.2
    kx = int(np.clip(np.ceil(np.sqrt(eta_x)), 1, 8))   # = 3
    ky = int(np.clip(np.ceil(np.sqrt(eta_y)), 1, 8))   # = 2
    offsets, s_child, R = grid_children(s, deg, kx, ky)
    children = mu + offsets
    n_children = kx * ky

    fig = plt.figure(figsize=(10.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.85])
    ax_l = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[0, 2])
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_children))

    # ---- panel local: lưới đều k_x * k_y trong hệ toạ độ riêng
    ax_l.set_title(f"Khung local: lưới $k_x{{\\times}}k_y$ = {kx}$\\times${ky} con")
    gx = np.arange(kx) - (kx - 1) / 2.0
    gy = np.arange(ky) - (ky - 1) / 2.0
    sep = (s / np.array([kx, ky], dtype=float)) * np.sqrt(12.0)
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2) * sep
    gauss_ellipse(ax_l, (0, 0), s, 0, k=1, fc="none", ec=C_OLD, lw=1.2, ls="--")
    for j in range(n_children):
        ax_l.plot(grid[j, 0], grid[j, 1], "s", color=colors[j], ms=7, zorder=5)
    ax_l.annotate("", xy=(0.10, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax_l.annotate("", xy=(0, 0.10), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax_l.text(0.102, -0.006, "$x_{loc}$", fontsize=7.5, va="top")
    ax_l.text(-0.006, 0.102, "$y_{loc}$", fontsize=7.5, ha="right", va="bottom")
    ax_l.set_xlim(-0.15, 0.17)
    ax_l.set_ylim(-0.15, 0.15)
    ax_l.set_aspect("equal")
    ax_l.grid(alpha=0.25, lw=0.5)
    ax_l.text(0.02, 0.02,
              f"$\\eta_x$={eta_x:.1f} → $k_x$=⌈√{eta_x:.1f}⌉={kx}\n$\\eta_y$={eta_y:.1f} → $k_y$=⌈√{eta_y:.1f}⌉={ky}",
              transform=ax_l.transAxes, fontsize=7.5,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_GREY, lw=0.5))

    # ---- panel world
    ax_w.set_title("Khung thế giới: $\\mu^{(j)} = \\mu_i + R(q_i)\\,$offset$^{(j)}$")
    tt = np.linspace(-0.13, 0.13, 200)
    curve = np.stack([tt, 2.2 * tt ** 2 - 0.012], 1) @ rot2d(deg).T
    ax_w.plot(curve[:, 0], curve[:, 1], color=C_OBJ, lw=2.0, zorder=1, label="chi tiết texture mịn")
    gauss_ellipse(ax_w, mu, s, deg, k=2, fc=C_OLD, ec=C_OLD, alpha=0.10, lw=1.4, ls="--", zorder=2)
    gauss_ellipse(ax_w, mu, s, deg, k=1, fc="none", ec=C_OLD, lw=0.8, ls="--", zorder=2)
    ax_w.plot(*mu, "x", color=C_OLD, ms=8, mew=2, zorder=6)
    tip = mu + R[:, 0] * 0.115
    ax_w.annotate(f"gốc $\\theta_i$ (nét đứt, 2σ)\nbị XOÁ, thay bằng {n_children} con", xy=tip,
                  xytext=(0.24, 0.19), fontsize=8, color=C_OLD, ha="right", va="top",
                  arrowprops=dict(arrowstyle="->", color=C_OLD, lw=0.8))
    for vec, lab, dx in [(R[:, 0] * 0.09, "$Re_x$ ($k_x$={})".format(kx), (0.10, -0.01)),
                         (R[:, 1] * 0.06, "$Re_y$ ($k_y$={})".format(ky), (-0.03, 0.0))]:
        ax_w.annotate("", xy=vec, xytext=mu, arrowprops=dict(arrowstyle="->", lw=1.3, color="#555555"))
        ax_w.text(vec[0] + dx[0], vec[1] + dx[1], lab, fontsize=7.5, color="#555555", ha="center", zorder=8,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    for j in range(n_children):
        gauss_ellipse(ax_w, children[j], s_child, deg, k=2, fc=colors[j], ec=colors[j], alpha=0.35, lw=1.0, zorder=4)
        gauss_ellipse(ax_w, children[j], s_child, deg, k=1, fc="none", ec=colors[j], lw=0.7, zorder=4)
        ax_w.plot(children[j, 0], children[j, 1], "s", color=colors[j], ms=5, zorder=7)
    ax_w.text(0.02, 0.98,
              f"con: $s$ = $s_i$/({kx},{ky})$^{{{KS_SCALE_POWER:.1f}}}$\n"
              f"    = ({s_child[0]:.4f}, {s_child[1]:.4f})\ngóc $q$ giữ = 40°, lưới đều tâm 0",
              transform=ax_w.transAxes, fontsize=8, ha="left", va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))
    ax_w.set_xlim(-0.25, 0.25)
    ax_w.set_ylim(-0.21, 0.21)
    ax_w.set_aspect("equal")
    ax_w.grid(alpha=0.25, lw=0.5)
    ax_w.legend(loc="lower left", fontsize=7.5)

    # ---- panel công thức
    ax_f.axis("off")
    ax_f.set_title("Công thức split dị hướng")
    txt = (
        "$k_a = \\mathrm{clip}(\\lceil\\sqrt{\\eta_{\\max,a}}\\rceil,\\,1,\\,8)$\n"
        "separation$_a = (s_a/k_a)\\sqrt{12}$\n"
        "offset$^{(j)}_{\\text{local}}$ = toạ độ lưới đều tâm 0 $\\times$ separation\n"
        "$\\mu^{(j)} = \\mu_i + R(q_i)\\,$offset$^{(j)}$\n"
        f"$s^{{(j)}} = s_i / k^{{{KS_SCALE_POWER:.1f}}}$,   $q^{{(j)}} = q_i$\n"
        "$\\alpha^{(j)} = \\alpha_i$,   SH$^{(j)}$ = SH$_i$\n"
        f"Gốc $\\theta_i$ bị xoá  →  N: 1 → {n_children}  (+{n_children-1})\n\n"
        f"Ví dụ trong hình: $k_x$={kx}, $k_y$={ky} → {n_children} con\n"
        f"$s_i$ = ({s[0]:.3f}, {s[1]:.3f})\n"
        f"$s^{{(j)}}$ = ({s_child[0]:.4f}, {s_child[1]:.4f})\n\n"
        "(khác 3DGS gốc: LƯỚI ĐỀU, không phải mẫu\n"
        "ngẫu nhiên $\\mathcal{N}(0,\\mathrm{diag}(s^2))$; số con theo\n"
        "TỪNG TRỤC, không cố định 2)"
    )
    ax_f.text(0.0, 0.97, txt, va="top", ha="left", fontsize=8, transform=ax_f.transAxes,
              linespacing=1.55)
    fig.suptitle(f"Split dị hướng SADGS: $k_a=\\lceil\\sqrt{{\\eta_{{\\max,a}}}}\\rceil$ mỗi trục "
                 f"→ {kx}$\\times${ky}={n_children} con trên lưới đều (khác 2 con ngẫu nhiên của 3DGS gốc)",
                 fontsize=9.5)
    save(fig, "fig_09_split_geometry.png")
    return offsets, children, kx, ky


# ---------------------------------------------------------------------------
# Hình 10: quét eta -> k trên cùng một Gaussian gốc (thay Monte Carlo mẫu ngẫu nhiên)
# ---------------------------------------------------------------------------
def fig10():
    s = np.array([0.06, 0.02])
    deg = 40.0
    eta_cases = [(1.2, 1.2), (4.0, 1.5), (9.0, 4.0)]   # (eta_x, eta_y) tăng dần

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    all_area_ratio = []
    all_k = []
    for ax, (eta_x, eta_y) in zip(axes, eta_cases):
        kx = int(np.clip(np.ceil(np.sqrt(eta_x)), 1, 8))
        ky = int(np.clip(np.ceil(np.sqrt(eta_y)), 1, 8))
        offsets, s_child, R = grid_children(s, deg, kx, ky)
        n_children = kx * ky
        area_ratio = n_children * (1.0 / (kx * ky) ** (2 * KS_SCALE_POWER))
        all_area_ratio.append(area_ratio)
        all_k.append((kx, ky))

        gauss_ellipse(ax, (0, 0), s, deg, k=1, fc="none", ec=C_OLD, lw=1.3, ls="--", label="gốc 1σ")
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_children))
        for j in range(n_children):
            gauss_ellipse(ax, offsets[j], s_child, deg, k=1, fc=colors[j], ec=colors[j],
                          alpha=0.55, lw=0.8, zorder=3)
            ax.plot(offsets[j, 0], offsets[j, 1], ".", color="black", ms=3, zorder=4)
        ax.plot(0, 0, "x", color="k", ms=7, mew=1.5, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-0.13, 0.13)
        ax.set_ylim(-0.13, 0.13)
        ax.set_title(f"$\\eta$=({eta_x:.1f},{eta_y:.1f}) → $k$=({kx},{ky}) → {n_children} con")
        ax.text(0.02, 0.02,
                f"diện tích {n_children} con / gốc = {area_ratio:.3f}",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_GREY, lw=0.5))
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("y (đơn vị cảnh)")
    axes[1].set_xlabel("x (đơn vị cảnh) — cùng Gaussian gốc $s_i$=(0.06, 0.02), $q_i$=40°, khác $\\eta$ quan sát")

    fig.suptitle("Quét $\\eta$: số con và cách chia lưới thay đổi theo mức aliasing đo được mỗi trục "
                 "(khác 3DGS gốc: luôn đúng 2 con bất kể mức độ)", fontsize=10)
    save(fig, "fig_10_split_samples.png")
    return dict(k_list=all_k, area_ratio=all_area_ratio)


# ---------------------------------------------------------------------------
# Hình 11: kế toán N
# ---------------------------------------------------------------------------
def fig11():
    # ví dụ khớp mục 4.4 của adc_3dgs_explained.md: 40 Gaussian split sinh 132 con
    # (không phải 2*40=80, vì k_x*k_y thay đổi theo eta từng Gaussian)
    N_old, n_clone, n_split_parents, n_split_children, n_prune = 1000, 60, 40, 132, 45
    n_split_net = n_split_children - n_split_parents
    steps = [("N cũ", N_old), ("+ clone\n(+n_clone)", n_clone),
             ("+ con split\n(+n_split_children)", n_split_children),
             ("− gốc split\n(−n_split_parents)", -n_split_parents),
             ("− prune\n(−n_prune)", -n_prune)]
    N_new = N_old + n_clone + n_split_net - n_prune

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # ---- waterfall
    run = 0
    xs = np.arange(len(steps) + 1)
    for k, (lab, v) in enumerate(steps):
        if k == 0:
            ax.bar(k, v, color=C_OLD, width=0.6)
            ax.text(k, v + 12, f"{v}", ha="center", fontsize=8.5)
            run = v
        else:
            col = "#2ca02c" if v > 0 else "#d62728"
            bottom = run if v > 0 else run + v
            ax.bar(k, abs(v), bottom=bottom, color=col, width=0.6)
            ax.plot([k - 1 + 0.3, k - 0.3], [run, run], color=C_GREY, lw=0.8, ls=":")
            ax.text(k, run + max(v, 0) + 12, f"{v:+d}", ha="center", fontsize=8.5, color=col)
            run += v
    ax.plot([len(steps) - 1 + 0.3, len(steps) - 0.3], [run, run], color=C_GREY, lw=0.8, ls=":")
    ax.bar(len(steps), N_new, color=C_OLD, width=0.6)
    ax.text(len(steps), N_new + 12, f"{N_new}", ha="center", fontsize=8.5, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps] + ["N mới"], fontsize=8)
    ax.set_ylim(900, 1330)
    ax.set_ylabel("số Gaussian N")
    ax.set_title("Kế toán N trong MỘT bước densify + prune")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.text(0.52, 0.04,
            "$N_{new} = N + n_{clone} + (n_{split\\_children} - n_{split\\_parents}) - n_{prune}$\n"
            f"= {N_old} + {n_clone} + ({n_split_children} − {n_split_parents}) − {n_prune} = {N_new}\n"
            "(split dị hướng: số con $k_x{\\times}k_y$ khác nhau mỗi Gaussian, không cố định +1)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.6))

    # ---- tăng trưởng N(t)
    n_steps = 144
    r = 0.015
    N0 = 100_000
    t = np.arange(n_steps + 1)
    N_exp = N0 * (1 + r) ** t
    N_lin = N0 * (1 + r * t)
    ax2.semilogy(t, N_exp, color=C_NEW, lw=1.8, label=f"luỹ thừa $N_0(1+r)^t$, r = 1.5%")
    ax2.semilogy(t, N_lin, color=C_OLD, lw=1.8, ls="--", label="tuyến tính $N_0(1+rt)$")
    ax2.axhline(N0, color=C_GREY, lw=0.8, ls=":")
    ax2.set_xlabel("bước densify t (mỗi 100 vòng, 500 < iter < 15000)")
    ax2.set_ylabel("N (log)")
    ax2.set_title("N(t) sau 144 bước densify: luỹ thừa vs tuyến tính")
    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(8e4, 1.5e6)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3, lw=0.5, which="both")
    ax2.annotate(f"t=144: {N_exp[-1]/1e3:,.0f}k  (×{N_exp[-1]/N0:.2f})", xy=(n_steps, N_exp[-1]),
                 xytext=(88, 1.05e6), fontsize=8, color=C_NEW,
                 arrowprops=dict(arrowstyle="->", color=C_NEW, lw=0.8))
    ax2.annotate(f"t=144: {N_lin[-1]/1e3:,.0f}k  (×{N_lin[-1]/N0:.2f})", xy=(n_steps, N_lin[-1]),
                 xytext=(60, 1.55e5), fontsize=8, color=C_OLD,
                 arrowprops=dict(arrowstyle="->", color=C_OLD, lw=0.8))
    ax2.text(2, 8.6e4, f"$N_0$ = {N0//1000}k", fontsize=8, color=C_GREY)
    for tm in (72,):
        ax2.plot(tm, N_exp[tm], "o", color=C_NEW, ms=4)
        ax2.text(tm + 4, N_exp[tm] * 0.82, f"t={tm}: {N_exp[tm]/1e3:.0f}k", fontsize=7.5, color=C_NEW)

    fig.suptitle("Mỗi bước: clone +1, split ròng +($k_xk_y$−1) tuỳ Gaussian, prune −1  →  "
                 "N tăng nhanh hơn 3DGS gốc nếu η lớn ở nhiều trục",
                 fontsize=10)
    save(fig, "fig_11_N_accounting.png")
    return dict(N_new=N_new, N_exp_end=N_exp[-1], N_lin_end=N_lin[-1], N_exp_72=N_exp[72])


if __name__ == "__main__":
    fig08()
    offsets, ch, kx, ky = fig09()
    st = fig10()
    acc = fig11()
    print("Đã lưu:")
    for p in saved:
        print("  ", p)
    print("\nSố liệu ví dụ:")
    print(f"  fig09 kx,ky = {kx},{ky}; offsets:", np.round(offsets, 4).tolist())
    print("  fig09 mu con:", np.round(ch, 4).tolist())
    print("  fig10 k_list:", st["k_list"], "area_ratio:", [round(v, 4) for v in st["area_ratio"]])
    print("  fig11:", {k: round(float(v), 1) for k, v in acc.items()})
