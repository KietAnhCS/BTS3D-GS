# -*- coding: utf-8 -*-
"""
BOT 05 — densify_and_clone_structgs & cơ chế ghép tensor vào optimizer.
Nguồn code: SADGS/scene/gaussian_model.py
  - densify_and_clone_structgs   : 934-956
  - densify_and_clone (3DGS)     : 893-910
  - densification_postfix        : 595-637
  - cat_tensors_to_optimizer     : 566-594
  - replace_tensor_to_optimizer  : 485-502
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


# ----------------------------------------------------------------------
# HÌNH 1: sơ đồ tensor trước/sau khi cat  (N -> N+M)
# cat_tensors_to_optimizer (566-594): param, exp_avg, exp_avg_sq đều cat dim=0
# ----------------------------------------------------------------------
def fig_cat():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    rows = [("param  $\\theta$", "#4C78A8"),
            ("exp_avg  $m$", "#F58518"),
            ("exp_avg_sq  $v$", "#54A24B")]

    y0, hgt, wN, wM = 4.6, 0.62, 2.85, 1.15

    ax.text(0.15, 5.75, "TRƯỚC (N Gaussian)", fontsize=11, fontweight="bold")
    for i, (lab, c) in enumerate(rows):
        y = y0 - i * 1.0
        ax.add_patch(Rectangle((1.75, y), wN, hgt, fc=c, ec="k", alpha=.75))
        ax.text(1.70, y + hgt / 2, lab, ha="right", va="center", fontsize=9)
        ax.text(1.75 + wN / 2, y + hgt / 2, "N", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")

    ax.add_patch(FancyArrowPatch((4.95, 3.60), (6.10, 3.60),
                                 arrowstyle="-|>", mutation_scale=18, lw=2, color="k"))
    ax.text(5.52, 4.15, "torch.cat(dim=0)", ha="center", fontsize=8.5)
    ax.text(5.52, 3.02, "+ M dòng", ha="center", fontsize=8.5, color="#B03A2E")

    ax.text(6.2, 5.75, "SAU (N+M Gaussian)", fontsize=11, fontweight="bold")
    for i, (lab, c) in enumerate(rows):
        y = y0 - i * 1.0
        ax.add_patch(Rectangle((6.2, y), wN, hgt, fc=c, ec="k", alpha=.75))
        ax.text(6.2 + wN / 2, y + hgt / 2, "N (giữ nguyên)", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        if i == 0:
            ax.add_patch(Rectangle((6.2 + wN, y), wM, hgt, fc="#B03A2E", ec="k", alpha=.85))
            ax.text(6.2 + wN + wM / 2, y + hgt / 2, "M: copy", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
        else:
            ax.add_patch(Rectangle((6.2 + wN, y), wM, hgt, fc="white", ec="k", hatch="//"))
            ax.text(6.2 + wN + wM / 2, y + hgt / 2, "M: 0", ha="center", va="center",
                    color="k", fontsize=9, fontweight="bold")

    txt = ("$\\theta' = \\mathrm{cat}(\\theta,\\ \\theta[\\mathcal{S}])$      "
           "$m' = \\mathrm{cat}(m,\\ \\mathbf{0}_M)$      "
           "$v' = \\mathrm{cat}(v,\\ \\mathbf{0}_M)$")
    ax.text(5.3, 1.40, txt, ha="center", fontsize=11)
    ax.text(5.3, 0.70,
            'del opt.state[p]  ->  group["params"][0] = nn.Parameter(cat)  ->  opt.state[p_new] = state\n'
            "(gaussian_model.py:578-586 — phải dựng lại param group vì id(Parameter) là khoá của opt.state)",
            ha="center", fontsize=8.6, color="#333333")
    ax.set_title("cat_tensors_to_optimizer: ghép M Gaussian mới vào Adam (566-594)", fontsize=12)
    save(fig, "05_cat_tensor_optimizer.png")


# ----------------------------------------------------------------------
# HÌNH 2: minh hoạ 2D vùng under-reconstruction được clone
# clone = copy nguyên xyz (937), KHÔNG dịch vị trí
# ----------------------------------------------------------------------
def fig_clone_2d():
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.1))
    xs = np.linspace(-1.6, 1.6, 400)
    for ax, title in zip(axs, ["Trước clone: under-reconstruction",
                               "Sau densify_and_clone_structgs"]):
        ax.plot(xs, 0.55 * np.sin(2.1 * xs) + 0.15, color="#888888", lw=6, alpha=.35,
                solid_capstyle="round")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=10.5)

    centers = [(-1.0, -0.6), (-0.25, 0.42), (0.55, -0.35), (1.15, 0.55)]
    sel = [False, True, True, False]   # metric_mask AND filter

    for (cx, cy), s in zip(centers, sel):
        ec = "#B03A2E" if s else "#4C78A8"
        axs[0].add_patch(Ellipse((cx, cy), 0.75, 0.48, angle=20, fc=ec, ec="k", alpha=.35))
        axs[0].plot(cx, cy, "k.", ms=5)
        if s:
            axs[0].annotate("chọn", (cx, cy), textcoords="offset points",
                            xytext=(0, 20), ha="center", fontsize=8, color="#B03A2E")

    for (cx, cy), s in zip(centers, sel):
        ec = "#B03A2E" if s else "#4C78A8"
        axs[1].add_patch(Ellipse((cx, cy), 0.75, 0.48, angle=20, fc=ec, ec="k", alpha=.30))
        if s:
            axs[1].add_patch(Ellipse((cx, cy), 0.75, 0.48, angle=20, fc="none",
                                     ec="#B03A2E", lw=2.2, ls="--"))
            axs[1].annotate("bản sao trùng khít\n(cùng $\\mu, s, q, \\alpha$, SH)", (cx, cy),
                            textcoords="offset points", xytext=(0, 24), ha="center",
                            fontsize=7.4, color="#B03A2E")
        axs[1].plot(cx, cy, "k.", ms=5)

    axs[1].text(0, -1.32,
                "Không có nhiễu vị trí: $\\mu_{new} = \\mu_{old}$ (gaussian_model.py:937)\n"
                "chỉ gradient các bước sau mới tách hai bản sao ra",
                ha="center", fontsize=8.0)
    fig.suptitle("Clone vùng under-reconstruction: $\\mathcal{S}$ = metric_mask $\\wedge$ filter (934-935)",
                 fontsize=11.5)
    fig.tight_layout()
    save(fig, "05_clone_under_reconstruction.png")


# ----------------------------------------------------------------------
# HÌNH 3: vùng quyết định 2 trục — 3DGS vs SADGS
# 3DGS  (893-897): ||g|| >= densify_grad_threshold AND max(s) <= percent_dense*extent
# SADGS (990, 998-1007): (custom_split_mask OR ||g||>=args.grad_thresh)
#                        AND max(s) <= args.dense*extent   AND metric_mask (1005)
# ----------------------------------------------------------------------
def fig_decision():
    extent = 5.0
    tau_g = 0.0002          # opt.grad_thresh (arguments/__init__.py:112)
    pd_3dgs = 0.01          # percent_dense mặc định gốc 3DGS
    dense_sadgs = 0.001     # args.dense (arguments/__init__.py:113)

    s_lo, s_hi = 1e-3, 1e-1
    g = np.linspace(0, 6e-4, 500)
    s = np.logspace(np.log10(s_lo), np.log10(s_hi), 500)
    G, S = np.meshgrid(g, s)

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)

    # --- 3DGS ---
    M3 = (G >= tau_g) & (S <= pd_3dgs * extent)
    axs[0].contourf(G * 1e4, S, M3.astype(float), levels=[.5, 1.5], colors=["#4C78A8"], alpha=.45)
    axs[0].axvline(tau_g * 1e4, color="k", ls="--", lw=1.3)
    axs[0].axhline(pd_3dgs * extent, color="k", ls=":", lw=1.3)
    axs[0].text(2.15, 6.8e-2, "grad $\\geq 2\\times 10^{-4}$", fontsize=8.5)
    axs[0].text(0.15, 5.4e-2, "max(s) $\\leq$ percent_dense $\\times$ extent = 0.05", fontsize=8)
    axs[0].text(4.0, 6e-3, "CLONE", fontsize=13, color="#26456E", fontweight="bold", ha="center")
    axs[0].set_title("3DGS: densify_and_clone (893-897)", fontsize=10.5)

    # --- SADGS ---
    MS = (G >= tau_g) & (S <= dense_sadgs * extent)
    MX = (S <= dense_sadgs * extent)
    axs[1].contourf(G * 1e4, S, MS.astype(float), levels=[.5, 1.5], colors=["#B03A2E"], alpha=.5)
    axs[1].contourf(G * 1e4, S, (MX & ~MS).astype(float), levels=[.5, 1.5],
                    colors=["#F58518"], alpha=.5)
    axs[1].axvline(tau_g * 1e4, color="k", ls="--", lw=1.3)
    axs[1].axhline(dense_sadgs * extent, color="k", ls=":", lw=1.3)
    axs[1].text(0.15, 5.8e-3, "max(s) $\\leq$ args.dense $\\times$ extent = 0.005", fontsize=8)
    axs[1].text(4.0, 1.7e-3, "grad", fontsize=8.5, color="#7B1D12", ha="center")
    axs[1].text(1.0, 1.7e-3, "eta", fontsize=8.5, color="#8A4B00", ha="center")
    axs[1].text(0.25, 1.4e-2,
                "cam: custom_split_mask (high_ratio > ngưỡng)\n"
                "mở CLONE ngay cả khi grad < ngưỡng;\n"
                "toàn bộ còn nhân thêm metric_mask\n"
                "(accum_view_count > 0.5)",
                fontsize=8, color="#8A4B00")
    axs[1].set_title("SADGS: final_clone_mask (990, 998-1007)", fontsize=10.5)

    for ax in axs:
        ax.set_xlabel("$\\|\\nabla_{xy}\\|\\ (\\times 10^{-4})$")
        ax.set_xlim(0, 6)
        ax.set_yscale("log")
        ax.set_ylim(s_lo, s_hi)
        ax.grid(alpha=.25, which="both")
    axs[0].set_ylabel("$\\max_i s_i$ (world scale, log)")
    fig.suptitle("Vùng quyết định CLONE (extent = 5.0)", fontsize=12)
    fig.tight_layout()
    save(fig, "05_decision_region_clone.png")


# ----------------------------------------------------------------------
# HÌNH 4: ảnh hưởng reset moment Adam lên bước cập nhật
# cat_tensors_to_optimizer (578-579) append 0 vào exp_avg/exp_avg_sq,
# nhưng 'step' là scalar chung của cả tensor => bias-correction vẫn dùng t lớn.
# ----------------------------------------------------------------------
def fig_adam_reset():
    b1, b2, eps = 0.9, 0.999, 1e-15
    T = 60

    def run(t0, m0, v0, gfun):
        m, v = m0, v0
        out = []
        for k in range(1, T + 1):
            gk = gfun(k)
            m = b1 * m + (1 - b1) * gk
            v = b2 * v + (1 - b2) * gk * gk
            t = t0 + k
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            out.append(mh / (np.sqrt(vh) + eps))
        return np.array(out)

    g_const = lambda k: 1.0
    steps = np.arange(1, T + 1)

    parent = run(3000, 1.0, 1.0, g_const)          # Gaussian cha: moment đã "nóng"
    child_inherit = run(3000, 0.0, 0.0, g_const)   # bản sao: m=v=0 nhưng step kế thừa t lớn
    child_fresh = run(0, 0.0, 0.0, g_const)        # nếu giả định step cũng reset về 0

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))

    axs[0].plot(steps, parent, lw=2, color="#4C78A8", label="Gaussian cha ($m, v$ đã hội tụ)")
    axs[0].plot(steps, child_inherit, lw=2.2, color="#B03A2E",
                label="Bản sao: $m_0 = v_0 = 0$, $t$ kế thừa")
    axs[0].plot(steps, child_fresh, lw=1.6, ls="--", color="#54A24B",
                label="Nếu $t$ cũng reset (không xảy ra)")
    axs[0].axhline(1.0, color="k", lw=.8, ls=":")
    r = (1 - b1) / np.sqrt(1 - b2)
    axs[0].axhline(r, color="#B03A2E", lw=.8, ls=":")
    axs[0].text(22, r + .25, "bước đầu $= (1-\\beta_1)/\\sqrt{1-\\beta_2} = 3.16$",
                fontsize=8.5, color="#B03A2E")
    axs[0].set_xlabel("iteration sau densify")
    axs[0].set_ylabel("$|\\Delta\\theta| / \\eta$")
    axs[0].set_title("Gradient hằng: $\\hat{m}_t / \\sqrt{\\hat{v}_t}$", fontsize=10.5)
    axs[0].legend(fontsize=7.5, loc="lower right")
    axs[0].grid(alpha=.3)

    rng = np.random.default_rng(0)
    noise = 1.0 + 0.6 * rng.standard_normal(T + 1)
    g_noisy = lambda k: noise[k]
    p2 = run(3000, 1.0, 1.0, g_noisy)
    c2 = run(3000, 0.0, 0.0, g_noisy)
    axs[1].plot(steps, np.cumsum(np.abs(p2)), lw=2, color="#4C78A8", label="Gaussian cha")
    axs[1].plot(steps, np.cumsum(np.abs(c2)), lw=2.2, color="#B03A2E",
                label="Bản sao (moment bị reset)")
    axs[1].set_xlabel("iteration sau densify")
    axs[1].set_ylabel("$\\sum |\\Delta\\theta| / \\eta$")
    axs[1].set_title("Quãng dịch chuyển tích luỹ (gradient nhiễu)", fontsize=10.5)
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=.3)

    fig.suptitle("Hệ quả reset exp_avg / exp_avg_sq = 0 (gaussian_model.py:578-579)", fontsize=11.5)
    fig.tight_layout()
    save(fig, "05_adam_moment_reset.png")


if __name__ == "__main__":
    fig_cat()
    fig_clone_2d()
    fig_decision()
    fig_adam_reset()
