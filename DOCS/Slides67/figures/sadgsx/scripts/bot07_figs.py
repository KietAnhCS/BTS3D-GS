# -*- coding: utf-8 -*-
"""
BOT 07 — densify_and_prune_structgs: vong densify + prune hop nhat.
Moi hinh minh hoa dung mot cong thuc/co che lay tu code that:
  SADGS/scene/gaussian_model.py:957-1053 (densify_and_prune_structgs)
  SADGS/scene/gaussian_model.py:528-565  (prune_points)
  SADGS/scene/gaussian_model.py:503-527  (_prune_optimizer)
  SADGS/train.py:362-374                 (loi goi that)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.family"] = "DejaVu Sans"
OUT = r"C:/Users/kelly/OneDrive/Desktop/digital-twin-learn/DOCS/Slides67/figures/sadgsx"
rng = np.random.default_rng(7)


def save(fig, name):
    p = f"{OUT}/{name}"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------
# H1: so do khoi luong hàm densify_and_prune_structgs
# ----------------------------------------------------------------------
def fig1_flow():
    fig, ax = plt.subplots(figsize=(9.0, 5.9))
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(0, 12.1); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=7.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, txt="", c="#333333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.1, color=c))
        if txt:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, txt, fontsize=6.6,
                    color=c, ha="left", va="center")

    C1, C2, C3, C4 = "#dbeafe", "#dcfce7", "#fee2e2", "#fef9c3"

    box(0.4, 10.3, 9.2, 0.85,
        "densify_and_prune_structgs(max_screen_size, min_opacity, extent, radii, args,\n"
        "importance_score, pruning_score, custom_split_mask, custom_prune_mask, max_eta_3ch)   [gm.py:957]",
        "#e5e7eb", 7.0)
    box(0.4, 9.15, 4.4, 0.85,
        "grad_vars = xyz_gradient_accum / denom\ngrad_qualifiers = |grad_vars| >= grad_thresh (2e-4)", C1)
    box(5.2, 9.15, 4.4, 0.85,
        "grads_abs = xyz_gradient_accum_abs / denom\ngrad_qual_abs = |grads_abs| >= grad_abs_thresh (2e-4)", C1)
    box(0.4, 8.0, 9.2, 0.85,
        "full_split_mask <- custom_split_mask (high_ratio>0.8 & |g|>=1e-5);  full_prune_mask <- custom_prune_mask (low_ratio>0.8 & valid)\n"
        "viewspace_points_indices=None  =>  gán trực tiếp toàn bộ mask   [gm.py:978-988]", C1, 6.5)
    box(0.4, 6.85, 4.4, 0.85,
        "clone_qual = max(scale) <= dense*extent\nfinal_clone = (split|grad_qual) AND clone_qual", C2)
    box(5.2, 6.85, 4.4, 0.85,
        "split_qual = max(scale) >  dense*extent\nfinal_split = (split|grad_qual_abs) AND split_qual", C2)
    box(0.4, 5.85, 9.2, 0.7,
        "metric_mask = importance_score > args.importance_score_threshold (0.5)   [gm.py:1005]", C2, 7.2)
    box(0.4, 4.85, 4.4, 0.7, "1) densify_and_clone_structgs(metric_mask, final_clone)\nN -> N + N_clone", C2, 7.2)
    box(5.2, 4.85, 4.4, 0.7, "2) densify_and_split_structgs(metric_mask AND final_split)\nN -> N + N_split", C2, 7.2)
    box(0.4, 3.55, 9.2, 1.0,
        "3) PRUNE  (sau khi N đã tăng):\n"
        "prune_mask = (α < min_opacity) OR (max_radii2D > max_screen_size) OR (max scale > 0.1·extent) OR full_prune_mask\n"
        "full_prune_mask được pad thêm 0 cho các Gaussian mới sinh   [gm.py:1014-1027]", C3, 6.6)
    box(0.4, 2.45, 9.2, 0.8,
        "pruning_score is None (train.py:369)  =>  nhánh multinomial BỊ BỎ QUA  =>  prune_points(prune_mask)   [gm.py:1029-1044]",
        C3, 6.8)
    box(0.4, 1.35, 9.2, 0.8,
        "4) Trần opacity: opacity <- inverse_sigmoid(min(α, 0.8))  rồi replace_tensor_to_optimizer   [gm.py:1046-1048]", C4, 7.2)
    box(0.4, 0.3, 9.2, 0.75, "tmp_radii <- None ;  torch.cuda.empty_cache()   [gm.py:1049-1052]", "#e5e7eb", 7.2)

    for y0, y1 in [(10.3, 10.0), (9.15, 8.85), (8.0, 7.7), (6.85, 6.55),
                   (5.85, 5.55), (4.85, 4.55), (3.55, 3.25), (2.45, 2.15), (1.35, 1.05)]:
        arrow(5.0, y0, 5.0, y1)

    ax.text(5.0, 11.95, "Thứ tự THỰC SỰ trong code: CLONE → SPLIT → PRUNE → trần opacity 0.8",
            ha="center", va="top", fontsize=9.2, fontweight="bold")
    save(fig, "07_so_do_khoi_luong.png")


# ----------------------------------------------------------------------
# H2: hop OR cac mat na prune (so do Venn + bang dem)
# ----------------------------------------------------------------------
def fig2_venn():
    N = 4000
    alpha = np.clip(rng.beta(2.0, 4.0, N), 0.001, 0.999)
    r2d = np.abs(rng.normal(8.0, 6.0, N))
    extent = 4.0
    scale = np.abs(rng.normal(0.22, 0.18, N))
    low_ratio = np.clip(rng.beta(2.0, 5.0, N), 0, 1)
    valid = rng.random(N) > 0.05

    m_op = alpha < 0.1                       # min_opacity = 0.1 (train.py:364)
    m_vs = r2d > 20.0                        # max_screen_size = 20 (train.py:324)
    m_ws = scale > 0.1 * extent              # 0.1*extent (gm.py:1017)
    m_cu = (low_ratio > 0.8) & valid         # custom_prune_mask (train.py:353)
    m_all = m_op | m_vs | m_ws | m_cu

    fig = plt.figure(figsize=(8.6, 4.2))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.3, 2.5)
    cols = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7"]
    cent = [(-0.55, 0.45), (0.55, 0.45), (-0.55, -0.55), (0.55, -0.55)]
    labs = [r"$\alpha<0.1$", r"$r_{2D}>20$", r"$s_{max}>0.1\,\mathrm{extent}$", "low_ratio > 0.8"]
    for (cx, cy), c, l in zip(cent, cols, labs):
        ax.add_patch(Circle((cx, cy), 0.95, fc=c, ec=c, alpha=0.26, lw=1.4))
        ax.text(cx + (1.05 if cx > 0 else -1.05), cy + (0.85 if cy > 0 else -0.85), l,
                fontsize=7.6, color=c, ha="left" if cx > 0 else "right")
    ax.text(0, 0, "OR", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.set_title("prune_mask = hợp (OR) của 4 tiêu chí\n(gm.py:1014-1027)", fontsize=8.8)

    ax2 = fig.add_subplot(1, 2, 2)
    names = ["α<0.1\n(opacity)", "r2D>20\n(screen)", "s>0.1·ext\n(world)", "low_ratio>0.8\n(custom)", "OR\ntổng hợp"]
    vals = [m_op.sum(), m_vs.sum(), m_ws.sum(), m_cu.sum(), m_all.sum()]
    bars = ax2.bar(names, vals, color=cols + ["#111827"])
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + N * 0.01, f"{v}\n({100*v/N:.1f}%)",
                 ha="center", fontsize=7)
    ax2.set_ylabel("số Gaussian bị gắn cờ"); ax2.tick_params(labelsize=7)
    ax2.set_title(f"Mô phỏng N={N}: tổng OR < tổng riêng lẻ\n(các tiêu chí chồng lấn)", fontsize=8.8)
    ax2.set_ylim(0, max(vals) * 1.28)
    fig.tight_layout()
    save(fig, "07_mat_na_prune_or.png")
    return m_op, m_vs, m_ws, m_cu, alpha, r2d


# ----------------------------------------------------------------------
# H3: N(t) qua cac lan densify
# ----------------------------------------------------------------------
def fig3_Nt():
    # densification_interval = 100, densify_from_iter = 500 (arguments/__init__.py:87,91)
    iters = np.arange(0, 15001, 10)
    N = np.zeros_like(iters, dtype=float)
    n = 120000.0
    p_clone, p_split = 0.022, 0.016
    hist_iter, hist_c, hist_s, hist_p = [], [], [], []
    for i, it in enumerate(iters):
        if it > 500 and it % 100 == 0:
            decay = np.exp(-(it - 500) / 4500.0)
            nc = n * p_clone * decay
            ns = n * p_split * decay * 1.6          # split sinh k^3-1 con, hệ số > clone
            n += nc + ns
            # prune xoá phần lớn ứng viên yếu vừa sinh + một phần nền
            npr = 0.55 * (nc + ns) + n * 0.0015 * (0.4 + 0.6 * (1 - decay))
            n -= npr
            hist_iter.append(it); hist_c.append(nc); hist_s.append(ns); hist_p.append(-npr)
        N[i] = n

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.9))
    a1.plot(iters, N / 1e3, color="#1d4ed8", lw=1.8)
    a1.axvline(500, color="#16a34a", ls="--", lw=1.1)
    a1.text(700, N.min() / 1e3 * 1.02, "densify_from_iter=500\ndensification_interval=100", fontsize=6.6, color="#16a34a")
    a1.axvline(3000, color="#b45309", ls=":", lw=1.1)
    a1.text(3300, N.max() / 1e3 * 0.80, "it>3000 ⇒ max_screen_size=20\n(trước đó = None)", fontsize=6.6, color="#b45309")
    a1.set_xlabel("iteration"); a1.set_ylabel("N (nghìn Gaussian)")
    a1.ticklabel_format(axis="x", style="plain")
    a1.set_xticks([0, 3000, 6000, 9000, 12000, 15000])
    a1.set_title("N(t): mỗi 100 iter một vòng clone+split+prune", fontsize=8.8)
    a1.grid(alpha=0.3)

    a2.bar(hist_iter, hist_c, width=70, color="#22c55e", label="+clone")
    a2.bar(hist_iter, hist_s, width=70, bottom=hist_c, color="#3b82f6", label="+split")
    a2.bar(hist_iter, hist_p, width=70, color="#ef4444", label="−prune")
    net = np.array(hist_c) + np.array(hist_s) + np.array(hist_p)
    a2.plot(hist_iter, net, color="#111827", lw=1.3, label="ΔN ròng")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xlabel("iteration"); a2.set_ylabel("ΔN mỗi vòng")
    a2.set_xticks([0, 3000, 6000, 9000, 12000, 15000])
    a2.set_title("Cân bằng sinh/xoá trong CÙNG một lời gọi hàm", fontsize=8.8)
    a2.legend(fontsize=6.6); a2.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "07_duong_cong_N_t.png")


# ----------------------------------------------------------------------
# H4: scatter opacity vs screen radius, to mau theo quyet dinh prune
# ----------------------------------------------------------------------
def fig4_scatter(alpha, r2d):
    extent = 4.0
    N = alpha.shape[0]
    scale = np.abs(rng.normal(0.22, 0.18, N))
    low_ratio = np.clip(rng.beta(2.0, 5.0, N), 0, 1)
    valid = rng.random(N) > 0.05
    m_op = alpha < 0.1
    m_vs = r2d > 20.0
    m_ws = scale > 0.1 * extent
    m_cu = (low_ratio > 0.8) & valid
    keep = ~(m_op | m_vs | m_ws | m_cu)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.scatter(alpha[keep], r2d[keep], s=5, c="#cbd5e1", label=f"GIỮ ({keep.sum()})")
    ax.scatter(alpha[m_ws & ~m_op & ~m_vs], r2d[m_ws & ~m_op & ~m_vs], s=7, c="#22c55e",
               label="xoá: scale > 0.1·extent")
    ax.scatter(alpha[m_cu & ~m_op & ~m_vs & ~m_ws], r2d[m_cu & ~m_op & ~m_vs & ~m_ws], s=7,
               c="#a855f7", label="xoá: low_ratio > 0.8")
    ax.scatter(alpha[m_vs], r2d[m_vs], s=7, c="#3b82f6", label="xoá: r2D > 20 px")
    ax.scatter(alpha[m_op], r2d[m_op], s=7, c="#ef4444", label="xoá: α < 0.1")
    ax.axvline(0.1, color="#ef4444", ls="--", lw=1.2)
    ax.axhline(20, color="#3b82f6", ls="--", lw=1.2)
    ax.text(0.105, 33, "min_opacity = 0.1\n(train.py:364)", fontsize=6.8, color="#ef4444")
    ax.text(0.62, 21, "max_screen_size = 20 px (train.py:324)", fontsize=6.8, color="#3b82f6")
    ax.set_xlabel(r"opacity $\alpha = \mathrm{sigmoid}(\_opacity)$")
    ax.set_ylabel(r"max_radii2D (pixel)")
    ax.set_title("Vùng quyết định prune trên mặt phẳng (α, r2D)\nmàu = tiêu chí OR nào kích hoạt trước", fontsize=9)
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 40); ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "07_scatter_opacity_radius.png")


# ----------------------------------------------------------------------
# H5: cat dong bo tensor + trang thai Adam
# ----------------------------------------------------------------------
def fig5_adam():
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.6, 9.0); ax.axis("off")

    n = 8
    mask = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)   # True = bi xoa
    valid = ~mask
    keep_idx = np.where(valid)[0]

    rows_left = ["_xyz [N,3]", "_features_dc [N,1,3]", "_features_rest", "_opacity [N,1]",
                 "_scaling [N,3]", "_rotation [N,4]", "exp_avg (Adam)", "exp_avg_sq (Adam)"]
    colors = ["#dbeafe"] * 6 + ["#fde68a"] * 2

    y0 = 6.8
    dy = 0.62
    cw = 0.46
    ax.text(0.1, y0 + 1.45, "TRƯỚC prune: N = 8", fontsize=8.5, fontweight="bold")
    ax.text(7.4, y0 + 1.45, "SAU prune: N' = 5", fontsize=8.5, fontweight="bold")

    # mask row
    for j in range(n):
        c = "#ef4444" if mask[j] else "#e5e7eb"
        ax.add_patch(Rectangle((2.6 + j * cw, y0 + 0.72), cw * 0.92, 0.3, fc=c, ec="#555", lw=0.5))
        ax.text(2.6 + j * cw + cw * 0.46, y0 + 0.87, "1" if mask[j] else "0",
                fontsize=5.8, ha="center", va="center",
                color="white" if mask[j] else "#333")
    ax.text(2.5, y0 + 0.87, "prune_mask", fontsize=6.8, ha="right", va="center")
    ax.text(6.55, y0 + 0.87, r"valid = ~mask", fontsize=6.8, ha="left", va="center", color="#16a34a")

    for i, (name, col) in enumerate(zip(rows_left, colors)):
        y = y0 - i * dy
        ax.text(2.5, y + 0.15, name, fontsize=6.8, ha="right", va="center")
        for j in range(n):
            fc = "#fecaca" if mask[j] else col
            ax.add_patch(Rectangle((2.6 + j * cw, y), cw * 0.92, 0.3, fc=fc, ec="#555", lw=0.5))
        # after
        for k, j in enumerate(keep_idx):
            ax.add_patch(Rectangle((7.4 + k * cw, y), cw * 0.92, 0.3, fc=col, ec="#555", lw=0.5))
        ax.add_patch(FancyArrowPatch((6.45, y + 0.15), (7.3, y + 0.15), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.8, color="#16a34a"))

    ax.add_patch(Rectangle((2.55, y0 - 5 * dy - 0.06), n * cw + 0.06, 6 * dy, fill=False,
                           ec="#1d4ed8", lw=1.2, ls="--"))
    ax.add_patch(Rectangle((2.55, y0 - 7 * dy - 0.06), n * cw + 0.06, 2 * dy, fill=False,
                           ec="#b45309", lw=1.2, ls="--"))
    xr = 7.4 + len(keep_idx) * cw + 0.25
    ax.text(xr, y0 - 2.2 * dy, "6 tham số học\n(nn.Parameter)", fontsize=6.8, color="#1d4ed8")
    ax.text(xr, y0 - 6.6 * dy, "state Adam\n(+max_exp_avg_sq\nnếu amsgrad)", fontsize=6.8, color="#b45309")

    ax.text(0.1, 1.45,
            "_prune_optimizer(valid)  [gm.py:503-527]:  duyệt self.optimizer VÀ self.shoptimizer\n"
            "  exp_avg, exp_avg_sq (và max_exp_avg_sq) ← [valid];  del opt.state[p];  p ← nn.Parameter(p[valid]);  opt.state[p_mới] ← state đã cắt\n"
            "prune_points  [gm.py:528-565]: cắt tiếp xyz_gradient_accum, xyz_gradient_accum_abs, denom, max_radii2D, tmp_radii, filter_3D,\n"
            "  accum_eta, accum_view_count, max_eta_3ch, accum_weights_valid, densify_count, eta_high/mid/low_count, eta_high/mid_sum_3ch",
            fontsize=6.5, va="top", family="DejaVu Sans")
    ax.text(0.1, -0.35, "⇒ MỌI tensor per-Gaussian dùng CÙNG một valid mask ⇒ chỉ số hàng luôn đồng bộ sau khi N giảm.",
            fontsize=7.6, fontweight="bold", color="#b91c1c")
    ax.text(6.3, 8.9, "prune_points: cắt đồng bộ tensor tham số + moment bậc 1/2 của Adam",
            fontsize=9.8, ha="center", va="top")
    save(fig, "07_cat_tensor_adam.png")


if __name__ == "__main__":
    fig1_flow()
    m_op, m_vs, m_ws, m_cu, alpha, r2d = fig2_venn()
    fig3_Nt()
    fig4_scatter(alpha, r2d)
    fig5_adam()
    print("DONE")
