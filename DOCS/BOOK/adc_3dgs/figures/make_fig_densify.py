"""
Hình minh hoạ tiêu chí densify structure-aware (clone / split) của SADGS.
Công thức (adc_3dgs_explained.md mục 3):
    high_ratio_i  = eta_hi_cnt_i / view_count_i
    split_signal_i = [high_ratio_i > split_ratio_threshold] AND [is_grad_high_i]
    metric_i        = [importance_i > importance_score_threshold]
    clone_i = metric_i AND split_signal_i AND [max(s_i) <= dense*extent]
    split_i = metric_i AND split_signal_i AND [max(s_i) >  dense*extent]
    split_ratio_threshold = 0.8, importance_score_threshold = 0.5, dense = 0.001
Prune theo kích thước (mục 5): max(s_i) > 0.1*extent (không đổi so với 3DGS gốc).

Xuất 3 hình vào cùng thư mục với script:
    fig_05_decision_plane.png
    fig_06_scene_extent.png
    fig_07_threshold_sweep.png
Chỉ dùng numpy + matplotlib. Chạy: python make_fig_densify.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ cấu hình
SEED = 20231
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_RATIO_THRESHOLD = 0.8    # multiview-consistency: split_ratio_threshold
IMPORTANCE_THRESHOLD = 0.5     # importance_score_threshold
DENSE = 0.001                  # args.dense (percent_dense analog, nhỏ hơn 3DGS gốc 10x)
PRUNE_SCALE = 0.1              # ngưỡng prune theo kích thước (× extent, không đổi)

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["legend.fontsize"] = 8

C_KEEP = "#d9d9d9"     # giữ nguyên
C_CLONE = "#a8dadc"    # clone (xanh lam nhạt)
C_SPLIT = "#f4a582"    # split (cam nhạt)
C_PRUNE = "#c9c9ff"    # vùng bị prune (tím nhạt)
C_TAU = "#c0392b"      # màu vạch tau_grad
C_DELTA = "#1f4e79"    # màu vạch delta*extent
C_PR = "#6a3d9a"       # màu vạch prune

saved = []


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(path)


# ------------------------------------------------------------------ tiện ích
def decide(high_ratio, s_rel, importance=1.0, is_grad_high=True):
    """Trả về 'keep' / 'clone' / 'split' theo công thức structure-aware mục 3."""
    split_signal = (high_ratio > SPLIT_RATIO_THRESHOLD) and is_grad_high
    metric = importance > IMPORTANCE_THRESHOLD
    if not (split_signal and metric):
        return "keep"
    return "clone" if s_rel <= DENSE else "split"


# ======================================================================
# HÌNH 5 — mặt phẳng quyết định
# ======================================================================
def fig_05():
    rng = np.random.default_rng(SEED)
    x_lo, x_hi = 0.0, 1.0
    y_lo, y_hi = 1e-3, 0.3

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.2, 5.6),
                                  gridspec_kw={"width_ratios": [2.9, 1.35]})
    axr.axis("off")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # 3 vùng màu (giả định importance đủ cao / is_grad_high đúng)
    ax.fill_betweenx([y_lo, y_hi], x_lo, SPLIT_RATIO_THRESHOLD, color=C_KEEP, alpha=0.9, zorder=0)
    ax.fill_betweenx([y_lo, DENSE], SPLIT_RATIO_THRESHOLD, x_hi, color=C_CLONE, alpha=0.9, zorder=0)
    ax.fill_betweenx([DENSE, y_hi], SPLIT_RATIO_THRESHOLD, x_hi, color=C_SPLIT, alpha=0.9, zorder=0)
    # vùng prune theo kích thước (gạch chéo phía trên 0.1*extent)
    ax.fill_betweenx([PRUNE_SCALE, y_hi], x_lo, x_hi, facecolor="none",
                     hatch="///", edgecolor=C_PR, linewidth=0, alpha=0.55, zorder=1)

    # vạch ngưỡng
    ax.axvline(SPLIT_RATIO_THRESHOLD, color=C_TAU, lw=1.8, zorder=2)
    ax.axhline(DENSE, color=C_DELTA, lw=1.8, zorder=2)
    ax.axhline(PRUNE_SCALE, color=C_PR, lw=1.6, ls="--", zorder=2)
    ax.text(SPLIT_RATIO_THRESHOLD * 1.02, 1.25e-3, "split\\_ratio\\_threshold=0.8",
            color=C_TAU, fontsize=9, ha="left", va="bottom", zorder=5)
    ax.text(0.02, DENSE * 1.12, r"$\mathrm{dense}\cdot extent = 0.001\,extent$",
            color=C_DELTA, fontsize=9, ha="left", va="bottom", zorder=5)
    ax.text(0.02, PRUNE_SCALE * 1.12, r"prune: $\max(s_i) > 0.1\,extent$",
            color=C_PR, fontsize=9, ha="left", va="bottom", zorder=5)

    # nhãn vùng
    ax.text(0.35, 0.04, "GIỮ NGUYÊN\nhigh\\_ratio $\\leq$ 0.8\n(hoặc is\\_grad\\_high sai / importance thấp)",
            ha="center", va="center", fontsize=9, color="#444", zorder=5)
    ax.text(0.90, 2.8e-3, "CLONE\nhigh\\_ratio $>$ 0.8 và\n$\\max(s_i) \\leq$ dense·extent",
            ha="center", va="center", fontsize=9, color="#0b525b", zorder=5)
    ax.text(0.90, 0.045, "SPLIT\nhigh\\_ratio $>$ 0.8 và\n$\\max(s_i) >$ dense·extent",
            ha="center", va="center", fontsize=9, color="#7f2704", zorder=5)

    # ~300 điểm mô phỏng: Beta cho high_ratio (thiên lệch về gần 1 vì nhiều
    # Gaussian đã hội tụ), log-normal cho s
    n = 300
    hr = rng.beta(1.6, 1.6, n)
    s = np.exp(rng.normal(np.log(0.006), 1.1, n))
    s = np.clip(s, y_lo * 1.2, y_hi / 1.2)
    grad_ok = rng.random(n) < 0.85     # is_grad_high đúng ở phần lớn (cổng phụ)
    importance = rng.beta(2.0, 1.3, n)  # phần lớn có importance khá cao
    cols = {"keep": "#6b6b6b", "clone": "#1b7d84", "split": "#c1440e"}
    dec = np.array([decide(h, si, imp, gk) for h, si, imp, gk in zip(hr, s, importance, grad_ok)])
    for k, c in cols.items():
        m = dec == k
        ax.scatter(hr[m], s[m], s=9, color=c, alpha=0.55, linewidths=0, zorder=3)

    # ví dụ A..E — trùng bảng mục 3.2 của adc_3dgs_explained.md
    examples = [
        ("A", 0.95, 0.0005, 1.0, 0.8, "clone"),
        ("B", 0.90, 0.02, 1.0, 0.7, "split"),
        ("C", 0.95, 0.02, 1.0, 0.2, "giữ (importance thấp)"),
        ("D", 0.50, 0.02, 1.0, 0.9, "giữ (chưa đủ nhất quán)"),
        ("E", 0.95, 0.0005, 0.0, 0.9, "giữ (is_grad_high sai)"),
    ]
    offsets = {"A": (8, -12), "B": (8, 6), "C": (-16, 8), "D": (8, 6), "E": (-16, -14)}
    for lab, h, sy, gk, imp, _ in examples:
        ax.scatter([h], [sy], s=70, marker="o", facecolor="white",
                   edgecolor="black", linewidths=1.3, zorder=6)
        ax.annotate(lab, (h, sy), xytext=offsets[lab], textcoords="offset points",
                    fontsize=10, fontweight="bold", zorder=7)

    # bảng nhỏ
    cell = [[lab, f"{h:.2f}", f"{sy:g}", "có" if gk else "không", f"{imp:.1f}", d]
            for lab, h, sy, gk, imp, d in examples]
    tb = axr.table(cellText=cell,
                   colLabels=["", "high\nratio", "max(s)\n/ext", "grad\nhigh", "import\n-ance", "Quyết định"],
                   colWidths=[0.07, 0.14, 0.15, 0.13, 0.13, 0.38],
                   cellLoc="left", colLoc="left",
                   bbox=[0.0, 0.56, 1.0, 0.40], zorder=8)
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    axr.text(0.0, 0.975, "Ví dụ A..E (khớp mục 3.2 explained.md)", fontsize=9,
             fontweight="bold", va="bottom", transform=axr.transAxes)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor("#888")
        cl.set_facecolor("white" if r > 0 else "#eeeeee")
        cl.set_alpha(0.95)
    axr.text(0.0, 0.535, "C, D, E đều nằm trong 'vùng densify' hình học nhưng\n"
             "bị chặn bởi importance/grad_high/high_ratio — minh hoạ\n"
             "vì sao multiview-consistency giảm false-positive.",
             fontsize=7.5, color=C_PR, va="top", transform=axr.transAxes)

    # legend
    handles = [
        Patch(color=C_KEEP, label="Giữ nguyên (hình học)"),
        Patch(color=C_CLONE, label="Clone (hình học)"),
        Patch(color=C_SPLIT, label="Split (hình học)"),
        Patch(facecolor="none", edgecolor=C_PR, hatch="///", label="Bị prune (max(s) > 0.1 extent)"),
        Line2D([], [], color=C_TAU, lw=1.8, label="split_ratio_threshold"),
        Line2D([], [], color=C_DELTA, lw=1.8, label="dense·extent"),
        Line2D([], [], marker="o", color="black", mfc="white", ls="", label="Điểm ví dụ A..E"),
    ]
    axr.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.46),
               framealpha=0.95, ncol=1, title="Chú giải")

    ax.set_xlabel("high_ratio = eta_hi_cnt / view_count  (tỉ lệ quan sát $\\eta$ cao, tuyến tính)")
    ax.set_ylabel(r"$\max(s_i)\,/\,extent$  (kích thước lớn nhất so với cảnh, thang log)")
    ax.set_title("Mặt phẳng quyết định densify structure-aware: giữ nguyên / clone / split\n"
                 "(vùng màu là điều kiện HÌNH HỌC; cần thêm is_grad_high VÀ importance — xem bảng)")
    save(fig, "fig_05_decision_plane.png")
    return examples


# ======================================================================
# HÌNH 6 — scene extent
# ======================================================================
def fig_06():
    rng = np.random.default_rng(SEED + 1)
    # camera nằm trên vòng quanh cảnh (chừa một khoảng trống), point cloud ở giữa
    n_cam = 28
    ang = np.linspace(0.12 * np.pi, 1.88 * np.pi, n_cam) + rng.normal(0, 0.03, n_cam)
    rad = 6.0 + rng.normal(0, 0.4, n_cam)
    cams = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    pts = rng.normal(0, 1.4, (350, 2)) * np.array([1.3, 0.8])

    center = cams.mean(axis=0)                       # tâm các camera
    dist = np.linalg.norm(cams - center, axis=1)
    extent = 1.1 * dist.max()                        # cách Inria tính
    i_far = int(dist.argmax())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 5.0),
                                  gridspec_kw={"width_ratios": [1.4, 1]})
    # --- panel trái
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color="#7a7a7a", alpha=0.6, label="Point cloud (SfM)")
    ax.scatter(cams[:, 0], cams[:, 1], marker="^", s=38, color="#1f4e79",
               edgecolor="white", linewidths=0.5, label="Camera")
    ax.scatter([center[0]], [center[1]], marker="x", s=70, color="black", lw=1.8,
               label="Tâm các camera", zorder=5)
    ax.plot([center[0], cams[i_far, 0]], [center[1], cams[i_far, 1]],
            color="#c0392b", lw=1.2, ls="--", zorder=4)
    mid = (center + cams[i_far]) / 2
    ax.annotate(r"$d_{max}$ (camera xa tâm nhất)", mid, xytext=(10, 0), textcoords="offset points",
                fontsize=8, color="#c0392b", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.add_patch(Circle(center, extent, fill=False, color="#c0392b", lw=1.8,
                        label=f"extent = 1.1 × d_max ≈ {extent:.2f}"))

    # 2 Gaussian ellipse với max(s) = dense·extent và 0.1·extent
    s_small = max(DENSE * extent, 1e-3)   # sàn hiển thị để ellipse còn thấy được
    s_big = PRUNE_SCALE * extent
    e1 = Ellipse((-2.4, 2.0), 2 * s_small, 2 * s_small * 0.55, angle=25,
                 facecolor=C_CLONE, edgecolor=C_DELTA, lw=1.5, zorder=6)
    e2 = Ellipse((2.6, -1.4), 2 * s_big, 2 * s_big * 0.5, angle=-20,
                 facecolor=C_PRUNE, edgecolor=C_PR, lw=1.5, alpha=0.85, zorder=6)
    ax.add_patch(e1)
    ax.add_patch(e2)
    ax.annotate(f"Gaussian có max(s) = dense·extent ≈ {DENSE*extent:.4f}\n(ngưỡng clone/split, rất nhỏ)",
                (-2.4, 2.0), xytext=(-4.6, 5.3), fontsize=8, color=C_DELTA, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_DELTA, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_DELTA, alpha=0.9))
    ax.annotate(f"Gaussian có max(s) = 0.1·extent ≈ {s_big:.2f}\n(ngưỡng prune theo kích thước)",
                (2.6 + s_big * 0.5, -1.4 - 0.3), xytext=(1.0, -4.3), fontsize=8, color=C_PR, ha="center",
                arrowprops=dict(arrowstyle="->", color=C_PR, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_PR, alpha=0.9))
    ax.set_aspect("equal")
    lim = extent * 1.12
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_xlabel("x (đơn vị cảnh)")
    ax.set_ylabel("y (đơn vị cảnh)")
    ax.set_title("extent theo cách tính của Inria:\n1.1 × khoảng cách xa nhất từ camera tới tâm các camera")
    ax.legend(loc="lower left", framealpha=0.95, fontsize=7.5)
    ax.grid(alpha=0.25)

    # --- panel phải: thước tỉ lệ
    bars = [
        (f"dense·extent ≈ {DENSE*extent:.4f}", s_small, C_CLONE, C_DELTA, "ngưỡng clone ↔ split (dense = 0.001)"),
        (f"0.1·extent ≈ {s_big:.2f}", s_big, C_PRUNE, C_PR, "ngưỡng prune theo kích thước"),
        (f"extent ≈ {extent:.2f}", extent, "#f8d7c4", "#c0392b", "kích thước cảnh (scene.cameras_extent)"),
    ]
    ys = [2, 1, 0]
    for (lab, L, fc, ec, note), y in zip(bars, ys):
        ax2.barh(y, L, height=0.5, color=fc, edgecolor=ec, lw=1.4)
        if L > 0.5 * extent:   # thanh dài: ghi chữ bên trong thanh
            ax2.text(0.02 * extent, y + 0.05, lab, va="center", fontsize=8.5, fontweight="bold")
            ax2.text(0.02 * extent, y - 0.17, note, va="center", fontsize=7.5, color="#444")
        else:
            ax2.text(L + extent * 0.02, y + 0.05, lab, va="center", fontsize=8.5, fontweight="bold")
            ax2.text(L + extent * 0.02, y - 0.17, note, va="center", fontsize=7.5, color="#444")
    ax2.set_yticks([])
    ax2.set_xlim(0, extent * 1.02)
    ax2.set_xticks(np.arange(0, extent, 1.0))
    ax2.set_ylim(-0.6, 2.9)
    ax2.set_xlabel("độ dài (đơn vị cảnh)")
    ax2.set_title("Thước tỉ lệ  dense·extent : 0.1·extent : extent\n= 1 : 100 : 1000  (dense=0.001)")
    ax2.text(0.02 * extent, 2.75,
             "Cùng dense=0.001 nhưng ngưỡng tuyệt đối dense·extent\nthay đổi theo kích thước cảnh (extent).",
             fontsize=8, va="top", color="#333")
    ax2.grid(axis="x", alpha=0.3)
    save(fig, "fig_06_scene_extent.png")
    return extent, s_small, s_big


# ======================================================================
# HÌNH 7 — sensitivity of thresholds
# ======================================================================
def fig_07():
    rng = np.random.default_rng(SEED + 2)
    N = 10000
    # high_ratio: Beta thiên lệch quanh 2 cực (đa số Gaussian đã "chốt" là
    # nhất quán cao hoặc nhất quán thấp qua nhiều view, ít Gaussian lửng lơ)
    hr = rng.beta(0.7, 0.7, N)
    # max(s)/extent: log-normal, median ~0.0006 (thang nhỏ hơn 3DGS gốc vì dense=0.001)
    s_rel = np.exp(rng.normal(np.log(0.0006), 0.9, N))

    sel = hr >= SPLIT_RATIO_THRESHOLD
    n_over = int(sel.sum())
    frac_over = n_over / N * 100

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 4.3),
                                       gridspec_kw={"width_ratios": [1.35, 1, 1]})
    # --- (a) histogram
    bins = np.linspace(0, 1, 60)
    counts, edges = np.histogram(hr, bins=bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color="#9e9e9e", edgecolor="white", lw=0.3, label="Tất cả Gaussian")
    over = edges[:-1] >= SPLIT_RATIO_THRESHOLD
    ax.bar(edges[:-1][over], counts[over], width=np.diff(edges)[over], align="edge",
           color="#c0392b", edgecolor="white", lw=0.3, label="Vượt split_ratio_threshold (ứng viên densify)")
    ax.axvspan(SPLIT_RATIO_THRESHOLD, 1.0, color="#c0392b", alpha=0.07)
    ax.axvline(SPLIT_RATIO_THRESHOLD, color=C_TAU, lw=1.8)
    ax.text(SPLIT_RATIO_THRESHOLD * 1.01, counts.max() * 1.22, "split_ratio_threshold=0.8",
            color=C_TAU, fontsize=9)
    ax.text(SPLIT_RATIO_THRESHOLD * 1.01, counts.max() * 1.02,
            f"{frac_over:.1f}% vượt ngưỡng\n({n_over}/{N} Gaussian)",
            color=C_TAU, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.38)
    ax.set_xlim(0, 1)
    ax.set_xlabel("high_ratio (tuyến tính)")
    ax.set_ylabel("số Gaussian")
    ax.set_title(f"(a) Phân bố mô phỏng high_ratio, N = {N}\n(Beta lưỡng cực: đa số đã 'chốt' cao/thấp)")
    ax.legend(loc="upper center", fontsize=7.5)

    # --- (b) quét split_ratio_threshold
    thrs = np.linspace(0.5, 0.98, 80)
    pct_densify = np.array([(hr >= t).mean() * 100 for t in thrs])
    pct_def = frac_over
    ax2.plot(thrs, pct_densify, color=C_TAU, lw=2)
    ax2.axvline(SPLIT_RATIO_THRESHOLD, color=C_TAU, lw=1, ls="--", alpha=0.6)
    ax2.axhline(pct_def, color=C_TAU, lw=1, ls="--", alpha=0.6)
    ax2.scatter([SPLIT_RATIO_THRESHOLD], [pct_def], s=80, color=C_TAU, edgecolor="black", zorder=6)
    ax2.annotate(f"mặc định = 0.8\n→ {pct_def:.1f}% ứng viên densify",
                 (SPLIT_RATIO_THRESHOLD, pct_def), xytext=(-70, 18), textcoords="offset points",
                 fontsize=8.5, color=C_TAU,
                 arrowprops=dict(arrowstyle="->", color=C_TAU))
    for t in (0.6, 0.9):
        v = (hr >= t).mean() * 100
        ax2.scatter([t], [v], s=30, color="white", edgecolor=C_TAU, zorder=6)
        ax2.annotate(f"{v:.1f}%", (t, v), xytext=(6, 4), textcoords="offset points",
                     fontsize=8, color=C_TAU)
    ax2.set_xlim(0.5, 0.98)
    ax2.set_ylim(0, pct_densify.max() * 1.25)
    ax2.set_xlabel("split_ratio_threshold")
    ax2.set_ylabel("% Gaussian là ứng viên densify (trước lọc importance)")
    ax2.set_title("(b) Quét split_ratio_threshold\nngưỡng thấp hơn → nhiều ứng viên hơn, dễ false-positive hơn")
    ax2.grid(alpha=0.3)

    # --- (c) quét dense
    denses = np.logspace(np.log10(2e-4), np.log10(5e-3), 80)
    pct_clone = np.array([(s_rel[sel] <= d).mean() * 100 for d in denses])
    clone_def = (s_rel[sel] <= DENSE).mean() * 100
    ax3.plot(denses, pct_clone, color=C_DELTA, lw=2, label="% clone (max(s) ≤ dense·extent)")
    ax3.plot(denses, 100 - pct_clone, color="#c1440e", lw=2, ls="--", label="% split (max(s) > dense·extent)")
    ax3.set_xscale("log")
    ax3.axvline(DENSE, color="#444", lw=1, ls="--", alpha=0.6)
    ax3.scatter([DENSE], [clone_def], s=80, color=C_DELTA, edgecolor="black", zorder=6)
    ax3.scatter([DENSE], [100 - clone_def], s=80, color="#c1440e", edgecolor="black", zorder=6)
    ax3.annotate(f"mặc định dense = 0.001\n→ {clone_def:.1f}% clone / {100 - clone_def:.1f}% split",
                 (DENSE, clone_def), xytext=(0.0016, 47), textcoords="data",
                 fontsize=8.5, color=C_DELTA, ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_DELTA))
    ax3.set_xlim(2e-4, 5e-3)
    ax3.set_ylim(0, 105)
    ax3.set_xticks([2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    ax3.set_xticklabels(["0.0002", "0.0005", "0.001", "0.002", "0.005"])
    ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax3.set_xlabel("dense = args.dense (thang log)")
    ax3.set_ylabel("% trong số Gaussian được densify")
    ax3.set_title(f"(c) Quét dense (giữ split_ratio_threshold=0.8, {n_over} ứng viên)\ndense lớn hơn → nhiều clone, ít split")
    ax3.legend(loc="lower right", fontsize=7.5)
    ax3.grid(alpha=0.3, which="both")

    save(fig, "fig_07_threshold_sweep.png")
    return frac_over, pct_def, clone_def, float(np.median(hr)), n_over


if __name__ == "__main__":
    ex = fig_05()
    extent, s_small, s_big = fig_06()
    frac_over, pct_def, clone_def, med_g, n_over = fig_07()

    print("Đã lưu:")
    for p in saved:
        print("  ", p)
    print("\nSố liệu ví dụ (hình 5):")
    for lab, h, sy, gk, imp, d in ex:
        print(f"  {lab}: high_ratio={h:.2f}, max(s)/extent={sy:g}, grad_high={gk}, importance={imp:.1f} -> {d}")
    print(f"\nHình 6: extent ≈ {extent:.3f}, dense·extent ≈ {DENSE*extent:.5f}, 0.1·extent ≈ {s_big:.3f}")
    print(f"Hình 7: median high_ratio ≈ {med_g:.2f}; {n_over}/10000 = {frac_over:.2f}% vượt split_ratio_threshold=0.8; "
          f"với dense=0.001: {clone_def:.1f}% clone / {100-clone_def:.1f}% split trong số ứng viên")
