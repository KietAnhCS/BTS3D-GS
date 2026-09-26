"""
Vẽ hình minh hoạ Chương 7 — Structure-Aware Densification (SADGS).
Import lại toàn bộ số liệu từ ch07_test.py (numpy + scipy, seed 0; không torch).

Chạy:  python DOCS/Report/test/scripts/ch07_plot.py
PNG:   DOCS/Report/test/figures/ch07_<tên>.png
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, HERE)

with contextlib.redirect_stdout(io.StringIO()):
    import ch07_test as T   # chạy toàn bộ test, seed 0

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.linewidth": 1.1, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.9, "legend.edgecolor": "0.75",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 450,
})
GCOL = ["#d62728", "#2ca02c", "#1f77b4", "#7f7f7f"]
GNAME = [r"$G_1$", r"$G_2$", r"$G_3$", r"$G_4$"]
N, V = T.N, T.V_CAMS


def save(fig, name):
    path = os.path.join(FIG, f"ch07_{name}.png")
    fig.savefig(path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.normpath(path))


def to_img(C):
    return np.clip(np.transpose(C, (1, 2, 0)), 0, 1)


# ============================================================================
# Hình 1: bản đồ lỗi 3 view x 4 cột (get_loss, freq_utils.py:92-96 — còn trong code, không
# phải đường chạy mặc định để suy importance/pruning trong SADGS)
# ============================================================================
fig, axes = plt.subplots(V, 4, figsize=(14, 7.6))
for v in range(V):
    e, eh, m = T.E[v], T.EHAT[v], T.MASK[v]
    ax = axes[v, 0]
    ax.imshow(to_img(T.REND[v]), interpolation="nearest")
    ax.set_title(rf"view {v+1}: $I_{{\mathrm{{rend}}}}$ ($\alpha=0.1$)", fontsize=10)
    ax = axes[v, 1]
    ax.imshow(to_img(T.GT[v]), interpolation="nearest")
    ax.set_title(rf"view {v+1}: $I_{{\mathrm{{gt}}}}$ ($\alpha=0.9$)", fontsize=10)
    ax = axes[v, 2]
    im = ax.imshow(e, cmap="magma", vmin=0, vmax=0.45, interpolation="nearest")
    ax.set_title(r"$e_v(x)=\frac{1}{3}\sum_{ch}|I_{\mathrm{rend}}-I_{\mathrm{gt}}|$ (get\_loss)", fontsize=10)
    ax.text(0.02, 0.97, f"min={e.min():.4f}\nmax={e.max():.4f}\nmean={e.mean():.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="none"))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax = axes[v, 3]
    ax.imshow(m, cmap=ListedColormap(["white", "#d62728"]), vmin=0, vmax=1, interpolation="nearest")
    thr = e.min() + 0.1 * (e.max() - e.min())
    ax.set_title(rf"$m_v=\mathbb{{1}}[\hat e_v>0.1]$  ($e>{thr:.4f}$)", fontsize=10)
    ax.text(0.02, 0.97, f"bật {int(m.sum())}/{T.H*T.W} px", transform=ax.transAxes, va="top",
            ha="left", fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.85, ec="gray"))
for ax in axes.ravel():
    ax.grid(False)
    ax.set_xticks([0, 16, 32, 47]); ax.set_yticks([0, 16, 31])
    ax.tick_params(labelsize=7)
fig.suptitle("get_loss (freq_utils.py:92-96): sai số màu $e_v(x)$, chuẩn hoá min–max $\\hat e_v$ và mặt "
             "nạ $m_v$ — còn trong code SADGS nhưng KHÔNG dùng để suy importance/pruning mặc định "
             "(3 view, 48×32 px)", fontsize=11.5)
fig.tight_layout()
save(fig, "error_maps")

# ============================================================================
# Hình 2: eta đa-view (proxy), active mask và importance_score = accum_view_count
# ============================================================================
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
v = 0
for i in range(N):
    ax = fig.add_subplot(gs[0, i])
    rgb = np.ones((T.H, T.W, 3))
    col = np.array(matplotlib.colors.to_rgb(GCOL[i]))
    f = T.FOOT[v][i]
    is_high = T.IS_HIGH[i, v]; is_low = T.IS_LOW[i, v]
    shade = col if is_high else (0.55 * col + 0.45 if is_low else 0.75 * col + 0.25)
    rgb[f] = shade
    ax.imshow(rgb, interpolation="nearest")
    ax.grid(False)
    mu2 = T.ITEMS[v][i]["mu2"]
    ax.plot(mu2[0], mu2[1], "k+", ms=10, mew=2)
    tag = "HIGH" if is_high else ("LOW" if is_low else "MID")
    ax.set_title(f"{GNAME[i]} view 1: eta={T.ETA[i,v]:.3f} ({tag})", fontsize=10, color=GCOL[i])
    ax.text(0.02, 0.97, f"active={int(T.ACTIVE[i,v])}", transform=ax.transAxes, va="top",
            fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="gray"))
    ax.set_xticks([0, 16, 32, 47]); ax.set_yticks([0, 16, 31]); ax.tick_params(labelsize=7)
leg = [Patch(fc="#333333", label="eta > TAU_HIGH=1.0 (đậm)"),
       Patch(fc="#bbbbbb", ec="gray", label="eta ≤ TAU_LOW=0.1 (nhạt)"),
       Patch(fc="#999999", label="MID (0.1 < eta ≤ 1.0)"),
       Line2D([], [], marker="+", color="k", ls="", ms=8, mew=2, label="$\\mu'_i$")]
fig.legend(handles=leg, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.945), frameon=True)

# bảng eta 4x3 heatmap
ax = fig.add_subplot(gs[1, 0:2])
im = ax.imshow(T.ETA, cmap="Blues")
for i in range(N):
    for vv in range(V):
        ax.text(vv, i, f"{T.ETA[i,vv]:.2f}", ha="center", va="center", fontsize=11,
                color="white" if T.ETA[i, vv] > T.ETA.mean() else "black")
ax.set_xticks(range(V)); ax.set_xticklabels([f"view {k+1}" for k in range(V)])
ax.set_yticks(range(N)); ax.set_yticklabels(GNAME)
for lab, c in zip(ax.get_yticklabels(), GCOL):
    lab.set_color(c)
ax.grid(False)
ax.set_title(r"eta$_i^{(v)}$ = axis_length_px / wavelength_min (proxy, freq_utils.py:338-348)", fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="eta")

# bar accum_view_count (importance_score) + ngưỡng
ax = fig.add_subplot(gs[1, 2:4])
x = np.arange(N)
b1 = ax.bar(x, T.IMPORTANCE, 0.5, color=GCOL, alpha=0.85, edgecolor=GCOL)
for k in range(N):
    ax.text(x[k], T.IMPORTANCE[k] + 0.05, f"{T.IMPORTANCE[k]}\nhigh_ratio={T.HIGH_RATIO[k]:.2f}\n"
            f"low_ratio={T.LOW_RATIO[k]:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.axhline(T.IMPORTANCE_SCORE_THR, color="k", ls="--", lw=1)
ax.text(3.45, T.IMPORTANCE_SCORE_THR + 0.05, "importance_score_threshold = 0.5", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(GNAME)
for lab, c in zip(ax.get_xticklabels(), GCOL):
    lab.set_color(c)
ax.set_ylim(0, V + 0.8)
ax.set_ylabel("accum_view_count = importance_score")
ax.set_title("importance_score (cả 4 > 0.5 → không ai bị chặn densify)", fontsize=11)
fig.suptitle("Thống kê eta đa-view (update_freq_stats_online, freq_utils.py:181-415): "
             "phân loại HIGH/LOW/MID và importance_score", fontsize=12, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.9])
save(fig, "footprint_counts")

# ============================================================================
# Hình 3: E_photo, low_ratio (Pruning), w_i multinomial minh hoạ
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
L1s = [T.l1(T.REND[v], T.GT[v]) for v in range(V)]
Ss = [T.ssim(T.REND[v], T.GT[v]) for v in range(V)]
xs = np.arange(V)
ax.bar(xs, [0.8 * l for l in L1s], 0.55, color="#4c72b0", label=r"$0.8\,\mathcal{L}_1$")
ax.bar(xs, [0.2 * (1 - s) for s in Ss], 0.55, bottom=[0.8 * l for l in L1s], color="#dd8452",
       label=r"$0.2\,(1-\mathrm{SSIM})$")
EPHOTO = np.array([(1 - T.LAMBDA) * L1s[v] + T.LAMBDA * (1 - Ss[v]) for v in range(V)])
for v in range(V):
    ax.text(xs[v], EPHOTO[v] + 0.006, f"{EPHOTO[v]:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(xs[v], 0.8 * L1s[v] / 2, f"L1={L1s[v]:.4f}", ha="center", va="center", fontsize=8, color="white")
    ax.text(xs[v], 0.8 * L1s[v] + 0.2 * (1 - Ss[v]) / 2, f"SSIM={Ss[v]:.3f}", ha="center", va="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([f"view {v+1}" for v in range(V)])
ax.set_ylim(0, 0.33)
ax.set_title(r"(a) $E^{(v)}_{\mathrm{photo}}=0.8\mathcal{L}_1+0.2(1-\mathrm{SSIM})$"
             "\n(freq_utils.py:98-102 compute_photometric_loss)", fontsize=9.5)
ax.legend(fontsize=9, loc="upper right")

ax = axes[1]
x = np.arange(N)
ax.bar(x - 0.2, T.RAW, 0.38, color=GCOL, alpha=0.45, edgecolor=GCOL)
for k in range(N):
    ax.text(x[k] - 0.2, T.RAW[k] + 0.05, f"{T.RAW[k]:.0f}", ha="center", fontsize=8)
ax.set_ylim(0, V + 0.8)
ax.set_ylabel("eta_low_count (nhạt, trục trái)", fontsize=9)
ax2 = ax.twinx()
ax2.bar(x + 0.2, T.PRUNING, 0.38, color=GCOL)
for k in range(N):
    ax2.text(x[k] + 0.2, T.PRUNING[k] + 0.02, f"{T.PRUNING[k]:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(T.PRUNE_RATIO_THR, color="k", ls="--", lw=1)
ax2.text(3.4, T.PRUNE_RATIO_THR + 0.02, "prune_ratio_threshold=0.8", fontsize=7.5, ha="right")
ax2.set_ylim(0, 1.18)
ax2.set_ylabel("low_ratio = Pruning$_i$ (đậm, trục phải)", fontsize=9)
ax2.grid(False)
ax.set_xticks(x); ax.set_xticklabels(GNAME)
for lab, c in zip(ax.get_xticklabels(), GCOL):
    lab.set_color(c)
ax.set_title("(b) eta_low_count → low_ratio (thay Pruning minmax kiểu FastGS cũ)", fontsize=9.5)

ax = axes[2]
ax.bar(x, T.W_I, 0.55, color=GCOL)
for k in range(N):
    lab = f"{T.W_I[k]:.3g}" if T.W_I[k] < 1e5 else r"$10^6$"
    ax.text(x[k], T.W_I[k] * 1.4, lab, ha="center", fontsize=10, fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(0.5, 1e7)
ax.set_xticks(x); ax.set_xticklabels(GNAME)
for lab, c in zip(ax.get_xticklabels(), GCOL):
    lab.set_color(c)
ax.set_ylabel(r"$w_i$ (log)")
ax.set_title("(c) $w_i=1/(10^{-6}+1-\\mathrm{low\\_ratio}_i)$\nMINH HOẠ — pruning_score=None trong "
             "train.py nên nhánh này KHÔNG chạy mặc định", fontsize=9)
p = T.W_I / T.W_I.sum()
ax.text(0.03, 0.95, "xác suất rút lần 1:\n" + "\n".join(f"{GNAME[k]}: {p[k]:.1e}" for k in range(N)),
        transform=ax.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray"))
fig.suptitle("Photometric loss từng view, low_ratio (điểm 'Pruning' đa-view) và trọng số multinomial "
             "minh hoạ (nhánh không mặc định trong SADGS)", fontsize=11.5)
fig.tight_layout()
save(fig, "scores")

# ============================================================================
# Hình 4 (ch7_clone_region): sơ đồ quyết định densify_and_prune_structgs
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
dext = T.DELTA * T.EXTENT
for ax, g, tau, tname, gname, qual in (
        (axes[0], T.GBAR, T.TAU_GRAD, r"$\tau_{\mathrm{grad}}=2\times10^{-4}$ (grad\_thresh)",
         r"$\bar g_i$ (có dấu)", T.clone_qualifiers),
        (axes[1], T.GBAR_ABS, T.TAU_GRAD_ABS, r"$\tau^{\mathrm{abs}}_{\mathrm{grad}}=2\times10^{-4}$ "
         "(grad\\_abs\\_thresh)", r"$\bar g^{\mathrm{abs}}_i$", T.split_qualifiers)):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(3e-5, 3e-1); ax.set_ylim(2e-4, 5)
    ax.axvspan(tau, 3e-1, ymin=0, ymax=1, color="#2ca02c", alpha=0.06)
    ax.axvline(tau, color="k", ls="--", lw=1.2)
    ax.axhline(dext, color="purple", ls="-.", lw=1.2)
    ax.text(tau * 1.1, 3.2, tname, fontsize=8.5, ha="left")
    ax.text(3.5e-5, dext * 1.25, rf"$\delta\cdot\mathrm{{extent}}={dext:.5f}$ (args.dense)", fontsize=9, color="purple")
    ax.text(0.03, 0.62, "vùng SPLIT\n(grad ≥ τ HOẶC high_ratio>0.8, s > δ·extent)", transform=ax.transAxes,
            fontsize=8.5, color="#2ca02c", ha="left", alpha=0.9)
    ax.text(0.62, 0.62, "SPLIT", transform=ax.transAxes, fontsize=16, color="#2ca02c", alpha=0.35, fontweight="bold")
    ax.text(0.62, 0.12, "CLONE", transform=ax.transAxes, fontsize=16, color="#1f77b4", alpha=0.35, fontweight="bold")
    ax.text(0.05, 0.12, "không densify", transform=ax.transAxes, fontsize=11, color="gray", alpha=0.8)
    for i in range(N):
        ax.scatter(g[i], T.MAXS[i], s=110, color=GCOL[i], edgecolor="k", zorder=5)
    for i in range(N):
        ax.text(0.97, 0.50 - 0.05 * i, f"{GNAME[i]}: ({g[i]:.2e}, {T.MAXS[i]:.3f})", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=GCOL[i],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.text(0.97, 0.55, "(gradient, max s) của 4 Gaussian:", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    ax.set_xlabel(gname + " (NDC, log)")
    ax.set_ylabel(r"$\max_k s_{i,k}$ (log)")
    ax.set_title(f"{gname} so với {tname}", fontsize=9.5)
axes[0].text(0.5, 0.02, "final_clone_mask = (custom_split_mask ∨ grad_qualifiers) ∧ clone_qualifiers",
             transform=axes[0].transAxes, ha="center", fontsize=7.5, color="gray")
axes[1].text(0.5, 0.02, "final_split_mask = (custom_split_mask ∨ grad_qualifiers_abs) ∧ split_qualifiers",
             transform=axes[1].transAxes, ha="center", fontsize=7.5, color="gray")
txt = ("SADGS (gaussian_model.py:957-1052, train.py:315-386): custom_split_mask từ "
       "high_ratio>split_ratio_threshold(0.8) & is_grad_high, cổng metric_mask=importance_score>0.5\n"
       + "  ".join(f"{GNAME[i]}: CLONE={bool(T.CLONE[i])} SPLIT={bool(T.SPLIT[i])}" for i in range(N))
       + f"\n→ clone {int(T.CLONE.sum())}, split {int(T.SPLIT.sum())} "
         "(δ·extent nhỏ hơn mọi s trong cảnh đồ chơi nên không ai vào nhánh clone)")
fig.text(0.5, -0.11, txt, ha="center", fontsize=9.5, bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0a800"))
fig.suptitle("Sơ đồ quyết định densify_and_prune_structgs: gradient trung bình vs kích thước Gaussian "
             "(vùng clone/split, SADGS)", fontsize=12)
fig.tight_layout()
save(fig, "densify_decision")

# ============================================================================
# Hình 5 (ch7_split_sampling): split dị hướng giải tích densify_and_split_structgs
# ============================================================================
children = {}
for (i, m, sc, k) in T.SPLIT_CHILDREN:
    children.setdefault(i, []).append((m, sc, k))
removed = set(int(k) for k in np.where(T.REMOVE)[0])

OFF_ORIG = [(-70, 22), (26, -16), (-30, 26), (34, -30)]
fig, axes = plt.subplots(1, 2, figsize=(16, 7.4))
for ax, (a, b, la, lb) in zip(axes, ((0, 1, "x", "y"), (0, 2, "x", "z"))):
    ax.set_aspect("equal")
    for i in range(N):
        mu = T.P[i]; s = T.S[i, 0]
        ax.add_patch(Circle((mu[a], mu[b]), s, fc=GCOL[i], alpha=0.07, ec=GCOL[i], lw=1.5, ls="--"))
        ax.plot(mu[a], mu[b], "o", color=GCOL[i], ms=8, mec="k", zorder=6)
        ax.annotate(f"{GNAME[i]} s={s:.3f}", (mu[a], mu[b]), xytext=OFF_ORIG[i], textcoords="offset points",
                    fontsize=8, color=GCOL[i], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=GCOL[i], lw=0.6, alpha=0.7))
        if i not in children:
            continue
        for j, (m, sc, k) in enumerate(children[i]):
            ax.add_patch(Circle((m[a], m[b]), sc, fc=GCOL[i], alpha=0.22, ec=GCOL[i], lw=1.0))
            ax.add_patch(FancyArrowPatch((mu[a], mu[b]), (m[a], m[b]), arrowstyle="-|>", mutation_scale=9,
                                         color=GCOL[i], lw=0.9, alpha=0.75, zorder=5))
            ax.plot(m[a], m[b], "s", color=GCOL[i], ms=4, mec="k", zorder=6)
    ax.set_xlabel(la); ax.set_ylabel(lb)
    ax.set_xlim(-4.2, 3.2)
    ax.set_ylim(-2.4, 2.6) if b == 1 else ax.set_ylim(-1.6, 2.4)
    ax.set_title(f"Mặt phẳng {la}{lb} (top-view) — lưới $k\\times k\\times k$ mỗi Gaussian split", fontsize=10)
leg = [Line2D([], [], marker="o", color="gray", mec="k", ls="--", label="gốc: vòng tròn bán kính $s_i$ (nét đứt)"),
       Line2D([], [], marker="s", color="gray", mec="k", ls="-",
              label="con: lưới $k^3$ điểm, offset=(s/k)$\\sqrt{12}$·(i-(k-1)/2), $s_{con}=s/k^{\\mathrm{ks\\_scale\\_power}}$")]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
k_list = ", ".join(f"{GNAME[i]}: k={children[i][0][2]} ({len(children[i])} con)" for i in sorted(children))
fig.suptitle("densify_and_split_structgs: split dị hướng giải tích theo k=ceil(√eta) mỗi trục "
             f"({k_list})\nN: {N} → {N + sum(len(c) for c in children.values()) - len(children)} → "
             f"{N + sum(len(c) for c in children.values()) - len(children) - int(T.REMOVE.sum())} "
             "(sau prune)", fontsize=11.5)
fig.text(0.5, 0.915, f"N: {N} → {N + sum(len(c) for c in children.values()) - len(children)} → "
         f"{N + sum(len(c) for c in children.values()) - len(children) - int(T.REMOVE.sum())}",
         ha="center", fontsize=14, fontweight="bold", bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0a800"))
fig.tight_layout(rect=[0, 0.03, 1, 0.9])
save(fig, "split")

# ============================================================================
# Hình 6 (ch7_n_growth): kịch bản N=10 (multinomial minh hoạ) + luỹ thừa N
# ============================================================================
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
x10 = np.arange(1, 11)
cc = ["#d62728" if c else "#9e9e9e" for c in T.C10]

ax = fig.add_subplot(gs[0, 0])
ax.bar(x10, T.PR10, color=cc, edgecolor="k", lw=0.5)
for k in range(10):
    ax.text(x10[k], T.PR10[k] + 0.02, f"{T.PR10[k]:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.15); ax.set_xticks(x10)
ax.set_title("Pruning$_i$ cho trước (đỏ = $i\\in\\mathcal{C}=\\{3,6,8,9,10\\}$, minh hoạ)", fontsize=9.5)
ax.set_xlabel("i")

ax = fig.add_subplot(gs[0, 1])
ax.bar(x10, T.W10, color=cc, edgecolor="k", lw=0.5)
for k in range(10):
    ax.text(x10[k], T.W10[k] * 1.4, f"{T.W10[k]:.4g}" if T.W10[k] < 1e5 else r"$10^6$", ha="center", fontsize=7.5)
ax.set_yscale("log"); ax.set_ylim(0.5, 1e7); ax.set_xticks(x10)
ax.set_title(r"$w_i=1/(10^{-6}+1-\mathrm{Pruning}_i)$ (log)", fontsize=10)
ax.set_xlabel("i")

ax = fig.add_subplot(gs[0, 2])
fS = T.freqS10 / 1000; fR = T.freq10 / 1000
ax.bar(x10 - 0.2, fS, 0.4, color="#bbbbbb", edgecolor="k", lw=0.5, label=r"tần suất $\in\mathcal{S}$ (được rút)")
ax.bar(x10 + 0.2, fR, 0.4, color=cc, edgecolor="k", lw=0.5, label=r"tần suất bị xoá $=\mathcal{C}\cap\mathcal{S}$")
for k in range(10):
    ax.text(x10[k] - 0.2, fS[k] + 0.015, f"{fS[k]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    ax.text(x10[k] + 0.2, fR[k] + 0.015, f"{fR[k]:.3f}", ha="center", fontsize=6.5, rotation=90, va="bottom")
ax.set_ylim(0, 1.45); ax.set_xticks(x10)
ax.set_title(f"1000 lần rút (seed 0), budget = ⌊0.5·5⌋ = {T.B10}\n(minh hoạ, không mặc định trong SADGS)",
             fontsize=9)
ax.set_xlabel("i")
ax.legend(fontsize=8, loc="upper left")
ax.text(0.03, 0.72, f"budget = {T.B10}\nxoá thật TB = {np.mean(T.nrem10):.3f}\n(min {min(T.nrem10)}, max {max(T.nrem10)})",
        transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0a800"))

ax = fig.add_subplot(gs[1, :])
base = (1 + 0.15) * (1 - 0.05)
n = np.arange(0, 146)
for N0, c in ((4, "#1f77b4"), (100_000, "#d62728")):
    ax.plot(n, N0 * base ** n, color=c, lw=2, label=rf"$N_0={N0:,}$")
    for nn in (28, 145):
        val = N0 * base ** nn
        ax.plot(nn, val, "o", color=c, ms=7, mec="k", zorder=5)
        ax.annotate(f"n={nn}: {val:.3g}", (nn, val), xytext=(-8, 10), textcoords="offset points",
                    fontsize=9, color=c, ha="right" if nn == 145 else "left")
ax.axvline(28, color="gray", ls=":"); ax.axvline(145, color="gray", ls=":")
ax.text(28, 1.5, "interval 500\n→ n = 28", ha="center", fontsize=8, color="gray")
ax.text(145, 1.5, "interval 100\n→ n = 145", ha="right", fontsize=8, color="gray")
ax.set_yscale("log"); ax.set_ylim(1, 1e11); ax.set_xlim(0, 150)
ax.set_xlabel("n = số lần densify"); ax.set_ylabel(r"$N_{\mathrm{cuối}}$ (log)")
ax.set_title(rf"$N_{{\mathrm{{cuối}}}}\approx N_0[(1+r_s)(1-r_p)]^n$, $r_s=0.15$, $r_p=0.05$ → cơ số "
             rf"$={base:.4f}$ (mô hình tăng trưởng minh hoạ, densification_window_width=200)", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
fig.suptitle("Kịch bản N=10: sampling multinomial trọng số minh hoạ (không mặc định trong SADGS); "
             "luỹ thừa của số Gaussian theo số lần densify", fontsize=12)
fig.tight_layout()
save(fig, "prune_multinomial")
