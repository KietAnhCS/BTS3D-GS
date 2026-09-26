"""BOT 15 - Tong hop dau-cuoi SADGS: pipeline, so sanh 3 phuong phap,
mo hinh chi phi, doan bay tang toc, nhanh code bi tat.

Moi hinh minh hoa dung 1 cong thuc/so do da trinh bay trong slide part_sadgsx_15.tex.
Tat ca tham so cua mo hinh chi phi la MINH HOA (illustrative), khong phai so do dac.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.normpath(OUT)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


# ---------------------------------------------------------------- FIG 1
# So do khoi 1 iteration SADGS (ve bang matplotlib patches, khong TikZ)
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(-8, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    C_PRE = "#dfe9f5"
    C_CORE = "#cfe8cf"
    C_FREQ = "#ffe2b8"
    C_ADC = "#f6cccc"
    C_OFF = "#e6e6e6"

    def box(x, y, w, h, text, color, ec="#333333", ls="-", fs=7.4):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.35,rounding_size=1.2",
                           fc=color, ec=ec, lw=1.1, linestyle=ls, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.25)

    def arrow(x1, y1, x2, y2, style="-|>", color="#444444", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, lw=1.2, color=color,
                                     linestyle=ls, zorder=1,
                                     connectionstyle="arc3,rad=0.0"))

    ax.text(50, 59.5, "Mot iteration SADGS  (train.py:202-440)",
            ha="center", fontsize=11, fontweight="bold")

    # Hang 0: tien xu ly (ngoai vong lap)
    box(2, 50, 42, 7,
        "TIEN XU LY (1 lan, ngoai vong lap)\nStructure Tensor cache moi anh  (train.py:160-200)",
        C_PRE)
    box(48, 50, 50, 7,
        "compute_3D_filter  (train.py:154-158)  --  mac dinh TAT\n"
        "sample_bbox_faces  (train.py:82-152)   --  mac dinh TAT",
        C_OFF, ls="--")

    # Hang 1
    box(2, 39, 20, 7, "1. update_learning_rate\n+ oneupSHdegree\n(train.py:214-217)", C_CORE)
    box(25, 39, 20, 7, "2. Chon camera\nrandom / fps\n(train.py:220-243)", C_CORE)
    box(48, 39, 22, 7, "3. render_structgs\n-> image, cov2D[N,7]\n(train.py:249-252)", C_CORE)
    box(73, 39, 25, 7, "4. Loss\nL = (1-l)L1 + l(1-SSIM)\n+ lambda_l2 * L2   (256-259)", C_CORE)
    arrow(22, 42.5, 25, 42.5)
    arrow(45, 42.5, 48, 42.5)
    arrow(70, 42.5, 73, 42.5)

    # Hang 2
    box(73, 28, 25, 7, "5. loss.backward()\n(train.py:263)", C_CORE)
    arrow(85.5, 39, 85.5, 35)
    box(45, 28, 25, 7,
        "6. update_freq_stats_online\nCHI khi iter%10==0\n(train.py:275-276)", C_FREQ)
    arrow(73, 31.5, 70, 31.5)
    box(18, 28, 24, 7, "7. Tich luy grad man hinh\nadd_densification_stats\n(train.py:318-319)", C_CORE)
    arrow(45, 31.5, 42, 31.5)

    # vong lap batch
    ax.add_patch(FancyArrowPatch((45, 35.2), (60, 38.6), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.2, color="#1f77b4",
                                 connectionstyle="arc3,rad=0.35", zorder=1))
    ax.text(50, 36.6, "lap batch_size lan (mac dinh 1)", fontsize=6.8,
            color="#1f77b4", ha="center")

    # Hang 3: ADC
    box(2, 15, 30, 9,
        "8. Tinh high_ratio / low_ratio\nsplit_mask = (high_ratio>0.8) & |grad|>=1e-5\n"
        "prune_mask = low_ratio>0.8\n(train.py:337-353)", C_ADC, fs=7.0)
    arrow(20, 28, 20, 24)
    box(35, 15, 30, 9,
        "9. densify_and_prune_structgs\nclone + split di huong + prune\n"
        "(train.py:362-374)\nchi khi iter%densification_interval==0", C_ADC, fs=7.0)
    arrow(32, 19.5, 35, 19.5)
    box(68, 15, 30, 9,
        "10. Reset 9 accumulator ve 0\naccum_eta, accum_view_count,\n"
        "eta_high/mid/low_*  (train.py:377-386)", C_ADC, fs=7.0)
    arrow(65, 19.5, 68, 19.5)

    # Hang 4
    box(68, 4, 30, 7, "11. reset_opacity(decay=0.1)\nmoi 3000 iter (train.py:402-403)", C_CORE)
    arrow(83, 15, 83, 11)
    box(35, 4, 30, 7, "12. prune opacity<0.1\ntai iter in prune_iterations\n(train.py:412-417)", C_CORE, fs=7.0)
    arrow(68, 7.5, 65, 7.5)
    box(2, 4, 30, 7, "13. optimizer step + zero_grad\nhybrid / sparse_adam\n(train.py:422-434)", C_CORE, fs=7.0)
    arrow(35, 7.5, 32, 7.5)

    # vong lap iteration
    ax.add_patch(FancyArrowPatch((2, 7.5), (2, 42.5), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.5, color="#8c564b",
                                 connectionstyle="arc3,rad=-0.30", zorder=1))
    ax.text(-6.2, 26, "iteration += 1", fontsize=7.5, color="#8c564b",
            rotation=90, va="center")

    handles = [Rectangle((0, 0), 1, 1, fc=c, ec="#333333") for c in
               [C_PRE, C_CORE, C_FREQ, C_ADC, C_OFF]]
    ax.legend(handles, ["Tien xu ly", "Loi 3DGS/FastGS", "Tan so (SADGS)",
                        "Densify/Prune (SADGS)", "Mac dinh TAT"],
              loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=7.5,
              frameon=False)
    save(fig, "15_pipeline_blocks.png")


# ---------------------------------------------------------------- FIG 2
# Duong cong N(t): so Gaussian theo iteration cho 3 phuong phap
def fig_N_curves():
    it = np.arange(0, 30001, 50)
    N0 = 0.15e6          # khoi tao tu SfM (minh hoa)
    d_from, d_until, d_int = 500, 15000, 100

    def growth(rate, cap, prune_at_reset=0.0, decay=1.0):
        N = np.zeros_like(it, dtype=float)
        cur = N0
        for i, t in enumerate(it):
            if d_from < t < d_until and t % d_int < 50:
                # tang truong bao hoa: moi vong ADC nhan them rate*N, gioi han cap
                cur = cur + rate * cur * max(0.0, 1 - cur / cap)
            if prune_at_reset > 0 and t % 3000 < 50 and d_from < t < d_until:
                cur = cur * (1 - prune_at_reset)
            if t in (4000, 8000):
                cur = cur * decay   # prune_iterations: opacity < 0.1 (train.py:412-417)
            N[i] = cur
        return N

    N_3dgs = growth(0.155, 6.0e6, prune_at_reset=0.05, decay=0.97)
    N_fast = growth(0.135, 3.2e6, prune_at_reset=0.06, decay=0.96)
    N_sadgs = growth(0.120, 2.4e6, prune_at_reset=0.10, decay=0.94)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(it, N_3dgs / 1e6, lw=2.0, label="3DGS (grad, split ngau nhien)", color="#d62728")
    ax.plot(it, N_fast / 1e6, lw=2.0, label="FastGS (grad + grad_abs)", color="#ff7f0e")
    ax.plot(it, N_sadgs / 1e6, lw=2.2, label=r"SADGS (high_ratio>0.8, prune low_ratio>0.8)",
            color="#2ca02c")

    ax.axvline(500, color="gray", ls=":", lw=1)
    ax.axvline(15000, color="gray", ls="--", lw=1)
    ax.text(700, 5.6, "densify_from_iter=500", fontsize=7, color="gray")
    ax.text(15300, 5.6, "densify_until_iter=15000", fontsize=7, color="gray")
    for r in range(3000, 15001, 3000):
        ax.axvline(r, color="#1f77b4", ls=":", lw=0.6, alpha=0.6)
    ax.text(3200, 0.25, "reset_opacity moi 3000 iter", fontsize=6.5, color="#1f77b4")

    ax.set_xlabel("iteration t")
    ax.set_ylabel("N(t)  [trieu Gaussian]")
    ax.set_title("Quy dao so Gaussian N(t) - mo hinh tang truong bao hoa\n"
                 r"$N \leftarrow N + r\,N\,(1-N/N_{max})$ moi vong densify (minh hoa)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "15_N_curves.png")


# ---------------------------------------------------------------- FIG 3
# Mo hinh chi phi t = a*N + b*P + c
def fig_cost_model():
    N = np.linspace(0.1e6, 6.0e6, 400)
    P_720 = 1280 * 720
    P_1080 = 1920 * 1080

    c = 3.0                 # ms: overhead co dinh (launch kernel, python, optimizer setup)
    b = 3.2e-6              # ms/pixel
    a_base = 4.0e-6         # ms/Gaussian: forward+backward+Adam

    def t_of(a, P, extra=0.0):
        return a * N + b * P + c + extra * N

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    ax = axes[0]
    ax.plot(N / 1e6, t_of(a_base, P_1080), lw=2, color="#d62728",
            label="3DGS 1080p (a=4.0e-6)")
    ax.plot(N / 1e6, t_of(a_base * 0.72, P_1080), lw=2, color="#ff7f0e",
            label="FastGS 1080p, compact box mult=0.7 (a=2.9e-6)")
    ax.plot(N / 1e6, t_of(a_base * 0.72, P_1080, extra=4.0e-7), lw=2.2, color="#2ca02c",
            label=r"SADGS 1080p (+ $\eta$ moi 10 iter: $+a_\eta N/10$)")
    ax.plot(N / 1e6, t_of(a_base * 0.72, P_720, extra=4.0e-7), lw=1.6, ls="--",
            color="#2ca02c", label="SADGS 720p")
    ax.set_xlabel("N  [trieu Gaussian]")
    ax.set_ylabel("t moi iteration  [ms]")
    ax.set_title(r"$t = a\,N + b\,P + c$   (tuyen tinh theo N va so pixel P)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # panel 2: phan ra dong gop tai N = 2.4e6
    ax = axes[1]
    Nq = 2.4e6
    a_s = a_base * 0.72
    comps = {
        "a*N\n(render+backward+Adam)": a_s * Nq,
        "b*P\n(1080p rasterize)": b * P_1080,
        "c\n(overhead co dinh)": c,
        r"$a_\eta N/10$" + "\n(freq stats)": 4.0e-7 * Nq,
        r"$a_D N/100$" + "\n(densify+prune)": 1.2e-6 * Nq / 1.0,
    }
    ks = list(comps.keys())
    vs = [comps[k] for k in ks]
    cols = ["#2ca02c", "#1f77b4", "#7f7f7f", "#ff7f0e", "#d62728"]
    ax.bar(range(len(vs)), vs, color=cols)
    for i, v in enumerate(vs):
        ax.text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, fontsize=6.2, rotation=16, ha="right")
    ax.set_ylabel("ms / iteration")
    ax.set_title(f"Phan ra chi phi tai N={Nq/1e6:.1f}M, P=1080p\n"
                 f"tong = {sum(vs):.2f} ms/iter", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Mo hinh chi phi thoi gian moi iteration (he so minh hoa)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "15_cost_model.png")


# ---------------------------------------------------------------- FIG 4
# Radar so sanh 3 phuong phap
def fig_radar():
    labels = ["Tin hieu\ndensify",
              "Split co\ncau truc",
              "Tieu chi\nprune",
              "Tiet kiem\nN",
              "Toc do\n/iteration",
              "Do don gian\n(it tham so)"]
    # thang diem 1-5, dua tren dac tinh doc duoc tu code (dinh tinh)
    vals = {
        "3DGS":  [2, 1, 2, 2, 2, 5],
        "FastGS": [3, 1, 3, 3, 4, 4],
        "SADGS": [5, 5, 5, 4, 3, 2],
    }
    cols = {"3DGS": "#d62728", "FastGS": "#ff7f0e", "SADGS": "#2ca02c"}

    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111, polar=True)
    for k, v in vals.items():
        vv = np.array(v + v[:1], dtype=float)
        ax.plot(ang, vv, lw=2, color=cols[k], label=k)
        ax.fill(ang, vv, color=cols[k], alpha=0.12)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title("So sanh 6 truc: 3DGS / FastGS / SADGS\n(thang 1-5, dinh tinh tu code)",
                 fontsize=10, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), fontsize=8)
    save(fig, "15_radar.png")


# ---------------------------------------------------------------- FIG 5
# Waterfall: dong gop tung don bay giam chi phi
def fig_waterfall():
    base = 30.0  # ms/iter gia dinh cho 3DGS o N=4.5M, 1080p
    steps = [
        ("Compact box\nmult=0.7\n(FastGS)", -5.6),
        ("Sparse/Fused Adam\noptimizer_type=hybrid", -3.4),
        ("Prune low_ratio>0.8\n=> N giam ~35%", -7.2),
        ("Freq stats chi\nmoi 10 iter", -1.1),
        ("Densify chi moi\n100 iter", -0.9),
        (r"Chi phi them: $\eta$" + "\n+ split di huong", +2.4),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = 0
    cur = base
    ax.bar(x, base, color="#d62728", width=0.62)
    ax.text(x, base + 0.5, f"{base:.1f}", ha="center", fontsize=8)
    labels = ["3DGS\n(co so)"]
    for name, d in steps:
        x += 1
        bottom = cur + d if d < 0 else cur
        ax.bar(x, abs(d), bottom=bottom,
               color="#2ca02c" if d < 0 else "#ff7f0e", width=0.62)
        ax.plot([x - 0.31 - 0.38, x - 0.31], [cur, cur], color="gray", lw=0.8, ls="--")
        ax.text(x, bottom + abs(d) + 0.5, f"{d:+.1f}", ha="center", fontsize=8)
        cur += d
        labels.append(name)
    x += 1
    ax.bar(x, cur, color="#2ca02c", width=0.62)
    ax.text(x, cur + 0.5, f"{cur:.1f}", ha="center", fontsize=8)
    labels.append("SADGS\n(tong)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.6, rotation=18, ha="right")
    ax.set_ylabel("t moi iteration  [ms]")
    ax.set_title("Bieu do thac nuoc: dong gop cua tung don bay len chi phi/iteration\n"
                 "(phan ra theo cac so hang cua $t = aN + bP + c$; gia tri minh hoa)",
                 fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, base * 1.18)
    save(fig, "15_waterfall.png")


# ---------------------------------------------------------------- FIG 6
# Timeline danh dau nhanh code bi tat
def fig_timeline_disabled():
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.set_xlim(-1200, 33500)
    ax.set_ylim(-1.2, 7.2)
    ax.axis("off")

    # truc thoi gian
    ax.annotate("", xy=(32500, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"))
    for t, lab in [(500, "500\ndensify_from"), (4000, "4000\nprune_iter"),
                   (8000, "8000\nprune_iter"), (15000, "15000\ndensify_until"),
                   (30000, "30000\nend")]:
        ax.plot([t, t], [-0.18, 0.18], color="#333333", lw=1.4)
        ax.text(t, -0.95, lab, ha="center", fontsize=7.2)

    # vung hoat dong
    ax.add_patch(Rectangle((500, 0.25), 14500, 0.55, fc="#cfe8cf", ec="none"))
    ax.text(7700, 0.52, "densify + prune + freq stats (dang CHAY)",
            ha="center", va="center", fontsize=7.6)
    ax.add_patch(Rectangle((15000, 0.25), 15000, 0.55, fc="#dfe9f5", ec="none"))
    ax.text(22500, 0.52, "chi toi uu (khong densify)", ha="center", va="center", fontsize=7.6)

    # cac nhanh bi tat
    off = [
        (500, 1.55, "expand_undersized_gs(tau_expand, avg_high_eta_3ch)\n"
                    "train.py:356-359  -- COMMENT OUT  (moi vong densify)"),
        (4000, 2.85, "compute_gaussian_score_structgs + final_prune_structgs\n"
                     "train.py:418-419  -- COMMENT OUT  (tai prune_iterations)"),
        (500, 4.15, "prune da view trong nhanh warmup (prune_points(low_ratio>0.8))\n"
                    "train.py:393-400  -- COMMENT OUT  (+ warmup_densification=False)"),
        (500, 5.45, "training_report(...) -> khong log PSNR/SSIM/LPIPS khi train\n"
                  "train.py:306  -- COMMENT OUT"),
        (30000, 6.55, "scene.save(iteration) cuoi training\ntrain.py:442  -- COMMENT OUT"),
    ]
    for t, y, txt in off:
        ax.plot([t, t], [0.85, y - 0.18], color="#c62828", lw=1.0, ls=":")
        ax.plot(t, 0.85, marker="x", ms=8, color="#c62828", mew=2)
        ha = "left" if t < 25000 else "right"
        dx = 600 if t < 25000 else -600
        p = FancyBboxPatch((t + dx, y - 0.34), 0, 0, boxstyle="round,pad=0.1")
        ax.add_patch(p)
        ax.text(t + dx, y, txt, ha=ha, va="center", fontsize=7.0,
                bbox=dict(boxstyle="round,pad=0.32", fc="#fdecea", ec="#c62828", lw=0.9))

    ax.set_title("Timeline huan luyen 30k iteration: nhanh code DA CAI nhung BI TAT (x do)",
                 fontsize=10.5)
    save(fig, "15_timeline_disabled.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_N_curves()
    fig_cost_model()
    fig_radar()
    fig_waterfall()
    fig_timeline_disabled()
    print("DONE")
