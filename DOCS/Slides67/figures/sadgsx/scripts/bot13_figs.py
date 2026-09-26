# -*- coding: utf-8 -*-
"""
BOT 13 — Siêu tham số SADGS & lịch trình huấn luyện thực tế.
Mọi con số lấy từ code thật:
  - SADGS/arguments/__init__.py (OptimizationParams)
  - SADGS/train.py:202-440 (vòng lặp training)
  - SADGS/utils/freq_utils.py:395-409 (TAU_HIGH=1.0, TAU_LOW=0.1)
  - SADGS/scene/gaussian_model.py:306-355 (optimizer + scheduler)
  - submodules/.../cuda_rasterizer/auxiliary.h:331-336 (t = mult * 2*log(255*alpha))
Output: DOCS/Slides67/figures/sadgsx/13_*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.normpath(OUT)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------------
# HÌNH 1 — Gantt / timeline lịch trình huấn luyện thật (train.py:202-440)
# ----------------------------------------------------------------------------
def fig_timeline():
    IT = 30000
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    # (nhãn, start, end, màu, ghi chú tần suất)
    rows = [
        ("Vòng lặp chính\n(train.py:202)", 1, IT, "#4C72B0", "mọi iteration"),
        ("oneupSHdegree\n(iter % 1 == 0)", 1, 4, "#DD8452", "SH 0->3 trong 3 iter"),
        ("update_freq_stats_online\n(iter % 10 == 0)", 10, 15000, "#55A868",
         "cập nhật eta mỗi 10 iter"),
        ("add_densification_stats\n(iter < densify_until)", 1, 15000, "#8172B3", "mọi iteration"),
        ("warmup_densification\n(iter % 100, nếu bật)", 600, 15000, "#C44E52",
         "mặc định TẮT (False)"),
        ("densify_and_prune_structgs\n(iter % 100 == 0)", 600, 15000, "#937860",
         "densification_interval=100"),
        ("size_threshold = 20\n(iter > opacity_reset_interval)", 3000, 15000, "#DA8BC3",
         "trước đó = None"),
        ("reset_opacity(0.1)\n(iter % 3000 == 0)", 3000, 15000, "#8C8C8C", "3k, 6k, 9k, 12k"),
        ("prune_iterations\n(opacity < 0.1)", 4000, 8000, "#CCB974", "mặc định [4000, 8000]"),
        ("Tinh chỉnh (không densify)", 15000, IT, "#64B5CD", "chỉ tối ưu tham số"),
    ]

    for i, (lab, s, e, c, note) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.add_patch(Rectangle((s, y - 0.32), max(e - s, 120), 0.64,
                               facecolor=c, edgecolor="black", lw=0.6, alpha=0.88))
        ax.text(IT + 700, y, note, va="center", ha="left", fontsize=7, color="#333333")

    # các mốc rời rạc
    for it in [3000, 6000, 9000, 12000]:
        ax.plot([it, it], [-0.6, len(rows) - 0.4], ls=":", lw=0.9, color="#8C8C8C", zorder=0)
    for it, lab in [(500, "densify_from\n500"), (15000, "densify_until\n15000"),
                    (30000, "iterations\n30000")]:
        ax.plot([it, it], [-0.6, len(rows) - 0.4], ls="--", lw=1.1, color="#C44E52", zorder=0)
        ax.text(it, len(rows) - 0.25, lab, fontsize=7, color="#C44E52",
                ha="center", va="bottom")
    for it in [4000, 8000]:
        ax.plot([it, it], [-0.6, len(rows) - 0.4], ls="-.", lw=0.9, color="#CCB974", zorder=0)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.2)
    ax.set_xlim(0, IT * 1.02)
    ax.set_ylim(-0.7, len(rows) + 0.3)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_title("Lịch trình huấn luyện thực tế của SADGS (train.py:202–440)", fontsize=10.5)
    ax.set_xticks([0, 500, 3000, 6000, 9000, 12000, 15000, 20000, 25000, 30000])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", alpha=0.18)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    save(fig, "13_lich_trinh.png")


# ----------------------------------------------------------------------------
# HÌNH 2 — Độ nhạy: prune_ratio_threshold, TAU_HIGH, mult
# ----------------------------------------------------------------------------
def simulate_eta(n_gs=20000, n_view=60, seed=0):
    """Mô phỏng eta_max (max trên 3 trục) mà update_freq_stats_online tích lũy:
    mỗi Gaussian có mức vi phạm tần số nền lognormal, mỗi view nhiễu quanh đó."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=-1.0, sigma=1.0, size=(n_gs, 1))   # phần lớn < 1
    noise = rng.lognormal(mean=0.0, sigma=0.45, size=(n_gs, n_view))
    return base * noise      # [N, V] eta_max_scalar mỗi quan sát


def fig_sensitivity():
    eta = simulate_eta()
    n_gs, n_view = eta.shape
    N0 = 1.0  # tỉ lệ so với số Gaussian ban đầu

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

    # --- (a) quét prune_ratio_threshold (mặc định 0.8), TAU_LOW = 0.1
    TAU_LOW = 0.1
    low_ratio = (eta <= TAU_LOW).sum(axis=1) / n_view
    taus = np.linspace(0.0, 1.0, 101)
    pruned = [(low_ratio > t).mean() * 100 for t in taus]
    ax = axes[0]
    ax.plot(taus, pruned, lw=2, color="#C44E52")
    ax.axvline(0.8, ls="--", color="black", lw=1.1)
    ax.text(0.8, max(pruned) * 0.92, " mặc định\n 0.8", fontsize=7.5)
    ax.set_xlabel("prune_ratio_threshold", fontsize=8.5)
    ax.set_ylabel("% Gaussian bị prune / lần densify", fontsize=8.5)
    ax.set_title("(a) Ngưỡng đồng thuận prune\n" r"$\frac{n_{low}}{n_{view}} > \tau_{prune}$",
                 fontsize=9)
    ax.grid(alpha=0.25)

    # --- (b) quét TAU_HIGH (freq_utils.py:395) với split_ratio_threshold = 0.8
    ax = axes[1]
    for srt, c in [(0.6, "#8172B3"), (0.8, "#4C72B0"), (0.9, "#55A868")]:
        th = np.linspace(0.2, 3.0, 80)
        split = [((eta > t).sum(axis=1) / n_view > srt).mean() * 100 for t in th]
        ax.plot(th, split, lw=2, color=c, label=f"split_ratio_threshold={srt}")
    ax.axvline(1.0, ls="--", color="black", lw=1.1)
    ax.text(1.02, 40, r" $\tau_{high}=1.0$" + "\n (freq_utils:395)", fontsize=7.5)
    ax.set_xlabel(r"$\tau_{high}$", fontsize=8.5)
    ax.set_ylabel("% Gaussian được split", fontsize=8.5)
    ax.set_title(r"(b) Ngưỡng vi phạm tần số $\eta > \tau_{high}$", fontsize=9)
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25)

    # --- (c) mult: t = mult * 2 log(255 alpha); bán trục bbox ~ sqrt(t) -> diện tích ~ t
    ax = axes[2]
    mult = np.linspace(0.05, 1.5, 150)
    for alpha, c in [(0.1, "#DD8452"), (0.5, "#4C72B0"), (0.9, "#55A868")]:
        t0 = 2.0 * np.log(255.0 * alpha)
        half = np.sqrt(np.clip(mult * t0, 0, None))   # bán trục bbox ~ sqrt(t), đơn vị sigma
        ax.plot(mult, half, lw=2, color=c, label=fr"$\alpha={alpha}$")
    ax.axvline(0.7, ls="--", color="black", lw=1.1)
    ax.text(0.72, 0.6, " mult=0.7\n bbox thu nhỏ\n $\\sqrt{0.7}\\approx 84\\%$", fontsize=7.2)
    ax.set_xlabel("mult (compact box)", fontsize=8.5)
    ax.set_ylabel(r"Bán trục bbox (đơn vị $\sigma$): $\sqrt{t}$", fontsize=8)
    ax.set_title("(c) mult điều khiển số tile mỗi splat\n" r"$t=\mathrm{mult}\cdot 2\log(255\alpha)$",
                 fontsize=9)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.25)

    fig.suptitle("Độ nhạy số Gaussian / chi phí render theo siêu tham số SADGS (mô phỏng)",
                 fontsize=10.5, y=1.04)
    save(fig, "13_do_nhay.png")


# ----------------------------------------------------------------------------
# HÌNH 3 — Bảng nhiệt ảnh hưởng tham số
# ----------------------------------------------------------------------------
def fig_heatmap():
    params = [
        "mult = 0.7",
        "dense = 0.001",
        "split_ratio_threshold = 0.8",
        "prune_ratio_threshold = 0.8",
        "freq_grad_threshold = 2e-5",
        "freq_opacity_threshold = 0.05",
        "freq_transmittance_threshold = 0.0",
        "eta_compute_mode = wavelength",
        "densification_interval = 100",
        "opacity_reset_decay = 0.1",
        "highfeature_lr = 0.005",
        "lambda_l2 = 2.0",
        "batch_size = 1",
        "adam_eps_order = 8",
        "warmup_densification = False",
    ]
    aspects = ["Số Gaussian", "Tốc độ\n(thời gian/iter)", "VRAM", "Chất lượng\n(PSNR/chi tiết)",
               "Ổn định\nhuấn luyện"]
    # thang -3..+3: mức độ ảnh hưởng khi TĂNG giá trị tham số
    M = np.array([
        [0, -3, -1, -1, 0],     # mult
        [+3, -2, -2, +1, -1],   # dense (tăng -> nhiều clone hơn)
        [-3, +2, +2, -2, +1],   # split_ratio_threshold
        [-3, +2, +2, -1, 0],    # prune_ratio_threshold
        [-2, +1, +1, -1, +1],   # freq_grad_threshold
        [-2, +1, +1, -1, +1],   # freq_opacity_threshold
        [-2, +1, +1, -1, +1],   # freq_transmittance_threshold
        [+1, 0, +1, +2, 0],     # eta_compute_mode
        [-2, +2, +1, -1, +2],   # densification_interval (tăng -> ít densify)
        [-1, 0, 0, +1, +2],     # opacity_reset_decay
        [0, -1, 0, +2, -1],     # highfeature_lr
        [0, 0, 0, +2, +1],      # lambda_l2
        [0, -2, -2, +1, +3],    # batch_size
        [0, 0, 0, +1, +2],      # adam_eps_order
        [+2, -2, -2, +1, +1],   # warmup_densification
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(aspects)))
    ax.set_xticklabels(aspects, fontsize=7.8)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=7.4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            ax.text(j, i, f"{v:+d}" if v else "0", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) >= 2 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Hướng ảnh hưởng khi TĂNG giá trị tham số", fontsize=7.8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Bảng nhiệt: ảnh hưởng định tính của siêu tham số SADGS", fontsize=10)
    ax.set_xticks(np.arange(-.5, len(aspects), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(params), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    save(fig, "13_heatmap.png")


# ----------------------------------------------------------------------------
# HÌNH 4 — Đường cong learning rate theo flag
# ----------------------------------------------------------------------------
def expon_lr(step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1_000_000):
    """Sao chép get_expon_lr_func — utils/general_utils.py:32."""
    step = np.asarray(step, dtype=float)
    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def fig_lr():
    it = np.arange(0, 30001, 10)
    S = 1.0  # spatial_lr_scale (chuẩn hoá)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))

    # (a) xyz + hằng số
    ax = axes[0]
    ax.plot(it, expon_lr(it, 0.00016 * S, 0.0000016 * S, 0, 0.01, 30000),
            lw=2, color="#4C72B0", label="xyz: 1.6e-4 → 1.6e-6 (expon)")
    for val, lab, c in [(0.05, "opacity_lr = 0.05", "#C44E52"),
                        (0.01, "scaling_lr = 0.01", "#55A868"),
                        (0.002, "rotation_lr = 0.002", "#DD8452"),
                        (0.0025, "lowfeature_lr = 0.0025 (f_dc)", "#8172B3"),
                        (0.005 / 20.0, "highfeature_lr/20 = 2.5e-4 (f_rest)", "#937860")]:
        ax.axhline(val, ls="--", lw=1.4, color=c, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=8.5)
    ax.set_ylabel("Learning rate (log)", fontsize=8.5)
    ax.set_title("(a) LR mặc định — gaussian_model.py:306–328", fontsize=9.5)
    ax.legend(fontsize=6.6, loc="center right")
    ax.grid(alpha=0.25, which="both")

    # (b) scale_rotation_scheduler bật / tắt
    ax = axes[1]
    ax.plot(it, expon_lr(it, 0.01 * 3.0, 0.01 * 0.5, 0, 0.01, 30000),
            lw=2, color="#55A868", label="scaling: 0.03 → 0.005 (scheduler BẬT)")
    ax.axhline(0.01, ls="--", lw=1.6, color="#55A868", alpha=0.7,
               label="scaling = 0.01 (mặc định, scheduler TẮT)")
    ax.plot(it, expon_lr(it, 0.002 * 3.0, 0.002 * 0.5, 0, 0.01, 30000),
            lw=2, color="#DD8452", label="rotation: 0.006 → 0.001 (scheduler BẬT)")
    ax.axhline(0.002, ls="--", lw=1.6, color="#DD8452", alpha=0.7,
               label="rotation = 0.002 (mặc định, scheduler TẮT)")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=8.5)
    ax.set_ylabel("Learning rate (log)", fontsize=8.5)
    ax.set_title("(b) Cờ scale_rotation_scheduler (mặc định False)", fontsize=9.5)
    ax.legend(fontsize=6.6)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Đường cong learning rate của SADGS theo từng flag", fontsize=10.5, y=1.03)
    save(fig, "13_lr.png")


if __name__ == "__main__":
    fig_timeline()
    fig_sensitivity()
    fig_heatmap()
    fig_lr()
    print("DONE")
