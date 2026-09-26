"""
real_adc_schedule.py
=====================
Sinh cac hinh "REAL" (khong bia, khong gia dinh) cho SADGS, dung DUY NHAT:
  - So lieu that trong DOCS/assets/history.csv  (log per-iteration cua 4 scene
    drjohnson / playroom / train / truck, 7 checkpoint moi scene: iter = 1000..7000)
  - So lieu that trong DOCS/assets/leaderboard.csv (ket qua cuoi cung, iter=7000)
  - Hang so lich trinh THAT doc truc tiep tu SADGS/arguments/__init__.py
  - Vi tri goi ham THAT doc truc tiep tu SADGS/train.py (dong duoc trich trong code)

KHONG co mang so bia dat ("GIA DINH"), KHONG ve duong cong tu ve-tay-cho-khop.
Neu mot hinh la so do luong/schema (khong co du lieu do), no duoc ghi ro
"so do minh hoa luong thuc thi, khong phai so do" trong tieu de/caption.

Hang so lich trinh (trich tu SADGS/arguments/__init__.py, dong so nhu ghi ro ben duoi):
  percent_dense               = 0.001   (dong 85)
  densification_interval      = 100     (dong 87)
  opacity_reset_interval      = 3000    (dong 89)
  opacity_reset_decay         = 0.1     (dong 90)
  densify_from_iter           = 500     (dong 91)
  densify_until_iter          = 15_000  (dong 92)
  prune_from_iter             = 6000    (dong 99)
  prune_until_iter            = 30_000  (dong 100)
  prune_interval              = 3000    (dong 101)
  split_ratio_threshold       = 0.8     (dong 148)
  prune_ratio_threshold       = 0.8     (dong 149)

Vi tri goi ham trong SADGS/train.py (xac minh bang Grep, khong doan):
  dong 316           : if iteration < opt.densify_until_iter:  (khoi ADC dang hoat dong)
  dong 321           : is_normal_densification = iteration > densify_from_iter
                        and iteration % densification_interval == 0
  dong 362-374        : gaussians.densify_and_prune_structgs(...)  (clone+split+prune
                        theo cau truc, dua tren split_ratio_threshold/prune_ratio_threshold)
  dong 402-403         : if iteration % opacity_reset_interval == 0: reset_opacity(...)
                        -> DAY LA nguyen nhan cac cu "sap PSNR" quan sat duoc tai
                        iter = 3000, 6000, ... trong history.csv (opacity ve gan 0,
                        anh render den, PSNR roi tham, roi optimizer phuc hoi dan)
  dong 419 (COMMENT)  : "# gaussians.final_prune_structgs(...)" -> buoc final-prune
                        trong code THAT da bi TAT (comment), khac voi mo ta ly thuyet
                        o cac phan truoc cua sach.

Chay:  python real_adc_schedule.py
Ghi ra: DOCS/BOOK/adc_figures/real_adc_0{1..7}_*.png
"""
import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(OUT, "..", "..", "assets"))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["font.size"] = 9.5
plt.rcParams["figure.constrained_layout.use"] = True

# ---------------------------------------------------------------------------
# 0. Doc du lieu THAT tu CSV (khong sua, khong noi suy, khong them gia tri)
# ---------------------------------------------------------------------------

def read_history(path):
    """Tra ve dict: scene -> list[row dict] (sap theo iter tang dan)."""
    by_scene = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["iter"] = int(row["iter"])
            for k in ("pct", "n_gauss", "ema_loss", "elapsed_s", "ram_gb", "vram_gb",
                      "psnr", "ssim", "lpips", "psnr_norm", "score",
                      "d_score", "d_psnr", "d_ssim", "d_lpips"):
                v = row.get(k, "")
                row[k] = float(v) if v not in ("", None) else None
            row["n_gauss"] = int(row["n_gauss"])
            by_scene[row["scene"]].append(row)
    for s in by_scene:
        by_scene[s].sort(key=lambda r: r["iter"])
    return by_scene


def read_leaderboard(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k not in ("scene", "size"):
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    return rows


HIST = read_history(os.path.join(ASSETS, "history.csv"))
LB = read_leaderboard(os.path.join(ASSETS, "leaderboard.csv"))
SCENES = ["drjohnson", "playroom", "train", "truck"]  # thu tu xuat hien trong history.csv
COLORS = {"drjohnson": "#2563eb", "playroom": "#16a34a", "train": "#dc2626", "truck": "#d97706"}

# Hang so lich trinh THAT (SADGS/arguments/__init__.py, dong so ghi trong docstring o tren)
PERCENT_DENSE = 0.001
DENSIFICATION_INTERVAL = 100
OPACITY_RESET_INTERVAL = 3000
OPACITY_RESET_DECAY = 0.1
DENSIFY_FROM_ITER = 500
DENSIFY_UNTIL_ITER = 15_000
PRUNE_FROM_ITER = 6000
PRUNE_UNTIL_ITER = 30_000
PRUNE_INTERVAL = 3000
SPLIT_RATIO_THRESHOLD = 0.8
PRUNE_RATIO_THRESHOLD = 0.8


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  ->", path)


def reset_lines(ax, max_iter=7000, label_first=True):
    """Ve cac duong doc tai boi so THAT cua opacity_reset_interval trong pham vi log."""
    k = OPACITY_RESET_INTERVAL
    first = True
    while k <= max_iter:
        ax.axvline(k, color="#9333ea", linestyle="--", linewidth=1.1, alpha=0.85, zorder=1)
        if first and label_first:
            ax.text(k, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, "",  # placeholder, dat label rieng ben ngoai
                    ha="center")
            first = False
        k += OPACITY_RESET_INTERVAL


# ---------------------------------------------------------------------------
# Hinh 1: PSNR that theo iteration, ca 4 scene, danh dau opacity_reset_interval
# ---------------------------------------------------------------------------
def fig01_training_curve_all_scenes():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for s in SCENES:
        rows = HIST[s]
        it = [r["iter"] for r in rows]
        ps = [r["psnr"] for r in rows]
        ax.plot(it, ps, "-o", color=COLORS[s], label=s, linewidth=1.8, markersize=4.5)

    for k in range(OPACITY_RESET_INTERVAL, 7001, OPACITY_RESET_INTERVAL):
        ax.axvline(k, color="#7c3aed", linestyle="--", linewidth=1.1, alpha=0.75, zorder=0)
    ax.text(3000, 3.0, f"reset_opacity()\niter % {OPACITY_RESET_INTERVAL} == 0\n(train.py:402-403)",
            fontsize=7.6, color="#7c3aed", ha="center", va="bottom",
            bbox=dict(fc="white", ec="#7c3aed", lw=0.6, alpha=0.9))
    ax.text(6000, 3.0, f"reset_opacity()\niter={6000}",
            fontsize=7.6, color="#7c3aed", ha="center", va="bottom",
            bbox=dict(fc="white", ec="#7c3aed", lw=0.6, alpha=0.9))

    ax.set_xlabel("iteration (checkpoint log that, buoc 1000)")
    ax.set_ylabel("PSNR (dB) - do that tren tap validation")
    ax.set_title("PSNR that theo iteration - 4 scene SADGS (DOCS/assets/history.csv)\n"
                  "Duong dut mau tim = moc opacity_reset_interval=3000 (arguments/__init__.py:89)",
                  fontsize=10.5)
    ax.set_xlim(500, 7300)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5, title="scene")
    savefig(fig, "real_adc_01_training_curve_all_scenes.png")


# ---------------------------------------------------------------------------
# Hinh 2: zoom drjohnson - cu sap/hoi phuc PSNR & SSIM quanh moc opacity reset
# ---------------------------------------------------------------------------
def fig02_opacity_reset_crash():
    rows = HIST["drjohnson"]
    it = [r["iter"] for r in rows]
    ps = [r["psnr"] for r in rows]
    ss = [r["ssim"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)

    ax1.plot(it, ps, "-o", color="#2563eb", linewidth=2, markersize=6)
    for k in range(3000, 7001, 3000):
        ax1.axvline(k, color="#7c3aed", linestyle="--", linewidth=1.1, alpha=0.8)
    p3000 = next(r["psnr"] for r in rows if r["iter"] == 3000)
    p4000 = next(r["psnr"] for r in rows if r["iter"] == 4000)
    p6000 = next(r["psnr"] for r in rows if r["iter"] == 6000)
    p7000 = next(r["psnr"] for r in rows if r["iter"] == 7000)
    ax1.annotate(f"iter=3000: PSNR = {p3000:.2f} dB\n(vua qua reset_opacity)",
                 xy=(3000, p3000), xytext=(3450, p3000 + 6),
                 arrowprops=dict(arrowstyle="->", color="#7c3aed"), fontsize=8, color="#7c3aed")
    ax1.annotate(f"iter=4000: PSNR = {p4000:.2f} dB\n(da phuc hoi)",
                 xy=(4000, p4000), xytext=(4250, p4000 - 9),
                 arrowprops=dict(arrowstyle="->", color="#16a34a"), fontsize=8, color="#16a34a")
    ax1.annotate(f"iter=6000: PSNR = {p6000:.2f} dB\n(reset lan 2)",
                 xy=(6000, p6000), xytext=(6300, p6000 + 6),
                 arrowprops=dict(arrowstyle="->", color="#7c3aed"), fontsize=8, color="#7c3aed")
    ax1.annotate(f"iter=7000: PSNR = {p7000:.2f} dB",
                 xy=(7000, p7000), xytext=(6250, p7000 - 10),
                 arrowprops=dict(arrowstyle="->", color="#16a34a"), fontsize=8, color="#16a34a")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Scene drjohnson - PSNR that sap va hoi phuc quanh moc opacity_reset_interval\n"
                  "(so lieu that: DOCS/assets/history.csv; ham reset_opacity() tai SADGS/train.py:402-403)",
                  fontsize=10)
    ax1.grid(alpha=0.25)

    ax2.plot(it, ss, "-o", color="#dc2626", linewidth=2, markersize=6)
    for k in range(3000, 7001, 3000):
        ax2.axvline(k, color="#7c3aed", linestyle="--", linewidth=1.1, alpha=0.8)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("SSIM")
    ax2.grid(alpha=0.25)

    savefig(fig, "real_adc_02_opacity_reset_crash.png")


# ---------------------------------------------------------------------------
# Hinh 3: N(t) - so luong Gaussian that theo iteration, 4 scene
# ---------------------------------------------------------------------------
def fig03_gaussian_growth():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for s in SCENES:
        rows = HIST[s]
        it = [r["iter"] for r in rows]
        n = [r["n_gauss"] for r in rows]
        ax.plot(it, n, "-o", color=COLORS[s], label=f"{s} (N(7000)={n[-1]:,})",
                linewidth=1.8, markersize=4.5)

    ax.axvline(DENSIFY_FROM_ITER, color="#059669", linestyle=":", linewidth=1.3)
    ax.text(DENSIFY_FROM_ITER, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] else 0,
            f"densify_from_iter={DENSIFY_FROM_ITER}", rotation=90, va="bottom", ha="right",
            color="#059669", fontsize=7.5)
    for k in range(3000, 7001, 3000):
        ax.axvline(k, color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.text(0.985, 0.04,
            f"Ghi chu: log chi den iter=7000. densify_until_iter={DENSIFY_UNTIL_ITER:,} va\n"
            f"prune_from_iter={PRUNE_FROM_ITER:,} (arguments/__init__.py:92,99) NAM NGOAI\n"
            f"pham vi log nay -> khong ve vi khong co so do that o do.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
            bbox=dict(fc="#fff7ed", ec="#d97706", lw=0.6))

    ax.set_xlabel("iteration")
    ax.set_ylabel("so luong Gaussian N(t) (cot n_gauss, so dem that)")
    ax.set_title("N(t) that: tang truong so Gaussian qua clone/split (SADGS/scene/gaussian_model.py)\n"
                  "Duong xanh la cham = densify_from_iter=500; duong tim dut = boi so opacity_reset_interval=3000",
                  fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    savefig(fig, "real_adc_03_gaussian_growth.png")


# ---------------------------------------------------------------------------
# Hinh 4: so do lich trinh (schema, khong phai so do dinh luong)
# ---------------------------------------------------------------------------
def fig04_schedule_timeline():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_title("So do minh hoa luong thuc thi ADC trong SADGS/train.py (KHONG PHAI SO DO DO DAC)\n"
                 "Hang so that: arguments/__init__.py; vi tri goi that: train.py (dong so ghi tren moi khoi)",
                 fontsize=10)

    T0, T1 = 0, 7000
    ax.set_xlim(T0 - 600, T1 + 600)
    ax.set_ylim(0, 7.6)
    ax.set_yticks([])
    ax.set_xlabel("iteration")

    # densify window
    ax.add_patch(mpatches.Rectangle((DENSIFY_FROM_ITER, 0.3), T1 - DENSIFY_FROM_ITER, 0.6,
                                     fc="#dbeafe", ec="#2563eb", lw=1.0, zorder=1))
    ax.text(T0 - 550, 0.6,
            f"cua so densify dang mo:\n[densify_from_iter={DENSIFY_FROM_ITER},\n"
            f"densify_until_iter={DENSIFY_UNTIL_ITER:,})\n(train.py:316)",
            ha="left", va="center", fontsize=7.4, color="#1e3a8a")

    # densification_interval ticks
    y_dens = 1.7
    it = DENSIFY_FROM_ITER + DENSIFICATION_INTERVAL
    while it <= T1:
        ax.plot([it, it], [y_dens - 0.15, y_dens + 0.15], color="#0891b2", lw=1.1)
        it += DENSIFICATION_INTERVAL
    ax.text(T1 / 2, y_dens + 0.35,
            f"moi {DENSIFICATION_INTERVAL} iter (densification_interval): densify_and_prune_structgs()\n"
            f"clone/split theo split_ratio_threshold={SPLIT_RATIO_THRESHOLD}, "
            f"prune theo prune_ratio_threshold={PRUNE_RATIO_THRESHOLD} (train.py:321-374)",
            ha="center", va="bottom", fontsize=7.3, color="#0e7490")

    # opacity reset markers
    y_reset = 3.3
    k = OPACITY_RESET_INTERVAL
    while k <= T1:
        ax.plot([k, k], [y_reset - 0.35, y_reset + 0.35], color="#7c3aed", lw=2.0)
        ax.text(k, y_reset + 0.5, f"reset_opacity(\ndecay={OPACITY_RESET_DECAY})",
                ha="center", va="bottom", fontsize=7.2, color="#7c3aed")
        k += OPACITY_RESET_INTERVAL
    ax.text(T0 - 550, y_reset, f"opacity_reset_interval\n={OPACITY_RESET_INTERVAL}\n(train.py:402-403)",
            ha="left", va="center", fontsize=7.4, color="#7c3aed")

    # prune_iterations / final prune (khong hoat dong)
    y_final = 5.0
    ax.add_patch(mpatches.Rectangle((PRUNE_FROM_ITER, y_final - 0.25), T1 - PRUNE_FROM_ITER, 0.5,
                                     fc="#fee2e2", ec="#dc2626", lw=1.0, hatch="//", zorder=1))
    ax.text(T0 - 550, y_final,
            f"prune_from_iter={PRUNE_FROM_ITER}..prune_until_iter={PRUNE_UNTIL_ITER:,}\n"
            f"prune_interval={PRUNE_INTERVAL} (arguments/__init__.py:99-101)\n"
            f"final_prune_structgs() BI COMMENT trong code that\n"
            f"(train.py:419, gach cheo = khong chay)",
            ha="left", va="center", fontsize=7.2, color="#b91c1c")

    ax.text(T1 + 550, 6.8, "Cham dut log that:\niter=7000\n(history.csv /\nleaderboard.csv)",
            ha="right", va="top", fontsize=7.6, color="#374151",
            bbox=dict(fc="white", ec="#374151", lw=0.6))
    ax.axvline(7000, color="#374151", lw=1.0, ls=":")

    savefig(fig, "real_adc_04_schedule_timeline_schema.png")


# ---------------------------------------------------------------------------
# Hinh 5: loss / VRAM / RAM that theo iteration
# ---------------------------------------------------------------------------
def fig05_loss_vram_ram():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [("ema_loss", "EMA loss (that)"), ("vram_gb", "VRAM (GB, that)"), ("ram_gb", "RAM (GB, that)")]
    for ax, (key, title) in zip(axes, metrics):
        for s in SCENES:
            rows = HIST[s]
            it = [r["iter"] for r in rows]
            v = [r[key] for r in rows]
            ax.plot(it, v, "-o", color=COLORS[s], label=s, linewidth=1.6, markersize=4)
        for k in range(3000, 7001, 3000):
            ax.axvline(k, color="#7c3aed", linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Chi phi/on dinh huan luyen THAT theo iteration (DOCS/assets/history.csv)", fontsize=10.5)
    savefig(fig, "real_adc_05_loss_vram_ram.png")


# ---------------------------------------------------------------------------
# Hinh 6: leaderboard that - ket qua cuoi cung (iter=7000) 4 scene
# ---------------------------------------------------------------------------
def fig06_final_leaderboard():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    scenes = [r["scene"] for r in LB]
    colors = [COLORS.get(s, "#6b7280") for s in scenes]

    ax = axes[0]
    ax.bar(scenes, [r["psnr"] for r in LB], color=colors)
    ax.set_title("PSNR cuoi cung (dB)", fontsize=9.5)
    ax.set_ylabel("dB")
    for i, r in enumerate(LB):
        ax.text(i, r["psnr"] + 0.3, f"{r['psnr']:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    ax.bar(scenes, [r["n_gauss"] for r in LB], color=colors)
    ax.set_title("So Gaussian cuoi cung N(7000)", fontsize=9.5)
    for i, r in enumerate(LB):
        ax.text(i, r["n_gauss"] + 2000, f"{int(r['n_gauss']):,}", ha="center", fontsize=7.5, rotation=0)

    ax = axes[2]
    ax.bar(scenes, [r["train_s"] for r in LB], color=colors)
    ax.set_title("Thoi gian huan luyen that (giay, 7000 iter)", fontsize=9.5)
    for i, r in enumerate(LB):
        ax.text(i, r["train_s"] + 1, f"{r['train_s']:.1f}s", ha="center", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Ket qua that cuoi cung tai iter=7000 (DOCS/assets/leaderboard.csv)", fontsize=10.5)
    savefig(fig, "real_adc_06_final_leaderboard.png")


# ---------------------------------------------------------------------------
# Hinh 7: delta metric (d_psnr/d_ssim/d_lpips) that - dao dong theo reset
# ---------------------------------------------------------------------------
def fig07_delta_oscillation():
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for s in SCENES:
        rows = [r for r in HIST[s] if r["d_psnr"] is not None]
        it = [r["iter"] for r in rows]
        d = [r["d_psnr"] for r in rows]
        ax.plot(it, d, "-o", color=COLORS[s], label=s, linewidth=1.8, markersize=4.5)
    ax.axhline(0, color="#374151", lw=1.0)
    for k in range(3000, 7001, 3000):
        ax.axvline(k, color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("d_psnr = PSNR(t) - PSNR(t-1000)  [cot that trong history.csv]")
    ax.set_title("Dao dong PSNR that (delta giua 2 checkpoint lien tiep) do reset_opacity()\n"
                  "gia tri am manh ngay sau moi boi so opacity_reset_interval=3000", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    savefig(fig, "real_adc_07_delta_psnr_oscillation.png")


if __name__ == "__main__":
    print("Doc du lieu that tu:", os.path.join(ASSETS, "history.csv"), "va leaderboard.csv")
    print("So scene:", list(HIST.keys()), "| so dong moi scene:", {s: len(v) for s, v in HIST.items()})
    fig01_training_curve_all_scenes()
    fig02_opacity_reset_crash()
    fig03_gaussian_growth()
    fig04_schedule_timeline()
    fig05_loss_vram_ram()
    fig06_final_leaderboard()
    fig07_delta_oscillation()
    print("Hoan tat. Tat ca hinh deu ve tu so lieu that trong CSV hoac hang so that trong code (khong bia).")
