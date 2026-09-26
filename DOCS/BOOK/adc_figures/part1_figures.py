"""
Hinh minh hoa cho PHAN 1 cua chuong 12 (Adaptive Density Control):
  - Lich chay ADC trong vong lap train (timeline 0 -> 30000)
  - N(t) dinh tinh qua cac giai doan

[XOA - DA XAC MINH, xem SADGS/train.py:362-374]
Cac ham fig_1_3..fig_1_12 (co che "buoc (1)-(2)-(3)": e_v pixel-error ->
min-max -> nguong tau_loss=0.1 -> metric_map, va luong PyTorch tuong ung)
DA BI XOA khoi file nay vi mo ta mot duong code KHONG chay trong pipeline
that:
  - utils/fast_utils.py KHONG con ton tai trong SADGS/ (chi con file .pyc cu).
  - Ham get_loss() tuong tu duoc dinh nghia o utils/freq_utils.py:92 nhung
    KHONG duoc goi o dau trong train.py (dead code, chua tung chay).
  - Loi goi THAT trong train.py (dong 362-374) la:
        gaussians.densify_and_prune_structgs(..., importance_score=gaussians.accum_view_count,
                                              pruning_score=None, ...)
    tuc Importance thuc te la SO LAN 1 Gaussian duoc nhin thay qua cac view
    (accum_view_count, nguong 0.5 trong arguments/__init__.py:124), KHONG
    phai so pixel loi trong footprint tung mo ta o day.
Vi day la dead code chua tung chay trong pipeline that (khong chi la minh
hoa gay hieu lam), toan bo cac ham/hinh lien quan da bi xoa thay vi chi
dinh chinh. Xem lich su git neu can tham khao lai noi dung cu.

Chay: python adc_figures/part1_figures.py
Chi dung numpy + matplotlib.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9


DISCLAIMER = ("Minh hoạ lịch ADC theo cấu hình mặc định của SADGS/train.py; "
              "densify/clone/split dùng gradient tích luỹ + eta (structure tensor); "
              "importance_score = accum_view_count, pruning_score = None.")


def save(fig, name):
    fig.text(0.5, 0.001, DISCLAIMER, ha="center", va="bottom", fontsize=6.3, color="#7a2e2e", wrap=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


# ---------------------------------------------------------------------------
# 1_1  Timeline lich chay ADC
# ---------------------------------------------------------------------------
def fig_1_1():
    from matplotlib.patches import Rectangle

    T = 30000
    dens_from, dens_until, interval, reset = 500, 15000, 500, 3000
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Band: tich luy thong ke gradient (t < 15000)
    ax.add_patch(Rectangle((0, 3.6), dens_until, 0.8, color="#c7d9f1", alpha=0.9))
    ax.text(dens_until / 2, 4.0, "tích luỹ accum / accum_abs / denom (t < 15000)",
            ha="center", va="center", fontsize=8.5)

    # Densify + prune (500 < t < 15000, moi 500 preset)
    dens = np.arange(dens_from + interval, dens_until, interval)
    ax.vlines(dens, 2.65, 3.35, color="#2a7d2e", lw=1.2)
    ax.text(dens_until + 300, 3.0, f"densify + prune (multinomial)\nmỗi {interval} vòng, {len(dens)} lần",
            va="center", fontsize=8.5, color="#2a7d2e")

    # Opacity reset (moi 3000, toan bo vi trong nhanh t < densify_until)
    resets = np.arange(reset, dens_until, reset)
    ax.vlines(resets, 1.65, 2.35, color="#b8860b", lw=2.2)
    ax.text(dens_until + 300, 2.0, "reset_opacity (mỗi 3000, chỉ khi t < 15000)\n"
            "alpha <- min(alpha, 0.01)", va="center", fontsize=8.5, color="#b8860b")

    # Final prune -- KHONG chay live (final_prune_structgs bi comment out trong train.py)
    fp = np.arange(18000, T, 3000)
    ax.vlines(fp, 0.65, 1.35, color="#999999", lw=2.5, linestyles="dashed")
    for t in fp:
        ax.text(t, 0.45, str(t), ha="center", fontsize=7.5, color="#999999")
    ax.text(16000, 1.55, "final_prune_structgs: BỊ COMMENT OUT trong train.py -- KHÔNG chạy live (đường đứt)",
            fontsize=8.5, color="#999999")

    # Moc quan trong
    for t, lab, dy in [(500, "500\ndensify_from", -0.6), (3000, "3000\nsize_threshold bật", -0.15),
                       (15000, "15000\ndensify_until", -0.15), (30000, "30000\nkết thúc", -0.15)]:
        ax.axvline(t, color="k", ls=":", lw=0.8)
        ax.text(t, dy, lab, ha="center", va="top", fontsize=7.5)

    ax.set_xlim(-300, T + 9000)
    ax.set_ylim(-1.3, 4.7)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["final prune (dead)", "opacity reset", "densify+prune", "thống kê grad"])
    ax.set_xlabel("vòng lặp t")
    ax.set_title("Lịch ADC theo cấu hình mặc định SADGS/train.py (final prune = dead code, đường đứt xám)")
    ax.grid(axis="x", alpha=0.2)
    save(fig, "1_1_timeline.png")


# ---------------------------------------------------------------------------
# 1_2  N(t) dinh tinh: 3DGS vs SADGS
# [MINH HOA / KHONG PHAI SO LIEU THAT] r_spawn/r_prune/r_final o day la hang so
# tu chon de ve duong cong "dinh tinh", KHONG doc tu SADGS/ hay tu 1 lan chay
# train.py that. PNG output cua ham nay da bi xoa (git rm) vi de gay hieu lam
# la so lieu do duoc; ham van giu lai o day de tham khao neu can ve lai.
# ---------------------------------------------------------------------------
def fig_1_2():
    T = 30000
    t = np.arange(0, T + 1, 100)
    N0 = 100_000

    def simulate(r_spawn, r_prune, r_final, interval=500):
        N = np.zeros_like(t, dtype=float)
        n = N0
        for k, tt in enumerate(t):
            if 500 < tt < 15000 and tt % interval == 0:
                n = n * (1 + r_spawn) * (1 - r_prune)
            if tt % 3000 == 0 and 15000 < tt < 30000:
                n = n * (1 - r_final)
            N[k] = n
        return N

    N_3dgs = simulate(0.085, 0.035, 0.0)
    N_sadgs = simulate(0.045, 0.030, 0.0)  # final prune KHONG chay live -> r_final=0

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, N_3dgs / 1e6, color="#7f8c8d", lw=2, label="3DGS gốc (isotropic clone/split)")
    ax.plot(t, N_sadgs / 1e6, color="#2a7d2e", lw=2, label="SADGS (accum_view_count>0.5, không final prune)")
    ax.axvspan(500, 15000, color="#c7d9f1", alpha=0.35, label="giai đoạn densify (500..15000)")
    ax.set_xlabel("vòng lặp t")
    ax.set_ylabel("N (triệu Gaussian)  —  MINH HOẠ ĐỊNH TÍNH")
    ax.set_title("N(t): mỗi lần densify là một hệ số nhân; final prune KHÔNG chạy live nên không có bậc giảm")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "1_2_N_qualitative.png")


if __name__ == "__main__":
    fig_1_1()
    # fig_1_2() da bi vo hieu hoa: PNG cua no bi xoa (git rm) vi de gay hieu lam
    # la so lieu do duoc thay vi minh hoa dinh tinh; xem chu thich tren ham fig_1_2.
    print("done.")
