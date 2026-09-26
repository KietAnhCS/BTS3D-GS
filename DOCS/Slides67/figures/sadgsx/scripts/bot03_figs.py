# -*- coding: utf-8 -*-
# BOT 03: Thong ke eta online da view & ti le nhat quan (multiview consistency ratio)
# Moi hinh minh hoa dung mot cong thuc trich tu code thuc:
#   SADGS/utils/freq_utils.py:181-417 (update_freq_stats_online)
#   SADGS/train.py:275-386 (high_ratio/low_ratio, split/prune, reset)
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(67)

TAU_HIGH = 1.0   # freq_utils.py:395
TAU_LOW = 0.1    # freq_utils.py:396
SPLIT_R = 0.8    # arguments/__init__.py:148
PRUNE_R = 0.8    # arguments/__init__.py:149


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------- Fig 1
# high_ratio = eta_high_count / accum_view_count  (train.py:341)
# low_ratio  = eta_low_count  / accum_view_count  (train.py:342)
def fig1():
    M = 24                       # so view tich luy
    N = 4000
    # 3 quan the GS: thuc su high-freq, trung gian, nen phang
    p_high = np.concatenate([rng.uniform(0.75, 0.98, N // 3),
                             rng.uniform(0.25, 0.6, N // 3),
                             rng.uniform(0.0, 0.15, N - 2 * (N // 3))])
    p_low = np.clip(0.9 - p_high + rng.normal(0, 0.05, N), 0, 1)

    hc = rng.binomial(M, p_high)          # eta_high_count
    lc = rng.binomial(M, p_low)           # eta_low_count
    high_ratio = hc / M
    low_ratio = lc / M

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax[0].hist(high_ratio, bins=40, color="#c0392b", alpha=0.85)
    ax[0].axvline(SPLIT_R, color="k", ls="--", lw=1.6)
    ax[0].text(SPLIT_R + 0.01, ax[0].get_ylim()[1] * 0.85,
               "split_ratio_threshold = 0.8", fontsize=8, rotation=90, va="top")
    ax[0].set_title(r"high_ratio = eta_high_count / accum_view_count", fontsize=9)
    ax[0].set_xlabel("high_ratio"); ax[0].set_ylabel("So Gaussian")
    sel = (high_ratio > SPLIT_R).sum()
    ax[0].text(0.03, 0.95, f"SPLIT: {sel}/{N} GS", transform=ax[0].transAxes,
               fontsize=8, va="top", bbox=dict(fc="#fdecea", ec="#c0392b"))

    ax[1].hist(low_ratio, bins=40, color="#2471a3", alpha=0.85)
    ax[1].axvline(PRUNE_R, color="k", ls="--", lw=1.6)
    ax[1].text(PRUNE_R + 0.01, ax[1].get_ylim()[1] * 0.85,
               "prune_ratio_threshold = 0.8", fontsize=8, rotation=90, va="top")
    ax[1].set_title(r"low_ratio = eta_low_count / accum_view_count", fontsize=9)
    ax[1].set_xlabel("low_ratio"); ax[1].set_ylabel("So Gaussian")
    selp = (low_ratio > PRUNE_R).sum()
    ax[1].text(0.03, 0.95, f"PRUNE: {selp}/{N} GS", transform=ax[1].transAxes,
               fontsize=8, va="top", bbox=dict(fc="#eaf2fa", ec="#2471a3"))

    fig.suptitle(f"Phan bo ti le nhat quan da view (M = {M} view/chu ky)", fontsize=10)
    save(fig, "03_ratio_hist.png")


# ---------------------------------------------------------------- Fig 2
# Ti le quyet dinh sai theo so view M: 1 view -> nhieu, M lon -> on dinh
def fig2():
    Ms = np.arange(1, 41)
    N = 6000
    trials = 30
    err_split = []
    for M in Ms:
        e1 = []
        for _ in range(trials):
            p = rng.uniform(0, 1, N)          # xac suat "quan sat thay eta cao"
            truth_split = p > SPLIT_R         # nhan dung (M -> vo cuc)
            hc = rng.binomial(M, p)
            e1.append(np.mean(((hc / M) > SPLIT_R) != truth_split))
        err_split.append(np.mean(e1))

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].plot(Ms, err_split, "o-", ms=3, color="#c0392b")
    ax[0].axvline(1, color="gray", ls=":")
    ax[0].annotate("M = 1 (quyet dinh 1 view): sai ~%.0f%%" % (100 * err_split[0]),
                   xy=(1, err_split[0]), xytext=(7, err_split[0] - 0.02), fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="gray"))
    ax[0].set_xlabel("M = accum_view_count")
    ax[0].set_ylabel("Ti le quyet dinh SPLIT sai")
    ax[0].set_title("Sai so quyet dinh giam theo so view", fontsize=9)

    # Do rong khoang tin cay 95% cua uoc luong ti le: 1.96*sqrt(p(1-p)/M)
    for p, c in [(0.8, "#c0392b"), (0.5, "#2471a3"), (0.2, "#27ae60")]:
        ax[1].plot(Ms, 1.96 * np.sqrt(p * (1 - p) / Ms), lw=2, color=c,
                   label=f"p = {p}")
    ax[1].axhline(0.1, color="k", ls="--", lw=1)
    ax[1].text(20, 0.105, "sai so muc tieu 0.1", fontsize=8)
    ax[1].set_xlabel("M = accum_view_count")
    ax[1].set_ylabel(r"$1.96\sqrt{p(1-p)/M}$")
    ax[1].set_title("Do bat dinh cua high_ratio ~ $1/\\sqrt{M}$", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.suptitle("Vi sao phai tich luy nhieu view thay vi quyet dinh tren 1 view", fontsize=10)
    save(fig, "03_view_stability.png")


# ---------------------------------------------------------------- Fig 3
# Vong doi: update moi 10 iter (train.py:275), densify moi
# densification_interval=100 (arguments:87), reset counters (train.py:377-386)
def fig3():
    densify_from, densify_until, interval = 500, 1500, 100
    its = np.arange(400, densify_until + 120)
    upd = its[(its % 10 == 0) & (its < densify_until)]
    dens = its[(its > densify_from) & (its % interval == 0) & (its < densify_until)]

    # accum_view_count cua 1 GS dien hinh: tang moi lan update, ve 0 khi densify
    acc = np.zeros_like(its, dtype=float)
    cur = 0.0
    for i, t in enumerate(its):
        if t in set(upd.tolist()):
            cur += 1.0
        if t in set(dens.tolist()):
            cur = 0.0
        acc[i] = cur

    fig, ax = plt.subplots(2, 1, figsize=(8.4, 4.2), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.6]})
    ax[0].vlines(upd, 0, 1, color="#27ae60", lw=0.8, label="update_freq_stats_online (moi 10 iter)")
    ax[0].vlines(dens, 0, 1.6, color="#c0392b", lw=2.0, label="densify + reset (moi 100 iter)")
    ax[0].axvline(densify_from, color="k", ls="--", lw=1.2)
    ax[0].text(densify_from + 5, 0.08, "densify_from_iter=500", fontsize=8)
    ax[0].axvline(densify_until, color="k", ls="-.", lw=1.2)
    ax[0].text(densify_until - 430, 0.08, "densify_until_iter=15000 (thu nho)", fontsize=8)
    ax[0].set_yticks([]); ax[0].set_ylim(0, 2.6)
    ax[0].legend(fontsize=7, loc="upper center", ncol=2)
    ax[0].set_title("Vong doi thong ke eta online", fontsize=10)

    ax[1].step(its, acc, where="post", color="#2471a3", lw=1.5)
    ax[1].fill_between(its, 0, acc, step="post", alpha=0.25, color="#2471a3")
    for d in dens:
        ax[1].axvline(d, color="#c0392b", lw=0.8, alpha=0.6)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("accum_view_count")
    ax[1].text(0.52, 0.86, "zero_() sau moi lan densify -> cua so truot ~10 mau/chu ky",
               transform=ax[1].transAxes, fontsize=8,
               bbox=dict(fc="#fdecea", ec="#c0392b"))
    save(fig, "03_timeline_reset.png")


# ---------------------------------------------------------------- Fig 4
# avg_high_eta_3ch = eta_high_sum_3ch / eta_high_count (train.py:350)
# max_high_eta = max_eta_3ch (train.py:351, freq_utils.py torch.max running)
def fig4():
    M = 20
    N = 1500
    base = rng.uniform(0.3, 2.2, N)
    obs = np.clip(base[:, None] * rng.lognormal(0, 0.45, (N, M)), 0, None)
    is_high = obs > TAU_HIGH
    cnt = is_high.sum(1)
    ok = cnt > 0
    avg_high = np.where(ok, np.where(is_high, obs, 0).sum(1) / np.maximum(cnt, 1), 0)
    max_eta = obs.max(1)

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax[0].scatter(avg_high[ok], max_eta[ok], s=5, alpha=0.35, color="#8e44ad")
    lim = [0, max(max_eta.max(), avg_high.max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
    ax[0].set_xlabel("avg_high_eta_3ch (trung binh)")
    ax[0].set_ylabel("max_eta_3ch (cuc dai qua view)")
    ax[0].set_title("max luon >= avg: max giu 'truong hop xau nhat'", fontsize=9)
    ax[0].legend(fontsize=8)

    ax[1].hist(avg_high[ok], bins=45, alpha=0.7, color="#27ae60", label="avg_high_eta_3ch")
    ax[1].hist(max_eta, bins=45, alpha=0.6, color="#c0392b", label="max_eta_3ch (dung de chia)")
    ax[1].axvline(TAU_HIGH, color="k", ls="--", lw=1.4)
    ax[1].text(TAU_HIGH + 0.05, ax[1].get_ylim()[1] * 0.9, "TAU_HIGH = 1.0", fontsize=8, rotation=90, va="top")
    ax[1].set_xlabel("eta"); ax[1].set_ylabel("So Gaussian")
    ax[1].set_title("avg lam muot nhieu, max quyet dinh kich thuoc con", fontsize=9)
    ax[1].legend(fontsize=8)
    save(fig, "03_avg_vs_max_eta.png")


# ---------------------------------------------------------------- Fig 5
# is_active_vis = (max_transmittance > tau_T) & (opacity > tau_o)   freq_utils.py:207
#                 & (||grad_xy|| > freq_grad_threshold)             freq_utils.py:218
def fig5():
    N = 20000
    T = rng.beta(2.0, 2.0, N)                    # max_transmittance
    o = rng.beta(1.3, 3.0, N)                    # get_opacity
    g = rng.lognormal(np.log(2e-5), 1.1, N)      # ||grad||_xy

    taus_T = np.linspace(0, 0.9, 40)
    keep_T = [(T > t).mean() * 100 for t in taus_T]
    taus_o = np.linspace(0, 0.5, 40)
    keep_o = [(o > t).mean() * 100 for t in taus_o]

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].plot(taus_T, keep_T, color="#2471a3", lw=2, label="loc transmittance")
    ax[0].plot(taus_o, keep_o, color="#c0392b", lw=2, label="loc opacity")
    ax[0].axvline(0.0, color="#2471a3", ls=":", lw=1.4)
    ax[0].axvline(0.05, color="#c0392b", ls=":", lw=1.4)
    ax[0].text(0.06, 55, "freq_opacity_threshold\n= 0.05", fontsize=7.5, color="#c0392b")
    ax[0].text(0.01, 95, "freq_transmittance\n_threshold = 0.0", fontsize=7.5, color="#2471a3")
    ax[0].set_xlabel("Nguong"); ax[0].set_ylabel("% mau con lai")
    ax[0].set_title("Tac dong tung bo loc len so mau hop le", fontsize=9)
    ax[0].legend(fontsize=8)

    stages = ["visibility\n_filter", "+ transmit\n> 0.0", "+ opacity\n> 0.05", "+ grad\n> 2e-5"]
    m1 = np.ones(N, bool)
    m2 = m1 & (T > 0.0)
    m3 = m2 & (o > 0.05)
    m4 = m3 & (g > 2e-5)
    vals = [m.sum() for m in (m1, m2, m3, m4)]
    bars = ax[1].bar(stages, vals, color=["#7f8c8d", "#2471a3", "#c0392b", "#27ae60"])
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{100*v/N:.0f}%", ha="center",
                   va="bottom", fontsize=8)
    ax[1].set_ylabel("So mau vao thong ke eta")
    ax[1].set_title("Chuoi loc is_active_vis (freq_utils.py:207-218)", fontsize=9)
    ax[1].tick_params(axis="x", labelsize=7.5)
    save(fig, "03_filter_impact.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("done")
