# -*- coding: utf-8 -*-
"""
BOT 08 — final_prune_structgs: prune cuoi theo pruning score & multi-view consistency.
Moi hinh minh hoa dung 1 cong thuc trich tu code that:
  SADGS/scene/gaussian_model.py:1059-1066  (final_prune_structgs)
  SADGS/train.py:412-419                   (prune_iterations, loi goi bi comment)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.abspath(OUT)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


rng = np.random.default_rng(8)
N = 20000
TAU_S = 0.9      # scores_mask = pruning_score > 0.9   (gaussian_model.py:1064)
MIN_OP = 0.1     # min_opacity = 0.1                   (train.py:419)

# ---------------------------------------------------------------- Fig 1
# final_prune = (opacity < min_opacity) OR (pruning_score > 0.9)
# hon hop: phan lon Gaussian nhat quan da view, mot duoi nho bat nhat manh
mix = rng.random(N) < 0.10
score = np.where(mix, rng.beta(9.0, 1.3, N), rng.beta(2.2, 4.5, N))
score = np.clip(score + rng.normal(0, 0.03, N), 0, 1)
opacity = np.clip(rng.beta(2.5, 2.0, N), 0, 1)
m_op = opacity < MIN_OP
m_sc = score > TAU_S
m_final = m_op | m_sc

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8))
bins = np.linspace(0, 1, 60)
ax[0].hist(score[~m_sc], bins=bins, color="#4C72B0", label="giữ lại")
ax[0].hist(score[m_sc], bins=bins, color="#C44E52", label="bị loại (score > 0.9)")
ax[0].axvline(TAU_S, color="k", ls="--", lw=1.6)
ax[0].text(TAU_S - 0.03, ax[0].get_ylim()[1] * 0.45, r"$\tau_s=0.9$",
           ha="right", fontsize=9)
ax[0].set_xlabel("pruning_score (bất nhất đa view)")
ax[0].set_ylabel("số Gaussian")
ax[0].set_title("Phân bố pruning_score & ngưỡng cắt", fontsize=10)
ax[0].legend(fontsize=8)

bins2 = np.linspace(0, 1, 60)
ax[1].hist(opacity[~m_op], bins=bins2, color="#55A868", label="giữ lại")
ax[1].hist(opacity[m_op], bins=bins2, color="#C44E52", label=r"bị loại ($\alpha<0.1$)")
ax[1].axvline(MIN_OP, color="k", ls="--", lw=1.6)
ax[1].text(MIN_OP + 0.03, ax[1].get_ylim()[1] * 0.45, "min_opacity=0.1", fontsize=9)
ax[1].set_xlabel(r"opacity $\alpha$")
ax[1].set_title("Nhánh opacity của cùng phép OR", fontsize=10)
ax[1].legend(fontsize=8)

fig.suptitle(r"final_prune $=\ (\alpha<0.1)\ \vee\ (s>0.9)$   —  loại %.1f%% (%d/%d)"
             % (100 * m_final.mean(), m_final.sum(), N), fontsize=11)
fig.tight_layout()
save(fig, "08_score_hist.png")

# ---------------------------------------------------------------- Fig 2
# Duong cong danh doi: thay doi tau_s -> ty le giu lai vs chat luong mo phong
taus = np.linspace(0.5, 1.0, 60)
keep = np.array([1.0 - ((opacity < MIN_OP) | (score > t)).mean() for t in taus])
# mo hinh chat luong: bao hoa theo so Gaussian giu lai (minh hoa)
psnr = 30.5 - 9.0 * np.exp(-9.0 * (keep - 0.55))

fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
ax1.plot(taus, 100 * keep, color="#4C72B0", lw=2, label="tỉ lệ Gaussian giữ lại")
ax1.set_xlabel(r"ngưỡng $\tau_s$ trên pruning_score")
ax1.set_ylabel("Gaussian giữ lại (%)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(taus, psnr, color="#C44E52", lw=2, ls="--", label="PSNR (mô phỏng)")
ax2.set_ylabel("PSNR (dB) — mô phỏng", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")
ax1.axvline(TAU_S, color="k", ls=":", lw=1.5)
ax1.annotate(r"$\tau_s=0.9$ (code)", xy=(TAU_S, 100 * keep[-12]),
             xytext=(0.66, 86), fontsize=9,
             arrowprops=dict(arrowstyle="->", lw=1))
ax1.set_title("Đánh đổi: số Gaussian giữ lại vs chất lượng", fontsize=11)
fig.tight_layout()
save(fig, "08_tradeoff.png")

# ---------------------------------------------------------------- Fig 3
# Timeline huan luyen, danh dau prune_iterations = [4000, 8000]
fig, ax = plt.subplots(figsize=(8.6, 3.2))
ax.hlines(0, 0, 30000, color="#333333", lw=2)
ax.fill_between([500, 15000], -0.12, 0.12, color="#4C72B0", alpha=0.18)
ax.text(7700, 0.18, "densify_and_prune_structgs\n(500 → 15 000)", ha="center", fontsize=8.5)

for it in (4000, 8000):
    ax.plot(it, 0, "v", color="#C44E52", ms=12)
    ax.annotate("iter %d" % it, xy=(it, 0), xytext=(it, -0.40), ha="center",
                fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
ax.text(16500, -0.55,
        "prune_iterations = [4000, 8000]\n"
        r"  • đang chạy: prune_points($\alpha<0.1$)" "\n"
        "  • final_prune_structgs(...): BỊ COMMENT",
        fontsize=8.5, va="center", ha="left",
        bbox=dict(boxstyle="round", fc="#FDF3F3", ec="#C44E52"))
for it, lab in ((500, "densify_from"), (15000, "densify_until"), (30000, "iterations")):
    ax.plot(it, 0, "|", color="#333333", ms=16)
    ax.text(it, 0.22, "%s\n%d" % (lab, it), ha="center", fontsize=8)

ax.set_xlim(-800, 31500)
ax.set_ylim(-0.9, 0.55)
ax.set_yticks([])
ax.set_xlabel("iteration")
ax.set_title("Timeline huấn luyện — điểm final prune (train.py:412–419)", fontsize=11)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
save(fig, "08_timeline.png")

# ---------------------------------------------------------------- Fig 4
# Kich thuoc mo hinh co / khong final prune
n_base = N
n_op_only = int((~m_op).sum())          # nhanh dang chay: chi opacity
n_full = int((~m_final).sum())          # neu bat final_prune_structgs
bytes_per_gs = 62 * 4                   # 62 float32/Gaussian (.ply 3DGS: xyz,normal,SH,opa,scale,rot)
labels = ["Không prune\ncuối", "Chỉ α<0.1\n(code đang chạy)", "final_prune_structgs\n(α<0.1 ∨ s>0.9)"]
vals = np.array([n_base, n_op_only, n_full])
mb = vals * bytes_per_gs / 1e6

fig, ax = plt.subplots(figsize=(6.6, 4.0))
bars = ax.bar(labels, mb, color=["#8C8C8C", "#4C72B0", "#55A868"], width=0.55)
for b, v, m in zip(bars, vals, mb):
    ax.text(b.get_x() + b.get_width() / 2, m, "%d GS\n%.2f MB" % (v, m),
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("kích thước .ply (MB, 62 float32/Gaussian)")
ax.set_ylim(0, mb.max() * 1.28)
ax.set_title("Kích thước mô hình có / không final prune (N=%d minh hoạ)" % N, fontsize=11)
ax.tick_params(axis="x", labelsize=8.5)
fig.tight_layout()
save(fig, "08_modelsize.png")

# ---------------------------------------------------------------- Fig 5
# Heatmap nhat quan da view: 40 Gaussian x 60 view (sampling_cameras num_cams=60)
G, V = 40, 60
base = np.concatenate([rng.uniform(0.0, 0.35, 28), rng.uniform(0.75, 1.0, 12)])
rng.shuffle(base)
M = np.clip(base[:, None] + rng.normal(0, 0.12, (G, V)), 0, 1)
row_score = M.mean(axis=1)
order = np.argsort(row_score)
M, row_score = M[order], row_score[order]

fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.0),
                       gridspec_kw={"width_ratios": [3, 1]})
im = ax[0].imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
ax[0].set_xlabel("view (60 camera từ sampling_cameras, fps)")
ax[0].set_ylabel("chỉ số Gaussian (đã sắp xếp)")
ax[0].set_title("Bất nhất tái dựng theo từng view", fontsize=10)
fig.colorbar(im, ax=ax[0], label="bất nhất/view")

ax[1].barh(np.arange(G), row_score,
           color=np.where(row_score > TAU_S, "#C44E52", "#4C72B0"))
ax[1].axvline(TAU_S, color="k", ls="--", lw=1.5)
ax[1].set_xlim(0, 1)
ax[1].set_ylim(-0.5, G - 0.5)
ax[1].set_yticks([])
ax[1].set_xlabel("pruning_score (trung bình đa view)")
ax[1].set_title(r"$s>\tau_s=0.9\Rightarrow$ loại (%d GS)" % int((row_score > TAU_S).sum()),
                fontsize=10)
fig.tight_layout()
save(fig, "08_mvconsistency.png")

print("DONE")
