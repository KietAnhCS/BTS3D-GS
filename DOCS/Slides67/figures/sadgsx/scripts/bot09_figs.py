# -*- coding: utf-8 -*-
"""BOT 09 - Optimizer, learning-rate schedule & opacity reset trong SADGS.
Moi con so lay tu code that:
  SADGS/arguments/__init__.py: position_lr_init=0.00016 (79), position_lr_final=1.6e-6 (80),
      position_lr_delay_mult=0.01 (81), position_lr_max_steps=30000 (82),
      opacity_lr=0.05 (76), scaling_lr=0.01 (77), rotation_lr=0.002 (78),
      lowfeature_lr=0.0025 (111), highfeature_lr=0.005 (110),
      opacity_reset_interval=3000 (89), opacity_reset_decay=0.1 (90), adam_eps_order=8 (156)
  SADGS/utils/general_utils.py: get_expon_lr_func (32-64)
  SADGS/scene/gaussian_model.py: training_setup (286-340), update_learning_rate (341-356),
      reset_opacity (415-433), oneupSHdegree (257-259)
  SADGS/train.py: oneupSHdegree moi iteration (216-217), reset_opacity (402-403)
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
    print("saved", p)


# ---------------------------------------------------------------- expon lr
def expon_lr(step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0,
             max_steps=1_000_000):
    """Ban sao chinh xac get_expon_lr_func (utils/general_utils.py:32-64)."""
    step = np.asarray(step, dtype=float)
    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


LR_INIT, LR_FINAL = 0.00016, 0.0000016
DELAY_MULT, MAX_STEPS = 0.01, 30000
SPATIAL = 1.0  # lr thuc te = gia tri nay * spatial_lr_scale (cameras_extent)


# ---- Fig 1: duong cong lr(xyz) theo iteration (log y), co/khong delay -----
def fig1():
    it = np.arange(0, 30001)
    lr_code = expon_lr(it, LR_INIT, LR_FINAL, 0, DELAY_MULT, MAX_STEPS)
    lr_delay = expon_lr(it, LR_INIT, LR_FINAL, 3000, DELAY_MULT, MAX_STEPS)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(it, lr_code, lw=2.2, color="#1f77b4",
            label="SADGS thuc te: lr_delay_steps=0 $\\Rightarrow$ delay_rate=1")
    ax.plot(it, lr_delay, lw=2.0, ls="--", color="#d62728",
            label="Neu lr_delay_steps=3000 (delay_mult=0.01)")
    ax.axvline(3000, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("lr(xyz)  [nhan spatial_lr_scale]")
    ax.set_title("Lich lr vi tri: $lr(t)=\\exp[(1-t)\\ln lr_{init}+t\\ln lr_{fin}]$,"
                 " $t=\\mathrm{clip}(k/30000,0,1)$", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.annotate("lr_init=1.6e-4", (0, LR_INIT), textcoords="offset points",
                xytext=(35, 6), fontsize=7.5)
    ax.annotate("lr_final=1.6e-6", (30000, LR_FINAL), textcoords="offset points",
                xytext=(-95, 8), fontsize=7.5)
    save(fig, "09_lr_xyz_schedule.png")


# ---- Fig 2: so sanh nhieu max_steps + lr hang so cac nhom khac -----------
def fig2():
    it = np.arange(0, 30001)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    ax = axes[0]
    for ms, c in zip([10000, 20000, 30000, 60000],
                     ["#9467bd", "#2ca02c", "#1f77b4", "#ff7f0e"]):
        ax.plot(it, expon_lr(it, LR_INIT, LR_FINAL, 0, DELAY_MULT, ms),
                lw=1.8, color=c, label=f"max_steps={ms}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("lr(xyz)")
    ax.set_title("Anh huong position_lr_max_steps", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    groups = [("xyz (scheduled)", None, "#1f77b4"),
              ("opacity", 0.05, "#d62728"),
              ("scaling", 0.01, "#2ca02c"),
              ("f_rest = highfeature_lr/20", 0.005 / 20.0, "#9467bd"),
              ("f_dc = lowfeature_lr", 0.0025, "#ff7f0e"),
              ("rotation", 0.002, "#8c564b")]
    for name, v, c in groups:
        if v is None:
            ax.plot(it, expon_lr(it, LR_INIT, LR_FINAL, 0, DELAY_MULT, MAX_STEPS),
                    lw=2.0, color=c, label=name)
        else:
            ax.plot(it, np.full_like(it, v, dtype=float), lw=1.8, color=c, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("learning rate")
    ax.set_title("lr rieng tung nhom tham so (training_setup:306-328)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.8, loc="lower left")
    fig.tight_layout()
    save(fig, "09_lr_groups_maxsteps.png")


# ---- Fig 3: histogram opacity truoc/sau reset ---------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_sigmoid(y):
    y = np.clip(y, 1e-12, 1 - 1e-12)
    return np.log(y / (1 - y))


def reset_opacity_code(raw, coef, decay=0.1):
    """reset_opacity (gaussian_model.py:415-433):
       a = sigmoid(raw)*coef ; a *= decay ; a /= coef ; raw_new = inv_sigmoid(a)
       => he so 3D-filter coef TRIET TIEU, ket qua = inv_sigmoid(decay*sigmoid(raw))."""
    a = sigmoid(raw) * coef
    a = a * decay
    a = a / coef
    return inv_sigmoid(a)


def fig3():
    rng = np.random.default_rng(0)
    n = 20000
    raw = rng.normal(0.0, 2.0, n)          # opacity tho truoc reset
    coef = rng.uniform(0.55, 0.98, n)      # he so 3D filter
    alpha_before = sigmoid(raw)
    raw_after = reset_opacity_code(raw, coef, 0.1)
    alpha_after = sigmoid(raw_after)
    alpha_3dgs = np.minimum(alpha_before, 0.01)   # 3DGS goc: min(a, 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    ax = axes[0]
    bins = np.linspace(0, 1, 60)
    ax.hist(alpha_before, bins=bins, color="#1f77b4", alpha=0.75, label="truoc reset")
    ax.hist(alpha_after, bins=bins, color="#d62728", alpha=0.75, label="sau reset (SADGS)")
    ax.axvline(0.1, color="k", ls=":", lw=1, label="nguong prune 0.1")
    ax.set_xlabel(r"$\alpha=\sigma(o)$")
    ax.set_ylabel("so Gaussian")
    ax.set_title(r"SADGS: $\alpha \leftarrow 0.1\,\alpha$ (nhan), decay_factor=0.1", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(alpha_before[:3000], alpha_after[:3000], s=4, color="#d62728",
               label="SADGS: $0.1\\alpha$")
    ax.scatter(alpha_before[:3000], alpha_3dgs[:3000], s=4, color="#2ca02c",
               label="3DGS goc: $\\min(\\alpha,0.01)$")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="y=x")
    ax.axhline(0.1, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\alpha$ truoc reset")
    ax.set_ylabel(r"$\alpha$ sau reset")
    ax.set_title("Anh xa reset: nhan (SADGS) vs kep hang so (3DGS)", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "09_opacity_reset_hist.png")


# ---- Fig 4: quy dao opacity qua nhieu chu ky reset ----------------------
def fig4():
    iters = np.arange(0, 30001, 10)
    resets = [k for k in range(3000, 30001, 3000)]   # opacity_reset_interval=3000
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    labels = ["Gaussian hoi phuc nhanh", "hoi phuc trung binh",
              "hoi phuc cham", "gan nhu khong hoi phuc", "bao hoa cao"]
    rates = [8e-4, 4e-4, 1.5e-4, 4e-5, 1.2e-3]
    a0 = [0.9, 0.6, 0.35, 0.2, 0.95]
    for c, lab, r, a_init in zip(colors, labels, rates, a0):
        a = a_init
        traj = []
        ri = 0
        for k in iters:
            if ri < len(resets) and k >= resets[ri]:
                a = 0.1 * a                     # reset_opacity: alpha <- decay*alpha
                ri += 1
            # hoi phuc do gradient opacity (opacity_lr=0.05): tien ve a_init
            a = a + r * (a_init - a) * 10
            traj.append(a)
        ax.plot(iters, traj, lw=1.7, color=c, label=lab)
    for r in resets:
        ax.axvline(r, color="gray", ls=":", lw=0.9)
    ax.axhline(0.1, color="k", ls="--", lw=1, label="nguong prune 0.1 (train.py:415)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\alpha$ (log)")
    ax.set_title("Quy dao opacity qua cac chu ky reset moi 3000 iter "
                 r"($\alpha \leftarrow 0.1\alpha$)", fontsize=9)
    ax.legend(fontsize=7.2, loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "09_opacity_reset_traj.png")


# ---- Fig 5: bac thang SH degree theo iteration --------------------------
def fig5():
    it = np.arange(0, 60)
    sadgs = np.minimum(it, 3)                       # goi moi iteration (train.py:216-217)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax = axes[0]
    ax.step(it, sadgs, where="post", lw=2.0, color="#1f77b4")
    ax.set_xlabel("iteration")
    ax.set_ylabel("active_sh_degree")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("SADGS: oneupSHdegree() moi iteration\n"
                 r"$d_k=\min(k,\,3)$ $\Rightarrow$ bao hoa tai k=3", fontsize=9)
    ax.grid(alpha=0.3)

    it2 = np.arange(0, 6000)
    ax = axes[1]
    ax.step(it2, np.minimum(it2, 3), where="post", lw=2.0, color="#1f77b4",
            label="SADGS (moi iter)")
    ax.step(it2, np.minimum(it2 // 1000, 3), where="post", lw=2.0, ls="--",
            color="#d62728", label="3DGS goc (moi 1000 iter)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("active_sh_degree")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("So sanh lich nang bac SH (max_sh_degree=3)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "09_sh_degree_schedule.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
