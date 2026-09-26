"""
deep17e_anisotropic_split.py
=============================
Tai lieu bo sung SAU cho muc 17.5 "Anisotropic split -- chia theo 3 truc doc
lap" (DOCS/BOOK/17-sadgs-structure-aware-densification.md). Muc 17.5 hien tai
da co 7 hinh (real_split_0X, real_eta_0X) nhung CHUA co hinh nao di sau vao
3 diem sau (chi neu bang loi van/cong thuc, khong co do thi rieng):

  (1) Rui ro no N khi KHONG co clamp max cho k (code thuc te chi
      torch.clamp(ks, min=1) -- gaussian_model.py:701 -- trong khi comment o
      dong 698 lai ghi "Clamp min=2 ... and max=8").
  (2) Anh huong cua tham so scale_power p trong s_new = s_old / k^p
      (gaussian_model.py:722).
  (3) Vi sao he so la sqrt(12) (chinh xac) chu khong phai mot hang so gan
      dung khac, trong cong thuc offset = (s_old/k)*sqrt(12)*grid
      (gaussian_model.py:775-776, 778).

DU LIEU: dung LAI dung Gaussian THAT tu checkpoint
    DOCS/BOOK/16-loi-giai/assets/point_cloud_iter5000.ply
(cung file, cung cach doc PLY thu cong, cung cong thuc kich hoat nhu
DOCS/BOOK/adc_figures/real_densification_compare.py -- xem file do de doi
chieu). Checkpoint nay chi co N=4 Gaussian that (canh do choi 4 diem,
DOCS/Report/test/00-scene.md); ca 4 deu dang huong hoan toan (rot=(1,0,0,0),
scale dong nhat 3 truc) nhung eta/ks THAT theo tung truc (tinh tu proxy
hinh hoc that omega_axis=1/khoang_cach_that_toi_lang_gieng_gan_nhat, giong
het real_densification_compare.py) van di huong ro ret vi khoang cach that
giua cac diem khong deu theo x/y/z. Gaussian dung lam vi du chinh trong file
nay la G1 cua checkpoint: scale that = exp(_scaling) = 0.53156249 (dong-nhat
3 truc), ks that = [1,4,3] (truc z: k=3) -- lay THANG tu du lieu that, khong
bia dat.

CONG THUC DUNG (trich dan SADGS/scene/gaussian_model.py):
  ks = clamp(eta, min=1).sqrt().ceil(); ks = clamp(ks, min=1)     (dong 699-701)
  new_scaling (linear truoc log) = current_scale / ks**scale_power (dong 722)
  grid toa do tam: coord_j = j - (k-1)/2                           (dong 763-765)
  stds_new = current_scale/ks; separations = stds_new*sqrt(12)     (dong 775-776)
  local_offset = separations * grid                                 (dong 778)

Khong dung torch (khong co san trong moi truong nay) -- toan bo numpy/scipy
thuan, cung phong cach real_densification_compare.py.

Chay:  python DOCS/BOOK/adc_figures/deep17e_anisotropic_split.py
Anh luu vao: DOCS/BOOK/adc_figures/deep17e_0X_*.png
"""

import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PLY = os.path.normpath(os.path.join(HERE, "..", "16-loi-giai", "assets", "point_cloud_iter5000.ply"))
OUT_DIR = HERE

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.0, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "#faf7f2",
    "savefig.dpi": 220,
})

GREEN = "#2c7a3f"
GREEN_LIGHT = "#5fa86e"
RED = "#b5432a"
BLUE = "#4a6fa5"
YELLOW = "#e8c34a"
GRAY = "#7f8a99"


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", os.path.normpath(path))


# ============================================================================
# 1) Doc PLY thu cong -- Y HET real_densification_compare.py (khong doi logic)
# ============================================================================
def read_ply_vertices(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY khong co end_header")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break
        assert header_lines[0] == "ply"
        fmt_line = [l for l in header_lines if l.startswith("format")][0]
        assert "binary_little_endian" in fmt_line, f"Dinh dang khong ho tro: {fmt_line}"

        n_vertex = None
        prop_names = []
        in_vertex_element = False
        for l in header_lines:
            if l.startswith("element vertex"):
                n_vertex = int(l.split()[-1])
                in_vertex_element = True
            elif l.startswith("element") and not l.startswith("element vertex"):
                in_vertex_element = False
            elif l.startswith("property") and in_vertex_element:
                parts = l.split()
                assert parts[1] == "float", f"Property khong phai float: {l}"
                prop_names.append(parts[2])

        n_props = len(prop_names)
        raw = f.read(n_vertex * n_props * 4)
        vals = struct.unpack("<" + "f" * (n_vertex * n_props), raw)
        arr = np.array(vals, dtype=np.float64).reshape(n_vertex, n_props)

    data = {name: arr[:, i] for i, name in enumerate(prop_names)}
    return data, n_vertex, prop_names


data, N_REAL, PROP_NAMES = read_ply_vertices(INPUT_PLY)
print(f"[PLY] {INPUT_PLY}")
print(f"[PLY] N Gaussian THAT = {N_REAL}")

XYZ_RAW = np.stack([data["x"], data["y"], data["z"]], axis=1)
SCALE_LOG_RAW = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)
SCALE = np.exp(SCALE_LOG_RAW)  # scaling_activation, gaussian_model.py:39

GAP_AXIS = np.zeros((N_REAL, 3))
for i in range(N_REAL):
    for ax in range(3):
        d = np.abs(XYZ_RAW[:, ax] - XYZ_RAW[i, ax])
        d[i] = np.inf
        GAP_AXIS[i, ax] = d.min()
OMEGA_AXIS = 1.0 / GAP_AXIS
ETA_AXIS = (SCALE * OMEGA_AXIS) ** 2                      # proxy that, giong real_densification_compare.py


def compute_ks(eta_axis, k_max=None):
    ks = np.clip(np.ceil(np.sqrt(np.clip(eta_axis, 1.0, None))), 1, None)
    if k_max is not None:
        ks = np.clip(ks, None, k_max)
    return ks.astype(int)


KS = compute_ks(ETA_AXIS)   # khong clamp max -- dung code that (dong 699-701)
print("\n[G1 that] scale =", SCALE[0], " eta_axis =", ETA_AXIS[0], " ks (khong clamp max) =", KS[0])

# Gaussian dai dien dung xuyen suot file nay: G1 that cua checkpoint.
G1_SCALE = float(SCALE[0, 0])            # 0.53156249... (dong nhat 3 truc, THAT)
G1_K_Z = int(KS[0, 2])                    # k that tren truc z cua G1 = 3


# ============================================================================
# HINH 1 -- Rui ro no N khi KHONG co clamp max cho k
#   Code that: ks = clamp(eta,min=1).sqrt().ceil(); ks=clamp(ks,min=1)  (699-701)
#   Comment (dong 698, KHONG khop code): "Clamp min=2 ... and max=8"
# ============================================================================
def fig_01_N_explosion():
    eta_sweep = np.linspace(1, 100, 400)
    k_sweep = np.ceil(np.sqrt(np.clip(eta_sweep, 1.0, None)))          # dong 699-701, that
    k_capped = np.clip(k_sweep, None, 8)                                  # neu ap dung dung comment dong 698

    # worst-case doi xung: ca 3 truc cung vi pham bang eta_sweep -> N=k^3
    N_uncapped = k_sweep ** 3
    N_capped = k_capped ** 3

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --- Panel A: duong cong N(eta) log-scale ---
    axA = axes[0]
    axA.plot(eta_sweep, N_uncapped, color=RED, lw=2.4,
             label=r"Code that: $N=k_x k_y k_z$, $k$ khong co tran (clamp(min=1) thoi, dong 701)")
    axA.plot(eta_sweep, N_capped, color=GREEN, lw=2.0, ls="--",
             label=r"Neu ap dung DUNG comment dong 698 ($k \leq 8$)")
    axA.set_yscale("log")
    axA.set_xlabel(r"$\eta$ (gia dinh CA 3 truc cung vi pham bang $\eta$ -- worst case doi xung)")
    axA.set_ylabel(r"$N=k_x k_y k_z$ (so Gaussian con tu 1 Gaussian cha, log scale)")
    axA.set_title("Hinh 1A: N no ra sao khi eta tang, KHONG co tran k")
    axA.axvspan(1, 9, color=GREEN_LIGHT, alpha=0.15)
    axA.axvspan(9, 100, color=RED, alpha=0.08)
    axA.text(4.2, N_uncapped.max() * 0.35, "vung thuc te\n(vi du sach o day)",
              fontsize=8.5, ha="center", color="#3a6b3a")
    axA.text(45, N_uncapped.max() * 0.35, "vung benh ly\n(outlier structure tensor)",
              fontsize=8.5, ha="center", color=RED)

    # danh dau eta=9 -> k=3 -> N=27 (vi du dung trong md)
    axA.scatter([9], [27], color="black", zorder=5, s=30)
    axA.annotate(r"$\eta=(9,9,9)\Rightarrow k=(3,3,3)\Rightarrow N=27$",
                 xy=(9, 27), xytext=(18, 3), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.8))

    # danh dau eta lon nhat THAT do duoc trong checkpoint (G2, truc z): 491.35
    eta_real_max = float(ETA_AXIS.max())
    k_real_max = int(np.ceil(np.sqrt(eta_real_max)))
    N_hypoth_sym = k_real_max ** 3
    axA.scatter([min(eta_real_max, 100)], [min(N_hypoth_sym, N_uncapped.max())],
                color=BLUE, marker="D", zorder=5, s=36)
    axA.annotate(
        rf"$\eta$ that lon nhat do duoc (G2, truc z)={eta_real_max:.1f}"
        f"\n-> k={k_real_max} (that, 1 truc); NEU ca 3 truc\ncung lon nhu vay: "
        f"N gia thuyet = {N_hypoth_sym}",
        xy=(min(eta_real_max, 100), min(N_hypoth_sym, N_uncapped.max())),
        xytext=(30, N_uncapped.max() * 0.02), fontsize=7.6, color=BLUE,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=BLUE))

    axA.legend(fontsize=7.8, loc="upper left")

    # --- Panel B: bar so sanh "code that" vs "neu ap dung comment max=8" tai vai eta cu the ---
    axB = axes[1]
    eta_marks = [9, 25, 50, 100]
    k_marks = [int(np.ceil(np.sqrt(e))) for e in eta_marks]
    k_marks_capped = [min(k, 8) for k in k_marks]
    N_marks = [k ** 3 for k in k_marks]
    N_marks_capped = [k ** 3 for k in k_marks_capped]

    x = np.arange(len(eta_marks))
    w = 0.35
    axB.bar(x - w / 2, N_marks, width=w, color=RED, label="Code that (khong tran)")
    axB.bar(x + w / 2, N_marks_capped, width=w, color=GREEN, label="Neu co clamp max=8 (theo comment)")
    axB.set_yscale("log")
    axB.set_xticks(x)
    axB.set_xticklabels([f"$\\eta$={e}\n(k={k})" for e, k in zip(eta_marks, k_marks)], fontsize=8.5)
    axB.set_ylabel("N (log scale)")
    axB.set_title("Hinh 1B: N thuc te vs. N neu co tran (vi du roi rac)")
    for xi, (nu, nc) in enumerate(zip(N_marks, N_marks_capped)):
        axB.text(xi - w / 2, nu * 1.15, str(nu), ha="center", fontsize=7.8, color=RED)
        axB.text(xi + w / 2, nc * 1.15, str(nc), ha="center", fontsize=7.8, color=GREEN)
    axB.legend(fontsize=8.2)

    fig.suptitle(
        "Rui ro no N: code that chi clamp(ks,min=1) (gaussian_model.py:701) -- "
        "KHONG co max, du comment dong 698 ghi 'max=8'",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "deep17e_01_N_explosion_no_clamp.png")

    print(f"\n[Hinh 1] eta=9 -> N=27; eta=100 -> N={100 and (int(np.ceil(np.sqrt(100)))**3)}; "
          f"eta that max do duoc={eta_real_max:.2f} (G2,truc z) -> k that={k_real_max} tren 1 truc "
          f"(N that cua G2 chi la {int(np.prod(KS[1]))}, vi 2 truc con lai eta nho hon nhieu -- "
          f"day la LY DO DI HUONG: khong phai ca 3 truc cung no).")


# ============================================================================
# HINH 2 -- Anh huong tham so scale_power p: s_new = s_old / k^p (dong 722)
#   Dung G1 THAT cua checkpoint, k=3 (that, truc z cua G1), so sanh p=1.0/1.5/2.0
# ============================================================================
def fig_02_scale_power():
    s_old = G1_SCALE          # 0.53156249..., THAT (G1, dong nhat 3 truc)
    k = G1_K_Z                  # 3, THAT (ks[0,2] cua G1)
    ps = [1.0, 1.5, 2.0]
    colors = [GREEN, "#c78a1e", RED]

    stds_new = s_old / k                                    # dong 775, KHONG phu thuoc p
    separation = stds_new * np.sqrt(12.0)                    # dong 776
    grid = np.arange(k) - (k - 1) / 2.0                         # dong 763-765
    positions = separation * grid                             # dong 778 (1D, vi day la 1 truc)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    s_news = []
    for ax, p, c in zip(axes, ps, colors):
        s_new = s_old / (k ** p)                               # dong 722
        s_news.append(s_new)
        diam = 2 * s_new
        gap = separation - diam
        # cha (net dut, mau xam) lam moc so sanh
        ax.add_patch(Circle((0, 0.72), s_old, facecolor="none", edgecolor=GRAY,
                             lw=1.4, ls="--", zorder=2))
        ax.text(0, 0.72 + s_old + 0.04, "cha (that, $s_{old}$="
                f"{s_old:.3f})", ha="center", fontsize=8, color=GRAY)
        for pos in positions:
            ax.add_patch(Circle((pos, 0), s_new, facecolor=c, edgecolor="black",
                                 lw=0.8, alpha=0.75, zorder=3))
        for pos in positions:
            ax.axvline(pos, color="black", lw=0.4, ls=":", alpha=0.3, zorder=1)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.75, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"$p$={p}\n"
                     r"$s_{new}=s_{old}/k^p$" + f" = {s_new:.4f}\n"
                     f"khoang trong giua 2 con = {gap:.4f}", fontsize=9.5)

    fig.suptitle(
        r"Hinh 2: hieu ung scale_power $p$ tren Gaussian $G_1$ THAT "
        r"($s_{old}$=" + f"{s_old:.5f}, k={k} that tren truc z, "
        "checkpoint point_cloud_iter5000.ply)",
        fontsize=11.5, fontweight="bold", y=1.05)
    fig.text(0.5, -0.04,
              r"Vi tri con (offset = $(s_{old}/k)\sqrt{12}\cdot grid$, dong 775-778) "
              r"KHONG doi theo $p$ -- chi kich thuoc con doi -> $p$ cang lon, con cang nho, "
              "khe ho giua cac con cang rong (gaussian_model.py:722).",
              ha="center", fontsize=9.2)
    fig.tight_layout()
    save(fig, "deep17e_02_scale_power_effect.png")

    print(f"\n[Hinh 2] s_old={s_old:.6f} k={k} separation={separation:.6f}")
    for p, sn in zip(ps, s_news):
        print(f"  p={p}: s_new={sn:.6f}  duong_kinh_con={2*sn:.6f}  "
              f"khe_ho={separation-2*sn:.6f}")


# ============================================================================
# HINH 3 -- Vi sao he so la sqrt(12) (dong 776), khong phai sqrt(3)/sqrt(6)
#   Dung G1 THAT, k=3 (that). So sanh 3 hang so cho separation = std_new * C:
#   C in {sqrt(3), sqrt(6), sqrt(12)}. Kiem chung theo DUNG suy luan trong
#   code/md: coi separation = do rong toan phan cua MOT o luoi (a = separation/2),
#   Var(Uniform(-a,a)) = a^2/3 phai bang std_new^2 -- CHI sqrt(12) thoa dung.
#   Kem theo, de trung thuc, tinh THEM phuong sai thuc nghiem cua chinh k=3
#   diem luoi roi rac (khac voi phuong sai "1 o luoi") de chi ro day la mot
#   xap xi cho k nho, khong phai dang thuc chinh xac cho moi k.
# ============================================================================
def fig_03_sqrt12():
    s_old = G1_SCALE
    k = G1_K_Z                     # 3, that
    std_new = s_old / k              # sigma_new = s_old/k, dong 775
    target_var = std_new ** 2

    constants = [("sqrt(3)", np.sqrt(3.0), "#c78a1e"),
                 ("sqrt(6)", np.sqrt(6.0), BLUE),
                 ("sqrt(12) -- code that", np.sqrt(12.0), GREEN)]

    grid = np.arange(k) - (k - 1) / 2.0     # [-1,0,1] cho k=3

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # --- Panel A: PDF Gaussian cha + vi tri luoi con cho tung hang so ---
    axA = axes[0]
    xs = np.linspace(-3 * s_old, 3 * s_old, 600)
    axA.plot(xs, norm.pdf(xs, 0, s_old), color="black", lw=1.8,
             label=r"PDF Gaussian CHA that, $\mathcal{N}(0,s_{old}^2)$, $s_{old}$="
                   f"{s_old:.3f}")
    axA.fill_between(xs, norm.pdf(xs, 0, s_old), color="black", alpha=0.05)

    y0 = -0.08 * norm.pdf(0, 0, s_old)
    for i, (name, C, color) in enumerate(constants):
        sep = std_new * C
        pos = sep * grid
        yy = y0 * (1 + 1.15 * i)
        axA.scatter(pos, [yy] * k, color=color, s=46, zorder=5,
                    label=f"{name}: separation={sep:.4f}")
        for p in pos:
            axA.plot([p, p], [0, yy], color=color, lw=0.5, ls=":", alpha=0.5)

    axA.axhline(0, color="gray", lw=0.6)
    axA.set_xlabel("truc (don vi that, cung don vi voi checkpoint)")
    axA.set_yticks([])
    axA.set_title(f"Hinh 3A: vi tri {k} diem luoi con (dong 763-778) voi 3 lua chon hang so")
    axA.legend(fontsize=7.6, loc="upper right")

    # --- Panel B: bar so sanh phuong sai (2 kieu) ---
    axB = axes[1]
    names = [c[0] for c in constants]
    percell_vars = []
    discrete_vars = []
    for name, C, color in constants:
        sep = std_new * C
        a = sep / 2.0
        percell_var = a ** 2 / 3.0                        # dung DUNG suy luan Var(Uniform(-a,a))=a^2/3
        pos = sep * grid
        discrete_var = float(np.mean(pos ** 2))              # phuong sai thuc nghiem cua k diem roi rac (mean=0)
        percell_vars.append(percell_var)
        discrete_vars.append(discrete_var)

    x = np.arange(len(names))
    w = 0.35
    axB.bar(x - w / 2, percell_vars, width=w, color=[c[2] for c in constants], alpha=0.9,
            label=r"Var(1 o luoi) = $(sep/2)^2/3$  (dung suy luan cua code)")
    axB.bar(x + w / 2, discrete_vars, width=w, color=[c[2] for c in constants], alpha=0.4,
            hatch="//", label=fr"Var thuc nghiem cua {k} diem luoi (mean$^2$)")
    axB.axhline(target_var, color=RED, lw=1.6, ls="--",
                label=r"muc tieu $\sigma_{new}^2$=" + f"{target_var:.5f}")
    axB.set_xticks(x)
    axB.set_xticklabels(names, fontsize=8.5)
    axB.set_ylabel("phuong sai")
    axB.set_title("Hinh 3B: chi Var(1 o luoi) voi sqrt(12) khop dung muc tieu")
    axB.legend(fontsize=7.4, loc="upper left")

    fig.suptitle(
        r"Hinh 3: vi sao he so la $\sqrt{12}$ (gaussian_model.py:776), tren $G_1$ that "
        r"($s_{old}$=" + f"{s_old:.5f}, k={k} that)",
        fontsize=11.8, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "deep17e_03_sqrt12_grid_variance.png")

    print(f"\n[Hinh 3] s_old={s_old:.6f} k={k} std_new={std_new:.6f} target_var={target_var:.6f}")
    for (name, C, _), pv, dv in zip(constants, percell_vars, discrete_vars):
        sep = std_new * C
        print(f"  {name}: separation={sep:.6f}  Var(1_o_luoi)={pv:.6f} "
              f"(khop_muc_tieu={'CO' if abs(pv-target_var)<1e-9 else 'KHONG'})  "
              f"Var_thuc_nghiem_{k}_diem={dv:.6f} (ti le so voi muc tieu = {dv/target_var:.2f}x)")
    print(f"  --> Ket luan so: sqrt(12) la LUA CHON DUY NHAT lam Var(1 o luoi)=(sep/2)^2/3 "
          f"khop CHINH XAC sigma_new^2, bat ke k. Phuong sai THUC NGHIEM cua chinh {k} diem "
          f"luoi roi rac lai LON HON muc tieu theo he so (k^2-1)={k**2-1} -- tuc sqrt(12) la "
          f"xap xi hop ly cho tung O LUOI rieng le (dung y code), KHONG phai la dang thuc chinh "
          f"xac cho phuong sai TOAN BO tap k diem con.")


if __name__ == "__main__":
    fig_01_N_explosion()
    fig_02_scale_power()
    fig_03_sqrt12()
    print(f"\nHoan tat: 3 hinh da luu vao {OUT_DIR}")
