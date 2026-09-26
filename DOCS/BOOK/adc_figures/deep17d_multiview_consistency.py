"""
deep17d_multiview_consistency.py — Hinh dao sau cho muc 17.4 "Multiview
consistency — high/mid/low ratio" cua sach SADGS.

MUC DICH: muc 17.4 trong DOCS/BOOK/17-sadgs-structure-aware-densification.md
hien KHONG co hinh anh nao (chi cong thuc + bang so sanh voi FastGS). Script
nay tao 3 hinh minh hoa CU THE bang so cho dung 3 khoang trong noi dung:

  (1) Vi sao dung TI SO (>0.8) qua nhieu view chu khong dung 1 view/1 dem tho:
      mo phong 3 "kich ban" quan sat mot Gaussian qua V=30 view (moi 10
      iteration mot lan, giong nhip that cua update_freq_stats_online), phan
      loai HIGH/MID/LOW dung dung cong thuc freq_utils.py:395-409, roi ve
      quy dao high_ratio/low_ratio tich luy theo so quan sat.

  (2) Vi sao "ti so" bat bien qua do phan giai con "Importance tuyet doi"
      (FastGS, xem bang so sanh cuoi muc 17.4) thi khong: mo phong Importance
      (dem pixel loi, ti le dien tich footprint ~ 1/r^2) va high_ratio
      (khong doi vi eta la MOT TI SO hai do dai cung co giang theo r) tren
      3-4 muc downsample r=1,2,4,8 giong quy uoc `-r` cua FastGS (chuong 12).

  (3) Truc eta_max that su trong khoang [0,5] voi 3 vung mau HIGH/MID/LOW
      theo dung tau_high=1.0/tau_low=0.1, chong len HISTOGRAM eta_max TINH
      THAT tu structure tensor cua mot anh that trong repo (tai su dung dung
      cong thuc DOCS/BOOK/adc_figures/real_eta_axis.py, khong bia so).

QUY UOC TRUNG THUC (tiep noi real_eta_axis.py / real_densification_compare.py):
  - Hinh (3): structure tensor, tri rieng, buoc song, eta deu tinh THAT bang
    numpy tren PIXEL THAT cua DOCS/assets/samples_train.png (cung anh, cung
    panel crop nhu real_eta_axis.py) — khong random cho ban than eta.
  - Hinh (1) va (2) MO PHONG (simulate) chuoi quan sat da-view vi repo khong
    co log eta that qua nhieu camera da luu san (eta accumulator KHONG duoc
    ghi ra PLY/checkpoint — xem docstring real_densification_compare.py).
    Moi con so mo phong deu duoc GHI RO trong code va trong nhan truc/tieu
    de hinh la "mo phong" (simulated), khong bao gio tuyen bo la du lieu that.
    random duoc dung DUNG voi tinh chat cho phep trong de bai: mo phong "Gaussian
    nay duoc V camera quan sat, moi lan cho ra mot eta_max" — tuc lay mau
    "chuoi quan sat qua nhieu goc nhin", khong phai bia dat ban than dai
    luong eta_max tu con so 0.

CONG THUC DUNG (CHEP LAI TU MA NGUON THAT SADGS/, trich dan file:line):
  - Phan loai high/mid/low theo eta_max — SADGS/utils/freq_utils.py:395-409:
        TAU_HIGH = 1.0 ; TAU_LOW = 0.1
        eta_max = eta_3ch.max(axis=-1)
        is_high = eta_max > TAU_HIGH
        is_low  = eta_max <= TAU_LOW
        is_mid  = ~is_high & ~is_low
        eta_high_count += is_high ; eta_mid_count += is_mid ; eta_low_count += is_low
  - Ti so nhat quan + nguong split/prune — SADGS/train.py:335-353 va
    SADGS/arguments/__init__.py:148-149 (split_ratio_threshold =
    prune_ratio_threshold = 0.8):
        high_ratio = eta_high_count / accum_view_count
        low_ratio  = eta_low_count  / accum_view_count
        split_mask = (high_ratio > 0.8) & (grad_norm >= 1e-5)
        prune_mask = (low_ratio  > 0.8) & (accum_view_count > 0)
  - eta (mode "wavelength") — SADGS/utils/freq_utils.py:325-348 (tai lap
    numpy dung nhu real_eta_axis.py::eig_and_wavelength / eta_sadgs):
        trace = Sxx+Syy ; det = Sxx*Syy - Sxy^2
        delta = sqrt(clamp((trace/2)^2 - det, min=0))
        lambda1 = trace/2 + delta                      # freq_utils.py:329
        wavelength_min = 1/(sqrt(lambda1) + 1e-5)       # freq_utils.py:334
        eta_j = axis_length_j / wavelength_min          # freq_utils.py:348

Moi truong: numpy + scipy.ndimage + PIL + matplotlib (Agg). KHONG torch.
Chay: python DOCS/BOOK/adc_figures/deep17d_multiview_consistency.py
"""

import os
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 0. Style (dong bo voi real_eta_axis.py / real_densification_compare.py)
# ----------------------------------------------------------------------------
COL_BG      = "#f4efe6"
COL_PANEL   = "#f8f4ec"
COL_DARK    = "#212a35"
COL_RED     = "#c0392b"
COL_GREEN   = "#3f7d5c"
COL_BLUE    = "#2f5f8a"
COL_GRAY    = "#8a8378"
COL_ORANGE  = "#d97b29"
COL_YELLOW  = "#f2c14e"
COL_YELLOW_E= "#b9821a"

plt.rcParams.update({
    "figure.facecolor": COL_BG,
    "axes.facecolor": COL_PANEL,
    "savefig.facecolor": COL_BG,
    "font.size": 10,
    "axes.edgecolor": COL_DARK,
})

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC_IMAGE = os.path.join(REPO, "DOCS", "assets", "samples_train.png")
PANEL_BOX = (468, 394, 908, 638)  # panel "gt 00097" — cung crop voi real_eta_axis.py

TAU_HIGH = 1.0   # freq_utils.py:395
TAU_LOW = 0.1    # freq_utils.py:396
TAU_SPLIT = 0.8  # arguments/__init__.py:148
TAU_PRUNE = 0.8  # arguments/__init__.py:149


def savefig(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


def classify(eta_max):
    """freq_utils.py:395-404 — tra ve mang nhan 'high'/'mid'/'low'."""
    is_high = eta_max > TAU_HIGH
    is_low = eta_max <= TAU_LOW
    labels = np.where(is_high, "high", np.where(is_low, "low", "mid"))
    return labels


# ============================================================================
# FIGURE 1 — 3 kich ban mo phong chuoi quan sat V view, quy dao high/low ratio
# ============================================================================
def simulate_case(rng, V, case):
    """Mo phong V quan sat eta_max (moi 10 iteration, giong nhip that cua
    update_freq_stats_online) cho MOT Gaussian, theo 3 kich ban vat ly hop ly:

    (a) consistent_high — Gaussian duoi-giai ("under-resolved") o hau het
        goc nhin: chi tiet anh nho hon truc Gaussian chieu len man hinh o
        gan nhu moi view -> eta_max mo phong lognormal quanh ~2.2 (>tau_high).
    (b) consistent_low — Gaussian da du min o hau het goc nhin (vung nen
        phang da khop tot) -> eta_max mo phong quanh ~0.03 (<tau_low).
    (c) edge_on_outlier — Gaussian mong, chi "chieu canh" (edge-on, giong
        mot dia phang nhin nghieng) o mot thieu so goc nhin hep -> da so
        view cho eta_max o vung MID/thap, nhung ~20-25% view (goc nhin hep,
        gan tiep tuyen) cho eta_max cao dot bien do truc chieu bi keo dai.
        Day CHINH LA truong hop nguong 0.8 duoc thiet ke de chan: vai view
        "outlier" khong du de vuot 80%.
    """
    if case == "consistent_high":
        eta = rng.lognormal(mean=np.log(2.2), sigma=0.35, size=V)
    elif case == "consistent_low":
        eta = np.clip(rng.lognormal(mean=np.log(0.03), sigma=0.5, size=V), 1e-4, None)
    elif case == "edge_on_outlier":
        # nen: phan lon view thay Gaussian o vung MID/thap (0.15-0.6)
        eta = rng.lognormal(mean=np.log(0.35), sigma=0.35, size=V)
        # ~22% view "edge-on": goc chieu hep lam truc keo dai dot bien
        is_edge_on = rng.random(V) < 0.22
        eta[is_edge_on] = rng.lognormal(mean=np.log(3.0), sigma=0.3, size=is_edge_on.sum())
    else:
        raise ValueError(case)
    return eta


def fig01_ratio_trajectories():
    rng = np.random.default_rng(17)
    V = 30  # so quan sat moi-10-iteration mo phong (trong khoang 20-40 de bai yeu cau)

    cases = {
        "consistent_high": dict(color=COL_RED,
            title="(a) Duoi-giai nhat quan\n(under-resolved o moi view)"),
        "consistent_low": dict(color=COL_BLUE,
            title="(b) Da du min nhat quan\n(fine o moi view)"),
        "edge_on_outlier": dict(color=COL_ORANGE,
            title="(c) Chi bi flag \"canh nghieng\"\no thieu so view (outlier)"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)

    summary_lines = []
    for ax, (case, spec) in zip(axes, cases.items()):
        eta_seq = simulate_case(rng, V, case)
        labels = classify(eta_seq)

        n = np.arange(1, V + 1)
        high_count = np.cumsum(labels == "high")
        low_count = np.cumsum(labels == "low")
        high_ratio = high_count / n
        low_ratio = low_count / n

        ax.plot(n, high_ratio, "-o", color=COL_RED, ms=3.5, lw=1.8, label="high\\_ratio")
        ax.plot(n, low_ratio, "-o", color=COL_BLUE, ms=3.5, lw=1.8, label="low\\_ratio")
        ax.axhline(TAU_SPLIT, color=COL_DARK, ls="--", lw=1.4,
                   label="$\\tau_{split}=\\tau_{prune}=0.8$" if ax is axes[0] else None)
        ax.set_title(spec["title"], fontsize=10.3, fontweight="bold", color=COL_DARK)
        ax.set_xlabel("So quan sat tich luy $M$ (moi 10 iteration)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(1, V)
        ax.set_facecolor(COL_PANEL)
        if ax is axes[0]:
            ax.set_ylabel("Ti so (eta\\_high\\_count hoac eta\\_low\\_count) / M")
            ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.62), fontsize=8.3, framealpha=0.9)

        final_hr, final_lr = high_ratio[-1], low_ratio[-1]
        decision = ("SPLIT" if final_hr > TAU_SPLIT else
                    ("PRUNE" if final_lr > TAU_PRUNE else "khong hanh dong"))
        ax.text(0.97, 0.50, f"M={V}: high\\_ratio={final_hr:.2f}\nlow\\_ratio={final_lr:.2f}\n"
                             f"$\\Rightarrow$ {decision}",
                transform=ax.transAxes, ha="right", va="center", fontsize=8.7, color=COL_DARK,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=COL_DARK, alpha=0.85, pad=0.35))
        summary_lines.append((case, final_hr, final_lr, decision))

    fig.suptitle("Hinh 1 (MO PHONG) — Quy dao high/low\\_ratio tich luy qua $M$ quan sat da-view\n"
                 "(phan loai theo freq\\_utils.py:395-409, ti so theo train.py:335-353)",
                 fontsize=12.5, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0.02, 1, 0.86])
    savefig(fig, "deep17d_01_ratio_trajectories.png")

    print("Hinh 1 - ket qua cuoi cung:")
    for case, hr, lr, dec in summary_lines:
        print(f"  {case}: high_ratio={hr:.3f} low_ratio={lr:.3f} -> {dec}")


# ============================================================================
# FIGURE 2 — Importance tuyet doi (FastGS) ~ 1/r^2 vs high_ratio bat bien
# ============================================================================
def fig02_resolution_invariance():
    rng = np.random.default_rng(23)
    r_values = np.array([1, 2, 4, 8])  # quy uoc -r cua FastGS (chuong 12)

    # --- "Importance" kieu FastGS: dem pixel-loi tuyet doi trong footprint.
    # Dien tich footprint (pixel^2) co theo 1/r^2 khi anh bi downsample he so r
    # (moi canh pixel co 1/r lan) -> so pixel loi dem duoc trong footprint do
    # cung giam xap xi 1/r^2 (gia dinh mat do loi/pixel khong doi).
    importance_r1_mean = 24.0  # gia tri neo tai r=1 (vai chuc pixel loi/footprint,
                                # cung bac voi nguong Importance>5 cua FastGS)
    n_gauss = 400
    base_importance = rng.gamma(shape=3.0, scale=importance_r1_mean / 3.0, size=n_gauss)

    # --- eta = axis_length_px / wavelength_min_px: CA HAI do dai deu do bang
    # pixel CUA ANH DA DOWNSAMPLE, nen ca hai cung co theo r -> ti so eta BAT
    # BIEN theo r (chi con nhieu do lay mau nho). Ta mo phong V=25 view cho
    # moi Gaussian, dung MOT phan bo eta_max co dinh (khong phu thuoc r,
    # +nhieu do lay mau nho de mo phong thuc te), roi tinh high_ratio.
    V = 25
    eta_max_samples = rng.lognormal(mean=np.log(1.6), sigma=0.5, size=(n_gauss, V))

    importance_curve = []
    high_ratio_curve = []
    high_ratio_std = []
    for r in r_values:
        # Importance giam theo 1/r^2 (dien tich footprint), + nhieu do lay mau nho
        imp_r = base_importance / (r ** 2) * rng.normal(1.0, 0.04, size=n_gauss)
        importance_curve.append(imp_r.mean())

        # eta bat bien theo r: chi them nhieu do lay mau RAT NHO (+-2%) de mo
        # phong sai so roi rac khi downsample (khong doi ban chat ti so)
        eta_r = eta_max_samples * rng.normal(1.0, 0.02, size=(n_gauss, V))
        labels_r = classify(eta_r)
        hr = (labels_r == "high").sum(axis=1) / V
        high_ratio_curve.append(hr.mean())
        high_ratio_std.append(hr.std())

    importance_curve = np.array(importance_curve)
    high_ratio_curve = np.array(high_ratio_curve)
    high_ratio_std = np.array(high_ratio_std)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ax0 = axes[0]
    ax0.plot(r_values, importance_curve, "-o", color=COL_ORANGE, lw=2.2, ms=7,
              label="Importance trung binh (dem pixel, mo phong)")
    ax0.axhline(5, color=COL_RED, ls="--", lw=1.5, label="nguong Importance $>5$ (co dinh)")
    # duong tham chieu ly thuyet ~ 1/r^2
    r_fine = np.linspace(1, 8, 100)
    ax0.plot(r_fine, importance_curve[0] / r_fine**2, ":", color=COL_GRAY, lw=1.6,
              label="tham chieu $\\propto 1/r^2$")
    ax0.set_xlabel("He so downsample $r$ (\"-r 1,2,4,8\")")
    ax0.set_ylabel("Importance (dem pixel loi tuyet doi, mo phong)")
    ax0.set_title("FastGS: nguong TUYET DOI, khong co giang theo $r$",
                  fontsize=10.5, fontweight="bold", color=COL_DARK)
    ax0.legend(fontsize=8.2, framealpha=0.9)
    ax0.set_facecolor(COL_PANEL)
    for r, v in zip(r_values, importance_curve):
        ax0.annotate(f"{v:.1f}", (r, v), textcoords="offset points", xytext=(4, 6), fontsize=8.5)

    ax1 = axes[1]
    ax1.errorbar(r_values, high_ratio_curve, yerr=high_ratio_std, fmt="-o",
                 color=COL_BLUE, lw=2.2, ms=7, capsize=4,
                 label="high\\_ratio trung binh $\\pm$ std (mo phong)")
    ax1.axhline(TAU_SPLIT, color=COL_RED, ls="--", lw=1.5, label="$\\tau_{split}=0.8$ (co dinh)")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("He so downsample $r$ (\"-r 1,2,4,8\")")
    ax1.set_ylabel("high\\_ratio (ti so, khong thu nguyen)")
    ax1.set_title("SAD-GS: nguong TI SO, bat bien theo $r$",
                  fontsize=10.5, fontweight="bold", color=COL_DARK)
    ax1.legend(fontsize=8.2, framealpha=0.9)
    ax1.set_facecolor(COL_PANEL)
    for r, v in zip(r_values, high_ratio_curve):
        ax1.annotate(f"{v:.2f}", (r, v), textcoords="offset points", xytext=(4, 6), fontsize=8.5)

    fig.suptitle("Hinh 2 (MO PHONG) — Importance tuyet doi ($\\propto 1/r^2$) vs high\\_ratio bat bien qua do phan giai\n"
                 "(so sanh dong cuoi bang 17.4: nguong pixel tuyet doi vs nguong ti so)",
                 fontsize=12.3, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    savefig(fig, "deep17d_02_resolution_invariance.png")

    print("Hinh 2 - Importance(r):", dict(zip(r_values.tolist(), np.round(importance_curve, 2).tolist())))
    print("Hinh 2 - high_ratio(r):", dict(zip(r_values.tolist(), np.round(high_ratio_curve, 3).tolist())))


# ============================================================================
# FIGURE 3 — truc eta_max That + 3 vung mau + histogram eta_max That
# ============================================================================
def gaussian_blur_np(img, sigma):
    if sigma <= 0:
        return img.copy()
    if img.ndim == 2:
        return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma=sigma, mode="reflect")
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)  # loss_utils.py:187
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)  # loss_utils.py:188


def get_structure_tensor_np(image_rgb, sigma=1.0, rho=1.0):
    """Tai lap SADGS/utils/loss_utils.py:170-230, dong bo voi real_eta_axis.py."""
    img_smooth = gaussian_blur_np(image_rgb, sigma)
    C = img_smooth.shape[-1]
    Ix = np.stack([ndimage.correlate(img_smooth[..., c], SOBEL_X, mode="constant", cval=0.0)
                   for c in range(C)], axis=-1)
    Iy = np.stack([ndimage.correlate(img_smooth[..., c], SOBEL_Y, mode="constant", cval=0.0)
                   for c in range(C)], axis=-1)
    Ixx = (Ix * Ix).sum(axis=-1)
    Ixy = (Ix * Iy).sum(axis=-1)
    Iyy = (Iy * Iy).sum(axis=-1)
    Sxx = gaussian_blur_np(Ixx, rho)
    Sxy = gaussian_blur_np(Ixy, rho)
    Syy = gaussian_blur_np(Iyy, rho)
    magnitude = Sxx + Syy
    max_val = magnitude.max() + 1e-6
    return Sxx / max_val, Sxy / max_val, Syy / max_val


def wavelength_field(Sxx, Sxy, Syy):
    """freq_utils.py:325-334."""
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((trace / 2) ** 2 - det, a_min=0.0, a_max=None))
    lambda1 = trace / 2 + delta
    return 1.0 / (np.sqrt(lambda1) + 1e-5)


# Kich thuoc Gaussian neo (GIA DINH HOP LY, giong real_eta_axis.py — khong co
# scene 3DGS/camera that de do truc chieu that; CHI kich thuoc neo la gia
# dinh, ban than truong wavelength_min la TINH THAT tu anh that).
SIGMA_MAJOR_PX = 18.0
SIGMA_MINOR_PX = 6.0


def fig03_threshold_bands():
    im = Image.open(SRC_IMAGE).convert("RGB")
    crop = im.crop(PANEL_BOX)
    img = np.asarray(crop).astype(np.float64) / 255.0

    Sxx, Sxy, Syy = get_structure_tensor_np(img, sigma=1.0, rho=1.0)
    wl_min = wavelength_field(Sxx, Sxy, Syy)

    eta_major_field = SIGMA_MAJOR_PX / wl_min   # freq_utils.py:348, truc chinh neo
    eta_minor_field = SIGMA_MINOR_PX / wl_min   # freq_utils.py:348, truc phu neo
    eta_max_field = np.maximum(eta_major_field, eta_minor_field)  # freq_utils.py:399 (max qua cac truc)

    eta_flat = eta_max_field.ravel()
    eta_flat_clipped = np.clip(eta_flat, 0, 5.0)

    frac_high = (eta_flat > TAU_HIGH).mean()
    frac_low = (eta_flat <= TAU_LOW).mean()
    frac_mid = 1.0 - frac_high - frac_low

    fig, ax = plt.subplots(figsize=(11, 5.4))

    # Vung mau HIGH/MID/LOW tren truc eta_max
    ax.axvspan(0, TAU_LOW, color=COL_BLUE, alpha=0.18)
    ax.axvspan(TAU_LOW, TAU_HIGH, color=COL_GRAY, alpha=0.18)
    ax.axvspan(TAU_HIGH, 5.0, color=COL_RED, alpha=0.18)

    ax.axvline(TAU_LOW, color=COL_BLUE, ls="--", lw=1.6)
    ax.axvline(TAU_HIGH, color=COL_RED, ls="--", lw=1.6)

    # Histogram THAT eta_max tren toan bo anh that (khong phai patch don le)
    ax.hist(eta_flat_clipped, bins=140, range=(0, 5.0), color=COL_DARK, alpha=0.75,
            density=True, label="histogram $\\eta_{max}$ THAT (moi pixel cua anh that)")

    ax.set_xlim(0, 5.0)
    ax.set_xlabel("$\\eta_{max} = \\max(\\eta_{major}, \\eta_{minor})$ (THAT, freq\\_utils.py:399)")
    ax.set_ylabel("Mat do (histogram, chuan hoa)")
    ax.set_facecolor(COL_PANEL)

    ymax = ax.get_ylim()[1]
    # LOW va MID la 2 dai rat hep (0-0.1 va 0.1-1.0 tren truc 0-5), nen nhan
    # duoc dat BEN NGOAI/PHIA TREN dai bang mui ten, dan tach nhau theo chieu
    # doc de khong de len nhau; chi HIGH (0.8 pham vi truc) du rong de dat
    # nhan ngay giua dai.
    ax.annotate(f"LOW ($\\eta\\leq{TAU_LOW}$)\nprune candidate\n({frac_low*100:.1f}% pixel that)",
                xy=(TAU_LOW / 2, ymax * 0.55), xytext=(0.75, ymax * 0.97),
                ha="left", va="top", fontsize=8.4, color=COL_BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL_BLUE, lw=1.3))
    ax.annotate(f"MID (khong hanh dong)\n({frac_mid*100:.1f}% pixel that)",
                xy=((TAU_LOW + TAU_HIGH) / 2, ymax * 0.30), xytext=(0.75, ymax * 0.75),
                ha="left", va="top", fontsize=8.4, color=COL_GRAY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COL_GRAY, lw=1.3))
    ax.text(3.0, ymax * 0.93, f"HIGH\n$\\eta>{TAU_HIGH}$\nsplit candidate\n"
            f"({frac_high*100:.1f}% pixel that)",
            ha="center", va="top", fontsize=9.2, color=COL_RED, fontweight="bold")

    ax.legend(loc="upper right", fontsize=8.6, framealpha=0.9)

    fig.suptitle("Hinh 3 (eta THAT tren anh that) — Truc phan loai HIGH/MID/LOW ($\\tau_{low}=0.1$, $\\tau_{high}=1.0$)\n"
                 "chong len histogram $\\eta_{max}$ tinh THAT tu structure tensor (samples\\_train.png, panel gt 00097)",
                 fontsize=12.3, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    savefig(fig, "deep17d_03_threshold_bands.png")

    print(f"Hinh 3 - eta_max THAT: frac_low={frac_low:.4f} frac_mid={frac_mid:.4f} frac_high={frac_high:.4f}")
    print(f"  (moi pixel cua panel {PANEL_BOX}, shape {img.shape[:2]}, "
          f"sigma neo=({SIGMA_MAJOR_PX},{SIGMA_MINOR_PX})px)")


# ============================================================================
# main
# ============================================================================
def main():
    fig01_ratio_trajectories()
    fig02_resolution_invariance()
    fig03_threshold_bands()
    print("Hoan tat — 3 hinh deep17d da luu trong", HERE)


if __name__ == "__main__":
    main()
