"""
real_eta_axis.py — Hinh THAT cho chuong 18.2 (chieu truc & ti so eta) va 18.4
(anisotropic split) cua sach SADGS.

QUY UOC TRUNG THUC (giong DOCS/Report/test/scripts/ch07_test.py):
Moi con so eta, buoc song lambda_min, huong truc chinh (eigenvector), va he so
tach ks trong cac hinh nay deu duoc TINH THAT bang numpy tren PIXEL THAT cua
mot anh that trong repo — khong co so lieu random/bia dat cho ban than eta.
Phan duy nhat mang tinh "gia dinh hop ly" (khong do duoc vi khong co scene
3DGS da train + camera that) la KICH THUOC PIXEL BAN DAU cua Gaussian neo vao
tung patch (sigma_major_px, sigma_minor_px) — day chi la mot con so neo de ve
hinh hoc truoc/sau split, duoc ghi ro trong code, KHONG anh huong toi gia tri
eta/huong truc (nhung dai luong nay chi phu thuoc vao anh that).

ANH NGUON THAT:
    DOCS/assets/samples_train.png, panel "gt 00097" (dong 2, cot 2), cat tai
    pixel box (468, 394, 908, 638) cua anh goc 1374x650 -> anh con 440x244.
    Panel nay co ca canh cheo tuong phan manh (soc xanh/cam tren dau may),
    vung troi phang, va vung tan cay/doi co ket cau hon hop — du 3 loai vung
    can cho bai tap (anisotropic / isotropic / ambiguous).

CONG THUC DUOC CHEP LAI TU MA NGUON THAT (SADGS/), TRICH DAN file:line:

1) Structure tensor Di Zenzo — SADGS/utils/loss_utils.py:170-230
   (ham get_structure_tensor_torch). Cac buoc (tai lap bang numpy/scipy):
     - lam mo Gauss truoc (sigma), dao ham Sobel 3x3 tung kenh RGB (loss_utils.py:187-196),
     - Ixx=Ix^2, Ixy=Ix*Iy, Iyy=Iy^2, CONG DON tren 3 kenh mau (loss_utils.py:198-208),
     - lam mo cua so tich hop (rho) -> Sxx,Sxy,Syy (loss_utils.py:210-217),
     - chuan hoa theo cuc dai toan anh cua (Sxx+Syy) (loss_utils.py:220-227).

2) Tri rieng lon nhat cua structure tensor va buoc song cuc bo —
   SADGS/utils/freq_utils.py:325-334 (trong update_freq_stats_online, mode
   "wavelength"):
     trace = Sxx+Syy; det = Sxx*Syy - Sxy^2
     delta = sqrt(clamp((trace/2)^2 - det, min=0))
     lambda1 = trace/2 + delta                      # freq_utils.py:329
     wavelength_min = 1 / (sqrt(lambda1) + 1e-5)     # freq_utils.py:334
   (lambda2 = trace/2 - delta la tri rieng con lai, dung de tinh ti so
   coherence/anisotropy lambda1/lambda2 — dai luong chuan cua structure
   tensor, KHONG phai eta cua SADGS, dung o day de chon HUONG truc chinh
   qua eigenvector — cong thuc goc atan2 chuan cho ma tran doi xung 2x2.)

3) Do dai truc chieu tren man hinh — SADGS/utils/freq_utils.py:342
     axis_lengths = sqrt(u_j^2 + v_j^2 + 1e-8)   # j = 1..3 truc chinh ellipsoid
   O day (khong co scene 3D that) ta neo truc j truc tiep theo eigenvector
   THAT cua structure tensor tai patch (huong do THAT), voi do dai la
   sigma_px NEO (gia dinh hop ly, ghi ro trong code).

4) Dinh nghia eta (mode mac dinh "wavelength") — SADGS/utils/freq_utils.py:348
     eta_j = axis_lengths_j / wavelength_min

5) So lat cat moi truc tu eta — SADGS/scene/gaussian_model.py:699-701
     ks = ceil( sqrt( clamp(eta, min=1.0) ) ); ks = clamp(ks, min=1)

6) Vi tri/ban kinh con sau split (luoi deu, xoay theo R, bao toan tam) —
   SADGS/scene/gaussian_model.py:722 (sigma' = sigma / k^scale_power, p=1),
   :763-765 (grid_x = ix - (kx-1)/2, ...), :775-776 (separations = (sigma/k)*sqrt(12)),
   :778 (local_offsets = separations * grid), :782-788 (xoay R(q), cong vao tam cha).

7) Gian Gaussian qua nho (expand_undersized_gs) —
   SADGS/scene/gaussian_model.py:833-860:
     eta < tau_expand (=1.0) va eta>0  =>  delta_log_scale = -0.5*log(eta)
     sigma_new = sigma_old * exp(delta_log_scale) = sigma_old / sqrt(eta)

Moi truong: numpy + scipy.ndimage (khong co torch). matplotlib de ve hinh.
Chay: python real_eta_axis.py  (luu PNG vao cung thu muc script).
"""

import os
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from matplotlib import gridspec

# ----------------------------------------------------------------------------
# 0. Style / palette (tham khao SADGS/static/teaser.png: nen kem phang, ellipse
#    vang, tieu de dam, mui ten ro net)
# ----------------------------------------------------------------------------
COL_BG      = "#f4efe6"
COL_PANEL   = "#f8f4ec"
COL_YELLOW  = "#f2c14e"
COL_YELLOW_E= "#b9821a"
COL_DARK    = "#212a35"
COL_RED     = "#c0392b"
COL_GREEN   = "#3f7d5c"
COL_BLUE    = "#2f5f8a"
COL_GRAY    = "#8a8378"
COL_ORANGE  = "#d97b29"

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
# Panel "gt 00097" trong luoi 3x2 cua samples_train.png (xem doc string)
PANEL_BOX = (468, 394, 908, 638)


# ----------------------------------------------------------------------------
# 1. Tai anh that + cat panel that
# ----------------------------------------------------------------------------
def load_real_patch():
    im = Image.open(SRC_IMAGE).convert("RGB")
    crop = im.crop(PANEL_BOX)
    arr = np.asarray(crop).astype(np.float64) / 255.0  # [H,W,3] in [0,1]
    return arr


# ----------------------------------------------------------------------------
# 2. Structure tensor Di Zenzo — tai lap loss_utils.py:170-230 bang numpy
# ----------------------------------------------------------------------------
def gaussian_blur_np(img, sigma):
    """Tuong duong torchvision.transforms.functional.gaussian_blur ap dung
    cho tung kenh (hoac anh 1 kenh), bien gioi 'reflect' (fast_gaussian_blur
    goi ham nay trong loss_utils.py, khong doi cong thuc Gauss)."""
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
    """Tai lap CHINH XAC cong thuc SADGS/utils/loss_utils.py:170-230
    (get_structure_tensor_torch), dau vao anh RGB [H,W,3] trong [0,1].
    Tra ve (Sxx, Sxy, Syy) da chuan hoa theo cuc dai toan anh (loss_utils.py:220-227).
    """
    # 1. Lam mo truoc (pre-smoothing), moi kenh doc lap — loss_utils.py:177-183
    img_smooth = gaussian_blur_np(image_rgb, sigma)

    # 2. Dao ham Sobel tung kenh (groups=C trong conv2d that) — loss_utils.py:185-196
    C = img_smooth.shape[-1]
    Ix = np.stack([ndimage.correlate(img_smooth[..., c], SOBEL_X, mode="constant", cval=0.0)
                   for c in range(C)], axis=-1)
    Iy = np.stack([ndimage.correlate(img_smooth[..., c], SOBEL_Y, mode="constant", cval=0.0)
                   for c in range(C)], axis=-1)

    # 3-4. San pham + cong don tren kenh mau (Di Zenzo) — loss_utils.py:198-208
    Ixx = (Ix * Ix).sum(axis=-1)
    Ixy = (Ix * Iy).sum(axis=-1)
    Iyy = (Iy * Iy).sum(axis=-1)

    # 5. Lam mo cua so tich hop (window integration) — loss_utils.py:210-217
    Sxx = gaussian_blur_np(Ixx, rho)
    Sxy = gaussian_blur_np(Ixy, rho)
    Syy = gaussian_blur_np(Iyy, rho)

    # 6. Chuan hoa theo cuc dai toan anh cua (Sxx+Syy) — loss_utils.py:220-227
    magnitude = Sxx + Syy
    max_val = magnitude.max() + 1e-6
    Sxx = Sxx / max_val
    Sxy = Sxy / max_val
    Syy = Syy / max_val
    return Sxx, Sxy, Syy


# ----------------------------------------------------------------------------
# 3. Tri rieng / eigenvector / buoc song — freq_utils.py:325-334 + chuan hoa
#    huong (atan2) cho ma tran doi xung 2x2
# ----------------------------------------------------------------------------
def eig_and_wavelength(Sxx, Sxy, Syy):
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((trace / 2) ** 2 - det, a_min=0.0, a_max=None))
    lambda1 = trace / 2 + delta          # freq_utils.py:329 (tri rieng lon nhat)
    lambda2 = trace / 2 - delta          # tri rieng con lai
    wavelength_min = 1.0 / (np.sqrt(lambda1) + 1e-5)   # freq_utils.py:334
    # Huong eigenvector ung voi lambda1 cho ma tran doi xung 2x2 [[Sxx,Sxy],[Sxy,Syy]]
    theta = 0.5 * np.arctan2(2 * Sxy, (Sxx - Syy) + 1e-12)
    coherence = lambda1 / (lambda2 + 1e-8)  # ti so anisotropy chuan (khong phai eta SADGS)
    return lambda1, lambda2, wavelength_min, theta, coherence


def eta_sadgs(axis_len_px, wavelength_min_px):
    """eta_j = ||a_j|| / wavelength_min — freq_utils.py:348 (mode 'wavelength')."""
    return axis_len_px / wavelength_min_px


def ks_from_eta(eta):
    """ks = ceil(sqrt(clamp(eta, min=1))) — gaussian_model.py:699-701."""
    return max(1, int(np.ceil(np.sqrt(max(eta, 1.0)))))


def delta_log_scale_expand(eta):
    """delta_log_scale = -0.5*log(eta) — gaussian_model.py:855 (chi khi 0<eta<tau_expand)."""
    eta = max(eta, 1e-6)
    return -0.5 * np.log(eta)


# ----------------------------------------------------------------------------
# 4. Chon 3 patch THAT tren anh THAT (toa do trong he quy chieu crop 440x244)
# ----------------------------------------------------------------------------
PATCHES = {
    "A_edge_stripes":  dict(x0=20,  y0=150, size=64, label="A — Soc cheo (canh manh)",
                             color=COL_RED),
    "B_flat_sky":      dict(x0=220, y0=20,  size=64, label="B — Troi (phang)",
                             color=COL_BLUE),
    "C_foliage_ambig": dict(x0=260, y0=130, size=64, label="C — Tan cay/doi (hon hop)",
                             color=COL_GREEN),
}

# Kich thuoc Gaussian neo (GIA DINH HOP LY — khong do duoc vi khong co scene
# 3DGS/camera that; chi dung de ve hinh hoc, KHONG anh huong den eta/huong).
SIGMA_MAJOR_PX = 18.0
SIGMA_MINOR_PX = 6.0


def analyze_patch(Sxx, Sxy, Syy, spec):
    x0, y0, s = spec["x0"], spec["y0"], spec["size"]
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    sxx_m, sxy_m, syy_m = Sxx[sl].mean(), Sxy[sl].mean(), Syy[sl].mean()
    lam1, lam2, wl_min, theta, coh = eig_and_wavelength(sxx_m, sxy_m, syy_m)
    eta_major = eta_sadgs(SIGMA_MAJOR_PX, wl_min)
    eta_minor = eta_sadgs(SIGMA_MINOR_PX, wl_min)
    ks_major = ks_from_eta(eta_major)
    ks_minor = ks_from_eta(eta_minor)
    return dict(Sxx=sxx_m, Sxy=sxy_m, Syy=syy_m, lambda1=lam1, lambda2=lam2,
                wavelength_min=wl_min, theta=theta, coherence=coh,
                eta_major=eta_major, eta_minor=eta_minor,
                ks_major=ks_major, ks_minor=ks_minor)


# ----------------------------------------------------------------------------
# 5. Helper ve
# ----------------------------------------------------------------------------
def style_axis(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_DARK)
        spine.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=COL_DARK, pad=8)


def draw_yellow_ellipse(ax, cx, cy, w, h, angle_deg, lw=2.0, alpha=1.0):
    e = Ellipse((cx, cy), width=w, height=h, angle=angle_deg,
                facecolor=COL_YELLOW, edgecolor=COL_YELLOW_E, lw=lw, alpha=alpha, zorder=5)
    ax.add_patch(e)
    return e


def savefig(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


# ============================================================================
# FIGURE 1 — anh that + 3 patch that + eigenvector THAT overlay
# ============================================================================
def fig01_patches(img, Sxx, Sxy, Syy, results):
    fig = plt.figure(figsize=(13, 7.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.18)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(img)
    style_axis(ax0, "Anh THAT: DOCS/assets/samples_train.png, panel \"gt 00097\" "
                     f"(box {PANEL_BOX}) — 3 patch that duoc phan tich")
    for key, spec in PATCHES.items():
        r = Rectangle((spec["x0"], spec["y0"]), spec["size"], spec["size"],
                       fill=False, edgecolor=spec["color"], lw=2.5, zorder=6)
        ax0.add_patch(r)
        ax0.text(spec["x0"], spec["y0"] - 4, spec["label"], color=spec["color"],
                  fontsize=9, fontweight="bold", va="bottom")

    for i, (key, spec) in enumerate(PATCHES.items()):
        ax = fig.add_subplot(gs[1, i])
        x0, y0, s = spec["x0"], spec["y0"], spec["size"]
        ax.imshow(img[y0:y0 + s, x0:x0 + s])
        res = results[key]
        # quiver luoi thua eigenvector THAT (chieu dai theo sqrt(coherence), chuan hoa)
        step = 8
        ys, xs = np.mgrid[0:s:step, 0:s:step]
        sub_theta = np.full_like(xs, res["theta"], dtype=float)
        mag = 3.0
        u = mag * np.cos(sub_theta)
        v = mag * np.sin(sub_theta)
        ax.quiver(xs, ys, u, v, color=spec["color"], scale=40, width=0.008,
                   headwidth=3, pivot="mid", zorder=6)
        style_axis(ax, spec["label"])
        txt = (f"$\\eta_{{maj}}$={res['eta_major']:.2f}  $\\eta_{{min}}$={res['eta_minor']:.2f}\n"
               f"$\\lambda_{{min}}$={res['wavelength_min']:.1f}px  coh={res['coherence']:.1f}\n"
               f"$\\theta$={np.degrees(res['theta']):.0f}$^\\circ$")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.3,
                color="white", va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor=COL_DARK, alpha=0.75, pad=0.35))

    fig.suptitle("Hinh 1 — Patch THAT + huong truc chinh (eigenvector) cua structure tensor THAT",
                  fontsize=13, fontweight="bold", color=COL_DARK, y=1.0)
    savefig(fig, "real_eta_01_patches.png")


# ============================================================================
# FIGURE 2 — heatmap coherence + huong tren toan anh (tinh THAT tung pixel)
# ============================================================================
def fig02_heatmap(img, Sxx, Sxy, Syy):
    lam1, lam2, wl_min, theta, coh = eig_and_wavelength(Sxx, Sxy, Syy)
    H, W = Sxx.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].imshow(img)
    style_axis(axes[0], "Anh that (patch 440x244)")
    for spec in PATCHES.values():
        r = Rectangle((spec["x0"], spec["y0"]), spec["size"], spec["size"],
                       fill=False, edgecolor=spec["color"], lw=2.0)
        axes[0].add_patch(r)

    log_coh = np.log10(np.clip(coh, 1e-2, 1e3))
    im1 = axes[1].imshow(log_coh, cmap="inferno")
    style_axis(axes[1], "$\\log_{10}$(coherence) = $\\log_{10}(\\lambda_1/\\lambda_2)$ — THAT tung pixel")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for spec in PATCHES.values():
        r = Rectangle((spec["x0"], spec["y0"]), spec["size"], spec["size"],
                       fill=False, edgecolor="white", lw=1.6, ls="--")
        axes[1].add_patch(r)

    axes[2].imshow(img)
    style_axis(axes[2], "Huong truc chinh THAT (quiver, mau = coherence)")
    step = 10
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    th = theta[0:H:step, 0:W:step]
    cvals = np.log10(np.clip(coh[0:H:step, 0:W:step], 1e-2, 1e3))
    u = np.cos(th); v = np.sin(th)
    q = axes[2].quiver(xs, ys, u, v, cvals, cmap="inferno", scale=45, width=0.004,
                        headwidth=2.5, pivot="mid")
    fig.colorbar(q, ax=axes[2], fraction=0.046, pad=0.04, label="log10(coherence)")

    fig.suptitle("Hinh 2 — Ban do coherence & huong truc chinh tren toan anh (tinh THAT tu Sxx,Sxy,Syy)",
                  fontsize=13, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "real_eta_02_heatmap_eta.png")


# ============================================================================
# FIGURE 3 — lambda1 / wavelength_min heatmap + bar chart 3 patch (THAT)
# ============================================================================
def fig03_wavelength(img, Sxx, Sxy, Syy, results):
    lam1, lam2, wl_min, theta, coh = eig_and_wavelength(Sxx, Sxy, Syy)
    fig = plt.figure(figsize=(14.5, 4.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.3)

    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(lam1, cmap="magma")
    style_axis(ax0, "$\\lambda_1$ (nang luong gradient toi da) — freq_utils.py:329")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(np.log10(np.clip(wl_min, 1, 2e5)), cmap="viridis")
    style_axis(ax1, "$\\log_{10}(\\lambda_{min})$ — buoc song cuc bo, freq_utils.py:334")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for spec in PATCHES.values():
        r = Rectangle((spec["x0"], spec["y0"]), spec["size"], spec["size"],
                       fill=False, edgecolor="white", lw=1.6, ls="--")
        ax1.add_patch(r)

    ax2 = fig.add_subplot(gs[2])
    names = [spec["label"].split(" — ")[0] for spec in PATCHES.values()]
    wls = [results[k]["wavelength_min"] for k in PATCHES]
    colors = [spec["color"] for spec in PATCHES.values()]
    bars = ax2.bar(names, wls, color=colors, edgecolor=COL_DARK)
    for b, v in zip(bars, wls):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}px", ha="center", va="bottom",
                  fontsize=9, fontweight="bold")
    ax2.set_ylabel("$\\lambda_{min}$ (pixel, THAT)")
    ax2.set_title("Buoc song THAT tai 3 patch", fontsize=11, fontweight="bold", color=COL_DARK)
    ax2.set_facecolor(COL_PANEL)

    fig.suptitle("Hinh 3 — Tri rieng lon nhat & buoc song cuc bo THAT (mau so cua eta)",
                  fontsize=13, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "real_eta_03_wavelength.png")


# ============================================================================
# FIGURE 4 — chieu truc: Sxx,Sxy,Syy that + ellipse cau truc + truc chieu
# ============================================================================
def fig04_axis_projection(img, results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (key, spec) in zip(axes, PATCHES.items()):
        x0, y0, s = spec["x0"], spec["y0"], spec["size"]
        ax.imshow(img[y0:y0 + s, x0:x0 + s], extent=(-s/2, s/2, s/2, -s/2))
        res = results[key]
        cx, cy = 0, 0
        angle_deg = np.degrees(res["theta"])
        # ellipse cau truc that (ti le theo lambda, chi de minh hoa huong/coherence THAT)
        w = 2 * SIGMA_MAJOR_PX
        h = 2 * SIGMA_MINOR_PX
        draw_yellow_ellipse(ax, cx, cy, w, h, angle_deg, lw=2.2, alpha=0.55)
        # hai mui ten truc chieu THAT (huong eigenvector that, do dai = sigma neo)
        dx1, dy1 = np.cos(res["theta"]) * SIGMA_MAJOR_PX, np.sin(res["theta"]) * SIGMA_MAJOR_PX
        dx2, dy2 = -np.sin(res["theta"]) * SIGMA_MINOR_PX, np.cos(res["theta"]) * SIGMA_MINOR_PX
        for dx, dy, c, lbl in [(dx1, dy1, COL_RED, f"truc chinh: {2*SIGMA_MAJOR_PX/res['wavelength_min']*0.5:.2f}"),
                                (dx2, dy2, COL_BLUE, "truc phu")]:
            ax.add_patch(FancyArrowPatch((0, 0), (dx, dy), color=c, lw=2.2,
                                          arrowstyle="-|>", mutation_scale=14, zorder=7))
            ax.add_patch(FancyArrowPatch((0, 0), (-dx, -dy), color=c, lw=2.2,
                                          arrowstyle="-|>", mutation_scale=14, zorder=7))
        style_axis(ax, spec["label"])
        st_txt = (f"$S_{{xx}}$={res['Sxx']:.3f}  $S_{{xy}}$={res['Sxy']:.3f}  $S_{{yy}}$={res['Syy']:.3f}\n"
                  f"$\\theta$(eigvec)={angle_deg:.0f}$^\\circ$   coherence={res['coherence']:.1f}")
        ax.text(0.5, -0.16, st_txt, transform=ax.transAxes, fontsize=8.6, ha="center",
                color=COL_DARK)
    fig.suptitle("Hinh 4 — Chieu truc: $(S_{xx},S_{xy},S_{yy})$ THAT va huong truc chinh eigenvector THAT\n"
                 "(kich thuoc ellipse la sigma neo gia dinh — xem docstring)",
                 fontsize=12.5, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    savefig(fig, "real_eta_04_axis_projection.png")


# ============================================================================
# FIGURE 5 — eta THAT -> ks (so lat cat) THAT theo cong thuc gaussian_model.py:699
# ============================================================================
def fig05_ks_split_count(results):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    names = [spec["label"].split(" — ")[0] for spec in PATCHES.values()]
    eta_maj = [results[k]["eta_major"] for k in PATCHES]
    eta_min = [results[k]["eta_minor"] for k in PATCHES]
    x = np.arange(len(names))
    w = 0.32
    b1 = ax.bar(x - w/2, eta_maj, width=w, color=COL_ORANGE, edgecolor=COL_DARK, label="$\\eta_{major}$ (truc chinh)")
    b2 = ax.bar(x + w/2, eta_min, width=w, color=COL_BLUE, edgecolor=COL_DARK, label="$\\eta_{minor}$ (truc phu)")
    ax.axhline(1.0, color=COL_RED, ls="--", lw=1.6, label="$\\tau_{high}=1.0$ (nguong split)")
    for b, k in zip(b1, PATCHES):
        v = results[k]["eta_major"]; ks = results[k]["ks_major"]
        ax.text(b.get_x()+b.get_width()/2, v, f"$\\eta$={v:.2f}\n$k$={ks}", ha="center", va="bottom", fontsize=8.5)
    for b, k in zip(b2, PATCHES):
        v = results[k]["eta_minor"]; ks = results[k]["ks_minor"]
        ax.text(b.get_x()+b.get_width()/2, v, f"$\\eta$={v:.2f}\n$k$={ks}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("$\\eta_j$ = axis$_j$ / $\\lambda_{min}$ (THAT, freq_utils.py:348)")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_facecolor(COL_PANEL)
    fig.suptitle("Hinh 5 — $\\eta$ THAT tren 3 patch $\\to$ $k=\\lceil\\sqrt{\\max(\\eta,1)}\\rceil$ THAT (gaussian_model.py:699-701)",
                 fontsize=12.5, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, "real_eta_05_ks_split_count.png")


# ============================================================================
# FIGURE 6 — hinh hoc split di huong THAT: truoc/sau, isotropic (3DGS) vs
# anisotropic (SADGS), dung eta+huong THAT cua patch A (soc cheo)
# ============================================================================
def split_children_grid(sigma_xy, kx, ky, theta, scale_power=1.0):
    """Tai lap dung cong thuc gaussian_model.py:722,763-765,775-778,782-788
    cho truong hop 2D (kx,ky trong he local, xoay boi theta)."""
    ix, iy = np.meshgrid(np.arange(kx), np.arange(ky), indexing="ij")
    gx = ix.astype(float) - (kx - 1) / 2.0     # :763
    gy = iy.astype(float) - (ky - 1) / 2.0     # :764
    sigma_new = sigma_xy / np.array([kx, ky])   # :775 (khong co scale_power)
    sep = sigma_new * np.sqrt(12.0)             # :776
    local_x = sep[0] * gx
    local_y = sep[1] * gy
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    pts = np.stack([local_x.ravel(), local_y.ravel()], axis=0)   # [2, N]
    world = R @ pts                                              # :786
    sigma_child = sigma_xy / (np.array([kx, ky]) ** scale_power)  # :722
    return world[0], world[1], sigma_child


def fig06_split_geometry(img, results):
    key = "A_edge_stripes"
    spec = PATCHES[key]
    res = results[key]
    x0, y0, s = spec["x0"], spec["y0"], spec["size"]
    theta = res["theta"]
    sigma_parent = np.array([SIGMA_MAJOR_PX, SIGMA_MINOR_PX])

    kx_real, ky_real = res["ks_major"], res["ks_minor"]  # THAT, tu eta THAT
    kx_iso, ky_iso = 2, 2  # 3DGS goc: N=2 co dinh (gaussian_model.py:866, mo phong 2D bang k=2,2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax in axes:
        ax.imshow(img[y0:y0 + s, x0:x0 + s], extent=(-s/2, s/2, s/2, -s/2))

    # Panel 1: truoc split (cha)
    draw_yellow_ellipse(axes[0], 0, 0, 2*sigma_parent[0], 2*sigma_parent[1],
                         np.degrees(theta), lw=2.4, alpha=0.7)
    style_axis(axes[0], f"Truoc split (cha)\n$\\sigma$=({sigma_parent[0]:.0f},{sigma_parent[1]:.0f})px, "
                          f"$\\theta$={np.degrees(theta):.0f}$^\\circ$ (THAT)")

    # Panel 2: 3DGS goc — isotropic N=4 (k=2,2), vi tri lay mau ngau nhien minh hoa bang luoi deu tuong duong
    cx, cy, sc = split_children_grid(sigma_parent, kx_iso, ky_iso, theta, scale_power=1.0)
    for xi, yi in zip(cx, cy):
        draw_yellow_ellipse(axes[1], xi, yi, 2*sc[0], 2*sc[1], np.degrees(theta), lw=1.4, alpha=0.85)
    style_axis(axes[1], f"3DGS goc (N=2, mo phong 2D k=2,2 co dinh)\n"
                          f"$\\sigma'$=({sc[0]:.1f},{sc[1]:.1f})px — CHIA DEU ca 2 truc")

    # Panel 3: SADGS that — anisotropic kx,ky tu eta THAT
    cx2, cy2, sc2 = split_children_grid(sigma_parent, kx_real, ky_real, theta, scale_power=1.0)
    for xi, yi in zip(cx2, cy2):
        draw_yellow_ellipse(axes[2], xi, yi, 2*sc2[0], 2*sc2[1], np.degrees(theta), lw=1.2, alpha=0.85)
    style_axis(axes[2], f"SADGS (dung eta THAT): $k_x$={kx_real}, $k_y$={ky_real}, N={kx_real*ky_real}\n"
                          f"$\\sigma'$=({sc2[0]:.1f},{sc2[1]:.1f})px — tach DUNG theo truc vi pham")

    fig.suptitle("Hinh 6 — Hinh hoc split di huong dung eta & huong truc THAT tai patch A (soc cheo)\n"
                 "(gaussian_model.py:699-701, :722, :763-788)",
                 fontsize=12.5, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    savefig(fig, "real_eta_06_split_geometry.png")


# ============================================================================
# FIGURE 7 — expand_undersized_gs: patch B (troi, phang) -> eta THAT rat nho
# ============================================================================
def fig07_expand_undersized(img, results):
    key = "B_flat_sky"
    spec = PATCHES[key]
    res = results[key]
    x0, y0, s = spec["x0"], spec["y0"], spec["size"]
    theta = res["theta"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    for ax in axes:
        ax.imshow(img[y0:y0 + s, x0:x0 + s], extent=(-s/2, s/2, s/2, -s/2))

    sigma_before = np.array([SIGMA_MAJOR_PX, SIGMA_MINOR_PX])
    draw_yellow_ellipse(axes[0], 0, 0, 2*sigma_before[0], 2*sigma_before[1],
                         np.degrees(theta), lw=2.2, alpha=0.6)
    style_axis(axes[0], f"Truoc expand\n$\\sigma$=({sigma_before[0]:.0f},{sigma_before[1]:.0f})px\n"
                          f"$\\eta_{{maj}}$(THAT)={res['eta_major']:.4f}  $\\eta_{{min}}$(THAT)={res['eta_minor']:.4f}")

    dls_maj = delta_log_scale_expand(res["eta_major"])   # gaussian_model.py:855
    dls_min = delta_log_scale_expand(res["eta_minor"])
    sigma_after = sigma_before * np.array([np.exp(dls_maj), np.exp(dls_min)])
    draw_yellow_ellipse(axes[1], 0, 0, 2*sigma_after[0], 2*sigma_after[1],
                         np.degrees(theta), lw=2.2, alpha=0.6)
    style_axis(axes[1], f"Sau expand ($\\eta\\to$1, gaussian_model.py:855)\n"
                          f"$\\sigma'$=({sigma_after[0]:.0f},{sigma_after[1]:.0f})px\n"
                          f"$\\Delta\\log\\sigma$=({dls_maj:.2f},{dls_min:.2f})")

    fig.suptitle("Hinh 7 — Gian Gaussian qua nho tren vung phang (patch B, troi) dung eta THAT\n"
                 "$\\Delta\\log\\sigma=-0.5\\log(\\eta)$ — gaussian_model.py:833-860",
                 fontsize=12.5, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    savefig(fig, "real_eta_07_expand_undersized.png")


# ============================================================================
# main
# ============================================================================
def main():
    img = load_real_patch()
    print("Anh that duoc dung:", SRC_IMAGE, "panel box", PANEL_BOX, "shape", img.shape)

    Sxx, Sxy, Syy = get_structure_tensor_np(img, sigma=1.0, rho=1.0)
    print("Structure tensor THAT tinh xong. max(Sxx+Syy) truoc chuan hoa da ap dung trong ham.")

    results = {k: analyze_patch(Sxx, Sxy, Syy, spec) for k, spec in PATCHES.items()}
    for k, r in results.items():
        print(f"[{k}] lambda1={r['lambda1']:.4f} wl_min={r['wavelength_min']:.2f}px "
              f"theta={np.degrees(r['theta']):.1f}deg coherence={r['coherence']:.2f} "
              f"eta_maj={r['eta_major']:.3f}(k={r['ks_major']}) eta_min={r['eta_minor']:.3f}(k={r['ks_minor']})")

    fig01_patches(img, Sxx, Sxy, Syy, results)
    fig02_heatmap(img, Sxx, Sxy, Syy)
    fig03_wavelength(img, Sxx, Sxy, Syy, results)
    fig04_axis_projection(img, results)
    fig05_ks_split_count(results)
    fig06_split_geometry(img, results)
    fig07_expand_undersized(img, results)
    print("Hoan tat — 7 hinh da luu trong", HERE)


if __name__ == "__main__":
    main()
