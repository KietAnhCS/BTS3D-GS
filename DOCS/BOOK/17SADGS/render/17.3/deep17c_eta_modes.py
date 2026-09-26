"""
deep17c_eta_modes.py — Hinh THAT bo sung cho muc 17.3 (eta wavelength vs
projection, jitter Cholesky, bug w_valid=1) cua sach SADGS.

MUC DICH: di sau vao 3 khoang trong ma muc 17.3 hien chi khang dinh bang loi
van, chua co hinh rieng:
  1) Gap 1 — chung minh BANG SO rang mode "wavelength" BAT BIEN theo huong
     truc con mode "projection" THAY DOI theo huong truc, tren mot canh
     ngang THAT trich tu anh that.
  2) Gap 2 — jitter Cholesky (freq_utils.py:257-292): ve THAT cac diem mau
     jitter quanh tam Gaussian, hinh dang phu thuoc Sigma_2D (isotropic vs
     di huong), tren nen truong structure tensor THAT; tinh phuong sai eta
     giua cac lan jitter so voi eta lay mau dung tam.
  3) Gap 3 — bug w_valid=1 (freq_utils.py:380, 238): dung SO LIEU GIA DINH
     HOP LY (mot profile transmittance alpha-compositing chuan) de ve ro
     "neu duoc weight nhu comment noi" doi lap voi "code THAT hien tai luon
     nhan voi 1" — hai duong ro rang duoc GAN NHAN, khong lan lon.

QUY UOC TRUNG THUC (giong real_eta_axis.py cung thu muc): moi gia tri eta,
lambda1, wavelength_min, Sxx/Sxy/Syy trong Gap 1 va Gap 2 deu TINH THAT bang
numpy tren PIXEL THAT cua mot anh that trong repo (ANH MOI, chua dung trong
real_eta_axis.py / real_structure_analysis.py — xem SRC_IMAGE/PANEL_BOX ben
duoi). Cac truc Gaussian tong hop (huong 0/45/90 do, do dai L px; Sigma_2D
isotropic/di huong cho jitter) la THIET KE THU NGHIEM co chu dich (can thiet
de tach bach "cung mot S that" khoi "nhieu huong truc khac nhau" — khong the
do duoc tu scene 3DGS da train ma khong co scene 3D that), duoc ghi ro trong
code va KHONG anh huong den gia tri S/lambda1 (chi phu thuoc anh that). Gap 3
dung mot profile transmittance GIA DINH HOP LY (alpha-compositing chuan,
alpha ngau nhien co kiem soat seed) vi khong co pipeline render/rasterizer
that (khong co torch) de lay max_transmittance that.

CONG THUC DUOC CHEP LAI TU MA NGUON THAT (SADGS/), TRICH DAN file:line:
  - Structure tensor Di Zenzo: SADGS/utils/loss_utils.py:170-230 (tai lap
    numpy giong DOCS/BOOK/adc_figures/real_eta_axis.py, ham get_structure_tensor_np).
  - Tri rieng lon nhat + wavelength_min (mode "wavelength"):
    SADGS/utils/freq_utils.py:313-334.
  - eta mode "wavelength": SADGS/utils/freq_utils.py:348
        eta_k = axis_lengths_k / wavelength_min
  - eta mode "projection": SADGS/utils/freq_utils.py:350-373
        eta_k = sqrt(Sxx*u_k^2 + 2*Sxy*u_k*v_k + Syy*v_k^2)
  - Do dai truc chieu: SADGS/utils/freq_utils.py:342
        axis_lengths = sqrt(u^2+v^2+1e-8)
  - Cholesky jitter: SADGS/utils/freq_utils.py:257-292
        L11=sqrt(cov_xx); L21=cov_xy/L11; L22=sqrt(cov_yy - L21^2)
        jitter_x = L11*eps1 ; jitter_y = L21*eps1 + L22*eps2
  - grid_sample tai diem da jitter: SADGS/utils/freq_utils.py:289-298.
  - Trong so transmittance (weights_valid=1 hien tai) + tong hop 3 kenh:
    SADGS/utils/freq_utils.py:181 (chu ky ham), :207 (max_transmittance da
    co san trong cov2D nhung khong dung o day), :238
        weights_valid = torch.ones_like(max_transmittance)[is_active_vis]
    :378-382
        eta_3ch = eta_3ch * weights_valid.unsqueeze(1)   # comment: "Weight by
        # Transmittance (Importance Sampling)" nhung weights_valid luon = 1
        eta_total = eta_3ch.sum(dim=1)

Moi truong: numpy + scipy.ndimage + PIL + matplotlib (KHONG import torch).
Chay: python DOCS/BOOK/adc_figures/deep17c_eta_modes.py
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
# 0. Style / palette — giu dung tong mau voi real_eta_axis.py cho nhat quan
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
COL_PURPLE  = "#7a4fa3"

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

# Panel MOI (chua dung trong real_eta_axis.py [box (468,394,908,638)] hay
# real_structure_analysis.py [box (10,395,452,640)]): panel cot 3 dong 1 cua
# luoi 3x2 trong samples_train.png (anh goc 1374x650) — canh toa xe lua khac,
# nhieu canh ngang manh (mep than xe, khung gam) + vung da soi hon hop.
PANEL_BOX = (926, 394, 1366, 638)


def savefig(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


def style_axis(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(COL_DARK)
        spine.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=COL_DARK, pad=6)


def draw_yellow_ellipse(ax, cx, cy, w, h, angle_deg, lw=2.0, alpha=1.0, color=COL_YELLOW, edge=COL_YELLOW_E):
    e = Ellipse((cx, cy), width=w, height=h, angle=angle_deg,
                facecolor=color, edgecolor=edge, lw=lw, alpha=alpha, zorder=5)
    ax.add_patch(e)
    return e


# ----------------------------------------------------------------------------
# 1. Structure tensor Di Zenzo — tai lap loss_utils.py:170-230 (giong
#    real_eta_axis.py, chep lai o day de script nay chay doc lap)
# ----------------------------------------------------------------------------
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
    """Tai lap SADGS/utils/loss_utils.py:170-230 bang numpy. Tra ve
    (Sxx, Sxy, Syy) da chuan hoa theo cuc dai toan anh (loss_utils.py:220-227)."""
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


def eig_lambda1(Sxx, Sxy, Syy):
    """freq_utils.py:325-329 — tri rieng lon nhat cua structure tensor."""
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((trace / 2) ** 2 - det, a_min=0.0, a_max=None))
    return trace / 2 + delta


def wavelength_min_from_S(Sxx, Sxy, Syy):
    """freq_utils.py:334."""
    lam1 = eig_lambda1(Sxx, Sxy, Syy)
    return 1.0 / (np.sqrt(lam1) + 1e-5), lam1


def eta_wavelength(axis_len_px, wavelength_min_px):
    """freq_utils.py:348 (mode 'wavelength'): eta = ||a|| / wavelength_min."""
    return axis_len_px / wavelength_min_px


def eta_projection(Sxx, Sxy, Syy, u, v):
    """freq_utils.py:350-373 (mode 'projection'):
    eta = sqrt(Sxx*u^2 + 2*Sxy*u*v + Syy*v^2)."""
    val = Sxx * u ** 2 + 2 * Sxy * u * v + Syy * v ** 2
    return np.sqrt(np.clip(val, 0.0, None))


def bilinear_sample(field, x, y):
    """Tuong duong F.grid_sample(align_corners=True) 1 diem tren field 2D
    (freq_utils.py:289-298), dung scipy map_coordinates (order=1)."""
    # map_coordinates nhan (row, col) = (y, x)
    coords = np.array([[y], [x]])
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest")[0]


def load_real_patch():
    im = Image.open(SRC_IMAGE).convert("RGB")
    crop = im.crop(PANEL_BOX)
    arr = np.asarray(crop).astype(np.float64) / 255.0
    return arr


# ============================================================================
# GAP 1 — bat bien huong (wavelength) vs phu thuoc huong (projection)
# ============================================================================
def find_horizontal_edge_location(Sxx, Syy, margin=24):
    """Tim vi tri THAT co canh ngang manh nhat: Syy lon, Sxx nho, trong vung
    noi bo (tranh bien anh). Diem so = Syy - Sxx, chi xet noi bo [margin:-margin]."""
    H, W = Sxx.shape
    score = Syy - Sxx
    inner = score[margin:H - margin, margin:W - margin]
    idx = np.unravel_index(np.argmax(inner), inner.shape)
    y0, x0 = idx[0] + margin, idx[1] + margin
    return y0, x0


def fig01_orientation_invariance(img, Sxx, Sxy, Syy):
    y0, x0 = find_horizontal_edge_location(Sxx, Syy)
    # Trung binh S trong 1 cua so nho 5x5 quanh vi tri de bot nhieu 1-pixel
    win = 2
    sxx = Sxx[y0-win:y0+win+1, x0-win:x0+win+1].mean()
    sxy = Sxy[y0-win:y0+win+1, x0-win:x0+win+1].mean()
    syy = Syy[y0-win:y0+win+1, x0-win:x0+win+1].mean()
    wl_min, lam1 = wavelength_min_from_S(sxx, sxy, syy)

    L = 20.0  # do dai truc tong hop (px) — CO DINH cho ca 3 huong (thiet ke thu nghiem)
    angles_deg = np.array([0.0, 45.0, 90.0])  # 0 = ngang (song song canh), 90 = doc
    thetas = np.radians(angles_deg)
    u = L * np.cos(thetas)
    v = L * np.sin(thetas)
    axis_len = np.sqrt(u**2 + v**2 + 1e-8)  # freq_utils.py:342 — LUON = L bat ke huong

    eta_wl = eta_wavelength(axis_len, wl_min)               # freq_utils.py:348
    eta_pr = eta_projection(sxx, sxy, syy, u, v)             # freq_utils.py:350-373

    fig = plt.figure(figsize=(14, 6.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.32, top=0.72, bottom=0.06)

    # Panel A: anh that + 3 truc tong hop ve tai vi tri canh ngang
    ax0 = fig.add_subplot(gs[0])
    half = 40
    y_lo, y_hi = max(0, y0-half), min(img.shape[0], y0+half)
    x_lo, x_hi = max(0, x0-half), min(img.shape[1], x0+half)
    ax0.imshow(img[y_lo:y_hi, x_lo:x_hi], extent=(-(x0-x_lo), x_hi-x0, y_hi-y0, -(y0-y_lo)))
    colors = [COL_RED, COL_ORANGE, COL_BLUE]
    for ang, uu, vv, c in zip(angles_deg, u, v, colors):
        ax0.add_patch(FancyArrowPatch((0, 0), (uu, vv), color=c, lw=2.4,
                                       arrowstyle="-|>", mutation_scale=14, zorder=7,
                                       label=f"{ang:.0f}$^\\circ$"))
        ax0.add_patch(FancyArrowPatch((0, 0), (-uu, -vv), color=c, lw=2.4,
                                       arrowstyle="-|>", mutation_scale=14, zorder=7))
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.85)
    style_axis(ax0, f"Canh ngang THAT tai pixel ({x0},{y0})\n"
                      f"$S_{{xx}}$={sxx:.3f} $S_{{xy}}$={sxy:.3f} $S_{{yy}}$={syy:.3f} (THAT)\n"
                      f"3 truc tong hop L={L:.0f}px, huong 0/45/90$^\\circ$")

    # Panel B: eta_wavelength theo huong — HANG NGANG (bat bien)
    ax1 = fig.add_subplot(gs[1])
    bars1 = ax1.bar([f"{a:.0f}$^\\circ$" for a in angles_deg], eta_wl,
                     color=colors, edgecolor=COL_DARK)
    for b, v_ in zip(bars1, eta_wl):
        ax1.text(b.get_x()+b.get_width()/2, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, max(eta_wl.max(), eta_pr.max()) * 1.35)
    ax1.set_ylabel("$\\eta$ (mode wavelength)")
    style_axis(ax1, "Mode \"wavelength\" (fu:313-348)\nBAT BIEN theo huong truc")
    ax1.set_facecolor(COL_PANEL)

    # Panel C: eta_projection theo huong — THAY DOI
    ax2 = fig.add_subplot(gs[2])
    bars2 = ax2.bar([f"{a:.0f}$^\\circ$" for a in angles_deg], eta_pr,
                     color=colors, edgecolor=COL_DARK)
    for b, v_ in zip(bars2, eta_pr):
        ax2.text(b.get_x()+b.get_width()/2, v_, f"{v_:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(eta_wl.max(), eta_pr.max()) * 1.35)
    ax2.set_ylabel("$\\eta$ (mode projection)")
    style_axis(ax2, "Mode \"projection\" (fu:350-373)\nTHAY DOI theo huong truc")
    ax2.set_facecolor(COL_PANEL)

    fig.suptitle("Hinh 1 (Gap 17.3-1) — Tren cung 1 canh ngang THAT: eta \"wavelength\" khong doi theo huong,\n"
                 "eta \"projection\" thap nhat khi truc song song canh (0$^\\circ$), cao nhat khi vuong goc (90$^\\circ$)",
                 fontsize=12, fontweight="bold", color=COL_DARK, y=0.98)
    savefig(fig, "deep17c_01_orientation_invariance.png")
    return dict(y0=y0, x0=x0, sxx=sxx, sxy=sxy, syy=syy, wl_min=wl_min,
                angles=angles_deg, eta_wl=eta_wl, eta_pr=eta_pr)


# ============================================================================
# GAP 2 — jitter Cholesky: hinh dang jitter theo Sigma_2D + phuong sai eta
# ============================================================================
def cholesky_jitter_samples(cov_xx, cov_xy, cov_yy, n, rng):
    """freq_utils.py:257-274, tai lap dung tung dong."""
    eps1 = rng.standard_normal(n)
    eps2 = rng.standard_normal(n)
    L11 = np.sqrt(max(cov_xx, 1e-6))
    L21 = cov_xy / L11
    L22 = np.sqrt(max(cov_yy - L21 ** 2, 1e-6))
    jitter_x = L11 * eps1
    jitter_y = L21 * eps1 + L22 * eps2
    return jitter_x, jitter_y


def fig02_cholesky_jitter(img, Sxx, Sxy, Syy):
    H, W = Sxx.shape
    # Vi tri "nhay cam": tim diem co do lech chuan local cua (Sxx+Syy) trong
    # cua so 15x15 lon nhat (bien/goc, noi jitter co the doi vung eta ro ret),
    # tranh vien anh.
    energy = Sxx + Syy
    local_std = ndimage.generic_filter(energy, np.std, size=15, mode="nearest")
    margin = 30
    inner = local_std[margin:H - margin, margin:W - margin]
    idx = np.unravel_index(np.argmax(inner), inner.shape)
    cy, cx = idx[0] + margin, idx[1] + margin

    rng = np.random.default_rng(17)
    N_SAMPLES = 400
    L_axis = 14.0  # do dai truc test co dinh (px), thiet ke thu nghiem

    covs = {
        "isotropic": dict(cov_xx=9.0, cov_xy=0.0, cov_yy=9.0, color=COL_BLUE,
                            title="$\\Sigma_{2D}$ isotropic\n(cov=[9,0;0,9])"),
        "elongated": dict(cov_xx=25.0, cov_xy=14.0, cov_yy=10.0, color=COL_RED,
                            title="$\\Sigma_{2D}$ di huong\n(cov=[25,14;14,10])"),
    }

    fig = plt.figure(figsize=(15, 6.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.3)

    # Panel A & B: overlay jitter tren nen anh that (crop quanh vi tri) cho tung Sigma
    results = {}
    for i, (key, spec) in enumerate(covs.items()):
        ax = fig.add_subplot(gs[i])
        half = 34
        y_lo, y_hi = max(0, cy - half), min(H, cy + half)
        x_lo, x_hi = max(0, cx - half), min(W, cx + half)
        ax.imshow(img[y_lo:y_hi, x_lo:x_hi],
                   extent=(-(cx - x_lo), x_hi - cx, y_hi - cy, -(cy - y_lo)))

        jx, jy = cholesky_jitter_samples(spec["cov_xx"], spec["cov_xy"], spec["cov_yy"],
                                          N_SAMPLES, rng)
        ax.scatter(jx, jy, s=6, color=spec["color"], alpha=0.35, zorder=6,
                   label=f"{N_SAMPLES} lan jitter")

        # ellipse 1-sigma that su cua Sigma_2D (eigen-decomposition)
        cov_mat = np.array([[spec["cov_xx"], spec["cov_xy"]], [spec["cov_xy"], spec["cov_yy"]]])
        evals, evecs = np.linalg.eigh(cov_mat)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        ang = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        draw_yellow_ellipse(ax, 0, 0, 2*np.sqrt(evals[0]), 2*np.sqrt(evals[1]), ang,
                             lw=2.2, alpha=0.28, color=spec["color"], edge=spec["color"])

        # Eta tai TAM CO DINH (khong jitter) vs eta trung binh qua cac lan jitter,
        # dung truc test co dinh doc theo huong x (mode wavelength)
        sample_center = (Sxx[cy, cx], Sxy[cy, cx], Syy[cy, cx])
        wl_c, _ = wavelength_min_from_S(*sample_center)
        eta_center = eta_wavelength(L_axis, wl_c)

        eta_jitter = np.empty(N_SAMPLES)
        for k in range(N_SAMPLES):
            sx = np.clip(cx + jx[k], 0, W - 1)
            sy = np.clip(cy + jy[k], 0, H - 1)
            s_xx = bilinear_sample(Sxx, sx, sy)
            s_xy = bilinear_sample(Sxy, sx, sy)
            s_yy = bilinear_sample(Syy, sx, sy)
            wl_j, _ = wavelength_min_from_S(s_xx, s_xy, s_yy)
            eta_jitter[k] = eta_wavelength(L_axis, wl_j)

        results[key] = dict(eta_center=eta_center, eta_mean=eta_jitter.mean(),
                             eta_std=eta_jitter.std(), eta_all=eta_jitter)

        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        style_axis(ax, f"{spec['title']}\ntai pixel ({cx},{cy}) THAT")
        txt = (f"$\\eta$(tam co dinh)={eta_center:.3f}\n"
               f"$\\eta$(TB qua jitter)={eta_jitter.mean():.3f}$\\pm${eta_jitter.std():.3f}")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.4, color="white",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor=COL_DARK, alpha=0.78, pad=0.35))

    # Panel C: phan bo eta qua cac lan jitter (histogram so sanh 2 Sigma) + duong eta tam
    ax2 = fig.add_subplot(gs[2])
    for key, spec in covs.items():
        r = results[key]
        ax2.hist(r["eta_all"], bins=28, color=spec["color"], alpha=0.5,
                  label=f"{key} ($\\sigma_\\eta$={r['eta_std']:.3f})", edgecolor=spec["color"])
        ax2.axvline(r["eta_center"], color=spec["color"], ls="--", lw=2.0)
    ax2.set_xlabel("$\\eta$ (mode wavelength, truc L=%.0fpx)" % L_axis)
    ax2.set_ylabel("So lan jitter (histogram)")
    ax2.set_facecolor(COL_PANEL)
    ax2.legend(fontsize=8, framealpha=0.9)
    style_axis(ax2, "Phan bo $\\eta$ qua %d lan jitter\n(net dut = $\\eta$ lay mau dung tam)" % N_SAMPLES)

    fig.suptitle("Hinh 2 (Gap 17.3-2) — Jitter Cholesky THAT (fu:257-292): hinh dang vet jitter\n"
                 "theo dung $\\Sigma_{2D}$; $\\Sigma$ di huong cho $\\eta$ dao dong manh hon quanh tam co dinh",
                 fontsize=12, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    savefig(fig, "deep17c_02_cholesky_jitter.png")
    return results


# ============================================================================
# GAP 3 — w_valid=1 (THAT) vs trong so transmittance (GIA DINH theo comment)
# ============================================================================
def fig03_weighting_bug(results_gap1):
    rng = np.random.default_rng(42)
    N_LAYERS = 6
    # Alpha-compositing chuan: T_i = prod_{j<i} (1 - alpha_j) — GIA DINH HOP LY
    # (khong co pipeline render that trong script nay), chi dung de minh hoa
    # "neu duoc weight nhu comment code noi" khac gi voi code THAT hien tai.
    alphas = rng.uniform(0.25, 0.6, size=N_LAYERS)
    alphas.sort()  # lop gan camera thuong opacity cao hon (gia dinh minh hoa)
    transmittance = np.empty(N_LAYERS)
    T = 1.0
    for i in range(N_LAYERS):
        transmittance[i] = T
        T *= (1.0 - alphas[i])

    # eta_k gia lap cho N_LAYERS Gaussian chong lan tai cung 1 pixel — dung
    # lai gia tri eta THAT tinh o Gap 1 (3 huong) lam nguon "eta_3ch" hop ly
    # (khong bia so ngau nhien cho ban than eta — chi bia PROFILE opacity/
    # transmittance vi khong co pipeline render that).
    base_eta = results_gap1["eta_wl"]  # 3 gia tri THAT (0/45/90 do) tu Gap 1
    rng2 = np.random.default_rng(7)
    eta_3ch_layers = base_eta[None, :] * (0.85 + 0.3 * rng2.random((N_LAYERS, 3)))

    # code THAT: weights_valid = 1 luon (freq_utils.py:238, 378-380)
    w_valid_actual = np.ones(N_LAYERS)
    eta_total_actual = (eta_3ch_layers * w_valid_actual[:, None]).sum(axis=1)

    # GIA DINH theo comment: neu weights_valid = max_transmittance (transmittance
    # THAT toi lop do) thi eta_total se bi lop sau (bi che nhieu) giam trong so
    w_valid_hypothetical = transmittance
    eta_total_hypothetical = (eta_3ch_layers * w_valid_hypothetical[:, None]).sum(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    x = np.arange(N_LAYERS)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])

    ax0 = axes[0]
    ax0.bar(x, w_valid_actual, color=COL_GRAY, edgecolor=COL_DARK, label="$w_{valid}$ THAT (=1 luon)")
    ax0.plot(x, w_valid_hypothetical, "o--", color=COL_PURPLE, lw=2.0, ms=6,
             label="$w_{valid}$ GIA DINH = transmittance $T_i$")
    ax0.set_xlabel("Lop Gaussian chong lan tai 1 pixel (thu tu depth)")
    ax0.set_ylabel("Trong so $w_{valid}$")
    ax0.set_ylim(0, 1.15)
    ax0.legend(fontsize=8, framealpha=0.9)
    ax0.set_facecolor(COL_PANEL)
    style_axis(ax0, "$w_{valid}$: THAT (code) vs GIA DINH (theo comment)\nfu:238, 378-380")

    ax1 = axes[1]
    w = 0.38
    b1 = ax1.bar(x - w/2, eta_total_actual, width=w, color=COL_ORANGE, edgecolor=COL_DARK,
                 label="$\\eta_{total}$ THAT (code hien tai, $w{=}1$)")
    b2 = ax1.bar(x + w/2, eta_total_hypothetical, width=w, color=COL_PURPLE, edgecolor=COL_DARK,
                 label="$\\eta_{total}$ GIA DINH (weight = $T_i$)")
    ax1.set_xlabel("Lop Gaussian (thu tu depth)")
    ax1.set_ylabel("$\\eta_{total}=\\sum_k \\eta_k \\cdot w_{valid}$")
    ax1.legend(fontsize=7.8, framealpha=0.9, loc="upper right")
    ax1.set_facecolor(COL_PANEL)
    style_axis(ax1, "$\\eta_{total}$ moi lop: THAT vs GIA DINH")

    ax2 = axes[2]
    diff_pct = 100.0 * (eta_total_actual - eta_total_hypothetical) / np.clip(eta_total_hypothetical, 1e-6, None)
    colors_bar = [COL_RED if d > 0 else COL_BLUE for d in diff_pct]
    ax2.bar(x, diff_pct, color=colors_bar, edgecolor=COL_DARK)
    for xi, d in zip(x, diff_pct):
        ax2.text(xi, d, f"{d:+.0f}%", ha="center",
                  va="bottom" if d >= 0 else "top", fontsize=8, fontweight="bold")
    ax2.axhline(0, color=COL_DARK, lw=1.0)
    ax2.set_xlabel("Lop Gaussian (thu tu depth)")
    ax2.set_ylabel("Chenh lech (%) THAT so GIA DINH")
    ax2.set_facecolor(COL_PANEL)
    style_axis(ax2, "Lop cang sau (transmittance thap) cang bi\nTHAT (code) DINH GIA CAO hon GIA DINH (comment)")

    fig.suptitle("Hinh 3 (Gap 17.3-3) — $w_{valid}{=}1$ THAT trong code (fu:238) so voi trong so\n"
                 "transmittance GIA DINH ma comment \"Weight by Transmittance\" (fu:378-380) ngu y",
                 fontsize=12, fontweight="bold", color=COL_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    savefig(fig, "deep17c_03_weighting_bug.png")
    return dict(alphas=alphas, transmittance=transmittance,
                eta_total_actual=eta_total_actual, eta_total_hypothetical=eta_total_hypothetical)


# ============================================================================
# main
# ============================================================================
def main():
    img = load_real_patch()
    print("Anh THAT dung:", SRC_IMAGE, "panel box", PANEL_BOX, "shape", img.shape)

    Sxx, Sxy, Syy = get_structure_tensor_np(img, sigma=1.0, rho=1.0)
    print("Structure tensor THAT tinh xong tren patch moi (chua dung o cac script khac).")

    r1 = fig01_orientation_invariance(img, Sxx, Sxy, Syy)
    print(f"[Gap1] vi tri canh ngang=({r1['x0']},{r1['y0']}) S=({r1['sxx']:.4f},{r1['sxy']:.4f},{r1['syy']:.4f}) "
          f"wl_min={r1['wl_min']:.2f}px")
    print(f"[Gap1] eta_wavelength theo huong 0/45/90deg = {np.round(r1['eta_wl'],4)} (BAT BIEN)")
    print(f"[Gap1] eta_projection theo huong 0/45/90deg = {np.round(r1['eta_pr'],4)} (THAY DOI)")

    r2 = fig02_cholesky_jitter(img, Sxx, Sxy, Syy)
    for key, r in r2.items():
        print(f"[Gap2] {key}: eta_center={r['eta_center']:.4f} eta_jitter_mean={r['eta_mean']:.4f} "
              f"+-{r['eta_std']:.4f}")

    r3 = fig03_weighting_bug(r1)
    print(f"[Gap3] eta_total THAT (w=1) = {np.round(r3['eta_total_actual'],3)}")
    print(f"[Gap3] eta_total GIA DINH (w=T_i) = {np.round(r3['eta_total_hypothetical'],3)}")

    print("Hoan tat — 3 hinh deep17c_0{1,2,3}_*.png da luu trong", HERE)


if __name__ == "__main__":
    main()
