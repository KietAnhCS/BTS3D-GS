# -*- coding: utf-8 -*-
"""
Overview / "teaser-style" figures for SADGS, built strictly from real formulas
in the SADGS codebase (no invented math). Pipeline mirrored from the paper
teaser: Input Image -> Laplacian(DoG) Scale Space -> Multiscale Structure-
Tensor Field -> 3D Gaussian Training (structure-aware splitting).

Source of every formula (line numbers as of this snapshot):
  fast_gaussian_blur                  SADGS/utils/loss_utils.py:113-168
  get_structure_tensor_torch          SADGS/utils/loss_utils.py:170-230
  get_multiscale_structure_tensor_v1  SADGS/utils/loss_utils.py:232-313
  get_multiscale_structure_tensor_v2  SADGS/utils/loss_utils.py:315-387
  eta (wavelength mode)               SADGS/utils/freq_utils.py:313-348
  densify_and_split_structgs          SADGS/scene/gaussian_model.py:638-832
  expand_undersized_gs                SADGS/scene/gaussian_model.py:833-857

Input image: real photo cropped from DOCS/assets/samples_train.png
(gt 00001 tile of the "train" COLMAP scene actually used to train SADGS).

Output: DOCS/Slides67/figures/sadgs_overview/*.png
"""
import os
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.ndimage import gaussian_filter

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
os.makedirs(OUT, exist_ok=True)


def save(fig, name, dpi=170):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# ---------------------------------------------------------------------------
# Real formulas re-implemented in numpy (must match SADGS/utils/loss_utils.py)
# ---------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def fast_blur(img_chw, sigma):
    """fast_gaussian_blur (loss_utils.py:113). Downsample branch skipped
    (CPU/numpy, same output up to interpolation aliasing at sigma>2)."""
    if sigma < 0.01:
        return img_chw
    return np.stack([gaussian_filter(c, sigma, mode="reflect") for c in img_chw], 0)


def structure_tensor(img_chw, sigma=1.0, rho=1.0, normalize=True):
    """get_structure_tensor_torch (loss_utils.py:170-230), Di Zenzo multi-channel ST."""
    smooth = fast_blur(img_chw, sigma)
    from scipy.ndimage import convolve
    Ix = np.stack([convolve(c, SOBEL_X, mode="reflect") for c in smooth], 0)
    Iy = np.stack([convolve(c, SOBEL_Y, mode="reflect") for c in smooth], 0)
    Ixx = (Ix ** 2).sum(0)
    Ixy = (Ix * Iy).sum(0)
    Iyy = (Iy ** 2).sum(0)
    Sxx = gaussian_filter(Ixx, rho, mode="reflect")
    Sxy = gaussian_filter(Ixy, rho, mode="reflect")
    Syy = gaussian_filter(Iyy, rho, mode="reflect")
    if normalize:
        mag = Sxx + Syy
        m = mag.max() + 1e-6
        Sxx, Sxy, Syy = Sxx / m, Sxy / m, Syy / m
    return Sxx, Sxy, Syy


def multiscale_structure_tensor_v1(img_chw, levels=4, base_sigma=1.0, octave_step=1.5,
                                    power_factor=3.0, smoothing_factor=1.0):
    """get_multiscale_structure_tensor_v1 (loss_utils.py:232-313)."""
    H, W = img_chw.shape[1:]
    Sxx_b, Sxy_b, Syy_b = structure_tensor(img_chw, sigma=base_sigma, rho=1.0, normalize=False)

    accum_Sxx = np.zeros((H, W))
    accum_Sxy = np.zeros((H, W))
    accum_Syy = np.zeros((H, W))
    accum_w = np.zeros((H, W))

    current = img_chw
    sigma_accum = 0.0
    levels_info = []
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        sigma_inc = math.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
        nxt = fast_blur(current, sigma_inc)

        band_response = np.sqrt(((current - nxt) ** 2).sum(0))
        if i > 0 and smoothing_factor > 0:
            band_response = gaussian_filter(band_response, target_sigma * 2.0, mode="reflect")

        integ_rho = target_sigma * 3.0
        Sxx_i = gaussian_filter(Sxx_b, integ_rho, mode="reflect")
        Sxy_i = gaussian_filter(Sxy_b, integ_rho, mode="reflect")
        Syy_i = gaussian_filter(Syy_b, integ_rho, mode="reflect")

        trace_i = Sxx_i + Syy_i + 1e-6
        Sxx_n, Sxy_n, Syy_n = Sxx_i / trace_i, Sxy_i / trace_i, Syy_i / trace_i

        w_i = band_response ** power_factor
        target = band_freq ** 2

        accum_Sxx += Sxx_n * w_i * target
        accum_Sxy += Sxy_n * w_i * target
        accum_Syy += Syy_n * w_i * target
        accum_w += w_i

        levels_info.append(dict(i=i, sigma=target_sigma, freq=band_freq,
                                 band=band_response.copy(), weight=w_i.copy(),
                                 blurred=nxt.copy()))
        current = nxt
        sigma_accum = target_sigma

    Sxx = accum_Sxx / (accum_w + 1e-6)
    Sxy = accum_Sxy / (accum_w + 1e-6)
    Syy = accum_Syy / (accum_w + 1e-6)
    return Sxx, Sxy, Syy, levels_info


def eigs_2x2(Sxx, Sxy, Syy):
    tr = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((tr / 2) ** 2 - det, 0, None))
    l1 = tr / 2 + delta
    l2 = tr / 2 - delta
    # v1 eigenvector angle: theta = 0.5*atan2(2Sxy, Sxx-Syy)  (freq_utils.py / standard ST)
    theta = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy + 1e-12)
    return l1, l2, theta


def k_of_eta(eta):
    """densify_and_split_structgs (gaussian_model.py:699):
    ks = torch.sqrt(torch.clamp(etavals, min=1.0)).ceil().int() ; clamp min=1"""
    return max(1, int(np.ceil(np.sqrt(max(1.0, eta)))))


# ---------------------------------------------------------------------------
# Load real photo
# ---------------------------------------------------------------------------
img_pil = Image.open(os.path.join(HERE, "input_photo.png")).convert("RGB")
img_pil = img_pil.resize((320, 176), Image.LANCZOS)
img_rgb = np.asarray(img_pil).astype(np.float64) / 255.0  # (H,W,3)
IMG = np.transpose(img_rgb, (2, 0, 1))  # (3,H,W) matches (C,H,W) in loss_utils

# ===========================================================================
# FIGURE 1 (slide 1 hero): Input -> Laplacian(DoG) Scale Space -> ST field
# ===========================================================================
LEVELS = 4
Sxx, Sxy, Syy, LV = multiscale_structure_tensor_v1(IMG, levels=LEVELS, base_sigma=1.0,
                                                     octave_step=1.5, power_factor=3.0)
L1, L2, THETA = eigs_2x2(Sxx, Sxy, Syy)

fig = plt.figure(figsize=(15, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.35], wspace=0.15)

# --- panel 1: input image ---
ax0 = fig.add_subplot(gs[0])
ax0.imshow(img_rgb)
ax0.set_title("Input Image\n(gt 00001, scene \"train\")", fontsize=11)
ax0.axis("off")

# --- panel 2: Laplacian (DoG) scale space stack, offset like a pyramid ---
ax1 = fig.add_subplot(gs[1])
ax1.axis("off")
ax1.set_title(r"Laplacian Scale Space  $b_i=\|I^{(i)}-G_{\Delta\sigma_i}{*}I^{(i)}\|_2$"
              "\n(loss_utils.py:270, Difference-of-Gaussians band)", fontsize=10.5)
H, W = img_rgb.shape[:2]
n = len(LV)
dx, dy = 0.10, 0.10
for i, lv in enumerate(LV):
    band = lv["band"]
    band_n = band / (band.max() + 1e-9)
    ext = [i * dx, i * dx + 1, -(i * dy), -(i * dy) + (H / W)]
    ax1.imshow(band_n, cmap="magma", extent=ext, zorder=n - i, alpha=0.97)
    ax1.add_patch(plt.Rectangle((ext[0], ext[2]), 1, H / W, fill=False,
                                 lw=0.8, edgecolor="k", zorder=n - i + 0.5))
    ax1.text(ext[0] + 1.02, ext[2] + (H / W) / 2,
              r"$i{=}%d,\ \sigma_i{=}%.2f,\ f_i{=}%.2f$" % (i, lv["sigma"], lv["freq"]),
              fontsize=7.5, va="center")
ax1.set_xlim(-0.05, 1 + n * dx + 0.9)
ax1.set_ylim(-(n - 1) * dy - 0.15, H / W + 0.15)
ax1.set_aspect("equal")

# --- panel 3: multiscale structure-tensor field as ellipses over the image ---
ax2 = fig.add_subplot(gs[2])
ax2.imshow(img_rgb, alpha=0.55)
step = 10
ys = np.arange(step // 2, H, step)
xs = np.arange(step // 2, W, step)
lam_ref = np.percentile(L1, 92) + 1e-9
for yy in ys:
    for xx in xs:
        l1v, l2v, th = L1[yy, xx], L2[yy, xx], THETA[yy, xx]
        if l1v < 1e-8:
            continue
        r1 = 3.2 * math.sqrt(max(l1v, 1e-8) / lam_ref)
        r2 = 3.2 * math.sqrt(max(l2v, 1e-8) / lam_ref)
        r1 = min(r1, 4.2)
        r2 = min(r2, r1)
        e = Ellipse((xx, yy), width=2 * r1, height=2 * r2,
                    angle=math.degrees(th), edgecolor="gold", facecolor="none",
                    linewidth=0.6, alpha=0.85)
        ax2.add_patch(e)
ax2.set_xlim(0, W)
ax2.set_ylim(H, 0)
ax2.set_title(r"Multiscale Structure-Tensor Field  $\hat J=\dfrac{\sum_i (J_i/\mathrm{tr}J_i)\,w_i f_i^2}{\sum_i w_i}$"
              "\n(loss_utils.py:300-311, get_multiscale_structure_tensor_v1)", fontsize=10.2)
ax2.axis("off")

fig.suptitle("2D Structure Analysis — real formulas, real photo (SADGS/utils/loss_utils.py)",
             fontsize=12.5, y=1.05)
save(fig, "01_2d_structure_analysis.png")

# ===========================================================================
# FIGURE 2 (slide 1 support): worked numeric example at 3 real pixels
# ===========================================================================
# Pick real, representative pixels automatically from the actual L1/coherence maps
# instead of guessing coordinates: flattest patch, strongest edge, strongest corner/texture.
COH = ((L1 - L2) / (L1 + L2 + 1e-9)) ** 2
margin = 16
valid = np.zeros_like(L1, dtype=bool)
valid[margin:H - margin, margin:W - margin] = True

flat_mask = valid & (L1 < np.percentile(L1[valid], 20))
flat_yx = np.unravel_index(np.argmin(np.where(flat_mask, L1, np.inf)), L1.shape)

edge_score = np.where(valid, L1 * COH, -1)
edge_yx = np.unravel_index(np.argmax(edge_score), L1.shape)

corner_score = np.where(valid, L1 * (1 - COH), -1)
corner_yx = np.unravel_index(np.argmax(corner_score), L1.shape)

picks = [
    ("flattest patch (sky/plain)", flat_yx[0], flat_yx[1]),
    ("strongest edge (coherent)", edge_yx[0], edge_yx[1]),
    ("strongest corner/texture", corner_yx[0], corner_yx[1]),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
crop_half = 14
for ax, (label, yy, xx) in zip(axes, picks):
    y0, y1 = max(0, yy - crop_half), min(H, yy + crop_half)
    x0, x1 = max(0, xx - crop_half), min(W, xx + crop_half)
    ax.imshow(img_rgb[y0:y1, x0:x1])
    ax.scatter([xx - x0], [yy - y0], s=40, facecolors="none", edgecolors="red", linewidths=1.5)
    l1v, l2v = L1[yy, xx], L2[yy, xx]
    coh = COH[yy, xx]
    wmin = 1.0 / (math.sqrt(max(l1v, 0)) + 1e-5)
    ax.set_title(f"{label}\n" r"$\lambda_1{=}%.4f,\ \lambda_2{=}%.4f$" "\n"
                 r"coherence$=%.2f$, $w_{\min}\!\approx\!%.1f$px" % (l1v, l2v, coh, wmin),
                 fontsize=9.3)
    ax.axis("off")
fig.subplots_adjust(top=0.72, wspace=0.1)
fig.suptitle(r"Worked example: eigen-decomposition of $\hat J$ at real pixels of the photo"
             "\n(freq_utils.py:325-334:  " r"$\lambda_1=\mathrm{tr}/2+\delta,\ w_{\min}=1/(\sqrt{\lambda_1}+10^{-5})$)",
             fontsize=11.0)
save(fig, "02_eigendecomp_examples.png")

# ===========================================================================
# FIGURE 3 (slide 2 hero): 3D Gaussian Training — structure-aware splitting
# ===========================================================================
np.random.seed(3)


def draw_gaussian_blob(ax, cx, cy, sx, sy, theta_deg, color, alpha=0.55, n=1):
    e = Ellipse((cx, cy), width=2 * sx, height=2 * sy, angle=theta_deg,
                facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(e)


def split_children(cx, cy, sigma, ks, theta_rad, scale_power=1.0):
    """Exact reimplementation of densify_and_split_structgs (gaussian_model.py:638-832)
    for a single 2D parent (kz axis fixed to 1)."""
    kx, ky = int(ks[0]), int(ks[1])
    sigma_new = sigma / np.array([kx, ky], dtype=float)
    new_scale = sigma / (np.array([kx, ky], dtype=float) ** scale_power)
    sep = sigma_new * math.sqrt(12.0)  # separations (:776)
    R = np.array([[math.cos(theta_rad), -math.sin(theta_rad)],
                  [math.sin(theta_rad), math.cos(theta_rad)]])
    pts = []
    for ix in range(kx):
        for iy in range(ky):
            gx = ix - (kx - 1) / 2.0
            gy = iy - (ky - 1) / 2.0
            local = np.array([gx, gy]) * sep
            world = R @ local
            pts.append((cx + world[0], cy + world[1], new_scale[0], new_scale[1]))
    return pts


fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

# --- (a) Initialization: sparse, isotropic Gaussians from SfM points ---
ax = axes[0]
ax.set_facecolor("#efe7da")
rng = np.random.default_rng(5)
pts0 = rng.normal(size=(60, 2)) * 1.0
for (x, y) in pts0:
    draw_gaussian_blob(ax, x, y, 0.55, 0.55, 0, "seagreen", alpha=0.35)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Initialization\n(SfM points, isotropic $\\sigma$)", fontsize=11)

# --- (b) baseline gradient split: k=2 on the longest axis only (fallback path) ---
ax = axes[1]
ax.set_facecolor("#efe7da")
theta = math.radians(35)
sigma = np.array([1.6, 0.6])
draw_gaussian_blob(ax, 0, 0, sigma[0] * 0.0, sigma[1] * 0.0, 0, "none", 0)  # noop for symmetry
ks_base = np.array([2, 1])  # gradient-based fallback: ks.scatter_ on argmax axis (:707-709)
kids = split_children(0, 0, sigma, ks_base, theta, scale_power=1.0)
for (x, y, sx, sy) in kids:
    draw_gaussian_blob(ax, x, y, sx, sy, math.degrees(theta), "steelblue", alpha=0.55)
draw_gaussian_blob(ax, 0, 0, sigma[0], sigma[1], math.degrees(theta), "gray", alpha=0.18)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Previous Densification\n(gradient split, $k{=}2$ longest axis,\ngaussian_model.py:707-709)",
              fontsize=10.5)

# --- (c) structure-aware split: k_x, k_y from eta per axis (independent) ---
ax = axes[2]
ax.set_facecolor("#efe7da")
eta_x, eta_y = 14.0, 3.0  # illustrative but computed with the real formula below
kx = k_of_eta(eta_x)
ky = k_of_eta(eta_y)
ks_sa = np.array([kx, ky])
kids = split_children(0, 0, sigma, ks_sa, theta, scale_power=1.0)
for (x, y, sx, sy) in kids:
    draw_gaussian_blob(ax, x, y, sx, sy, math.degrees(theta), "seagreen", alpha=0.55)
draw_gaussian_blob(ax, 0, 0, sigma[0], sigma[1], math.degrees(theta), "gray", alpha=0.15)
# dashed guide lines like the teaser
for (x, y, sx, sy) in kids:
    ax.plot([0, x], [0, y], ls="--", lw=0.6, color="k", alpha=0.5)
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(("Our Densification (Structure-Aware)\n"
              r"$k_i=\lceil\sqrt{\max(\eta_i,1)}\rceil$: $\eta_x{=}%.0f\Rightarrow k_x{=}%d$,  "
              r"$\eta_y{=}%.0f\Rightarrow k_y{=}%d$""\n(gaussian_model.py:688-722)")
             % (eta_x, kx, eta_y, ky), fontsize=10.0)

fig.suptitle("3D Gaussian Training — structure-aware splitting (per-axis, frequency-driven)",
             fontsize=12.8, y=1.06)
save(fig, "03_3d_gaussian_training_split.png")

# ===========================================================================
# FIGURE 4 (slide 2 support): eta formula -> k mapping, worked curve
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

ax = axes[0]
eta_range = np.linspace(0, 20, 400)
k_curve = np.array([k_of_eta(e) for e in eta_range])
ax.plot(eta_range, np.sqrt(np.clip(eta_range, 1, None)), color="steelblue", lw=1.6,
        label=r"$\sqrt{\max(\eta,1)}$")
ax.step(eta_range, k_curve, color="seagreen", lw=1.8, where="post",
         label=r"$k=\lceil\sqrt{\max(\eta,1)}\rceil$ (int, clamp $k\geq1$)")
ax.axhline(1, color="gray", lw=0.6, ls=":")
ax.axvline(1, color="gray", lw=0.6, ls=":")
ax.set_xlabel(r"$\eta$  (axis length / local wavelength, freq_utils.py:348)")
ax.set_ylabel("splits along this axis, $k$")
ax.set_title("Per-axis split count from aliasing ratio $\\eta$\n(gaussian_model.py:688-701)", fontsize=10.5)
ax.legend(fontsize=8.5)

ax = axes[1]
# eta = axis_length / wavelength_min ; wavelength_min = 1/(sqrt(lambda1)+1e-5)
axis_len = np.linspace(0.5, 25, 200)  # pixels, projected gaussian axis
for lam1, style in [(0.02, "-"), (0.08, "--"), (0.20, ":")]:
    wmin = 1.0 / (math.sqrt(lam1) + 1e-5)
    eta = axis_len / wmin
    ax.plot(axis_len, eta, style, label=r"$\lambda_1{=}%.2f\Rightarrow w_{\min}{=}%.1f$px" % (lam1, wmin))
ax.axhline(1.0, color="red", lw=0.8, ls="--", label=r"$\eta{=}1$ (Nyquist)")
ax.set_xlabel("projected Gaussian axis length (px)")
ax.set_ylabel(r"$\eta$")
ax.set_title(r"$\eta=\dfrac{\text{axis length}}{w_{\min}}$  (freq_utils.py:313-348, mode=\"wavelength\")",
             fontsize=10.5)
ax.legend(fontsize=7.5)

fig.suptitle("From structure tensor to split count: the exact $\\eta\\!\\to\\!k$ chain used in training",
             fontsize=11.8, y=1.04)
save(fig, "04_eta_to_k_chain.png")

print("DONE")
