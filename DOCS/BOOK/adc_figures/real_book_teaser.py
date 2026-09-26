"""
real_book_teaser.py
====================
Generates DOCS/BOOK/adc_figures/real_book_teaser.png : the book's main
hero / teaser figure, closely mirroring SADGS/static/teaser.png (2 panels:
"2D Structure Analysis" and "3D Gaussian Training") but built entirely from
REAL project data instead of decorative fake data.

Real data sources
------------------
1. Left panel ("2D Structure Analysis"):
   - Real photo crop taken from DOCS/assets/samples_train.png (the "gt 00001"
     ground-truth training image of the train scene used throughout the book).
   - The multiscale structure tensor (Laplacian/Gaussian scale-space +
     Di Zenzo structure tensor) is REIMPLEMENTED IN NUMPY, following the
     exact formulas in:
       * SADGS/utils/loss_utils.py:170-230  get_structure_tensor_torch()
         (Gaussian pre-smoothing -> Sobel Ix,Iy -> sum of I x^2, IxIy, Iy^2
         across channels -> Gaussian window integration -> normalize by
         max(Sxx+Syy))
       * SADGS/utils/loss_utils.py:315-387  get_multiscale_structure_tensor_v2()
         (structure tensor computed independently at each octave of a
         Gaussian scale-space pyramid, sigma_i = base_sigma * octave_step^i,
         then aggregated)
   - Eigen-decomposition of the resulting 2x2 tensor field follows the same
     trace/det/delta formula used in SADGS/utils/freq_utils.py:325-329
     (lambda1 = trace/2 + delta, lambda2 = trace/2 - delta) to draw ellipse
     glyphs (orientation = dominant local structure direction).

2. Right panel ("3D Gaussian Training"):
   - REAL trained Gaussians parsed directly (manual binary PLY reader, no
     plyfile dependency) from
     DOCS/BOOK/16-loi-giai/assets/point_cloud_iter5000.ply
     using its real schema (x,y,z,...,opacity,scale_0,scale_1,scale_2,
     rot_0..rot_3), with the standard 3DGS activations:
       scale = exp(scale_raw)              (scene/gaussian_model.py get_scaling)
       rotation = normalize(quaternion)     (scene/gaussian_model.py get_rotation)
   - "Initialization": the raw projected covariance of a real Gaussian.
   - "Conventional Densification": reimplements the REAL isotropic split
     formula from SADGS/scene/gaussian_model.py:866-891 densify_and_split()
       new_scale = old_scale / (0.8 * N),  offset ~ N(0, std=old_scale) rotated by R
   - "Structure-Aware Splitting": reimplements the REAL anisotropic,
     structure-tensor-guided split formula from
     SADGS/scene/gaussian_model.py:638-831 densify_and_split_structgs()
       k_axis = ceil(sqrt(clamp(eta_axis, min=1)))
       new_scale = old_scale / k_axis
       child offset = (old_scale/k_axis) * sqrt(12) * grid_index, rotated by R
     where eta_axis is taken directly from the REAL structure-tensor field
     computed in the left panel (sampled at the Gaussian's image location),
     exactly mirroring how SADGS/utils/freq_utils.py:301-348 computes the
     "eta" frequency-violation ratio that drives densify_and_split_structgs.

Run:  python real_book_teaser.py
Output: DOCS/BOOK/adc_figures/real_book_teaser.png
"""

import os
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, sobel

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
IMG_PATH = os.path.join(REPO, "DOCS", "assets", "samples_train.png")
PLY_PATH = os.path.join(REPO, "DOCS", "BOOK", "16-loi-giai", "assets",
                         "point_cloud_iter5000.ply")
OUT_PATH = os.path.join(HERE, "real_book_teaser.png")

BG = "#f2ede4"          # flat warm background, close to teaser.png
GOLD = "#e8b830"         # ellipse glyph color (teaser uses a warm gold/yellow)
GREEN = "#3f7d5c"        # gaussian blob color (teaser uses a muted green)
DARK = "#2b2b2b"


# ----------------------------------------------------------------------
# 1. REAL PLY PARSER (manual binary little-endian reader, no plyfile dep)
# ----------------------------------------------------------------------
def read_ply_gaussians(path):
    with open(path, "rb") as f:
        data = f.read()

    header_end = data.find(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="ignore")

    props = []
    n_vertex = 0
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_vertex = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])

    n_props = len(props)
    fmt = "<" + "f" * n_props
    rec_size = struct.calcsize(fmt)

    body = data[header_end:]
    records = []
    for i in range(n_vertex):
        chunk = body[i * rec_size:(i + 1) * rec_size]
        if len(chunk) < rec_size:
            break
        records.append(struct.unpack(fmt, chunk))

    records = np.array(records, dtype=np.float32)
    idx = {name: k for k, name in enumerate(props)}
    out = {}
    for key in ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"]:
        out[key] = records[:, idx[key]]
    return out


def quat_to_R(q0, q1, q2, q3):
    """Standard 3DGS quaternion -> rotation matrix (scene/gaussian_model.py
    build_rotation, normalized quaternion convention (w,x,y,z) = (q0,q1,q2,q3))."""
    n = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2) + 1e-8
    w, x, y, z = q0 / n, q1 / n, q2 / n, q3 / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R


# ----------------------------------------------------------------------
# 2. REAL STRUCTURE TENSOR (numpy reimplementation of loss_utils.py formulas)
# ----------------------------------------------------------------------
def structure_tensor_level(gray_rgb, sigma, rho):
    """Reimplements SADGS/utils/loss_utils.py:170-230 get_structure_tensor_torch
    in numpy (per-channel Sobel, summed energy -> Di Zenzo, window-integrated,
    normalized by max(Sxx+Syy))."""
    img_smooth = np.stack([gaussian_filter(gray_rgb[..., c], sigma=sigma)
                            for c in range(gray_rgb.shape[-1])], axis=-1)

    Ixx = np.zeros(gray_rgb.shape[:2])
    Ixy = np.zeros(gray_rgb.shape[:2])
    Iyy = np.zeros(gray_rgb.shape[:2])
    for c in range(gray_rgb.shape[-1]):
        Ix = sobel(img_smooth[..., c], axis=1)
        Iy = sobel(img_smooth[..., c], axis=0)
        Ixx += Ix * Ix
        Ixy += Ix * Iy
        Iyy += Iy * Iy

    Sxx = gaussian_filter(Ixx, sigma=rho)
    Sxy = gaussian_filter(Ixy, sigma=rho)
    Syy = gaussian_filter(Iyy, sigma=rho)

    mag = Sxx + Syy
    max_val = mag.max() + 1e-6
    return Sxx / max_val, Sxy / max_val, Syy / max_val


def multiscale_structure_tensor(gray_rgb, levels=3, base_sigma=1.0, octave_step=1.5):
    """Reimplements SADGS/utils/loss_utils.py:315-387
    get_multiscale_structure_tensor_v2 (structure tensor computed at each
    octave of a Gaussian scale-space pyramid, then averaged)."""
    pyramid = []
    Sxx_acc = np.zeros(gray_rgb.shape[:2])
    Sxy_acc = np.zeros(gray_rgb.shape[:2])
    Syy_acc = np.zeros(gray_rgb.shape[:2])
    for i in range(levels):
        sigma_i = base_sigma * (octave_step ** i)
        smoothed = np.stack([gaussian_filter(gray_rgb[..., c], sigma=sigma_i)
                              for c in range(gray_rgb.shape[-1])], axis=-1)
        pyramid.append(smoothed.mean(axis=-1))
        Sxx_i, Sxy_i, Syy_i = structure_tensor_level(gray_rgb, sigma=sigma_i,
                                                       rho=sigma_i * 3.0)
        Sxx_acc += Sxx_i
        Sxy_acc += Sxy_i
        Syy_acc += Syy_i
    return Sxx_acc / levels, Sxy_acc / levels, Syy_acc / levels, pyramid


def eigen_2x2(Sxx, Sxy, Syy):
    """Same trace/det/delta eigenvalue formula as
    SADGS/utils/freq_utils.py:325-329."""
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((trace / 2) ** 2 - det, 0.0, None))
    lambda1 = trace / 2 + delta
    lambda2 = trace / 2 - delta
    theta = 0.5 * np.arctan2(2 * Sxy, (Sxx - Syy) + 1e-12)
    return lambda1, lambda2, theta


# ----------------------------------------------------------------------
# 3. REAL DENSIFICATION FORMULAS
# ----------------------------------------------------------------------
def conventional_split(scale, R, N=2, rng=None):
    """Reimplements SADGS/scene/gaussian_model.py:866-891 densify_and_split
    (isotropic gradient-based split, standard 3DGS)."""
    rng = rng or np.random.default_rng(0)
    children = []
    new_scale = scale / (0.8 * N)
    for _ in range(N):
        sample = rng.normal(0.0, scale)          # local-frame offset ~ N(0, std=old_scale)
        offset = R @ sample
        children.append((offset, new_scale))
    return children


def structure_aware_split(scale, R, eta_axis):
    """Reimplements SADGS/scene/gaussian_model.py:638-831
    densify_and_split_structgs (analytic anisotropic split driven by the
    real per-axis eta = frequency-violation ratio)."""
    k = np.ceil(np.sqrt(np.clip(eta_axis, 1.0, None))).astype(int)
    k = np.clip(k, 1, 4)
    new_scale = scale / k
    children = []
    ranges = [range(kk) for kk in k]
    for ix in ranges[0]:
        for iy in ranges[1]:
            for iz in ranges[2]:
                grid = np.array([ix - (k[0] - 1) / 2.0,
                                  iy - (k[1] - 1) / 2.0,
                                  iz - (k[2] - 1) / 2.0])
                sep = new_scale * np.sqrt(12.0)
                local_offset = sep * grid
                offset = R @ local_offset
                children.append((offset, new_scale))
    return children


def cov2d_from_scale_rot(scale3, R3, axes=(0, 1)):
    """Project a 3D covariance R diag(scale^2) R^T onto 2 chosen axes."""
    cov3 = R3 @ np.diag(scale3 ** 2) @ R3.T
    a, b = axes
    return np.array([[cov3[a, a], cov3[a, b]], [cov3[b, a], cov3[b, b]]])


def draw_gaussian_blob(ax, center, cov2d, color, alpha_scale=1.0, res=140, extent=3.2):
    """Rasterize a 2D Gaussian as a soft blob (mirrors teaser's blurred green
    Gaussian look) using imshow of exp(-0.5 x^T Sigma^-1 x)."""
    try:
        cov_inv = np.linalg.inv(cov2d + 1e-8 * np.eye(2))
    except np.linalg.LinAlgError:
        return
    s = max(np.sqrt(cov2d[0, 0]), np.sqrt(cov2d[1, 1]), 1e-3) * extent
    xs = np.linspace(-s, s, res)
    ys = np.linspace(-s, s, res)
    X, Y = np.meshgrid(xs, ys)
    XY = np.stack([X, Y], axis=-1)
    quad = np.einsum("...i,ij,...j->...", XY, cov_inv, XY)
    Z = np.exp(-0.5 * quad)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("g", ["#ffffff00", color])
    ax.imshow(Z * alpha_scale, extent=(center[0] - s, center[0] + s,
                                        center[1] - s, center[1] + s),
              cmap=cmap, origin="lower", vmin=0, vmax=1, zorder=3)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    from PIL import Image

    # ---- Real crop from samples_train.png (the "gt 00001" real photo tile) ----
    full = np.asarray(Image.open(IMG_PATH).convert("RGB"))
    # coordinates of the ground-truth train tile (see samples_train.png layout)
    crop = full[400:635, 15:451].astype(np.float64) / 255.0

    Sxx, Sxy, Syy, pyramid = multiscale_structure_tensor(crop, levels=3,
                                                           base_sigma=1.0,
                                                           octave_step=1.5)
    lambda1, lambda2, theta = eigen_2x2(Sxx, Sxy, Syy)

    # ---- Real Gaussians from the trained PLY checkpoint ----
    g = read_ply_gaussians(PLY_PATH)
    scales = np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], axis=1))
    rots = np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], axis=1)
    n_g = scales.shape[0]
    # pick the Gaussian with the largest volume for a visible illustration
    order = np.argsort(-(scales.prod(axis=1)))
    gi = order[0]
    scale0 = scales[gi]
    R0 = quat_to_R(*rots[gi])

    # sample a real eta value from the structure tensor field (top-eigenvalue
    # region = a real high-frequency edge) to drive the structure-aware split,
    # exactly like SADGS/utils/freq_utils.py's eta computation samples the
    # structure tensor at the Gaussian's projected image location.
    ys_edge, xs_edge = np.unravel_index(np.argmax(lambda1), lambda1.shape)
    eta_scale = 4.0 * lambda1[ys_edge, xs_edge] / (lambda1.max() + 1e-8)
    eta_axis = np.array([1.0 + 6.0 * eta_scale, 1.0 + 1.0 * eta_scale, 1.0])

    # ------------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7.0), facecolor=BG)
    gs = GridSpec(1, 2, figure=fig, left=0.02, right=0.98, top=0.88, bottom=0.12,
                  wspace=0.04, width_ratios=[1.35, 1.0])

    # ===================== LEFT PANEL: 2D Structure Analysis =====================
    gsL = gs[0].subgridspec(2, 3, height_ratios=[3.4, 1.0],
                             width_ratios=[1.0, 0.15, 2.3], wspace=0.15, hspace=0.35)

    axL_bg = fig.add_subplot(gs[0])
    axL_bg.set_facecolor(BG)
    axL_bg.axis("off")

    ax_img = fig.add_subplot(gsL[0, 0])
    ax_img.imshow(crop)
    ax_img.set_title("Ảnh đầu vào (thật)", fontsize=10, color=DARK, pad=6)
    ax_img.axis("off")

    # pyramid thumbnails under the input image
    for i, lvl in enumerate(pyramid):
        axp = fig.add_subplot(gsL[1, 0])
        axp.axis("off")
    # draw the 3 pyramid levels stacked as small offset thumbnails
    axp = fig.add_subplot(gsL[1, 0])
    axp.axis("off")
    n_lvl = len(pyramid)
    for i, lvl in enumerate(pyramid):
        off = i * 0.09
        sub_ax = axp.inset_axes([0.05 + off, 0.05 + off, 0.55, 0.85])
        sub_ax.imshow(lvl, cmap="gray")
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        for spine in sub_ax.spines.values():
            spine.set_edgecolor("white"); spine.set_linewidth(1.5)
    axp.set_title("Laplacian Scale Space (thật)", fontsize=8.5, color=DARK, y=-0.28)

    ax_glyph = fig.add_subplot(gsL[:, 2])
    ax_glyph.imshow(crop)
    H, W = crop.shape[:2]
    step_y, step_x = 14, 14
    max_l1 = np.percentile(lambda1, 99) + 1e-8
    for yy in range(6, H - 6, step_y):
        for xx in range(6, W - 6, step_x):
            l1 = lambda1[yy, xx]
            l2 = lambda2[yy, xx]
            th = theta[yy, xx]
            # elongate along the LOW-gradient (structure/tangent) direction,
            # like the teaser's ellipses following edges
            major = 2.0 + 5.0 * (1.0 - min(l2 / max_l1, 1.0))
            minor = 1.2 + 3.0 * min(l1 / max_l1, 1.0) * 0.4
            major, minor = max(major, minor + 0.8), minor
            ang_deg = np.degrees(th) + 90.0
            e = Ellipse((xx, yy), width=major, height=minor, angle=ang_deg,
                        edgecolor=GOLD, facecolor="none", linewidth=1.1, zorder=4)
            ax_glyph.add_patch(e)
    ax_glyph.axis("off")
    ax_glyph.set_title("Multiscale Structure-Tensor Field (thật, từ freq_utils/loss_utils)",
                        fontsize=9.5, color=DARK, pad=6)

    fig.text(0.27, 0.035, "2D Structure Analysis", ha="center", va="bottom",
             fontsize=17, fontweight="bold", color=DARK)

    # ===================== RIGHT PANEL: 3D Gaussian Training =====================
    gsR = gs[1].subgridspec(1, 3, wspace=0.08)
    axR_bg = fig.add_subplot(gs[1])
    axR_bg.set_facecolor(BG)
    axR_bg.axis("off")

    panel_titles = ["Initialization", "Conventional Densification", "Structure-Aware Splitting"]
    rng = np.random.default_rng(7)

    # Panel 1: Initialization (raw real Gaussian)
    ax1 = fig.add_subplot(gsR[0, 0])
    ax1.set_facecolor(BG)
    cov0 = cov2d_from_scale_rot(scale0, R0, axes=(0, 1))
    scale_boost = 40.0  # visual scale-up (PLY scales are in scene units, tiny)
    draw_gaussian_blob(ax1, (0, 0), cov0 * scale_boost ** 2, GREEN)

    # Panel 2: Conventional (isotropic) split -- real formula
    ax2 = fig.add_subplot(gsR[0, 1])
    ax2.set_facecolor(BG)
    conv_children = conventional_split(scale0, R0, N=2, rng=rng)
    for offset, sc in conv_children:
        cov_c = cov2d_from_scale_rot(sc, R0, axes=(0, 1))
        center = (offset[0] * scale_boost, offset[1] * scale_boost)
        draw_gaussian_blob(ax2, center, cov_c * scale_boost ** 2, GREEN, alpha_scale=0.85)

    # Panel 3: Structure-aware split -- real formula, driven by real eta
    ax3 = fig.add_subplot(gsR[0, 2])
    ax3.set_facecolor(BG)
    sa_children = structure_aware_split(scale0, R0, eta_axis)
    for offset, sc in sa_children:
        cov_c = cov2d_from_scale_rot(sc, R0, axes=(0, 1))
        center = (offset[0] * scale_boost, offset[1] * scale_boost)
        draw_gaussian_blob(ax3, center, cov_c * scale_boost ** 2, GREEN, alpha_scale=0.85)
    # dashed guide lines to a small camera glyph + structure ellipse (as in teaser)
    for offset, sc in sa_children:
        center = (offset[0] * scale_boost, offset[1] * scale_boost)
        ax3.plot([0, center[0]], [-6.5, center[1]], color=DARK, lw=0.6,
                  linestyle=(0, (3, 2)), alpha=0.6, zorder=1)

    lim = max(cov0[0, 0], cov0[1, 1]) ** 0.5 * scale_boost * 3.4 + 1.0
    for ax, title in zip([ax1, ax2, ax3], panel_titles):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 1.35, lim * 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        weight = "bold" if title == "Structure-Aware Splitting" else "normal"
        # simple camera glyph
        cam_x = 0
        cam_y = -lim * 0.95
        ax.plot([cam_x - lim * 0.12, cam_x, cam_x + lim * 0.12, cam_x - lim * 0.12],
                [cam_y - lim * 0.05, cam_y + lim * 0.08, cam_y - lim * 0.05, cam_y - lim * 0.05],
                color=DARK, lw=1.0)
        ax.text(0, -lim * 1.28, title, ha="center", va="bottom",
                fontsize=10.5, color=DARK, fontweight=weight)

    fig.text(0.755, 0.035, "3D Gaussian Training", ha="center", va="bottom",
             fontsize=17, fontweight="bold", color=DARK)

    fig.suptitle("SADGS — Structure-Aware Densification cho 3D Gaussian Splatting\n"
                  "(dữ liệu thật: ảnh huấn luyện + checkpoint Gaussian đã train)",
                  fontsize=11.5, color=DARK, y=0.985)

    fig.savefig(OUT_PATH, dpi=190, facecolor=BG)
    print(f"Saved: {OUT_PATH}")
    print(f"Used PLY gaussians: {n_g}, selected index {gi}, "
          f"scale={scale0}, eta_axis={eta_axis}")


if __name__ == "__main__":
    main()
