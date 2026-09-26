"""
real_structure_analysis.py
===========================
"2D Structure Analysis" pipeline cua SADGS, ve lai bang numpy/scipy THUAN
(khong torch, khong du lieu gia/np.random cho so lieu khoa hoc) va chay tren
mot anh THAT, sau do xuat nhieu hinh chi tiet nhieu panel (gridspec) kieu
teaser cua paper (SADGS/static/teaser.png, panel trai "2D Structure Analysis":
Input Image -> Laplacian Scale Space -> Multiscale Structure-Tensor Field).

Input anh that
--------------
DOCS/assets/samples_train.png la mot LUOI 3x2 anh so sanh (render vs ground
truth) ghep san, khong phai mot anh don. De co mot anh that lien tuc (khong
duong vien lat) cho phan tich cau truc, script nay CAT (crop) o toa do pixel
co dinh (10, 395)-(452, 640) dung ra o "gt 00001" -- anh dau máy xe lua that
trong bo du lieu Tanks&Temples "train" -- roi dung nguyen anh RGB cat duoc
(khong np.random, khong ve tay) lam dau vao cho toan bo pipeline ben duoi.
  Anh goc:  DOCS/assets/samples_train.png  (1374x650, luoi 3x2, xem file de
            doi chieu toa do crop)
  Anh dung: crop (10,395)-(452,640) => 442x245 px, anh dau may + troi + duong
            ray -- nhieu canh manh (than xe, so hieu "713", duong chan troi,
            thanh ray) nen phu hop de minh hoa structure tensor / eta.

Cong thuc THAT duoc chep lai (KHONG phai freq_utils.py nhu suy doan ban dau --
sau khi doc toan bo SADGS/utils/freq_utils.py, 4 ham nay thuc su nam trong
SADGS/utils/loss_utils.py; freq_utils.py chi IMPORT get_structure_tensor_torch
tu loss_utils o dong 5). Trich dan dong cu the:
  - SADGS/utils/loss_utils.py:113-168  fast_gaussian_blur
        (danh gia lai: cho sigma<=2.0 code goc goi thang
        torchvision.transforms.functional.gaussian_blur voi kernel_size =
        (2*4*sigma+1)|1, padding 'reflect' -- day la Gaussian blur tach truc
        chuan; nhanh "downsample roi blur roi upsample" chi la toi uu toc do
        cho sigma lon tren GPU, khong doi dinh nghia toan hoc. Ban numpy o
        day dung scipy.ndimage.gaussian_filter(..., mode='reflect') cho MOI
        sigma -- tuong duong chinh xac voi nhanh sigma<=2.0 cua code goc, va
        la xap xi lien tuc rat sat cho nhanh downsample o sigma lon.)
  - SADGS/utils/loss_utils.py:170-230  get_structure_tensor_torch
        (blur tien xu ly sigma -> Sobel Ix,Iy tung kenh RGB (conv2d,
        padding=1, tuc zero-pad, groups=C) -> Ixx=Ix^2, Ixy=Ix*Iy, Iyy=Iy^2
        -> CONG DON qua 3 kenh mau (phuong phap Di Zenzo, dong 206-208) ->
        blur cua so tich hop rho -> chuan hoa toan cuc theo max(Sxx+Syy).)
  - SADGS/utils/loss_utils.py:232-313  get_multiscale_structure_tensor_v1
        (tinh structure tensor MOT LAN o base_sigma, roi voi moi octave i:
        lam mo dan anh (DoG band_response), blur LAI 3 thanh phan cua base
        tensor voi integration_rho = target_sigma*3 -> chuan hoa "chi con
        huong" (chia cho trace) -> nhan trong so band_response^power_factor
        * (1/octave_step^i)^2 -> cong don -> chia tong trong so.)
  - SADGS/utils/loss_utils.py:315-387  get_multiscale_structure_tensor_v2
        (giong v1 nhung o moi octave i TINH LAI structure tensor tren chinh
        anh da lam mo o scale do (get_structure_tensor_torch(next_smooth,
        sigma=target_sigma, rho=target_sigma*3)) thay vi chi blur lai tensor
        goc -- "true multi-scale" theo docstring goc dong 317-320.)
Moi con so trong 9 hinh PNG xuat ra deu la ket qua THAT cua cac ham numpy
ben duoi chay tren anh crop that o tren -- khong co np.random cho so lieu
khoa hoc (chi dung numpy o buoc lay mau luoi hien thi ellipse, deu la
np.linspace/arange, khong phai random).

Chay:  python DOCS/BOOK/adc_figures/real_structure_analysis.py
"""
import math
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from PIL import Image

# ----------------------------------------------------------------------------
# Duong dan
# ----------------------------------------------------------------------------
HERE = __file__
import os
BOOK_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(BOOK_DIR, "..", "..", ".."))
SRC_IMAGE = os.path.join(REPO_ROOT, "DOCS", "assets", "samples_train.png")
OUT_DIR = BOOK_DIR

# Bang mau flat-design dong nhat (tham khao style teaser.png)
BG_CREAM = "#efe9e1"
INK = "#20242b"
ACCENT_YELLOW = "#f2b705"
ACCENT_TEAL = "#1f6f78"
ACCENT_RED = "#c0392b"
ACCENT_BLUE = "#2b5c8a"
CMAP_FIELD = "magma"
CMAP_DIV = "coolwarm"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


# ============================================================================
# 1. Ham loi -- chep dung cong thuc SADGS/utils/loss_utils.py (numpy thuan)
# ============================================================================

def fast_gaussian_blur(img, sigma):
    """loss_utils.py:113-168 fast_gaussian_blur (nhanh sigma<=2.0, xem docstring
    dau file). img: (H,W,C) hoac (H,W) float32. Blur tung kenh doc lap,
    bien 'reflect' (giong padding_mode mac dinh cua torchvision.gaussian_blur)."""
    if sigma < 0.01:
        return img.copy()
    if img.ndim == 2:
        return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma=sigma, mode="reflect")
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel_channel(img_c):
    """loss_utils.py:187-196: F.conv2d(..., padding=1) = correlation, zero-pad."""
    Ix = ndimage.correlate(img_c, SOBEL_X, mode="constant", cval=0.0)
    Iy = ndimage.correlate(img_c, SOBEL_Y, mode="constant", cval=0.0)
    return Ix, Iy


def get_structure_tensor(image, sigma=1.0, rho=1.0, return_extras=False):
    """loss_utils.py:170-230 get_structure_tensor_torch, numpy thuan.
    image: (H,W,C) float32 trong [0,1]. Tra ve Sxx,Sxy,Syy (H,W) da chuan hoa
    theo max(Sxx+Syy) toan cuc (dong 219-227)."""
    H, W, C = image.shape
    img_smooth = fast_gaussian_blur(image, sigma)

    Ixx = np.zeros((H, W)); Ixy = np.zeros((H, W)); Iyy = np.zeros((H, W))
    Ix_ch = np.zeros((H, W, C)); Iy_ch = np.zeros((H, W, C))
    for c in range(C):
        Ix, Iy = sobel_channel(img_smooth[..., c])
        Ix_ch[..., c] = Ix; Iy_ch[..., c] = Iy
        Ixx += Ix * Ix
        Ixy += Ix * Iy
        Iyy += Iy * Iy

    Sxx = fast_gaussian_blur(Ixx, rho)
    Sxy = fast_gaussian_blur(Ixy, rho)
    Syy = fast_gaussian_blur(Iyy, rho)

    magnitude = Sxx + Syy
    max_val = magnitude.max() + 1e-6
    Sxx, Sxy, Syy = Sxx / max_val, Sxy / max_val, Syy / max_val

    if return_extras:
        return Sxx, Sxy, Syy, dict(img_smooth=img_smooth, Ix=Ix_ch, Iy=Iy_ch,
                                    Ixx=Ixx, Ixy=Ixy, Iyy=Iyy)
    return Sxx, Sxy, Syy


def _eig2x2(Sxx, Sxy, Syy):
    """Eigen-decomposition dang dong cho ma tran doi xung 2x2 [[Sxx,Sxy],[Sxy,Syy]].
    Tra ve lambda1>=lambda2>=0 va goc theta cua eigenvector ung voi lambda1
    (huong gradient chinh / phap tuyen canh)."""
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((trace / 2) ** 2 - det, 0.0, None))
    lambda1 = trace / 2 + delta
    lambda2 = np.clip(trace / 2 - delta, 0.0, None)
    theta = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    return lambda1, lambda2, theta


def multiscale_structure_tensor(image, levels=3, base_sigma=1.0, octave_step=1.5,
                                 smoothing_factor=1.0, power_factor=3.0, mode="v1",
                                 collect_levels=False):
    """loss_utils.py:232-313 (v1) / :315-387 (v2), numpy thuan.
    mode='v1' -> blur lai base tensor moi octave; mode='v2' -> tinh lai
    structure tensor tren anh da blur o tung octave ("true multiscale")."""
    H, W, C = image.shape
    base_Sxx, base_Sxy, base_Syy = get_structure_tensor(image, sigma=base_sigma)

    current_smooth = image.copy()
    accum_Sxx = np.zeros((H, W)); accum_Sxy = np.zeros((H, W)); accum_Syy = np.zeros((H, W))
    accum_weight = np.zeros((H, W))
    sigma_accum = 0.0
    levels_info = []

    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        sigma_inc = math.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
        next_smooth = fast_gaussian_blur(current_smooth, sigma_inc)

        band_response = np.sqrt(np.sum((current_smooth - next_smooth) ** 2, axis=-1))
        if i > 0 and smoothing_factor > 0:
            band_response = fast_gaussian_blur(band_response, target_sigma * 2.0)

        integration_rho = target_sigma * 3.0

        if mode == "v1":
            Sxx_i = fast_gaussian_blur(base_Sxx, integration_rho)
            Sxy_i = fast_gaussian_blur(base_Sxy, integration_rho)
            Syy_i = fast_gaussian_blur(base_Syy, integration_rho)
        elif mode == "v2":
            Sxx_i, Sxy_i, Syy_i = get_structure_tensor(next_smooth, sigma=target_sigma,
                                                         rho=integration_rho)
        else:
            raise ValueError(mode)

        trace_i = Sxx_i + Syy_i + 1e-6
        Sxx_n = Sxx_i / trace_i; Sxy_n = Sxy_i / trace_i; Syy_n = Syy_i / trace_i

        weight_i = band_response ** power_factor
        tgt = band_freq ** 2

        accum_Sxx += Sxx_n * weight_i * tgt
        accum_Sxy += Sxy_n * weight_i * tgt
        accum_Syy += Syy_n * weight_i * tgt
        accum_weight += weight_i

        if collect_levels:
            levels_info.append(dict(i=i, target_sigma=target_sigma, band_freq=band_freq,
                                     next_smooth=next_smooth.copy(),
                                     band_response=band_response.copy(),
                                     weight=weight_i.copy(),
                                     Sxx=Sxx_i, Sxy=Sxy_i, Syy=Syy_i))

        current_smooth = next_smooth
        sigma_accum = target_sigma

    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)

    if collect_levels:
        return final_Sxx, final_Sxy, final_Syy, levels_info
    return final_Sxx, final_Sxy, final_Syy


# ============================================================================
# 2. Load anh that + tien ich ve
# ============================================================================

def load_real_crop():
    im = Image.open(SRC_IMAGE).convert("RGB")
    crop = im.crop((10, 395, 452, 640))  # "gt 00001" trong luoi 3x2, xem docstring
    arr = np.asarray(crop).astype(np.float64) / 255.0
    return arr, crop.size  # (H,W,3), (W,H)


def style_panel(ax, title=None, facecolor=BG_CREAM):
    ax.set_facecolor(facecolor)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=6)


def draw_ellipse_field(ax, image, Sxx, Sxy, Syy, step=14, scale=5.0, color=ACCENT_YELLOW,
                        max_axis=None):
    """Ve truong ellipse cau truc tren anh that, cung tinh than voi panel
    'Multiscale Structure-Tensor Field' trong teaser.png. Ellipse duoc dung tu
    eigen-decomposition cua structure tensor: truc dai doc theo huong canh
    (eigenvector ung voi lambda_min), do dai ti le 1/sqrt(lambda)."""
    H, W = Sxx.shape
    ax.imshow(image, extent=[0, W, H, 0])
    lambda1, lambda2, theta = _eig2x2(Sxx, Sxy, Syy)
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    l1v = lambda1[np.ix_(ys, xs)]; l2v = lambda2[np.ix_(ys, xs)]; th = theta[np.ix_(ys, xs)]
    a = 1.0 / np.sqrt(l1v + 1e-3)
    b = 1.0 / np.sqrt(l2v + 1e-3)
    if max_axis is None:
        max_axis = step * 0.85
    norm_max = max(a.max(), b.max(), 1e-6)
    a = a / norm_max * max_axis
    b = b / norm_max * max_axis
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            e = Ellipse((x, y), width=2 * b[yi, xi], height=2 * a[yi, xi],
                        angle=np.degrees(th[yi, xi]) + 90,
                        edgecolor=color, facecolor="none", linewidth=0.9, alpha=0.95)
            ax.add_patch(e)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


# ============================================================================
# 3. Cac hinh
# ============================================================================

def fig01_pyramid(image, levels_info, out):
    fig = plt.figure(figsize=(13, 7.5), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, len(levels_info) + 1, figure=fig,
                            hspace=0.35, wspace=0.15, top=0.86, bottom=0.08,
                            left=0.03, right=0.98)
    fig.suptitle("Laplacian / DoG Scale Space -- loss_utils.py:262-267 (v1/v2 loop)",
                 fontsize=15, fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image); style_panel(ax0, "Input Image\n(anh that, crop 442x245)")
    ax0b = fig.add_subplot(gs[1, 0])
    ax0b.axis("off")
    ax0b.text(0.02, 0.9, "octave_step = 1.5\nbase_sigma = 1.0\nlevels = 3",
              fontsize=10, va="top", family="monospace", color=INK,
              transform=ax0b.transAxes)
    ax0b.text(0.02, 0.35,
              "target_sigma_i = base_sigma\n  * octave_step^i\nband_freq_i = 1/octave_step^i",
              fontsize=9.5, va="top", family="monospace", color=ACCENT_TEAL,
              transform=ax0b.transAxes)

    for j, lvl in enumerate(levels_info):
        axg = fig.add_subplot(gs[0, j + 1])
        axg.imshow(lvl["next_smooth"])
        style_panel(axg, f"Level {lvl['i']}: blur(sigma={lvl['target_sigma']:.2f})")

        axb = fig.add_subplot(gs[1, j + 1])
        im = axb.imshow(lvl["band_response"], cmap=CMAP_FIELD)
        style_panel(axb, f"|band_response| (DoG)\nband_freq={lvl['band_freq']:.3f}")
        fig.colorbar(im, ax=axb, fraction=0.046, pad=0.03)

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig02_gradients(image, extras, out):
    Ix = extras["Ix"]; Iy = extras["Iy"]
    Ix_rgb_sum = np.sum(Ix, axis=-1)
    Iy_rgb_sum = np.sum(Iy, axis=-1)
    fig = plt.figure(figsize=(13, 8), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.15, hspace=0.32,
                            top=0.87, bottom=0.06, left=0.03, right=0.98)
    fig.suptitle("Sobel Gradients per Channel -- loss_utils.py:187-208 (RGB conv2d, groups=C)",
                 fontsize=15, fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(image); style_panel(ax0, "Input (smoothed sigma=1)")
    chans = ["R", "G", "B"]
    for c in range(3):
        axc = fig.add_subplot(gs[0, c + 1])
        im = axc.imshow(Ix[..., c], cmap=CMAP_DIV,
                         norm=Normalize(-np.abs(Ix[..., c]).max(), np.abs(Ix[..., c]).max()))
        style_panel(axc, f"Ix channel {chans[c]}")
        fig.colorbar(im, ax=axc, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(Ix_rgb_sum, cmap=CMAP_DIV,
                     norm=Normalize(-np.abs(Ix_rgb_sum).max(), np.abs(Ix_rgb_sum).max()))
    style_panel(ax1, "sum_c Ix (Di Zenzo, pre-square)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    for c in range(3):
        axc = fig.add_subplot(gs[1, c + 1])
        im = axc.imshow(Iy[..., c], cmap=CMAP_DIV,
                         norm=Normalize(-np.abs(Iy[..., c]).max(), np.abs(Iy[..., c]).max()))
        style_panel(axc, f"Iy channel {chans[c]}")
        fig.colorbar(im, ax=axc, fraction=0.046, pad=0.03)

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig03_structure_tensor(Sxx, Sxy, Syy, image, out):
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    fig = plt.figure(figsize=(13, 6.8), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.32,
                            top=0.85, bottom=0.06, left=0.03, right=0.98)
    fig.suptitle("Single-scale Structure Tensor S = [[Sxx,Sxy],[Sxy,Syy]] "
                 "-- loss_utils.py:198-227 (sigma=rho=1)", fontsize=15,
                 fontweight="bold", color=INK)
    data = [("Sxx", Sxx), ("Sxy", Sxy), ("Syy", Syy), ("trace = Sxx+Syy", trace),
            ("det = Sxx*Syy-Sxy^2", det), ("input image", None)]
    for k, (name, arr) in enumerate(data):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        if arr is None:
            ax.imshow(image); style_panel(ax, name)
        else:
            im = ax.imshow(arr, cmap=CMAP_FIELD)
            style_panel(ax, name)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig04_eigen(Sxx, Sxy, Syy, image, out):
    lambda1, lambda2, theta = _eig2x2(Sxx, Sxy, Syy)
    eta = np.sqrt((lambda1 + 1e-8) / (lambda2 + 1e-8))
    fig = plt.figure(figsize=(13, 8.4), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.35,
                            top=0.87, bottom=0.06, left=0.03, right=0.98)
    fig.suptitle("Eigen-decomposition of S -- lambda_{1,2} = tr/2 +- sqrt((tr/2)^2-det), "
                 "eta = sqrt(lambda1/lambda2)", fontsize=14.5, fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(image); style_panel(ax0, "Input image")
    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(lambda1, cmap=CMAP_FIELD)
    style_panel(ax1, "lambda1 (max, canh manh)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(lambda2, cmap=CMAP_FIELD)
    style_panel(ax2, "lambda2 (min, vung phang)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)

    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(np.degrees(theta), cmap="twilight", vmin=-90, vmax=90)
    style_panel(ax3, "theta (do) -- huong gradient chinh")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)

    ax4 = fig.add_subplot(gs[1, 1])
    step = 10
    H, W = Sxx.shape
    ys, xs = np.arange(step // 2, H, step), np.arange(step // 2, W, step)
    U = np.cos(theta[np.ix_(ys, xs)]); V = np.sin(theta[np.ix_(ys, xs)])
    ax4.imshow(image[..., 0], cmap="gray", alpha=0.5, extent=[0, W, H, 0])
    ax4.quiver(xs, ys, U, V, color=ACCENT_RED, scale=30, width=0.004, pivot="mid")
    ax4.quiver(xs, ys, -U, -V, color=ACCENT_RED, scale=30, width=0.004, pivot="mid")
    style_panel(ax4, "Eigenvector field (huong lambda1)")
    ax4.set_xlim(0, W); ax4.set_ylim(H, 0)

    ax5 = fig.add_subplot(gs[1, 2])
    im = ax5.imshow(np.log10(eta), cmap="viridis")
    style_panel(ax5, "log10(eta), eta=sqrt(lambda1/lambda2)")
    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.03)

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig05_ellipse_field(image, Sxx, Sxy, Syy, out):
    fig = plt.figure(figsize=(13.5, 6.2), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06, top=0.85, bottom=0.05,
                            left=0.02, right=0.98)
    fig.suptitle("2D Structure Analysis -- Input Image -> Structure-Tensor Ellipse Field "
                 "(kieu SADGS/static/teaser.png)", fontsize=15, fontweight="bold", color=INK)
    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(image); style_panel(ax0, "Input Image")
    ax1 = fig.add_subplot(gs[0, 1])
    draw_ellipse_field(ax1, image, Sxx, Sxy, Syy, step=13, color=ACCENT_YELLOW)
    style_panel(ax1, "Structure-Tensor Field (ellipse: truc dai theo canh,\n"
                      "do dai ~ 1/sqrt(lambda))")
    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig06_multiscale(image, levels_info, final_Sxx, final_Sxy, final_Syy, mode_label, out):
    n = len(levels_info)
    fig = plt.figure(figsize=(13, 9), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(3, n + 1, figure=fig, wspace=0.18, hspace=0.35,
                            top=0.88, bottom=0.05, left=0.03, right=0.98)
    fig.savefig  # noqa (placeholder to keep linter calm)
    fig.suptitle(f"Multiscale Structure Tensor ({mode_label}) -- "
                 "loss_utils.py:" + ("232-313" if mode_label == "v1" else "315-387"),
                 fontsize=15, fontweight="bold", color=INK)

    for j, lvl in enumerate(levels_info):
        ax_w = fig.add_subplot(gs[0, j])
        im = ax_w.imshow(lvl["weight"], cmap=CMAP_FIELD)
        style_panel(ax_w, f"level {lvl['i']}: weight=band_resp^3")
        fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.03)

        trace_i = lvl["Sxx"] + lvl["Syy"] + 1e-6
        eta_i = np.sqrt((lvl["Sxx"] + 1e-8) / (lvl["Syy"] + 1e-8)) if False else None
        ax_s = fig.add_subplot(gs[1, j])
        im = ax_s.imshow(lvl["Sxx"] - lvl["Syy"], cmap=CMAP_DIV,
                          norm=Normalize(-np.abs(lvl["Sxx"] - lvl["Syy"]).max(),
                                         np.abs(lvl["Sxx"] - lvl["Syy"]).max()))
        style_panel(ax_s, f"level {lvl['i']}: Sxx-Syy (anisotropy sign)")
        fig.colorbar(im, ax=ax_s, fraction=0.046, pad=0.03)

    ax_img = fig.add_subplot(gs[0, n]); ax_img.imshow(image); style_panel(ax_img, "Input")
    lambda1f, lambda2f, _ = _eig2x2(final_Sxx, final_Sxy, final_Syy)
    ax_f1 = fig.add_subplot(gs[1, n])
    im = ax_f1.imshow(lambda1f, cmap=CMAP_FIELD)
    style_panel(ax_f1, "Aggregated lambda1")
    fig.colorbar(im, ax=ax_f1, fraction=0.046, pad=0.03)

    ax_e = fig.add_subplot(gs[2, :])
    draw_ellipse_field(ax_e, image, final_Sxx, final_Sxy, final_Syy, step=13,
                        color=ACCENT_TEAL if mode_label == "v1" else ACCENT_RED)
    style_panel(ax_e, f"Final aggregated ellipse field ({mode_label})")

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig08_v1_vs_v2(image, S1, S2, out):
    Sxx1, Sxy1, Syy1 = S1; Sxx2, Sxy2, Syy2 = S2
    l1a, l2a, _ = _eig2x2(Sxx1, Sxy1, Syy1)
    l1b, l2b, _ = _eig2x2(Sxx2, Sxy2, Syy2)
    eta_a = np.sqrt((l1a + 1e-8) / (l2a + 1e-8))
    eta_b = np.sqrt((l1b + 1e-8) / (l2b + 1e-8))
    diff = eta_a - eta_b

    fig = plt.figure(figsize=(13.5, 8.4), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.32,
                            top=0.87, bottom=0.05, left=0.03, right=0.98)
    fig.suptitle("Multiscale v1 (blur-base-tensor) vs v2 (recompute-per-level) "
                 "-- loss_utils.py:232-313 vs :315-387", fontsize=14.5,
                 fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(image); style_panel(ax0, "Input")
    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(np.log10(eta_a), cmap="viridis"); style_panel(ax1, "v1: log10(eta)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(np.log10(eta_b), cmap="viridis"); style_panel(ax2, "v2: log10(eta)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)

    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(diff, cmap=CMAP_DIV, norm=Normalize(-np.abs(diff).max(), np.abs(diff).max()))
    style_panel(ax3, "eta_v1 - eta_v2")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)

    ax4 = fig.add_subplot(gs[1, 1])
    draw_ellipse_field(ax4, image, Sxx1, Sxy1, Syy1, step=15, color=ACCENT_TEAL)
    style_panel(ax4, "v1 ellipse field")
    ax5 = fig.add_subplot(gs[1, 2])
    draw_ellipse_field(ax5, image, Sxx2, Sxy2, Syy2, step=15, color=ACCENT_RED)
    style_panel(ax5, "v2 ellipse field")

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


def fig09_eta_heatmap(image, Sxx, Sxy, Syy, out):
    lambda1, lambda2, _ = _eig2x2(Sxx, Sxy, Syy)
    eta = np.sqrt((lambda1 + 1e-8) / (lambda2 + 1e-8))
    q50, q80, q95 = np.percentile(eta, [50, 80, 95])
    high_mask = eta >= q80
    low_mask = eta <= q50

    fig = plt.figure(figsize=(13, 6.5), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.16, top=0.83, bottom=0.06,
                            left=0.03, right=0.98)
    fig.suptitle("Eta anisotropy heatmap (eta=sqrt(lambda1/lambda2), single-scale S) "
                 "-- proxy 2D cho ty so ket cau, XEM CHU Y", fontsize=13.5,
                 fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(eta, cmap="inferno", vmax=q95)
    style_panel(ax0, f"eta map (clip @ p95={q95:.2f})")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(gs[0, 1])
    overlay = np.array(image)
    class_img = np.zeros((*eta.shape, 3))
    class_img[..., 0] = np.where(high_mask, 1.0, overlay[..., 0] * 0.5)
    class_img[..., 1] = np.where(low_mask, 0.6, overlay[..., 1] * 0.5)
    class_img[..., 2] = overlay[..., 2] * 0.3
    ax1.imshow(class_img)
    style_panel(ax1, f"do: eta>=p80 ({q80:.2f}) | xanh la: eta<=p50 ({q50:.2f})")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(eta.ravel(), bins=60, color=ACCENT_TEAL, edgecolor="none")
    for q, name, c in [(q50, "p50", ACCENT_BLUE), (q80, "p80", ACCENT_YELLOW), (q95, "p95", ACCENT_RED)]:
        ax2.axvline(q, color=c, linestyle="--", linewidth=1.5, label=f"{name}={q:.2f}")
    ax2.legend(fontsize=8)
    ax2.set_facecolor(BG_CREAM)
    ax2.set_title("Phan bo eta tren toan anh", fontsize=10, fontweight="bold")

    note = ("Luu y: day la eta 2D per-pixel = sqrt(lambda_max/lambda_min) cua\n"
            "structure tensor anh (thuoc do bat dang huong cuc bo).\n"
            "KHAC voi eta_3ch per-Gaussian trong freq_utils.py:313-348\n"
            "(update_freq_stats_online), vi eta do can truc Gaussian 3D da\n"
            "chieu (axes_2d) -- khong ton tai trong bai toan phan tich anh 2D\n"
            "thuan tuy nay. Nguong tau_high=1.0/tau_low=0.1 cua 3 kenh do KHONG\n"
            "duoc ap dung o day; p50/p80/p95 chi la phan vi thong ke de minh hoa.")
    fig.text(0.02, -0.02, note, fontsize=8, family="monospace", color=INK, va="top")

    fig.savefig(out, dpi=150, facecolor=BG_CREAM, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig10_summary(image, levels_info, final_Sxx, final_Sxy, final_Syy, out):
    fig = plt.figure(figsize=(15, 5.6), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.06, 1.6],
                            wspace=0.12, top=0.78, bottom=0.08, left=0.02, right=0.98)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image); style_panel(ax0, "Input Image")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off"); ax1.set_facecolor(BG_CREAM)
    n = len(levels_info)
    for k, lvl in enumerate(levels_info):
        y0 = 0.08 + k * 0.30
        h = 0.24
        arr = lvl["next_smooth"]
        ax1.imshow(arr, extent=[0.05, 0.95, y0, y0 + h], aspect="auto", zorder=n - k)
        ax1.text(1.0, y0 + h / 2, f"  sigma={lvl['target_sigma']:.2f}", fontsize=7.5,
                  va="center", ha="left", transform=ax1.transData)
    ax1.set_xlim(0, 1.3); ax1.set_ylim(0, 1)
    ax1.set_title("Laplacian Scale Space", fontsize=11, fontweight="bold", color=INK, pad=6)

    ax_arrow = fig.add_subplot(gs[0, 2]); ax_arrow.axis("off")
    ax_arrow.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5),
                       xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))

    ax2 = fig.add_subplot(gs[0, 3])
    draw_ellipse_field(ax2, image, final_Sxx, final_Sxy, final_Syy, step=13,
                        color=ACCENT_YELLOW)
    style_panel(ax2, "Multiscale Structure-Tensor Field")

    fig.text(0.5, 0.965, "2D Structure Analysis", fontsize=20, fontweight="bold",
              color=INK, ha="center")
    fig.text(0.5, 0.90,
             "Input Image -> Laplacian Scale Space -> Multiscale Structure-Tensor Field  "
             "(real numpy re-impl. of loss_utils.py:113-387, real crop of samples_train.png)",
             fontsize=10, color=INK, ha="center")

    fig.savefig(out, dpi=160, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")


# ============================================================================
# main
# ============================================================================

def main():
    print(__doc__)
    print(f"Input anh that: {SRC_IMAGE}")
    print("Cong thuc that: SADGS/utils/loss_utils.py:113-387 "
          "(fast_gaussian_blur, get_structure_tensor_torch, "
          "get_multiscale_structure_tensor_v1, get_multiscale_structure_tensor_v2)")

    image, size = load_real_crop()
    print(f"Da crop anh that: {size[0]}x{size[1]} px tu {os.path.basename(SRC_IMAGE)} "
          f"(vung (10,395)-(452,640))")

    Sxx1, Sxy1, Syy1, extras1 = get_structure_tensor(image, sigma=1.0, rho=1.0,
                                                       return_extras=True)
    print(f"Structure tensor don-scale (sigma=rho=1): Sxx in [{Sxx1.min():.4f},{Sxx1.max():.4f}], "
          f"Syy in [{Syy1.min():.4f},{Syy1.max():.4f}], Sxy in [{Sxy1.min():.4f},{Sxy1.max():.4f}]")

    Sxx_v1, Sxy_v1, Syy_v1, lvl_v1 = multiscale_structure_tensor(
        image, levels=3, base_sigma=1.0, octave_step=1.5, mode="v1", collect_levels=True)
    Sxx_v2, Sxy_v2, Syy_v2, lvl_v2 = multiscale_structure_tensor(
        image, levels=3, base_sigma=1.0, octave_step=1.5, mode="v2", collect_levels=True)
    print("Da chay xong multiscale v1 va v2 (3 level, base_sigma=1.0, octave_step=1.5).")

    fig01_pyramid(image, lvl_v2, os.path.join(OUT_DIR, "real_st_01_pyramid.png"))
    fig02_gradients(image, extras1, os.path.join(OUT_DIR, "real_st_02_gradients.png"))
    fig03_structure_tensor(Sxx1, Sxy1, Syy1, image,
                            os.path.join(OUT_DIR, "real_st_03_structure_tensor.png"))
    fig04_eigen(Sxx1, Sxy1, Syy1, image, os.path.join(OUT_DIR, "real_st_04_eigen_decomp.png"))
    fig05_ellipse_field(image, Sxx1, Sxy1, Syy1,
                         os.path.join(OUT_DIR, "real_st_05_ellipse_field.png"))
    fig06_multiscale(image, lvl_v1, Sxx_v1, Sxy_v1, Syy_v1, "v1",
                      os.path.join(OUT_DIR, "real_st_06_multiscale_v1.png"))
    fig06_multiscale(image, lvl_v2, Sxx_v2, Sxy_v2, Syy_v2, "v2",
                      os.path.join(OUT_DIR, "real_st_07_multiscale_v2.png"))
    fig08_v1_vs_v2(image, (Sxx_v1, Sxy_v1, Syy_v1), (Sxx_v2, Sxy_v2, Syy_v2),
                   os.path.join(OUT_DIR, "real_st_08_v1_vs_v2_compare.png"))
    fig09_eta_heatmap(image, Sxx1, Sxy1, Syy1,
                       os.path.join(OUT_DIR, "real_st_09_eta_heatmap.png"))
    fig10_summary(image, lvl_v2, Sxx_v2, Sxy_v2, Syy_v2,
                  os.path.join(OUT_DIR, "real_st_10_summary_pipeline.png"))

    print("\nHoan tat: 10 hinh PNG that da duoc ghi vao", OUT_DIR)


if __name__ == "__main__":
    main()
