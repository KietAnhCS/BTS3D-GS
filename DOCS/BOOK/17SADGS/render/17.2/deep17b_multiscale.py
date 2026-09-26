"""
deep17b_multiscale.py
======================
Hinh minh hoa THEM cho muc 17.2 "Multiscale structure tensor -- hai bien the
v1/v2" cua sach SADGS, di SAU vao 3 khoang trong ma ban md hien chi khang
dinh bang loi (khong co hinh so lieu that):

  (1) Bang so sanh chi phi v1 vs v2 trong md chi la mo ta bang chu ("1 lan
      Sobel + L lan blur" vs "L lan Sobel + L lan blur") -- khong co so do
      that. Hinh 01 o day CHAY that ca hai bien the tren CUNG mot anh that,
      voi st_levels = 2,3,4,6, do (a) wall-clock that (time.perf_counter),
      (b) dem SO LAN GOI THAT ham sobel_channel/fast_gaussian_blur (khong
      phai cong thuc ly thuyet) trong ca hai bien the, va (c) do do lech
      that giua trace(v1) va trace(v2) (mean absolute difference) theo so
      level.
  (2) md chi noi octave_step (r) va power_factor (p) "dieu khien do sac net
      cua winner-take-all giua cac thang" nhung chi minh hoa MOT bo tham so
      co dinh (r=1.5, p=3.0). Hinh 02 quet that r trong {1.2,1.5,2.0} va p
      trong {1,3,6} tren CUNG anh crop that, xuat luoi 3x3 ban do trace da
      gop that.
  (3) md khang dinh "chia cho tr_i truoc khi gop" de moi octave chi dong gop
      HUONG, con DO LON hoan toan do f_i^2 (trong so wi) ap dat -- day la
      mot dang thuc dai so: vi (Sxx_i+Syy_i)/tr_i = 1 (xap xi, bo qua eps),
      nen trace cuoi = sum_i(w_i * f_i^2) / sum_i(w_i), KHONG PHU THUOC noi
      dung anh ngoai tru qua trong so w_i. Hinh 03 tinh trace nay bang HAI
      cach doc lap tren cung anh that -- (i) chay thuat toan that roi lay
      Sxx+Syy, (ii) cong thuc dai so weighted-average f_i^2 -- va ve ban do
      phan du (residual) giua hai cach, gan bang 0 o moi pixel, la bang
      chung so hoc cho dang thuc.

Cong thuc THAT duoc chep lai tu SADGS/utils/loss_utils.py (numpy thuan,
KHONG torch -- torch khong duoc cai trong moi truong nay). Trich dan dong
cu the (giong DOCS/BOOK/adc_figures/real_structure_analysis.py, doc lai o
day de file nay TU CHAY DOC LAP, khong import file kia):
  - loss_utils.py:113-168  fast_gaussian_blur
  - loss_utils.py:170-230  get_structure_tensor_torch
  - loss_utils.py:232-313  get_multiscale_structure_tensor_v1
        (blur LAI base tensor moi octave, dong 282-284)
  - loss_utils.py:315-387  get_multiscale_structure_tensor_v2
        ("true multiscale": tinh lai structure tensor tren anh da blur o
        tung octave, dong 357)
  - loss_utils.py:296-298  comment goc "We want the final sum to have
        Trace = freq^2" -- day chinh la dang thuc duoc kiem chung o Hinh 03.

Input anh that
--------------
DOCS/assets/samples_drjohnson.png la luoi 3x2 (render vs ground-truth) cua
scene "drjohnson" (Tanks&Temples-style). Script nay CAT o toa do pixel co
dinh (924,381)-(1365,671) -- dung o "gt IMG_6500" trong luoi -- ra mot anh
that 441x290 px cua mot tu sach go co canh cua kinh loi (mullion cheo),
nam vang tren canh cua, va khung anh treo tuong -- nhieu chi tiet nhieu
thang (tu go/tuong phang, luoi kinh nhieu tan so cao, chu ky lap deu dan
cua cac o kinh) rat phu hop de minh hoa do nhay tham so multiscale. Day LA
mot anh khac (scene khac, vung crop khac) so voi crop (10,395)-(452,640)
cua samples_train.png ma real_structure_analysis.py da dung -- khong trung
lap du lieu voi cac hinh real_st_*.png da co.

Khong dung np.random cho bat ky so lieu khoa hoc nao trong bai. Chay:
  python DOCS/BOOK/adc_figures/deep17b_multiscale.py
"""
import math
import os
import time

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from PIL import Image

# ----------------------------------------------------------------------------
# Duong dan
# ----------------------------------------------------------------------------
HERE = __file__
BOOK_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(BOOK_DIR, "..", "..", ".."))
SRC_IMAGE = os.path.join(REPO_ROOT, "DOCS", "assets", "samples_drjohnson.png")
OUT_DIR = BOOK_DIR

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
# 1. Ham loi -- chep dung cong thuc SADGS/utils/loss_utils.py (numpy thuan),
#    co instrument dem so lan goi that (khong phai uoc luong ly thuyet) de
#    do chi phi o Hinh 01.
# ============================================================================

class OpCounter:
    """Dem SO LAN GOI THAT cac ham chi phi chinh trong pipeline, de do chi
    phi v1 vs v2 bang so lieu that thay vi cong thuc O(...) uoc luong."""
    def __init__(self):
        self.sobel_calls = 0      # so lan goi sobel_channel (1 lan = 1 kenh anh)
        self.blur_calls = 0       # so lan goi fast_gaussian_blur tren 1 mang 2D/kenh
        self.blur_pixel_ops = 0   # tong so pixel*kenh da di qua Gaussian blur


def fast_gaussian_blur(img, sigma, counter=None):
    """loss_utils.py:113-168 fast_gaussian_blur, numpy thuan (scipy Gaussian
    tuong duong chinh xac voi nhanh sigma<=2 cua ban torch, xem
    real_structure_analysis.py dong 28-36 cho phan tich chi tiet)."""
    if sigma < 0.01:
        if counter is not None:
            counter.blur_calls += 1
        return img.copy()
    if img.ndim == 2:
        if counter is not None:
            counter.blur_calls += 1
            counter.blur_pixel_ops += img.shape[0] * img.shape[1]
        return ndimage.gaussian_filter(img, sigma=sigma, mode="reflect")
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = ndimage.gaussian_filter(img[..., c], sigma=sigma, mode="reflect")
        if counter is not None:
            counter.blur_calls += 1
            counter.blur_pixel_ops += img.shape[0] * img.shape[1]
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel_channel(img_c, counter=None):
    """loss_utils.py:187-196: F.conv2d(..., padding=1) = correlation, zero-pad."""
    Ix = ndimage.correlate(img_c, SOBEL_X, mode="constant", cval=0.0)
    Iy = ndimage.correlate(img_c, SOBEL_Y, mode="constant", cval=0.0)
    if counter is not None:
        counter.sobel_calls += 1
    return Ix, Iy


def get_structure_tensor(image, sigma=1.0, rho=1.0, counter=None):
    """loss_utils.py:170-230, numpy thuan. Tra ve Sxx,Sxy,Syy da chuan hoa
    theo max(Sxx+Syy) toan cuc (dong 219-227)."""
    H, W, C = image.shape
    img_smooth = fast_gaussian_blur(image, sigma, counter)

    Ixx = np.zeros((H, W)); Ixy = np.zeros((H, W)); Iyy = np.zeros((H, W))
    for c in range(C):
        Ix, Iy = sobel_channel(img_smooth[..., c], counter)
        Ixx += Ix * Ix
        Ixy += Ix * Iy
        Iyy += Iy * Iy

    Sxx = fast_gaussian_blur(Ixx, rho, counter)
    Sxy = fast_gaussian_blur(Ixy, rho, counter)
    Syy = fast_gaussian_blur(Iyy, rho, counter)

    magnitude = Sxx + Syy
    max_val = magnitude.max() + 1e-6
    Sxx, Sxy, Syy = Sxx / max_val, Sxy / max_val, Syy / max_val
    return Sxx, Sxy, Syy


def multiscale_structure_tensor(image, levels=4, base_sigma=1.0, octave_step=1.5,
                                 smoothing_factor=1.0, power_factor=3.0, mode="v1",
                                 counter=None, collect_levels=False):
    """loss_utils.py:232-313 (v1) / :315-387 (v2), numpy thuan. Neu counter
    duoc truyen vao (OpCounter), dem SO LAN GOI THAT sobel_channel va
    fast_gaussian_blur trong toan bo qua trinh -- day la so lieu chi phi
    THAT dung o Hinh 01, khong phai cong thuc ly thuyet."""
    H, W, C = image.shape
    base_Sxx, base_Sxy, base_Syy = get_structure_tensor(image, sigma=base_sigma, counter=counter)

    current_smooth = image.copy()
    accum_Sxx = np.zeros((H, W)); accum_Sxy = np.zeros((H, W)); accum_Syy = np.zeros((H, W))
    accum_weight = np.zeros((H, W))
    sigma_accum = 0.0
    levels_info = []

    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        sigma_inc = math.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
        next_smooth = fast_gaussian_blur(current_smooth, sigma_inc, counter)

        band_response = np.sqrt(np.sum((current_smooth - next_smooth) ** 2, axis=-1))
        if i > 0 and smoothing_factor > 0:
            band_response = fast_gaussian_blur(band_response, target_sigma * 2.0, counter)

        integration_rho = target_sigma * 3.0

        if mode == "v1":
            Sxx_i = fast_gaussian_blur(base_Sxx, integration_rho, counter)
            Sxy_i = fast_gaussian_blur(base_Sxy, integration_rho, counter)
            Syy_i = fast_gaussian_blur(base_Syy, integration_rho, counter)
        elif mode == "v2":
            Sxx_i, Sxy_i, Syy_i = get_structure_tensor(next_smooth, sigma=target_sigma,
                                                         rho=integration_rho, counter=counter)
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
                                     weight=weight_i.copy(), trace_i=trace_i.copy()))

        current_smooth = next_smooth
        sigma_accum = target_sigma

    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)

    if collect_levels:
        return final_Sxx, final_Sxy, final_Syy, accum_weight, levels_info
    return final_Sxx, final_Sxy, final_Syy, accum_weight


def load_real_crop():
    """Anh that: crop (924,381)-(1365,671) cua samples_drjohnson.png -- o
    'gt IMG_6500' trong luoi 3x2, tu sach go + cua kinh loi mullion cheo,
    KHAC anh/vung crop ma real_structure_analysis.py da dung (samples_train,
    (10,395)-(452,640))."""
    im = Image.open(SRC_IMAGE).convert("RGB")
    crop = im.crop((924, 381, 1365, 671))
    arr = np.asarray(crop).astype(np.float64) / 255.0
    return arr, crop.size


def style_panel(ax, title=None, facecolor=BG_CREAM, fontsize=10):
    ax.set_facecolor(facecolor)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight="bold", color=INK, pad=5)


# ============================================================================
# 2. Hinh 01 -- chi phi v1 vs v2 (wall-clock + so lan goi that) va do lech
#    trace(v1)-trace(v2), theo st_levels that
# ============================================================================

def fig01_cost_divergence(image, out):
    levels_list = [2, 3, 4, 6]
    times_v1, times_v2 = [], []
    sobel_v1, sobel_v2 = [], []
    blur_v1, blur_v2 = [], []
    diverg = []  # mean |trace_v1 - trace_v2|

    for L in levels_list:
        c1 = OpCounter()
        t0 = time.perf_counter()
        Sxx1, Sxy1, Syy1, _ = multiscale_structure_tensor(
            image, levels=L, base_sigma=1.0, octave_step=1.5, power_factor=3.0,
            mode="v1", counter=c1)
        t1 = time.perf_counter()
        times_v1.append(t1 - t0)
        sobel_v1.append(c1.sobel_calls)
        blur_v1.append(c1.blur_calls)

        c2 = OpCounter()
        t0 = time.perf_counter()
        Sxx2, Sxy2, Syy2, _ = multiscale_structure_tensor(
            image, levels=L, base_sigma=1.0, octave_step=1.5, power_factor=3.0,
            mode="v2", counter=c2)
        t1 = time.perf_counter()
        times_v2.append(t1 - t0)
        sobel_v2.append(c2.sobel_calls)
        blur_v2.append(c2.blur_calls)

        trace1 = Sxx1 + Syy1
        trace2 = Sxx2 + Syy2
        diverg.append(float(np.mean(np.abs(trace1 - trace2))))

        print(f"[cost] levels={L}: v1 time={times_v1[-1]*1000:.1f}ms "
              f"sobel_calls={sobel_v1[-1]} blur_calls={blur_v1[-1]}  |  "
              f"v2 time={times_v2[-1]*1000:.1f}ms sobel_calls={sobel_v2[-1]} "
              f"blur_calls={blur_v2[-1]}  |  mean|trace1-trace2|={diverg[-1]:.5f}")

    fig = plt.figure(figsize=(14, 4.6), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32, top=0.82, bottom=0.14,
                            left=0.06, right=0.98)
    fig.suptitle("v1 vs v2 -- chi phi THAT (wall-clock + so lan goi Sobel/blur) va do lech "
                 "trace, theo st_levels -- loss_utils.py:232-387", fontsize=13.5,
                 fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(levels_list, [t * 1000 for t in times_v1], "o-", color=ACCENT_TEAL, label="v1")
    ax0.plot(levels_list, [t * 1000 for t in times_v2], "o-", color=ACCENT_RED, label="v2")
    ax0.set_xlabel("st_levels (L)"); ax0.set_ylabel("wall-clock (ms)")
    ax0.set_title("Thoi gian chay that", fontsize=10.5, fontweight="bold")
    ax0.legend(fontsize=9); ax0.set_facecolor(BG_CREAM)
    ax0.set_xticks(levels_list)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(levels_list, sobel_v1, "s--", color=ACCENT_TEAL, label="v1: sobel_calls")
    ax1.plot(levels_list, sobel_v2, "s-", color=ACCENT_RED, label="v2: sobel_calls")
    ax1.plot(levels_list, blur_v1, "^--", color=ACCENT_BLUE, label="v1: blur_calls")
    ax1.plot(levels_list, blur_v2, "^-", color=ACCENT_YELLOW, label="v2: blur_calls")
    ax1.set_xlabel("st_levels (L)"); ax1.set_ylabel("so lan goi (dem that)")
    ax1.set_title("So lan goi sobel_channel / fast_gaussian_blur", fontsize=10.5,
                  fontweight="bold")
    ax1.legend(fontsize=7.5); ax1.set_facecolor(BG_CREAM)
    ax1.set_xticks(levels_list)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(levels_list, diverg, "o-", color=ACCENT_RED)
    ax2.set_xlabel("st_levels (L)"); ax2.set_ylabel("mean |trace_v1 - trace_v2|")
    ax2.set_title("Do lech that v1 vs v2 (trung binh toan anh)", fontsize=10.5,
                  fontweight="bold")
    ax2.set_facecolor(BG_CREAM)
    ax2.set_xticks(levels_list)
    for L, d in zip(levels_list, diverg):
        ax2.annotate(f"{d:.4f}", (L, d), textcoords="offset points", xytext=(0, 6),
                     fontsize=8, ha="center", color=INK)

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")
    return dict(levels_list=levels_list, times_v1=times_v1, times_v2=times_v2,
                sobel_v1=sobel_v1, sobel_v2=sobel_v2, blur_v1=blur_v1, blur_v2=blur_v2,
                diverg=diverg)


# ============================================================================
# 3. Hinh 02 -- do nhay octave_step (r) va power_factor (p): luoi 3x3
# ============================================================================

def fig02_rp_sensitivity(image, out):
    r_list = [1.2, 1.5, 2.0]
    p_list = [1, 3, 6]

    fig = plt.figure(figsize=(13.5, 13.2), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.08, hspace=0.24,
                            top=0.85, bottom=0.04, left=0.06, right=0.98)
    fig.suptitle("Do nhay that: quet octave_step r va power_factor p\n"
                 "get_multiscale_structure_tensor_v1, loss_utils.py:232-313 (levels=4, base_sigma=1)",
                 fontsize=13, fontweight="bold", color=INK, y=0.985)

    traces = {}
    for ri, r in enumerate(r_list):
        for pi, p in enumerate(p_list):
            Sxx, Sxy, Syy, _ = multiscale_structure_tensor(
                image, levels=4, base_sigma=1.0, octave_step=r, power_factor=p, mode="v1")
            trace = Sxx + Syy
            traces[(r, p)] = trace
            ax = fig.add_subplot(gs[ri, pi])
            im = ax.imshow(trace, cmap=CMAP_FIELD, vmin=0, vmax=1.0)
            style_panel(ax, f"r={r}, p={p}\nmean_tr={trace.mean():.4f}", fontsize=9)
            if pi == 0:
                ax.text(-0.08, 0.5, f"r={r}", transform=ax.transAxes, fontsize=11,
                        fontweight="bold", color=ACCENT_TEAL, rotation=90,
                        va="center", ha="center")
            if ri == 0:
                ax.set_xlabel("")
            print(f"[rp-sweep] r={r} p={p}: trace mean={trace.mean():.5f} "
                  f"std={trace.std():.5f}")

    fig.text(0.5, 0.905,
              "r nho -> nhieu octave gan nhau -> tron muot theo thang; r lon -> octave thua, "
              "nhay cam hon voi 1-2 thang trội.  p lon -> winner-take-all (chi octave co\n"
              "band_response manh nhat thong tri); p nho -> cac octave tron deu hon.",
              fontsize=9.5, ha="center", color=INK)

    fig.savefig(out, dpi=145, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")
    return traces


# ============================================================================
# 4. Hinh 03 -- kiem chung dang thuc: trace_final = weighted-average(f_i^2)
# ============================================================================

def fig03_trace_identity_check(image, out):
    L = 4
    Sxx, Sxy, Syy, accum_weight, levels_info = multiscale_structure_tensor(
        image, levels=L, base_sigma=1.0, octave_step=1.5, power_factor=3.0,
        mode="v1", collect_levels=True)
    actual_trace = Sxx + Syy  # (i) trace THAT tu thuat toan chay day du

    # (ii) cong thuc dai so doc lap: vi Sxx_i/tr_i + Syy_i/tr_i = tr_i/tr_i ~ 1
    # (bo qua eps=1e-6 trong tr_i = Sxx_i+Syy_i+1e-6), nen
    #   trace_final = sum_i w_i * f_i^2 * 1 / sum_i w_i
    # KHONG can biet gia tri Sxx_i,Syy_i cu the -- chi can w_i va f_i.
    numer = np.zeros_like(actual_trace)
    for lvl in levels_info:
        numer += lvl["weight"] * (lvl["band_freq"] ** 2)
    analytic_trace = numer / (accum_weight + 1e-6)

    residual = actual_trace - analytic_trace
    max_abs_res = float(np.max(np.abs(residual)))
    mean_abs_res = float(np.mean(np.abs(residual)))
    rel_res = max_abs_res / (float(np.max(np.abs(actual_trace))) + 1e-12)
    print(f"[identity] max|residual|={max_abs_res:.3e}  mean|residual|={mean_abs_res:.3e} "
          f"  relative(max/max|trace|)={rel_res:.3e}")

    fig = plt.figure(figsize=(14, 7.2), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.22, hspace=0.34,
                            top=0.86, bottom=0.07, left=0.04, right=0.98)
    fig.suptitle(
        r"Kiem chung dang thuc: $\mathrm{tr}(\hat S)=\dfrac{\sum_i w_i f_i^2}{\sum_i w_i}$"
        "  (chia cho tr_i truoc khi gop => octave chi dong gop huong) -- "
        "loss_utils.py:296-298,309-311", fontsize=13, fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(image); style_panel(ax0, "Input that (crop)")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(actual_trace, cmap=CMAP_FIELD, vmin=0, vmax=1.0)
    style_panel(ax1, "(i) trace THAT tu thuat toan\n(Sxx+Syy sau khi chay day du)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(analytic_trace, cmap=CMAP_FIELD, vmin=0, vmax=1.0)
    style_panel(ax2, "(ii) cong thuc dai so\nweighted-avg(f_i^2) tu w_i,f_i")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

    ax3 = fig.add_subplot(gs[1, 0])
    vlim = max(1e-6, np.abs(residual).max())
    im3 = ax3.imshow(residual, cmap=CMAP_DIV, norm=Normalize(-vlim, vlim))
    style_panel(ax3, f"Residual (i)-(ii)\nmax|res|={max_abs_res:.2e}")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residual.ravel(), bins=60, color=ACCENT_TEAL, edgecolor="none")
    ax4.axvline(0, color=ACCENT_RED, linestyle="--", linewidth=1.3)
    ax4.set_facecolor(BG_CREAM)
    ax4.set_title(f"Phan bo residual\nmean|res|={mean_abs_res:.2e}", fontsize=10,
                  fontweight="bold", color=INK)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off"); ax5.set_facecolor(BG_CREAM)
    txt = (
        "Vi sao dang thuc dung:\n"
        f"  tr_i = Sxx_i+Syy_i+1e-6 (mau so chuan hoa)\n"
        f"  Sxx_i/tr_i + Syy_i/tr_i = tr_i/tr_i (approx 1)\n"
        f"  => trace(octave i sau chuan hoa) approx 1\n"
        f"  => trace_final = sum_i w_i*f_i^2*1 / sum_i w_i\n"
        f"     = weighted-average cua f_i^2, trong so w_i\n\n"
        f"levels={L}, r=1.5\n"
        f"f_i^2 = " + ", ".join(f"{(1/1.5**i)**2:.3f}" for i in range(L)) + "\n\n"
        f"max|residual| = {max_abs_res:.3e}\n"
        f"mean|residual| = {mean_abs_res:.3e}\n"
        f"(sai so con lai chi den tu eps=1e-6\n"
        f" trong tr_i, khong phai loi cong thuc)"
    )
    ax5.text(0.02, 0.98, txt, fontsize=9.3, va="top", ha="left", family="monospace",
             color=INK, transform=ax5.transAxes)

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")
    return dict(max_abs_res=max_abs_res, mean_abs_res=mean_abs_res)


# ============================================================================
# main
# ============================================================================

def main():
    print(__doc__)
    print(f"Input anh that: {SRC_IMAGE}")
    image, size = load_real_crop()
    print(f"Da crop anh that: {size[0]}x{size[1]} px tu {os.path.basename(SRC_IMAGE)} "
          f"(vung (924,381)-(1365,671))")

    fig01_cost_divergence(image, os.path.join(OUT_DIR, "deep17b_01_cost_divergence.png"))
    fig02_rp_sensitivity(image, os.path.join(OUT_DIR, "deep17b_02_rp_sensitivity.png"))
    fig03_trace_identity_check(image, os.path.join(OUT_DIR, "deep17b_03_trace_identity.png"))

    print("\nHoan tat: 3 hinh PNG that da duoc ghi vao", OUT_DIR)


if __name__ == "__main__":
    main()
