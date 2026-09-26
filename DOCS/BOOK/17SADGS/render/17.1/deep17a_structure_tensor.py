"""
deep17a_structure_tensor.py
============================
Ba hinh "dao sau" (deep-dive) bo sung cho muc 17.1 Structure tensor, lap 3
khoang trong ma real_structure_analysis.py CHUA chung minh bang so lieu that
(chi khang dinh bang loi van):

  1. deep17a_01_iso_luminant_dizenzo.png
     Vi sao Di Zenzo cong nang luong 3 kenh RGB thay vi dung luminance: tim
     mot diem canh THAT trong anh crop noi ty le grad_dizenzo/grad_luminance
     lon nhat, roi in ra hai con so that de so sanh truc tiep.

  2. deep17a_02_rho_integration_window.png
     Vai tro cua rho (cua so tich hop, buoc 4): ve CUNG Ixx,Ixy,Iyy TRUOC
     (rho~0, tuc chi san pham diem anh don, ma tran hang-1 "gia tao coherent")
     va SAU khi lam mo G_rho voi hai gia tri rho khac nhau, tren cung mot anh
     that -- kem coherence trung binh de dinh luong: Grho bien uoc luong
     pointwise gion (luon "coherent" gia tao) thanh uoc luong khong gian
     THUC SU phan biet duoc canh don huong voi goc/texture da huong.

  3. deep17a_03_brightness_normalization.png
     Buoc 5 (chuan hoa theo max(Sxx+Syy)): render CUNG mot crop that o hai
     muc do sang/tuong phan khac nhau (x0.5 va x1.5), chung minh bang so
     Sxx/Sxy/Syy TRUOC chuan hoa rat khac nhau nhung SAU chuan hoa gan nhu
     giong het (tru vung bi clip saturate).

Input anh that
--------------
DOCS/assets/samples_drjohnson.png la luoi 3x2 (render vs ground-truth) cua
scene Tanks&Temples "drjohnson" (CHUA dung trong real_structure_analysis.py,
file do dung samples_train.png). Script nay CAT panel duoi-trai "gt IMG_6292"
o toa do pixel co dinh (11,381)-(451,669) => anh 440x288, mot phong that voi
tuong xanh la nhat, cua go trang, ghe go, dan suoi va binh chua chay mau do
-- nhieu bien the mau/do sang thich hop de kiem tra ca 3 khoang trong tren.
  Anh goc: DOCS/assets/samples_drjohnson.png (1374x681, luoi 3x2)
  Anh dung: crop (11,381)-(451,669) => 440x288 px

Cong thuc THAT duoc chep lai tu SADGS/utils/loss_utils.py:170-230
(get_structure_tensor_torch) -- xem docstring tung ham numpy ben duoi de
doi chieu tung dong. Khong dung torch (repo nay khong cai torch), toan bo
bang numpy/scipy/PIL thuan, giong 100% phong cach real_structure_analysis.py.
Khong np.random cho bat ky so lieu khoa hoc nao (chi la anh THAT / phep bien
doi tuyen tinh THAT tren anh THAT).

Chay:  python DOCS/BOOK/adc_figures/deep17a_structure_tensor.py
"""
import os
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
from PIL import Image

# ----------------------------------------------------------------------------
# Duong dan
# ----------------------------------------------------------------------------
HERE = __file__
BOOK_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.abspath(os.path.join(BOOK_DIR, "..", "..", ".."))
SRC_IMAGE = os.path.join(REPO_ROOT, "DOCS", "assets", "samples_drjohnson.png")
OUT_DIR = BOOK_DIR

# Bang mau flat-design dong nhat voi real_structure_analysis.py
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
    """loss_utils.py:113-168 fast_gaussian_blur (nhanh cho sigma<=2.0, xem
    docstring real_structure_analysis.py de biet danh gia tuong duong voi
    nhanh downsample o sigma lon). img: (H,W,C) hoac (H,W) float64."""
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
    """loss_utils.py:187-196: F.conv2d(..., padding=1, groups=C) = correlation
    voi zero-pad, tung kenh doc lap."""
    Ix = ndimage.correlate(img_c, SOBEL_X, mode="constant", cval=0.0)
    Iy = ndimage.correlate(img_c, SOBEL_Y, mode="constant", cval=0.0)
    return Ix, Iy


def raw_dizenzo(image, sigma=1.0):
    """loss_utils.py:177-208 (buoc 1-3, TRUOC tich hop rho o buoc 4). Tra ve
    img_smooth, Ix_ch, Iy_ch (H,W,C) va Ixx,Ixy,Iyy (H,W) = tong binh phuong/
    tich dao ham qua 3 kenh RGB (Di Zenzo, dong 206-208), CHUA lam mo G_rho."""
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
    return img_smooth, Ix_ch, Iy_ch, Ixx, Ixy, Iyy


def integrate_rho(Ixx, Ixy, Iyy, rho):
    """loss_utils.py:210-217 (buoc 4): G_rho * (Ixx,Ixy,Iyy). rho=0 tra ve
    nguyen ban (khong tich hop) de lam panel 'truoc khi blur'."""
    if rho < 0.01:
        return Ixx.copy(), Ixy.copy(), Iyy.copy()
    return (fast_gaussian_blur(Ixx, rho), fast_gaussian_blur(Ixy, rho),
            fast_gaussian_blur(Iyy, rho))


def normalize_batch(Sxx, Sxy, Syy):
    """loss_utils.py:219-227 (buoc 5): chia ca 3 kenh cho max(Sxx+Syy)+1e-6.
    Tra ve them max_val de bao cao so lieu that."""
    magnitude = Sxx + Syy
    max_val = magnitude.max() + 1e-6
    return Sxx / max_val, Sxy / max_val, Syy / max_val, max_val


def get_structure_tensor(image, sigma=1.0, rho=1.0):
    """Pipeline day du loss_utils.py:170-230, dung lai 2 ham tren."""
    _, _, _, Ixx, Ixy, Iyy = raw_dizenzo(image, sigma)
    Sxx, Sxy, Syy = integrate_rho(Ixx, Ixy, Iyy, rho)
    Sxx, Sxy, Syy, max_val = normalize_batch(Sxx, Sxy, Syy)
    return Sxx, Sxy, Syy, max_val


def eig2x2(Sxx, Sxy, Syy):
    """Cong thuc dong cho ma tran doi xung 2x2, dung lai freq_utils.py:325-329."""
    trace = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((trace / 2) ** 2 - det, 0.0, None))
    lambda1 = trace / 2 + delta
    lambda2 = np.clip(trace / 2 - delta, 0.0, None)
    return lambda1, lambda2


def coherence(lambda1, lambda2):
    denom = lambda1 + lambda2 + 1e-8
    return ((lambda1 - lambda2) / denom) ** 2


# ============================================================================
# 2. Load anh that + tien ich ve
# ============================================================================

def load_real_crop():
    im = Image.open(SRC_IMAGE).convert("RGB")
    # panel duoi-trai "gt IMG_6292" trong luoi 3x2, xem docstring dau file
    crop = im.crop((11, 381, 451, 669))
    arr = np.asarray(crop).astype(np.float64) / 255.0
    return arr, crop.size  # (H,W,3), (W,H)


def style_panel(ax, title=None, facecolor=BG_CREAM):
    ax.set_facecolor(facecolor)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=INK, pad=5)


# ============================================================================
# 3. Hinh 1 -- vi sao cong 3 kenh RGB thay vi luminance (Di Zenzo)
# ============================================================================

def fig01_iso_luminant(image, out):
    H, W, C = image.shape
    img_smooth, Ix_ch, Iy_ch, Ixx, Ixy, Iyy = raw_dizenzo(image, sigma=1.0)

    # Luminance ITU-R BT.709, tinh tren CUNG anh da lam mo sigma=1 de so sanh
    # cong bang (cung buoc 1) -- roi lay Sobel giong het buoc 2.
    lum = 0.2126 * img_smooth[..., 0] + 0.7152 * img_smooth[..., 1] + 0.0722 * img_smooth[..., 2]
    Lx, Ly = sobel_channel(lum)
    grad_lum = np.sqrt(Lx ** 2 + Ly ** 2)
    grad_dizenzo = np.sqrt(Ixx + Iyy)  # buoc 3 THO, truoc tich hop rho o buoc 4

    # Tim diem canh THAT ("iso-luminant-like") noi ty le dizenzo/luminance
    # lon nhat: loai bien anh va loai vung dizenzo qua yeu (nhieu nen) bang
    # nguong percentile 70 tren chinh grad_dizenzo.
    margin = 6
    mask = np.zeros((H, W), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    thresh = np.percentile(grad_dizenzo[mask], 70)
    mask &= grad_dizenzo >= thresh
    ratio = np.where(mask, grad_dizenzo / (grad_lum + 1e-3), -1.0)
    py, px = np.unravel_index(np.argmax(ratio), ratio.shape)

    v_grad_lum = grad_lum[py, px]
    v_grad_dz = grad_dizenzo[py, px]
    v_ratio = ratio[py, px]
    ch_names = ["R", "G", "B"]
    ch_grad = [np.sqrt(Ix_ch[py, px, c] ** 2 + Iy_ch[py, px, c] ** 2) for c in range(3)]
    rgb_at = image[py, px]

    fig = plt.figure(figsize=(13.5, 8.6), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.22, hspace=0.35,
                            top=0.87, bottom=0.06, left=0.03, right=0.98)
    fig.suptitle("Vi sao cong nang luong 3 kenh RGB (Di Zenzo) thay vi luminance -- "
                 "loss_utils.py:198-208 vs gradient luminance (khong nam trong SADGS)",
                 fontsize=13.5, fontweight="bold", color=INK)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.add_patch(Circle((px, py), 7, edgecolor=ACCENT_YELLOW, facecolor="none", linewidth=2.2))
    style_panel(ax0, "Input that (crop 440x288) + diem tim duoc")

    half = 22
    y0, y1 = max(0, py - half), min(H, py + half)
    x0, x1 = max(0, px - half), min(W, px + half)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(image[y0:y1, x0:x1], extent=[x0, x1, y1, y0])
    ax1.add_patch(Circle((px, py), 3, edgecolor=ACCENT_YELLOW, facecolor="none", linewidth=2.2))
    style_panel(ax1, f"Zoom quanh ({px},{py})\nRGB tai diem = ({rgb_at[0]:.2f},{rgb_at[1]:.2f},{rgb_at[2]:.2f})")

    ax2 = fig.add_subplot(gs[0, 2])
    bars_x = ["|grad R|", "|grad G|", "|grad B|", "grad\nluminance", "grad\nDi Zenzo\n(sqrt(Ixx+Iyy))"]
    bars_v = [ch_grad[0], ch_grad[1], ch_grad[2], v_grad_lum, v_grad_dz]
    bars_c = [ACCENT_RED, ACCENT_TEAL, ACCENT_BLUE, "#888888", ACCENT_YELLOW]
    ax2.bar(bars_x, bars_v, color=bars_c, edgecolor=INK, linewidth=0.8)
    ax2.set_facecolor(BG_CREAM)
    for i, v in enumerate(bars_v):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax2.set_title(f"Tai diem ({px},{py}): ty le dizenzo/luminance = {v_ratio:.1f}x",
                  fontsize=10, fontweight="bold", color=INK)
    ax2.tick_params(axis="x", labelsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(grad_lum, cmap=CMAP_FIELD, vmax=np.percentile(grad_lum, 99))
    ax3.add_patch(Circle((px, py), 7, edgecolor=ACCENT_YELLOW, facecolor="none", linewidth=2))
    style_panel(ax3, f"grad luminance |(Lx,Ly)|\ntai diem = {v_grad_lum:.4f}")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(grad_dizenzo, cmap=CMAP_FIELD, vmax=np.percentile(grad_dizenzo, 99))
    ax4.add_patch(Circle((px, py), 7, edgecolor=ACCENT_YELLOW, facecolor="none", linewidth=2))
    style_panel(ax4, f"grad Di Zenzo sqrt(Ixx+Iyy)\ntai diem = {v_grad_dz:.4f}")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.03)

    ax5 = fig.add_subplot(gs[1, 2])
    ratio_show = np.clip(ratio, 0, np.percentile(ratio[mask], 99))
    im5 = ax5.imshow(ratio_show, cmap="viridis")
    ax5.add_patch(Circle((px, py), 7, edgecolor=ACCENT_RED, facecolor="none", linewidth=2))
    style_panel(ax5, "ty le dizenzo/luminance\n(chi vung dizenzo top-30%)")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.03)

    note = (f"So lieu that tai pixel ({px},{py}): grad_luminance = {v_grad_lum:.4f}, "
            f"grad_DiZenzo = {v_grad_dz:.4f}  =>  Di Zenzo lon hon luminance "
            f"{v_ratio:.1f} lan tai diem canh mau nay (RGB doi nhieu, luminance gan nhu khong doi).")
    fig.text(0.5, 0.005, note, fontsize=9.5, ha="center", color=INK, family="monospace")

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")
    return dict(px=px, py=py, grad_lum=v_grad_lum, grad_dizenzo=v_grad_dz, ratio=v_ratio)


# ============================================================================
# 4. Hinh 2 -- vai tro cua rho (cua so tich hop) o buoc 4
# ============================================================================

def fig02_rho_window(image, out):
    _, _, _, Ixx, Ixy, Iyy = raw_dizenzo(image, sigma=1.0)

    rhos = [0.0, 1.5, 6.0]
    rows = []
    for rho in rhos:
        Sxx, Sxy, Syy = integrate_rho(Ixx, Ixy, Iyy, rho)
        lambda1, lambda2 = eig2x2(Sxx, Sxy, Syy)
        coh = coherence(lambda1, lambda2)
        rows.append(dict(rho=rho, Sxx=Sxx, Sxy=Sxy, Syy=Syy, coh=coh, mean_coh=coh.mean()))

    vmax_xx = max(np.percentile(np.abs(r["Sxx"]), 99) for r in rows)
    vmax_xy = max(np.percentile(np.abs(r["Sxy"]), 99) for r in rows)
    vmax_yy = max(np.percentile(np.abs(r["Syy"]), 99) for r in rows)

    fig = plt.figure(figsize=(14, 10), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.2, hspace=0.32,
                            top=0.90, bottom=0.05, left=0.03, right=0.98)
    fig.suptitle("Vai tro cua rho (cua so tich hop) -- loss_utils.py:210-217, buoc 4: "
                 "Ixx,Ixy,Iyy TRUOC vs SAU khi lam mo G_rho (cung anh that, cung sigma=1)",
                 fontsize=13.5, fontweight="bold", color=INK)

    col_names = ["Sxx = Grho*Ixx", "Sxy = Grho*Ixy", "Syy = Grho*Iyy", "coherence ((l1-l2)/(l1+l2))^2"]
    for r, row in enumerate(rows):
        label = "TRUOC blur (rho=0, tho)" if row["rho"] == 0.0 else f"SAU blur rho={row['rho']}"
        ax0 = fig.add_subplot(gs[r, 0])
        im0 = ax0.imshow(row["Sxx"], cmap=CMAP_FIELD, vmax=vmax_xx)
        style_panel(ax0, f"{label}\nSxx")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)

        ax1 = fig.add_subplot(gs[r, 1])
        im1 = ax1.imshow(row["Sxy"], cmap=CMAP_DIV, norm=Normalize(-vmax_xy, vmax_xy))
        style_panel(ax1, "Sxy")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

        ax2 = fig.add_subplot(gs[r, 2])
        im2 = ax2.imshow(row["Syy"], cmap=CMAP_FIELD, vmax=vmax_yy)
        style_panel(ax2, "Syy")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

        ax3 = fig.add_subplot(gs[r, 3])
        im3 = ax3.imshow(row["coh"], cmap="viridis", vmin=0, vmax=1)
        style_panel(ax3, f"coherence, mean={row['mean_coh']:.3f}")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)

    note = ("So lieu that (coherence trung binh toan anh):  " +
            "  |  ".join(f"rho={row['rho']}: {row['mean_coh']:.3f}" for row in rows) +
            f"  => GIAM {rows[0]['mean_coh']:.3f} -> {rows[-1]['mean_coh']:.3f} khi rho tang. "
            "Ly do: o rho=0, Ixx=Ix^2 la ma tran hang-1 TAI TUNG DIEM ANH (mot huong gradient duy nhat) "
            "nen coherence~1 'gia tao' o hau het pixel co canh (khong phan biet duoc canh that voi goc/texture); "
            "chi sau khi Grho THUC SU trung binh khong gian, cac vung nhieu huong gan nhau (goc cua, khung cua so, "
            "chan ghe) moi lo ro coherence thap -- day chinh la ly do buoc 4 bat buoc phai co Grho.")
    fig.text(0.5, 0.008, note, fontsize=9, ha="center", color=INK, family="monospace")

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")
    return rows


# ============================================================================
# 5. Hinh 3 -- chuan hoa theo batch (buoc 5) khong troi theo do sang
# ============================================================================

def fig03_brightness_normalization(image, out):
    scales = [0.5, 1.0, 1.5]
    results = []
    for s in scales:
        img_s = np.clip(image * s, 0.0, 1.0)
        _, _, _, Ixx, Ixy, Iyy = raw_dizenzo(img_s, sigma=1.0)
        Sxx_raw, Sxy_raw, Syy_raw = integrate_rho(Ixx, Ixy, Iyy, rho=1.0)
        mag_raw = Sxx_raw + Syy_raw
        max_val = mag_raw.max() + 1e-6
        Sxx_n = Sxx_raw / max_val
        Sxy_n = Sxy_raw / max_val
        Syy_n = Syy_raw / max_val
        results.append(dict(scale=s, img=img_s, mag_raw=mag_raw, max_val=max_val,
                             Sxx_n=Sxx_n, Sxy_n=Sxy_n, Syy_n=Syy_n))

    ref = results[1]  # scale = 1.0
    diff05 = np.abs(results[0]["Sxx_n"] - ref["Sxx_n"])
    diff15 = np.abs(results[2]["Sxx_n"] - ref["Sxx_n"])

    vmax_raw = max(r["mag_raw"].max() for r in results)

    fig = plt.figure(figsize=(13.5, 11.5), facecolor=BG_CREAM)
    gs = gridspec.GridSpec(4, 3, figure=fig, wspace=0.2, hspace=0.4,
                            top=0.93, bottom=0.05, left=0.03, right=0.98)
    fig.suptitle("Chuan hoa theo batch (buoc 5) -- loss_utils.py:219-227: cung anh that "
                 "render o 0.5x / 1.0x / 1.5x do sang, TRUOC vs SAU khi chia cho max(Sxx+Syy)",
                 fontsize=13, fontweight="bold", color=INK)

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(r["img"])
        style_panel(ax, f"Input x{r['scale']} do sang")

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(r["mag_raw"], cmap=CMAP_FIELD, vmax=vmax_raw)
        style_panel(ax, f"TRUOC chuan hoa: Sxx+Syy\nmax = {r['max_val']:.5f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    for j, r in enumerate(results):
        ax = fig.add_subplot(gs[2, j])
        im = ax.imshow(r["Sxx_n"], cmap=CMAP_FIELD, vmin=0, vmax=1)
        style_panel(ax, f"SAU chuan hoa: Sxx (x{r['scale']})\nmax = {r['Sxx_n'].max():.4f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax_d1 = fig.add_subplot(gs[3, 0])
    im = ax_d1.imshow(diff05, cmap="inferno", vmax=max(diff05.max(), 1e-6))
    style_panel(ax_d1, f"|Sxx_n(0.5x) - Sxx_n(1.0x)|\nmax diff = {diff05.max():.4f}")
    fig.colorbar(im, ax=ax_d1, fraction=0.046, pad=0.03)

    ax_d2 = fig.add_subplot(gs[3, 1])
    im = ax_d2.imshow(diff15, cmap="inferno", vmax=max(diff15.max(), 1e-6))
    style_panel(ax_d2, f"|Sxx_n(1.5x) - Sxx_n(1.0x)|\nmax diff = {diff15.max():.4f}")
    fig.colorbar(im, ax=ax_d2, fraction=0.046, pad=0.03)

    ax_txt = fig.add_subplot(gs[3, 2])
    ax_txt.axis("off")
    ratio_0501 = results[0]["max_val"] / ref["max_val"]
    ratio_1501 = results[2]["max_val"] / ref["max_val"]
    txt = (
        f"So lieu that (max(Sxx+Syy) TRUOC chuan hoa):\n"
        f"  x0.5 do sang: {results[0]['max_val']:.5f}\n"
        f"  x1.0 do sang: {ref['max_val']:.5f}\n"
        f"  x1.5 do sang: {results[2]['max_val']:.5f}\n\n"
        f"Ty le max/max_ref: x0.5 -> {ratio_0501:.3f} (~0.5^2={0.5**2:.3f})\n"
        f"                    x1.5 -> {ratio_1501:.3f} (~1.5^2={1.5**2:.3f}, "
        f"lech do clip [0,1] o vung sang qua)\n\n"
        f"SAU chuan hoa: max|diff Sxx| = {diff05.max():.4f} (x0.5) / "
        f"{diff15.max():.4f} (x1.5)\n"
        f"=> x0.5: GIONG HET (khong clip, k^2 trieu tieu dung theo dai so).\n"
        f"=> x1.5: giong het o da so pixel, LECH manh chi o vung sang bi\n"
        f"   bao hoa [0,1] (cua so/khung cua trang) -- dung dung diem yeu\n"
        f"   cua gia dinh tuyen tinh khi anh bi over-expose."
    )
    ax_txt.text(0.0, 1.0, txt, fontsize=9, va="top", family="monospace", color=INK,
                transform=ax_txt.transAxes)

    fig.savefig(out, dpi=150, facecolor=BG_CREAM)
    plt.close(fig)
    print(f"[OK] {out}")
    return dict(max_vals=[r["max_val"] for r in results], diff05=diff05.max(), diff15=diff15.max())


# ============================================================================
# main
# ============================================================================

def main():
    print(__doc__)
    print(f"Input anh that: {SRC_IMAGE}")
    image, size = load_real_crop()
    print(f"Da crop anh that: {size[0]}x{size[1]} px tu {os.path.basename(SRC_IMAGE)} "
          f"(vung (11,381)-(451,669))")

    r1 = fig01_iso_luminant(image, os.path.join(OUT_DIR, "deep17a_01_iso_luminant_dizenzo.png"))
    print(f"  -> diem ({r1['px']},{r1['py']}): grad_luminance={r1['grad_lum']:.4f}, "
          f"grad_dizenzo={r1['grad_dizenzo']:.4f}, ty le={r1['ratio']:.1f}x")

    r2 = fig02_rho_window(image, os.path.join(OUT_DIR, "deep17a_02_rho_integration_window.png"))
    print("  -> mean coherence theo rho: " +
          ", ".join(f"rho={row['rho']}:{row['mean_coh']:.3f}" for row in r2))

    r3 = fig03_brightness_normalization(image, os.path.join(OUT_DIR, "deep17a_03_brightness_normalization.png"))
    print(f"  -> max(Sxx+Syy) truoc chuan hoa (0.5x/1.0x/1.5x): {r3['max_vals']}")
    print(f"  -> max diff Sxx sau chuan hoa: 0.5x={r3['diff05']:.5f}, 1.5x={r3['diff15']:.5f}")

    print("\nHoan tat: 3 hinh PNG that da duoc ghi vao", OUT_DIR)


if __name__ == "__main__":
    main()
