# -*- coding: utf-8 -*-
"""
BOT 11 — Hình minh hoạ cho các hàm mất mát của SADGS.
Nguồn công thức: SADGS/utils/loss_utils.py, SADGS/utils/freq_utils.py, SADGS/train.py
Mọi công thức được cài lại NGUYÊN VĂN từ code (numpy thay cho torch).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


# ----------------------------------------------------------------------
# HÌNH 1: L1 vs L2 vs tone-curve theo sai số (đường cong + đạo hàm)
#   loss_utils.py:22  l1_loss  = mean|p - g|
#   loss_utils.py:25  l2_loss  = mean (p - g)^2
#   loss_utils.py:28  tone_curve_loss = mean( ((p-g)/(sg(p)+eps))^norm ), eps=1e-6, norm=2
# ----------------------------------------------------------------------
def fig1():
    e = np.linspace(-0.5, 0.5, 801)          # sai số e = p - g
    eps = 1e-6
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))

    ax[0].plot(e, np.abs(e), "k-", lw=2, label=r"$L_1=|e|$")
    ax[0].plot(e, e ** 2, "b-", lw=2, label=r"$L_2=e^2$")
    for p, c in [(0.15, "#d62728"), (0.40, "#ff7f0e"), (0.80, "#2ca02c")]:
        ax[0].plot(e, (e / (p + eps)) ** 2, c=c, lw=1.8, ls="--",
                   label=r"tone, $\hat{p}=%.2f$" % p)
    ax[0].set_ylim(0, 1.2)
    ax[0].set_xlabel(r"sai số $e=p-g$")
    ax[0].set_ylabel("giá trị mất mát")
    ax[0].set_title("Giá trị mất mát", fontsize=10)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)

    ax[1].plot(e, np.sign(e), "k-", lw=2, label=r"$dL_1/de=\mathrm{sign}(e)$")
    ax[1].plot(e, 2 * e, "b-", lw=2, label=r"$dL_2/de=2e$")
    for p, c in [(0.15, "#d62728"), (0.40, "#ff7f0e"), (0.80, "#2ca02c")]:
        ax[1].plot(e, 2 * e / (p + eps) ** 2, c=c, lw=1.8, ls="--",
                   label=r"tone: $2e/\hat{p}^2$, $\hat{p}=%.2f$" % p)
    ax[1].set_ylim(-12, 12)
    ax[1].set_xlabel(r"sai số $e=p-g$")
    ax[1].set_ylabel("đạo hàm theo e")
    ax[1].set_title(r"Đạo hàm (stop-grad ở mẫu số $\hat{p}$)", fontsize=10)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)

    fig.suptitle("loss_utils.py:22/25/28 — L1, L2, tone-curve", fontsize=11)
    save(fig, "11_l1_l2_tone.png")


# ----------------------------------------------------------------------
# HÌNH 2: cửa sổ Gaussian 11x11 + bản đồ SSIM
#   loss_utils.py:36-44 gaussian(11, 1.5), create_window
#   loss_utils.py:56-76 _ssim, C1=0.01^2, C2=0.03^2
# ----------------------------------------------------------------------
def gaussian_1d(window_size=11, sigma=1.5):
    g = np.array([np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
                  for x in range(window_size)])
    return g / g.sum()


def conv2d_same(img, w1d):
    k = len(w1d); pad = k // 2
    t = np.pad(img, pad, mode="constant")
    out = np.zeros_like(img, dtype=float)
    tmp = np.zeros_like(t)
    for i, v in enumerate(w1d):                     # theo hàng
        tmp[:, pad:pad + img.shape[1]] += v * t[:, i:i + img.shape[1]]
    for i, v in enumerate(w1d):                     # theo cột
        out += v * tmp[i:i + img.shape[0], pad:pad + img.shape[1]]
    return out


def ssim_map(a, b, w1d):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = conv2d_same(a, w1d), conv2d_same(b, w1d)
    m1s, m2s, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = conv2d_same(a * a, w1d) - m1s
    s2 = conv2d_same(b * b, w1d) - m2s
    s12 = conv2d_same(a * b, w1d) - m12
    return ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1s + m2s + C1) * (s1 + s2 + C2))


def make_pair(n=128, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    img = 0.45 + 0.18 * np.sin(2 * np.pi * x / 26.0)
    img[n // 2:, :] += 0.18                                  # cạnh ngang
    img[20:50, 20:50] = 0.85                                 # ô sáng
    img = np.clip(img, 0, 1)
    dist = img.copy()
    dist[20:50, 20:50] += 0.12                               # lỗi cục bộ (sai độ sáng)
    dist[70:110, 70:110] += 0.25 * rng.standard_normal((40, 40))   # nhiễu cấu trúc
    return img, np.clip(dist, 0, 1)


def fig2():
    w1d = gaussian_1d(11, 1.5)
    w2d = np.outer(w1d, w1d)
    a, b = make_pair()
    sm = ssim_map(a, b, w1d)

    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(w2d, cmap="viridis")
    ax.set_title(r"Cửa sổ Gaussian $11\times11$, $\sigma=1.5$" "\n" r"$\sum w=%.4f$" % w2d.sum(),
                 fontsize=9)
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
    fig.colorbar(im, ax=ax, fraction=.046)

    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(a, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("ảnh GT", fontsize=9); ax1.axis("off")
    ax2 = fig.add_subplot(gs[1, 1]); ax2.imshow(b, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("ảnh render (có lỗi)", fontsize=9); ax2.axis("off")

    ax3 = fig.add_subplot(gs[:, 2])
    im3 = ax3.imshow(sm, cmap="RdYlGn", vmin=0, vmax=1)
    ax3.set_title("SSIM map — trung bình = %.3f\n" % sm.mean() +
                  r"$1-$SSIM $=%.3f$" % (1 - sm.mean()), fontsize=9)
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, fraction=.046)

    fig.suptitle("loss_utils.py:36-79 — cửa sổ Gaussian và bản đồ SSIM", fontsize=11)
    save(fig, "11_ssim_window.png")


# ----------------------------------------------------------------------
# HÌNH 3: L1 vs SSIM nhạy với loại lỗi nào
# ----------------------------------------------------------------------
def fig3():
    w1d = gaussian_1d(11, 1.5)
    a, _ = make_pair()
    n = a.shape[0]
    rng = np.random.default_rng(3)
    amps = np.linspace(0, 0.30, 13)

    rng_fixed = rng.standard_normal((n, n))

    def blur_mix(img, t, w):
        bl = conv2d_same(conv2d_same(img, w), w)
        k = min(1.0, t / 0.30)
        return np.clip((1 - k) * img + k * bl, 0, 1)

    kinds = {
        "Lệch sáng toàn ảnh (bias)": lambda t: np.clip(a + t, 0, 1),
        "Nhiễu trắng (cấu trúc)":    lambda t: np.clip(a + t * 3.0 * rng_fixed, 0, 1),
        "Làm mờ (mất chi tiết)":     lambda t: blur_mix(a, t, w1d),
    }

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for (name, f), c in zip(kinds.items(), colors):
        l1s, ds = [], []
        for t in amps:
            b = f(t)
            l1s.append(np.abs(b - a).mean())
            ds.append(1.0 - ssim_map(a, b, w1d).mean())
        l1s, ds = np.array(l1s), np.array(ds)
        ax[0].plot(amps, l1s, "-o", ms=3, c=c, label=name)
        ax[1].plot(amps, ds, "-o", ms=3, c=c, label=name)
        ax[2].plot(amps[1:], ds[1:] / (l1s[1:] + 1e-9), "-o", ms=3, c=c, label=name)
    ax[0].set_title(r"$L_1$ = mean$|p-g|$", fontsize=10)
    ax[1].set_title(r"D-SSIM $=1-$SSIM", fontsize=10)
    ax[2].axhline(1.0, color="k", ls=":", lw=1)
    ax[2].set_yscale("log")
    ax[2].set_title("Tỉ số D-SSIM / $L_1$\n(>1: SSIM nhạy hơn)", fontsize=9)
    for k in (0, 1, 2):
        ax[k].set_xlabel("cường độ lỗi"); ax[k].grid(alpha=.3); ax[k].legend(fontsize=7)
    ax[0].set_ylabel("giá trị")
    fig.suptitle(r"L1 phạt mạnh lệch sáng; SSIM phạt mạnh mất/sai cấu trúc"
                 "\n" r"train.py:259 dùng $0.8\,L_1+0.2\,(1-\mathrm{SSIM})+2.0\,L_2$", fontsize=10)
    save(fig, "11_l1_vs_ssim.png")


# ----------------------------------------------------------------------
# HÌNH 4: frequency_loss  (loss_utils.py:80-110)
#   det = a*c - b*b ; sigma_area = sqrt(det+1e-6) ; sigma_linear = sqrt(sigma_area+1e-6)
#   freq_target = sqrt(Sxx+Syy+1e-6) ; wavelength_limit = 1/(freq_target+1e-6)
#   loss = relu(sigma_linear - wavelength_limit).mean()
# ----------------------------------------------------------------------
def fig4():
    n = 200
    y, x = np.mgrid[0:n, 0:n]
    # chirp: tần số tăng dần theo x  -> vùng phải là texture tần số cao
    phase = 2 * np.pi * (0.005 * x + 0.00018 * x ** 2)
    img = 0.5 + 0.45 * np.sin(phase)

    # ước lượng freq_target theo cùng ý nghĩa Sxx+Syy đã chuẩn hoá (loss_utils.py:222-227)
    gx = np.gradient(img, axis=1); gy = np.gradient(img, axis=0)
    S = gx ** 2 + gy ** 2
    # tích phân cửa sổ (rho-smoothing) như get_structure_tensor_torch: loss_utils.py:215-217
    rho = gaussian_1d(41, 8.0)
    S = conv2d_same(S, rho)
    S = S / (S.max() + 1e-6)                       # chuẩn hoá theo max: loss_utils.py:222-227
    freq_prof = np.sqrt(S.mean(axis=0) + 1e-6)
    wl_prof = 1.0 / (freq_prof + 1e-6)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    ax[0].imshow(img, cmap="gray", extent=[0, n, n, 0])
    for cx, s, ok in [(40, 26, True), (110, 26, False), (170, 26, False),
                      (170, 6, True)]:
        col = "#2ca02c" if ok else "#d62728"
        el = matplotlib.patches.Ellipse((cx, 100 if s > 10 else 150), 2 * s, 2 * s,
                                        fill=False, ec=col, lw=2)
        ax[0].add_patch(el)
    ax[0].set_title("Gaussian chiếu lên texture chirp\n(đỏ: bán kính > bước sóng)", fontsize=9)
    ax[0].set_xlabel("x (pixel)")

    ax[1].plot(np.arange(n), wl_prof, "b-", lw=2, label=r"$\lambda_{lim}=1/f_{target}$")
    ax[1].axhline(26, color="#d62728", ls="--", lw=1.5, label=r"$\sigma_{lin}=26$ (Gaussian to)")
    ax[1].axhline(6, color="#2ca02c", ls="--", lw=1.5, label=r"$\sigma_{lin}=6$ (Gaussian nhỏ)")
    ax[1].fill_between(np.arange(n), wl_prof, 26, where=(wl_prof < 26),
                       color="#d62728", alpha=.2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("x (pixel)"); ax[1].set_ylabel("pixel")
    ax[1].set_title("Vùng tô đỏ: bị phạt\n" r"$\mathrm{relu}(\sigma_{lin}-\lambda_{lim})>0$", fontsize=9)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)

    sig = np.linspace(0, 40, 400)
    for wl, c in [(5, "#d62728"), (15, "#ff7f0e"), (30, "#2ca02c")]:
        ax[2].plot(sig, np.maximum(sig - wl, 0), c=c, lw=2,
                   label=r"$\lambda_{lim}=%d$ px" % wl)
    ax[2].set_xlabel(r"$\sigma_{lin}=\sqrt{\sqrt{\det\Sigma_{2D}}}$")
    ax[2].set_ylabel("phạt mỗi Gaussian")
    ax[2].set_title(r"$\mathrm{relu}(\sigma_{lin}-\lambda_{lim})$", fontsize=9)
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=7)

    fig.suptitle("loss_utils.py:80-110 — frequency_loss(means2D, cov2D, st_map, H, W)", fontsize=11)
    save(fig, "11_frequency_loss.png")


# ----------------------------------------------------------------------
# HÌNH 5: estimate_required_gaussians (loss_utils.py:394-437)
#   count_coverage = H*W*base_density
#   count_detail   = sum(Sxx+Syy) * detail_sensitivity
# ----------------------------------------------------------------------
def fig5():
    H, W = 768, 1024
    total_pixels = H * W
    # năng lượng cấu trúc tổng, mô phỏng 3 loại cảnh (st_map đã chuẩn hoá max ~1)
    scenes = {"cảnh phẳng (E=0.01·HW)": 0.01 * total_pixels,
              "cảnh vừa (E=0.05·HW)":  0.05 * total_pixels,
              "cảnh nhiều texture (E=0.15·HW)": 0.15 * total_pixels}

    bd = np.linspace(0.0, 0.05, 200)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for (name, E), c in zip(scenes.items(), ["#2ca02c", "#ff7f0e", "#d62728"]):
        ax[0].plot(bd, (total_pixels * bd + E * 0.5) / 1e3, c=c, lw=2, label=name)
    ax[0].axvline(0.01, color="k", ls=":", lw=1.2)
    ax[0].text(0.0105, ax[0].get_ylim()[1] * .55, "mặc định\nbase_density=0.01", fontsize=7)
    ax[0].set_xlabel("base_density"); ax[0].set_ylabel("số Gaussian ước lượng (nghìn)")
    ax[0].set_title(r"$N=HW\cdot b + E\cdot d$, $d=0.5$, $HW=1024{\times}768$", fontsize=9)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)

    ds = np.linspace(0.0, 1.5, 200)
    for (name, E), c in zip(scenes.items(), ["#2ca02c", "#ff7f0e", "#d62728"]):
        ax[1].plot(ds, (total_pixels * 0.01 + E * ds) / 1e3, c=c, lw=2, label=name)
    ax[1].axvline(0.5, color="k", ls=":", lw=1.2)
    ax[1].text(0.52, ax[1].get_ylim()[1] * .2, "mặc định\ndetail_sensitivity=0.5", fontsize=7)
    ax[1].axhline(total_pixels * 0.01 / 1e3, color="gray", ls="--", lw=1,
                  label="phần phủ (coverage) = 7.9k")
    ax[1].set_xlabel("detail_sensitivity"); ax[1].set_ylabel("số Gaussian ước lượng (nghìn)")
    ax[1].set_title("Phần chi tiết tỉ lệ tuyến tính với năng lượng cấu trúc E", fontsize=9)
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)

    fig.suptitle("loss_utils.py:394-437 — estimate_required_gaussians (KHÔNG được gọi trong train.py)",
                 fontsize=10)
    save(fig, "11_estimate_gaussians.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("DONE")
