"""
Mô phỏng matplotlib cho TOÀN BỘ công thức trong SH.md
(đối chiếu SADGS/utils/sh_utils.py và computeColorFromSH)

Mỗi hàm fig_x_y() ứng với đúng một công thức / một mục trong tài liệu,
xuất ra 1 file PNG trong cùng thư mục sh_figures/.

Chạy: python sh_visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.special import sph_harm_y
import os

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["figure.constrained_layout.use"] = True


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved:", path)


def sphere_surface(ax, THETA, PHI, values, cmap="RdBu_r", vcenter=None):
    """Ve mat cau ban kinh CO DINH r=1, mau sac the hien gia tri ham (khong
    bien dang ban kinh) — tranh hinh dang meo mo, de doc hon."""
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    vmax = np.max(np.abs(values)) + 1e-12
    norm = (values + vmax) / (2 * vmax) if vcenter is None else (values - values.min()) / (values.max() - values.min() + 1e-12)
    colors = plt.get_cmap(cmap)(norm)
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def lobe_surface(ax, THETA, PHI, values, cmap="RdBu_r"):
    """Ve dang 'canh hoa' kinh dien cua SH: ban kinh = |gia tri|, mau = dau."""
    Rr = np.abs(values)
    X = Rr * np.sin(THETA) * np.cos(PHI)
    Y = Rr * np.sin(THETA) * np.sin(PHI)
    Z = Rr * np.cos(THETA)
    vmax = np.max(np.abs(values)) + 1e-12
    colors = plt.get_cmap(cmap)((values + vmax) / (2 * vmax))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def fig_2_2_low_order_real_sh_xyz():
    """Eq: Y00, Y1,-1=C1 y, Y10=C1 z, Y11=C1 x và hằng số C0,C1."""
    C0 = 0.5 * np.sqrt(1 / np.pi)
    C1 = 0.5 * np.sqrt(3 / np.pi)

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    THETA, PHI = np.meshgrid(theta, phi)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)

    funcs = {
        r"$Y_{00}=C_0$": np.full_like(x, C0),
        r"$Y_{1,-1}=C_1\,y$": C1 * y,
        r"$Y_{1,0}=C_1\,z$": C1 * z,
        r"$Y_{1,1}=C_1\,x$": C1 * x,
    }
    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    for i, (title, F) in enumerate(funcs.items()):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        lobe_surface(ax, THETA, PHI, F)
        ax.set_title(title, fontsize=11)
    fig.suptitle(rf"2.2 — SH thực bậc thấp theo $(x,y,z)$   ($C_0={C0:.4f}$, $C_1={C1:.4f}$)", fontsize=13)
    save(fig, "2_2_low_order_real_sh.png")


# ---------------------------------------------------------------------------
# PHẦN III
# ---------------------------------------------------------------------------

def fig_3_2_truncation_and_coeff_count():
    """Eq: c_i(d) ~ sum_{l=0}^{D} sum_m k_lm Y_lm  và  số hệ số = (D+1)^2."""
    theta = np.linspace(1e-4, np.pi - 1e-4, 60)
    phi = np.linspace(0, 2 * np.pi, 120)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    true_c = 0.5 + 0.3 * np.cos(THETA) + 0.2 * np.sin(THETA) ** 2 * np.cos(2 * PHI)
    dth = theta[1] - theta[0]; dph = phi[1] - phi[0]
    dOmega = np.sin(THETA) * dth * dph

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), subplot_kw={"projection": "3d"}, constrained_layout=True)
    for ax, D in zip(axes, [0, 1, 2, 3]):
        recon = np.zeros_like(THETA, dtype=complex)
        for l in range(D + 1):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, m, THETA, PHI)
                k_lm = np.sum(true_c * np.conj(Y) * dOmega)
                recon += k_lm * Y
        sphere_surface(ax, THETA, PHI, recon.real, cmap="viridis", vcenter=True)
        ncoef = (D + 1) ** 2
        ax.set_title(f"D={D}\n(D+1)²={ncoef} hệ số", fontsize=10)
    fig.suptitle(r"3.1/3.2 — Cắt cụt chuỗi SH tại bậc $D$: $c_i(\vec d)\approx\sum_{l=0}^{D}\sum_m k_{i,lm}Y_{lm}$", fontsize=13)
    save(fig, "3_2_truncation_reconstruction.png")

    fig2, ax = plt.subplots(figsize=(5.5, 4.3), constrained_layout=True)
    Ds = np.arange(0, 8)
    ax.bar(Ds, (Ds + 1) ** 2, color="steelblue")
    for D in Ds:
        ax.text(D, (D + 1) ** 2 + 1, str((D + 1) ** 2), ha="center", fontsize=8)
    ax.set_xlabel("D"); ax.set_ylabel("số hệ số / kênh màu")
    ax.set_title(r"Số hệ số cần lưu: $\sum_{l=0}^{D}(2l+1)=(D+1)^2$", fontsize=12)
    save(fig2, "3_2b_coefficient_count.png")


def fig_3_3_compute_color_from_sh():
    """Eq đầy đủ: c_i(d) = max(0, 0.5 + C0 k00 + sum_{l=1}^{3} ...). Mô phỏng
    màu đầy đủ của 1 Gaussian với bộ hệ số k_{lm} ngẫu nhiên, thể hiện cả
    hiệu ứng clamp (max(0,.))."""
    rng = np.random.default_rng(7)
    theta = np.linspace(1e-4, np.pi - 1e-4, 80)
    phi = np.linspace(0, 2 * np.pi, 160)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    fig = plt.figure(figsize=(13, 4.6), constrained_layout=True)
    channel_names = ["R", "G", "B"]
    for ci, cname in enumerate(channel_names):
        # DC nho + bac cao lon => co vung bi am, de thay hieu ung clamp
        k = {(l, m): rng.normal(0, 0.35) for l in range(4) for m in range(-l, l + 1)}
        k[(0, 0)] = rng.uniform(-0.1, 0.3)

        c_raw = np.zeros_like(THETA)
        for l in range(4):
            for m in range(-l, l + 1):
                Y = sph_harm_y(l, m, THETA, PHI).real
                c_raw += k[(l, m)] * Y
        c_raw = 0.5 + c_raw
        c_clamped = np.clip(c_raw, 0, 1)

        ax = fig.add_subplot(1, 3, ci + 1, projection="3d")
        X = np.sin(THETA) * np.cos(PHI); Y_ = np.sin(THETA) * np.sin(PHI); Z = np.cos(THETA)
        facecolor = np.zeros(THETA.shape + (4,))
        facecolor[..., ci] = c_clamped
        facecolor[..., 3] = 1.0
        ax.plot_surface(X, Y_, Z, facecolors=facecolor, rstride=2, cstride=2, linewidth=0)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
        frac_clamped = np.mean(c_raw < 0) * 100
        ax.set_title(f"kênh {cname}\n{frac_clamped:.1f}% điểm bị clamp(0,·)", fontsize=10)
    fig.suptitle(r"3.2 — $c_i(\vec d)=\max(0,\ 0.5+C_0k_{00}+\sum_{l=1}^{3}\sum_m C_l^{(m)}P_l^{(m)}(x,y,z)k_{lm})$", fontsize=12)
    save(fig, "3_3_compute_color_from_sh.png")


ALL_FIGS = [
    fig_2_2_low_order_real_sh_xyz,
    fig_3_2_truncation_and_coeff_count,
    fig_3_3_compute_color_from_sh,
]

if __name__ == "__main__":
    import sys
    names = sys.argv[1:]
    if names:
        by_name = {f.__name__: f for f in ALL_FIGS}
        for n in names:
            by_name[n]()
    else:
        for f in ALL_FIGS:
            f()
    print("DONE — tất cả hình đã lưu trong", OUT)
