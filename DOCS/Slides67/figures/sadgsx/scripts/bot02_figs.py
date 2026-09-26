# -*- coding: utf-8 -*-
"""
BOT 02 - SADGS: Chieu truc Gaussian len man hinh & ti so cau truc eta.
Moi cong thuc trong slide <=> 1 hinh o day.
Nguon code that:
  SADGS/utils/freq_utils.py:116-178  compute_projected_axes_subset
  SADGS/utils/freq_utils.py:313-348  eta_compute_mode == "wavelength"
  SADGS/utils/freq_utils.py:350-373  eta_compute_mode == "projection"
  SADGS/utils/freq_utils.py:395-409  TAU_HIGH=1.0 / TAU_LOW=0.1
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

plt.rcParams["font.family"] = "DejaVu Sans"

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


def quat_to_R(q):
    """Giong utils/general_utils.py build_rotation (q = w,x,y,z, da chuan hoa)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------- FIG 1
# Sigma3D = R S S^T R^T  ->  ellipse 2D tren man hinh theo depth
def fig1():
    s = np.array([0.30, 0.12, 0.06])
    R = quat_to_R([0.92, 0.10, 0.20, 0.32])
    S = np.diag(s)
    M = R @ S
    Sigma3 = M @ M.T

    fx = fy = 1000.0
    depths = [2.0, 4.0, 8.0]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))

    # panel 0: ma tran hiep phuong sai 3D
    ax = axes[0]
    im = ax.imshow(Sigma3, cmap="viridis")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{Sigma3[i, j]:.3f}", ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_title(r"$\Sigma_{3D}=R\,S\,S^\top R^\top$" + "\n" + r"$s=(0.30,0.12,0.06)$",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for k, z in enumerate(depths):
        ax = axes[k + 1]
        # J tai tam anh (vec_x=vec_y=0) => J = [[fx/z,0,0],[0,fy/z,0]]
        J = np.array([[fx / z, 0.0, 0.0], [0.0, fy / z, 0.0]])
        Sigma2 = J @ Sigma3 @ J.T
        vals, vecs = np.linalg.eigh(Sigma2)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
        w, h = 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0])
        ax.add_patch(Ellipse((0, 0), w, h, angle=ang, facecolor="#4C78A8",
                             alpha=0.45, edgecolor="#1f4e79", lw=1.6))
        # 3 truc chieu: cot cua J @ (R*s)
        axes_cam = R * s[None, :]          # cot j = truc j trong khong gian camera
        proj = J @ axes_cam                # [2,3]
        cols = ["#E45756", "#F58518", "#54A24B"]
        for j in range(3):
            ax.arrow(0, 0, proj[0, j], proj[1, j], color=cols[j], lw=1.8,
                     head_width=4, length_includes_head=True,
                     label=f"truc {j+1}: {np.hypot(*proj[:, j]):.0f} px")
        L = 190
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"depth z = {z:.0f} m", fontsize=9)
        ax.set_xlabel("u (pixel)")
        if k == 0:
            ax.set_ylabel("v (pixel)")
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(r"$\Sigma_{2D}=J\,\Sigma_{3D}J^\top$ — Gaussian co dinh, anh chieu co lai theo $1/z$",
                 fontsize=10)
    save(fig, "02_cov3d_to_cov2d.png")


# ---------------------------------------------------------------- FIG 2
# Jacobian phoi canh + cach lay 3 truc 2D (freq_utils.py:137-178)
def fig2():
    fx = fy = 1000.0
    z = 4.0
    x0, y0 = 1.2, 0.5     # vec_x, vec_y (freq_utils.py:134-135)
    inv_z = 1.0 / z
    J = np.array([[fx * inv_z, 0.0, -fx * x0 * inv_z ** 2],
                  [0.0, fy * inv_z, -fy * y0 * inv_z ** 2]])

    s = np.array([0.30, 0.14, 0.07])
    R = quat_to_R([0.92, 0.10, 0.20, 0.32])
    axes_cam = R * s[None, :]              # freq_utils.py:165 axes_cam = R_total * scales
    ax_x, ax_y, ax_z = axes_cam[0], axes_cam[1], axes_cam[2]
    u_vec = J[0, 0] * ax_x + J[0, 2] * ax_z    # freq_utils.py:174
    v_vec = J[1, 1] * ax_y + J[1, 2] * ax_z    # freq_utils.py:175
    lengths = np.sqrt(u_vec ** 2 + v_vec ** 2)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))

    a0.imshow(J, cmap="coolwarm", vmin=-np.abs(J).max(), vmax=np.abs(J).max())
    for i in range(2):
        for j in range(3):
            a0.text(j, i, f"{J[i, j]:.1f}", ha="center", va="center", fontsize=10)
    a0.set_xticks(range(3), ["x", "y", "z"])
    a0.set_yticks(range(2), ["u", "v"])
    a0.set_title(r"$J$: hang $u=(f_x/z,\;0,\;-f_x x/z^2)$, hang $v=(0,\;f_y/z,\;-f_y y/z^2)$"
                 + f"\n$z={z}$, $f_x=f_y={fx:.0f}$", fontsize=8)

    cols = ["#E45756", "#F58518", "#54A24B"]
    for j in range(3):
        a1.arrow(0, 0, u_vec[j], v_vec[j], color=cols[j], lw=2.2, head_width=3,
                 length_includes_head=True,
                 label=r"$\|a_%d\|=%.1f$ px" % (j + 1, lengths[j]))
    ang = np.linspace(0, 2 * np.pi, 200)
    A = np.stack([u_vec, v_vec])            # [2,3]
    # bien 1-sigma: A @ (he so tren mat cau) -> xap xi bang ellipse tu A A^T
    C = A @ A.T
    vals, vecs = np.linalg.eigh(C)
    ang_e = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    a1.add_patch(Ellipse((0, 0), 2 * np.sqrt(vals[-1]), 2 * np.sqrt(vals[0]),
                         angle=ang_e, facecolor="#4C78A8", alpha=0.25,
                         edgecolor="#1f4e79"))
    L = 90
    a1.set_xlim(-L, L); a1.set_ylim(-L, L); a1.set_aspect("equal"); a1.grid(alpha=0.3)
    a1.set_xlabel("u (pixel)"); a1.set_ylabel("v (pixel)")
    a1.set_title(r"$u_j=J_{00}a^x_j+J_{02}a^z_j$,  $v_j=J_{11}a^y_j+J_{12}a^z_j$"
                 "\n(freq_utils.py:174-178)", fontsize=9)
    a1.legend(fontsize=8)
    save(fig, "02_jacobian_axes.png")


# ---------------------------------------------------------------- FIG 3
# Do dai truc chieu theo khoang cach: |a| ~ f*s/z
def fig3():
    fx = 1000.0
    z = np.linspace(0.6, 12.0, 400)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))

    for s_, c in zip([0.30, 0.12, 0.05], ["#E45756", "#F58518", "#54A24B"]):
        a0.plot(z, fx * s_ / z, color=c, lw=2, label=f"s = {s_:.2f} m")
    a0.axhline(1.0, color="k", ls="--", lw=1, label="1 pixel")
    a0.set_yscale("log"); a0.set_xlabel("depth z (m)")
    a0.set_ylabel("do dai truc chieu (pixel)")
    a0.set_title(r"$\|a_j\|\approx f\,s_j/z$ (tam anh)", fontsize=10)
    a0.grid(alpha=0.3, which="both"); a0.legend(fontsize=8)

    # eta theo z voi buoc song co dinh
    for wl, c in zip([2.0, 6.0, 20.0], ["#9467bd", "#1f77b4", "#2ca02c"]):
        a1.plot(z, (fx * 0.12 / z) / wl, color=c, lw=2,
                label=r"$\lambda_{\min}=%.0f$ px" % wl)
    a1.axhline(1.0, color="#d62728", ls="--", lw=1.4, label=r"$\tau_{high}=1.0$")
    a1.axhline(0.1, color="#7f7f7f", ls=":", lw=1.4, label=r"$\tau_{low}=0.1$")
    a1.set_yscale("log"); a1.set_xlabel("depth z (m)"); a1.set_ylabel(r"$\eta$")
    a1.set_title(r"$\eta=\|a_j\|/\lambda_{\min}$ voi $s=0.12$ m", fontsize=10)
    a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=8)
    save(fig, "02_axis_vs_depth.png")


# ---------------------------------------------------------------- FIG 4
# Structure tensor -> lambda1 -> buoc song cuc bo
def fig4():
    # anh soc voi tan so tang dan
    W = 256
    xx, yy = np.meshgrid(np.arange(W), np.arange(W))
    freq = 0.02 + 0.28 * (xx / W)                 # chu ky / pixel
    img = 0.5 + 0.5 * np.sin(2 * np.pi * freq * xx)

    # Sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    ky = kx.T

    def conv(a, k):
        out = np.zeros_like(a)
        for i in range(3):
            for j in range(3):
                out[1:-1, 1:-1] += k[i, j] * a[i:i + W - 2, j:j + W - 2]
        return out

    Ix, Iy = conv(img, kx), conv(img, ky)
    Sxx, Sxy, Syy = Ix ** 2, Ix * Iy, Iy ** 2
    tr = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    delta = np.sqrt(np.clip((tr / 2) ** 2 - det, 0, None))
    lam1 = tr / 2 + delta
    wl = 1.0 / (np.sqrt(lam1) + 1e-5)

    # lambda_1 la NANG LUONG TAN SO CUC DAI -> lay max tren cua so cot
    core = slice(8, W - 8)
    lam1_col = lam1[core, :].max(axis=0)                 # [W]
    # running-max tren cua so +-8 px de lay NANG LUONG CUC DAI cuc bo
    hw = 8
    lam1_run = np.array([lam1_col[max(0, i - hw):i + hw + 1].max() for i in range(W)])
    wl_col = 1.0 / (np.sqrt(lam1_run) + 1e-5)
    # tan so tuc thoi cua chirp: d/dx[freq(x)*x]
    inst_f = 0.02 + 2 * 0.28 * np.arange(W) / W
    period = 1.0 / inst_f                                 # chu ky that theo cot

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("anh: soc co tan so tang dan", fontsize=9)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    im1 = axs[1].imshow(lam1, cmap="magma")
    axs[1].set_title(r"$\lambda_1=\frac{tr}{2}+\sqrt{(\frac{tr}{2})^2-\det}$"
                     "\n(freq_utils.py:325-329)", fontsize=9)
    axs[1].set_xticks([]); axs[1].set_yticks([])
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    sl = (period >= 2.6) & (period <= 25.0)
    axs[2].plot(period[sl], wl_col[sl], lw=1.8, color="#1f77b4",
                label=r"$\lambda_{\min}=1/(\sqrt{\lambda_1}+10^{-5})$")
    axs[2].set_xlabel("chu ky that cua soc (pixel)")
    axs[2].set_ylabel(r"$\lambda_{\min}$ (pixel)", labelpad=2)
    axs[2].set_yscale("log"); axs[2].set_xscale("log")
    axs[2].set_xticks([2, 3, 5, 10, 20, 40])
    axs[2].set_xticklabels(["2", "3", "5", "10", "20", "40"])
    axs[2].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs[2].set_yticks([0.2, 0.3, 0.5, 1.0, 2.0])
    axs[2].set_yticklabels(["0.2", "0.3", "0.5", "1", "2"])
    axs[2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs[2].grid(alpha=0.3, which="major"); axs[2].legend(fontsize=8, loc="upper left")
    axs[2].set_title("soc cang min $\\Rightarrow$ $\\lambda_{\\min}$ cang nho", fontsize=9)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "02_wavelength_eigen.png")


# ---------------------------------------------------------------- FIG 5
# Heatmap eta tren luoi (kich thuoc Gaussian) x (buoc song texture)
def fig5():
    axis_px = np.logspace(np.log10(0.05), np.log10(60), 300)     # ||a_j|| pixel
    lam = np.logspace(np.log10(0.5), np.log10(40), 300)          # lambda_min pixel
    A, L = np.meshgrid(axis_px, lam, indexing="ij")
    eta = A / L

    fig, ax = plt.subplots(figsize=(7.2, 5))
    pc = ax.pcolormesh(L, A, np.log10(eta), cmap="RdYlGn_r",
                       vmin=-2.5, vmax=2.5, shading="auto")
    cb = fig.colorbar(pc, ax=ax); cb.set_label(r"$\log_{10}\eta$")
    cs = ax.contour(L, A, eta, levels=[0.1, 1.0], colors=["#333333", "#000000"],
                    linewidths=[1.6, 2.2], linestyles=[":", "-"])
    ax.clabel(cs, fmt={0.1: r"$\tau_{low}=0.1$", 1.0: r"$\tau_{high}=1.0$"}, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"buoc song texture $\lambda_{\min}$ (pixel)")
    ax.set_ylabel(r"do dai truc chieu $\|a_j\|$ (pixel)")
    ax.set_title(r"$\eta=\|a_j\|/\lambda_{\min}$ — 3 vung quyet dinh"
                 "\n(freq_utils.py:348, 395-409)", fontsize=10)
    ax.text(1.2, 25, "SPLIT\n" + r"$\eta>\tau_{high}$", fontsize=11, weight="bold",
            color="#7f0000", ha="center")
    ax.text(6.0, 1.2, "GIU\n" + r"$\tau_{low}<\eta\leq\tau_{high}$", fontsize=10,
            weight="bold", color="#5c4b00", ha="center")
    ax.text(22, 0.13, "PRUNE\n" + r"$\eta\leq\tau_{low}$", fontsize=11, weight="bold",
            color="#0b4a0b", ha="center")
    save(fig, "02_eta_heatmap.png")


# ---------------------------------------------------------------- FIG 6
# Truc giac Nyquist: sine lay mau boi Gaussian to / vua / nho
def fig6():
    x = np.linspace(-12, 12, 2000)
    lam = 4.0                          # buoc song texture (pixel)
    sig = 2.0 * np.pi / lam
    f = np.sin(2 * np.pi * x / lam)

    sizes = [6.0, 1.6, 0.3]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, s in zip(axs, sizes):
        g = np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
        # ket qua loc: Gaussian lam mo sine => bien do nhan exp(-0.5 s^2 omega^2)
        atten = np.exp(-0.5 * (s * sig) ** 2)
        ax.plot(x, f, color="#bbbbbb", lw=1.2, label="texture")
        ax.plot(x, atten * f, color="#1f77b4", lw=2.2,
                label=f"sau khi Gaussian lay mau (x{atten:.2f})")
        ax.plot(x, g / g.max(), color="#E45756", lw=1.8, ls="--",
                label=r"Gaussian $\|a\|=%.2f$" % s)
        eta = s / lam
        lab = ("QUA TO $\\Rightarrow$ SPLIT" if eta > 1.0 else
               ("VUA $\\Rightarrow$ GIU" if eta > 0.1 else "QUA NHO $\\Rightarrow$ PRUNE"))
        ax.set_title(r"$\eta=\|a\|/\lambda_{\min}=%.2f$  —  %s" % (eta, lab), fontsize=9)
        ax.set_xlabel("pixel"); ax.grid(alpha=0.3); ax.set_ylim(-1.3, 1.3)
        ax.legend(fontsize=6.5, loc="lower right")
    axs[0].set_ylabel("cuong do")
    fig.suptitle(r"Nyquist: Gaussian rong hon $\lambda_{\min}$ thi lam mat chi tiet $\Rightarrow$ can SPLIT",
                 fontsize=10)
    save(fig, "02_nyquist.png")


# ---------------------------------------------------------------- FIG 7
# 3 kenh eta_3ch = 3 truc rieng, so sanh 2 mode + phan loai
def fig7():
    rng = np.random.default_rng(7)
    fx = fy = 1000.0
    z = 4.0
    J = np.array([[fx / z, 0, 0], [0, fy / z, 0]])
    lam_min = 3.0
    s = np.array([0.20, 0.08, 0.02])
    R = quat_to_R([0.88, 0.2, 0.15, 0.4])
    axes_cam = R * s[None, :]
    P = J @ axes_cam
    u, v = P[0], P[1]
    L = np.sqrt(u ** 2 + v ** 2)
    eta_w = L / lam_min                                  # mode "wavelength"
    # mode "projection": eta = sqrt(Sxx u^2 + 2 Sxy u v + Syy v^2)
    Sxx, Sxy, Syy = 1.0 / lam_min ** 2, 0.0, 0.25 / lam_min ** 2
    eta_p = np.sqrt(Sxx * u ** 2 + 2 * Sxy * u * v + Syy * v ** 2)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    idx = np.arange(3)
    w = 0.36
    a0.bar(idx - w / 2, eta_w, w, color="#4C78A8", label='mode "wavelength"')
    a0.bar(idx + w / 2, eta_p, w, color="#F58518", label='mode "projection"')
    a0.axhline(1.0, color="#d62728", ls="--", label=r"$\tau_{high}=1.0$")
    a0.axhline(0.1, color="#7f7f7f", ls=":", label=r"$\tau_{low}=0.1$")
    a0.set_yscale("log")
    a0.set_xticks(idx, [r"$\eta_1$ (truc 1)", r"$\eta_2$ (truc 2)", r"$\eta_3$ (truc 3)"])
    a0.set_ylabel(r"$\eta$"); a0.grid(alpha=0.3, axis="y"); a0.legend(fontsize=7.5)
    a0.set_title(r"$\eta_{3ch}$: 1 gia tri / 1 truc chinh cua Gaussian", fontsize=10)

    # phan loai qua nhieu view: eta_max theo view -> high/mid/low count
    nview = 40
    etas = 1.1 * np.exp(rng.normal(0, 1.4, nview))
    high = etas > 1.0
    low = etas <= 0.1
    mid = ~high & ~low
    a1.scatter(np.arange(nview)[high], etas[high], c="#d62728", s=28, label="high (SPLIT)")
    a1.scatter(np.arange(nview)[mid], etas[mid], c="#F58518", s=28, label="mid")
    a1.scatter(np.arange(nview)[low], etas[low], c="#7f7f7f", s=28, label="low (PRUNE)")
    a1.axhline(1.0, color="#d62728", ls="--"); a1.axhline(0.1, color="#7f7f7f", ls=":")
    a1.set_yscale("log"); a1.set_xlabel("view thu i"); a1.set_ylabel(r"$\eta_{\max}=\max_j \eta_j$")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_title("high_ratio = %.2f > 0.8 ?  low_ratio = %.2f > 0.8 ?"
                 % (high.mean(), low.mean()), fontsize=9)
    save(fig, "02_eta_3ch.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("DONE")
