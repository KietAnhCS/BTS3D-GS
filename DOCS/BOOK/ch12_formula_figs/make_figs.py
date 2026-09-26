"""
make_figs.py — Hinh minh hoa CONG THUC cho Chuong 12 (12-adaptive-density-control.md).

QUAN TRONG: day la du lieu TOY/MINH HOA de doc hieu cong thuc, KHONG PHAI ket qua
train that tren GPU (repo khong co history.csv/leaderboard.csv that cho SADGS,
xem Chuong 14). Moi con so trong cac ham duoi day duoc chon de di dung cong thuc
that trong SADGS/*.py (trich dan o docstring tung ham), khong phai so do dac.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 10.5,
})

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
GRAY = "#718096"
PURPLE = "#6b46c1"


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Fig 1 — 12.0: ADC goc 3DGS — gradient threshold + clone/split theo max(scale)
# ---------------------------------------------------------------------------
def fig1_gradient_adc_3dgs():
    rng = np.random.default_rng(0)
    n = 220
    grad = rng.gamma(shape=1.5, scale=6e-5, size=n)
    scale = rng.lognormal(mean=np.log(0.01), sigma=0.9, size=n)
    tau_grad = 2e-4
    delta_extent = 0.012  # dense * extent minh hoa

    above = grad >= tau_grad
    clone = above & (scale <= delta_extent)
    split = above & (scale > delta_extent)
    keep = ~above

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(grad[keep], scale[keep], s=18, c=GRAY, alpha=0.6, label="giữ nguyên (grad < τ)")
    ax.scatter(grad[clone], scale[clone], s=34, c=BLUE, alpha=0.85, label="clone (grad ≥ τ, nhỏ)")
    ax.scatter(grad[split], scale[split], s=34, c=ORANGE, alpha=0.85, label="split (grad ≥ τ, to)")
    ax.axvline(tau_grad, color=RED, ls="--", lw=1.4)
    ax.axhline(delta_extent, color=RED, ls="--", lw=1.4)
    ax.text(tau_grad * 1.05, ax.get_ylim()[1] * 0.85, r"$\bar g_i \geq \tau_{\mathrm{grad}}$",
            color=RED, fontsize=10)
    ax.text(grad.max() * 0.55, delta_extent * 1.15, r"$\max s_i > \delta \cdot \mathrm{extent}$",
            color=RED, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\bar g_i = \|\mathrm{accum}_i / \mathrm{denom}_i\|$  (gradient màn hình tích luỹ)")
    ax.set_ylabel(r"$\max_j s_i^{(j)}$  (trục scale lớn nhất, log)")
    ax.set_title("§12.0 — ADC gốc 3DGS: một tiêu chí gradient, chia theo 2 vùng\n"
                  "(dữ liệu TOY minh hoạ quy tắc, không phải số đo thật)")
    ax.legend(loc="lower right", framealpha=0.95)
    savefig(fig, "12_fig1_gradient_adc_3dgs.png")


# ---------------------------------------------------------------------------
# Fig 2 — 12.3.3: eta = do dai truc chieu / buoc song cuc bo (wavelength mode)
# ---------------------------------------------------------------------------
def fig2_eta_wavelength():
    x = np.linspace(0, 40, 800)
    wavelength = 6.0  # px, w_min minh hoa
    texture = 0.5 + 0.5 * np.sin(2 * np.pi * x / wavelength)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(x, texture, color=GRAY, lw=1.4, label="texture GT cục bộ (tần số cao nhất)")
    ax.fill_between(x, 0, texture, color=GRAY, alpha=0.12)

    # Gaussian nho: truc = 3px < wavelength -> eta < 1
    cx1, ell1 = 10, 3.0
    ax.add_patch(patches.Ellipse((cx1, 0.5), width=2 * ell1, height=0.9,
                                  fc=BLUE, ec=BLUE, alpha=0.35, lw=1.6))
    ax.annotate("", xy=(cx1 - ell1, 1.35), xytext=(cx1 + ell1, 1.35),
                arrowprops=dict(arrowstyle="<->", color=BLUE))
    ax.text(cx1, 1.45, r"$\ell = 3$px, $\eta=\ell/w_{\min}=0.5$" "\n(nhỏ hơn texture ⇒ ổn)",
            ha="center", color=BLUE, fontsize=9)

    # Gaussian to: truc = 14px > wavelength -> eta > 1
    cx2, ell2 = 28, 14.0
    ax.add_patch(patches.Ellipse((cx2, 0.5), width=2 * ell2, height=0.9,
                                  fc=ORANGE, ec=ORANGE, alpha=0.35, lw=1.6))
    ax.annotate("", xy=(cx2 - ell2, -0.45), xytext=(cx2 + ell2, -0.45),
                arrowprops=dict(arrowstyle="<->", color=ORANGE))
    ax.text(cx2, -0.6, r"$\ell = 14$px, $\eta=\ell/w_{\min}\approx2.33 > 1$" "\n(dài hơn cả chu kỳ texture ⇒ split)",
            ha="center", color=ORANGE, fontsize=9)

    ax.set_xlim(0, 40)
    ax.set_ylim(-1.0, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("vị trí trên màn hình (pixel)")
    ax.set_title(r"§12.3.3 — $\eta=\ell/w_{\min}$: so độ dài trục chiếu Gaussian với bước sóng texture cục bộ"
                 "\n(1D minh hoạ; $w_{\\min}=1/(\\sqrt{\\lambda_1}+\\varepsilon)$ đo từ structure tensor)")
    savefig(fig, "12_fig2_eta_wavelength.png")


# ---------------------------------------------------------------------------
# Fig 3 — 12.3.3: wavelength vs projection mode tren mot canh nghieng
# ---------------------------------------------------------------------------
def fig3_eta_modes_compare():
    # Structure tensor cua mot canh gan doc (bien thien manh theo x)
    Sxx, Sxy, Syy = 1.0, 0.0, 0.05
    tr = Sxx + Syy
    det = Sxx * Syy - Sxy ** 2
    lam1 = tr / 2 + np.sqrt(max((tr / 2) ** 2 - det, 0))

    thetas = np.linspace(0, np.pi, 200)
    # truc Gaussian huong theta, do dai co dinh minh hoa
    ell = 10.0
    u, v = np.cos(thetas), np.sin(thetas)

    eta_wave = np.full_like(thetas, ell * np.sqrt(lam1))  # khong phu thuoc huong
    eta_proj = ell * np.sqrt(np.clip(Sxx * u ** 2 + 2 * Sxy * u * v + Syy * v ** 2, 0, None))

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.plot(np.degrees(thetas), eta_wave, color=BLUE, lw=2.2, label=r'mode "wavelength" — $\eta$ hằng số, không nhìn hướng trục')
    ax.plot(np.degrees(thetas), eta_proj, color=ORANGE, lw=2.2, label=r'mode "projection" — $\eta=\ell\sqrt{S_{xx}u^2+2S_{xy}uv+S_{yy}v^2}$')
    ax.axhline(1.0, color=RED, ls="--", lw=1.2, label=r"ngưỡng split $\tau_{\text{high}}=1.0$")
    ax.axvline(0, color=GREEN, ls=":", lw=1.2)
    ax.axvline(90, color=GREEN, ls=":", lw=1.2)
    ax.text(2, ax.get_ylim()[1] * 0.94, "trục song song cạnh\n(θ=0°)", color=GREEN, fontsize=8.5)
    ax.text(92, ax.get_ylim()[1] * 0.94, "trục vuông góc cạnh\n(θ=90°)", color=GREEN, fontsize=8.5)
    ax.set_xlabel(r"hướng trục Gaussian $\theta$ so với cạnh (độ)")
    ax.set_ylabel(r"$\eta(\theta)$")
    ax.set_title("§12.3.3 — Cùng một Gaussian, hai công thức $\\eta$ cho kết quả khác nhau theo hướng\n"
                 r"(cạnh gần-dọc minh hoạ: $S_{xx}\gg S_{yy}$)")
    ax.legend(loc="upper center", framealpha=0.95, fontsize=9)
    savefig(fig, "12_fig3_eta_modes_compare.png")


# ---------------------------------------------------------------------------
# Fig 4 — 12.4: phan loai high/mid/low qua nhieu view + ti le
# ---------------------------------------------------------------------------
def fig4_highmidlow_classify():
    rng = np.random.default_rng(3)
    n_views = 50
    # mot Gaussian "that su qua to": phan lon view cho eta cao
    eta_seq = np.clip(rng.normal(1.6, 0.5, n_views), 0.02, None)
    tau_high, tau_low = 1.0, 0.1
    is_high = eta_seq > tau_high
    is_low = eta_seq <= tau_low
    is_mid = ~is_high & ~is_low
    high_ratio = is_high.mean()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    colors = np.where(is_high, ORANGE, np.where(is_low, BLUE, GRAY))
    ax.bar(np.arange(n_views), eta_seq, color=colors, width=0.8)
    ax.axhline(tau_high, color=RED, ls="--", lw=1.3, label=r"$\tau_{\text{high}}=1.0$")
    ax.axhline(tau_low, color=PURPLE, ls="--", lw=1.3, label=r"$\tau_{\text{low}}=0.1$")
    ax.set_xlabel("chỉ số lần quan sát (mỗi 10 vòng lặp × mỗi view thấy Gaussian)")
    ax.set_ylabel(r"$\eta^{\max}_i$ đo được ở view đó")
    ax.set_title(r"§12.4 — Chuỗi quan sát $\eta^{\max}$ của MỘT Gaussian qua $50$ lần nhìn thấy")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[1]
    counts = [is_high.sum(), is_mid.sum(), is_low.sum()]
    labels = [f"high\n({is_high.sum()})", f"mid\n({is_mid.sum()})", f"low\n({is_low.sum()})"]
    ax.bar(labels, counts, color=[ORANGE, GRAY, BLUE])
    ax.set_ylabel("số lần quan sát (accum_*_count)")
    ax.set_title(f"high_ratio = {is_high.sum()}/{n_views} = {high_ratio:.2f}\n"
                 f"> split_ratio_threshold (0.8) ⇒ {'SPLIT' if high_ratio > 0.8 else 'chưa đủ'}")
    fig.suptitle("Multi-view consistency: chỉ quyết định khi ĐA SỐ view đồng thuận, không phải 1 view lệch",
                 y=1.06, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "12_fig4_highmidlow_classify.png")


# ---------------------------------------------------------------------------
# Fig 5 — 12.6: k_j = ceil(sqrt(eta_j)) va N_con = kx*ky*kz
# ---------------------------------------------------------------------------
def fig5_split_k_formula():
    eta = np.linspace(0.01, 12, 600)
    k = np.ceil(np.sqrt(np.clip(eta, 1.0, None))).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.plot(eta, k, color=BLUE, lw=2.2, drawstyle="steps-post")
    ax.axvline(1.0, color=RED, ls="--", lw=1.2, label=r"$\eta=1$ (ngưỡng bắt đầu chia)")
    for kk in range(1, 5):
        lo = kk ** 2 if kk > 1 else 0
        hi = (kk + 1) ** 2
        ax.axvspan(max(lo, 0.01), hi, color=BLUE, alpha=0.05 if kk % 2 else 0.1)
    ax.set_xlabel(r"$\eta^{(j)}$ (một trục)")
    ax.set_ylabel(r"$k_j=\lceil\sqrt{\max(\eta_j,1)}\rceil$")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_title(r"§12.6 — Số bản mỗi trục là hàm bậc thang của $\eta$" "\n(gaussian_model.py:699)")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    combos = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2), (3, 2, 1), (3, 3, 3)]
    Ncon = [kx * ky * kz for kx, ky, kz in combos]
    labels = [f"({kx},{ky},{kz})" for kx, ky, kz in combos]
    bars = ax.bar(labels, Ncon, color=GREEN)
    for b, n in zip(bars, Ncon):
        ax.text(b.get_x() + b.get_width() / 2, n + 0.3, str(n), ha="center", fontsize=9)
    ax.axhline(2, color=GRAY, ls=":", lw=1.4, label="N=2 cố định (3DGS/FastGS gốc)")
    ax.set_xlabel(r"$(k_x,k_y,k_z)$")
    ax.set_ylabel(r"$N_{\text{con}}=k_xk_yk_z$")
    ax.set_title("So Gaussian con: cố định 2 (3DGS/FastGS)\nvs thay đổi theo mức vi phạm từng trục (SADGS)")
    ax.legend(fontsize=9)
    savefig(fig, "12_fig5_split_k_formula.png")


# ---------------------------------------------------------------------------
# Fig 6 — 12.7: tran opacity 0.8 va opacity reset qua inverse_sigmoid
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(p):
    return np.log(p / (1 - p))


def fig6_opacity_cap_reset():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    # Panel a: quy dinh cua alpha qua 3 lan densify + 1 lan reset (toy)
    ax = axes[0]
    t = np.linspace(0, 3200, 400)
    alpha = 0.05 + 0.4 * (1 - np.exp(-t / 900))
    alpha = np.clip(alpha, 0, 0.95)
    densify_at = [500, 1000, 1500, 2000, 2500]
    for d in densify_at:
        alpha = np.where(t >= d, np.minimum(alpha, np.where(t >= d, 0.8, alpha)), alpha)
    reset_at = 3000
    alpha_reset = np.where(t >= reset_at, np.minimum(alpha, 0.1), alpha)

    ax.plot(t, np.clip(0.05 + 0.4 * (1 - np.exp(-t / 900)) + 0.5 * (t > 2600) * (1 - np.exp(-(t - 2600) / 300)),
                        0, 0.97), color=GRAY, lw=1.6, ls="--", label=r"$\alpha$ nếu KHÔNG có trần/reset (minh hoạ)")
    ax.plot(t, alpha_reset, color=BLUE, lw=2.2, label=r"$\alpha$ thực tế (trần 0.8 + reset 0.1)")
    for d in densify_at:
        ax.axvline(d, color=ORANGE, ls=":", lw=1, alpha=0.7)
    ax.axvline(reset_at, color=RED, ls="--", lw=1.6)
    ax.axhline(0.8, color=ORANGE, ls="--", lw=1.2)
    ax.axhline(0.1, color=RED, ls=":", lw=1.2)
    ax.text(reset_at + 40, 0.55, "opacity_reset\n" r"$\alpha\!\leftarrow\!\sigma^{-1}(\min(\alpha,0.1))$", color=RED, fontsize=8.5)
    ax.text(60, 0.82, r"trần $\alpha\leftarrow\min(\alpha,0.8)$ sau mỗi densify", color=ORANGE, fontsize=8.5)
    ax.set_xlabel("iteration (minh hoạ, không phải log thật)")
    ax.set_ylabel(r"$\alpha$ (opacity)")
    ax.set_title(r"§12.7 — Trần $\alpha\leq0.8$ sau mỗi densify, reset $\alpha\leq0.1$ mỗi opacity_reset_interval")
    ax.legend(fontsize=8, loc="center right")

    # Panel b: sigmoid / inverse_sigmoid
    ax = axes[1]
    x = np.linspace(-6, 6, 400)
    ax.plot(x, sigmoid(x), color=BLUE, lw=2.2, label=r"$\sigma(x)=1/(1+e^{-x})$")
    p = np.linspace(0.01, 0.99, 400)
    ax2 = ax.twiny()
    ax2.plot(inv_sigmoid(p), p, color=ORANGE, lw=2.2, label=r"$\sigma^{-1}(p)=\log(p/(1-p))$")
    ax2.set_xlabel(r"$\sigma^{-1}(p)$  (trục trên, dùng để LƯU opacity dạng logit)", color=ORANGE)
    ax.set_xlabel(r"$x$  (trục dưới)")
    ax.set_ylabel(r"$\alpha=\sigma(x)$")
    ax.set_title(r"Vì sao cần $\sigma^{-1}$: opacity lưu ở dạng logit," "\n" r"reset $\alpha=0.1$ nghĩa là gán $x=\sigma^{-1}(0.1)\approx-2.2$")
    ax.axhline(0.1, color=RED, ls=":", lw=1)
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    savefig(fig, "12_fig6_opacity_cap_reset.png")


# ---------------------------------------------------------------------------
# Fig 7 — 12.8: lich chay day du trong train.py (timeline)
# ---------------------------------------------------------------------------
def fig7_schedule_timeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.2))

    densify_from, densify_until = 500, 15000
    densification_interval = 100
    opacity_reset_interval = 3000
    total_iters = 16000

    ax.axhspan(-0.5, 4.5, xmin=densify_from / total_iters, xmax=densify_until / total_iters,
               color=BLUE, alpha=0.06)
    ax.text((densify_from + densify_until) / 2, 4.3, "cửa sổ densify [500, 15000)",
            ha="center", color=BLUE, fontsize=9)

    # eta-accum moi 10 vong (qua day -> ve nhu mot dai lien tuc)
    ax.broken_barh([(0, densify_until)], (3.6, 0.5), facecolors=GREEN, alpha=0.5)
    ax.text(200, 3.85, r"$\eta$-accum: update_freq_stats_online mỗi 10 vòng", color=GREEN, fontsize=8.5, va="center")

    # grad-accum
    ax.broken_barh([(0, densify_until)], (2.6, 0.5), facecolors=GRAY, alpha=0.6)
    ax.text(200, 2.85, "grad-accum: add_densification_stats — MỖI vòng", color="#444", fontsize=8.5, va="center")

    # densify+prune moi 100 vong
    dp_iters = np.arange(densify_from + densification_interval, densify_until, densification_interval)
    ax.scatter(dp_iters, np.full_like(dp_iters, 1.9), marker="v", color=ORANGE, s=28,
               label=f"densify_and_prune_structgs (mỗi {densification_interval}, {len(dp_iters)} lần)")

    # opacity reset
    reset_iters = np.arange(opacity_reset_interval, densify_until, opacity_reset_interval)
    ax.scatter(reset_iters, np.full_like(reset_iters, 1.1), marker="D", color=RED, s=40,
               label=f"opacity_reset (mỗi {opacity_reset_interval}, {len(reset_iters)} lần)")

    # prune-only
    prune_iters = [4000, 8000]
    ax.scatter(prune_iters, np.full_like(prune_iters, 0.3, dtype=float), marker="x", color=PURPLE, s=60,
               label="prune-only [α<0.1] tại prune_iterations={4000,8000}")

    ax.axvline(densify_until, color=BLUE, ls="--", lw=1.3)
    ax.text(densify_until + 100, 0.0, "densify_until_iter=15000\n(final_prune_structgs tồn tại\nnhưng lời gọi BỊ COMMENT)",
            color=BLUE, fontsize=8, va="bottom")

    ax.set_xlim(0, total_iters)
    ax.set_ylim(-0.6, 5.0)
    ax.set_yticks([])
    ax.set_xlabel("iteration")
    ax.set_title("§12.8 — Lịch chạy đầy đủ trong train.py (cấu hình mặc định)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    savefig(fig, "12_fig7_schedule_timeline.png")


if __name__ == "__main__":
    fig1_gradient_adc_3dgs()
    fig2_eta_wavelength()
    fig3_eta_modes_compare()
    fig4_highmidlow_classify()
    fig5_split_k_formula()
    fig6_opacity_cap_reset()
    fig7_schedule_timeline()
