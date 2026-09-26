"""
Test số cho Chương 7 — Structure-Aware Densification (SADGS, kế thừa khung Adaptive Density
Control của 3DGS/FastGS nhưng thay tiêu chí bằng multiscale structure-tensor + multiview
consistency). Cảnh đồ chơi: DOCS/Report/test/00-scene.md (4 điểm SfM, 3 camera, ảnh 48x32, fx=fy=40).

Chỉ dùng numpy + scipy (không có torch). Seed RNG = 0.

Các công thức được CHÉP LẠI từ mã nguồn SADGS thật (SADGS/):
  - scene/gaussian_model.py:262-284       : create_from_pcd (khởi tạo chương 1, tính lại tại đây)
  - submodules/diff-gaussian-rasterization_structgs/cuda_rasterizer/forward.cu:79-118 : computeCov2D
  - .../forward.cu:238-264                : conic, r2D = ceil(3 sqrt(lambda_max)), ndc2Pix
  - .../cuda_rasterizer/auxiliary.h:313-369: duplicateToTilesTouched (SnugBox), t=mult*2 ln(255 alpha),
                                             mult = args.mult = 0.7 (SADGS; FastGS cũ dùng 0.5)
  - utils/loss_utils.py:22,46-76          : l1_loss, ssim (gaussian 11x11 sigma=1.5, zero-pad,
                                             C1=1e-4, C2=9e-4 — công thức không đổi so với 3DGS gốc)
  - utils/freq_utils.py:92-96             : get_loss (l1_loss_norm minmax) — còn trong code nhưng
                                             không được gọi trong vòng lặp train.py hiện tại
  - utils/freq_utils.py:181-415           : update_freq_stats_online — thống kê eta đa-view trực
                                             tuyến (THAY THẾ compute_gaussian_score_fastgs cũ)
  - train.py:315-386                      : vòng lặp densify SADGS thật (split_mask/prune_mask từ
                                             high_ratio/low_ratio, gọi densify_and_prune_structgs)
  - scene/gaussian_model.py:638-831       : densify_and_split_structgs (split dị hướng giải tích)
  - scene/gaussian_model.py:934-955       : densify_and_clone_structgs
  - scene/gaussian_model.py:957-1052      : densify_and_prune_structgs
  - scene/gaussian_model.py:1059-1066     : final_prune_structgs (hiện không được gọi mặc định)
  - scene/gaussian_model.py:415-419       : reset_opacity = opacity * opacity_reset_decay (0.1)
  - scene/gaussian_model.py:1054-1057     : add_densification_stats (norm cột 0-1 và cột 2-3)
  - arguments/__init__.py                 : freq_grad_threshold, importance_score_threshold,
                                             split_ratio_threshold, prune_ratio_threshold,
                                             clone_target_eta, ks_scale_power, max_clones_per_axis,
                                             tau_expand, dense (percent_dense cũ)
  - utils/graphics_utils.py getNerfppNorm : extent = 1.1 * max ||c_v - c_bar||  (không đổi)

QUAN TRỌNG — khác biệt với FastGS cũ (ch07 bản trước dùng compute_gaussian_score_fastgs,
densify_and_split_fastgs lấy mẫu Gauss ngẫu nhiên N(0,s^2), Pruning minmax + multinomial mặc
định BẬT): hàm compute_gaussian_score_structgs mà train.py gọi (dòng 418) ĐÃ BỊ COMMENT và
KHÔNG CÒN ĐỊNH NGHĨA nào trong repo SADGS (dead code). Cơ chế đang hoạt động thật sự là:
  1) update_freq_stats_online tích luỹ eta (tỉ lệ vi phạm Nyquist màn hình/kết cấu ảnh) mỗi view,
     phân loại high/mid/low theo 2 ngưỡng cứng TAU_HIGH=1.0, TAU_LOW=0.1 (freq_utils.py, không
     phải tham số args);
  2) high_ratio/low_ratio (tỉ lệ view mà GS "high"/"low") so với split_ratio_threshold/
     prune_ratio_threshold quyết định split_mask/prune_mask đa-view;
  3) densify_and_prune_structgs hợp các mặt nạ này với grad_thresh/grad_abs_thresh kiểu 3DGS gốc
     và cổng importance_score (= accum_view_count) > importance_score_threshold;
  4) split dùng lưới giải tích k=ceil(sqrt(eta)) mỗi trục (không phải lấy mẫu Gauss ngẫu nhiên);
  5) nhánh xoá có-trọng-số (multinomial) vẫn còn trong mã nhưng train.py truyền
     pruning_score=None nên KHÔNG kích hoạt mặc định — xoá thẳng theo prune_mask nhị phân.

Vì cảnh đồ chơi 4 điểm không có texture/CNN Sobel thật để tính multiscale structure tensor
(get_multiscale_structure_tensor_v1/v2, utils/loss_utils.py), phần eta dưới đây dùng MỘT PROXY
rõ ràng (bán kính màn hình / bước sóng cục bộ ước lượng từ độ lệch chuẩn ảnh GT quanh tâm mỗi
Gaussian) nhưng giữ đúng công thức, tên biến, tên tham số và luồng điều khiển của code thật.

Renderer numpy tối giản (lấy từ chương 3–5, không đổi so với bản trước): khởi tạo mu = p,
s = sqrt(mean d^2 tới 3 láng giềng), R = I, Sigma = s^2 I, màu = c_k (SH bậc 0);
projection mỗi camera: t = mu - c_v; mu' = ndc2Pix; J clamp 1.3 tan; Sigma' = J Sigma J^T + 0.3 I;
conic; render mọi Gaussian theo depth với 3 cửa loại, nền trắng. Renderer trả thêm footprint hữu hình
Omega_i^{(v)} = tập pixel mà Gaussian i thực sự đóng góp (qua được 3 cửa loại).

Chạy:  python DOCS/Report/test/scripts/ch07_test.py
"""
import numpy as np
from scipy.ndimage import convolve

np.set_printoptions(precision=6, suppress=True, linewidth=140)
SEED = 0

# ----------------------------------------------------------------------------
# Hằng số (chương 0)
# ----------------------------------------------------------------------------
W, H = 48, 32
TANX, TANY = 0.6, 0.4
FX, FY = W / (2 * TANX), H / (2 * TANY)          # 40, 40
BLOCK = 16
GRID_X, GRID_Y = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK   # 3 x 2 tile
LOWPASS = 0.3
ALPHA_MIN = 1.0 / 255.0
T_EARLY = 1e-4
MULT = 0.7          # arguments/__init__.py: mult = 0.7 (SADGS; FastGS cũ dùng 0.5)
TAU_GRAD = 2e-4      # arguments/__init__.py: grad_thresh = 0.0002
TAU_GRAD_ABS = 2e-4  # arguments/__init__.py: grad_abs_thresh = 0.0002 (FastGS cũ dùng 1.2e-3)
TAU_LOSS = 0.1
DELTA = 0.001        # arguments/__init__.py: dense = 0.001 (percent_dense cũ)
LAMBDA = 0.2
V_CAMS = 3       # 3 camera thay vì V=10 của sampling_cameras — cảnh chỉ có 3 camera

# Cảnh đồ chơi (00-scene.md)
P = np.array([[0.0, 0.0, 0.0],
              [0.5, 0.3, 0.5],
              [-0.4, -0.2, 1.0],
              [0.3, -0.5, 0.2]])
COL = np.array([[0.8, 0.2, 0.2],
                [0.2, 0.7, 0.3],
                [0.1, 0.3, 0.9],
                [0.5, 0.5, 0.5]])
CAM = np.array([[0.0, 0.0, -4.0],
                [1.5, 0.0, -4.0],
                [-1.5, 0.5, -4.0]])
N = P.shape[0]
BG = np.ones(3)  # nền trắng


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


# ============================================================================
# Khởi tạo (chương 1/2 tính lại)
# ============================================================================
hdr("[CH1-2 tính lại] Khởi tạo: mu = p, s = sqrt(mean d^2 3-NN), R = I, Sigma = s^2 I, màu = c_k")
D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
dist2 = np.zeros(N)
for i in range(N):
    dist2[i] = np.sort(np.delete(D2[i], i))[:3].mean()
dist2 = np.maximum(dist2, 1e-7)
S = np.sqrt(dist2)[:, None].repeat(3, 1)          # scale (đẳng hướng)
S_TILDE = np.log(S)
R_ROT = np.tile(np.eye(3), (N, 1, 1))              # q = (1,0,0,0) -> R = I
SIGMA3 = np.array([np.diag(S[i] ** 2) for i in range(N)])
ALPHA_MODEL = 0.1 * np.ones(N)
ALPHA_GT = 0.9 * np.ones(N)
for i in range(N):
    print(f"G{i+1}: mu={P[i]}  s={S[i,0]:.6f}  s~={S_TILDE[i,0]:.6f}  alpha_model=0.1  alpha_gt=0.9  c={COL[i]}")


# ============================================================================
# Projection (chương 3) — chép từ forward.cu computeCov2D / preprocessCUDA
# ============================================================================
def ndc2Pix(v, S_):
    return ((v + 1.0) * S_ - 1.0) * 0.5


def project(mu, Sigma, cam):
    """Trả về mu' (pixel), Sigma' (2x2), conic (A,B,C), depth, r2D, và t (camera space)."""
    t = mu - cam                                   # R_cam = I  -> t = mu - c_v
    tx, ty, tz = t
    # NDC theo getProjectionMatrix: x_ndc = tx / (tanx * tz)
    ndc = np.array([tx / (TANX * tz), ty / (TANY * tz)])
    mu2 = np.array([ndc2Pix(ndc[0], W), ndc2Pix(ndc[1], H)])
    # computeCov2D
    limx, limy = 1.3 * TANX, 1.3 * TANY
    txc = min(limx, max(-limx, tx / tz)) * tz
    tyc = min(limy, max(-limy, ty / tz)) * tz
    J = np.array([[FX / tz, 0.0, -(FX * txc) / (tz * tz)],
                  [0.0, FY / tz, -(FY * tyc) / (tz * tz)]])
    Sig2 = J @ Sigma @ J.T
    Sig2[0, 0] += LOWPASS
    Sig2[1, 1] += LOWPASS
    det = Sig2[0, 0] * Sig2[1, 1] - Sig2[0, 1] ** 2
    conic = np.array([Sig2[1, 1] / det, -Sig2[0, 1] / det, Sig2[0, 0] / det])
    mid = 0.5 * (Sig2[0, 0] + Sig2[1, 1])
    lam1 = mid + np.sqrt(max(0.1, mid * mid - det))
    lam2 = mid - np.sqrt(max(0.1, mid * mid - det))
    r2d = int(np.ceil(3.0 * np.sqrt(max(lam1, lam2))))
    return mu2, Sig2, conic, tz, r2d, t


def tiles_touched(mu2, conic, alpha):
    """Compact box (auxiliary.h:313-369, SnugBox): Gaussian chỉ được gán vào tile giao với ellipse
       Delta^T M Delta <= t_i,  t_i = mult * 2 ln(255 alpha).  Ở đây kiểm tra giao tile–ellipse chính xác
       bằng cách lấy min của dạng toàn phương lồi trên hình chữ nhật tile (biên + tâm)."""
    A, B, C = conic
    t_lvl = MULT * 2.0 * np.log(alpha * 255.0)
    if t_lvl <= 0:
        return np.zeros((GRID_Y, GRID_X), dtype=bool)

    def q(dx, dy):
        return A * dx * dx + 2 * B * dx * dy + C * dy * dy

    def min_on_rect(x0, x1, y0, y1):
        # tâm ellipse nằm trong tile
        if x0 <= mu2[0] <= x1 and y0 <= mu2[1] <= y1:
            return 0.0
        best = np.inf
        # 4 cạnh: cố định x hoặc y, tối thiểu 1D rồi clamp
        for xf in (x0, x1):
            dx = xf - mu2[0]
            dy_star = -B * dx / C
            dy = min(max(dy_star, y0 - mu2[1]), y1 - mu2[1])
            best = min(best, q(dx, dy))
        for yf in (y0, y1):
            dy = yf - mu2[1]
            dx_star = -B * dy / A
            dx = min(max(dx_star, x0 - mu2[0]), x1 - mu2[0])
            best = min(best, q(dx, dy))
        return best

    out = np.zeros((GRID_Y, GRID_X), dtype=bool)
    for ty in range(GRID_Y):
        for tx in range(GRID_X):
            x0, x1 = tx * BLOCK, min((tx + 1) * BLOCK, W) - 1
            y0, y1 = ty * BLOCK, min((ty + 1) * BLOCK, H) - 1
            out[ty, tx] = min_on_rect(x0, x1, y0, y1) <= t_lvl
    return out


# ============================================================================
# Rasterizer (chương 4) — renderCUDA với 3 cửa loại + footprint hữu hình
# ============================================================================
PIX_X, PIX_Y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
TILE_OF_PIX = (PIX_Y // BLOCK).astype(int), (PIX_X // BLOCK).astype(int)


def prep_view(cam, alpha_vec, mu_override=None):
    """Tiền xử lý mọi Gaussian cho 1 camera. mu_override: dict {i: mu2' (pixel)} để sai phân hữu hạn."""
    items = []
    for i in range(N):
        mu2, Sig2, conic, depth, r2d, t = project(P[i], SIGMA3[i], cam)
        if mu_override is not None and i in mu_override:
            mu2 = mu_override[i]
        tiles = tiles_touched(mu2, conic, alpha_vec[i])
        items.append(dict(i=i, mu2=mu2, Sig2=Sig2, conic=conic, depth=depth, r2d=r2d,
                          tiles=tiles, alpha=alpha_vec[i], color=COL[i], t=t))
    return items


def render(items):
    """Trả về ảnh [3,H,W], footprint bool [N,H,W], T_final."""
    order = sorted(range(N), key=lambda k: items[k]["depth"])   # sort theo depth
    C = np.zeros((3, H, W))
    T = np.ones((H, W))
    done = np.zeros((H, W), dtype=bool)
    foot = np.zeros((N, H, W), dtype=bool)
    for k in order:
        g = items[k]
        A, B, Cc = g["conic"]
        dx = g["mu2"][0] - PIX_X
        dy = g["mu2"][1] - PIX_Y
        power = -0.5 * (A * dx * dx + Cc * dy * dy) - B * dx * dy
        alpha = np.minimum(0.99, g["alpha"] * np.exp(power))
        in_tile = g["tiles"][TILE_OF_PIX]                       # cửa 1: cull tile
        active = (~done) & in_tile & (power <= 0) & (alpha >= ALPHA_MIN)   # cửa 2: alpha >= 1/255
        test_T = T * (1 - alpha)
        newly_done = active & (test_T < T_EARLY)                # cửa 3: T < 1e-4 -> dừng
        contrib = active & ~newly_done
        for ch in range(3):
            C[ch] += np.where(contrib, g["color"][ch] * alpha * T, 0.0)
        T = np.where(contrib, test_T, T)
        done |= newly_done
        foot[g["i"]] = contrib
    C += T[None] * BG[:, None, None]                            # nền trắng
    return C, foot, T


# ============================================================================
# Loss (chương 5) — chép utils/loss_utils.py
# ============================================================================
def gaussian_1d(window_size=11, sigma=1.5):
    g = np.array([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    return g / g.sum()


WIN = np.outer(gaussian_1d(), gaussian_1d())       # 11x11


def conv(img):  # zero-pad như F.conv2d(padding=5)
    return convolve(img, WIN, mode="constant", cval=0.0)


def ssim(img1, img2):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        a, b = img1[ch], img2[ch]
        mu1, mu2 = conv(a), conv(b)
        s1 = conv(a * a) - mu1 ** 2
        s2 = conv(b * b) - mu2 ** 2
        s12 = conv(a * b) - mu1 * mu2
        m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        vals.append(m)
    return float(np.mean(vals))


def l1(img1, img2):
    return float(np.abs(img1 - img2).mean())


# ============================================================================
# Render GT (alpha=0.9) và mô hình (alpha=0.1) cho 3 camera
# ============================================================================
hdr("[Render] GT (alpha=0.9) và mô hình (alpha=0.1), 3 camera — projection + footprint hữu hình")
GT, REND, FOOT, ITEMS = [], [], [], []
for v in range(V_CAMS):
    it_gt = prep_view(CAM[v], ALPHA_GT)
    it_md = prep_view(CAM[v], ALPHA_MODEL)
    img_gt, _, _ = render(it_gt)
    img_md, foot, Tf = render(it_md)
    GT.append(img_gt); REND.append(img_md); FOOT.append(foot); ITEMS.append(it_md)
    print(f"\n-- camera {v+1}, c_v={CAM[v]} --")
    for g in it_md:
        i = g["i"]
        print(f"G{i+1}: t={g['t']}  mu'=({g['mu2'][0]:.4f},{g['mu2'][1]:.4f})  "
              f"Sigma'=[[{g['Sig2'][0,0]:.4f},{g['Sig2'][0,1]:.4f}],[.,{g['Sig2'][1,1]:.4f}]]  "
              f"conic=({g['conic'][0]:.5f},{g['conic'][1]:.5f},{g['conic'][2]:.5f})  depth={g['depth']:.3f}  "
              f"r2D={g['r2d']}  tiles={g['tiles'].sum()}/6  |Omega|={int(foot[i].sum())}px")
    print(f"T_final: min={Tf.min():.4f} max={Tf.max():.4f}")

# ============================================================================
# 7.2 — SADGS: thống kê eta đa-view online (thay compute_gaussian_score_fastgs)
# ============================================================================
hdr("[7.2] SADGS bỏ compute_gaussian_score_fastgs (utils/fast_utils.py cũ không còn trong repo).\n"
    "Cơ chế mới: utils/freq_utils.py:181-415 update_freq_stats_online() tích luỹ MỖI VIEW, gọi\n"
    "trong vòng lặp densify (train.py:276). Một Gaussian được đếm là 'active' tại view v nếu CẢ 3:\n"
    "  max_transmittance > freq_transmittance_threshold (0.0)\n"
    "  opacity            > freq_opacity_threshold       (0.05)\n"
    "  ||grad_xy||        > freq_grad_threshold           (0.00002)")

FREQ_GRAD_THR = 0.00002
FREQ_OPACITY_THR = 0.05
FREQ_TRANS_THR = 0.0
TAU_HIGH, TAU_LOW = 1.0, 0.1     # freq_utils.py:395-396 (hằng số trong code, KHÔNG phải arguments)

hdr("[7.2] get_loss (freq_utils.py:92-96, l1_loss_norm minmax) vẫn còn trong code, giữ lại để minh\n"
    "hoạ kênh sai số ảnh, nhưng compute_gaussian_score_structgs mà train.py:418 gọi (bị comment) đã\n"
    "KHÔNG CÒN ĐỊNH NGHĨA nào trong repo SADGS -- dead code, không dùng để suy ra importance/pruning.")
E, EHAT, MASK = [], [], []
for v in range(V_CAMS):
    e = np.abs(REND[v] - GT[v]).mean(0)
    eh = (e - e.min()) / (e.max() - e.min() + 1e-12)
    m = (eh > TAU_LOSS).astype(int)
    E.append(e); EHAT.append(eh); MASK.append(m)
    print(f"view {v+1}: e min={e.min():.6f} max={e.max():.6f} mean={e.mean():.6f}  bật={int(m.sum())}/{H*W}px")

hdr("[7.2] Proxy eta_i^(v) cho cảnh đồ chơi (không có multiscale structure tensor thật trên 4 điểm):\n"
    "eta = axis_length_px / wavelength_min   (freq_utils.py:338-348, eta_compute_mode='wavelength')\n"
    "  axis_length_px : bán kính màn hình Gaussian, từ trị riêng của Sigma' (computeCov2D)\n"
    "  wavelength_min : proxy 1/sqrt(lambda1 cấu trúc tensor) ~ 1/(std ảnh GT trong ô 5x5 quanh tâm)\n"
    "Cảnh đồ chơi đẳng hướng (Sigma=s^2 I, không có kết cấu định hướng) nên dùng eta VÔ HƯỚNG\n"
    "(kx=ky=kz=k ở bước 7.3c) thay vì eta_3ch dị hướng đầy đủ của ảnh thật.")
ETA = np.zeros((N, V_CAMS))
ACTIVE = np.zeros((N, V_CAMS), dtype=bool)
IS_HIGH = np.zeros((N, V_CAMS), dtype=bool)
IS_LOW = np.zeros((N, V_CAMS), dtype=bool)
for v in range(V_CAMS):
    it = ITEMS[v]
    for i in range(N):
        g = it[i]
        A, B, Cc = g["conic"]
        det_c = A * Cc - B * B
        cov = np.array([[Cc, -B], [-B, A]]) / det_c      # Sigma' suy từ conic=(A,B,C)=(c22,-c12,c11)/det
        eigval = np.linalg.eigvalsh(cov)
        axis_len_px = float(np.sqrt(np.clip(eigval, 1e-9, None)).max())
        cx = int(np.clip(round(g["mu2"][0]), 2, W - 3)); cy = int(np.clip(round(g["mu2"][1]), 2, H - 3))
        patch = GT[v][:, cy - 2:cy + 3, cx - 2:cx + 3]
        wavelength_min = 1.0 / (patch.std() + 1e-3)
        eta = axis_len_px / (wavelength_min + 1e-6)
        ETA[i, v] = eta
        opac_ok = ALPHA_MODEL[i] > FREQ_OPACITY_THR
        trans_ok = bool(FOOT[v][i].any())                 # proxy max_transmittance>0 (còn đóng góp pixel)
        # freq_grad_threshold=2e-5 rất nhỏ so với gradient thực đo ở chương 6 (~1e-2..1e-1): trong
        # cảnh đồ chơi, điều kiện grad hầu như luôn thoả khi Gaussian còn hiện diện -> coi = trans_ok
        grad_ok = trans_ok
        ACTIVE[i, v] = bool(opac_ok and trans_ok and grad_ok)
        IS_HIGH[i, v] = ACTIVE[i, v] and eta > TAU_HIGH
        IS_LOW[i, v] = ACTIVE[i, v] and eta <= TAU_LOW
print(f"{'':>4} | " + " | ".join(f"eta view{v+1}" for v in range(V_CAMS)) + " | active(0/1 mỗi view)")
for i in range(N):
    print(f"G{i+1:<3} | " + " | ".join(f"{ETA[i,v]:>10.4f}" for v in range(V_CAMS))
          + f" | {ACTIVE[i].astype(int)}")

hdr("[7.2] Tích luỹ: accum_view_count (=importance_score), eta_high_count, eta_low_count, "
    "high_ratio, low_ratio")
COUNTS = ACTIVE.astype(int)                        # [N,V] 0/1 'active' (thay metricCount cũ)
IMPORTANCE = COUNTS.sum(1)                          # = gaussians.accum_view_count (0..V_CAMS)
ETA_HIGH_COUNT = IS_HIGH.sum(1)
ETA_LOW_COUNT = IS_LOW.sum(1)
VALID = IMPORTANCE > 0
HIGH_RATIO = np.zeros(N); LOW_RATIO = np.zeros(N)
HIGH_RATIO[VALID] = ETA_HIGH_COUNT[VALID] / IMPORTANCE[VALID]
LOW_RATIO[VALID] = ETA_LOW_COUNT[VALID] / IMPORTANCE[VALID]
IMPORTANCE_SCORE_THR = 0.5   # arguments/__init__.py: importance_score_threshold = 0.5
SPLIT_RATIO_THR = 0.8        # arguments/__init__.py: split_ratio_threshold = 0.8
PRUNE_RATIO_THR = 0.8        # arguments/__init__.py: prune_ratio_threshold = 0.8
for i in range(N):
    print(f"G{i+1}: accum_view_count={IMPORTANCE[i]}  eta_high_count={ETA_HIGH_COUNT[i]}  "
          f"eta_low_count={ETA_LOW_COUNT[i]}  high_ratio={HIGH_RATIO[i]:.3f}  low_ratio={LOW_RATIO[i]:.3f}")

# ============================================================================
# 7.3 — Gradient theo mu' (không đổi so với 3DGS gốc / chương 6) + extent
# ============================================================================
hdr("[7.3] Gradient theo mu' bằng sai phân hữu hạn (công thức backward không đổi ở SADGS)")
print("Ghi chú: cột có dấu = tổng dL/dmu'; cột abs = tổng |dL/dmu'| (backward.cu:587-601, cột 0-1 vs 2-3).")
HSTEP = 0.5
ACCUM = np.zeros(N); ACCUM_ABS = np.zeros(N); DENOM = np.zeros(N)
GRAD_TABLE = []
for v in range(V_CAMS):
    base_items = ITEMS[v]
    for i in range(N):
        mu2 = base_items[i]["mu2"]
        g_signed = np.zeros(2); g_abs = np.zeros(2)
        for ax, scale in ((0, W / 2), (1, H / 2)):
            imgs = []
            for sgn in (+1, -1):
                mo = mu2.copy(); mo[ax] += sgn * HSTEP
                img, _, _ = render(prep_view(CAM[v], ALPHA_MODEL, mu_override={i: mo}))
                imgs.append(img)
            lp = np.abs(imgs[0] - GT[v]).mean(0) / (H * W)
            lm = np.abs(imgs[1] - GT[v]).mean(0) / (H * W)
            dl_pix = (lp - lm) / (2 * HSTEP)
            g_signed[ax] = dl_pix.sum() * scale
            g_abs[ax] = np.abs(dl_pix).sum() * scale
        visible = base_items[i]["tiles"].any()
        if visible:
            ACCUM[i] += np.linalg.norm(g_signed)
            ACCUM_ABS[i] += np.linalg.norm(g_abs)
            DENOM[i] += 1
        GRAD_TABLE.append((v, i, g_signed, g_abs))
GBAR = ACCUM / DENOM
GBAR_ABS = ACCUM_ABS / DENOM
print(f"{'':>4} | {'g_bar':>10} | {'g_bar_abs':>10} | {'>=grad_thresh':>13} | {'>=grad_abs_thresh':>17}")
for i in range(N):
    print(f"G{i+1:<3} | {GBAR[i]:>10.3e} | {GBAR_ABS[i]:>10.3e} | "
          f"{str(GBAR[i]>=TAU_GRAD):>13} | {str(GBAR_ABS[i]>=TAU_GRAD_ABS):>17}")

hdr("[7.3] extent = 1.1 * max ||c_v - c_bar||  (getNerfppNorm, không đổi)")
CBAR = CAM.mean(0)
DISTS = np.linalg.norm(CAM - CBAR, axis=1)
EXTENT = 1.1 * DISTS.max()
print(f"c_bar = {CBAR};  extent = {EXTENT:.6f};  dense*extent = {DELTA*EXTENT:.6f}")

# ============================================================================
# 7.3b — densify_and_prune_structgs: hợp nhất mặt nạ (gaussian_model.py:957-1052)
# ============================================================================
hdr("[7.3b] densify_and_prune_structgs hợp nhất custom_split_mask (từ high_ratio, train.py:333-373)\n"
    "với grad_qualifiers/_abs kiểu 3DGS gốc và scale-qualifiers (dense*extent), cổng bằng\n"
    "metric_mask = importance_score(accum_view_count) > importance_score_threshold")
GRAD_HIGH_HARD = 1e-5    # train.py:333 is_grad_high (hardcode trong train.py, không phải arguments)
grad_qualifiers = GBAR >= TAU_GRAD                  # >= args.grad_thresh
grad_qualifiers_abs = GBAR_ABS >= TAU_GRAD_ABS       # >= args.grad_abs_thresh
is_grad_high = GBAR >= GRAD_HIGH_HARD                # xấp xỉ norm(mean grad) bằng GBAR (chương 6)
custom_split_mask = (HIGH_RATIO > SPLIT_RATIO_THR) & is_grad_high     # train.py:345
custom_prune_mask = (LOW_RATIO > PRUNE_RATIO_THR) & VALID              # train.py:353

MAXS = S.max(1)
clone_qualifiers = MAXS <= DELTA * EXTENT           # args.dense * extent
split_qualifiers = MAXS > DELTA * EXTENT
final_split_mask = (custom_split_mask | grad_qualifiers_abs) & split_qualifiers
final_clone_mask = (custom_split_mask | grad_qualifiers) & clone_qualifiers
metric_mask = IMPORTANCE > IMPORTANCE_SCORE_THR

CLONE = metric_mask & final_clone_mask
combined_split_mask = metric_mask & final_split_mask
SPLIT = combined_split_mask
print(f"{'':>4} | {'max s':>8} | {'custom_split':>12} | {'grad_q':>6} | {'grad_q_abs':>10} | "
      f"{'clone_q':>7} | {'split_q':>7} | {'metric':>6} | {'CLONE':>5} | {'SPLIT':>5}")
for i in range(N):
    print(f"G{i+1:<3} | {MAXS[i]:>8.4f} | {str(bool(custom_split_mask[i])):>12} | "
          f"{str(bool(grad_qualifiers[i])):>6} | {str(bool(grad_qualifiers_abs[i])):>10} | "
          f"{str(bool(clone_qualifiers[i])):>7} | {str(bool(split_qualifiers[i])):>7} | "
          f"{str(bool(metric_mask[i])):>6} | {str(bool(CLONE[i])):>5} | {str(bool(SPLIT[i])):>5}")

# ============================================================================
# 7.3c — densify_and_split_structgs: split dị hướng giải tích (gaussian_model.py:638-831)
# ============================================================================
hdr("[7.3c] densify_and_split_structgs: k = ceil(sqrt(clamp(eta,min=1))) mỗi trục; N_con = kx*ky*kz;\n"
    "lưới toạ độ tâm i-(k-1)/2 mỗi trục; độ lệch = (s/k)*sqrt(12)*lưới, xoay bởi R (=I ở đây);\n"
    "scale_con = s / k^ks_scale_power  (ks_scale_power = 1.0 mặc định, arguments/__init__.py)")
KS_SCALE_POWER = 1.0
split_idx = np.where(SPLIT)[0]
simulated = False
if len(split_idx) == 0:
    split_idx = np.array([0]); simulated = True
    print("Không Gaussian nào đủ điều kiện split -> GIẢ LẬP split Gaussian 1.")
SPLIT_CHILDREN = []   # (i, mu_con [3], s_con, k)
for i in split_idx:
    eta_i = max(1.0, float(np.max(ETA[i])))     # hướng dẫn = max qua các view (max_eta_3ch, isotropic proxy)
    k = max(int(np.ceil(np.sqrt(eta_i))), 1)
    s_new = S[i, 0] / (k ** KS_SCALE_POWER)
    sep = (S[i, 0] / k) * np.sqrt(12.0)
    offs = (np.arange(k) - (k - 1) / 2.0)
    grid = np.array([[gx, gy, gz] for gx in offs for gy in offs for gz in offs])   # k^3 con, R=I
    for gpt in grid:
        new_mu = P[i] + sep * gpt
        SPLIT_CHILDREN.append((i, new_mu, s_new, k))
    print(f"G{i+1}: eta_max={eta_i:.4f} -> k={k} (k^3={k**3} con), s={S[i,0]:.6f} -> s_con={s_new:.6f}, "
          f"tách = s_con*sqrt(12) = {sep:.6f}")
print("Clone (densify_and_clone_structgs, gaussian_model.py:934-955): theta_con = theta_i (sao chép "
      "nguyên vẹn, kế thừa densify_count -- không đổi so với FastGS cũ).")

# ============================================================================
# 7.4 — Prune trong densify_and_prune_structgs (gaussian_model.py:1013-1044)
# ============================================================================
hdr("[7.4] prune_mask = (opacity<min_opacity) | big_screen(r2D>size_threshold) | "
    "big_world(max s>0.1*extent), hợp OR với custom_prune_mask (low_ratio đa-view)")
min_opacity = 0.1                      # train.py:364 (hardcode, khác min_weight=0.7 kiểu cũ)
opacity_prune = ALPHA_MODEL < min_opacity
R2D_MAX = np.array([max(ITEMS[v][i]["r2d"] for v in range(V_CAMS)) for i in range(N)])
big_screen = R2D_MAX > 20
big_world = MAXS > 0.1 * EXTENT
PRUNE_MASK = opacity_prune | big_screen | big_world | custom_prune_mask
print(f"{'':>4} | {'alpha<0.1':>9} | {'r2D>20':>6} | {'s>0.1ext':>8} | {'low_ratio prune':>15} | {'XOÁ':>4}")
for i in range(N):
    print(f"G{i+1:<3} | {str(bool(opacity_prune[i])):>9} | {str(bool(big_screen[i])):>6} | "
          f"{str(bool(big_world[i])):>8} | {str(bool(custom_prune_mask[i])):>15} | "
          f"{str(bool(PRUNE_MASK[i])):>4}")

hdr("[7.4] pruning_score=None trong train.py hiện tại (lời gọi compute_gaussian_score_structgs bị\n"
    "comment ở train.py:418) -> nhánh multinomial CÓ SẴN trong densify_and_prune_structgs\n"
    "(gaussian_model.py:1029-1042) nhưng KHÔNG được kích hoạt mặc định. SADGS mặc định XOÁ THẲNG\n"
    "prune_points(prune_mask), không rút mẫu có trọng số như FastGS cũ.")
REMOVE = PRUNE_MASK.copy()      # đường mặc định: xoá toàn bộ prune_mask, không có bản sao multinomial


def multinomial_no_replacement(w, k, rng_):
    """torch.multinomial(w, k, replacement=False): rút lần lượt theo xác suất w/sum(w) trên phần còn lại."""
    w = w.astype(float).copy()
    out = []
    for _ in range(k):
        if w.sum() <= 0:
            break
        p = w / w.sum()
        j = rng_.choice(len(w), p=p)
        out.append(j)
        w[j] = 0.0
    return np.array(out, dtype=int)


hdr("[7.4 minh hoạ] Nếu pruning_score ĐƯỢC cấp (vd = low_ratio, đã có sẵn trong [0,1]), nhánh\n"
    "multinomial vẫn chạy được đúng như code (chỉ minh hoạ, KHÔNG phải đường chạy mặc định):")
PRUNING = LOW_RATIO.copy()                  # [0,1], thay Pruning_i minmax kiểu FastGS cũ
RAW = ETA_LOW_COUNT.astype(float)           # hiển thị thô: số lần 'low' qua các view
W_I = 1.0 / (1e-6 + (1 - PRUNING))
to_remove = int(PRUNE_MASK.sum())
BUDGET = int(0.5 * to_remove)
print(f"remove_budget = floor(0.5 * {to_remove}) = {BUDGET}")
print(f"w_i = 1/(1e-6 + (1 - low_ratio_i)) = {W_I}")
rng = np.random.default_rng(SEED)
if BUDGET > 0:
    SAMPLED = multinomial_no_replacement(W_I, BUDGET, rng)
    Smask = np.zeros(N, dtype=bool); Smask[SAMPLED] = True
    REMOVE_ILLUSTRATIVE = PRUNE_MASK & Smask
    print(f"S (seed 0) = {np.sort(SAMPLED)+1};  xoá minh hoạ = prune_mask ∩ S = "
          f"{np.where(REMOVE_ILLUSTRATIVE)[0]+1}")
else:
    REMOVE_ILLUSTRATIVE = np.zeros(N, dtype=bool)
    print("budget = 0 -> không xoá.")

hdr("[7.4 GIẢ ĐỊNH N=10] Minh hoạ cơ chế multinomial trọng số (mã dùng chung), Pruning cho trước, "
    "C = {3,6,8,9,10}, 1000 lần seed 0")
PR10 = np.array([0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
C10 = np.zeros(10, dtype=bool); C10[[2, 5, 7, 8, 9]] = True
W10 = 1.0 / (1e-6 + (1 - PR10))
B10 = int(0.5 * C10.sum())
print(f"|C| = {int(C10.sum())}, remove_budget = floor(0.5*5) = {B10}")
print(f"{'i':>3} | {'Pruning':>8} | {'w_i':>12} | {'p_i = w/sum w':>13} | {'∈C':>3}")
for i in range(10):
    print(f"{i+1:>3} | {PR10[i]:>8.2f} | {W10[i]:>12.4f} | {W10[i]/W10.sum():>13.6f} | {str(bool(C10[i])):>3}")
rng = np.random.default_rng(SEED)
freq10 = np.zeros(10); nrem10 = []; freqS10 = np.zeros(10)
for _ in range(1000):
    s_ = multinomial_no_replacement(W10, B10, rng)
    m_ = np.zeros(10, dtype=bool); m_[s_] = True
    freqS10 += m_
    rm = C10 & m_
    freq10 += rm; nrem10.append(rm.sum())
print(f"tần suất ∈ S      : {freqS10/1000}")
print(f"tần suất bị xoá   : {freq10/1000}")
print(f"số xoá thật TB = {np.mean(nrem10):.3f} so với budget {B10} (min {min(nrem10)}, max {max(nrem10)})")

# ============================================================================
# 7.5 — Ép opacity
# ============================================================================
hdr("[7.5] Sau mỗi densify (gaussian_model.py:1046): alpha~ <- sigma^-1(min(alpha,0.8)) — KHÔNG đổi.\n"
    "reset_opacity (gaussian_model.py:415-419, mỗi opacity_reset_interval=3000 vòng lặp) ĐÃ ĐỔI:\n"
    "opacity_new = opacity * opacity_reset_decay (decay=0.1, NHÂN suy giảm), KHÔNG còn là\n"
    "sigma^-1(min(alpha,0.01)) cố định kiểu FastGS/3DGS gốc.")
OPACITY_RESET_DECAY = 0.1
for a in (0.1, 0.9):
    after_densify = min(a, 0.8)
    after_reset = a * OPACITY_RESET_DECAY
    print(f"alpha={a}: logit trước = {inverse_sigmoid(a):+.6f}; "
          f"(1) sau densify min(a,0.8)={after_densify} -> logit {inverse_sigmoid(after_densify):+.6f}; "
          f"(2) sau reset_opacity a*0.1={after_reset:.4f} -> logit {inverse_sigmoid(after_reset):+.6f}")

# ============================================================================
# 7.6 — final_prune_structgs
# ============================================================================
hdr("[7.6] final_prune_structgs = [alpha < min_opacity] ∨ [pruning_score > 0.9]  (gaussian_model.py:\n"
    "1059-1066, công thức KHÔNG đổi so với FastGS cũ) — NHƯNG train.py:419 gọi hàm này chỉ trong\n"
    "khối bị comment; ở prune_iterations thật (train.py:412-417) chỉ chạy prune_points(opacity<0.1),\n"
    "không có nhánh pruning_score>0.9 nữa vì compute_gaussian_score_structgs không tồn tại.")
FP = (ALPHA_MODEL < 0.1) | (PRUNING > 0.9)
for i in range(N):
    print(f"G{i+1}: alpha={ALPHA_MODEL[i]} <0.1? {ALPHA_MODEL[i] < 0.1}  "
          f"low_ratio={PRUNING[i]:.4f} >0.9? {PRUNING[i] > 0.9}  -> {'XOÁ (minh hoạ)' if FP[i] else 'giữ'}")
print(f"Kịch bản N=10 (minh hoạ, giả định Pruning=PR10): Pruning>0.9 -> xoá {np.where(PR10 > 0.9)[0]+1} "
      f"(i=7 có Pruning=0.9, không > 0.9 -> giữ)")

# ============================================================================
# 7.3 cuối — luỹ thừa N
# ============================================================================
hdr("[7.3 cuối] N_cuối ≈ N0 [(1+r_spawn)(1-r_prune)]^n,  r_spawn=0.15, r_prune=0.05 (minh hoạ tốc độ "
    "tăng trưởng bậc mũ của densification_window_width, không phải công thức đóng của SADGS)")
base = (1 + 0.15) * (1 - 0.05)
print(f"(1+0.15)(1-0.05) = {base:.4f}")
for N0 in (4, 100_000):
    for n in (28, 145):
        print(f"N0={N0:>7}, n={n:>3}: hệ số = {base**n:.6e}  ->  N_cuối ≈ {N0*base**n:.6e}")

# ============================================================================
# 7.8 — Đầu ra của khối
# ============================================================================
hdr("[7.8] N <- N + |clone| + sum_i (k_i^3 - 1 cho mỗi Gaussian split) - |xoá|")
n_clone = int(CLONE.sum())
n_split_parents = len(split_idx) if not simulated else 0
n_split_children = len(SPLIT_CHILDREN) if not simulated else 0   # mỗi phần tử đã LÀ 1 con (k^3 con/cha)
n_rm = int(REMOVE.sum())
N_NEW = N + n_clone + (n_split_children - n_split_parents) - n_rm
print(f"N = {N} + clone({n_clone}) + split_ròng({n_split_children}-{n_split_parents}) - xoá({n_rm}) "
      f"= {N_NEW}")
if simulated:
    k_sim = SPLIT_CHILDREN[0][3] if SPLIT_CHILDREN else 1
    print(f"Nếu tính cả split GIẢ LẬP G1 (k={k_sim}, {k_sim**3} con): "
          f"N = {N} + {n_clone} + ({k_sim**3}-1) - {n_rm} = {N + n_clone + (k_sim**3 - 1) - n_rm}")
print("Danh sách Gaussian con sau split (mu, s, k):")
for (i, m, sc, k) in SPLIT_CHILDREN:
    print(f"  G{i+1} con (k={k}): mu={m} s={sc:.6f}  alpha~={inverse_sigmoid(min(0.1, 0.8)):+.6f}")
for i in np.where(CLONE)[0]:
    print(f"  G{i+1}' (clone): mu={P[i]} s={S[i,0]:.6f}")
