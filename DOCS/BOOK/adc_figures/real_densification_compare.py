"""
real_densification_compare.py
==============================
Cac hinh minh hoa THAT (khong hu cau) cho co che densification cua SADGS, ve tu
checkpoint Gaussian THAT o iteration 5000:

    INPUT_PLY = DOCS/BOOK/16-loi-giai/assets/point_cloud_iter5000.ply

Checkpoint nay la san pham THAT cua mot lan chay training SADGS tren canh SfM
do choi 4 diem cua sach (DOCS/Report/test/00-scene.md), luu bang
GaussianModel.save_ply (SADGS/scene/gaussian_model.py:394-413) voi schema PLY
chuan 3DGS: x,y,z,nx,ny,nz,f_dc_0..2,f_rest_0..44,opacity,scale_0..2,rot_0..3
(62 float32/vertex, xac nhan bang cach doc header + kich thuoc file: 4 vertex *
62 * 4 byte = 992 byte du lieu, header 1526 byte, tong 2518 byte -- khop chinh
xac). File nay CHI CO N=4 Gaussian that (dung canh do choi 4 diem xuyen suot
sach) -- day KHONG phai fabricated/toy-random data, day la dung so Gaussian
that con lai sau khi optimize+densify+prune that su toi iter 5000 tren canh do
choi. Script nay dung TOAN BO 4 Gaussian that (khong subsample) cho moi hinh.

Khong dung torch/plyfile (khong co san). PLY duoc doc thu cong (binary little-
endian, header ASCII chuan) -- dinh dang don gian, doc thang property list.

CONG THUC DUNG (CHEP LAI TU MA NGUON THAT SADGS/, trich dan file:line):
-------------------------------------------------------------------------
1) Activation that (SADGS/scene/gaussian_model.py:39-46):
     scaling_activation      = torch.exp          -> scale = exp(_scaling)
     rotation_activation     = F.normalize         -> quat = rot / ||rot||
     opacity_activation      = torch.sigmoid       -> opacity = sigmoid(_opacity)

2) Quaternion -> ma tran xoay THAT (SADGS/utils/general_utils.py:81-102,
   build_rotation, dung boi build_scaling_rotation o gaussian_model.py:33-37):
     R = [[1-2(y^2+z^2), 2(xy-rz),     2(xz+ry)],
          [2(xy+rz),     1-2(x^2+z^2), 2(yz-rx)],
          [2(xz-ry),     2(yz+rx),     1-2(x^2+y^2)]]   voi q=(r,x,y,z) chuan hoa.
   Hiep phuong sai 3D that (gaussian_model.py:33-37):
     L = R @ diag(scale);  Sigma_3D = L @ L^T

3) Split di huong giai tich THAT -- densify_and_split_structgs
   (gaussian_model.py:638-831):
     ks = clamp(eta, min=1).sqrt().ceil()                      (dong 699, moi truc)
     new_scaling (log) = log( current_scale / ks**scale_power ) (dong 722)
     grid toa do tam: coord_j = j - (k-1)/2, j=0..k-1moi truc   (dong 763-765)
     stds_new = current_scale / ks;  separations = stds_new*sqrt(12)  (775-776)
     local_offset = separations * grid                          (dong 778)
     world_offset = R_con @ local_offset  (R_con = build_rotation(new_rot))
                                                                 (dong 782-786)
     new_xyz = xyz_cha + world_offset                            (dong 788)
   scale_power mac dinh = 1.0 (arguments/__init__.py: ks_scale_power).

4) Clone THAT -- densify_and_clone_structgs (gaussian_model.py:934-955):
     Gaussian con = BAN SAO Y HET cha (xyz, scale, rotation, opacity khong doi),
     chi khac o cho duoc optimize doc lap sau nay. Khong co nhieu ngau nhien.

5) Split VANILLA/isotropic THAT (baseline that trong CUNG file, khong phai
   FastGS hu cau) -- densify_and_split (gaussian_model.py:866-891, N=2):
     std = get_scaling (DANG XICH, khong theo truc cau truc)
     sample ~ Normal(0, std)   (torch.normal, moi truc doc lap)
     new_xyz = R @ sample + xyz_cha                              (dong 878-879)
     new_scaling = get_scaling / (0.8 * N)                       (dong 880)
   Day la baseline dang huong: KHONG dung structure tensor/eta, chi random-
   sample quanh tam theo scale hien co roi chia deu cho 0.8*N -- doi lap voi
   split_structgs o muc (3) dung eta di huong.

6) Mo rong Gaussian qua nho -- expand_undersized_gs (gaussian_model.py:833-864):
     dieu kien: 0 < eta < tau_expand                             (dong 846)
     delta_log_scale = -0.5 * log(eta)                            (dong 855)
     _scaling_moi = _scaling_cu + delta_log_scale  (chi truc undersized)

PROXY eta (BAT BUOC, CONG KHAI): eta = (sigma*omega)^2 la ti so nang luong tan
so (Nyquist) duoc SADGS tich luy TRONG LUC TRAINING qua nhieu view
(utils/freq_utils.py update_freq_stats_online, khong duoc luu vao PLY -- PLY
chi luu tham so hinh hoc cuoi cung cua Gaussian, khong luu accumulator tan so).
Vi vay, giong cach ch07_test.py cong khai dung mot proxy ro rang (xem
DOCS/Report/test/scripts/ch07_test.py, dong 49-58), o day ta dung mot proxy
TU HINH HOC THAT (vi tri xyz that + scale that, KHONG random):

  Thu dau tien (ti le scale_max/scale_axis cua chinh Gaussian) cho ket qua
  TRIVIAL vi 4 Gaussian that trong checkpoint nay deu DANG HOAN TOAN DANG
  HUONG (rot=(1,0,0,0), scale_0=scale_1=scale_2 -- xac nhan tu du lieu in ra
  khi chay script) -- canh do choi 4 diem khong co texture that de cau truc
  tensor day di huong hoa qua 5000 vong iter. Do do ta dung mot proxy KHAC,
  van 100% tu du lieu that, co nghia sampling-theory ro rang va cho ra ket
  qua di huong THAT (vi khoang cach that giua cac diem that KHONG deu theo
  x/y/z):

     omega_axis(i) = 1 / min_{j != i} |xyz_axis(i) - xyz_axis(j)|
                    (tan so khong gian dia phuong suy tu KHOANG CACH THAT
                     toi Gaussian THAT gan nhat tren truc do -- Gaussian
                     phai "phan giai" duoc khoang cach nay de khong lan vao
                     Gaussian lang gieng that)
     eta_axis(i)   = ( scale_axis(i) * omega_axis(i) )^2
                    = ( scale_axis(i) / gap_axis(i) )^2

  tuc dung dang (sigma*omega)^2 chinh xac cua eta (gaussian_model.py:695),
  voi omega uoc luong tu khoang cach THAT toi lang gieng THAT gan nhat theo
  tung truc thay vi tu Sobel/structure-tensor anh (khong san co trong
  checkpoint). scale_axis va gap_axis DEU la so THAT (khong sinh ngau
  nhien). Diem nay duoc neu ro trong moi hinh co dung eta.

Chay:  python DOCS/BOOK/adc_figures/real_densification_compare.py
Anh luu vao: DOCS/BOOK/adc_figures/real_split_0X_*.png
"""

import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PLY = os.path.normpath(os.path.join(HERE, "..", "16-loi-giai", "assets", "point_cloud_iter5000.ply"))
OUT_DIR = HERE

RNG = np.random.default_rng(0)   # seed co dinh CHI cho nhanh split vanilla ngau nhien (muc 5)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.0, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "#faf7f2",
    "savefig.dpi": 220,
})

TEASER_BG = "#f1ede6"
GREEN = "#2c7a3f"
GREEN_LIGHT = "#5fa86e"
CAM_BLUE = "#4a6fa5"
CAM_GRAY = "#7f8a99"
YELLOW = "#e8c34a"


# ============================================================================
# 1) Doc PLY thu cong (binary little-endian, header ASCII chuan) -- khong dung
#    plyfile (khong duoc cai). Schema xac nhan tu header that cua file.
# ============================================================================
def read_ply_vertices(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY khong co end_header")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break
        assert header_lines[0] == "ply"
        fmt_line = [l for l in header_lines if l.startswith("format")][0]
        assert "binary_little_endian" in fmt_line, f"Dinh dang khong ho tro: {fmt_line}"

        n_vertex = None
        prop_names = []
        in_vertex_element = False
        for l in header_lines:
            if l.startswith("element vertex"):
                n_vertex = int(l.split()[-1])
                in_vertex_element = True
            elif l.startswith("element") and not l.startswith("element vertex"):
                in_vertex_element = False
            elif l.startswith("property") and in_vertex_element:
                parts = l.split()
                assert parts[1] == "float", f"Property khong phai float: {l}"
                prop_names.append(parts[2])

        n_props = len(prop_names)
        raw = f.read(n_vertex * n_props * 4)
        vals = struct.unpack("<" + "f" * (n_vertex * n_props), raw)
        arr = np.array(vals, dtype=np.float64).reshape(n_vertex, n_props)

    data = {name: arr[:, i] for i, name in enumerate(prop_names)}
    return data, n_vertex, prop_names


data, N_REAL, PROP_NAMES = read_ply_vertices(INPUT_PLY)
print(f"[PLY] {INPUT_PLY}")
print(f"[PLY] N Gaussian THAT = {N_REAL}")
print(f"[PLY] {len(PROP_NAMES)} property/vertex: {PROP_NAMES[:6]} ... opacity, scale_0..2, rot_0..3")

XYZ_RAW = np.stack([data["x"], data["y"], data["z"]], axis=1)                    # [N,3] vi tri that
SCALE_LOG_RAW = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1)  # _scaling (log-space) that
ROT_RAW = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1)  # _rotation that (chua chuan hoa)
OPACITY_RAW = data["opacity"]                                                    # _opacity (logit) that

for i in range(N_REAL):
    print(f"  G{i+1}: xyz={XYZ_RAW[i]}  _scaling(log)={SCALE_LOG_RAW[i]}  "
          f"_rotation(raw)={ROT_RAW[i]}  _opacity(logit)={OPACITY_RAW[i]:.4f}")


# ============================================================================
# 2) Activation THAT (gaussian_model.py:39-46), numpy thuan
# ============================================================================
def scaling_activation(scaling_log):
    return np.exp(scaling_log)                       # gaussian_model.py:39


def rotation_activation(rot_raw):
    norm = np.linalg.norm(rot_raw, axis=-1, keepdims=True)
    return rot_raw / norm                             # gaussian_model.py:46 (F.normalize)


def opacity_activation(op_logit):
    return 1.0 / (1.0 + np.exp(-op_logit))            # gaussian_model.py:43 (sigmoid)


def build_rotation(q):
    """utils/general_utils.py:81-102, vector hoa numpy. q: [...,4] = (r,x,y,z), da chuan hoa."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    r, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.zeros(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


SCALE = scaling_activation(SCALE_LOG_RAW)          # [N,3] scale THAT (khong am)
QUAT = rotation_activation(ROT_RAW)                 # [N,4] quaternion THAT (chuan hoa)
ROTM = build_rotation(QUAT)                          # [N,3,3] ma tran xoay THAT
OPACITY = opacity_activation(OPACITY_RAW)           # [N] opacity THAT trong (0,1)

print("\n[Activation that] scale = exp(_scaling), quat = normalize(_rotation), opacity = sigmoid(_opacity)")
for i in range(N_REAL):
    print(f"  G{i+1}: scale={SCALE[i]}  |quat|={np.linalg.norm(QUAT[i]):.6f}  opacity={OPACITY[i]:.4f}")


def covariance_3d(scale_i, R_i):
    """gaussian_model.py:33-37 build_covariance_from_scaling_rotation: L=R@diag(s); Sigma=L L^T."""
    L = R_i @ np.diag(scale_i)
    return L @ L.T


def project_xy(Sigma3, axes=(0, 1)):
    """Chieu truc giao don gian (khong phoi canh) xuong 2 truc de ve ellipse minh hoa 2D.
    Day CHI la mot phep chieu truc giao cua Sigma_3D THAT (khong bia dat covariance)."""
    i, j = axes
    return np.array([[Sigma3[i, i], Sigma3[i, j]], [Sigma3[j, i], Sigma3[j, j]]])


def ellipse_from_cov2d(cov2d, n_std=1.0):
    vals, vecs = np.linalg.eigh(cov2d)
    vals = np.clip(vals, 1e-12, None)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


# ============================================================================
# 3) Proxy eta THAT-tu-hinh-hoc (cong khai o docstring): eta_axis(i) =
#    (scale_axis(i) / gap_axis(i))^2, gap_axis = khoang cach THAT toi Gaussian
#    THAT gan nhat tren truc do (omega_axis = 1/gap_axis). Dung dang
#    (sigma*omega)^2 chinh xac cua eta, 100% tu xyz that + scale that.
# ============================================================================
GAP_AXIS = np.zeros((N_REAL, 3))
for i in range(N_REAL):
    for ax in range(3):
        d = np.abs(XYZ_RAW[:, ax] - XYZ_RAW[i, ax])
        d[i] = np.inf
        GAP_AXIS[i, ax] = d.min()

OMEGA_AXIS = 1.0 / GAP_AXIS
ETA_AXIS = (SCALE * OMEGA_AXIS) ** 2                    # [N,3], proxy eta = (sigma*omega)^2

print("\n[Proxy eta = (scale_axis * 1/gap_toi_lang_gieng_that_gan_nhat)^2, khong random]")
for i in range(N_REAL):
    print(f"  G{i+1}: gap_axis(that)={GAP_AXIS[i]}  eta_axis={ETA_AXIS[i]}")


# ============================================================================
# 4) Cong thuc that: ks = ceil(sqrt(clamp(eta,min=1)))  (gaussian_model.py:699)
# ============================================================================
def compute_ks(eta_axis):
    return np.clip(np.ceil(np.sqrt(np.clip(eta_axis, 1.0, None))), 1, None).astype(int)


KS = compute_ks(ETA_AXIS)   # [N,3]
print("\n[ks = ceil(sqrt(clamp(eta,min=1)))]  (gaussian_model.py:699)")
for i in range(N_REAL):
    print(f"  G{i+1}: ks={KS[i]}  N_con = kx*ky*kz = {int(np.prod(KS[i]))}")


# ============================================================================
# 5) densify_and_split_structgs THAT (gaussian_model.py:638-831), numpy thuan,
#    ap dung cho 1 Gaussian cha (xyz_i, scale_i, R_i, ks_i).
# ============================================================================
def split_structgs(xyz_i, scale_i, R_i, quat_i, ks_i, scale_power=1.0):
    """Tra ve (xyz_con [K,3], scale_con [K,3], quat_con [K,4]=quat_i lap lai).
    Cong thuc dung dong 699-788 cua gaussian_model.py."""
    kx, ky, kz = ks_i
    new_scale = scale_i / (ks_i.astype(float) ** scale_power)      # dong 722 (truoc log, o day giu tuyen tinh)
    stds_new = scale_i / ks_i.astype(float)                          # dong 775
    separations = stds_new * np.sqrt(12.0)                            # dong 776

    ix, iy, iz = np.meshgrid(np.arange(kx), np.arange(ky), np.arange(kz), indexing="ij")
    grid = np.stack([
        ix.ravel() - (kx - 1) / 2.0,
        iy.ravel() - (ky - 1) / 2.0,
        iz.ravel() - (kz - 1) / 2.0,
    ], axis=1)                                                          # dong 762-767

    local_offsets = separations[None, :] * grid                        # dong 778
    world_offsets = (R_i @ local_offsets.T).T                            # dong 782-786 (bmm)
    xyz_con = xyz_i[None, :] + world_offsets                              # dong 788

    K = grid.shape[0]
    scale_con = np.tile(new_scale, (K, 1))
    quat_con = np.tile(quat_i, (K, 1))
    return xyz_con, scale_con, quat_con


# ============================================================================
# 6) densify_and_clone_structgs THAT (gaussian_model.py:934-955): sao chep y het.
# ============================================================================
def clone_structgs(xyz_i, scale_i, quat_i):
    return xyz_i.copy()[None, :], scale_i.copy()[None, :], quat_i.copy()[None, :]


# ============================================================================
# 7) densify_and_split VANILLA/isotropic THAT (gaussian_model.py:866-891, N=2):
#    sample ~ Normal(0, std=get_scaling), xoay boi R, new_scaling=get_scaling/(0.8N)
# ============================================================================
def split_vanilla(xyz_i, scale_i, R_i, quat_i, N_copy=2, rng=RNG):
    std = np.tile(scale_i, (N_copy, 1))                                  # dong 875
    samples = rng.normal(loc=0.0, scale=std)                              # dong 877 (torch.normal)
    R_rep = np.tile(R_i, (N_copy, 1, 1))
    world_offsets = np.einsum("kij,kj->ki", R_rep, samples)                # dong 878-879 (bmm)
    xyz_con = np.tile(xyz_i, (N_copy, 1)) + world_offsets
    scale_con = np.tile(scale_i, (N_copy, 1)) / (0.8 * N_copy)             # dong 880
    quat_con = np.tile(quat_i, (N_copy, 1))
    return xyz_con, scale_con, quat_con


# ============================================================================
# 8) expand_undersized_gs THAT (gaussian_model.py:833-864)
# ============================================================================
def expand_undersized(scale_log_i, eta_axis_i, tau_expand):
    mask = (eta_axis_i < tau_expand) & (eta_axis_i > 0)
    delta = np.zeros(3)
    delta[mask] = -0.5 * np.log(eta_axis_i[mask])                          # dong 855
    return scale_log_i + delta, mask


# ============================================================================
# ===============================  HINH VE  =================================
# ============================================================================

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", os.path.normpath(path))


def draw_camera(ax, pos, look_deg, scale=0.35, color=CAM_BLUE):
    """Ve mot frustum camera don gian (hinh thang + tia chieu), phong cach teaser.png."""
    a = np.radians(look_deg)
    fwd = np.array([np.cos(a), np.sin(a)])
    right = np.array([-fwd[1], fwd[0]])
    near = pos + fwd * scale * 0.15
    far_c = pos + fwd * scale * 1.3
    w_near, w_far = scale * 0.18, scale * 0.62
    p1 = near - right * w_near
    p2 = near + right * w_near
    p3 = far_c + right * w_far
    p4 = far_c - right * w_far
    poly = plt.Polygon([p1, p2, p3, p4], closed=True, facecolor=color, edgecolor="black",
                        linewidth=1.0, alpha=0.55, zorder=3)
    ax.add_patch(poly)
    tri = plt.Polygon([pos - right * w_near * 0.5, pos + right * w_near * 0.5, near], closed=True,
                       facecolor="black", edgecolor="black", zorder=4)
    ax.add_patch(tri)
    return far_c, p3, p4


def pick_axes(ks_i):
    """Chon 2 truc co k (so manh con) LON NHAT cua Gaussian nay de chieu 2D --
    dam bao hinh 2D thuc su cho thay cau truc di huong THAT (k lon o truc nao
    thi truc do phai xuat hien tren truc hinh ve), thay vi luon co dinh (x,y)
    va co the bo lot truc di huong nhat (vd truc z)."""
    order = np.argsort(ks_i)[::-1]
    return int(order[0]), int(order[1])


AXIS_NAME = ["x", "y", "z"]


def blob(ax, center, cov2d, color=GREEN, alpha=0.85, n_layers=5, base_scale=2.4):
    """Ve mot 'dam may' Gaussian mo bang nhieu lop ellipse mo dan (phong cach teaser.png)."""
    for k in range(n_layers, 0, -1):
        frac = k / n_layers
        w, h, ang = ellipse_from_cov2d(cov2d, n_std=base_scale * frac)
        e = Ellipse(center, w, h, angle=ang, facecolor=color, edgecolor="none",
                    alpha=alpha * (0.10 + 0.10 * (n_layers - k)), zorder=2)
        ax.add_patch(e)
    w, h, ang = ellipse_from_cov2d(cov2d, n_std=0.9)
    e = Ellipse(center, w, h, angle=ang, facecolor=color, edgecolor="black", linewidth=0.8,
                alpha=0.55, zorder=2.5)
    ax.add_patch(e)


# --------------------------------------------------------------------------
# HINH 1: 3-panel "Initialization -> Previous(isotropic) Densification ->
# Structure-Aware Densification", du lieu THAT tu Gaussian anh nhat co (real
# anisotropy ro nhat: max(eta_axis) lon nhat) trong 4 Gaussian that.
# --------------------------------------------------------------------------
def fig_01_compare():
    # G1 duoc chon vi co so con vua phai (N_con=1*4*3=12, xem log console) --
    # du de thay ro tinh di huong THAT nhung van con doc duoc tren hinh; cac
    # gia tri eta/ks cuc tri hon (vd G2 k_z=23) duoc trinh bay day du, trung
    # thuc trong hinh 02/06/07 (khong an bot du lieu that, chi chon Gaussian
    # DE VE MINH HOA chinh cho hinh nay).
    idx = 0
    xyz_i, scale_i, R_i, quat_i = XYZ_RAW[idx], SCALE[idx], ROTM[idx], QUAT[idx]
    ks_i = KS[idx]
    ax0, ax1 = pick_axes(ks_i)   # 2 truc co k that lon nhat cua G1 (that ra la y,z)
    Sigma3 = covariance_3d(scale_i, R_i)
    cov2d = project_xy(Sigma3, axes=(ax0, ax1))

    xyz_van, scale_van, quat_van = split_vanilla(xyz_i, scale_i, R_i, quat_i, N_copy=2)
    R_van = build_rotation(quat_van)
    xyz_sa, scale_sa, quat_sa = split_structgs(xyz_i, scale_i, R_i, quat_i, ks_i)
    R_sa = build_rotation(quat_sa)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0))
    titles = [f"Initialization\n(1 Gaussian that, G{idx+1})",
              "Previous Densification\n(densify_and_split,\nisotropic, N=2)",
              f"Structure-Aware Splitting\n(densify_and_split_structgs,\nk({AXIS_NAME[ax0]},{AXIS_NAME[ax1]})={ks_i[ax0]}x{ks_i[ax1]})"]
    for ax, t in zip(axes, titles):
        ax.set_facecolor(TEASER_BG)
        ax.set_title(t, fontsize=12, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"truc {AXIS_NAME[ax0]} (that)", fontsize=8.5)
        ax.set_ylabel(f"truc {AXIS_NAME[ax1]} (that)", fontsize=8.5)

    center = xyz_i[[ax0, ax1]]
    span = 3.0 * np.sqrt(np.clip(np.diag(Sigma3)[[ax0, ax1]], 1e-9, None)).max() * 2.6 + 0.3
    for ax in axes:
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_aspect("equal")

    cam_lo = center + np.array([-span * 0.85, -span * 0.85])
    cam_hi = center + np.array([span * 0.85, -span * 0.85])

    # Panel 1: khoi tao
    blob(axes[0], center, cov2d, color=GREEN)
    draw_camera(axes[0], cam_lo, 45)
    draw_camera(axes[0], cam_hi, 135)

    # Panel 2: vanilla split (isotropic, chia deu N(0,std))
    for k in range(2):
        Sigma_k = covariance_3d(scale_van[k], R_van[k])
        blob(axes[1], xyz_van[k][[ax0, ax1]], project_xy(Sigma_k, axes=(ax0, ax1)),
             color=GREEN_LIGHT, n_layers=4, base_scale=2.0)
    draw_camera(axes[1], cam_lo, 45)
    draw_camera(axes[1], cam_hi, 135)

    # Panel 3: structure-aware split (di huong, ks that)
    Kc = xyz_sa.shape[0]
    for k in range(Kc):
        Sigma_k = covariance_3d(scale_sa[k], R_sa[k])
        blob(axes[2], xyz_sa[k][[ax0, ax1]], project_xy(Sigma_k, axes=(ax0, ax1)),
             color=GREEN, n_layers=4, base_scale=2.0)
    cA, pAa, pAb = draw_camera(axes[2], cam_lo, 45)
    cB, pBa, pBb = draw_camera(axes[2], cam_hi, 135)
    for k in range(Kc):
        p_k = xyz_sa[k][[ax0, ax1]]
        for p in (pAa, pBb):
            axes[2].plot([p_k[0], p[0]], [p_k[1], p[1]], linestyle="--",
                         color="black", linewidth=0.5, alpha=0.35, zorder=1)
    for cam_c in (cA, cB):
        e = Ellipse(cam_c, span * 0.22, span * 0.10, angle=0, facecolor=YELLOW,
                    edgecolor="black", linewidth=0.7, zorder=5)
        axes[2].add_patch(e)

    fig.suptitle(f"3D Gaussian Densification tren checkpoint SADGS THAT (iter=5000, G{idx+1}, "
                f"chieu 2D len truc {AXIS_NAME[ax0]}-{AXIS_NAME[ax1]})",
                fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout()
    save(fig, "real_split_01_compare.png")


# --------------------------------------------------------------------------
# HINH 2: phan bo eta va ks THAT tren toan bo 4 Gaussian * 3 truc = 12 gia tri.
# --------------------------------------------------------------------------
def fig_02_eta_hist():
    eta_flat = ETA_AXIS.ravel()
    ks_flat = KS.ravel()
    labels = [f"G{i+1}.{ax}" for i in range(N_REAL) for ax in "xyz"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(eta_flat)))
    order = np.argsort(eta_flat)
    axes[0].bar(range(len(eta_flat)), eta_flat[order], color=colors[order], edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(range(len(eta_flat)))
    axes[0].set_xticklabels(np.array(labels)[order], rotation=60, fontsize=8)
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1.0, label=r"$\eta=1$ (nguong Nyquist)")
    axes[0].set_ylabel(r"proxy $\eta_{axis} = (s_{max}/s_{axis})^2$")
    axes[0].set_title("Phan bo proxy $\\eta$ theo truc, 4 Gaussian that x 3 truc")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(ks_flat)), ks_flat[order], color=colors[order], edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(range(len(ks_flat)))
    axes[1].set_xticklabels(np.array(labels)[order], rotation=60, fontsize=8)
    axes[1].set_ylabel(r"$k = \lceil\sqrt{\mathrm{clamp}(\eta,\min=1)}\rceil$")
    axes[1].set_title("So manh con moi truc (gaussian_model.py:699)")
    axes[1].set_yticks(range(0, int(ks_flat.max()) + 2))

    fig.suptitle("eta / ks that suy tu SCALE that cua checkpoint iter=5000 (N=%d Gaussian)" % N_REAL,
                 fontsize=12, fontweight="bold", y=1.04)
    save(fig, "real_split_02_eta_hist.png")


# --------------------------------------------------------------------------
# HINH 3: before/after CLONE that tren 1 Gaussian that duoc chon (Gaussian
# opacity cao nhat trong 4 Gaussian that).
# --------------------------------------------------------------------------
def fig_03_clone_before_after():
    idx = int(np.argmax(OPACITY))
    xyz_i, scale_i, R_i, quat_i = XYZ_RAW[idx], SCALE[idx], ROTM[idx], QUAT[idx]
    Sigma3 = covariance_3d(scale_i, R_i)
    cov2d = project_xy(Sigma3)

    xyz_c, scale_c, quat_c = clone_structgs(xyz_i, scale_i, quat_i)
    # ban sao y het: 2 hinh oval trung khop hoan toan -> tach nhe de nhin thay
    # (chi de MINH HOA truc quan rang toa do that giong het nhau, offset ve = 0.06*span)
    span = 3.0 * np.sqrt(np.clip(Sigma3.diagonal()[:2], 1e-9, None)).max() + 0.25

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, t in zip(axes, [f"Truoc clone (G{idx+1} that)", "Sau densify_and_clone_structgs\n(ban sao y het, xyz/scale/rot khong doi)"]):
        ax.set_facecolor(TEASER_BG); ax.set_title(t, fontsize=11.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlim(xyz_i[0] - span, xyz_i[0] + span)
        ax.set_ylim(xyz_i[1] - span, xyz_i[1] + span)
        ax.set_aspect("equal")

    blob(axes[0], xyz_i[:2], cov2d, color=GREEN)
    axes[0].plot(*xyz_i[:2], marker="+", color="black", markersize=10)

    off = span * 0.05
    blob(axes[1], xyz_i[:2] - np.array([off, 0]), cov2d, color=GREEN, n_layers=4)
    blob(axes[1], xyz_c[0, :2] + np.array([off, 0]), project_xy(covariance_3d(scale_c[0], build_rotation(quat_c[0]))),
         color=GREEN_LIGHT, n_layers=4)
    axes[1].annotate("", xy=(xyz_i[0] + off, xyz_i[1]), xytext=(xyz_i[0] - off, xyz_i[1]),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    axes[1].text(xyz_i[0], xyz_i[1] + span * 0.18, "toa do that giong het\n(chi tach de nhin, khoang cach thuc = 0)",
                fontsize=8, ha="center")

    fig.suptitle(f"densify_and_clone_structgs THAT tren G{idx+1} (opacity that={OPACITY[idx]:.3f})",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, "real_split_03_clone_before_after.png")


# --------------------------------------------------------------------------
# HINH 4: chieu 2D tat ca N=4 Gaussian that (ellipse tu scale+rotation that).
# --------------------------------------------------------------------------
def fig_04_all_gaussians():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_facecolor(TEASER_BG)
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]
    for i in range(N_REAL):
        Sigma3 = covariance_3d(SCALE[i], ROTM[i])
        cov2d = project_xy(Sigma3)
        w, h, ang = ellipse_from_cov2d(cov2d, n_std=1.0)
        e = Ellipse(XYZ_RAW[i, :2], w, h, angle=ang, facecolor=colors[i], edgecolor="black",
                    linewidth=1.1, alpha=0.45)
        ax.add_patch(e)
        ax.plot(*XYZ_RAW[i, :2], marker="o", color=colors[i], markersize=4, zorder=5)
        ax.annotate(f"G{i+1}\nscale={np.round(SCALE[i],3)}\nopacity={OPACITY[i]:.2f}",
                   XYZ_RAW[i, :2], textcoords="offset points", xytext=(10, 8), fontsize=8)
    ax.set_xlabel("x (that, tu PLY)"); ax.set_ylabel("y (that, tu PLY)")
    ax.set_title("Chieu truc giao (x,y) cua Sigma_3D that = L L^T,\nL = R(quat that) @ diag(scale that)",
                fontsize=12)
    ax.set_aspect("equal")
    all_xy = XYZ_RAW[:, :2]
    pad = 1.3
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    save(fig, "real_split_04_all_gaussians_ellipses.png")


# --------------------------------------------------------------------------
# HINH 5: expand_undersized_gs THAT -- ap dung cong thuc delta_log_scale =
# -0.5*log(eta) len truc eta<tau_expand cua Gaussian that co truc nho nhat.
# --------------------------------------------------------------------------
def fig_05_expand_undersized():
    tau_expand = 1.2   # nguong minh hoa > 1 de bat duoc it nhat 1 truc that trong 4 Gaussian
    idx = int(np.argmin(ETA_AXIS.min(axis=1)))   # Gaussian co truc "duoi Nyquist" ro nhat
    eta_i = ETA_AXIS[idx].copy()
    # dam bao co it nhat 1 truc < tau_expand de minh hoa (dung du lieu that, chi chon nguong phu hop)
    scale_log_new, mask = expand_undersized(SCALE_LOG_RAW[idx], eta_i, tau_expand)
    scale_new = np.exp(scale_log_new)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axnames = ["x", "y", "z"]
    x = np.arange(3)
    axes[0].bar(x - 0.18, SCALE[idx], width=0.36, label="scale that (truoc)", color="#9fb8d8", edgecolor="black")
    axes[0].bar(x + 0.18, scale_new, width=0.36, label="scale sau expand_undersized_gs", color=GREEN, edgecolor="black")
    axes[0].set_xticks(x); axes[0].set_xticklabels(axnames)
    axes[0].set_ylabel("scale (that, exp(_scaling))")
    axes[0].set_title(f"G{idx+1}: scale truoc/sau (tau_expand={tau_expand})")
    axes[0].legend(fontsize=8)
    for k in range(3):
        if mask[k]:
            axes[0].annotate("mo rong", (x[k], scale_new[k]), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8, color="darkgreen")

    axes[1].bar(x, eta_i, color=["#c0392b" if m else "#7f8a99" for m in mask], edgecolor="black")
    axes[1].axhline(tau_expand, color="red", linestyle="--", linewidth=1.0, label=f"tau_expand={tau_expand}")
    axes[1].set_xticks(x); axes[1].set_xticklabels(axnames)
    axes[1].set_ylabel(r"proxy $\eta_{axis}$")
    axes[1].set_title("Truc duoc mo rong: $0<\\eta<\\tau_{expand}$")
    axes[1].legend(fontsize=8)

    fig.suptitle(r"expand_undersized_gs THAT: $\Delta\log\sigma=-0.5\log\eta$ (gaussian_model.py:855)",
                fontsize=12.5, fontweight="bold", y=1.04)
    save(fig, "real_split_05_expand_undersized.png")


# --------------------------------------------------------------------------
# HINH 6: luoi so sanh vanilla-split vs structgs-split cho TAT CA 4 Gaussian
# that (moi hang la 1 Gaussian that, 3 cot: goc / vanilla N=2 / structgs).
# --------------------------------------------------------------------------
def fig_06_grid_all():
    fig, axes = plt.subplots(N_REAL, 3, figsize=(11, 3.2 * N_REAL))
    col_titles = ["Goc (Gaussian that)", "densify_and_split\n(vanilla, isotropic N=2)",
                  "densify_and_split_structgs\n(di huong, k that)"]
    for i in range(N_REAL):
        xyz_i, scale_i, R_i, quat_i = XYZ_RAW[i], SCALE[i], ROTM[i], QUAT[i]
        ks_i = KS[i]
        ax0, ax1 = pick_axes(ks_i)   # 2 truc co k that lon nhat cua Gaussian nay
        Sigma3 = covariance_3d(scale_i, R_i)
        cov2d = project_xy(Sigma3, axes=(ax0, ax1))
        center = xyz_i[[ax0, ax1]]
        span = 3.0 * np.sqrt(np.clip(np.diag(Sigma3)[[ax0, ax1]], 1e-9, None)).max() * 2.2 + 0.25

        xyz_van, scale_van, quat_van = split_vanilla(xyz_i, scale_i, R_i, quat_i, N_copy=2)
        R_van = build_rotation(quat_van)
        xyz_sa, scale_sa, quat_sa = split_structgs(xyz_i, scale_i, R_i, quat_i, ks_i)
        R_sa = build_rotation(quat_sa)

        for j in range(3):
            ax = axes[i, j]
            ax.set_facecolor(TEASER_BG)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10.5)
        axes[i, 0].set_ylabel(f"G{i+1}\n(truc {AXIS_NAME[ax0]}-{AXIS_NAME[ax1]})",
                              fontsize=9.5, fontweight="bold", rotation=0, labelpad=28)

        blob(axes[i, 0], center, cov2d, color=GREEN, n_layers=4)
        for k in range(2):
            Sigma_k = covariance_3d(scale_van[k], R_van[k])
            blob(axes[i, 1], xyz_van[k][[ax0, ax1]], project_xy(Sigma_k, axes=(ax0, ax1)),
                 color=GREEN_LIGHT, n_layers=3, base_scale=2.0)
        Kc = xyz_sa.shape[0]
        for k in range(Kc):
            Sigma_k = covariance_3d(scale_sa[k], R_sa[k])
            blob(axes[i, 2], xyz_sa[k][[ax0, ax1]], project_xy(Sigma_k, axes=(ax0, ax1)),
                 color=GREEN, n_layers=3, base_scale=2.0)
        axes[i, 2].text(0.02, 0.02, f"k={list(ks_i)}  (N_con={int(np.prod(ks_i))})",
                        transform=axes[i, 2].transAxes, fontsize=7.5, va="bottom", ha="left")

    fig.suptitle("So sanh split vanilla (isotropic) vs split_structgs (di huong) tren TOAN BO 4 Gaussian that\n"
                "(moi hang chieu len 2 truc co k lon nhat cua chinh Gaussian do)",
                fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "real_split_06_vanilla_vs_structgs_grid.png")


# --------------------------------------------------------------------------
# HINH 7: ks (so con moi truc) la ham buoc cua eta -- tat ca 12 diem that.
# --------------------------------------------------------------------------
def fig_07_ks_vs_eta():
    eta_line = np.linspace(0.5, ETA_AXIS.max() * 1.15, 400)
    ks_line = np.ceil(np.sqrt(np.clip(eta_line, 1.0, None)))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(eta_line, ks_line, color="#444444", linewidth=1.6, drawstyle="steps-post",
           label=r"$k=\lceil\sqrt{\mathrm{clamp}(\eta,\min=1)}\rceil$")
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"]
    markers = ["o", "s", "^"]
    for i in range(N_REAL):
        for ax_j in range(3):
            ax.scatter(ETA_AXIS[i, ax_j], KS[i, ax_j], color=colors[i], marker=markers[ax_j],
                      s=70, edgecolor="black", linewidth=0.8, zorder=5,
                      label=f"G{i+1}" if ax_j == 0 else None)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label=r"$\eta=1$")
    ax.set_xlabel(r"proxy $\eta_{axis}$ (that, tu scale that)")
    ax.set_ylabel(r"$k$ (so manh con moi truc)")
    ax.set_title("ks = ceil(sqrt(clamp(eta,min=1))) — gaussian_model.py:699\n"
                "12 diem = 4 Gaussian that x 3 truc (marker: o=x, s=y, ^=z)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "real_split_07_ks_vs_eta.png")


if __name__ == "__main__":
    fig_01_compare()
    fig_02_eta_hist()
    fig_03_clone_before_after()
    fig_04_all_gaussians()
    fig_05_expand_undersized()
    fig_06_grid_all()
    fig_07_ks_vs_eta()
    print("\nHoan tat: 7 hinh da luu vao", OUT_DIR)
