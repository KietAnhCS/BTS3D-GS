"""
deep17f_overview_comparison.py
================================
Bốn hình schematic (matplotlib patches/text/arrows, KHÔNG cần torch/scipy/PIL)
tổng kết trực quan cho các mục 17.0 / 17.6 / 17.7 của chương SAD-GS.

Đây là bản VẼ LẠI (visual transcription) trung thực của các bảng/đoạn văn ĐÃ CÓ
trong DOCS/BOOK/17-sadgs-structure-aware-densification.md — không phát sinh kết
luận mới, chỉ đổi định dạng từ bảng markdown / văn xuôi sang sơ đồ.

Nguồn cho từng hình (xem chi tiết trong docstring mỗi hàm vẽ):
  01 -> bảng "Cụm từ README | Cơ chế trong code | Mục" ở mục 17.0 (dòng ~40-45
       của file .md) + trích README SADGS/README.md:7-10.
  02 -> bảng 5 điểm yếu ở mục 17.7 (dòng ~416-422 của file .md), đối chiếu với
       mục 12.0.3 của chương 12.
  03 -> bảng "Vấn đề mới mà SAD-GS giải quyết" ở mục 17.7 (dòng ~426-430).
  04 -> tổng hợp các công thức trung tâm 17.1->17.5 (structure tensor, eta,
       high/low ratio, anisotropic split) thành một sơ đồ luồng ngang duy nhất
       (hình MỚI, không sao chép bảng nào, chỉ tổng hợp công thức đã có).

Output: DOCS/BOOK/adc_figures/deep17f_0{1,2,3,4}_*.png

Run:  python DOCS/BOOK/adc_figures/deep17f_overview_comparison.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

BG = "#f7f4ee"
DARK = "#242222"
GOLD = "#e8b830"
GREEN = "#3f7d5c"
RED = "#b3423a"
BLUE = "#3a6ea8"
GREY = "#8a8378"
BOXFACE = "#ffffff"

# status colors for the icon-matrix (fig 02)
COL_NOT_FIXED = "#e3d9cf"       # 3DGS gốc: gần như luôn "chưa sửa" baseline -> nhạt
COL_FIXED = "#3f7d5c"           # đã sửa hẳn (xanh lá)
COL_FIXED_DIFF = "#3a6ea8"      # sửa theo cơ chế khác / triệt để hơn (xanh dương)
COL_PARTIAL = "#e8b830"         # có nhánh nhưng bị vô hiệu hoá (vàng/hổ phách)
COL_NA = "#c9c2b6"              # không liên quan (xám)


def rounded_box(ax, xy, w, h, text, fc=BOXFACE, ec=DARK, fontsize=9.5,
                 fontweight="normal", textcolor=DARK, lw=1.3, zorder=3):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, fontweight=fontweight,
             zorder=zorder + 1, wrap=True)
    return box


def arrow(ax, p0, p1, color=DARK, lw=1.6, style="-|>", connstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, connectionstyle=connstyle,
                         zorder=2)
    ax.add_patch(a)


# ----------------------------------------------------------------------
# FIGURE 1 — README quote -> code mechanism -> section (mục 17.0)
# ----------------------------------------------------------------------
def fig01_readme_mapping():
    """
    INPUT: bảng "Cụm từ README | Cơ chế trong code | Mục" ở mục 17.0 của
           17-sadgs-structure-aware-densification.md (dòng ~40-45), cùng
           trích dẫn SADGS/README.md:7-10.
    NGUỒN: 3 cụm từ khoá trong câu README + cơ chế code tương ứng:
      "multiscale image structure"                    -> get_multiscale_structure_tensor_v1/v2
                                                           (SADGS/utils/loss_utils.py:232-387) -> §17.2
      "compares projected extent with texture"         -> eta = do dai truc chieu / buoc song
                                                           (SADGS/utils/freq_utils.py:313-373) -> §17.3
      "multiview consistency"                          -> ti le view eta cao/thap tren tong so
                                                           view nhin thay Gaussian
                                                           (SADGS/utils/freq_utils.py:395-409) -> §17.4
    OUTPUT: sơ đồ 3 hàng, mỗi hàng: [khung cụm từ README] --> [khung cơ chế code]
            --> [khung số mục], để người đọc thấy ngay README không phải khẩu
            hiệu marketing mà ánh xạ 1-1 sang code cụ thể (đúng luận điểm văn
            xuôi ngay dưới bảng gốc trong .md).
    """
    fig, ax = plt.subplots(figsize=(12.5, 6.4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6.25, 6.05,
            "README $\\to$ cơ chế code $\\to$ mục sách  (mục 17.0)",
            ha="center", va="center", fontsize=15, fontweight="bold", color=DARK)
    ax.text(6.25, 5.6,
            '"...uses multiscale image structure ... compares each Gaussian\'s projected\n'
            'screen-space extent with local texture structure ... multiview consistency."',
            ha="center", va="center", fontsize=8.8, style="italic", color=GREY)
    ax.text(6.25, 5.28, "— SADGS/README.md:7-10", ha="center", va="center",
            fontsize=7.8, color=GREY)

    rows = [
        ("\u201cmultiscale image\nstructure\u201d",
         "get_multiscale_structure_tensor_v1 / v2\nloss_utils.py:232-387",
         "§17.2", GOLD),
        ("\u201ccompares projected extent\nwith local texture structure\u201d",
         "$\\eta = \\ell_k / w_{\\min}$\nfreq_utils.py:313-373",
         "§17.3", BLUE),
        ("\u201cmultiview\nconsistency\u201d",
         "high_ratio / low_ratio\nfreq_utils.py:395-409",
         "§17.4", GREEN),
    ]

    y0 = 1.0
    row_h = 1.15
    gap = 0.35
    w1, w2, w3 = 3.1, 4.6, 1.5
    x1, x2, x3 = 0.4, 3.9, 8.9

    for i, (readme_txt, code_txt, sec, accent) in enumerate(rows):
        y = y0 + (2 - i) * (row_h + gap)
        rounded_box(ax, (x1, y), w1, row_h, readme_txt, fc="#fffaf0",
                    ec=accent, fontsize=9.3, fontweight="bold")
        rounded_box(ax, (x2, y), w2, row_h, code_txt, fc=BOXFACE, ec=DARK,
                    fontsize=8.6)
        rounded_box(ax, (x3, y), w3, row_h, sec, fc=accent, ec=accent,
                    fontsize=12, fontweight="bold", textcolor="white")
        arrow(ax, (x1 + w1, y + row_h / 2), (x2, y + row_h / 2), color=accent)
        arrow(ax, (x2 + w2, y + row_h / 2), (x3, y + row_h / 2), color=accent)

    ax.text(6.25, 0.35,
            "Điểm mấu chốt: cả 3 cơ chế được \u201cthay vào đúng chỗ split/clone\u201d của khung ADC nâng cao "
            "(mục 17.0, đoạn cuối) — không thay thế toàn bộ pipeline, chỉ đổi tín hiệu đo lỗi.",
            ha="center", va="center", fontsize=8.3, color=DARK)

    out = os.path.join(HERE, "deep17f_01_readme_mapping.png")
    fig.savefig(out, dpi=175, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ----------------------------------------------------------------------
# FIGURE 2 — icon matrix: 5 weaknesses x {3DGS gốc, SAD-GS}
# ----------------------------------------------------------------------
def fig02_weakness_matrix():
    """
    INPUT: bảng 5 điểm yếu ở mục 17.7 của 17-sadgs-structure-aware-densification.md
           (dòng ~416-422), đối chiếu 5 điểm yếu gốc ở §12.0.3 của
           DOCS/BOOK/12-adaptive-density-control.md.
    NGUỒN VERDICT (giữ nguyên nội dung kỹ thuật, chỉ gộp về 2 cột — mã nguồn
    SAD-GS/train.py, scene/gaussian_model.py kế thừa nguyên khung ADC nâng cao
    này rồi thay đổi/giữ nguyên từng phần, nên trình bày trực tiếp dưới dạng
    "3DGS gốc có vấn đề gì -> SAD-GS xử lý ra sao" mà không tách một bước
    trung gian riêng):
      #1 Densify noi render da dung          -> SAD-GS: KHONG truc tiep (tieu chi khac, chi OR them)
      #2 Split bien, gradient co dau trieu tieu -> SAD-GS: fixed KHAC/triet de hon (eta khong dung gradient)
      #3 Prune xoa ca cum sau reset opacity  -> SAD-GS: DA SUA (multinomial, ke thua nguyen ven) +
                                                 nhanh mo rong theo eta thap CO trong code nhung bi TAT mac dinh
      #4 alpha bao hoa 0.99, chon Gaussian    -> SAD-GS: DA SUA (tran alpha<=0.8, ke thua nguyen ven)
      #5 N khong giam sau t=15000             -> SAD-GS: DA SUA (final prune 4 lan, ke thua nguyen ven)
    OUTPUT: lưới 5 hàng (điểm yếu) x 2 cột (3DGS gốc / SAD-GS), mỗi ô tô màu +
            icon chữ theo 5 trạng thái: chưa sửa (nhạt), đã sửa (xanh lá), sửa
            theo cơ chế khác/triệt để hơn (xanh dương), có nhánh nhưng bị tắt
            (vàng), không liên quan (xám).
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    weaknesses = [
        "#1 Densify ở vùng\nrender đã đúng\n(không hỏi ảnh có sai)",
        "#2 Split biên: gradient\ncó dấu triệt tiêu",
        "#3 Prune xoá cả cụm\nsau reset opacity",
        "#4 $\\alpha\\to0.99$ bão hoà,\nchôn Gaussian phía sau",
        "#5 $N$ không giảm\nsau $t=15000$",
    ]
    cols = ["3DGS gốc", "SAD-GS"]

    # (color, short label) per cell, row-major, 2 cols each
    cells = [
        [(COL_NOT_FIXED, "chưa sửa"),
         (COL_PARTIAL, "KHÔNG trực tiếp\n(tiêu chí khác, chỉ OR thêm)")],
        [(COL_NOT_FIXED, "chưa sửa"),
         (COL_FIXED_DIFF, "SỬA KHÁC & TRIỆT ĐỂ HƠN\n(eta không dùng gradient)")],
        [(COL_NOT_FIXED, "chưa sửa"),
         (COL_FIXED, "ĐÃ SỬA (multinomial,\nkế thừa nguyên vẹn) + nhánh\n$\\eta$ thấp CÓ nhưng bị TẮT")],
        [(COL_NOT_FIXED, "chưa sửa"),
         (COL_FIXED, "ĐÃ SỬA\n(trần $\\alpha\\leq0.8$, kế thừa)")],
        [(COL_NOT_FIXED, "chưa sửa"),
         (COL_FIXED, "ĐÃ SỬA\n(final prune 4 lần, kế thừa)")],
    ]

    n_rows = len(weaknesses)
    n_cols = 2
    left = 3.3
    row_h = 1.18
    col_w = 3.7
    col_gap = 0.15
    row_gap = 0.1
    header_h = 0.55
    total_w = left + n_cols * (col_w + col_gap) + 0.3
    title_h = 0.9
    n_legend = 5
    legend_h = n_legend * 0.42 + 0.3
    total_h = title_h + header_h + n_rows * (row_h + row_gap) + legend_h

    fig.set_size_inches(10.5, 10.5 * total_h / total_w)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)

    # cursor moves top-down
    cursor = total_h

    cursor -= title_h
    ax.text(total_w / 2, cursor + title_h / 2,
            "SAD-GS sửa được điểm yếu nào của 3DGS gốc?\n(đối chiếu §12.0.3, mục 17.7)",
            ha="center", va="center", fontsize=13, fontweight="bold", color=DARK)

    # column headers
    header_y = cursor - header_h
    for c, name in enumerate(cols):
        x = left + c * (col_w + col_gap)
        rounded_box(ax, (x, header_y), col_w, header_h, name, fc=DARK, ec=DARK,
                    fontsize=10.5, fontweight="bold", textcolor="white")
    cursor = header_y

    # row labels + cells
    for r, wtext in enumerate(weaknesses):
        y = cursor - row_h
        rounded_box(ax, (0.0, y), 3.1, row_h, wtext, fc="#efe9df", ec=DARK,
                    fontsize=8.4, fontweight="bold")
        for c in range(n_cols):
            x = left + c * (col_w + col_gap)
            color, label = cells[r][c]
            textcolor = "white" if color in (COL_FIXED, COL_FIXED_DIFF) else DARK
            rounded_box(ax, (x, y), col_w, row_h, label, fc=color, ec=DARK,
                        lw=1.0, fontsize=8.0, textcolor=textcolor,
                        fontweight="bold" if color != COL_NOT_FIXED else "normal")
        cursor = y - row_gap

    # legend
    cursor -= 0.15
    legend_items = [
        (COL_NOT_FIXED, "chưa sửa (baseline 3DGS gốc)"),
        (COL_FIXED, "SAD-GS đã sửa (kế thừa cơ chế nền, không đổi)"),
        (COL_FIXED_DIFF, "SAD-GS sửa theo cơ chế khác / triệt để hơn"),
        (COL_PARTIAL, "có nhánh code nhưng bị vô hiệu hoá mặc định"),
        (COL_NA, "không liên quan / không áp dụng"),
    ]
    lx = 0.1
    for color, text in legend_items:
        ly = cursor - 0.42
        ax.add_patch(Rectangle((lx, ly), 0.28, 0.28, facecolor=color, edgecolor=DARK, lw=0.8))
        ax.text(lx + 0.4, ly + 0.14, text, fontsize=7.8, va="center", ha="left", color=DARK)
        cursor = ly

    out = os.path.join(HERE, "deep17f_02_weakness_matrix.png")
    fig.savefig(out, dpi=175, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ----------------------------------------------------------------------
# FIGURE 3 — new problems SAD-GS solves beyond the 5 from §12.0.3
# ----------------------------------------------------------------------
def fig03_new_problems():
    """
    INPUT: bảng "Vấn đề mới mà SAD-GS giải quyết, ngoài 5 điểm của 12.0.3" ở mục
           17.7 (dòng ~426-430 của .md).
    NGUỒN (3 hàng, giữ nguyên verdict):
      (a) Gaussian to hơn texture GT dù render đúng màu -> 3DGS gốc: Không thấy
          -> SAD-GS giải bằng eta = l/w_min
      (b) Chia Gaussian không đều theo hướng (dẹt chỉ vi phạm 1-2 trục) ->
          3DGS gốc: Không thấy -> SAD-GS giải bằng (kx,ky,kz) độc lập §17.5
      (c) Ngưỡng densify không co giãn theo độ phân giải (Importance tuyệt đối)
          -> 3DGS gốc/ADC nguyên bản: ĐÚNG, là điểm yếu thật -> SAD-GS giải bằng
             high/low_ratio (ngưỡng dạng tỉ lệ)
    OUTPUT: 3 hàng dạng "vấn đề -> 3DGS gốc có thấy? -> SAD-GS giải bằng gì",
            mỗi hàng là một luồng ngang 3 khung + mũi tên, câu (c) tô đỏ vì đây
            là điểm yếu THẬT đã được §17.4 thừa nhận, khác (a)/(b) là "khái
            niệm 3DGS gốc không có" chứ không hẳn là lỗi.
    """
    fig, ax = plt.subplots(figsize=(13, 6.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)

    ax.text(6.5, 6.25,
            "Vấn đề mới SAD-GS giải quyết — ngoài 5 điểm của §12.0.3  (mục 17.7)",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color=DARK)

    rows = [
        ("Gaussian to hơn texture GT\ndù render đang đúng màu\n(dưới-lấy-mẫu hình học)",
         "KHÔNG THẤY\n(Importance chỉ đếm lỗi màu\nhiện tại)", GREY,
         "$\\eta=\\ell/w_{\\min}$\n(so kích thước với\nbước sóng ảnh)"),
        ("Chia Gaussian không đều\ntheo hướng (dẹt chỉ vi phạm\n1-2 trục vẫn bị chia đều 3 trục)",
         "KHÔNG THẤY\n(split isotropic, luôn\nchia trục dài nhất)", GREY,
         "$(k_x,k_y,k_z)$ độc lập\n(§17.5,\ndensify_and_split_structgs)"),
        ("Ngưỡng densify không co giãn\ntheo độ phân giải ảnh\n(Importance tuyệt đối)",
         "ĐÚNG — ĐIỂM YẾU\nTHẬT CỦA ADC GỐC", RED,
         "high_ratio / low_ratio\n(ngưỡng dạng tỉ lệ,\nbất biến độ phân giải)"),
    ]

    x1, w1 = 0.3, 4.0
    x2, w2 = 4.7, 3.7
    x3, w3 = 8.9, 3.8
    row_h = 1.45
    gap = 0.35
    y0 = 0.6

    for i, (prob, seen, seen_color, fix) in enumerate(rows):
        y = y0 + (2 - i) * (row_h + gap)
        rounded_box(ax, (x1, y), w1, row_h, prob, fc="#fffaf0", ec=DARK,
                    fontsize=8.6, fontweight="bold")
        rounded_box(ax, (x2, y), w2, row_h, "3DGS gốc thấy vấn đề này?\n\n" + seen,
                    fc=BOXFACE, ec=seen_color, fontsize=8.3, textcolor=seen_color,
                    fontweight="bold")
        rounded_box(ax, (x3, y), w3, row_h, "SAD-GS giải bằng:\n\n" + fix,
                    fc=GREEN, ec=GREEN, fontsize=8.3, textcolor="white",
                    fontweight="bold")
        arrow(ax, (x1 + w1, y + row_h / 2), (x2, y + row_h / 2), color=DARK)
        arrow(ax, (x2 + w2, y + row_h / 2), (x3, y + row_h / 2), color=DARK)

    out = os.path.join(HERE, "deep17f_03_new_problems.png")
    fig.savefig(out, dpi=175, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ----------------------------------------------------------------------
# FIGURE 4 — end-to-end pipeline recap 17.1 -> 17.5
# ----------------------------------------------------------------------
def fig04_pipeline_recap():
    """
    INPUT: các công thức trung tâm của mục 17.1-17.5 trong
           17-sadgs-structure-aware-densification.md — hình MỚI (tổng hợp),
           không sao chép nguyên một bảng nào, chỉ nối các \\boxed{...} đã có
           thành một sơ đồ luồng.
    NGUỒN từng khối (SADGS/... : dòng, tương ứng mục sách):
      17.1 Structure tensor Di Zenzo   loss_utils.py:170-230
      17.2 Multiscale (v1/v2)          loss_utils.py:232-387
      17.3 eta = l/w_min               freq_utils.py:116-373
      17.4 high/low ratio, tau=0.8     freq_utils.py:395-409, arguments:148-149
      17.5 anisotropic split k=ceil(sqrt(eta)) gaussian_model.py:638-831
    OUTPUT: một hàng 5 khối nối bằng mũi tên, mỗi khối có tên mục + công thức
            rút gọn + 1 số ngưỡng thật (1.0 / 0.1 / 0.8) lấy đúng từ mục 17.4,
            dùng làm hình mở đầu hoặc hình tổng kết đóng chương.
    """
    fig, ax = plt.subplots(figsize=(15, 5.0), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.0)

    ax.text(7.5, 4.65, "Toàn tuyến SAD-GS: từ ảnh GT tới Gaussian con dị hướng (§17.1 → §17.5)",
            ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

    stages = [
        ("§17.1\nStructure tensor",
         "Di Zenzo (RGB)\n$S=(S_{xx},S_{xy},S_{yy})$",
         GOLD, "loss_utils.py\n:170-230"),
        ("§17.2\nMultiscale",
         "gộp octave, trọng số\n$w_i=R_i^{3}$, $f_i^2$",
         "#c98a2b", "loss_utils.py\n:232-387"),
        ("§17.3\n$\\eta$",
         "$\\eta_k=\\dfrac{\\ell_k}{w_{\\min}}$\n$w_{\\min}=\\frac{1}{\\sqrt{\\lambda_1}+10^{-5}}$",
         BLUE, "freq_utils.py\n:116-373"),
        ("§17.4\nMultiview ratio",
         "high: $\\eta{>}1.0$\nlow: $\\eta{\\leq}0.1$\nsplit: ratio$>0.8$",
         "#7a4fa0", "freq_utils.py\n:395-409"),
        ("§17.5\nAnisotropic split",
         "$k_{\\text{axis}}=\\lceil\\sqrt{\\max(\\eta,1)}\\rceil$\n$N_{\\text{con}}=k_xk_yk_z$",
         GREEN, "gaussian_model.py\n:638-831"),
    ]

    n = len(stages)
    w = 2.5
    gap = 0.45
    total_w = n * w + (n - 1) * gap
    x0 = (15 - total_w) / 2
    y = 1.7
    h = 2.4

    for i, (title, formula, color, src) in enumerate(stages):
        x = x0 + i * (w + gap)
        rounded_box(ax, (x, y), w, h, "", fc=BOXFACE, ec=color, lw=2.2)
        ax.text(x + w / 2, y + h - 0.4, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - 0.05, formula, ha="center", va="center",
                fontsize=8.6, color=DARK)
        ax.text(x + w / 2, y + 0.28, src, ha="center", va="center",
                fontsize=6.8, color=GREY, style="italic")
        if i < n - 1:
            arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2), color=DARK, lw=2.0)

    ax.text(7.5, 0.55,
            "Nhánh ảnh (17.1-17.2) chạy 1 lần trước train; nhánh Gaussian (17.3-17.4) mỗi 10 iteration; "
            "§17.5 chỉ chạy khi split_mask=True.",
            ha="center", va="center", fontsize=8.4, color=DARK)

    out = os.path.join(HERE, "deep17f_04_pipeline_recap.png")
    fig.savefig(out, dpi=175, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    fig01_readme_mapping()
    fig02_weakness_matrix()
    fig03_new_problems()
    fig04_pipeline_recap()
