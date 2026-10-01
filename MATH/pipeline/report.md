# Công thức toán học trong `pipeline/report.py`

File này chủ yếu là **trực quan hoá** (pandas + matplotlib): gộp lịch sử train, dựng bảng leaderboard, vẽ biểu đồ so sánh. Không có công thức mới — chỉ hiển thị lại các đại lượng đã tính ở nơi khác (`pipeline/score.py`, `pipeline/trainer.py`). Dưới đây là các phép tính số học thực sự xuất hiện trong file.

## 1. Công thức điểm hiển thị trên biểu đồ (`plot_training`)

Trích nguyên văn (`pipeline/report.py`, dòng 66–67, 74–76):

```python
    ax.plot(part["iter"], part["score"], marker="o", ms=3, label=scene)
ax.set(xlabel="iteration", ylabel="Score", title="Score = .4(1-LPIPS)+.3SSIM+.3PSNR_norm")
...
    ax.plot(part["iter"], part["psnr"], marker="o", ms=3, label=f"{scene} PSNR")
    ax.plot(part["iter"], part["ssim"] * 30, ls="--", lw=1, label=f"{scene} SSIM×30")
    ax.plot(part["iter"], part["lpips"] * 30, ls=":", lw=1, label=f"{scene} LPIPS×30")
```

Tiêu đề biểu đồ (dòng 67) nhắc lại định nghĩa điểm tổng hợp (định nghĩa gốc nằm ở `pipeline/score.py`, không tính lại trong file này — `report.py` chỉ đọc cột `score` đã có sẵn trong `history`):

$$
\text{Score} = 0.4\,(1-\text{LPIPS}) + 0.3\,\text{SSIM} + 0.3\,\text{PSNR}_{norm}
$$

Trên cùng một trục (dòng 75–76), SSIM và LPIPS được co giãn để cùng thang với PSNR (chỉ phục vụ hiển thị, không phải công thức khoa học):

$$
\text{SSIM}_{plot} = 30\cdot\text{SSIM}, \qquad \text{LPIPS}_{plot} = 30\cdot\text{LPIPS}
$$

## 2. Tiến bộ giữa hai lần chấm (`d_score`)

Trích nguyên văn (`pipeline/report.py`, dòng 82–87):

```python
    if "d_score" in history.columns:
        for scene in scenes:
            part = history[history.scene == scene].dropna(subset=["d_score"])
            ax.bar(part["iter"], part["d_score"], width=max(1, history["iter"].max() / 40),
                   alpha=.6, label=scene)
        ax.axhline(0, c="k", lw=.8)
```

Cột `d_score` chỉ được **đọc và vẽ** (dạng bar chart theo `iter`) tại đây — giá trị của nó được tính sẵn ở `pipeline/trainer.py`:

$$
\Delta\text{Score}_t = \text{Score}_t - \text{Score}_{t-1}
$$

## 3. Trung bình leaderboard (`leaderboard`)

Trích nguyên văn (`pipeline/report.py`, dòng 42–43):

```python
    numeric = frame.select_dtypes("number")
    frame.loc["MEAN"] = numeric.mean().reindex(frame.columns)
```

Với $M$ là tập các cột số của bảng kết quả theo scene $s=1,\dots,N$:

$$
\text{MEAN}_m = \frac{1}{N}\sum_{s=1}^{N} \text{frame}_{s,m}, \qquad m \in M
$$

(`numeric.mean()` của pandas — trung bình cộng số học đơn giản theo từng cột, bỏ qua `NaN` theo mặc định của pandas.)

## 4. Trung bình trong `plot_leaderboard`

Trích nguyên văn (`pipeline/report.py`, dòng 129–131):

```python
    axes[0].bar(frame.index, frame[column], color="#3b7dd8")
    axes[0].axhline(frame[column].mean(), ls="--", c="crimson",
                    label=f"trung bình {frame[column].mean():.4f}")
```

Đường trung bình vẽ trên biểu đồ cột:

$$
\bar{x} = \frac{1}{N}\sum_{s=1}^{N} x_s
$$

với $x_s$ là cột `score` (hoặc `live_score` khi không có ảnh test thật, theo logic chọn cột ở dòng 115–122: `for candidate in ("score", "live_score")`).

## 5. Chọn mẫu ảnh minh hoạ (`show_samples`)

Trích nguyên văn (`pipeline/report.py`, dòng 179):

```python
    picks = manifest["files"][:: max(1, len(manifest["files"]) // n)][:n]
```

Lấy $n$ ảnh cách đều trong danh sách $K=$ `len(manifest["files"])` file bằng bước nhảy nguyên (slicing `[::step][:n]` của Python):

$$
\text{step} = \max(1, \lfloor K/n \rfloor), \qquad \text{indices} = \{0, \text{step}, 2\cdot\text{step}, \dots\}[:n]
$$

Không phải công thức toán học theo nghĩa khoa học, chỉ là lấy mẫu đều (uniform subsampling) để hiển thị.

---

## Kiểm chứng tính đúng sai

Đối chiếu từng công thức với `pipeline/report.py` thật (203 dòng):

| Mục | Mô tả trong tài liệu | Dòng code thật | Khớp? |
|---|---|---|---|
| Công thức Score (tiêu đề `plot_training`) | $0.4(1-\text{LPIPS})+0.3\text{SSIM}+0.3\text{PSNR}_{norm}$ | dòng 67 (chuỗi tiêu đề, không tính lại giá trị) | **Khớp** — chỉ là chuỗi hiển thị, giá trị `score` đã có sẵn trong `history` |
| $\text{SSIM}_{plot}=30\cdot\text{SSIM}$, $\text{LPIPS}_{plot}=30\cdot\text{LPIPS}$ | nhân 30 để cùng thang PSNR | dòng 75–76 | **Khớp** |
| $\Delta\text{Score}_t$ | chỉ vẽ cột `d_score` có sẵn | dòng 82–87 | **Khớp** — không tính lại `d_score`, chỉ `dropna` + vẽ bar |
| $\text{MEAN}_m$ | `numeric.mean().reindex(...)` | dòng 42–43 | **Khớp** |
| $\bar x$ (đường trung bình `plot_leaderboard`) | `frame[column].mean()` | dòng 130–131 | **Khớp** |
| $\text{step}=\max(1,\lfloor K/n\rfloor)$ | slicing `[::max(1, len(...)//n)][:n]` | dòng 179 | **Khớp** (`//` là chia nguyên Python, đúng $\lfloor\cdot\rfloor$ vì $K,n>0$) |

Không phát hiện sai sót toán học nào so với bản mô tả cũ — toàn bộ nội dung đã đúng, chỉ bổ sung trích dẫn code nguyên văn kèm số dòng.

## Ví dụ số

Giả sử leaderboard có $N=3$ scene với cột `score` $=(0.72,\ 0.65,\ 0.80)$ (các scene khác coi như `NaN` ở cột này). Áp dụng mục 3–4:

$$
\text{MEAN}_{\text{score}} = \frac{0.72+0.65+0.80}{3} = \frac{2.17}{3} = 0.72333\ldots
$$

Dòng `frame.loc["MEAN", "score"]` sẽ in ra (theo dòng 48: `print(f"... {frame.loc['MEAN', 'score']:.4f}")`) chuỗi `0.7233`. Tương tự, nếu `show_samples` gặp $K=10$ file và $n=3$: $\text{step}=\max(1,\lfloor10/3\rfloor)=3$, chọn `files[0], files[3], files[6], files[9]` rồi cắt `[:3]` → lấy đúng 3 ảnh đầu tiên của dãy bước nhảy 3: `files[0], files[3], files[6]`.

---

## Bảng tương ứng cú pháp ↔ công thức

| Cú pháp / code gốc (dòng) | Công thức tương ứng |
|---|---|
| `title="Score = .4(1-LPIPS)+.3SSIM+.3PSNR_norm"` (dòng 67) | $\text{Score} = 0.4(1-\text{LPIPS}) + 0.3\,\text{SSIM} + 0.3\,\text{PSNR}_{norm}$ |
| `part["ssim"] * 30` (dòng 75) | $\text{SSIM}_{plot} = 30 \cdot \text{SSIM}$ |
| `part["lpips"] * 30` (dòng 76) | $\text{LPIPS}_{plot} = 30 \cdot \text{LPIPS}$ |
| `ax.bar(part["iter"], part["d_score"], ...)` (dòng 85) | $\Delta\text{Score}_t = \text{Score}_t - \text{Score}_{t-1}$ (giá trị tính sẵn ở `trainer.py`) |
| `numeric.mean().reindex(...)` → `frame.loc["MEAN"]` (dòng 42–43) | $\text{MEAN}_m = \frac{1}{N}\sum_{s=1}^N \text{frame}_{s,m}$ |
| `frame[column].mean()` (dòng 130) | $\bar{x} = \frac{1}{N}\sum_{s=1}^N x_s$ |
| `manifest["files"][:: max(1, len(...)//n)][:n]` (dòng 179) | $\text{step} = \max(1,\lfloor K/n\rfloor)$, lấy mẫu đều |
