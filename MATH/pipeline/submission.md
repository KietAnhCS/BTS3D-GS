# Công thức toán học trong `pipeline/submission.py`

File này **không định nghĩa công thức toán mới**: nó render test camera, gọi lại `composite_score` từ `pipeline/score.py` (xem `score.md`), và đóng gói/kiểm tra file ZIP. Phần toán duy nhất là phép **trung bình cộng** ba metric trước khi đưa vào công thức Score, cùng vài công thức định dạng tên file/kích thước thuần kỹ thuật.

---

## 1. Trung bình metric trên toàn bộ ảnh render (`render_scene`, `pipeline/submission.py` dòng 68–95)

Trích nguyên văn phần tích luỹ và trung bình hoá (dòng 68–95):

```python
    psnrs, ssims, lpipss, files = [], [], [], []
    with torch.no_grad():
        for index, cam in enumerate(tqdm(cams, desc=f"render[{scene}]", dynamic_ncols=True), start=1):
            rendered = torch.clamp(render_structgs(cam, gaussians, pipe, background, cfg.mult)["render"],
                                   0.0, 1.0)
            name = _image_name(index, cfg)
            torchvision.utils.save_image(rendered, os.path.join(out_dir, name))
            files.append(dict(file=name, source=cam.image_name,
                              width=cam.image_width, height=cam.image_height))
            if score:
                gt = torch.clamp(cam.original_image.to("cuda")[:3], 0.0, 1.0)
                psnrs.append(psnr_fn(rendered, gt).mean().item())
                ssims.append(ssim_fn(rendered.unsqueeze(0), gt.unsqueeze(0)).item())
                lpipss.append(lpips_fn(rendered, gt, net_type=cfg.lpips_net_report).mean().item())
                del gt
            del rendered
            cam.original_image = None                  # trả VRAM ngay, ảnh này xong việc

    info = dict(scene=scene, images=len(files), out_dir=out_dir,
                width=files[0]["width"] if files else None,
                height=files[0]["height"] if files else None,
                camera_source="test_poses.csv" if pose_info else f"llffhold={cfg.llffhold}",
                poses=pose_info, files=files)
    if psnrs:
        psnr_val = sum(psnrs) / len(psnrs)
        ssim_val = sum(ssims) / len(ssims)
        lpips_val = sum(lpipss) / len(lpipss)
        score_val, psnr_norm = composite_score(psnr_val, ssim_val, lpips_val, cfg.psnr_max)
```

Với $M$ ảnh có ground-truth trong một scene (dòng 92–94):

$$
\overline{\mathrm{PSNR}} = \frac{1}{M}\sum_{j=1}^{M}\mathrm{PSNR}_j,\qquad
\overline{\mathrm{SSIM}} = \frac{1}{M}\sum_{j=1}^{M}\mathrm{SSIM}_j,\qquad
\overline{\mathrm{LPIPS}} = \frac{1}{M}\sum_{j=1}^{M}\mathrm{LPIPS}_j
$$

Ba giá trị này được truyền vào `composite_score(psnr_val, ssim_val, lpips_val, cfg.psnr_max)` (dòng 95, định nghĩa ở `pipeline/score.py`) để ra $\mathrm{Score}$ và $\mathrm{PSNR}_{norm}$ — **không lặp lại công thức Score ở đây**, chỉ gọi hàm có sẵn. Lưu ý mỗi $\mathrm{PSNR}_j,\mathrm{SSIM}_j,\mathrm{LPIPS}_j$ tự nó cũng đã là trung bình trên không gian ảnh (`psnr_fn(...).mean()`, `ssim_fn(...)`) nên $\overline{\mathrm{PSNR}}$ ở đây là trung bình **hai lớp**: trung bình pixel/kênh trong mỗi ảnh (ở `image_utils.py`/`loss_utils.py`), rồi trung bình tiếp qua $M$ ảnh (ở đây).

## 2. Định dạng tên file ảnh (`_image_name`, `pipeline/submission.py` dòng 18–19)

Trích nguyên văn (dòng 18–19):

```python
def _image_name(index, cfg):
    return f"{index:0{cfg.submission_digits}d}{cfg.submission_ext}"
```

Không phải công thức toán nhưng là một phép ánh xạ số nguyên → chuỗi, đáng ghi vì được dùng để kiểm tra tính liên tục của bộ ảnh:

$$
\text{name}(i) = \mathrm{zero\_pad}(i,\ d) \,\Vert\, \text{ext}, \qquad d=\texttt{cfg.submission\_digits}
$$

ví dụ $d=4$: $\text{name}(1) = \texttt{"0001.png"}$.

## 3. Kiểm tra kích thước & số lượng ảnh so với CSV (`verify`, `pipeline/submission.py` dòng 160–177)

Trích nguyên văn phần so khớp (dòng 160–177):

```python
            wanted = [_image_name(i, cfg) for i in range(1, len(files) + 1)]
            if files != wanted:
                problems.append(f"{scene}: tên file không liên tục ({files[:3]} ...)")
            with zf.open(f"{scene}/{files[0]}") as handle:
                width, height = Image.open(io.BytesIO(handle.read())).size
            row = dict(scene=scene, images=len(files), width=width, height=height)

            # đối chiếu thẳng với test_poses.csv: số ảnh và kích thước render
            csv_path = testposes.poses_csv(cfg, scene)
            if csv_path:
                want = testposes.read_rows(csv_path)
                row["expected"] = len(want)
                if len(files) != len(want):
                    problems.append(f"{scene}: {len(files)} ảnh nhưng CSV có {len(want)} pose")
                want_w, want_h = int(float(want[0]["width"])), int(float(want[0]["height"]))
                if (width, height) != (want_w, want_h):
                    problems.append(f"{scene}: ảnh {width}x{height} nhưng CSV yêu cầu "
                                    f"{want_w}x{want_h}")
```

So khớp trực tiếp (không phải công thức dẫn xuất, chỉ là phép so sánh bằng):

$$
|\text{files}| \stackrel{?}{=} |\text{test\_poses.csv}|, \qquad (W_{render}, H_{render}) \stackrel{?}{=} (W_{csv}, H_{csv})
$$

## 4. Dung lượng file (`build_zip`, `pipeline/submission.py` dòng 133)

Trích nguyên văn (dòng 133):

```python
    size_mb = os.path.getsize(cfg.submission_zip) / 1024 ** 2
```

$$
\text{size}_{MB} = \frac{\text{size}_{bytes}}{1024^2}
$$

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức LaTeX |
|---|---|
| `sum(psnrs)/len(psnrs)` | $\overline{\mathrm{PSNR}}=\frac1M\sum_j \mathrm{PSNR}_j$ |
| `sum(ssims)/len(ssims)` | $\overline{\mathrm{SSIM}}=\frac1M\sum_j \mathrm{SSIM}_j$ |
| `sum(lpipss)/len(lpipss)` | $\overline{\mathrm{LPIPS}}=\frac1M\sum_j \mathrm{LPIPS}_j$ |
| `composite_score(psnr_val, ssim_val, lpips_val, cfg.psnr_max)` | gọi công thức $\mathrm{Score}$ từ `score.py` (không định nghĩa lại) |
| `f"{index:0{cfg.submission_digits}d}{cfg.submission_ext}"` | $\text{name}(i)=\mathrm{zero\_pad}(i,d)\,\Vert\,\text{ext}$ |
| `len(files) != len(want)` | $|\text{files}| \ne |\text{CSV}|$ |
| `(width, height) != (want_w, want_h)` | $(W_{render},H_{render}) \ne (W_{csv},H_{csv})$ |
| `os.path.getsize(cfg.submission_zip) / 1024 ** 2` | $\text{size}_{MB}=\text{size}_{bytes}/1024^2$ |
