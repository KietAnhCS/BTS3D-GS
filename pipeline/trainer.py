"""Vòng lặp huấn luyện SADGS (structgs) + móc theo dõi Score ngay trong lúc train.

Mirror sát `train.py` gốc của repo (render_structgs, densify_and_prune_structgs,
structure-tensor multiscale, update_freq_stats_online, batch training) — chỉ khác
ở chỗ thêm eval/checkpoint định kỳ và co giãn lịch densify theo `cfg.iterations`
(xem cfg.densify_until_frac / cfg.opacity_reset_frac trong pipeline/config.py).
"""

import gc
import os
import random
import time
from argparse import ArgumentParser, Namespace

from pipeline import data as data_mod
from pipeline.env import mem, show_mem
from pipeline.score import evaluate_cameras


def build_args(cfg, scene, model_path=None, iterations=None, resolution=None, extra=(),
               write_cfg=True):
    """Dựng Namespace tham số của SADGS cho một scene rồi ghi `cfg_args` để render lại."""
    from arguments import ModelParams, PipelineParams, OptimizationParams

    model_path = model_path or cfg.model_path(scene)
    n_iter = int(iterations or cfg.iterations)
    # Densify (và các lần reset opacity bên trong nó) phải kết thúc sớm hơn vòng
    # cuối, nếu không model dừng ngay sau một lần reset và PSNR sụp.
    densify_until = max(1, int(round(getattr(cfg, "densify_until_frac", 0.5) * n_iter)))
    # opacity_reset_interval gốc (3000, mặc định của OptimizationParams) được đặt cho
    # lịch 30k -- co giãn theo cùng nguyên tắc với densify_until_iter ở trên, nếu không
    # reset rơi lệch tỷ lệ và gây sụp PSNR giữa chừng (xem pipeline/config.py).
    opacity_reset = max(1, int(round(getattr(cfg, "opacity_reset_frac", 0.2) * n_iter)))
    parser = ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    argv = ["-s", data_mod.scene_path(cfg, scene), "-m", model_path,
            "-i", cfg.images_dir,
            "-r", str(cfg.resolution if resolution is None else resolution), "--eval",
            "--iterations", str(n_iter),
            "--position_lr_max_steps", str(n_iter),
            "--densify_until_iter", str(densify_until),
            "--opacity_reset_interval", str(opacity_reset),
            "--mult", str(cfg.mult)]
    if cfg.white_background:
        argv.append("-w")
    argv += list(cfg.train_extra_args) + list(extra)
    args = parser.parse_args(argv)

    os.makedirs(model_path, exist_ok=True)
    if write_cfg:
        with open(os.path.join(model_path, "cfg_args"), "w") as handle:
            handle.write(str(Namespace(**vars(args))))
    return args, lp.extract(args), op.extract(args), pp.extract(args)


def _save_checkpoint(scene_obj, iteration, previous=None, drop_previous=True):
    """Lưu `.ply` rồi xoá checkpoint giữa chừng trước đó (đĩa Colab không rộng)."""
    import shutil

    scene_obj.save(iteration)
    if drop_previous and previous is not None and previous != iteration:
        stale = os.path.join(scene_obj.model_path, "point_cloud", f"iteration_{previous}")
        shutil.rmtree(stale, ignore_errors=True)
    return iteration


def _precompute_structure_tensors(dataset, opt, scene_obj):
    """Structure tensor đa tỉ lệ cho từng ảnh train (xem utils/loss_utils.py, train.py).

    Gộp theo (height, width) rồi tính theo batch để không phải gọi lại cho mỗi ảnh.
    """
    import torch

    from utils.loss_utils import get_multiscale_structure_tensor_v1, get_multiscale_structure_tensor_v2

    cache = {}
    cameras_by_res = {}
    for cam in scene_obj.getTrainCameras():
        res = (cam.image_height, cam.image_width)
        cameras_by_res.setdefault(res, []).append(cam)

    batch_size = 100
    fn = get_multiscale_structure_tensor_v1 if opt.st_mode == "v1" else get_multiscale_structure_tensor_v2
    with torch.no_grad():
        for cams in cameras_by_res.values():
            for i in range(0, len(cams), batch_size):
                batch_cams = cams[i:i + batch_size]
                img_batch = torch.stack([cam.original_image.cuda() for cam in batch_cams], dim=0)
                st_batch = fn(img_batch, levels=opt.st_levels)
                for j, cam in enumerate(batch_cams):
                    cache[cam.image_name] = st_batch[j:j + 1]
    return cache


def train_scene(cfg, scene, iterations=None, tag=None, keep_model=False, quiet_eval=False):
    """Huấn luyện một scene.

    Trả về (result, gaussians, scene_obj); gaussians/scene_obj = None khi `keep_model=False`
    (mặc định — giải phóng RAM/VRAM ngay để scene kế tiếp có chỗ chạy).
    """
    import torch
    from tqdm.auto import tqdm

    from scene import Scene, GaussianModel
    from utils.loss_utils import l1_loss, l2_loss
    from utils.general_utils import safe_state
    from fused_ssim import fused_ssim as fast_ssim
    from gaussian_renderer import render_structgs
    from utils.freq_utils import sampling_cameras, update_freq_stats_online

    iterations = int(iterations or cfg.iterations)
    tag = tag or scene
    safe_state(True)

    args, dataset, opt, pipe = build_args(cfg, scene, iterations=iterations)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene_obj = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.compute_3d_filter:
        gaussians.compute_3D_filter(cameras=scene_obj.getTrainCameras())
    else:
        gaussians.filter_3D = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    structure_tensor_cache = _precompute_structure_tensors(dataset, opt, scene_obj)

    test_cams = scene_obj.getTestCameras()
    holdout = list(test_cams) if len(test_cams) else scene_obj.getTrainCameras()[::8]
    holdout_kind = "test" if len(test_cams) else "train_subset"

    # 3DGS gốc dùng 4000/8000 trên lịch 30k -> co theo cùng tỷ lệ với densify_until_iter.
    prune_iterations = {max(1, round(0.267 * opt.densify_until_iter)),
                        max(1, round(0.533 * opt.densify_until_iter))}

    history, prev = [], None
    viewpoint_stack, ema_loss = [], 0.0
    saved_at = None
    last = dict(score=float("nan"), psnr=float("nan"))
    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    def _refill_stack():
        stack = scene_obj.getTrainCameras().copy()
        if opt.camera_sampling == "fps":
            stack = sampling_cameras(stack, mode="fps", num_cams=len(stack))
        else:
            random.shuffle(stack)
        return stack

    score_every = max(1, min(cfg.score_every, iterations))
    pbar = tqdm(range(1, iterations + 1), desc=f"train[{tag}]", dynamic_ncols=True)
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        gaussians.oneupSHdegree()

        bg = torch.rand(3, device="cuda") if opt.random_background else background

        batch_loss = 0.0
        radii = visibility_filter = None
        for _ in range(opt.batch_size):
            if not viewpoint_stack:
                viewpoint_stack = _refill_stack()
            cam = viewpoint_stack.pop(0)

            render_pkg = render_structgs(cam, gaussians, pipe, bg, opt.mult)
            image, viewspace, visibility_filter, radii, cov2D = (
                render_pkg["render"], render_pkg["viewspace_points"],
                render_pkg["visibility_filter"].squeeze(1), render_pkg["radii"], render_pkg["cov2D"])

            gt = cam.original_image.cuda()
            ll1 = l1_loss(image, gt)
            ll2 = l2_loss(image, gt)
            ssim_value = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim_value) + opt.lambda_l2 * ll2
            loss.backward()
            batch_loss += loss.item()

            with torch.no_grad():
                if iteration < opt.densify_until_iter and iteration % 10 == 0:
                    update_freq_stats_online(cam, gaussians, cov2D, visibility_filter,
                                             structure_tensor_cache, viewspace_point_tensor=viewspace,
                                             grad_threshold=opt.freq_grad_threshold,
                                             transmittance_threshold=opt.freq_transmittance_threshold,
                                             opacity_threshold=opt.freq_opacity_threshold,
                                             eta_compute_mode=opt.eta_compute_mode)
            del image, gt, viewspace, ll1, ll2, ssim_value, cov2D

        with torch.no_grad():
            ema_loss = 0.4 * (batch_loss / opt.batch_size) + 0.6 * ema_loss

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                      radii[visibility_filter])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)

                is_normal = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
                is_warmup = (opt.warmup_densification and iteration > opt.densify_from_iter
                            and iteration % 100 == 0 and not is_normal)
                if is_normal:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    grads = gaussians.xyz_gradient_accum / gaussians.denom
                    grads[grads.isnan()] = 0.0
                    is_grad_high = torch.norm(grads, dim=-1) >= 1e-5

                    valid_mask = gaussians.accum_view_count > 0
                    high_ratio = torch.zeros_like(gaussians.accum_view_count)
                    low_ratio = torch.zeros_like(gaussians.accum_view_count)
                    high_ratio[valid_mask] = gaussians.eta_high_count[valid_mask] / gaussians.accum_view_count[valid_mask]
                    low_ratio[valid_mask] = gaussians.eta_low_count[valid_mask] / gaussians.accum_view_count[valid_mask]

                    split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high
                    prune_mask = (low_ratio > opt.prune_ratio_threshold) & valid_mask

                    max_high_eta = gaussians.max_eta_3ch
                    gaussians.densify_and_prune_structgs(
                        max_screen_size=size_threshold, min_opacity=0.1,
                        extent=scene_obj.cameras_extent, radii=radii, args=opt,
                        importance_score=gaussians.accum_view_count, pruning_score=None,
                        custom_split_mask=split_mask, custom_prune_mask=prune_mask,
                        viewspace_points_indices=None, max_eta_3ch=max_high_eta)

                    gaussians.accum_eta.zero_()
                    gaussians.accum_view_count.zero_()
                    gaussians.max_eta_3ch.zero_()
                    gaussians.accum_weights_valid.zero_()
                    gaussians.eta_high_count.zero_()
                    gaussians.eta_high_sum_3ch.zero_()
                    gaussians.eta_mid_count.zero_()
                    gaussians.eta_mid_sum_3ch.zero_()
                    gaussians.eta_low_count.zero_()
                elif is_warmup:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold,
                                               0.005, scene_obj.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(opt.opacity_reset_decay)

            if (iteration % 100 == 0 and iteration > opt.densify_until_iter
                    and iteration < iterations - 100 and opt.compute_3d_filter):
                gaussians.compute_3D_filter(cameras=scene_obj.getTrainCameras())

            if iteration in prune_iterations:
                prune_mask = (gaussians.get_opacity < 0.1).squeeze()
                gaussians.prune_points(prune_mask)

            if iteration < iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "hybrid":
                    visible = radii > 0
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.shoptimizer.step(visible, radii.shape[0])
                    gaussians.shoptimizer.zero_grad(set_to_none=True)

            # --- % tiến trình + điểm số hiện ngay trên thanh tqdm ---
            if iteration % 10 == 0:
                pbar.set_postfix_str(
                    f"loss {ema_loss:.4f} | G {gaussians._xyz.shape[0]:,} | "
                    f"Score {last['score']:.4f} | PSNR {last['psnr']:.2f}", refresh=False)

            if iteration % score_every == 0 or iteration == iterations:
                torch.cuda.empty_cache()
                evaluation = evaluate_cameras(gaussians, holdout, pipe, background, opt.mult,
                                              cfg.eval_views, cfg.psnr_max, cfg.lpips_net_live)
                usage = mem()
                row = dict(scene=scene, iter=iteration, pct=100.0 * iteration / iterations,
                           n_gauss=int(gaussians._xyz.shape[0]), ema_loss=ema_loss,
                           elapsed_s=time.time() - start,
                           ram_gb=usage["ram_used_gb"], vram_gb=usage.get("vram_alloc_gb", 0.0),
                           **(evaluation or {}))
                if evaluation is not None:
                    if prev is not None:
                        for key in ("score", "psnr", "ssim", "lpips"):
                            row[f"d_{key}"] = row[key] - prev[key]
                    prev, last = row, row
                    if not quiet_eval:
                        delta = row.get("d_score")
                        delta_txt = f"{delta:+.4f}" if delta is not None else "  n/a "
                        tqdm.write(
                            f"[{tag}] {row['pct']:5.1f}% iter {iteration:6d}"
                            f" | Score {row['score']:.4f} ({delta_txt})"
                            f" | PSNR {row['psnr']:5.2f} (norm {row['psnr_norm']:.3f})"
                            f" SSIM {row['ssim']:.4f} LPIPS {row['lpips']:.4f}"
                            f" | G {row['n_gauss']:>9,} | loss {ema_loss:.4f}"
                            f" | RAM {row['ram_gb']:.1f} VRAM {row['vram_gb']:.1f} GB")
                history.append(row)
                if usage["ram_used_gb"] > cfg.ram_soft_limit_gb:
                    gc.collect()
                    torch.cuda.empty_cache()

            # --- checkpoint định kỳ: mất session Colab thì vẫn còn .ply gần nhất ---
            if cfg.save_every and iteration % cfg.save_every == 0 and iteration < iterations:
                saved_at = _save_checkpoint(scene_obj, iteration, saved_at,
                                            cfg.keep_last_checkpoint)
                tqdm.write(f"[{tag}] checkpoint: point_cloud/iteration_{iteration}")

    pbar.close()
    _save_checkpoint(scene_obj, iterations, saved_at, cfg.keep_last_checkpoint)
    total_time = time.time() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    final = history[-1] if history else {}
    result = dict(scene=scene, tag=tag, model_path=scene_obj.model_path, iterations=iterations,
                  holdout_kind=holdout_kind, total_time_s=total_time, peak_vram_gb=peak_vram,
                  n_gauss=int(gaussians._xyz.shape[0]), history=history)
    for key in ("score", "psnr", "ssim", "lpips", "psnr_norm"):
        if key in final:
            result[key] = final[key]
    print(f"[{tag}] xong: {result['n_gauss']:,} gaussians | {total_time:.1f}s"
          f" | Score {result.get('score', float('nan')):.4f} | peak VRAM {peak_vram:.2f} GB")

    if not keep_model:
        del gaussians, scene_obj
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        show_mem(f"sau train {tag}")
        return result, None, None
    return result, gaussians, scene_obj
