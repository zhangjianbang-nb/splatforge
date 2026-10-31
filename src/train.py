"""Training entry point: fit a 3D Gaussian Splatting scene to posed images."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .dataset import Scene, init_point_cloud, init_point_cloud_visual_hull
from .gaussian_model import GaussianModel
from .metrics import psnr, dssim_loss
from .rasterizer import rasterize


def render(model, cam, background, tile_size, pair_budget, aa_sigma_px,
           radius_cap=64.0, with_aux=False):
    means = model.means
    cov = model.cov3d()
    op = model.opacities().squeeze(-1)
    col = model.colors(cam.w2c_t.new_tensor(cam.center))
    out = rasterize(means, cov, op, col, cam.w2c_r, cam.w2c_t,
                    cam.fx, cam.fy, cam.cx, cam.cy, cam.W, cam.H,
                    background=background, tile_size=tile_size,
                    pair_budget=pair_budget, aa_sigma_px=aa_sigma_px,
                    radius_cap=radius_cap)
    return out if with_aux else out[0]


@torch.no_grad()
def evaluate(model, cams, background, tile_size, pair_budget, aa_sigma_px, radius_cap):
    model.eval_mode = True
    ps, ss = [], []
    from .metrics import ssim
    for cam in cams:
        img = render(model, cam, background, tile_size, pair_budget,
                     aa_sigma_px, radius_cap)
        ps.append(psnr(img.clamp(0, 1), cam.image))
        ss.append(ssim(img.clamp(0, 1).unsqueeze(0), cam.image.unsqueeze(0)))
    return sum(ps) / len(ps), sum(ss) / len(ss)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    scene = Scene(args.data_root, downscale=args.downscale).to(device)
    print(f"scene={os.path.basename(args.data_root)} resolution={scene.W}x{scene.H} "
          f"train={len(scene.train)} val={len(scene.val)} test={len(scene.test)} "
          f"extent={scene.extent:.3f}", flush=True)

    if args.init == "hull":
        pts, cols, voxel = init_point_cloud_visual_hull(
            scene, resolution=args.hull_res, max_points=args.init_points,
            seed=args.seed)
        # splats should span roughly the voxel spacing so the carved surface is
        # covered, but not much more or the pair count explodes
        init_scale = max(0.6 * voxel, 1.5 * scene.pixel_world)
    else:
        pts, cols = init_point_cloud(scene, n_points=args.init_points,
                                     seed=args.seed)
        init_scale = 1.5 * scene.pixel_world
    print(f"init={args.init}: {pts.shape[0]} points, init_scale={init_scale:.5f} "
          f"(world per pixel {scene.pixel_world:.5f})", flush=True)
    model = GaussianModel(sh_degree=args.sh_degree, device=device,
                          init_points=pts, init_colors=cols,
                          spatial_lr_scale=scene.extent, init_scale=init_scale)

    log = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        model.update_learning_rate(it, args.iters)
        if it % args.sh_warmup == 0:
            model.oneup_sh_degree()

        cam = scene.train[torch.randint(len(scene.train), (1,)).item()]
        img, aux = render(model, cam, scene.background, args.tile_size,
                          args.pair_budget, args.aa_sigma_px, args.radius_cap,
                          with_aux=True)
        gt = cam.image
        l1 = (img - gt).abs().mean()
        loss = (1.0 - args.lambda_dssim) * l1 + args.lambda_dssim * dssim_loss(
            img.unsqueeze(0), gt.unsqueeze(0))
        loss.backward()

        with torch.no_grad():
            grad = model._xyz.grad
            if grad is not None:
                # W is a rotation, so the world-space gradient norm equals the
                # view-space one used by the original densification heuristic
                model.add_densification_stats(grad.norm(dim=-1), aux["visible"])
                model.max_radii2d[aux["visible"]] = torch.maximum(
                    model.max_radii2d[aux["visible"]], aux["radii"][aux["visible"]])
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if args.densify_until and it < args.densify_until and it >= args.densify_from \
                and it % args.densify_every == 0:
            n_clone, n_split = model.densify_and_prune(
                args.densify_grad_threshold, args.min_opacity, scene.extent,
                args.max_screen_size, args.percent_dense, args.max_gaussians)
            log.append({"iter": it, "event": "densify", "n": model._xyz.shape[0],
                        "clone": n_clone, "split": n_split})
            with torch.no_grad():
                sc = model.scales().max(dim=1).values
                big = float((sc > args.percent_dense * scene.extent).float().mean())
                med = float(sc.median())
            print(f"[{it:6d}] densify -> {model._xyz.shape[0]} gaussians "
                  f"(+{n_clone} clones, +{n_split} splits) "
                  f"scale med={med:.4f} frac_above_split_thr={big:.3f}", flush=True)

        if it % args.opacity_reset_every == 0 and it < args.densify_until:
            model.reset_opacity()
            print(f"[{it:6d}] opacity reset", flush=True)

        if it % args.log_every == 0 or it == 1:
            dt = time.time() - t0
            msg = (f"[{it:6d}] loss={loss.item():.4f} psnr_train={psnr(img.clamp(0,1), gt):.2f} "
                   f"n={model._xyz.shape[0]} pairs={aux['num_pairs']} "
                   f"sh={model.active_sh_degree} {dt/max(it,1)*1000:.0f}ms/it")
            print(msg, flush=True)
            log.append({"iter": it, "loss": loss.item(), "n": model._xyz.shape[0]})

        if it % args.eval_every == 0 or it == args.iters:
            p, s = evaluate(model, scene.val, scene.background, args.tile_size,
                            args.pair_budget, args.aa_sigma_px, args.radius_cap)
            print(f"[{it:6d}] VAL psnr={p:.2f} ssim={s:.4f} "
                  f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
            log.append({"iter": it, "val_psnr": p, "val_ssim": s})

    # ---- final artifacts ----
    p_val, s_val = evaluate(model, scene.val, scene.background, args.tile_size,
                            args.pair_budget, args.aa_sigma_px, args.radius_cap)
    p_test, s_test = evaluate(model, scene.test, scene.background, args.tile_size,
                              args.pair_budget, args.aa_sigma_px, args.radius_cap)
    print(f"FINAL val psnr={p_val:.2f} ssim={s_val:.4f} | "
          f"test psnr={p_test:.2f} ssim={s_test:.4f}", flush=True)

    model.to_ply(os.path.join(args.out, "point_cloud.ply"), args.aa_sigma_px)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"val_psnr": p_val, "val_ssim": s_val,
                   "test_psnr": p_test, "test_ssim": s_test,
                   "n_gaussians": int(model._xyz.shape[0]),
                   "resolution": [scene.W, scene.H],
                   "train_minutes": (time.time() - t0) / 60.0,
                   "log": log}, f, indent=2)
    print(f"artifacts written to {args.out}", flush=True)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--init_points", type=int, default=80000)
    p.add_argument("--init", choices=["hull", "rays"], default="hull")
    p.add_argument("--hull_res", type=int, default=192)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--sh_warmup", type=int, default=1000)
    p.add_argument("--lambda_dssim", type=float, default=0.2)
    p.add_argument("--tile_size", type=int, default=8)
    p.add_argument("--pair_budget", type=int, default=131072)
    p.add_argument("--aa_sigma_px", type=float, default=0.3)
    p.add_argument("--radius_cap", type=float, default=64.0)
    p.add_argument("--densify_from", type=int, default=500)
    p.add_argument("--densify_until", type=int, default=9000)
    p.add_argument("--densify_every", type=int, default=100)
    p.add_argument("--densify_grad_threshold", type=float, default=2e-4)
    p.add_argument("--min_opacity", type=float, default=0.005)
    p.add_argument("--max_screen_size", type=float, default=64.0)
    p.add_argument("--percent_dense", type=float, default=0.01)
    p.add_argument("--max_gaussians", type=int, default=400000)
    p.add_argument("--opacity_reset_every", type=int, default=3000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
