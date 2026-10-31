"""Diagnose densification: measure the per-Gaussian positional gradient
distribution that drives adaptive density control, and check whether plain
optimisation (no densification) converges."""
import argparse
import sys

import torch

sys.path.insert(0, "/home/whu/splatforge")
from src.dataset import Scene, init_point_cloud_visual_hull  # noqa: E402
from src.gaussian_model import GaussianModel  # noqa: E402
from src.metrics import psnr, dssim_loss  # noqa: E402
from src.rasterizer import rasterize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/whu/splatforge/data/nerf_synthetic/lego")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--hull_res", type=int, default=160)
    ap.add_argument("--init_points", type=int, default=80000)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    scene = Scene(args.data_root, downscale=args.downscale).to(device)
    pts, cols, voxel = init_point_cloud_visual_hull(
        scene, resolution=args.hull_res, max_points=args.init_points)
    init_scale = max(0.6 * voxel, 1.5 * scene.pixel_world)
    print(f"points={pts.shape[0]} voxel={voxel:.4f} init_scale={init_scale:.5f} "
          f"extent={scene.extent:.3f} pixel_world={scene.pixel_world:.5f}")
    model = GaussianModel(sh_degree=3, device=device, init_points=pts,
                          init_colors=cols, spatial_lr_scale=scene.extent,
                          init_scale=init_scale)

    t0 = None
    import time
    t0 = time.time()
    for it in range(1, args.iters + 1):
        model.update_learning_rate(it, args.iters)
        cam = scene.train[torch.randint(len(scene.train), (1,)).item()]
        img, aux = rasterize(model.means, model.cov3d(),
                             model.opacities().squeeze(-1),
                             model.colors(cam.w2c_t.new_tensor(cam.center)),
                             cam.w2c_r, cam.w2c_t, cam.fx, cam.fy, cam.cx, cam.cy,
                             cam.W, cam.H, background=scene.background,
                             tile_size=8, pair_budget=131072, aa_sigma_px=0.3)
        gt = cam.image
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * dssim_loss(
            img.unsqueeze(0), gt.unsqueeze(0))
        loss.backward()
        with torch.no_grad():
            g = model._xyz.grad
            gn = g.norm(dim=-1)
            vis = aux["visible"]
            if it % 50 == 0 or it == 1:
                gv = gn[vis]
                qs = torch.quantile(gv, torch.tensor([0.5, 0.9, 0.99, 0.999],
                                                     device=gv.device))
                print(f"[{it:4d}] loss={loss.item():.4f} psnr={psnr(img.clamp(0,1), gt):.2f} "
                      f"n={model._xyz.shape[0]} pairs={aux['num_pairs']} "
                      f"visible={int(vis.sum())} "
                      f"grad q50={qs[0]:.2e} q90={qs[1]:.2e} q99={qs[2]:.2e} q999={qs[3]:.2e} "
                      f"max={gv.max():.2e} mean={gv.mean():.2e} "
                      f"lr_xyz={model.optimizer.param_groups[0]['lr']:.2e} "
                      f"{time.time()-t0:.0f}s", flush=True)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        ps = []
        for cam in scene.val[:20]:
            im, _ = rasterize(model.means, model.cov3d(),
                              model.opacities().squeeze(-1),
                              model.colors(cam.w2c_t.new_tensor(cam.center)),
                              cam.w2c_r, cam.w2c_t, cam.fx, cam.fy, cam.cx, cam.cy,
                              cam.W, cam.H, background=scene.background,
                              tile_size=8, pair_budget=131072, aa_sigma_px=0.3)
            ps.append(psnr(im.clamp(0, 1), cam.image))
        print(f"no-densify VAL psnr={sum(ps)/len(ps):.2f} after {args.iters} iters")


if __name__ == "__main__":
    main()
