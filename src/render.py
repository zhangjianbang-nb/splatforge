"""Render novel views from a trained checkpoint and export visual artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import torch
from PIL import Image

from .dataset import Camera, Scene
from .gaussian_model import GaussianModel
from .metrics import psnr, ssim
from .rasterizer import rasterize


def load_model(scene, device, sh_degree=3, init_points=20000, seed=0):
    """Rebuild a GaussianModel with the same parameter layout as training.

    The parameters are overwritten by ``load_ply`` immediately afterwards, so
    the placeholder cloud is deliberately small and cheap to build.
    """
    from .dataset import init_point_cloud
    pts, cols = init_point_cloud(scene, n_points=init_points, seed=seed)
    return GaussianModel(sh_degree=sh_degree, device=device,
                         init_points=pts, init_colors=cols,
                         spatial_lr_scale=scene.extent, init_scale=0.01)


def load_ply(model, path):
    """Load the binary PLY written by GaussianModel.to_ply back into parameters."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        count = int([h for h in header if h.startswith("element vertex")][0].split()[-1])
        data = np.frombuffer(f.read(count * 62 * 4), dtype=np.float32).reshape(count, 62)
    xyz = data[:, 0:3]
    feats = data[:, 6:54].reshape(count, 3, 16).transpose(0, 2, 1)   # back to [N,16,3]
    opacity = data[:, 54:55]
    scale = data[:, 55:58]
    rot = data[:, 58:62]
    dev = model.device
    model._xyz = torch.from_numpy(xyz.copy()).to(dev).requires_grad_(True)
    model._features = torch.from_numpy(feats.copy()).to(dev).requires_grad_(True)
    model._opacity = torch.from_numpy(opacity.copy()).to(dev).requires_grad_(True)
    model._scaling = torch.from_numpy(scale.copy()).to(dev).requires_grad_(True)
    model._rotation = torch.from_numpy(rot.copy()).to(dev).requires_grad_(True)
    model.active_sh_degree = model.sh_degree
    return model


def render_camera(model, cam, background, tile_size, pair_budget, aa_sigma_px, radius_cap):
    means = model.means
    cov = model.cov3d()
    op = model.opacities().squeeze(-1)
    col = model.colors(torch.as_tensor(cam.center, device=means.device, dtype=means.dtype))
    img, _ = rasterize(means, cov, op, col, cam.w2c_r, cam.w2c_t,
                       cam.fx, cam.fy, cam.cx, cam.cy, cam.W, cam.H,
                       background=background, tile_size=tile_size,
                       pair_budget=pair_budget, aa_sigma_px=aa_sigma_px,
                       radius_cap=radius_cap)
    return img.clamp(0, 1)


def scene_up(scene):
    """The scene's up axis: the direction the training cameras orbit around.

    NeRF-synthetic is authored z-up, but the capture cameras are what actually
    define the axis — they circle the object in the plane perpendicular to it,
    so the least-spread direction of their positions is up.
    """
    centers = np.stack([c.center for c in scene.train])
    mean = centers.mean(axis=0)
    axis = np.linalg.svd(centers - mean)[2][-1]
    axis = axis / np.linalg.norm(axis)
    if np.dot(axis, mean - scene.center) < 0:
        axis = -axis
    return axis


def turntable_cameras(scene, n_frames=120, radius_scale=1.0, height_frac=0.35, up=None):
    """Orbit the camera around the scene centre, perpendicular to its up axis."""
    up = scene_up(scene) if up is None else np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)

    centers = np.stack([c.center for c in scene.train])
    radius = np.linalg.norm(centers - scene.center, axis=1).mean() * radius_scale
    cam0 = scene.train[0]

    # An orthonormal pair spanning the orbit plane.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, up)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(up, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    cams = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        eye = (scene.center
               + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
               + height_frac * radius * up)
        # look-at construction; the scene's own up keeps the object upright
        fwd = scene.center - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        R = np.stack([right, true_up, -fwd], axis=1)     # columns
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = eye
        cams.append(Camera(c2w, cam0.fx, cam0.fy, cam0.cx, cam0.cy, cam0.W, cam0.H))
    return cams


def save_image(img, path):
    arr = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True, help="directory holding point_cloud.ply")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--tile_size", type=int, default=8)
    p.add_argument("--pair_budget", type=int, default=131072)
    p.add_argument("--aa_sigma_px", type=float, default=0.3)
    p.add_argument("--radius_cap", type=float, default=64.0)
    p.add_argument("--turntable_frames", type=int, default=120)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    scene = Scene(args.data_root, downscale=args.downscale).to(device)

    model = load_model(scene, device)
    model = load_ply(model, os.path.join(args.ckpt, "point_cloud.ply"))
    print(f"loaded {model._xyz.shape[0]} gaussians from {args.ckpt}", flush=True)

    report = {"n_gaussians": int(model._xyz.shape[0]), "splits": {}}
    with torch.no_grad():
        for split_name, cams in (("val", scene.val), ("test", scene.test)):
            ps, ss = [], []
            for i, cam in enumerate(cams):
                img = render_camera(model, cam, scene.background, args.tile_size,
                                    args.pair_budget, args.aa_sigma_px, args.radius_cap)
                ps.append(psnr(img, cam.image))
                ss.append(ssim(img.unsqueeze(0), cam.image.unsqueeze(0)))
                if i < 8:
                    save_image(img, os.path.join(args.out, f"{split_name}_{i:03d}.png"))
                    save_image(cam.image, os.path.join(args.out, f"{split_name}_{i:03d}_gt.png"))
            report["splits"][split_name] = {
                "psnr": sum(ps) / len(ps), "ssim": sum(ss) / len(ss), "n_views": len(ps)}
            print(f"{split_name}: psnr={report['splits'][split_name]['psnr']:.2f} "
                  f"ssim={report['splits'][split_name]['ssim']:.4f}", flush=True)

        # turntable
        tdir = os.path.join(args.out, "turntable")
        os.makedirs(tdir, exist_ok=True)
        cams = turntable_cameras(scene, args.turntable_frames)
        for i, cam in enumerate(cams):
            cam.to(device)
            img = render_camera(model, cam, scene.background, args.tile_size,
                                args.pair_budget, args.aa_sigma_px, args.radius_cap)
            save_image(img, os.path.join(tdir, f"frame_{i:04d}.png"))
    print(f"wrote turntable frames to {tdir}", flush=True)

    with open(os.path.join(args.out, "render_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # try to encode a video if ffmpeg is available
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        mp4 = os.path.join(args.out, "turntable.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "30", "-i",
            os.path.join(tdir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4],
            check=False, capture_output=True)
        print(f"video: {mp4}", flush=True)


if __name__ == "__main__":
    main()
