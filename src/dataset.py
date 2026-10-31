"""Scene / camera loading for the Blender-style NeRF-synthetic dataset.

Format
------
Each scene directory contains ``transforms_{train,val,test}.json`` plus a tree
of PNGs.  The JSON gives ``camera_angle_x`` (horizontal field of view, radians)
and a list of frames, each with a 4x4 OpenGL camera-to-world matrix.  There is
no SfM point cloud, so Gaussians are seeded by space-carving a visual hull from
the alpha masks (see ``init_point_cloud_visual_hull``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Camera:
    """Pinhole camera with an OpenGL-style world-to-camera transform."""

    def __init__(self, c2w, fx, fy, cx, cy, W, H, image=None, mask=None, name=""):
        self.c2w = np.asarray(c2w, dtype=np.float32)
        self.R = self.c2w[:3, :3]
        self.t = self.c2w[:3, 3]
        self.w2c_r = torch.from_numpy(self.R.T.copy())
        self.w2c_t = torch.from_numpy((-self.R.T @ self.t).copy())
        self.c2w_r = torch.from_numpy(self.R.copy())
        self.c2w_t = torch.from_numpy(self.t.copy())
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.W, self.H = W, H
        self.image = image          # [3, H, W] float32 in [0, 1], white-composited
        self.mask = mask            # [1, H, W] float32 foreground mask
        self.name = name
        self.center = self.t.copy()

    def to(self, device):
        self.w2c_r = self.w2c_r.to(device)
        self.w2c_t = self.w2c_t.to(device)
        self.c2w_r = self.c2w_r.to(device)
        self.c2w_t = self.c2w_t.to(device)
        if self.image is not None:
            self.image = self.image.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self


def load_image(path, white_background=True):
    """Load an RGBA render: composite over white, and return the alpha mask."""
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    rgb, a = arr[:, :, :3], arr[:, :, 3:4]
    bg = np.array([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
                  dtype=np.float32)
    comp = rgb * a + bg * (1.0 - a)
    return (torch.from_numpy(comp.transpose(2, 0, 1).copy()),
            torch.from_numpy(a.transpose(2, 0, 1).copy()))


def _downsample(img, factor):
    if factor <= 1:
        return img
    h, w = img.shape[1], img.shape[2]
    # "area" averaging is itself the correct anti-aliased box filter
    return F.interpolate(img.unsqueeze(0), size=(h // factor, w // factor),
                         mode="area").squeeze(0)


def _resolve_image_path(root, file_path):
    for cand in (file_path, file_path + ".png", file_path + ".jpg"):
        p = os.path.normpath(os.path.join(root, cand))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no image for {file_path} under {root}")


def load_split(root, split, downscale=2, white_background=True, load_images=True):
    meta = json.load(open(os.path.join(root, f"transforms_{split}.json")))
    cams = []
    for frame in meta["frames"]:
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        path = _resolve_image_path(root, frame["file_path"])
        img, mask = load_image(path, white_background) if load_images \
            else (None, None)
        if img is None:
            w = h = int(round(800 / downscale))
        else:
            img = _downsample(img, downscale)
            # max-pooling keeps thin foreground structures alive when shrinking
            mask = F.max_pool2d(mask.unsqueeze(0),
                                downscale, downscale).squeeze(0)
            h, w = img.shape[1], img.shape[2]
        fx = fy = 0.5 * w / np.tan(meta["camera_angle_x"] / 2.0)
        cams.append(Camera(c2w, fx, fy, w / 2.0, h / 2.0, w, h, img, mask,
                           name=os.path.basename(path)))
    return cams


class Scene:
    def __init__(self, root, downscale=2, white_background=True):
        self.root = root
        self.downscale = downscale
        self.white_background = white_background
        self.train = load_split(root, "train", downscale, white_background)
        self.val = load_split(root, "val", downscale, white_background)
        self.test = load_split(root, "test", downscale, white_background)
        centers = np.stack([c.center for c in self.train])
        # Surround-view cameras do not sit symmetrically around the object, so
        # their mean position is *not* the scene centre.  The cameras do all
        # look at the same point, so solve for the least-squares intersection of
        # their viewing rays instead: minimise sum ||(I - f f^T)(p - c)||^2.
        forwards = np.stack([-c.R[:, 2] for c in self.train])   # OpenGL forward
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for c, f in zip(centers, forwards):
            P = np.eye(3) - np.outer(f, f)
            A += P
            b += P @ c
        self.center = np.linalg.solve(A, b)
        self.mean_depth = float(np.linalg.norm(centers - self.center, axis=1).mean())
        self.extent = float(np.linalg.norm(centers - self.center, axis=1).max() * 1.1)
        self.W, self.H = self.train[0].W, self.train[0].H
        self.background = (1.0, 1.0, 1.0) if white_background else (0.0, 0.0, 0.0)
        # world size of one pixel at the mean viewing distance
        self.pixel_world = self.mean_depth / self.train[0].fx

    def to(self, device):
        for c in self.train + self.val + self.test:
            c.to(device)
        return self


def _project(points, cam):
    """Project world points into a camera.  Returns pixel coords and validity."""
    view = (points - cam.c2w_t) @ cam.c2w_r
    depth = -view[:, 2]
    valid = depth > 1e-3
    inv = 1.0 / depth.clamp_min(1e-6)
    xs = cam.cx + cam.fx * view[:, 0] * inv
    ys = cam.cy - cam.fy * view[:, 1] * inv
    inb = valid & (xs >= 0) & (xs < cam.W) & (ys >= 0) & (ys < cam.H)
    xi = xs.long().clamp(0, cam.W - 1)
    yi = ys.long().clamp(0, cam.H - 1)
    return xi, yi, inb


@torch.no_grad()
def init_point_cloud_visual_hull(scene, resolution=128, side_scale=1.25,
                                 max_points=120000, seed=0, min_seen_frac=0.3):
    """Seed Gaussians on a visual hull obtained by space carving.

    Random or ray-sampled seeding scatters points through the camera frustums,
    so most Gaussians start in empty space and the optimiser spends thousands of
    iterations deleting them.  The dataset's alpha masks give a free silhouette
    for every training view, so we carve a voxel grid instead: a voxel survives
    only if it projects onto foreground in *every* view that sees it.  Keeping
    only voxels on the boundary of the surviving set gives a surface point cloud
    that hugs the real object, plus a colour for each point sampled from the
    views that see it.
    """
    device = scene.train[0].w2c_r.device
    R = int(resolution)
    side = side_scale * scene.mean_depth

    idx = torch.arange(R, device=device)
    gx, gy, gz = torch.meshgrid(idx, idx, idx, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).float() + 0.5
    pts = torch.as_tensor(scene.center, device=device, dtype=torch.float32) \
        + (pts / R - 0.5) * side

    inside = torch.ones(pts.shape[0], dtype=torch.bool, device=device)
    seen_count = torch.zeros(pts.shape[0], dtype=torch.int32, device=device)
    for cam in scene.train:
        xi, yi, inb = _project(pts, cam)
        fg = cam.mask[0][yi, xi] > 0.5
        # outside the image: this view says nothing; inside: must be foreground
        inside &= (~inb) | fg
        seen_count += inb.int()

    # Voxels far outside every camera frustum are never constrained by any view,
    # so carving alone leaves them standing.  Real surface points are inside the
    # image for most of the surrounding cameras, so requiring a minimum number
    # of observing views removes that unconstrained shell.
    min_seen = max(2, int(min_seen_frac * len(scene.train)))
    hull = inside & (seen_count >= min_seen)
    print(f"space carving: {int(hull.sum())} voxels inside the visual hull "
          f"of {hull.numel()} (>= {min_seen} observing views)", flush=True)
    if int(hull.sum()) == 0:
        raise RuntimeError("space carving produced an empty hull")

    # keep only surface voxels: those with at least one non-hull 6-neighbour
    h = hull.view(R, R, R).float().view(1, 1, R, R, R)
    pad = F.pad(h, (1, 1, 1, 1, 1, 1))
    neighbours = (pad[:, :, :-2, 1:-1, 1:-1] + pad[:, :, 2:, 1:-1, 1:-1]
                  + pad[:, :, 1:-1, :-2, 1:-1] + pad[:, :, 1:-1, 2:, 1:-1]
                  + pad[:, :, 1:-1, 1:-1, :-2] + pad[:, :, 1:-1, 1:-1, 2:])
    surf = (h > 0.5) & (neighbours < 6.0)
    surf_idx = torch.nonzero(surf.view(-1)).flatten()
    pts = pts[surf_idx]
    print(f"visual hull surface: {pts.shape[0]} points", flush=True)

    # average each surface point's colour over the views that see it
    col_sum = torch.zeros(pts.shape[0], 3, device=device)
    count = torch.zeros(pts.shape[0], device=device)
    for cam in scene.train:
        xi, yi, inb = _project(pts, cam)
        take = inb & (cam.mask[0][yi, xi] > 0.5)
        col_sum[take] += cam.image[:, yi[take], xi[take]].t()
        count[take] += 1.0
    cols = col_sum / count.clamp_min(1.0)[:, None]

    if pts.shape[0] > max_points:
        g = torch.Generator(device="cpu").manual_seed(seed)
        sel = torch.randperm(pts.shape[0], generator=g)[:max_points]
        pts, cols = pts[sel], cols[sel]

    voxel = side / R
    return pts.cpu(), cols.cpu(), voxel


def init_point_cloud(scene, n_points=20000, seed=0):
    """Fallback seeding: sample points along training rays."""
    rng = np.random.default_rng(seed)
    pts, cols = [], []
    per_cam = max(1, n_points // len(scene.train))
    for cam in scene.train:
        img = cam.image.cpu().numpy()
        px = rng.integers(0, cam.W, size=per_cam)
        py = rng.integers(0, cam.H, size=per_cam)
        t = rng.normal(scene.mean_depth, 0.35 * scene.mean_depth, size=per_cam)
        t = np.clip(t, 0.2 * scene.mean_depth, 2.0 * scene.mean_depth)
        x = (px + 0.5 - cam.cx) / cam.fx * t
        y = (py + 0.5 - cam.cy) / cam.fy * t
        dir_cam = np.stack([x, y, -t], axis=1)          # OpenGL: forward is -z
        pts.append(dir_cam @ cam.R.T + cam.t)
        cols.append(img[:, py, px].T)
    return (torch.from_numpy(np.concatenate(pts, 0).astype(np.float32)),
            torch.from_numpy(np.concatenate(cols, 0).astype(np.float32)))


def knn_mean_distance(points, k=4, sample=8192):
    """Mean distance to the k nearest neighbours."""
    if points.shape[0] > sample:
        idx = torch.randperm(points.shape[0])[:sample]
        points = points[idx]
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    return d.topk(k, largest=False).values.mean().item()
