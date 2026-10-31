"""Correctness check for the vectorised rasteriser against a naive reference.

The reference rasteriser computes exactly the same quantity, but by brute
force: for every pixel it gathers every Gaussian whose 3-sigma footprint covers
it, sorts them front-to-back by depth, and alpha-composites in a Python loop.
That is O(H*W*N) and far too slow for training, but it is simple enough to
trust, so agreement between the two is strong evidence the fast path is right.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/whu/splatforge")

from src.rasterizer import rasterize, project_gaussians, COV_REG  # noqa: E402


def quat_to_rotmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def make_cov(scales, quats):
    R = quat_to_rotmat(quats)
    M = R * scales[:, None, :]
    return M @ M.transpose(1, 2)


def naive_reference(means, cov3d, opacities, colors, w2c_r, w2c_t,
                    fx, fy, cx, cy, W, H, background, aa_sigma_px=0.0,
                    tile_size=8):
    """Brute-force per-pixel front-to-back compositing (float64, no chunks)."""
    m2, conic, depth, oscale, x0, x1, y0, y1, radius, visible = project_gaussians(
        means, cov3d, w2c_r, w2c_t, fx, fy, cx, cy, W, H,
        aa_sigma_px=aa_sigma_px, radius_cap=64.0)

    m2 = m2.double().cpu().numpy()
    conic = conic.double().cpu().numpy()
    depth = depth.double().cpu().numpy()
    op = (opacities * oscale).double().cpu().numpy()
    col = colors.double().cpu().numpy()
    vis = visible.cpu().numpy()

    ts = tile_size
    tx0n = (x0.cpu().numpy().astype(np.int64) // ts)
    tx1n = (x1.cpu().numpy().astype(np.int64) // ts)
    ty0n = (y0.cpu().numpy().astype(np.int64) // ts)
    ty1n = (y1.cpu().numpy().astype(np.int64) // ts)

    img = np.zeros((H, W, 3), dtype=np.float64)
    for py in range(H):
        for px in range(W):
            cx_p, cy_p = px + 0.5, py + 0.5
            # Same pair set as the fast path.  A Gaussian contributes to every
            # pixel of every tile its bounding box touches -- not only to pixels
            # inside the box -- which is what the reference CUDA kernel does.
            tpx, tpy = px // ts, py // ts
            idx = np.nonzero(vis & (tx0n <= tpx) & (tx1n >= tpx)
                             & (ty0n <= tpy) & (ty1n >= tpy))[0]
            if idx.size == 0:
                img[py, px] = background
                continue
            order = idx[np.argsort(depth[idx], kind="stable")]
            T = 1.0
            acc = np.zeros(3)
            for i in order:
                a_i, b_i, c_i = conic[i, 0], conic[i, 1], conic[i, 2]
                ddx = cx_p - m2[i, 0]
                ddy = cy_p - m2[i, 1]
                pw = a_i * ddx * ddx + c_i * ddy * ddy + 2 * b_i * ddx * ddy
                al = float(np.clip(op[i] * np.exp(-0.5 * pw), 0.0, 0.9999))
                acc += T * al * col[i]
                T *= (1.0 - al)
            img[py, px] = acc + T * np.asarray(background, dtype=np.float64)
    return img


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N, W, H = 60, 48, 40
    means = (torch.rand(N, 3, device=device) - 0.5) * 1.2
    scales = torch.rand(N, 3, device=device) * 0.08 + 0.02
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    cov3d = make_cov(scales, quats)
    opacities = torch.rand(N, device=device) * 0.9 + 0.05
    colors = torch.rand(N, 3, device=device)

    # camera at +z looking back toward -z
    w2c_r = torch.eye(3, device=device)
    w2c_t = torch.tensor([0.0, 0.0, -3.0], device=device)
    fx = fy = 60.0
    cx, cy = W / 2.0, H / 2.0
    bg = (1.0, 1.0, 1.0)

    for aa in (0.0, 0.3):
        fast, aux = rasterize(means, cov3d, opacities, colors, w2c_r, w2c_t,
                              fx, fy, cx, cy, W, H, background=bg,
                              tile_size=8, pair_budget=4096, aa_sigma_px=aa)
        ref = naive_reference(means, cov3d, opacities, colors, w2c_r, w2c_t,
                              fx, fy, cx, cy, W, H, bg, aa_sigma_px=aa,
                              tile_size=8)
        fast_np = fast.double().cpu().numpy().transpose(1, 2, 0)
        diff = np.abs(fast_np - ref)
        denom = max(1e-8, float(np.abs(ref).mean()))
        print(f"aa_sigma_px={aa}: max_abs_diff={diff.max():.3e} "
              f"mean_abs_diff={diff.mean():.3e} rel={diff.mean()/denom:.3e} "
              f"pairs={aux['num_pairs']}")
        # remaining gap is float32 (fast path) vs float64 (reference) accumulation
        assert diff.max() < 1e-4, f"mismatch too large for aa={aa}"
        assert diff.mean() / denom < 1e-5, f"mean mismatch too large for aa={aa}"

    # gradient sanity: a rendered pixel must be differentiable w.r.t. opacity
    means_g = means.clone().requires_grad_(True)
    op_g = opacities.clone().requires_grad_(True)
    cov_g = make_cov(scales, quats)
    img, _ = rasterize(means_g, cov_g, op_g, colors, w2c_r, w2c_t,
                       fx, fy, cx, cy, W, H, background=bg,
                       tile_size=8, pair_budget=4096, aa_sigma_px=0.0)
    img.mean().backward()
    print("grad means finite:", bool(torch.isfinite(means_g.grad).all()),
          "grad opacity finite:", bool(torch.isfinite(op_g.grad).all()),
          "opacity grad norm:", float(op_g.grad.norm()))
    assert torch.isfinite(means_g.grad).all() and torch.isfinite(op_g.grad).all()

    print("OK: vectorised rasteriser matches the naive reference")


if __name__ == "__main__":
    main()
