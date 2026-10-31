"""Throughput benchmark of the vectorised rasteriser at training scale."""
import sys
import time

import torch

sys.path.insert(0, "/home/whu/splatforge")
from src.rasterizer import rasterize  # noqa: E402


def bench(N, W, H, tile_size, pair_budget, iters=20, backward=True):
    dev = "cuda"
    torch.manual_seed(0)
    means = ((torch.rand(N, 3, device=dev) - 0.5) * 3.0).requires_grad_(True)
    scales = torch.rand(N, 3, device=dev) * 0.05 + 0.005
    quats = torch.randn(N, 4, device=dev)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    from src.gaussian_model import quat_to_rotmat
    R = quat_to_rotmat(quats)
    M = R * scales[:, None, :]
    cov3d = M @ M.transpose(1, 2)
    op = (torch.rand(N, device=dev) * 0.9 + 0.05).requires_grad_(True)
    col = torch.rand(N, 3, device=dev)

    w2c_r = torch.eye(3, device=dev)
    w2c_t = torch.tensor([0.0, 0.0, -4.0], device=dev)
    fx = fy = 0.5 * W / torch.tan(torch.tensor(0.6911 / 2)).item()
    cx, cy = W / 2.0, H / 2.0

    # warmup
    for _ in range(3):
        img, aux = rasterize(means, cov3d, op, col, w2c_r, w2c_t, fx, fy, cx, cy,
                             W, H, tile_size=tile_size, pair_budget=pair_budget,
                             aa_sigma_px=0.3)
        if backward:
            img.mean().backward()
            means.grad = None
            op.grad = None
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        img, aux = rasterize(means, cov3d, op, col, w2c_r, w2c_t, fx, fy, cx, cy,
                             W, H, tile_size=tile_size, pair_budget=pair_budget,
                             aa_sigma_px=0.3)
        if backward:
            img.mean().backward()
            means.grad = None
            op.grad = None
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    print(f"N={N} {W}x{H} tile={tile_size} budget={pair_budget} "
          f"pairs={aux['num_pairs']} tiles={aux['n_tiles']} "
          f"{'fwd+bwd' if backward else 'fwd'}: {dt*1000:.1f} ms/iter "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    for N in (50000, 200000, 400000):
        torch.cuda.reset_peak_memory_stats()
        bench(N, 400, 400, 8, 131072)
    print("--- tile size / budget sweep at N=200k ---")
    for ts in (8, 16):
        for pb in (65536, 131072, 262144):
            torch.cuda.reset_peak_memory_stats()
            bench(200000, 400, 400, ts, pb, iters=10)
