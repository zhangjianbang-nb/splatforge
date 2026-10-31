"""Check that gradient checkpointing actually bounds peak memory."""
import sys
import torch

sys.path.insert(0, "/home/whu/splatforge")
from src.gaussian_model import quat_to_rotmat  # noqa: E402
from src.rasterizer import rasterize  # noqa: E402


def run(N, W, H, pair_budget, checkpoint_chunks, scale=0.012, depth=4.0):
    dev = "cuda"
    torch.manual_seed(0)
    means = ((torch.rand(N, 3, device=dev) - 0.5) * 1.5).requires_grad_(True)
    sc = torch.full((N, 3), scale, device=dev)
    quats = torch.randn(N, 4, device=dev)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    R = quat_to_rotmat(quats)
    M = R * sc[:, None, :]
    cov3d = M @ M.transpose(1, 2)
    op = (torch.rand(N, device=dev) * 0.9 + 0.05).requires_grad_(True)
    col = torch.rand(N, 3, device=dev)

    w2c_r = torch.eye(3, device=dev)
    w2c_t = torch.tensor([0.0, 0.0, -depth], device=dev)
    fx = fy = 0.5 * W / torch.tan(torch.tensor(0.6911 / 2)).item()
    cx, cy = W / 2.0, H / 2.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    img, aux = rasterize(means, cov3d, op, col, w2c_r, w2c_t, fx, fy, cx, cy,
                         W, H, tile_size=8, pair_budget=pair_budget,
                         aa_sigma_px=0.3, checkpoint_chunks=checkpoint_chunks)
    img.mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del img, means, op
    return aux["num_pairs"], peak


if __name__ == "__main__":
    N = 120000
    for pb in (32768, 131072):
        for ck in (True, False):
            pairs, peak = run(N, 400, 400, pb, ck)
            print(f"N={N} budget={pb} checkpoint={ck}: pairs={pairs} "
                  f"peak={peak:.2f} GB", flush=True)
