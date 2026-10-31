"""Vectorised, tile-based, differentiable 3D Gaussian Splatting rasteriser (pure PyTorch).

Why a new rasteriser
--------------------
The reference 3DGS implementation rasterises with a hand-written CUDA kernel:
one thread per pixel, walking the per-tile Gaussian list front-to-back and
accumulating alpha-composited colour.  Reproducing that in pure PyTorch is
non-trivial, because a per-tile Python loop (2500 tiles per iteration at
400x400) would dominate the runtime.

Key idea
--------
Once the (Gaussian, tile) pairs are sorted by tile id and then by depth, alpha
compositing *is* a segmented cumulative product along the pair axis:

    T_i = prod_{j < i, same tile} (1 - alpha_j)
    colour += T_i * alpha_i * c_i

A segmented cumprod is expressible with two fully-vectorised primitives:
a global ``cumsum`` in log space, plus subtraction of the per-segment prefix
value.  The whole rasteriser therefore reduces to a short loop over chunks of
the sorted pair list, where every operation is a batched tensor op that
autograd can differentiate.

Two implementation details matter for speed and memory:

* **Layout.** Per-(pair, pixel) tensors are stored as ``[k, P]`` (pixels on the
  outer axis, pairs on the contiguous inner axis) rather than ``[P, k]``.  The
  scan runs along the contiguous axis, which is many times faster than a
  strided scan over 131072 rows.
* **Lazy geometry.** The per-pair/per-pixel offsets, squared distances and
  pixel indices are rebuilt inside each chunk instead of being materialised
  once for the whole pair list, which keeps peak memory to a few hundred MB.

Log space matters: the naive ``cumprod(a) / cumprod_at_segment_start`` form
divides by a product that underflows to zero for long tile lists, producing
NaNs.  Working with ``log1p(-alpha)`` and ``exp(cumsum - prefix)`` is stable.

Segments (tiles) can straddle chunk boundaries, so a running per-tile log
transmittance ("carry") is threaded through the chunk loop with
``index_copy``, which keeps the graph differentiable.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

DEPTH_EPS = 0.2            # near plane for view-space culling
COV_REG = 0.3              # diagonal regulariser added to the 2D covariance
LOG_ALPHA_FLOOR = -23.0    # floor on log(1 - alpha); keeps exp() finite


# --------------------------------------------------------------------------- #
#  Projection: 3D Gaussians -> 2D screen-space conics
# --------------------------------------------------------------------------- #
def project_gaussians(
    means, cov3d, w2c_r, w2c_t, fx, fy, cx, cy, W, H,
    aa_sigma_px=0.0, radius_cap=64.0,
):
    """EWA-splat a batch of 3D Gaussians into screen space.

    Returns mean2d, conic (the inverse 2x2 screen covariance laid out as
    ``[a, b, c]`` meaning ``[[a, b], [b, c]]``), depth, an opacity compensation
    factor, the clipped integer bounding box, the 3-sigma radius, and a
    visibility mask.
    """
    view = means @ w2c_r.t() + w2c_t              # world -> camera (OpenGL: -z forward)
    depth = -view[:, 2]

    # Sigma_cam = R Sigma_world R^T
    cov_cam = torch.einsum("ij,njk,lk->nil", w2c_r, cov3d, w2c_r)

    inv_d = 1.0 / depth.clamp_min(1e-6)
    inv_d2 = inv_d * inv_d
    z = torch.zeros_like(inv_d)
    # Jacobian of the perspective projection (with y flipped for image coords)
    row0 = torch.stack([fx * inv_d, z, fx * view[:, 0] * inv_d2], dim=-1)
    row1 = torch.stack([z, -fy * inv_d, -fy * view[:, 1] * inv_d2], dim=-1)
    jac = torch.stack([row0, row1], dim=1)        # [N, 2, 3]

    cov2d = torch.einsum("nij,njk,nlk->nil", jac, cov_cam, jac)

    a = cov2d[:, 0, 0] + COV_REG
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + COV_REG
    det = (a * c - b * b).clamp_min(1e-9)

    # Mip-style screen-space anti-aliasing: dilate the 2D Gaussian by
    # sigma_px pixels and compensate opacity so the splat integral is preserved.
    #   opacity' = opacity * sqrt(det(Sigma) / det(Sigma + sigma^2 I))
    opacity_scale = torch.ones_like(det)
    if aa_sigma_px > 0.0:
        s2 = aa_sigma_px * aa_sigma_px
        det_s = (a + s2) * (c + s2) - b * b
        opacity_scale = torch.sqrt(det / det_s.clamp_min(1e-9))
        a = a + s2
        c = c + s2
        det = det_s.clamp_min(1e-9)

    conic = torch.stack([c / det, -b / det, a / det], dim=-1)   # Sigma_2d^{-1}

    mean2d_x = cx + fx * view[:, 0] * inv_d
    mean2d_y = cy - fy * view[:, 1] * inv_d

    # Largest eigenvalue of Sigma_2d -> 3-sigma splat radius in pixels
    mid = 0.5 * (a + c)
    tmp = (mid * mid - det).clamp_min(0.1).sqrt()
    radius = (3.0 * (mid + tmp).sqrt()).clamp(max=radius_cap)

    x0 = torch.floor(mean2d_x - radius)
    x1 = torch.ceil(mean2d_x + radius)
    y0 = torch.floor(mean2d_y - radius)
    y1 = torch.ceil(mean2d_y + radius)
    visible = ((depth > DEPTH_EPS)
               & (x1 >= 0) & (x0 <= W - 1)
               & (y1 >= 0) & (y0 <= H - 1))
    return (torch.stack([mean2d_x, mean2d_y], dim=-1), conic, depth,
            opacity_scale, x0, x1, y0, y1, radius, visible)


# --------------------------------------------------------------------------- #
#  Pair construction: which (Gaussian, tile) pairs exist
# --------------------------------------------------------------------------- #
def build_pairs(x0, x1, y0, y1, visible, n_tx, tile_size, W, H,
                cell_budget=4_000_000):
    """Expand the visible Gaussians into a flat (gaussian_index, tile_id) list.

    Screen-space bounding boxes are clipped to the image, so every tile id
    lands in ``[0, n_tx * n_ty)``.  The masked-grid expansion is chunked over
    Gaussians so the temporary buffer stays bounded even when a few splats are
    very large.
    """
    vis = torch.nonzero(visible).flatten()
    if vis.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=x0.device)
        return empty, empty

    tx0 = (x0[vis].clamp(0, W - 1) / tile_size).long()
    tx1 = (x1[vis].clamp(0, W - 1) / tile_size).long()
    ty0 = (y0[vis].clamp(0, H - 1) / tile_size).long()
    ty1 = (y1[vis].clamp(0, H - 1) / tile_size).long()
    nx = tx1 - tx0 + 1
    ny = ty1 - ty0 + 1

    max_nx, max_ny = int(nx.max()), int(ny.max())
    cells = max(1, max_nx * max_ny)
    chunk = max(2048, min(65536, cell_budget // cells))

    offs_y = torch.arange(max_ny, device=x0.device)
    offs_x = torch.arange(max_nx, device=x0.device)

    tids_parts, gidx_parts = [], []
    for s in range(0, vis.numel(), chunk):
        sub = vis[s:s + chunk]
        nxs, nys = nx[s:s + chunk], ny[s:s + chunk]
        tx0s, ty0s = tx0[s:s + chunk], ty0[s:s + chunk]
        grid_y = ty0s[:, None] + offs_y[None, :]
        grid_x = tx0s[:, None] + offs_x[None, :]
        mask = ((offs_y[None, :] < nys[:, None])[:, :, None]
                & (offs_x[None, :] < nxs[:, None])[:, None, :])
        tids_parts.append((grid_y[:, :, None] * n_tx + grid_x[:, None, :])[mask])
        gidx_parts.append(torch.repeat_interleave(sub, nxs * nys))

    return torch.cat(tids_parts), torch.cat(gidx_parts)


# --------------------------------------------------------------------------- #
#  One chunk of the compositing loop
# --------------------------------------------------------------------------- #
def _chunk_composite(ca, cb, cc, seg_op, seg_col, m2x, m2y, tx_off, ty_off,
                     seg_tids, lx, ly, carry_log):
    """Alpha-composite one chunk of the sorted pair list.

    Everything here is batched over ``[k, p]`` (pixels on the outer axis, pairs
    on the contiguous inner axis) so the log-space scan runs along a contiguous
    dimension.  ``carry_log`` is the per-tile, per-pixel log transmittance
    entering this chunk; the returned value is the updated one.

    This is a module-level function so the caller can wrap it in
    ``torch.utils.checkpoint``: without that, autograd would retain every
    intermediate of every chunk simultaneously, and a dense scene (millions of
    pairs) exhausts GPU memory during backward.
    """
    dx = m2x - (tx_off + lx[:, None] + 0.5)             # [k, p]
    dy = m2y - (ty_off + ly[:, None] + 0.5)

    power = (ca[None, :] * dx * dx + cc[None, :] * dy * dy
             + 2.0 * cb[None, :] * dx * dy)
    alpha = (seg_op[None, :] * torch.exp(-0.5 * power)).clamp(0.0, 0.9999)

    log_a = torch.log1p(-alpha).clamp_min(LOG_ALPHA_FLOOR)
    cs = torch.cumsum(log_a, dim=1)                      # inclusive prefix

    # tile boundaries inside this chunk (pairs are sorted by tile id)
    new_seg = torch.zeros(seg_tids.shape[0], dtype=torch.bool, device=seg_tids.device)
    new_seg[0] = True
    new_seg[1:] = seg_tids[1:] != seg_tids[:-1]
    starts = torch.nonzero(new_seg).flatten()
    ends = torch.cat([starts[1:] - 1,
                      torch.tensor([seg_tids.shape[0] - 1], device=seg_tids.device)])
    seg_tile = seg_tids[starts]

    # prefix transmittance: for a tile whose splat list began in an earlier
    # chunk take the carried value, otherwise take the value from this chunk
    pre = torch.empty(lx.shape[0], starts.shape[0], device=cs.device, dtype=cs.dtype)
    pre[:, 0] = carry_log[seg_tile[0]]
    if starts.shape[0] > 1:
        pre[:, 1:] = cs[:, starts[1:] - 1]

    seg_id = torch.repeat_interleave(
        torch.arange(starts.shape[0], device=cs.device), ends - starts + 1)

    t_excl = torch.exp(cs - log_a - pre[:, seg_id])      # transmittance before i
    contrib = (t_excl * alpha)[:, :, None] * seg_col[None, :, :]   # [k, p, 3]

    # Sum each tile's splats here, inside the checkpointed function, rather than
    # scattering per-pair contributions into the image.  Scattering directly
    # would make autograd retain one index tensor and one contribution view per
    # chunk -- both O(pairs) -- which is what exhausted memory on dense scenes.
    # Collapsing to per-tile sums first makes everything retained O(tiles).
    tile_contrib = torch.zeros(starts.shape[0], contrib.shape[0], 3,
                               device=contrib.device, dtype=contrib.dtype)
    tile_contrib = tile_contrib.index_add(0, seg_id, contrib.permute(1, 0, 2))

    new_carry = carry_log.index_copy(0, seg_tile, (cs[:, ends] - pre).t())
    return tile_contrib, new_carry, seg_tile


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def rasterize(
    means, cov3d, opacities, colors, w2c_r, w2c_t,
    fx, fy, cx, cy, W, H,
    background=(1.0, 1.0, 1.0),
    tile_size: int = 8,
    pair_budget: int = 131072,
    aa_sigma_px: float = 0.0,
    radius_cap: float = 64.0,
    checkpoint_chunks: bool = True,
):
    """Render one image.  Returns ``(image [3,H,W], aux dict)``."""
    device, dtype = means.device, means.dtype
    n_tx = (W + tile_size - 1) // tile_size
    n_ty = (H + tile_size - 1) // tile_size
    n_tiles = n_tx * n_ty
    W_pad, H_pad = n_tx * tile_size, n_ty * tile_size
    k = tile_size * tile_size

    (mean2d, conic, depth, opacity_scale, x0, x1, y0, y1, radius, visible) = \
        project_gaussians(means, cov3d, w2c_r, w2c_t, fx, fy, cx, cy, W, H,
                          aa_sigma_px=aa_sigma_px, radius_cap=radius_cap)

    tids, gidx = build_pairs(x0, x1, y0, y1, visible, n_tx, tile_size, W, H)
    if tids.numel() == 0:
        img = torch.empty(3, H, W, device=device, dtype=dtype)
        img[:] = torch.tensor(background, device=device, dtype=dtype).view(3, 1, 1)
        return img, {"visible": visible, "radii": radius, "num_pairs": 0,
                     "n_tiles": n_tiles}

    # --- sort pairs: primary tile id, secondary depth (front to back) ---
    depth_pair = depth[gidx].detach()
    o1 = torch.argsort(depth_pair, stable=True)
    tids, gidx = tids[o1], gidx[o1]
    o2 = torch.argsort(tids, stable=True)
    tids, gidx = tids[o2], gidx[o2]

    # --- per-pair data (small, kept for the whole call) ---
    m2 = mean2d[gidx]                                  # [P, 2]
    cn = conic[gidx]                                   # [P, 3]
    op = opacities[gidx] * opacity_scale[gidx]         # [P]
    col = colors[gidx]                                 # [P, 3]

    tx = tids % n_tx
    ty = tids // n_tx
    base_pix = (ty * tile_size) * W_pad + (tx * tile_size)      # [P]

    # pixel geometry inside a tile: [k]
    loc = torch.arange(k, device=device)
    lx = (loc % tile_size).to(dtype)
    ly = (loc // tile_size).to(dtype)
    offs_pix = (loc // tile_size) * W_pad + (loc % tile_size)   # [k]
    tx_off = (tx * tile_size).to(dtype)                         # [P]
    ty_off = (ty * tile_size).to(dtype)

    bg = torch.tensor(background, device=device, dtype=dtype)

    # --- chunk boundaries, aligned to tile boundaries where possible ---
    counts = torch.bincount(tids, minlength=n_tiles).tolist()
    bounds, start, acc = [], 0, 0
    for c in counts:
        if acc + c > pair_budget and acc > 0:
            bounds.append((start, start + acc))
            start += acc
            acc = 0
        acc += c
    if acc:
        bounds.append((start, start + acc))

    # --- chunked compositing, accumulated per tile ---
    tile_acc = torch.zeros(n_tiles, k, 3, device=device, dtype=dtype)
    carry_log = torch.zeros(n_tiles, k, device=device, dtype=dtype)

    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled()
    for lo, hi in bounds:
        args = (cn[lo:hi, 0], cn[lo:hi, 1], cn[lo:hi, 2], op[lo:hi], col[lo:hi],
                m2[lo:hi, 0][None, :], m2[lo:hi, 1][None, :],
                tx_off[lo:hi][None, :], ty_off[lo:hi][None, :],
                tids[lo:hi], lx, ly, carry_log)
        if use_checkpoint:
            contrib, carry_log, seg_tile = checkpoint(
                _chunk_composite, *args, use_reentrant=False)
        else:
            contrib, carry_log, seg_tile = _chunk_composite(*args)
        tile_acc = tile_acc.index_add(0, seg_tile, contrib)

    # --- scatter tiles to pixels, then composite the background ---
    tile_ids = torch.arange(n_tiles, device=device)
    ttx, tty = tile_ids % n_tx, tile_ids // n_tx
    tpix = (tty[:, None] * tile_size * W_pad + ttx[:, None] * tile_size
            + offs_pix[None, :])                       # [n_tiles, k]
    t_bg = torch.exp(carry_log).reshape(-1)
    image_flat = torch.zeros(W_pad * H_pad, 3, device=device, dtype=dtype)
    image_flat = image_flat.index_add(0, tpix.reshape(-1), tile_acc.reshape(-1, 3))
    image_flat = image_flat.index_add(
        0, tpix.reshape(-1), t_bg[:, None] * bg[None, :])

    image = image_flat.view(H_pad, W_pad, 3)[:H, :W].permute(2, 0, 1)
    return image, {
        "visible": visible,
        "radii": radius,
        "num_pairs": int(tids.numel()),
        "n_tiles": n_tiles,
    }
