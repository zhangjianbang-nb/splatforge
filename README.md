# SplatForge — 3D Gaussian Splatting from scratch, in pure PyTorch

A from-scratch implementation of **3D Gaussian Splatting** (Kerbl et al.,
SIGGRAPH 2023) for novel-view synthesis, with one substantive algorithmic
redesign: the differentiable rasteriser is written entirely in PyTorch, with
**no CUDA kernel and no Python loop over tiles**.

See [`REPORT.md`](REPORT.md) for results, the engineering log, and the
verification details.

## The problem

3DGS represents a scene as anisotropic 3D Gaussians and renders by projecting
each one to a screen-space ellipse, sorting per tile by depth, and
alpha-compositing front-to-back:

```
T_i = prod_{j<i, same tile} (1 - alpha_j)
C   = sum_i T_i * alpha_i * c_i
```

`T_i` depends on every earlier splat in the same tile, so the reference
implementation rasterises with one CUDA thread per pixel walking its tile's
sorted list. The sequential scan is trivial in CUDA but awkward in PyTorch,
where a per-tile Python loop (2500 tiles at 400×400) would dominate runtime.

## The design: log-space segmented cumulative product

Once (Gaussian, tile) pairs are sorted by tile id and then by depth, the
per-tile prefix product is a **segmented cumulative product** along one axis:

```
cs   = cumsum(log1p(-alpha))                  # inclusive prefix, log space
T_i  = exp(cs[i] - log_a[i] - prefix_of_tile_i)
```

One `cumsum`, one gather and one `exp` over a whole chunk of pairs — all batched
tensor ops that autograd differentiates. The rasteriser becomes a short loop
over chunks of the sorted pair list.

Three things make it work:

1. **Log space is mandatory.** `cumprod(a) / cumprod_at_start` divides by a
   product that underflows to zero for long tile lists and yields NaNs.
   `log1p(-alpha)` clamped at −23 keeps it finite.
2. **Layout.** Per-(pair, pixel) tensors are `[k, P]` — pixels outer, pairs on
   the contiguous inner axis — so the scan is contiguous. The first version used
   `[P, k]` and ran **6.4× slower** (568 → 88.5 ms/iter).
3. **Chunk boundaries.** A tile's splat list can straddle two chunks, so a
   per-tile running log-transmittance is threaded through the loop with
   `index_copy`, which stays differentiable.

Memory is handled by **gradient checkpointing per chunk** plus **per-tile
aggregation inside the checkpointed function**, so everything autograd retains
is O(tiles) rather than O(pairs). Measured at 780 k pairs: 3.43 GB → 0.78 GB.

## Screen-space anti-aliasing

Splats that shrink below a pixel alias and pop. The 2D covariance is dilated by
`σ_px² I` with opacity compensation `opacity · sqrt(det Σ / det Σ')`, preserving
the splat integral — the 2D analogue of Mip-Splatting's 3D smoothing filter.
Controlled by `--aa_sigma_px` (default 0.3).

## Initialisation by space carving

NeRF-synthetic has no SfM points. Random seeding gives an initial PSNR of 11.6
and stalls. Instead, the dataset's alpha masks are used to carve a **visual
hull** by voxel space carving: a voxel survives only if it projects onto
foreground in every view that sees it; boundary voxels become surface points
with colours sampled from the views that see them.

Two traps: voxels outside every frustum are never constrained (fixed by
requiring a minimum number of observing views), and the scene centre is *not*
the mean camera position for surround-view data (fixed by least-squares
intersection of the viewing rays — for `lego` the cameras are 4.031 units from
the origin while their mean sits at z ≈ 2.28).

Result: initial PSNR 18.6, and val PSNR 24.55 after 600 iterations with
densification disabled.

## Layout

```
src/rasterizer.py        vectorised tile rasteriser (core algorithm)
src/gaussian_model.py    parameters, activations, SH, density control
src/dataset.py           Blender/NeRF-synthetic loader, space carving, seeding
src/train.py             training loop
src/render.py            novel-view rendering, PLY export, turntable
src/metrics.py           PSNR / SSIM
web/serve.py             local server: static viewer + inference endpoint
web/index.html           viewer UI
web/splat-viewer.js      WebGL2 Gaussian rasteriser + 3DGS PLY parser
web/sort-worker.js       depth sort (LSD radix, off the main thread)
tests/test_rasterizer.py agreement check vs. brute-force reference
tests/diag_grads.py      gradient-distribution diagnostic
tests/bench.py           throughput benchmark
tests/bench_mem.py       checkpointing memory check
scripts/download_dataset.py  dataset fetcher (hf-mirror.com)
scripts/run_all.sh       download -> train -> render
```

## Usage

```bash
bash scripts/run_all.sh lego 15000 2      # scene, iterations, downscale
```

or step by step:

```bash
.venv/bin/python scripts/download_dataset.py lego
.venv/bin/python -m src.train --data_root data/nerf_synthetic/lego \
    --out outputs/lego --device cuda:0 --downscale 2 --iters 15000
.venv/bin/python -m src.render --data_root data/nerf_synthetic/lego \
    --ckpt outputs/lego --out outputs/lego/renders --device cuda:0
```

Everything is pinned to **GPU 1** (`CUDA_VISIBLE_DEVICES=1`); GPU 0 hosts the
Bonsai llama-server and is never used.

## Web viewer — SplatForge Studio

`web/` is a browser front end that renders a trained checkpoint live: a WebGL2
rasteriser for the Gaussians plus a panel that runs real PyTorch inference on
the same model, so the browser render and the reference render can be compared
side by side.

```bash
.venv/bin/python web/serve.py --port 8770     # then open http://127.0.0.1:8770/
```

The viewer reads `outputs/<scene>/point_cloud.ply` **directly** — there is no
export step and no second copy of the weights. The raw parameters are activated
in the shader (`exp` for scale, `sigmoid` for opacity) and the 16 SH
coefficients drive view-dependent colour, so orbiting the object reproduces the
model's own appearance changes.

How it renders, and why:

- **Screen-space ellipses.** Each Gaussian's 3D covariance is projected through
  the camera Jacobian to a 2D covariance, eigen-decomposed, and drawn as an
  instanced quad at ±3σ. The fragment shader evaluates the Gaussian falloff and
  discards below 1/255.
- **Back-to-front compositing.** Splats are depth-sorted and blended with
  premultiplied `ONE, ONE_MINUS_SRC_ALPHA`. A 32-bit LSD radix sort over
  monotonic float keys runs in a worker (`web/sort-worker.js`) — a comparison
  sort of 200k+ elements would stutter the drag, and this keeps even the linear
  sort off the render loop.
- **Indirection, not re-upload.** Per-splat data lives in float textures indexed
  by `gl_InstanceID`; only a small integer texture holds the sorted permutation,
  so re-sorting costs 4 bytes per splat instead of the whole cloud.
- **Adaptive tiling.** Splats are packed into texture rows sized to the device's
  `MAX_TEXTURE_SIZE`, which keeps the viewer working on software rasterisers
  (SwiftShader caps textures at 8192) as well as real GPUs.
- **Scene up axis.** NeRF-synthetic is authored z-up while an orbit camera is
  y-up, so the cloud is rotated by a uniform matrix built from the training
  cameras' orbit axis. Without it the object renders lying on its side and the
  orbit swings over its poles.

Front-end controls: scene picker, background, splat scale, SH degree (3/1/0),
and depth-sort policy (continuous / on release / off). A 3DGS `.ply` can also be
dragged straight onto the canvas.

Endpoints:

| Route | Purpose |
|---|---|
| `GET /api/scenes` | trained checkpoints with framing, up axis, and metrics |
| `POST /api/infer` | forward pass on GPU 1 → `{"render", "groundTruth", "psnr", ...}` |
| `GET /ply/<scene>` | streams the checkpoint PLY |
| `GET /media/<scene>/<file>` | renders directory (turntable video) |

Inference is pinned to GPU 1 via `CUDA_VISIBLE_DEVICES`, and the model is cached
in-process after the first call (~12 s load, ~150 ms per subsequent view).

## Verification

`tests/test_rasterizer.py` renders a random scene with both the vectorised
rasteriser and a brute-force per-pixel reference (which enumerates every
Gaussian whose tile the pixel belongs to, sorts front-to-back, and composites in
a Python loop). They agree to float32 round-off — max absolute difference
8.9e-07 without anti-aliasing, 9.9e-07 with it. The same test checks that
gradients w.r.t. means and opacities are finite.
