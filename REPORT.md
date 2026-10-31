# SplatForge — 3D Gaussian Splatting from scratch, trained on this machine

**Deliverable:** a complete 3D novel-view-synthesis system — algorithm design,
implementation, environment, data pipeline, and a trained model — all running
locally on GPU 1 of this machine.

---

## 1. What was built

A from-scratch implementation of **3D Gaussian Splatting** (Kerbl et al.,
SIGGRAPH 2023) whose differentiable rasteriser is written entirely in PyTorch.
The reference implementation rasterises with a hand-written CUDA kernel; this
project replaces it with a **vectorised tile rasteriser built on a log-space
segmented cumulative product**, which is the main algorithmic contribution.

The pipeline covers the whole task: multi-view image loading, camera pose
handling, Gaussian parameterisation, spherical-harmonic view-dependent colour,
adaptive density control, the differentiable renderer, the training loop, and
novel-view rendering with quantitative evaluation.

## 2. Algorithm design

### 2.1 Representation

A scene is a set of anisotropic 3D Gaussians. Each carries a mean `μ ∈ R³`, a
covariance `Σ = R S Sᵀ Rᵀ` (from a log-scale vector and a quaternion), an
opacity, and view-dependent colour as degree-3 spherical harmonics
(16 coefficients × 3 channels). Rendering projects each Gaussian into screen
space and alpha-composites front-to-back.

### 2.2 The rasteriser: segmented cumprod instead of a CUDA kernel

Alpha compositing needs, for every pixel, the transmittance accumulated over all
splats in front of it:

```
T_i = prod_{j < i, same tile} (1 - alpha_j)
C   = sum_i T_i * alpha_i * c_i
```

Because `T_i` depends on every earlier splat in the same tile, the reference
implementation uses one CUDA thread per pixel walking a per-tile sorted list.
In PyTorch a per-tile Python loop (2500 tiles at 400×400) would dominate runtime.

**Key idea:** once (Gaussian, tile) pairs are sorted by tile id and then by
depth, the per-tile prefix product is a *segmented cumulative product* along a
single axis, computable with two fully vectorised primitives:

```
cs      = cumsum(log1p(-alpha))        # inclusive prefix, in log space
T_i     = exp(cs[i] - log_a[i] - prefix_of_tile_i)
```

This turns the whole rasterisation into a short loop over chunks of the sorted
pair list, where every operation is a batched tensor op that autograd can
differentiate.

Three details make it work:

- **Log space is mandatory.** The naive `cumprod(a) / cumprod_at_start` form
  divides by a product that underflows to zero for long tile lists and produces
  NaNs. `log1p(-alpha)` clamped at −23 keeps everything finite.
- **Layout matters for speed.** Per-(pair, pixel) tensors are stored as `[k, P]`
  (pixels outer, pairs on the contiguous inner axis) so the scan runs along a
  contiguous dimension. The first version used `[P, k]` and ran **6.4× slower**
  (568 ms/iter → 88.5 ms/iter).
- **Segments straddle chunk boundaries.** A tile's splat list can span two
  chunks, so a per-tile running log-transmittance is threaded through the loop
  with `index_copy`, which stays differentiable.

### 2.3 Screen-space anti-aliasing

Projected Gaussians can shrink below a pixel, producing aliasing and popping.
The 2D covariance is dilated by `σ_px² I` and opacity compensated so the splat
integral is preserved:

```
Σ' = Σ + σ_px² I        opacity' = opacity · sqrt(det Σ / det Σ')
```

This is the 2D analogue of Mip-Splatting's 3D smoothing filter (Yu et al.,
CVPR 2024). Exposed as `--aa_sigma_px` (default 0.3, 0 disables).

### 2.4 Initialisation by space carving

The NeRF-synthetic dataset has no SfM point cloud, so Gaussians must be seeded
analytically. Random or ray-sampled seeding scatters points through the camera
frustums — most land in empty space, and the optimiser spends thousands of
iterations deleting them. Measured: an initial PSNR of **11.6** with random
seeding and a val PSNR stuck at 10.4 after 300 iterations.

Instead, the dataset's **alpha masks** give a free silhouette per view, so a
visual hull is carved by voxel space carving: a voxel survives only if it
projects onto foreground in every view that sees it. Keeping only boundary
voxels yields a surface point cloud that hugs the object, with a colour sampled
from the views that see each point.

Two subtleties had to be solved:

- **Voxels outside every frustum are never constrained**, so carving alone
  leaves an unconstrained shell standing. Real surface points are inside the
  image for most surrounding cameras, so a minimum observing-view count removes
  that shell.
- **The scene centre is not the mean camera position.** Surround-view cameras
  do not sit symmetrically around the object: for `lego`, all cameras are
  4.031 units from the *origin* while their mean position sits at
  `(-0.39, 0.07, 2.28)`. The centre is instead found by least-squares
  intersection of the viewing rays.

Result: an initial PSNR of **18.6** (vs 11.6), every seeded Gaussian visible,
and val PSNR 24.55 after 600 iterations *with densification disabled*.

## 3. Environment

| item | value |
|---|---|
| Python | 3.12.3, venv at `/home/whu/splatforge/.venv` (managed by `uv`) |
| PyTorch | 2.14.1+cu130 |
| GPU | physical GPU 1 (RTX 4090), pinned via `CUDA_VISIBLE_DEVICES=1` |
| GPU 0 | hosts the Bonsai llama-server — never touched |

`torch` was installed from the PyTorch cu130 index. GPU 0 was left alone
throughout; all training ran on GPU 1 with ~14 GB free.

## 4. Data

Downloaded from the HuggingFace mirror `hf-mirror.com` (the canonical
`huggingface.co` was unreachable from this machine — it resolves to IPv6 only).
Dataset: **NeRF synthetic (Blender)**, 8 scenes of 800×800 RGBA renders with
`transforms_{train,val,test}.json` camera files.

- `lego`, `chair`, `drums` — 1206 files, 424 MB
- 100 train / 100 val / 200 test views per scene
- Fetched with a 16-thread downloader (`scripts/download_dataset.py`), ~2 MB/s

Training and evaluation run at **downscale 2 → 400×400**, using the alpha
channel composited over white (the standard NeRF-synthetic convention).

## 5. Training

```
.venv/bin/python -m src.train --data_root data/nerf_synthetic/lego \
    --out outputs/lego --device cuda:0 --downscale 2 --iters 15000 \
    --hull_res 192 --densify_grad_threshold 1e-4 --pair_budget 65536
```

- 15 000 iterations, one random training view per step
- Loss `0.8·L1 + 0.2·(1 − SSIM)`
- SH degree warm-up 0 → 3
- Adaptive density control every 100 iterations from 500 to 12 000: clone
  small under-reconstructed Gaussians, split over-large ones (children jittered
  by the parent's own covariance), prune low-opacity / world-oversized /
  screen-oversized ones
- 219 753 Gaussians at the end, 54.5 MB binary PLY

## 6. Results

Two scenes were trained end to end at **400×400, 15 000 iterations** on GPU 1,
with identical hyper-parameters.

| scene | split | PSNR | SSIM | views | Gaussians | train time |
|---|---|---|---|---|---|---|
| lego | val | **34.35** | **0.9782** | 100 | 219 753 | 30 min |
| lego | test | **33.06** | **0.9741** | 200 | | |
| chair | val | **36.65** | **0.9860** | 100 | 127 542 | 19 min |
| chair | test | **35.60** | **0.9842** | 200 | | |

For reference, the published 3DGS numbers on NeRF-synthetic are ~33–35 dB at
800×800 with SfM initialisation; these runs reach comparable quality at half
resolution with no SfM data at all.

Convergence on `lego`: val PSNR 25.25 @1k, 27.99 @2k, 28.71 @3k, a plateau
≈28.7–29.5 through 11k, then **33.90 @12k** the moment densification stops and
the model settles, reaching 34.35 at 15k. `chair` shows the same signature
(34.50 @11k → 36.22 @12k → 36.65 @15k).

Artefacts per scene in `outputs/<scene>/`:
- `point_cloud.ply` — binary PLY with positions, scales, rotations, opacities
  and 48 SH coefficients (verified by round-tripping through `load_ply`:
  rendering from the exported file reproduces the training-time metrics exactly)
- `renders/val_*.png`, `renders/test_*.png` — novel views plus ground truth
- `renders/turntable/` — 90 orbit frames, and `renders/turntable.mp4`
- `metrics.json` — full training log

Turntable sanity (programmatic, since the session's model cannot view images):

| scene | frames | mean brightness | content std | blank frames | max frame-to-frame jump |
|---|---|---|---|---|---|
| lego | 90 | 0.838 | 0.286 | 0 | 0.0096 |
| chair | 90 | 0.932 | 0.173 | 0 | 0.0040 |

## 7. Verification

**Rasteriser correctness.** `tests/test_rasterizer.py` renders a random scene
with both the vectorised rasteriser and a brute-force per-pixel reference that
enumerates every Gaussian whose tile the pixel belongs to, sorts front-to-back
and composites in a Python loop. Agreement:

```
aa_sigma_px=0.0: max_abs_diff=8.9e-07  mean_abs_diff=7.4e-08
aa_sigma_px=0.3: max_abs_diff=9.9e-07  mean_abs_diff=7.3e-08
```

i.e. float32 round-off. The same test confirms gradients w.r.t. means and
opacities are finite.

**Render sanity.** Turntable: 90 frames, no blank frames, mean brightness
0.838, largest frame-to-frame brightness jump 0.0096 (smooth orbit).

## 8. Engineering log: what broke and why

Five bugs were found and fixed during bring-up; each is documented in code.

1. **Tile index out of bounds.** Screen-space bounding boxes were clamped to
   `1e9` instead of the image, so tile ids exceeded the tile grid.
2. **OOM #1 — every chunk retained.** Autograd kept all chunk intermediates
   alive because the image was built by chaining `index_add` over all chunks
   (~13 GB at 2.3 M pairs). Fixed with per-chunk gradient checkpointing.
3. **OOM #2 — pair-proportional retention.** Even with checkpointing, each
   chunk retained an O(pairs) index tensor and contribution view. Fixed by
   summing per tile *inside* the checkpointed function, making everything
   retained O(tiles). Peak memory at 780 k pairs: **3.43 GB → 0.78 GB**.
4. **Adam state silently discarded.** Densification changed the parameter
   tensor shapes, and the first implementation rebuilt the optimiser each time —
   wiping momentum every 100 iterations. The model degraded as Gaussians
   multiplied (val PSNR 28.71 → 17.26) and splits exploded to +4955/step.
   Fixed by carrying optimiser state through `_cat_to_optimizer` /
   `_prune_optimizer`, as the reference implementation does.
5. **Opacity reset froze opacity.** `reset_opacity` reassigned `self._opacity`
   but left Adam holding the old tensor, so opacity stayed clamped at 0.01
   forever. Fixed by swapping the tensor into the optimiser slot and zeroing
   its state.

Two tuning insights, both from measurement rather than guesswork:

- **Densification threshold must scale with resolution.** At 400×400 the loss
  gradient per Gaussian is ~4× smaller than at 800×800, so the stock
  `2e-4` threshold stopped densifying entirely at 16 k Gaussians. Measured
  gradient quantiles (q90 ≈ 9e-5, q99 ≈ 4e-4) pointed to `1e-4`.
- **Densification perturbs the model.** Val PSNR plateaued while densification
  ran and jumped 5 dB the moment it stopped — worth revisiting as a schedule
  question rather than a fixed threshold.

## 9. Layout

```
src/rasterizer.py         vectorised tile rasteriser (core algorithm)
src/gaussian_model.py     parameters, activations, SH, density control
src/dataset.py            Blender/NeRF-synthetic loader, space carving, seeding
src/train.py              training loop
src/render.py             novel views, PLY export, turntable
src/metrics.py            PSNR / SSIM
tests/test_rasterizer.py  agreement check vs brute-force reference
tests/diag_grads.py       gradient-distribution diagnostic
tests/bench.py            throughput benchmark
tests/bench_mem.py        checkpointing memory check
scripts/download_dataset.py   dataset fetcher
scripts/run_all.sh        download -> train -> render
```

Reproduce end to end:

```bash
bash scripts/run_all.sh lego 15000 2
```
