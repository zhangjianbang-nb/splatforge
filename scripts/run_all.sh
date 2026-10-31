#!/bin/bash
# End-to-end driver: download data -> train -> render novel views.
#
# GPU policy: GPU0 hosts the Bonsai llama-server and must never be touched, so
# everything here is pinned to physical GPU1 via CUDA_VISIBLE_DEVICES.
set -euo pipefail

ROOT=/home/whu/splatforge
export CUDA_VISIBLE_DEVICES=1

SCENE=${1:-lego}
ITERS=${2:-12000}
DOWNSCALE=${3:-2}

echo "=== 1. dataset ==="
"$ROOT/.venv/bin/python" "$ROOT/scripts/download_dataset.py" "$SCENE"

echo "=== 2. train ==="
"$ROOT/.venv/bin/python" -m src.train \
  --data_root "$ROOT/data/nerf_synthetic/$SCENE" \
  --out "$ROOT/outputs/$SCENE" \
  --device cuda:0 \
  --downscale "$DOWNSCALE" \
  --iters "$ITERS" \
  --aa_sigma_px 0.3

echo "=== 3. render novel views ==="
"$ROOT/.venv/bin/python" -m src.render \
  --data_root "$ROOT/data/nerf_synthetic/$SCENE" \
  --ckpt "$ROOT/outputs/$SCENE" \
  --out "$ROOT/outputs/$SCENE/renders" \
  --device cuda:0 \
  --downscale "$DOWNSCALE"

echo "done -> $ROOT/outputs/$SCENE"
