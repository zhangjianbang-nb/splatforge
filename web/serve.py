#!/usr/bin/env python3
"""Local server for the SplatForge 3D viewer.

Serves the static front end in ``web/`` and adds the endpoints it needs:

  GET  /api/scenes        trained checkpoints: framing, metrics, asset URLs
  POST /api/infer         real forward pass through the trained model (GPU 1),
                          rendered to ``web/assets/<scene>_infer.png``
  GET  /ply/<scene>       streams ``outputs/<scene>/point_cloud.ply``
  GET  /media/<scene>/<f> streams anything under ``outputs/<scene>/renders``

Usage::

    .venv/bin/python web/serve.py --port 8770

Inference is pinned to GPU 1 through ``CUDA_VISIBLE_DEVICES`` so the Bonsai
server on GPU 0 is never touched.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
OUTPUTS = REPO / "outputs"
DATA = REPO / "data" / "nerf_synthetic"
ASSETS = WEB / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

CHUNK = 1 << 20
FLOATS_PER_VERTEX = 62          # 3DGS PLY layout written by GaussianModel.to_ply

# ---------------------------------------------------------------- PLY helpers


def ply_vertex_count(path: Path) -> tuple[int, int]:
    """Return (vertex count, byte offset of the binary body)."""
    with open(path, "rb") as f:
        count = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line == "end_header":
                return count, f.tell()


_framing_cache: dict[str, dict] = {}


def scene_framing(scene: str) -> dict:
    """Robust bounds of the trained cloud, plus the training camera pose.

    Percentiles rather than min/max: a 3DGS scene always grows a few floaters
    far outside the object, and framing on min/max would leave the object a
    speck in the middle of the viewport.
    """
    ply = OUTPUTS / scene / "point_cloud.ply"
    key = f"{ply}:{ply.stat().st_mtime}"
    if key in _framing_cache:
        return _framing_cache[key]

    n, offset = ply_vertex_count(ply)
    with open(ply, "rb") as f:
        f.seek(offset)
        raw = np.frombuffer(f.read(n * FLOATS_PER_VERTEX * 4), dtype=np.float32)
    xyz = raw.reshape(n, FLOATS_PER_VERTEX)[:, :3]

    lo = np.percentile(xyz, 2, axis=0)
    hi = np.percentile(xyz, 98, axis=0)
    center = ((lo + hi) * 0.5).tolist()
    radius = float(np.linalg.norm(hi - lo) * 0.5)

    eye = None
    target = None
    up = [0.0, 1.0, 0.0]
    fov_y = 50.0
    tf = DATA / scene / "transforms_train.json"
    if tf.exists():
        meta = json.loads(tf.read_text())
        c2w = np.stack([np.array(fr["transform_matrix"], dtype=np.float64)
                        for fr in meta["frames"]])
        eyes = c2w[:, :3, 3]
        mean_eye = eyes.mean(axis=0)
        # The training cameras orbit the object, so the least-spread direction of
        # their positions is the scene's up axis.  NeRF-synthetic is authored
        # z-up, which a y-up orbit camera would otherwise render lying on its side.
        spread = np.linalg.svd(eyes - mean_eye)[2]
        axis = spread[-1] / np.linalg.norm(spread[-1])
        if np.dot(axis, mean_eye - np.asarray(center)) < 0:
            axis = -axis
        up = axis.tolist()

        # Showcase view: the training camera nearest 25 degrees elevation.  The
        # mean camera sits near the pole of the capture dome, so it would frame
        # the object almost top-down.
        offsets = eyes - np.asarray(center)
        height = offsets @ axis
        planar = np.linalg.norm(offsets - np.outer(height, axis), axis=1)
        elevation = np.degrees(np.arctan2(height, planar + 1e-9))
        pick = int(np.argmin(np.abs(elevation - 25.0)))
        fwd = -c2w[pick, :3, 2]
        fwd = fwd / np.linalg.norm(fwd)
        eye = eyes[pick].tolist()
        # Reproduce that camera's exact orientation by looking along its forward axis.
        t = float(np.dot(np.asarray(center) - eyes[pick], fwd))
        target = (eyes[pick] + t * fwd).tolist()

        # load_split derives fx = fy = 0.5*w / tan(camera_angle_x/2); the images
        # are square, so the vertical fov equals camera_angle_x.
        fov_y = float(np.degrees(meta["camera_angle_x"]))

    metrics = {}
    mf = OUTPUTS / scene / "metrics.json"
    if mf.exists():
        metrics = json.loads(mf.read_text())

    info = {
        "id": scene,
        "plyUrl": f"/ply/{scene}",
        "nGaussians": int(n),
        "center": center,
        "target": target or center,
        "radius": radius,
        "up": up,
        "initialEye": eye or [center[0], center[1] + radius * 0.2, center[2] + radius * 2.2],
        "fovY": fov_y,
        "valPsnr": metrics.get("val_psnr"),
        "testPsnr": metrics.get("test_psnr"),
        "valSsim": metrics.get("val_ssim"),
        "resolution": metrics.get("resolution"),
        "trainMinutes": metrics.get("train_minutes"),
    }
    render_dir = OUTPUTS / scene / "renders"
    video = render_dir / "turntable.mp4"
    if video.exists():
        # Version the URL by mtime so a regenerated turntable is not served
        # from the browser cache.
        info["turntableUrl"] = f"/media/{scene}/turntable.mp4?v={int(video.stat().st_mtime)}"
    _framing_cache[key] = info
    return info


def list_scenes() -> list[dict]:
    scenes = sorted(p.parent.name for p in OUTPUTS.glob("*/point_cloud.ply"))
    return [scene_framing(s) for s in scenes]


# ------------------------------------------------------------- torch inference

_models: dict[str, tuple] = {}
_model_lock = threading.Lock()
_infer_cache: dict[str, dict] = {}


def get_model(scene: str):
    """Load (and cache) the trained model on GPU 1."""
    with _model_lock:
        if scene in _models:
            return _models[scene]
        import torch
        from src.dataset import Scene as SceneDS
        from src.render import load_model, load_ply

        device = torch.device("cuda:0")     # CUDA_VISIBLE_DEVICES=1 -> GPU 1
        dataset = SceneDS(str(DATA / scene), downscale=2).to(device)
        model = load_model(dataset, device)
        model = load_ply(model, str(OUTPUTS / scene / "point_cloud.ply"))
        _models[scene] = (model, dataset, torch)
        return _models[scene]


def run_inference(scene: str, mode: str, idx: int) -> dict:
    """Render a view with the trained model — an actual forward pass."""
    cache_key = f"{scene}:{mode}:{idx}"
    if cache_key in _infer_cache:
        return _infer_cache[cache_key]

    from src.metrics import psnr
    from src.render import render_camera, save_image, turntable_cameras

    model, dataset, torch = get_model(scene)
    t0 = time.time()
    with torch.no_grad():
        if mode == "test":
            cams = dataset.test
            cam = cams[idx % len(cams)]
            img = render_camera(model, cam, dataset.background, 8, 131072, 0.3, 64.0)
            save_image(img, ASSETS / f"{scene}_infer.png")
            save_image(cam.image, ASSETS / f"{scene}_infer_gt.png")
            result = {
                "mode": "test",
                "idx": idx % len(cams),
                "render": f"/assets/{scene}_infer.png",
                "groundTruth": f"/assets/{scene}_infer_gt.png",
                "psnr": float(psnr(img, cam.image)),
                "note": "held-out test view — never seen during training",
            }
        else:
            cams = turntable_cameras(dataset, 360)
            cam = cams[idx % 360]
            cam.to(model.device)
            img = render_camera(model, cam, dataset.background, 8, 131072, 0.3, 64.0)
            save_image(img, ASSETS / f"{scene}_infer.png")
            result = {
                "mode": "novel",
                "idx": idx % 360,
                "render": f"/assets/{scene}_infer.png",
                "note": "novel view — an orbit angle absent from the training set",
            }
    result["milliseconds"] = int((time.time() - t0) * 1000)
    result["nGaussians"] = int(model.means.shape[0])
    result["device"] = "GPU 1"
    _infer_cache[cache_key] = result
    return result


# ---------------------------------------------------------------- HTTP layer

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".ply": "application/octet-stream",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SplatForge/1.0"

    def log_message(self, fmt, *args):      # quieter default logging
        sys.stderr.write("  %s\n" % (fmt % args))

    # ---------- helpers ----------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code, message):
        self._send_json({"error": message}, code)

    def _stream(self, path: Path, ctype: str | None = None, cache: bool = False):
        if not path.is_file():
            self._send_error_json(404, f"not found: {path.name}")
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype or
                         MIME.get(path.suffix, mimetypes.guess_type(str(path))[0] or
                                  "application/octet-stream"))
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control",
                         "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ---------- routes ----------
    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            return self._stream(WEB / "index.html", cache=False)
        if path == "/api/scenes":
            try:
                return self._send_json({"scenes": list_scenes()})
            except Exception as exc:                       # pragma: no cover
                return self._send_error_json(500, str(exc))
        if path == "/api/health":
            return self._send_json({"ok": True, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        if path.startswith("/ply/"):
            scene = path[len("/ply/"):].strip("/")
            if "/" in scene or not scene:
                return self._send_error_json(400, "bad scene name")
            return self._stream(OUTPUTS / scene / "point_cloud.ply")
        if path.startswith("/media/"):
            rest = path[len("/media/"):].strip("/")
            parts = rest.split("/", 1)
            if len(parts) != 2:
                return self._send_error_json(400, "expected /media/<scene>/<file>")
            scene, name = parts
            target = (OUTPUTS / scene / "renders" / name).resolve()
            renders = (OUTPUTS / scene / "renders").resolve()
            if not str(target).startswith(str(renders)):
                return self._send_error_json(400, "bad path")
            return self._stream(target, cache=True)
        if path.startswith("/assets/"):
            name = path[len("/assets/"):].strip("/")
            if "/" in name or not name:
                return self._send_error_json(400, "bad asset name")
            return self._stream(ASSETS / name)

        # static front end
        name = path.lstrip("/")
        if "/" in name or not name:
            return self._send_error_json(404, "not found")
        return self._stream(WEB / name)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/infer":
            return self._send_error_json(404, "not found")

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send_error_json(400, "invalid JSON body")

        scene = str(payload.get("scene", ""))
        mode = str(payload.get("mode", "test"))
        idx = int(payload.get("idx", 0))
        if mode not in ("test", "novel"):
            return self._send_error_json(400, "mode must be 'test' or 'novel'")
        if not (OUTPUTS / scene / "point_cloud.ply").exists():
            return self._send_error_json(404, f"no trained checkpoint for '{scene}'")

        try:
            return self._send_json(run_inference(scene, mode, idx))
        except Exception as exc:
            return self._send_error_json(500, f"inference failed: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    scenes = list_scenes()
    print(f"SplatForge viewer  ->  http://{args.host}:{args.port}/")
    for s in scenes:
        print(f"  scene {s['id']}: {s['nGaussians']} gaussians"
              f"{f', val PSNR {s['valPsnr']:.2f}' if s.get('valPsnr') else ''}")
    print("  inference pinned to GPU 1 (CUDA_VISIBLE_DEVICES=1)")

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
