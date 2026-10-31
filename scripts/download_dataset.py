#!/usr/bin/env python3
"""Download NeRF-synthetic scenes from the HuggingFace mirror (hf-mirror.com).

The mirror hosts the classic Blender/NeRF-synthetic dataset (8 scenes, 800x800
PNGs + transforms_*.json camera files).  We only need the images; poses come
from the JSON files.
"""
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

MIRROR = "https://hf-mirror.com/datasets/pablovela5620/nerf-synthetic-mirror/resolve/main"
ROOT = "/home/whu/splatforge/data/nerf_synthetic"

SPLIT_COUNTS = {"train": 100, "val": 100, "test": 200}


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path, "skip"
    tmp = path + ".part"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(tmp) < 1000:
                raise IOError("too small")
            os.replace(tmp, path)
            return path, "ok"
        except Exception as e:  # noqa: BLE001 - retry any transport error
            if attempt == 3:
                return path, f"FAIL {e}"
    return path, "FAIL"


def main():
    scenes = sys.argv[1:] or ["lego"]
    jobs = []
    for scene in scenes:
        for split, n in SPLIT_COUNTS.items():
            for i in range(n):
                url = f"{MIRROR}/{scene}/{split}/r_{i}.png"
                path = f"{ROOT}/{scene}/{split}/r_{i}.png"
                jobs.append((url, path))
        for name in ("transforms_train.json", "transforms_val.json", "transforms_test.json"):
            jobs.append((f"{MIRROR}/{scene}/{name}", f"{ROOT}/{scene}/{name}"))

    print(f"downloading {len(jobs)} files for scenes {scenes}", flush=True)
    ok = skip = fail = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch, u, p): (u, p) for u, p in jobs}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            path, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(" ", path, status, flush=True)
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
