"""Photometric variants of a reconstructed dome background, for background randomization (Phase 10).

The reconstructed dome matches one photo, taken under one lighting condition. Each variant keeps every pixel's
content (no blur, no generative change) and changes how the room is lit:
- exposure (in linear light) and contrast
- white balance, warm or cool
- saturation
- a smooth directional illumination field over azimuth / elevation (a window or lamp on another side)
- a few soft local brightness blobs (lamps, monitors, people passing)
- a small yaw roll of the panorama (background content shifts sideways)
Variant 0 is the unchanged dome, so the photographed condition stays in the set.

Usage:
  python make_dome_variants.py dome.png OUT_DIR [--n 12] [--seed 0] [--exposure 0.75 1.3] [--wb 0.08]
      [--contrast 0.1] [--saturation 0.15] [--gradient 0.25] [--blobs 3] [--yaw-deg 5]
Writes OUT_DIR/dome_00.png ... and OUT_DIR/variants.json (parameters per file) and OUT_DIR/contact.jpg.
Reference the directory in dr.json: "background": {"dir": ".../dome_variants", "intensity": [500, 900]}.
"""
import argparse
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dome")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exposure", type=float, nargs=2, default=[0.75, 1.3])
    ap.add_argument("--wb", type=float, default=0.08)
    ap.add_argument("--contrast", type=float, default=0.1)
    ap.add_argument("--saturation", type=float, default=0.15)
    ap.add_argument("--gradient", type=float, default=0.25)
    ap.add_argument("--blobs", type=int, default=3)
    ap.add_argument("--yaw-deg", type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = cv2.imread(a.dome, cv2.IMREAD_UNCHANGED)
    alpha = src[..., 3:] if src.shape[2] == 4 else None
    img = src[..., :3].astype(np.float32) / 255.0
    h, w = img.shape[:2]
    az = (np.arange(w) / w * 2 * np.pi)[None, :].repeat(h, 0)
    el = (0.5 - (np.arange(h) + 0.5) / h)[:, None].repeat(w, 1) * np.pi
    rng = np.random.default_rng(a.seed)
    meta, thumbs = [], []
    for i in range(a.n):
        if i == 0:
            out, p = img.copy(), {"original": True}
        else:
            p = {"exposure": float(rng.uniform(*a.exposure)), "warm": float(rng.uniform(-a.wb, a.wb)),
                 "tint": float(rng.uniform(-a.wb, a.wb) / 3), "contrast": float(rng.uniform(1 - a.contrast, 1 + a.contrast)),
                 "saturation": float(rng.uniform(1 - a.saturation, 1 + a.saturation)),
                 "grad_amp": float(rng.uniform(0, a.gradient)), "grad_az": float(rng.uniform(0, 2 * np.pi)),
                 "grad_el": float(rng.uniform(-0.5, 0.5)), "yaw_deg": float(rng.uniform(-a.yaw_deg, a.yaw_deg)), "blobs": []}
            lin = img ** 2.2
            field = 1 + p["grad_amp"] * (np.cos(az - p["grad_az"]) * np.cos(el) + p["grad_el"] * np.sin(el))
            for _ in range(rng.integers(0, a.blobs + 1)):
                b = {"az": float(rng.uniform(0, 2 * np.pi)), "el": float(rng.uniform(-0.3, 0.6)),
                     "r": float(rng.uniform(0.15, 0.5)), "gain": float(rng.uniform(0.7, 1.4))}
                ang = np.arccos(np.clip(np.cos(el) * np.cos(b["el"]) * np.cos(az - b["az"]) + np.sin(el) * np.sin(b["el"]), -1, 1))
                field = field * (1 + (b["gain"] - 1) * np.exp(-(ang / b["r"]) ** 2))
                p["blobs"].append(b)
            lin = lin * p["exposure"] * field[..., None]
            lin = lin * np.array([1 - p["warm"], 1 + p["tint"], 1 + p["warm"]], np.float32)
            out = np.clip(lin, 0, 1) ** (1 / 2.2)
            grey = out.mean(2, keepdims=True)
            out = grey + (out - grey) * p["saturation"]
            mean = out.mean()
            out = np.clip((out - mean) * p["contrast"] + mean, 0, 1)
            out = np.roll(out, int(round(p["yaw_deg"] / 360 * w)), axis=1)
        o8 = (out * 255 + 0.5).astype(np.uint8)
        if alpha is not None:
            o8 = np.concatenate([o8, alpha], 2)
        name = f"dome_{i:02d}.png"
        cv2.imwrite(os.path.join(a.out, name), o8)
        meta.append({"file": name, **p})
        thumbs.append(cv2.resize(o8[..., :3], (512, 256), interpolation=cv2.INTER_AREA))
    cols = 4
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    cv2.imwrite(os.path.join(a.out, "contact.jpg"),
                np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)]))
    json.dump(meta, open(os.path.join(a.out, "variants.json"), "w"), indent=1)
    print(f"[dome_variants] wrote {a.n} variants to {a.out}")


if __name__ == "__main__":
    main()
