"""Equirectangular dome-light background (robot frame) from cleaned real photos with known camera poses.

Works for any environment (lab, kitchen, office, outdoors). Each photo (policy-camera images and posed
supplementary photos) is projected onto the sphere along its own viewing rays. Directions whose ray from
that photo's camera hits a near-field rectangle (a horizontal surface that is modelled as geometry, e.g.
the work surface) are skipped, because near-field content must be real geometry with parallax.
No blur or smooth hole filling is applied. Directions no photo covers get a flat fill and are written
to out_holes.png; the script reports how much of each --check camera's view is a hole, which must be 0
for the external camera (outpaint more with inpaint.py --pad instead of blurring).

The dome mapping (u = -azimuth / 2pi, v = 0.5 - elevation / pi, +x forward at u = 0) matches RoboLab's
/World/background DomeLight (texture_format latlong); it was verified with axis-direction renders.

config.json:
{"out": ".../lab_background.png", "size": [4096, 2048], "floor_rgb": [41, 40, 42],
 "near_field": [{"x": [0.05, 0.75], "y": [-0.25, 0.40], "z": 0.0, "margin": 0.05}],
 "photos": [{"image": "ext_clean.png", "fit": "fit_ext.json", "inpaint": "ext_clean_inpaint.json",
             "weight": 10, "exclude": "ext_keep_out.json"}],
 "check": ["fit_ext.json"]}
fit: output of fit_camera.py. inpaint: sidecar of inpaint.py (gives the pad offsets). exclude: polygons
in original photo pixels that must not be used (e.g. regions you did not inpaint).

Usage: python make_dome.py config.json
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cam_math import gl_quat_to_cv_R


def dome_dirs(W, H):
    uu, vv = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    phi, theta = -2 * np.pi * uu, np.pi / 2 - np.pi * vv
    return np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], -1)


def project_dirs(dirs, fit, scale=1.0, off=(0, 0)):
    R = gl_quat_to_cv_R(fit["quat_xyzw_gl"])
    c = dirs @ R
    fx = fit["fx"] * scale
    cx, cy = fit.get("cx", fit["size"][0] / 2) * scale + off[0], fit.get("cy", fit["size"][1] / 2) * scale + off[1]
    front = c[..., 2] > 0.05
    z = np.maximum(c[..., 2], 1e-6)
    return np.where(front, fx * c[..., 0] / z + cx, -1), np.where(front, fx * c[..., 1] / z + cy, -1), front


def near_field(dirs, pos, nf):
    if isinstance(nf, list):
        out = np.zeros(dirs.shape[:2], bool)
        for r in nf:
            out |= near_field(dirs, pos, r)
        return out
    m = nf.get("margin", 0.05)
    dz = np.where(dirs[..., 2] < -1e-6, dirs[..., 2], -1e-6)
    t = (nf["z"] - pos[2]) / dz
    hx, hy = pos[0] + t * dirs[..., 0], pos[1] + t * dirs[..., 1]
    return (dirs[..., 2] < 0) & (hx > nf["x"][0] - m) & (hx < nf["x"][1] + m) & (hy > nf["y"][0] - m) & (hy < nf["y"][1] + m)


def main():
    cfg = json.load(open(sys.argv[1]))
    base = os.path.dirname(os.path.abspath(sys.argv[1]))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    W, H = cfg.get("size", [4096, 2048])
    dirs = dome_dirs(W, H)
    acc, wsum = np.zeros((H, W, 3)), np.zeros((H, W))
    for ph in cfg["photos"]:
        img = cv2.imread(rel(ph["image"])).astype(np.float32)
        ih, iw = img.shape[:2]
        fit = json.load(open(rel(ph["fit"])))
        T, R_, B, L = json.load(open(rel(ph["inpaint"])))["pad_trbl"] if ph.get("inpaint") else (0, 0, 0, 0)
        s = (iw - L - R_) / fit["size"][0]
        valid = np.ones((ih, iw), np.uint8)
        if ph.get("exclude"):
            for poly in json.load(open(rel(ph["exclude"]))):
                cv2.fillPoly(valid, [(np.array(poly) * s + [L, T]).astype(np.int32)], 0)
        px, py, front = project_dirs(dirs, fit, s, (L, T))
        inb = front & (px >= 0) & (px < iw - 1) & (py >= 0) & (py < ih - 1)
        if "near_field" in cfg:
            inb &= ~near_field(dirs, fit["pos"], cfg["near_field"])
        samp = cv2.remap(img, px.astype(np.float32), py.astype(np.float32), cv2.INTER_LANCZOS4)
        v = cv2.remap(valid.astype(np.float32), px.astype(np.float32), py.astype(np.float32), cv2.INTER_NEAREST)
        edge = np.clip(np.minimum.reduce([px / iw, 1 - px / iw, py / ih, 1 - py / ih]) * 8, 0, 1)
        w = np.where(inb, v * edge, 0) * ph.get("weight", 1.0)
        acc += samp * w[..., None]
        wsum += w
        print(ph["image"], "dome coverage", round(float((w > 0).mean()), 3))
    pan = acc / np.maximum(wsum, 1e-6)[..., None]
    hole = wsum < 1e-3
    upper = (~hole) & (dirs[..., 2] > 0)
    fill_up = pan[upper].mean(0) if upper.any() else np.array([200.0, 200.0, 200.0])
    floor = np.array(cfg.get("floor_rgb", [41, 40, 42])[::-1], float)
    pan[hole] = np.where(dirs[hole][:, 2:3] > 0, fill_up, floor)
    out = cfg["out"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.clip(pan, 0, 255).astype(np.uint8))
    cv2.imwrite(out.rsplit(".", 1)[0] + "_holes.png", hole.astype(np.uint8) * 255)
    print("wrote", out, "hole fraction", round(float(hole.mean()), 3))
    for f in cfg.get("check", []):
        fit = json.load(open(rel(f)))
        px, py, front = project_dirs(dirs, fit)
        vis = front & (px >= 0) & (px < fit["size"][0]) & (py >= 0) & (py < fit["size"][1])
        if "near_field" in cfg:
            vis &= ~near_field(dirs, fit["pos"], cfg["near_field"])
        frac = float((hole & vis).sum() / max(vis.sum(), 1))
        print(f"check {f}: {frac:.4f} of its visible background is hole -> {'ok' if frac < 1e-3 else 'OUTPAINT MORE'}")


if __name__ == "__main__":
    main()
