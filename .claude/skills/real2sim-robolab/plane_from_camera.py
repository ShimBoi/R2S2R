"""Locate a patterned support plane (height, pattern yaw and phase) in the robot frame from one image taken by a
camera whose pose is already known, typically the wrist camera at the recorded joint state.

Label-free: every detected pattern cell is back-projected onto the candidate plane z, and (z, yaw) are searched
so the back-projected cells form a lattice of period P (circular coherence). Use it to anchor the table in the
robot frame before fitting an external camera (compose_ext.py).

Scale: with an unknown pattern period P, z and P trade off exactly (z - cam_z scales with P). Pass your best
guess for P and resolve the scale later with scale_scan (render_check.py --cands) against the robot silhouette,
or with a ruler measurement.

Usage:
  python plane_from_camera.py cam.json lattice.json --period P [--z-range -0.1 0.4] [--yaw-range 0 90]
      [--max-v 600] [--point name:u,v,dz ...] --out plane.json
cam.json: {"pos": [x,y,z], "quat_xyzw_gl": [...], "fx": F, "size": [W, H]} (render_views/refine output, or
          a pose dumped from sim at the recorded joints).
lattice.json: fit_lattice.py output for this image (only "uv" is used).
--point: back-project pixel (u, v) onto the plane raised by dz (e.g. the middle height of an object).
--max-v: ignore cells below this image row (gripper fingers, close clutter).
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation


def rays(uv, fx, W, H, R):
    return np.c_[(uv[:, 0] - W / 2) / fx, -(uv[:, 1] - H / 2) / fx, -np.ones(len(uv))] @ R.T


def on_plane(pos, d, z):
    s = (z - pos[2]) / d[:, 2]
    return pos[None] + s[:, None] * d, s


def coherence(xy, yaw, P):
    c, s = np.cos(yaw), np.sin(yaw)
    q = xy @ np.array([[c, -s], [s, c]]) / P
    m = np.exp(2j * np.pi * q).mean(0)
    return 2 - np.abs(m).sum(), np.angle(m) / (2 * np.pi), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("lattice")
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--z-range", type=float, nargs=2, default=[-0.1, 0.4])
    ap.add_argument("--yaw-range", type=float, nargs=2, default=[0, 90])
    ap.add_argument("--max-v", type=float, default=1e9)
    ap.add_argument("--point", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = json.load(open(a.cam))
    pos, R, fx = np.array(c["pos"]), Rotation.from_quat(c["quat_xyzw_gl"]).as_matrix(), c["fx"]
    W, H = c.get("size", [1280, 720])
    uv = np.array(json.load(open(a.lattice))["uv"])
    uv = uv[uv[:, 1] < a.max_v]
    d = rays(uv, fx, W, H, R)
    best = None
    for z in np.arange(a.z_range[0], a.z_range[1], 0.0025):
        P3, s = on_plane(pos, d, z)
        if (s <= 0).any():
            continue
        for yaw in np.radians(np.arange(a.yaw_range[0], a.yaw_range[1], 0.25)):
            sc, ph, _ = coherence(P3[:, :2], yaw, a.period)
            if best is None or sc < best[0]:
                best = (sc, z, yaw, ph)
    sc, z, yaw, ph = best
    P3, _ = on_plane(pos, d, z)
    _, _, q = coherence(P3[:, :2], yaw, a.period)
    r = (q - ph - np.round(q - ph)) * a.period * 1000
    out = dict(z=float(z), yaw_deg=float(np.degrees(yaw)), phase=ph.tolist(), period=a.period,
               coherence=float(2 - sc), resid_mm_rms=float(np.sqrt((r ** 2).sum(1).mean())), cam=c, points={})
    print(f"[plane] z {z:.4f} yaw {np.degrees(yaw):.2f} deg phase {np.round(ph, 3)} coherence {2 - sc:.3f}/2 "
          f"resid {out['resid_mm_rms']:.1f} mm rms over {len(uv)} cells")
    for spec in a.point:
        name, rest = spec.split(":")
        u, v, dz = map(float, rest.split(","))
        X, _ = on_plane(pos, rays(np.array([[u, v]]), fx, W, H, R), z + dz)
        out["points"][name] = X[0].tolist()
        print(f"[plane] {name} ({u:.0f},{v:.0f}) +{dz:.3f} -> {np.round(X[0], 4).tolist()}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
