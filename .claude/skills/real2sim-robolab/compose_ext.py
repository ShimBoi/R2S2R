"""External-camera pose in the robot frame from (1) its own view of a patterned plane (fit_lattice.py: fx, tilt,
height in pattern units), (2) the plane located in the robot frame by another camera (plane_from_camera.py:
height, pattern yaw), and (3) one anchor point seen by both cameras (pins the in-plane translation).

The pattern's 4-fold symmetry leaves 4 candidate yaws; the right one is picked by where a known robot point
(default: the base, (0, 0, 0.15)) lands in the image (--check-uv), or listed for you to judge.

Unknown pattern period: pass --scales. Each scale s multiplies the period and moves the plane and anchor
along the reference camera's rays (the scale ambiguity of plane_from_camera.py). Every scale gives one
consistent candidate; pick the right one with the robot silhouette (render_check.py --cands) or a ruler.

Usage:
  python compose_ext.py lattice_ext.json plane.json --anchor obj_a:722,388 [--check-uv 1150,560]
      [--check-xyz 0 0 0.15] [--scales 0.9 0.95 1 1.05 1.1 1.15 1.2] --out cands.json
Writes cands.json: one entry per scale {scale, period, z, pos, quat_xyzw_gl, fx, size, anchor_xyz, check_uv},
the best yaw per scale, ready for render_check.py --cands and for render_views.py --ext-fit.
"""
import argparse
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CV2GL = np.diag([1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lattice")
    ap.add_argument("plane")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--check-xyz", type=float, nargs=3, default=[0.0, 0.0, 0.15])
    ap.add_argument("--check-uv", default=None)
    ap.add_argument("--scales", type=float, nargs="*", default=[1.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lat, pl = json.load(open(a.lattice)), json.load(open(a.plane))
    W, H = lat["size"]
    fx = lat["fx"]
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    Rgc = cv2.Rodrigues(np.array(lat["rvec"]))[0]
    C = -Rgc.T @ np.array(lat["tvec"])
    if C[2] < 0:
        Rgc, C = Rgc @ np.diag([1.0, -1.0, -1.0]), C * np.array([1, -1, -1])
    unit0 = pl["period"] / np.sqrt(abs(np.linalg.det(np.array(lat["T"], float))))
    name, uvs = a.anchor.split(":")
    auv = np.array([float(v) for v in uvs.split(",")])
    A0, Z0, cam_ref = np.array(pl["points"][name]), pl["z"], np.array(pl["cam"]["pos"])
    check_uv = np.array([float(v) for v in a.check_uv.split(",")]) if a.check_uv else None
    out = []
    for s in a.scales:
        z = cam_ref[2] + (Z0 - cam_ref[2]) * s
        A = cam_ref + (A0 - cam_ref) * s
        h = abs(C[2]) * unit0 * s
        cands = []
        for k in range(4):
            for mirror in (False, True):
                ang = np.radians(pl["yaw_deg"]) + k * np.pi / 2
                Rwg = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
                if mirror:
                    Rwg = Rwg @ np.diag([1.0, -1.0, 1.0])
                Rwc = Rwg @ Rgc.T
                if np.linalg.det(Rwc) < 0:
                    continue
                ray = Rwc @ np.linalg.solve(K, [auv[0], auv[1], 1.0])
                if ray[2] >= 0:
                    continue
                pos = A - (A[2] - (z + h)) / ray[2] * ray
                pc = Rwc.T @ (np.array(a.check_xyz) - pos)
                cuv = (K @ pc / pc[2])[:2] if pc[2] > 0 else np.array([np.nan, np.nan])
                err = np.linalg.norm(cuv - check_uv) if check_uv is not None else 0.0
                q = Rotation.from_matrix(Rwc @ CV2GL).as_quat()
                cands.append(dict(scale=s, period=pl["period"] * s, z=float(z), pos=pos.tolist(), quat_xyzw_gl=q.tolist(),
                                  fx=fx, size=[W, H], anchor_xyz=A.tolist(), check_uv=cuv.tolist(), yaw_k=k,
                                  check_err=float(err) if np.isfinite(err) else 1e9))
        cands.sort(key=lambda c: c["check_err"])
        if check_uv is None:
            for c in cands:
                print(f"[compose] scale {s:.3f} yaw_k {c['yaw_k']} pos {np.round(c['pos'], 3).tolist()} check -> {np.round(c['check_uv'], 0).tolist()}")
        best = cands[0]
        print(f"[compose] scale {s:.3f} period {best['period']:.4f} plane z {z:.4f} cam {np.round(best['pos'], 4).tolist()} "
              f"check {np.round(best['check_uv'], 0).tolist()} (err {best['check_err']:.0f}px)")
        out.append(best)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
