"""Initial camera pose (and optionally focal length) in the robot base frame from one policy-camera image.
No calibration pattern is needed: any >= 4 labelled 2D-3D correspondences work.

pnp (default): points with known robot-frame positions, e.g. user-measured fixture corners, the robot base,
     tape marks, objects of known size, and robot body origins at the recorded joint state
     (refine_camera.py --dump-keypoints writes them; reference a body with "body" instead of "xyz").
     Follow with refine_camera.py (render-and-compare on the robot) for the final pose.
vp (optional): only when a straight-line grid lies on a plane (floor tiles, a checked cloth, a pegboard):
     rotation (and focal) from its two orthogonal vanishing points, translation from >= 2 labelled points.

points.json:
 {"image": "ext.png",
  "intrinsics": {"fx":..,"fy":..,"cx":..,"cy":.., "dist": [k1,k2,p1,p2,k3]} | {"K": [9 values], "dist": [...]}
                | {"hfov_deg":..} | {"focal_mm":.., "sensor_width_mm":..}   (omit to estimate fx with --fit-focal)
  "points": [{"name": "base_front", "uv": [u, v], "xyz": [x, y, z]},
             {"name": "elbow", "uv": [u, v], "body": "panda_link4"}, ...]}
uv are pixels of the image at its native resolution; xyz are metres in the robot base frame.
If "dist" is given the points are undistorted first (the sim camera is then an ideal pinhole).

Usage:
  python fit_camera.py pnp points.json out_prefix [--fit-focal] [--keypoints kp.json] [--plane X0 X1 Y0 Y1 Z]
  python fit_camera.py vp points.json out_prefix --roi x0,y0,x1,y1 [--axes auto|0+,1-] [--plane ...]
Writes out_prefix.json (pos, quat_xyzw_gl, fx, size, rms_px) and out_prefix_overlay.png.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cam_math import K_from_fx, cfg_snippet, cv_R_to_gl_quat, describe, fx_from_hfov, hfov_from_fx, project


def load(path, keypoints=None):
    spec = json.load(open(path))
    img = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(path)), spec["image"]))
    H, W = img.shape[:2]
    kp = json.load(open(keypoints)) if keypoints else {}
    uv = np.array([p["uv"] for p in spec["points"]], float)
    xyz = np.array([p["xyz"] if "xyz" in p else kp[p["body"]] for p in spec["points"]], float)
    names = [p["name"] for p in spec["points"]]
    intr = spec.get("intrinsics", {})
    K = None
    if "K" in intr:
        K = np.array(intr["K"], float).reshape(3, 3)
    elif "fx" in intr:
        K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]])
    elif "hfov_deg" in intr:
        K = K_from_fx(fx_from_hfov(intr["hfov_deg"], W), W, H)
    elif "focal_mm" in intr:
        K = K_from_fx(intr["focal_mm"] / intr["sensor_width_mm"] * W, W, H)
    if K is not None and intr.get("dist"):
        uv = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, np.array(intr["dist"], float), P=K).reshape(-1, 2)
        img = cv2.undistort(img, K, np.array(intr["dist"], float))
    return img, W, H, uv, xyz, names, K


def pnp(uv, xyz, K):
    ok, rv, tv = cv2.solvePnP(xyz, uv, K, None, flags=cv2.SOLVEPNP_SQPNP if len(uv) >= 3 else cv2.SOLVEPNP_ITERATIVE)
    rv, tv = cv2.solvePnPRefineLM(xyz, uv, K, None, rv, tv)
    R_cr = cv2.Rodrigues(rv)[0]
    return R_cr.T, (-R_cr.T @ tv).ravel()


def rms(uv, xyz, pos, q, K):
    pr, _ = project(xyz, pos, q, K)
    e = np.linalg.norm(pr - uv, axis=1)
    return float(np.sqrt((e ** 2).mean())), e


def segments(img, roi, min_len):
    H, W = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    seg = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0].reshape(-1, 4)
    x0, y0, x1, y1 = np.array(roi) * [W, H, W, H]
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    L = np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1)
    keep = (L > min_len) & (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)
    return seg[keep]


def ransac_vp(sg, iters=3000, tol_deg=1.0, rng=np.random.default_rng(0)):
    p = np.c_[sg[:, :2], np.ones(len(sg))]
    q = np.c_[sg[:, 2:], np.ones(len(sg))]
    L = np.cross(p, q)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    mids = (sg[:, :2] + sg[:, 2:]) / 2
    dirs = sg[:, 2:] - sg[:, :2]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(sg), 2, replace=False)
        v = np.cross(L[i], L[j])
        to = v[:2] - mids * v[2]
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
        inl = np.degrees(np.arccos(np.clip(np.abs((to * dirs).sum(1)), 0, 1))) < tol_deg
        if best is None or inl.sum() > best.sum():
            best = inl
    v = np.linalg.svd(L[best])[2][-1]
    return v, best


def vp_rotation(img, roi, axes, K, min_len):
    H, W = img.shape[:2]
    seg = segments(img, roi, min_len)
    v1, in1 = ransac_vp(seg)
    v2, in2 = ransac_vp(seg[~in1])
    P = [v1[:2] / v1[2], v2[:2] / v2[2]]
    print("segments", len(seg), "VP inliers", int(in1.sum()), int(in2.sum()), "VPs", np.round(P, 1).tolist())
    c = np.array([W / 2, H / 2])
    f2 = -np.dot(P[0] - c, P[1] - c)
    f_vp = float(np.sqrt(f2)) if f2 > 0 else float("nan")
    print(f"focal from VPs: {f_vp:.1f}px (hfov {hfov_from_fx(f_vp, W) if f2 > 0 else float('nan'):.1f}deg)")
    if K is None:
        K = K_from_fx(f_vp, W, H)
    Ki = np.linalg.inv(K)
    d = [Ki @ np.r_[p, 1] for p in P]
    d = [x / np.linalg.norm(x) for x in d]
    d[1] = d[1] - d[0] * np.dot(d[1], d[0])
    d[1] /= np.linalg.norm(d[1])
    combos = [axes] if axes != "auto" else [f"{i}{a},{1 - i}{b}" for i in (0, 1) for a in "+-" for b in "+-"]
    cands = []
    for c in combos:
        (ix, sx), (iy, sy) = [(int(t[0]), 1.0 if t[1] == "+" else -1.0) for t in c.split(",")]
        xr, yr = sx * d[ix], sy * d[iy]
        zr = np.cross(xr, yr)
        if zr[1] < 0:
            cands.append((c, np.stack([xr, yr, zr])))
    if not cands:
        raise SystemExit("axes mapping gives robot +z pointing down in the image; flip one sign in --axes")
    vis = img.copy()
    for sg_, col in ((seg[in1], (0, 0, 255)), (seg[~in1][in2], (255, 0, 0))):
        for a in sg_.astype(int):
            cv2.line(vis, tuple(a[:2]), tuple(a[2:]), col, 2)
    return cands, K, f_vp, vis


def translation_given_R(R_robot_cv, uv, xyz, K):
    R_cr = R_robot_cv.T
    Ki = np.linalg.inv(K)
    A, b = [], []
    for (u, v), X in zip(uv, xyz):
        r = Ki @ np.array([u, v, 1.0])
        rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        A.append(rx)
        b.append(rx @ (R_cr @ X))
    bc = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    return R_robot_cv @ bc


def draw(img, uv, xyz, names, pos, q, K, plane):
    vis = img.copy()
    pr, _ = project(xyz, pos, q, K)
    for (a, b), (c, d), n in zip(uv, pr, names):
        cv2.circle(vis, (int(a), int(b)), 7, (0, 0, 255), 2)
        cv2.drawMarker(vis, (int(c), int(d)), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, n, (int(a) + 8, int(b) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if plane:
        x0, x1, y0, y1, z = plane
        for t in np.linspace(0, 1, 11):
            for p0, p1 in (((x0 + t * (x1 - x0), y0, z), (x0 + t * (x1 - x0), y1, z)),
                           ((x0, y0 + t * (y1 - y0), z), (x1, y0 + t * (y1 - y0), z))):
                pts, depth = project(np.array([p0, p1]), pos, q, K)
                if (depth > 0).all():
                    cv2.line(vis, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), (0, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pnp", "vp"])
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--fit-focal", action="store_true", help="pnp: search fx (square pixels, centred principal point)")
    ap.add_argument("--roi", default="0,0,1,1", help="vp: normalized x0,y0,x1,y1 containing the planar grid")
    ap.add_argument("--axes", default="auto", help="vp: e.g. 0+,1- = VP0 is robot +x, -VP1 is robot +y; auto = lowest rms")
    ap.add_argument("--min-len", type=float, default=25.0)
    ap.add_argument("--plane", type=float, nargs=5, help="X0 X1 Y0 Y1 Z of a horizontal surface, drawn as a grid")
    ap.add_argument("--keypoints", help="robot body origins from refine_camera.py --dump-keypoints")
    a = ap.parse_args()
    img, W, H, uv, xyz, names, K = load(a.points, a.keypoints)
    vp_vis = None
    if a.mode == "pnp":
        if a.fit_focal or K is None:
            best = None
            for fx in np.linspace(0.3 * W, 2.0 * W, 341):
                Kc = K_from_fx(fx, W, H)
                R, pos = pnp(uv, xyz, Kc)
                e = rms(uv, xyz, pos, cv_R_to_gl_quat(R), Kc)[0]
                if best is None or e < best[0]:
                    best = (e, Kc)
            K = best[1]
        R, pos = pnp(uv, xyz, K)
    else:
        cands, K, _, vp_vis = vp_rotation(img, [float(v) for v in a.roi.split(",")], a.axes, K, a.min_len)
        fits = []
        for name, Rc in cands:
            pc = translation_given_R(Rc, uv, xyz, K)
            fits.append((rms(uv, xyz, pc, cv_R_to_gl_quat(Rc), K)[0], name, Rc, pc))
            print(f"axes {name}: rms {fits[-1][0]:.2f}px")
        _, name, R, pos = min(fits, key=lambda f: f[0])
        print("chose axes", name)
    q = cv_R_to_gl_quat(R)
    err, per = rms(uv, xyz, pos, q, K)
    fx = float((K[0, 0] + K[1, 1]) / 2)
    out = dict(pos=pos.round(4).tolist(), quat_xyzw_gl=q.round(5).tolist(), fx=round(fx, 2), size=[W, H],
               cx=float(K[0, 2]), cy=float(K[1, 2]), hfov_deg=round(hfov_from_fx(fx, W), 3), rms_px=round(err, 2),
               per_point={n: round(float(e), 2) for n, e in zip(names, per)})
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    cv2.imwrite(a.out + "_overlay.png", draw(img, uv, xyz, names, pos, q, K, a.plane))
    if vp_vis is not None:
        cv2.imwrite(a.out + "_vp_lines.png", vp_vis)
    print(json.dumps(out, indent=1))
    print(describe(q))
    off_c = np.hypot(K[0, 2] - W / 2, K[1, 2] - H / 2)
    if off_c > 0.01 * W:
        print(f"WARNING principal point is {off_c:.1f}px off-centre; Isaac pinhole is centred unless you use "
              "OpenCvPinholeDistortionCfg or horizontal/vertical_aperture_offset")
    print(cfg_snippet("MyExternalCameraCfg", pos, q, fx, W, H))


if __name__ == "__main__":
    main()
