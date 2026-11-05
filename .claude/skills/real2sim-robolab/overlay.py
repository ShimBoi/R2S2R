"""Score how well a sim render lines up with the real camera image, and write visual diffs.

Compares edges only inside --score polygons (static structure that exists in both: surface edges and
corners, the robot base, textured surfaces, fixed fixtures). Leave out the arm, the movable objects and
anything the dome background provides, since those are not expected to match pixel for pixel.

Reports the residual similarity transform sim -> real (shift at image centre, scale, rotation) from
ECC on blurred edge maps, and the symmetric edge chamfer distance. All pixel numbers are normalized
to a 1280-px-wide image so thresholds do not depend on resolution.

Usage:
  python overlay.py real.png sim.png out_prefix [--score polys.json] [--max-shift 6.4 --max-scale 0.01
         --max-rot 0.3 --max-chamfer 4]
polys.json: [[[u, v], [u, v], ...], ...] in real-image pixels.
Writes out_prefix_blend.png, out_prefix_edges.png (real magenta, sim green), out_prefix_checker.png.
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import math
import sys

import cv2
import numpy as np


def edges(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    m = float(np.median(g))
    return cv2.Canny(g, int(max(0, 0.66 * m)), int(min(255, 1.33 * m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real")
    ap.add_argument("sim")
    ap.add_argument("out")
    ap.add_argument("--score")
    ap.add_argument("--max-shift", type=float, default=6.4)
    ap.add_argument("--max-scale", type=float, default=0.01)
    ap.add_argument("--max-rot", type=float, default=0.3)
    ap.add_argument("--max-chamfer", type=float, default=4.0)
    a = ap.parse_args()
    real, sim = cv2.imread(a.real), cv2.imread(a.sim)
    H, W = real.shape[:2]
    if abs(sim.shape[1] / sim.shape[0] - W / H) > 0.01:
        print(f"FAIL aspect ratio differs: real {W}x{H}, sim {sim.shape[1]}x{sim.shape[0]}")
        sys.exit(1)
    sim = cv2.resize(sim, (W, H), interpolation=cv2.INTER_AREA if sim.shape[1] > W else cv2.INTER_CUBIC)
    k = 1280.0 / W
    mask = np.zeros((H, W), np.uint8)
    if a.score:
        for poly in json.load(open(a.score)):
            cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    else:
        mask[:] = 255
    er, es = edges(real) & mask, edges(sim) & mask

    cv2.imwrite(a.out + "_blend.png", cv2.addWeighted(real, 0.5, sim, 0.5, 0))
    vis = (real * 0.35).astype(np.uint8)
    vis[er > 0] = (255, 0, 255)
    vis[es > 0] = (0, 255, 0)
    vis[(er > 0) & (es > 0)] = (255, 255, 255)
    cv2.polylines(vis, [np.array(p, np.int32) for p in (json.load(open(a.score)) if a.score else [])], True, (0, 200, 255), 1)
    cv2.imwrite(a.out + "_edges.png", vis)
    s = max(8, W // 32)
    yy, xx = np.mgrid[0:H, 0:W]
    chk = ((yy // s + xx // s) % 2 == 0)[..., None]
    cv2.imwrite(a.out + "_checker.png", np.where(chk, real, sim))

    dr = cv2.distanceTransform(255 - er, cv2.DIST_L2, 3)
    ds = cv2.distanceTransform(255 - es, cv2.DIST_L2, 3)
    if er.sum() == 0 or es.sum() == 0:
        print("FAIL no edges inside the score region")
        sys.exit(1)
    cham = 0.5 * (np.median(dr[es > 0]) + np.median(ds[er > 0])) * k

    blur = lambda e: cv2.GaussianBlur(e.astype(np.float32) / 255, (0, 0), max(2.0, W / 400))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(blur(er), blur(es), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
        A, t = warp[:, :2].astype(float), warp[:, 2].astype(float)
        c = np.array([W / 2, H / 2])
        shift = (A @ c + t - c) * k
        scale = math.sqrt(abs(np.linalg.det(A))) - 1
        rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    except cv2.error as e:
        print("ECC did not converge:", str(e).splitlines()[-1])
        shift, scale, rot = np.array([np.inf, np.inf]), np.inf, np.inf

    checks = [("shift_px@1280", float(np.hypot(*shift)), a.max_shift), ("scale", abs(scale), a.max_scale),
              ("rot_deg", abs(rot), a.max_rot), ("chamfer_px@1280", float(cham), a.max_chamfer)]
    ok = True
    for n, v, lim in checks:
        ok &= v <= lim
        print(f"{n:16s} {v:8.3f}  limit {lim:6.3f}  {'ok' if v <= lim else 'FAIL'}")
    print(f"shift (dx, dy) = ({shift[0]:+.2f}, {shift[1]:+.2f}) px@1280: positive dx means sim content sits right of the real content")
    m = mask > 0
    mr, ms = real[m].reshape(-1, 3).mean(0)[::-1], sim[m].reshape(-1, 3).mean(0)[::-1]
    print(f"mean RGB in score region: real {mr.round(1).tolist()} sim {ms.round(1).tolist()} "
          f"gain real/sim {(mr / np.maximum(ms, 1)).round(3).tolist()} (exposure / white balance, informational)")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
