"""Detect a periodic planar pattern (checked/gingham cloth, floor tiles, pegboard, grid mat) in one image and fit
it: lattice -> image homography, plus a single-view calibration (fx, k1, plane pose in lattice units).

No 3D points or measurements are needed. The pattern's cells must form a square lattice on a plane. Blobs are
the pattern's most distinctive cells (gingham: the darkest, most saturated squares, where both stripes cross),
picked by an HSV box. Indices are assigned by RANSAC over local quads, then grown over the whole image; the
reduced square basis is chosen by single-view calibration RMS (a wrong, sheared basis gives large RMS).

Usage:
  python fit_lattice.py image.png out.json [--hsv H0 H1 S0 S1 V0 V1 | --auto] [--fx-init 600] [--fix-fx F]
      [--roi x0,y0,x1,y1] [--min-area 15] [--max-area 8000] [--hfov-range 50 130]
--fix-fx: use for near-fronto-parallel views (wrist cameras), where focal length is not observable.
--hfov-range: plausible horizontal FOV of the camera; sheared-basis fits land outside it.
Writes out.json {H (lattice->image), lat, uv, basis T, fx, k1, rvec, tvec, rms, inliers} and out_overlay.png
(green = matched blobs, red = predicted lattice). Check the overlay: red dots must sit on every pattern cell.
"""
import argparse
import json

import cv2
import numpy as np

BASES = ([[1, 0], [0, 1]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[1, 0], [-1, 1]], [[1, -1], [0, 1]])


def detect(img, hsv_box, roi, amin, amax):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    h0, h1, s0, s1, v0, v1 = hsv_box
    m = ((hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) & (hsv[..., 1] >= s0) & (hsv[..., 1] <= s1)
         & (hsv[..., 2] >= v0) & (hsv[..., 2] <= v1)).astype(np.uint8)
    x0, y0, x1, y1 = roi
    m[:y0], m[y1:], m[:, :x0], m[:, x1:] = 0, 0, 0, 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if amin < st[i, 4] < amax and 0.3 < st[i, 2] / max(st[i, 3], 1) < 3.5
            and st[i, 4] / (st[i, 2] * st[i, 3]) > 0.45]
    return cen[keep]


def auto_hsv(img):
    """Most saturated third of the dominant hue (the 'both stripes' cells of a gingham)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    sat = hsv[hsv[:, 1] > 40]
    hue = int(np.bincount(sat[:, 0], minlength=180).argmax())
    band = sat[np.abs(sat[:, 0] - hue) < 10]
    return [max(0, hue - 10), hue + 10, int(np.percentile(band[:, 1], 66)), 255, 0, 255]


def grow(pts, H, iters=4, tol=0.28, rng=25):
    I, J = np.meshgrid(np.arange(-rng, rng + 1), np.arange(-rng, rng + 1))
    L = np.stack([I.ravel(), J.ravel()], 1).astype(float)
    Lm = Um = np.zeros((0, 2))
    for _ in range(iters):
        P = cv2.perspectiveTransform(L[None], H)[0]
        sp = np.minimum(np.linalg.norm(cv2.perspectiveTransform((L + [1, 0])[None], H)[0] - P, axis=1),
                        np.linalg.norm(cv2.perspectiveTransform((L + [0, 1])[None], H)[0] - P, axis=1))
        d = np.linalg.norm(P[:, None] - pts[None], axis=-1)
        j = d.argmin(1)
        ok = (d[np.arange(len(L)), j] < tol * sp) & np.isfinite(P).all(1)
        _, first = np.unique(j[ok], return_index=True)
        Lm, Um = L[ok][first], pts[j[ok]][first]
        if len(Lm) < 5:
            break
        H, _ = cv2.findHomography(Lm, Um, 0)
    return H, Lm, Um


def ransac_lattice(pts):
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    best = (0, None)
    for p in range(len(pts)):
        nn = np.argsort(d[p])[:6]
        for qi in nn:
            for ri in nn:
                if qi >= ri:
                    continue
                q, r = pts[qi] - pts[p], pts[ri] - pts[p]
                if abs(q[0] * r[1] - q[1] * r[0]) < 0.3 * np.linalg.norm(q) * np.linalg.norm(r):
                    continue
                s = pts[p] + q + r
                si = int(np.argmin(np.linalg.norm(pts - s, axis=1)))
                if np.linalg.norm(pts[si] - s) > 0.3 * min(np.linalg.norm(q), np.linalg.norm(r)):
                    continue
                H0 = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [0, 1], [1, 1]]),
                                                 np.float32([pts[p], pts[qi], pts[ri], pts[si]]))
                H, Lm, Um = grow(pts, H0.astype(float))
                if len(Lm) > best[0]:
                    best = (len(Lm), (H, Lm, Um))
    return best[1]


def calibrate(L, U, size, fx0, fix_fx, hfov_range=(50, 130)):
    W, Hh = size
    best = None
    for T in BASES:
        M = L @ np.array(T, float).T
        obj = np.c_[M, np.zeros(len(M))].astype(np.float32)
        for free_k1 in (False, True):
            fl = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                  | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
            if not free_k1:
                fl |= cv2.CALIB_FIX_K1
            if fix_fx:
                fl |= cv2.CALIB_FIX_FOCAL_LENGTH
            m = np.ones(len(M), bool)
            try:
                for _ in range(5):
                    K0 = np.array([[fx0, 0, W / 2], [0, fx0, Hh / 2], [0, 0, 1.0]])
                    rms, K, D, rv, tv = cv2.calibrateCamera([obj[m]], [U[m].astype(np.float32)], (W, Hh), K0,
                                                            np.zeros(5), flags=fl)
                    e = np.linalg.norm(cv2.projectPoints(obj, rv[0], tv[0], K, D)[0].reshape(-1, 2) - U, axis=1)
                    m = e < max(2.5 * np.median(e), 3)
            except cv2.error:
                continue
            hfov = np.degrees(2 * np.arctan(W / 2 / K[0, 0]))
            if not fix_fx and not (hfov_range[0] <= hfov <= hfov_range[1]):
                continue
            cand = dict(T=T, fx=float(K[0, 0]), k1=float(D.ravel()[0]), rvec=rv[0].ravel().tolist(),
                        tvec=tv[0].ravel().tolist(), rms=float(rms), inliers=m.tolist(), free_k1=free_k1)
            if best is None or cand["rms"] < best["rms"] - 0.3 or (abs(cand["rms"] - best["rms"]) <= 0.3 and not free_k1 and best["free_k1"]):
                best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--hsv", type=int, nargs=6)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--roi", default=None)
    ap.add_argument("--min-area", type=float, default=15)
    ap.add_argument("--max-area", type=float, default=8000)
    ap.add_argument("--fx-init", type=float, default=600)
    ap.add_argument("--fix-fx", type=float)
    ap.add_argument("--hfov-range", type=float, nargs=2, default=[50, 130])
    a = ap.parse_args()
    img = cv2.imread(a.image)
    Hh, W = img.shape[:2]
    roi = [int(v) for v in a.roi.split(",")] if a.roi else [0, 0, W, Hh]
    box = a.hsv if a.hsv else auto_hsv(img)
    pts = detect(img, box, roi, a.min_area, a.max_area)
    H, L, U = ransac_lattice(pts)
    e = np.linalg.norm(cv2.perspectiveTransform(L[None], H)[0] - U, axis=1)
    print(f"[lattice] hsv {box} blobs {len(pts)} matched {len(L)} homography rms {np.sqrt((e ** 2).mean()):.2f}px")
    cal = calibrate(L, U, (W, Hh), a.fix_fx or a.fx_init, a.fix_fx is not None, a.hfov_range)
    if cal:
        print(f"[lattice] basis {cal['T']} fx {cal['fx']:.1f} (hfov {np.degrees(2 * np.arctan(W / 2 / cal['fx'])):.1f}) "
              f"k1 {cal['k1']:.3f} calib rms {cal['rms']:.2f}px inliers {sum(cal['inliers'])}/{len(L)}")
    out = dict(image=a.image, size=[W, Hh], hsv=box, H=H.tolist(), lat=L.tolist(), uv=U.tolist(), **(cal or {}))
    json.dump(out, open(a.out, "w"))
    ov = img.copy()
    for u, v in U:
        cv2.circle(ov, (int(u), int(v)), 7, (0, 255, 0), 2)
    for i in range(-25, 26):
        for j in range(-25, 26):
            p = cv2.perspectiveTransform(np.array([[[i, j]]], float), H)[0, 0]
            if 0 <= p[0] < W and 0 <= p[1] < Hh:
                cv2.circle(ov, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    cv2.imwrite(a.out.replace(".json", "_overlay.png"), ov)


if __name__ == "__main__":
    main()
