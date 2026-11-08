"""Score one scene version against the real policy-camera images and decide PASS / FAIL. Use it after every
change (camera, scale, geometry, objects, lighting) and keep iterating versions until it passes.

Inputs are the real images, a labels file you write once per scene, and a render_check.py output directory.
Checks (thresholds overridable in labels.json "limits"; pixels normalized to 1280 px width):

  robot_iou      ext robot silhouette IoU vs the real robot mask                       >= 0.65
  obj_px         per object: sim mask centroid vs labelled real centroid             <= 15 px ext, 25 px wrist
  obj_iou        per object labelled with a polygon: visible-region IoU               >= 0.6
  static_chamfer edge chamfer inside static polygons (surface edges, fixtures)        <= 6 px
  color_gain     real/sim mean RGB per region, every channel within                   +-10 % ext, +-25 % wrist
  pattern_scale  ratio of the dominant pattern period real/sim in a box (if "pattern")  within +-5 %

labels.json:
{"ext":   {"image": "ext.png", "robot_mask": "robot_mask.png", "objects": {"obj_a": [722, 390], ...},
           "static_polys": [[[u, v], ...], ...], "color_regions": [[x0, y0, x1, y1], ...]},
 "wrist": {"image": "wrist.png", "objects": {"obj_a": [480, 300]}, "color_regions": [...], "pattern": true},
 "limits": {"robot_iou": 0.65, ...}}
Paths are relative to labels.json. An object label is either a point (centre of the visible body, by eye) or a
polygon [[u, v], ...] outlining the visible part; prefer polygons for large or partly occluded objects, since a
point and a mask centroid only agree if both cover the same visible region.

Usage:
  python check_version.py labels.json render_dir --name v6 [--log versions.md]
Writes render_dir/check.json, render_dir/sheet.jpg (real | sim | edge overlay with object markers, per camera)
and appends one line to the versions log. Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlay import edges

LIMITS = dict(robot_iou=0.65, obj_iou=0.6, obj_px_ext=15.0, obj_px_wrist=25.0, static_chamfer=6.0, gain_ext=0.10,
              gain_wrist=0.25, pattern_scale=0.05)


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) else None


def spectrum(img, box, pad=4):
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    n = pad * max(g.shape)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g, s=(n, n))))
    f = np.fft.fftshift(np.fft.fftfreq(n))
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.hypot(fy, fx)
    F[(r < 3.0 / min(g.shape)) | (fy < 0)] = 0
    return F, fy, fx, r


def spacing(real, sim, box):
    """Dominant pattern period (px) in real, and sim's period along the same direction (+-15 deg)."""
    Fr, fy, fx, r = spectrum(real, box)
    i = np.unravel_index(Fr.argmax(), Fr.shape)
    ang = np.arctan2(fy[i], fx[i])
    Fs, _, _, _ = spectrum(sim, box)
    d = np.abs(np.angle(np.exp(1j * (np.arctan2(fy, fx) - ang))))
    Fs[np.minimum(d, np.pi - d) > np.radians(15)] = 0
    j = np.unravel_index(Fs.argmax(), Fs.shape)
    return float(1 / r[i]), float(1 / r[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("render_dir")
    ap.add_argument("--name", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    lab = json.load(open(a.labels))
    base = os.path.dirname(os.path.abspath(a.labels))
    lim = dict(LIMITS, **lab.get("limits", {}))
    name = a.name or os.path.basename(os.path.normpath(a.render_dir))
    results, rows = [], []

    def add(check, cam, value, limit, ok, detail=""):
        results.append(dict(check=check, cam=cam, value=None if value is None else round(float(value), 3),
                            limit=limit, ok=bool(ok), detail=detail))

    for cam in ("ext", "wrist"):
        if cam not in lab:
            continue
        L = lab[cam]
        real = cv2.imread(os.path.join(base, L["image"]))
        sim = cv2.imread(os.path.join(a.render_dir, f"{cam}.png"))
        H, W = real.shape[:2]
        sim = cv2.resize(sim, (W, H))
        k = 1280.0 / W
        if L.get("robot_mask"):
            rm = cv2.resize(cv2.imread(os.path.join(base, L["robot_mask"]), 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127
            sm = cv2.resize(cv2.imread(os.path.join(a.render_dir, "masks", f"{cam}_robot.png"), 0), (W, H)) > 127
            iou = (rm & sm).sum() / max((rm | sm).sum(), 1)
            add("robot_iou", cam, iou, lim["robot_iou"], iou >= lim["robot_iou"])
        sim_pts = {}
        for obj, uv in L.get("objects", {}).items():
            p = os.path.join(a.render_dir, "masks", f"{cam}_{obj}.png")
            sm = cv2.resize(cv2.imread(p, 0), (W, H)) > 127 if os.path.exists(p) else None
            c = centroid(sm) if sm is not None else None
            limit = lim[f"obj_px_{cam}"]
            if c is None:
                add("obj_px", cam, None, limit, False, f"{obj}: not visible in sim")
                continue
            sim_pts[obj] = c
            if isinstance(uv[0], (list, tuple)):
                rmask = np.zeros((H, W), np.uint8)
                cv2.fillPoly(rmask, [np.array(uv, np.int32)], 255)
                rmask = rmask > 0
                iou = (rmask & sm).sum() / max((rmask | sm).sum(), 1)
                add("obj_iou", cam, iou, lim["obj_iou"], iou >= lim["obj_iou"], f"{obj}")
                uv = np.round(centroid(rmask)).astype(int).tolist()
                L["objects"][obj] = uv
            e = np.linalg.norm(c - np.array(uv, float)) * k
            add("obj_px", cam, e, limit, e <= limit, f"{obj}: sim {np.round(c).astype(int).tolist()} real {uv}")
        if L.get("static_polys"):
            reg = np.zeros((H, W), np.uint8)
            for poly in L["static_polys"]:
                cv2.fillPoly(reg, [np.array(poly, np.int32)], 255)
            er, es = edges(real) & reg, edges(sim) & reg
            if er.sum() and es.sum():
                dr = cv2.distanceTransform(255 - er, cv2.DIST_L2, 3)
                ds = cv2.distanceTransform(255 - es, cv2.DIST_L2, 3)
                ch = 0.5 * (np.minimum(dr[es > 0], 50).mean() + np.minimum(ds[er > 0], 50).mean()) * k
                add("static_chamfer", cam, ch, lim["static_chamfer"], ch <= lim["static_chamfer"])
        for (x0, y0, x1, y1) in L.get("color_regions", []):
            g = real[y0:y1, x0:x1].reshape(-1, 3).mean(0) / np.maximum(sim[y0:y1, x0:x1].reshape(-1, 3).mean(0), 1)
            dev = np.abs(g - 1).max()
            add("color_gain", cam, dev, lim[f"gain_{cam}"], dev <= lim[f"gain_{cam}"],
                f"region {[x0, y0, x1, y1]} gain BGR {np.round(g, 3).tolist()}")
        if L.get("pattern"):
            box = L["pattern"] if isinstance(L["pattern"], list) else [0, 0, W, int(H * 0.6)]
            sr, ss = spacing(real, sim, box)
            if sr and ss:
                d = abs(sr / ss - 1)
                add("pattern_scale", cam, d, lim["pattern_scale"], d <= lim["pattern_scale"], f"real {sr:.1f}px sim {ss:.1f}px")
        ov = (cv2.cvtColor(cv2.cvtColor(real, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.45).astype(np.uint8)
        ov[edges(real) > 0] = (255, 0, 255)
        ov[edges(sim) > 0] = (0, 255, 0)
        for obj, uv in L.get("objects", {}).items():
            cv2.drawMarker(ov, tuple(int(v) for v in uv), (255, 0, 255), cv2.MARKER_CROSS, 24, 3)
            if obj in sim_pts:
                cv2.drawMarker(ov, tuple(int(v) for v in sim_pts[obj]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 3)
        row = np.hstack([real, sim, ov])
        for i, t in enumerate(("REAL " + cam, f"SIM {name}", "edges real magenta / sim green")):
            cv2.putText(row, t, (20 + W * i, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        rows.append(cv2.resize(row, (3 * 640, int(640 * H / W))))
    ok = all(r["ok"] for r in results)
    for r in results:
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['check']:15s} {r['cam']:6s} {str(r['value']):>8s}  limit {r['limit']}  {r['detail']}")
    print("PASS" if ok else "FAIL")
    json.dump(dict(name=name, passed=ok, results=results), open(os.path.join(a.render_dir, "check.json"), "w"), indent=1)
    cv2.imwrite(os.path.join(a.render_dir, "sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if a.log:
        fails = [f"{r['check']}/{r['cam']}" + (f" ({r['detail'].split(':')[0]})" if r["check"] == "obj_px" else "")
                 for r in results if not r["ok"]]
        with open(a.log, "a") as f:
            f.write(f"| {name} | {'PASS' if ok else 'FAIL'} | {sum(r['ok'] for r in results)}/{len(results)} | "
                    f"{', '.join(fails) or '-'} |\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
