"""Cut a metric, top-down texture of any planar surface (work surface, mat, cloth, tray, wall panel) from a photo.

Use whichever photo (policy camera or supplementary) sees the surface most frontally and sharply.
Give the pixel corners of a known rectangle on the surface (in order: the corner that maps to texture
(0, 0), then along +u, then the opposite corner, then along +v) and its metric size. Output is
px_per_m * size pixels, resampled with Lanczos straight from the photo (no blur, no inpainting).
With --tile N M, also writes a seamless crop of N x M pattern periods for a repeating texture.

Usage:
  python rectify.py photo.jpg out.png --quad u0,v0 u1,v1 u2,v2 u3,v3 --size-m 0.70 0.65 [--px-per-m 1500]
         [--tile 4 4 --period-m 0.06 0.06]
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out")
    ap.add_argument("--quad", nargs=4, required=True)
    ap.add_argument("--size-m", type=float, nargs=2, required=True)
    ap.add_argument("--px-per-m", type=float, default=1500)
    ap.add_argument("--tile", type=int, nargs=2)
    ap.add_argument("--period-m", type=float, nargs=2)
    a = ap.parse_args()
    img = cv2.imread(a.photo)
    src = np.array([[float(v) for v in p.split(",")] for p in a.quad], np.float32)
    w, h = int(round(a.size_m[0] * a.px_per_m)), int(round(a.size_m[1] * a.px_per_m))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    rect = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(a.out, rect)
    area = abs(cv2.contourArea(src))
    print(f"wrote {a.out} {w}x{h}; source pixels per output pixel {area / (w * h):.2f} "
          "(< 1 means the photo is upsampled there: lower --px-per-m or use a closer photo)")
    if a.tile:
        pw, ph = int(round(a.period_m[0] * a.px_per_m)), int(round(a.period_m[1] * a.px_per_m))
        cx, cy = w // 2 - pw * a.tile[0] // 2, h // 2 - ph * a.tile[1] // 2
        tile = rect[cy:cy + ph * a.tile[1], cx:cx + pw * a.tile[0]]
        tp = a.out.rsplit(".", 1)[0] + "_tile.png"
        cv2.imwrite(tp, tile)
        print(f"wrote {tp} {tile.shape[1]}x{tile.shape[0]}: UV period {a.period_m[0] * a.tile[0]:.4f} x "
              f"{a.period_m[1] * a.tile[1]:.4f} m (use as material period_m in the gen_scene.py spec)")


if __name__ == "__main__":
    main()
