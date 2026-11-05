"""Colour-match a texture taken from a supplementary photo to how the same surface looks in a policy camera.

Supplementary photos (phone, other cameras) usually differ in white balance, exposure and tone curve.
The policy-camera image is the appearance ground truth, so fit a per-channel affine map (gain, offset)
from the source region to the reference region and apply it to the whole texture. Regions should show
the same material under similar lighting (avoid shadows and specular highlights in either).

Usage:
  python color_match.py texture.png policy_cam.png out.png --src-region src_polys.json --ref-region ref_polys.json
polys: [[[u, v], ...], ...] in the pixels of the respective image. Without a region the whole image is used.
"""
import argparse
import json

import cv2
import numpy as np


def region(img, path):
    m = np.zeros(img.shape[:2], np.uint8)
    if path:
        for poly in json.load(open(path)):
            cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    else:
        m[:] = 255
    return img[m > 0].reshape(-1, 3).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texture")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("--src-region")
    ap.add_argument("--ref-region")
    a = ap.parse_args()
    tex, ref = cv2.imread(a.texture), cv2.imread(a.reference)
    s, r = region(tex, a.src_region), region(ref, a.ref_region)
    gain = r.std(0) / np.maximum(s.std(0), 1e-6)
    off = r.mean(0) - gain * s.mean(0)
    out = np.clip(tex.astype(np.float64) * gain + off, 0, 255).astype(np.uint8)
    cv2.imwrite(a.out, out)
    print("BGR gain", gain.round(3).tolist(), "offset", off.round(1).tolist(),
          "| src mean", s.mean(0).round(1).tolist(), "ref mean", r.mean(0).round(1).tolist())


if __name__ == "__main__":
    main()
