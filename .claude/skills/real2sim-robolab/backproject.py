"""Turn pixels of a fitted camera into robot-frame positions and object dimensions.

point:  pixel -> intersection with the horizontal plane z (surface height plus any offset).
object: base / top / left / right pixels of an object standing on the plane -> base position (where the
        visible bottom edge touches the plane), height (top pixel on the vertical through the base) and width
        (left-right pixels at that height). For round objects add the radius along the viewing direction to
        get the centre (printed as "centre").

Usage:
  python backproject.py cam.json point --z Z name:u,v [name:u,v ...]
  python backproject.py cam.json object --z Z name:bu,bv:tu,tv:lu,lv:ru,rv [...]
cam.json: {"pos", "quat_xyzw_gl", "fx", "size"} (compose_ext.py candidate, refine output, render.json camera).
"""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam")
    ap.add_argument("mode", choices=["point", "object"])
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    c = json.load(open(a.cam))
    pos, R, fx = np.array(c["pos"]), Rotation.from_quat(c["quat_xyzw_gl"]).as_matrix(), c["fx"]
    W, H = c.get("size", [1280, 720])

    def ray(u, v):
        return R @ [(u - W / 2) / fx, -(v - H / 2) / fx, -1.0]

    def on_z(u, v, z):
        d = ray(u, v)
        return pos + (z - pos[2]) / d[2] * d

    for it in a.items:
        name, *px = it.split(":")
        pts = [tuple(map(float, p.split(","))) for p in px]
        if a.mode == "point":
            X = on_z(*pts[0], a.z)
            print(f"{name:12s} ({pts[0][0]:.0f},{pts[0][1]:.0f}) -> ({X[0]:.4f}, {X[1]:.4f}, {X[2]:.4f})")
            continue
        B = on_z(*pts[0], a.z)
        d = ray(*pts[1])
        A = np.array([[d[0], 0.0], [d[1], 0.0], [d[2], -1.0]])
        s, _ = np.linalg.lstsq(A, np.array([B[0] - pos[0], B[1] - pos[1], -pos[2]]), rcond=None)[0]
        top = (pos + s * d)[2]
        width = np.linalg.norm(on_z(*pts[2], top) - on_z(*pts[3], top))
        view = (B - pos)[:2] / np.linalg.norm((B - pos)[:2])
        centre = B[:2] + view * width / 2
        print(f"{name:12s} base ({B[0]:.4f}, {B[1]:.4f}) centre ({centre[0]:.4f}, {centre[1]:.4f}) "
              f"height {top - a.z:.4f} width {width:.4f}")


if __name__ == "__main__":
    main()
