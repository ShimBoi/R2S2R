"""Camera math shared by the real2sim-robolab helpers.

Frames: robot base frame (+x forward, +y left, +z up, origin at the robot base = env origin).
Isaac Lab 3 quaternions are (x, y, z, w). Camera OffsetCfg uses convention="opengl" (looks along -Z, +Y up).
OpenCV / PnP uses the "cv" camera frame (looks along +Z, +Y down).

CLI:
  python cam_math.py lookat --pos x y z --target x y z [--roll 0] [--hfov 101.9 --size 1280 720]
  python cam_math.py cfg --pos x y z --quat x y z w [--hfov H | --fx F] --size W H
  python cam_math.py describe --quat x y z w
"""
import argparse
import math

import numpy as np

CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q if w >= 0 else -q


def lookat_opengl(pos, target, roll_deg=0.0):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    r, u = c * r + s * u, -s * r + c * u
    return R_to_quat_xyzw(np.stack([r, u, -f], 1))


def gl_quat_to_cv_R(q):
    """Robot-from-camera rotation in the cv camera frame, from an opengl OffsetCfg quaternion."""
    return quat_xyzw_to_R(q) @ CV_TO_GL


def cv_R_to_gl_quat(R_robot_cv):
    return R_to_quat_xyzw(R_robot_cv @ CV_TO_GL)


def K_from_fx(fx, W, H, cx=None, cy=None):
    return np.array([[fx, 0, W / 2 if cx is None else cx], [0, fx, H / 2 if cy is None else cy], [0, 0, 1.0]])


def fx_from_hfov(hfov_deg, W):
    return W / 2 / math.tan(math.radians(hfov_deg) / 2)


def hfov_from_fx(fx, W):
    return 2 * math.degrees(math.atan(W / 2 / fx))


def pinhole_cfg(fx, W, H, focal_length=None):
    """PinholeCameraCfg fields that give fx at render width W (Isaac: fx = W * focal_length / horizontal_aperture)."""
    focal_length = focal_length or 2.1
    ha = W * focal_length / fx
    return dict(focal_length=round(focal_length, 4), horizontal_aperture=round(ha, 4), vertical_aperture=round(ha * H / W, 4))


def project(xyz, pos, quat_gl, K):
    """Project robot-frame points (N,3) to pixels with an opengl-convention camera pose."""
    R = gl_quat_to_cv_R(quat_gl)
    pc = (np.asarray(xyz, float) - np.asarray(pos, float)) @ R
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def describe(q):
    R = quat_xyzw_to_R(q)
    f = -R[:, 2]
    return (f"fwd={np.round(f, 3)} yaw={np.degrees(np.arctan2(f[1], f[0])):.1f} "
            f"pitch={np.degrees(np.arcsin(f[2])):.1f} up={np.round(R[:, 1], 3)}")


def cfg_snippet(name, pos, q, fx, W, H):
    p = pinhole_cfg(fx, W, H)
    return f'''@configclass
class {name}:
    over_shoulder_left_camera = TiledCameraCfg(
        prim_path="{{ENV_REGEX_NS}}/over_shoulder_left_camera",
        height={H},
        width={W},
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length={p["focal_length"]},
            focus_distance=28.0,
            horizontal_aperture={p["horizontal_aperture"]},
            vertical_aperture={p["vertical_aperture"]},
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), rot=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}), convention="opengl"
        ),
    )
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lookat", "cfg", "describe"])
    ap.add_argument("--pos", type=float, nargs=3)
    ap.add_argument("--target", type=float, nargs=3)
    ap.add_argument("--quat", type=float, nargs=4)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--fx", type=float)
    ap.add_argument("--size", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--name", default="MyExternalCameraCfg")
    a = ap.parse_args()
    q = np.array(a.quat) if a.quat else None
    if a.cmd == "lookat":
        q = lookat_opengl(a.pos, a.target, a.roll)
        print("rot (xyzw, opengl):", np.round(q, 4).tolist(), describe(q))
    if a.cmd == "describe":
        print(describe(q))
        return
    W, H = a.size
    fx = a.fx or (fx_from_hfov(a.hfov, W) if a.hfov else None)
    if fx:
        print(f"fx={fx:.1f}px hfov={hfov_from_fx(fx, W):.2f}deg vfov={2 * math.degrees(math.atan(H / 2 / fx)):.2f}deg")
        print(cfg_snippet(a.name, a.pos, q, fx, W, H))


if __name__ == "__main__":
    main()
