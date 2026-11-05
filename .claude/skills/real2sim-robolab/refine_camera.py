"""Render-and-compare camera refinement, using the robot itself as the calibration target.
Run in your Isaac Lab environment on a GPU (see reference.md, "Running").

The sim robot is posed at the joint angles recorded with the real image. The camera pose (and optionally
fx) is optimized with Nelder-Mead so that the sim robot silhouette and edges line up with the real image.
The sim robot mask comes from rendering with and without the robot's meshes visible, so no segmentation
sensor is needed. Works for any camera that sees part of the robot: external cameras (no planar pattern
needed) and the wrist camera (it sees the gripper fingers).

Usage:
  python refine_camera.py --task-file t.py --task T --camera ext|wrist --real real.png --joint-pos j1..j7
      [--gripper 0|1] [--real-mask robot_mask.png] [--score static_polys.json] [--init fit.json]
      [--fx F_at_real_width | --hfov DEG] [--fit-focal] [--iters 200] [--render-width 640]
      [--object-poses poses.json] [--preset WRIST_LAB] --out prefix
  python refine_camera.py ... --dump-keypoints kp.json   (robot body origins at --joint-pos, then exit)
--real-mask: white = robot/gripper pixels in the real image (hand-drawn polygon fill or a segmenter).
--score:     extra static regions (real-image pixels) whose edges must also align (surface edges, fixtures).
--init:      fit_camera.py output for ext; without it the current camera cfg pose is the start.
Writes prefix.json (pos, quat_xyzw_gl, fx, size; for wrist also offset_pos / offset_rot on the gripper
base_link), prefix_best.png and prefix_edges.png.
"""
import argparse
import json
import math
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--preset", default="WRIST_LAB")
    ap.add_argument("--camera", choices=["ext", "wrist"], default="ext")
    ap.add_argument("--real")
    ap.add_argument("--real-mask")
    ap.add_argument("--score")
    ap.add_argument("--joint-pos", type=float, nargs=7, required=True)
    ap.add_argument("--gripper", type=float, default=0.0)
    ap.add_argument("--init")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--hfov", type=float)
    ap.add_argument("--fit-focal", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--render-width", type=int, default=640)
    ap.add_argument("--object-poses")
    ap.add_argument("--dump-keypoints")
    ap.add_argument("--gripper-link", default="base_link")
    ap.add_argument("--out", default="refine")
    a = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, enable_cameras=True)

    import cv2
    import numpy as np
    import omni.usd
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from pxr import Usd, UsdGeom
    from robolab.core.environments.config import parse_env_cfg
    from robolab.registrations.droid import camera_presets
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from scipy.optimize import minimize
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cam_math import R_to_quat_xyzw, cfg_snippet, describe, quat_xyzw_to_R
    from overlay import edges

    real = cv2.imread(a.real) if a.real else None
    RW, RH = (real.shape[1], real.shape[0]) if real is not None else (1280, 720)
    W = min(a.render_width, RW)
    H = int(round(RH * W / RW))
    key = "over_shoulder_left_camera" if a.camera == "ext" else "wrist_cam"

    auto_register_droid_envs(task=a.task_file, cameras=getattr(camera_presets, a.preset), viewport_camera=False)
    cfg = parse_env_cfg(a.task, device="cuda:0", seed=0, num_envs=1)
    cfg.recorders = None
    for c in (cfg.scene.over_shoulder_left_camera, cfg.scene.wrist_cam):
        c.height, c.width = H, W
        c.spawn.clipping_range = (0.01, 20.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    dev = env.device
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:1]
    ids = torch.tensor([0], device=dev)

    hold = torch.zeros(1, 8, device=dev)
    hold[:, :7] = torch.tensor([a.joint_pos], device=dev)
    hold[:, 7] = float(a.gripper > 0.5)
    jp = robot.data.joint_pos.torch.clone()
    jp[:, :7] = hold[:, :7]
    robot.write_joint_position_to_sim_index(position=jp)
    if a.object_poses:
        for name, p in json.load(open(a.object_poses)).items():
            pose = torch.tensor([[*p[:3], p[4], p[5], p[6], p[3]]], device=dev, dtype=torch.float32)
            pose[:, :3] += origin
            env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)

    def step(n):
        o = None
        for _ in range(n):
            o = env.step(hold)[0]
        return o

    step(30)
    names = robot.body_names
    bpos = (robot.data.body_pos_w.torch[0] - origin[0]).cpu().numpy()
    bquat = robot.data.body_quat_w.torch[0].cpu().numpy()
    if a.dump_keypoints:
        json.dump({n: bpos[i].round(5).tolist() for i, n in enumerate(names)}, open(a.dump_keypoints, "w"), indent=1)
        print("[refine] wrote robot body origins (env-local) to", a.dump_keypoints, flush=True)
        if real is None:
            os._exit(0)

    cam = env.scene[key]
    stage = omni.usd.get_context().get_stage()
    robot_meshes = [UsdGeom.Imageable(p) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/robot"))
                    if p.IsA(UsdGeom.Gprim)]
    if a.init:
        f0 = json.load(open(a.init))
        pos0, q0 = np.array(f0["pos"], float), np.array(f0["quat_xyzw_gl"], float)
        fx0 = f0["fx"] * W / f0["size"][0]
    else:
        pos0 = (cam.data.pos_w.torch[0] - origin[0]).cpu().numpy().astype(float)
        q0 = cam.data.quat_w_opengl.torch[0].cpu().numpy().astype(float)
        fx0 = float(cam.data.intrinsic_matrices.torch[0, 0, 0])
    if a.fx:
        fx0 = a.fx * W / RW
    if a.hfov:
        fx0 = W / 2 / math.tan(math.radians(a.hfov) / 2)
    R0 = quat_xyzw_to_R(q0)

    def euler(r):
        rx, ry, rz = np.radians(r)
        cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
        return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))

    def params_to_pose(p):
        return pos0 + p[:3], R_to_quat_xyzw(R0 @ euler(p[3:6])), fx0 * math.exp(p[6])

    def render(p, with_robot=True):
        pos, q, fx = params_to_pose(p)
        cam.set_world_poses(origin + torch.tensor([pos], device=dev, dtype=torch.float32),
                            torch.tensor(np.array([q]), device=dev, dtype=torch.float32), convention="opengl")
        cam.set_intrinsic_matrices(torch.tensor([[[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]], device=dev, dtype=torch.float32))
        for m in robot_meshes:
            m.MakeVisible() if with_robot else m.MakeInvisible()
        img = step(4)["image_obs"][key][0].cpu().numpy().astype(np.uint8)[..., ::-1].copy()
        for m in robot_meshes:
            m.MakeVisible()
        return img

    realr = cv2.resize(real, (W, H), interpolation=cv2.INTER_AREA)
    rmask = None
    if a.real_mask:
        rmask = cv2.resize(cv2.imread(a.real_mask, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST) > 127
    static = np.zeros((H, W), np.uint8)
    if a.score:
        for poly in json.load(open(a.score)):
            cv2.fillPoly(static, [(np.array(poly) * W / RW).astype(np.int32)], 255)
    er_all = edges(realr)
    k = 1280.0 / W
    dil = np.ones((max(3, W // 40),) * 2, np.uint8)
    hist = []

    def cost(p):
        img = render(p)
        smask = np.abs(img.astype(int) - render(p, False).astype(int)).sum(-1) > 30
        region = cv2.dilate(((smask | (rmask if rmask is not None else smask)) * 255).astype(np.uint8), dil) | static
        er, es = er_all & region, edges(img) & region
        if er.sum() == 0 or es.sum() == 0:
            return 1e3
        dr = cv2.distanceTransform(255 - er, cv2.DIST_L2, 3)
        ds = cv2.distanceTransform(255 - es, cv2.DIST_L2, 3)
        c = 0.5 * (np.minimum(dr[es > 0], 50).mean() + np.minimum(ds[er > 0], 50).mean()) * k
        if rmask is not None:
            iou = (smask & rmask).sum() / max((smask | rmask).sum(), 1)
            c += 20.0 * (1 - iou)
        hist.append(c)
        if len(hist) % 10 == 0:
            print(f"[refine] eval {len(hist)} cost {c:.3f} best {min(hist):.3f}", flush=True)
        return c

    n = 7 if a.fit_focal else 6
    f = (lambda p: cost(p)) if a.fit_focal else (lambda p: cost(np.r_[p, 0.0]))
    steps = np.array([0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 0.02][:n])
    x0 = np.zeros(n)
    start = f(x0)
    res = minimize(f, x0, method="Nelder-Mead",
                   options=dict(maxfev=a.iters, xatol=1e-4, fatol=1e-3, initial_simplex=np.vstack([x0, x0 + np.diag(steps)])))
    p = res.x if a.fit_focal else np.r_[res.x, 0.0]
    pos, q, fx = params_to_pose(p)
    best = render(p)
    out = dict(camera=a.camera, pos=np.round(pos, 4).tolist(), quat_xyzw_gl=np.round(q, 5).tolist(),
               fx=round(fx * RW / W, 2), size=[RW, RH], hfov_deg=round(2 * math.degrees(math.atan(W / 2 / fx)), 3),
               cost_start=round(float(start), 3), cost_final=round(float(res.fun), 3), joint_pos=a.joint_pos)
    if a.camera == "wrist":
        i = names.index(a.gripper_link)
        Rb = quat_xyzw_to_R(bquat[i])
        out["offset_pos"] = np.round(Rb.T @ (pos - bpos[i]), 4).tolist()
        out["offset_rot"] = np.round(R_to_quat_xyzw(Rb.T @ quat_xyzw_to_R(q)), 5).tolist()
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    cv2.imwrite(a.out + "_best.png", best)
    vis = (realr * 0.35).astype(np.uint8)
    vis[er_all > 0] = (255, 0, 255)
    vis[edges(best) > 0] = (0, 255, 0)
    cv2.imwrite(a.out + "_edges.png", vis)
    print(json.dumps(out, indent=1))
    print(describe(q))
    if a.camera == "ext":
        print(cfg_snippet("MyExternalCameraCfg", pos, q, fx * RW / W, RW, RH))
    else:
        print(f"wrist OffsetCfg(pos={tuple(out['offset_pos'])}, rot={tuple(out['offset_rot'])}, convention=\"opengl\")")
    os._exit(0)


if __name__ == "__main__":
    main()
