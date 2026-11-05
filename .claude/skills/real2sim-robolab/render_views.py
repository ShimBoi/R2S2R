"""Render a RoboLab task's external and wrist cameras for side-by-side comparison with real photos.
Run inside the container on a GPU node you own (see reference.md, "Running").

Usage:
  python render_views.py --task-file my_task.py --task MyTask --out OUT [--preset WRIST_LAB]
      [--ext-fit fit_ext.json] [--wrist-offset px py pz qx qy qz qw] [--res 1280 720]
      [--joint-pos j1 .. j7] [--object-poses poses.json] [--views views.json] [--bg dome.png --intensity 700]
--ext-fit:       override the external camera pose and fx at runtime with fit_camera.py output (before the
                 camera is committed to a CameraCfg class), saved as ext_fit.png.
--wrist-offset:  override the wrist camera offset on Robotiq base_link (opengl convention, xyzw).
--object-poses:  {"obj_a": [x, y, z, qw, qx, qy, qz], ...} env-local; place objects as in the photo.
--views:         [{"name", "pos", "target", "hfov_deg", "roll_deg", "joint_pos", "wrist": bool}, ...]
Writes ext_default.png, wrist_default.png, ext_fit.png, <view>.png [<view>_wrist.png].
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="WRIST_LAB")
    ap.add_argument("--ext-fit")
    ap.add_argument("--wrist-offset", type=float, nargs=7)
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--joint-pos", type=float, nargs=7)
    ap.add_argument("--object-poses")
    ap.add_argument("--views")
    ap.add_argument("--bg")
    ap.add_argument("--intensity", type=float)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, enable_cameras=True)

    import numpy as np
    import omni.usd
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from PIL import Image
    from pxr import Sdf
    from robolab.core.environments.config import parse_env_cfg
    from robolab.registrations.droid import camera_presets
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cam_math import lookat_opengl

    os.makedirs(a.out, exist_ok=True)
    os.environ["ROBOLAB_EVAL_ONLY"] = "1" if a.eval_only else "0"
    auto_register_droid_envs(task=a.task_file, cameras=getattr(camera_presets, a.preset), viewport_camera=False)
    cfg = parse_env_cfg(a.task, device="cuda:0", seed=0, num_envs=1)
    cfg.recorders = None
    W, H = a.res
    for c in (cfg.scene.over_shoulder_left_camera, cfg.scene.wrist_cam):
        c.height, c.width = H, W
        c.spawn.clipping_range = (0.01, 20.0)
    if a.wrist_offset:
        cfg.scene.wrist_cam.offset.pos = tuple(a.wrist_offset[:3])
        cfg.scene.wrist_cam.offset.rot = tuple(a.wrist_offset[3:])
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()

    stage = omni.usd.get_context().get_stage()
    bg = stage.GetPrimAtPath("/World/background")
    if a.bg:
        bg.GetAttribute("inputs:texture:file").Set(Sdf.AssetPath(os.path.abspath(a.bg)))
    if a.intensity is not None:
        bg.GetAttribute("inputs:intensity").Set(a.intensity)

    dev = env.device
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:1]
    hold = torch.cat([robot.data.joint_pos.torch[:, :7], torch.zeros(1, 1, device=dev)], 1)

    def set_joints(q):
        hold[:, :7] = torch.tensor([q], device=dev)
        jp = robot.data.joint_pos.torch.clone()
        jp[:, :7] = hold[:, :7]
        robot.write_joint_position_to_sim_index(position=jp)

    def place_objects():
        if not a.object_poses:
            return
        for name, p in json.load(open(a.object_poses)).items():
            pose = torch.tensor([[*p[:3], p[4], p[5], p[6], p[3]]], device=dev, dtype=torch.float32)
            pose[:, :3] += origin
            obj = env.scene[name]
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=dev))
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=torch.tensor([0], device=dev))

    def step(n):
        o = None
        for _ in range(n):
            o = env.step(hold)[0]
        return o

    def save(obs, key, name):
        Image.fromarray(obs["image_obs"][key][0].cpu().numpy().astype(np.uint8)).save(os.path.join(a.out, name + ".png"))

    if a.joint_pos:
        set_joints(a.joint_pos)
    place_objects()
    obs = step(20)
    save(obs, "over_shoulder_left_camera", "ext_default")
    save(obs, "wrist_cam", "wrist_default")
    cam = env.scene["over_shoulder_left_camera"]
    print("[render] ext K", cam.data.intrinsic_matrices.torch[0].cpu().numpy().round(2).tolist(), flush=True)

    def set_cam(pos, quat, fx_at_res):
        cam.set_world_poses(origin + torch.tensor([pos], device=dev, dtype=torch.float32),
                            torch.tensor(np.array([quat]), device=dev, dtype=torch.float32), convention="opengl")
        K = torch.tensor([[[fx_at_res, 0, W / 2], [0, fx_at_res, H / 2], [0, 0, 1]]], device=dev, dtype=torch.float32)
        cam.set_intrinsic_matrices(K)

    if a.ext_fit:
        f = json.load(open(a.ext_fit))
        set_cam(f["pos"], f["quat_xyzw_gl"], f["fx"] * W / f["size"][0])
        save(step(6), "over_shoulder_left_camera", "ext_fit")
        print("[render] ext_fit", flush=True)
    for v in json.load(open(a.views)) if a.views else []:
        if "joint_pos" in v:
            set_joints(v["joint_pos"])
        fx = W / 2 / math.tan(math.radians(v.get("hfov_deg", 90.0)) / 2)
        set_cam(v["pos"], lookat_opengl(v["pos"], v["target"], v.get("roll_deg", 0.0)), fx)
        obs = step(6)
        save(obs, "over_shoulder_left_camera", v["name"])
        if v.get("wrist"):
            save(obs, "wrist_cam", v["name"] + "_wrist")
        print("[render]", v["name"], flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
