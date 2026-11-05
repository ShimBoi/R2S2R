"""Render a scene version for check_version.py: both policy cameras at the recorded joint state and the
photographed object layout, plus a visibility mask per entity (robot and every listed object) for each camera.
Optionally scan candidate external-camera poses (compose_ext.py output) and score each by robot-silhouette IoU
against the real robot mask, to resolve scale / pose ambiguity.
Run inside the container on a GPU node you own (see reference.md, "Running").

Entity masks come from the cameras' instance-id segmentation (prim path of each pixel); if that output is
unavailable they fall back to rendering with and without the entity's meshes. Physics settles for --settle
steps first.

Usage:
  python render_check.py --task-file t.py --task T --preset P --out DIR --joint-pos j1..j7 [--gripper 0]
      [--entities robot bread plate lid=shell_main+top ...] [--object-poses poses.json] [--ext-fit fit.json]
      [--cands cands.json --real-robot-mask robot_mask.png] [--res 1280 720] [--steps 12] [--settle 90]
An entity is a prim-path token (any prim under a prim with that name), or label=tok1+tok2 for the union of
several (e.g. only the parts of a link that the real label outlines).
Writes DIR/ext.png, DIR/wrist.png, DIR/masks/{ext,wrist}_<entity>.png, DIR/render.json (camera poses, K);
with --cands also DIR/scan/c<i>.png, c<i>_robot.png and DIR/scan.json [{i, scale, iou}], best first.
Set the task's photographed-layout switch (e.g. an env var read by the task's events) before running, or pass
--object-poses.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joint-pos", type=float, nargs=7, required=True)
    ap.add_argument("--gripper", type=float, default=0.0)
    ap.add_argument("--entities", nargs="*", default=["robot"])
    ap.add_argument("--object-poses")
    ap.add_argument("--ext-fit")
    ap.add_argument("--cands")
    ap.add_argument("--real-robot-mask")
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--settle", type=int, default=90)
    a = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, enable_cameras=True)

    import cv2
    import numpy as np
    import omni.usd
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from pxr import UsdGeom
    from robolab.core.environments.config import parse_env_cfg
    from robolab.registrations.droid import camera_presets
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    auto_register_droid_envs(task=a.task_file, cameras=getattr(camera_presets, a.preset), viewport_camera=False)
    cfg = parse_env_cfg(a.task, device="cuda:0", seed=0, num_envs=1)
    cfg.recorders = None
    W, H = a.res
    for c in (cfg.scene.over_shoulder_left_camera, cfg.scene.wrist_cam):
        c.height, c.width = H, W
        c.spawn.clipping_range = (0.01, 20.0)
        c.data_types = list(c.data_types) + ["instance_id_segmentation_fast"]
        c.colorize_instance_id_segmentation = False
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    dev = env.device
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:1]
    hold = torch.zeros(1, 8, device=dev)
    hold[:, :7] = torch.tensor([a.joint_pos], device=dev)
    hold[:, 7] = a.gripper
    jp = robot.data.joint_pos.torch.clone()
    jp[:, :7] = hold[:, :7]
    robot.write_joint_position_to_sim_index(position=jp)
    if a.object_poses:
        for name, p in json.load(open(a.object_poses)).items():
            pose = torch.tensor([[*p[:3], p[4], p[5], p[6], p[3]]], device=dev, dtype=torch.float32)
            pose[:, :3] += origin
            obj = env.scene[name]
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([0], device=dev))
    stage = omni.usd.get_context().get_stage()
    env_root = "/World/envs/env_0/"
    toks = {e.split("=")[0]: set(e.split("=")[-1].split("+")) for e in a.entities}
    a.entities = list(toks)

    def match(path, ent):
        path = str(path)
        return path.startswith(env_root) and bool(toks[ent] & set(path[len(env_root):].split("/")))
    meshes = {}
    for ent in a.entities:
        meshes[ent] = [UsdGeom.Imageable(p) for p in stage.Traverse() if match(p.GetPath(), ent) and p.IsA(UsdGeom.Gprim)]
        print(f"[check] entity {ent}: {len(meshes[ent])} meshes", flush=True)
    cam = env.scene["over_shoulder_left_camera"]

    def set_ext(c):
        cam.set_world_poses(origin + torch.tensor([c["pos"]], device=dev, dtype=torch.float32),
                            torch.tensor([c["quat_xyzw_gl"]], device=dev, dtype=torch.float32), convention="opengl")
        fx = c["fx"] * W / c.get("size", [W, H])[0]
        cam.set_intrinsic_matrices(torch.tensor([[[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]], device=dev, dtype=torch.float32))

    def shoot(hidden=None):
        for m in meshes.get(hidden, []):
            m.MakeInvisible()
        o = None
        for _ in range(a.steps):
            o = env.step(hold)[0]
        for m in meshes.get(hidden, []):
            m.MakeVisible()
        im = {k: o["image_obs"][key][0].cpu().numpy().astype(np.uint8)[..., ::-1].copy()
              for k, key in (("ext", "over_shoulder_left_camera"), ("wrist", "wrist_cam"))}
        return im

    def diff_mask(a_, b_):
        m = (np.abs(a_.astype(int) - b_.astype(int)).sum(-1) > 30).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) * 255

    def seg_masks(key):
        """Entity masks from instance-id segmentation; None if the sensor output is unavailable."""
        c = env.scene[key]
        out = c.data.output.get("instance_id_segmentation_fast") if hasattr(c.data.output, "get") else None
        if out is None:
            return None
        arr = (out.torch if hasattr(out, "torch") else out)[0].cpu().numpy().reshape(H, W, -1).astype(np.int64)
        info = c.data.info[0] if isinstance(c.data.info, (list, tuple)) else c.data.info
        id2path = info["instance_id_segmentation_fast"]["idToLabels"]

        nc = arr.shape[-1]
        ids = sum(arr[..., j] << (8 * (nc - 1 - j)) for j in range(nc))

        def key_code(k):
            v = [int(x) for x in str(k).strip("()[] ").split(",")]
            return sum(x << (8 * (len(v) - 1 - j)) for j, x in enumerate(v))
        res = {}
        for ent in a.entities:
            sel = [key_code(i) for i, p in id2path.items() if match(p, ent)]
            res[ent] = (np.isin(ids, sel) * 255).astype(np.uint8)
        return res

    if a.ext_fit:
        set_ext(json.load(open(a.ext_fit)))
    for _ in range(a.settle):
        env.step(hold)
    full = shoot()
    for k, im in full.items():
        cv2.imwrite(os.path.join(a.out, f"{k}.png"), im)
    segs = {}
    for k, key in (("ext", "over_shoulder_left_camera"), ("wrist", "wrist_cam")):
        try:
            segs[k] = seg_masks(key)
        except Exception as e:
            print(f"[check] segmentation unavailable for {k} ({e}); using visibility-toggle masks", flush=True)
            segs[k] = None
    for ent in a.entities:
        hid = shoot(ent) if any(v is None for v in segs.values()) else None
        for k in full:
            m = segs[k][ent] if segs[k] is not None else diff_mask(full[k], hid[k])
            cv2.imwrite(os.path.join(a.out, "masks", f"{k}_{ent}.png"), m)
        print(f"[check] masks for {ent}: ext {int((cv2.imread(os.path.join(a.out, 'masks', f'ext_{ent}.png'), 0) > 0).sum())} px", flush=True)
    info = {}
    for k, key in (("ext", "over_shoulder_left_camera"), ("wrist", "wrist_cam")):
        c = env.scene[key]
        info[k] = dict(pos=(c.data.pos_w.torch[0] - origin[0]).cpu().numpy().round(5).tolist(),
                       quat_xyzw_gl=c.data.quat_w_opengl.torch[0].cpu().numpy().round(6).tolist(),
                       fx=float(c.data.intrinsic_matrices.torch[0, 0, 0]), size=[W, H])
    json.dump(dict(cameras=info, joint_pos=a.joint_pos, entities=a.entities), open(os.path.join(a.out, "render.json"), "w"), indent=1)

    if a.cands:
        real = cv2.resize(cv2.imread(a.real_robot_mask, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_NEAREST) > 127
        os.makedirs(os.path.join(a.out, "scan"), exist_ok=True)
        res = []
        for i, c in enumerate(json.load(open(a.cands))):
            set_ext(c)
            with_r, without = shoot(), shoot("robot")
            m = diff_mask(with_r["ext"], without["ext"]) > 0
            iou = float((m & real).sum() / max((m | real).sum(), 1))
            cv2.imwrite(os.path.join(a.out, "scan", f"c{i}.png"), with_r["ext"])
            cv2.imwrite(os.path.join(a.out, "scan", f"c{i}_robot.png"), m.astype(np.uint8) * 255)
            res.append(dict(i=i, scale=c.get("scale"), iou=round(iou, 4)))
            print(f"[check] cand {i} scale {c.get('scale')} robot IoU {iou:.3f}", flush=True)
        res.sort(key=lambda r: -r["iou"])
        json.dump(res, open(os.path.join(a.out, "scan.json"), "w"), indent=1)
    print("[check] done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
