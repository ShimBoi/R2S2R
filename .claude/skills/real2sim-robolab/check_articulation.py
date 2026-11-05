"""Physics check for an articulated object: release the moving part at several joint positions and record where
it settles, then render both policy cameras closed and open. Run in your Isaac Lab environment on a GPU.

What to look for:
- closed after settling reads ~0 (nothing overlaps or props it open)
- released below the balance angle it closes fully; above it falls to the open limit (a lid), or it stays where
  released (a damped door / drawer), matching how the real one behaves
- nothing jitters or drifts at rest
- every cloned env's fixture sits at its own env's pose (env_spread ~0)

Usage, fixture built by robolab.core.sim2real.articulated_usd with root "fixed" / "floating" and registered with
fixtures.articulated_fixture_cfg (the default for new objects):
  python check_articulation.py --task-file t.py --task T --preset P --fixture lidded_box --joint hinge \
      [--release 40 70 85 95] [--open 100] [--steps 10 --samples 6] --out DIR
Legacy "rigid_links" object, moving link registered as a RigidObjectCfg (its default pose is the closed pose):
  python check_articulation.py ... --link box_lid --anchor x y z --axis ax ay az --out DIR
Run with ROBOLAB_DR=0 so every env starts at the same pose. Revolute values in degrees; add --prismatic for slides
(values in metres). Writes DIR/closed_{ext,wrist}.png, DIR/open_{ext,wrist}.png, DIR/articulation.json.
"""
import argparse
import json
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--fixture")
    ap.add_argument("--joint")
    ap.add_argument("--link")
    ap.add_argument("--anchor", type=float, nargs=3)
    ap.add_argument("--axis", type=float, nargs=3)
    ap.add_argument("--prismatic", action="store_true")
    ap.add_argument("--release", type=float, nargs="*", default=[40, 70, 85, 95])
    ap.add_argument("--open", type=float, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not ((a.fixture and a.joint) or (a.link and a.anchor and a.axis)):
        ap.error("give --fixture and --joint, or --link with --anchor and --axis")

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, enable_cameras=True)

    import cv2
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from robolab.core.environments.config import parse_env_cfg
    from robolab.registrations.droid import camera_presets
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from scipy.spatial.transform import Rotation

    os.makedirs(a.out, exist_ok=True)
    auto_register_droid_envs(task=a.task_file, cameras=getattr(camera_presets, a.preset), viewport_camera=False)
    cfg = parse_env_cfg(a.task, device="cuda:0", seed=0, num_envs=a.num_envs)
    cfg.recorders = None
    cfg.terminations = None
    for c in (cfg.scene.over_shoulder_left_camera, cfg.scene.wrist_cam):
        c.height, c.width = 720, 1280
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    dev = env.device
    robot = env.scene["robot"]
    ids = torch.tensor([0], device=dev)
    hold = torch.cat([robot.data.joint_pos.torch[:, :7], torch.zeros(a.num_envs, 1, device=dev)], 1)
    scale = 1.0 if a.prismatic else math.pi / 180

    def step(n):
        o = None
        for _ in range(n):
            o = env.step(hold)[0]
        return o

    step(30)
    if a.fixture:
        art = env.scene[a.fixture]
        j = art.joint_names.index(a.joint)
        base = art.data.root_pos_w.torch - env.scene.env_origins
        moving = art.data.body_pos_w.torch[:, 1:].mean(1) - art.data.root_pos_w.torch
        spread = float((moving - moving[0]).norm(dim=1).max())
        print(f"[articulation] fixture roots per env {[[round(float(v), 3) for v in b[:2]] for b in base]}", flush=True)

        def joint_value():
            return float(art.data.joint_pos.torch[0, j]) / scale

        def set_joint(v):
            jp = art.data.joint_pos.torch[ids].clone()
            jp[:, j] = v * scale
            art.write_joint_position_to_sim_index(position=jp, env_ids=ids)
            art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=ids)
    else:
        link = env.scene[a.link]
        origin = env.scene.env_origins[0].cpu().numpy()
        rel = (link.data.root_pos_w.torch - env.scene.env_origins).cpu().numpy()
        spread = float(np.abs(rel - rel[0]).max())
        p0 = link.data.root_pos_w.torch[0].cpu().numpy() - origin
        q0 = Rotation.from_quat(link.data.root_quat_w.torch[0].cpu().numpy())
        anchor, axis = np.array(a.anchor), np.array(a.axis) / np.linalg.norm(a.axis)

        def joint_value():
            p = link.data.root_pos_w.torch[0].cpu().numpy() - origin
            if a.prismatic:
                return float(np.dot(p - p0, axis))
            rv = (Rotation.from_quat(link.data.root_quat_w.torch[0].cpu().numpy()) * q0.inv()).as_rotvec()
            return math.degrees(float(np.dot(rv, axis)))

        def set_joint(v):
            if a.prismatic:
                p, q = p0 + axis * v, q0
            else:
                r = Rotation.from_rotvec(axis * math.radians(v))
                p, q = anchor + r.apply(p0 - anchor), r * q0
            pose = torch.tensor([[*(p + origin), *q.as_quat()]], device=dev, dtype=torch.float32)
            link.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            link.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)

    print(f"[articulation] env_spread {spread:.4f} m over {a.num_envs} envs {'ok' if spread < 0.005 else 'FAIL'}", flush=True)

    def save(o, name):
        for k, key in (("ext", "over_shoulder_left_camera"), ("wrist", "wrist_cam")):
            cv2.imwrite(os.path.join(a.out, f"{name}_{k}.png"), o["image_obs"][key][0].cpu().numpy().astype(np.uint8)[..., ::-1])

    res = {"env_spread": round(spread, 4), "closed": round(joint_value(), 2), "release": {}}
    save(step(5), "closed")
    print(f"[articulation] closed after settling: {res['closed']}", flush=True)
    for v in a.release:
        set_joint(v)
        traj = []
        for _ in range(a.samples):
            step(a.steps)
            traj.append(round(joint_value(), 2))
        res["release"][str(v)] = traj
        print(f"[articulation] released at {v}: {traj}", flush=True)
    set_joint(a.open)
    save(step(30), "open")
    res["open_settled"] = round(joint_value(), 2)
    print(f"[articulation] opened to {a.open}: settled at {res['open_settled']}", flush=True)
    json.dump(res, open(os.path.join(a.out, "articulation.json"), "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
