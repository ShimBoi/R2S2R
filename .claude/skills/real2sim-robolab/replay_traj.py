"""Replay a real robot trajectory open loop in a RoboLab task and log what the objects do.
Run inside the container on a GPU node you own (see reference.md, "Running").

traj.npz: joint_position [T, 7] (measured, rad), gripper_position [T] (0 open .. 1 closed),
          optional action_joint_position [T, 7] (commanded targets; preferred when present), fps (default 15).
poses.json: {"obj_a": [x, y, z, qw, qx, qy, qz], ...} env-local initial object poses measured from the
          first real frame (same format as the task's presets).

Usage:
  python replay_traj.py --task-file my_task.py --task MyTask --traj traj.npz --object-poses poses.json --out OUT
      [--preset WRIST_LAB] [--frames-every 5] [--final-poses real_final.json]
Writes OUT/log.npz (sim joints, joint tracking error, object poses as x y z qx qy qz qw, grasp contact per step)
and frames.
Sim control is 15 Hz (dt 1/120, decimation 8); other rates are resampled by linear interpolation.
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--object-poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="WRIST_LAB")
    ap.add_argument("--frames-every", type=int, default=5)
    ap.add_argument("--final-poses")
    a = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, enable_cameras=True)

    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from PIL import Image
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.task.conditionals import object_grabbed
    from robolab.registrations.droid import camera_presets
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.traj)
    q = d["joint_position"].astype(np.float32)
    g = d["gripper_position"].astype(np.float32).reshape(-1)
    tgt = d["action_joint_position"].astype(np.float32) if "action_joint_position" in d else np.r_[q[1:], q[-1:]]
    fps = float(d["fps"]) if "fps" in d else 15.0
    if abs(fps - 15.0) > 1e-3:
        t_old = np.arange(len(q)) / fps
        t_new = np.arange(0, t_old[-1], 1 / 15.0)
        interp = lambda x: np.stack([np.interp(t_new, t_old, x[:, i]) for i in range(x.shape[1])], 1)
        q, tgt, g = interp(q), interp(tgt), np.interp(t_new, t_old, g)
    poses = json.load(open(a.object_poses))

    auto_register_droid_envs(task=a.task_file, cameras=getattr(camera_presets, a.preset), viewport_camera=False)
    cfg = parse_env_cfg(a.task, device="cuda:0", seed=0, num_envs=1)
    cfg.recorders = None
    for c in (cfg.scene.over_shoulder_left_camera, cfg.scene.wrist_cam):
        c.height, c.width = 360, 640
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    dev = env.device
    ids = torch.tensor([0], device=dev)
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:1]

    jp = robot.data.joint_pos.torch.clone()
    jp[:, :7] = torch.tensor(q[:1], device=dev)
    robot.write_joint_position_to_sim_index(position=jp)
    for name, p in poses.items():
        pose = torch.tensor([[*p[:3], p[4], p[5], p[6], p[3]]], device=dev, dtype=torch.float32)
        pose[:, :3] += origin
        env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)
    act = torch.zeros(1, 8, device=dev)
    act[:, :7] = torch.tensor(q[:1], device=dev)
    act[:, 7] = float(g[0] > 0.5)
    for _ in range(10):
        env.step(act)

    log = {"sim_q": [], "err_q": [], "grabbed": []}
    log.update({f"pose_{n}": [] for n in poses})
    for t in range(len(tgt)):
        act[:, :7] = torch.tensor(tgt[t:t + 1], device=dev)
        act[:, 7] = float(g[t] > 0.5)
        obs = env.step(act)[0]
        sq = robot.data.joint_pos.torch[0, :7].cpu().numpy()
        log["sim_q"].append(sq)
        log["err_q"].append(np.abs(sq - q[min(t + 1, len(q) - 1)]))
        log["grabbed"].append([bool(object_grabbed(env, object=n)[0]) for n in poses])
        for n in poses:
            o = env.scene[n].data
            log[f"pose_{n}"].append(np.r_[(o.root_pos_w.torch[0] - origin[0]).cpu().numpy(), o.root_quat_w.torch[0].cpu().numpy()])
        if t % a.frames_every == 0:
            for k, s in (("over_shoulder_left_camera", "ext"), ("wrist_cam", "wrist")):
                Image.fromarray(obs["image_obs"][k][0].cpu().numpy().astype(np.uint8)).save(f"{a.out}/{s}_{t:04d}.png")
    np.savez(f"{a.out}/log.npz", objects=list(poses), **{k: np.array(v) for k, v in log.items()})
    e = np.array(log["err_q"])
    print(f"[replay] {len(tgt)} steps, joint tracking error mean {e.mean():.4f} rad, max {e.max():.4f} rad", flush=True)
    gr = np.array(log["grabbed"])
    for i, n in enumerate(poses):
        first = int(np.argmax(gr[:, i])) if gr[:, i].any() else -1
        fin = log[f"pose_{n}"][-1]
        msg = f"[replay] {n}: first gripper contact step {first}, final pos {np.round(fin[:3], 3).tolist()}"
        if a.final_poses:
            real = np.array(json.load(open(a.final_poses))[n][:3])
            msg += f", real final {real.round(3).tolist()}, error {np.linalg.norm(fin[:3] - real):.3f} m"
        print(msg, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
