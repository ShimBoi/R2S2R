"""Render a grid of domain-randomized envs from each policy camera, for the Phase 10 plan check: the real
photos should look like plausible members of the set, and no sample may be absurd (camera inside geometry,
black or saturated frames, objects floating). Also places each real image in the distribution of sim image
statistics (mean brightness and per-channel ratios) and resets twice, so per-reset resampling is visible.
Run in your Isaac Lab environment on a GPU, with the task's DR terms on (ROBOLAB_DR unset or 1).

Usage:
  python check_dr.py --task-file t.py --task T --preset P --out DIR --real ext=ext.png wrist=wrist.png
      [--num-envs 16] [--res 640 360] [--steps 10]
--real maps the short camera names used in file names (ext, wrist) to the real images; the short names
follow --cams (default ext=over_shoulder_left_camera wrist=wrist_cam).
Writes DIR/grid_<cam>_r{0,1}.jpg (real image top-left, outlined), DIR/dr_stats.json and prints a summary:
the real image's percentile in each statistic (outside 5-95 means the ranges do not cover it).
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
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--cams", nargs="*", default=["ext=over_shoulder_left_camera", "wrist=wrist_cam"])
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--res", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True, enable_cameras=True)

    import cv2
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from robolab.core.environments.config import parse_env_cfg
    from robolab.registrations.droid import camera_presets
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    os.makedirs(a.out, exist_ok=True)
    cams = dict(c.split("=") for c in a.cams)
    real = {k: cv2.imread(v)[..., ::-1] for k, v in (r.split("=") for r in a.real)}
    auto_register_droid_envs(task=a.task_file, cameras=getattr(camera_presets, a.preset), viewport_camera=False)
    cfg = parse_env_cfg(a.task, device="cuda:0", seed=0, num_envs=a.num_envs)
    cfg.recorders = None
    cfg.terminations = None
    for key in cams.values():
        c = getattr(cfg.scene, key)
        c.width, c.height = a.res
    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]

    def stats(img):
        f = img.reshape(-1, 3).astype(np.float64)
        m = f.mean(0)
        return {"mean": float(m.mean()), "r/g": float(m[0] / max(m[1], 1e-6)), "b/g": float(m[2] / max(m[1], 1e-6)),
                "std": float(f.mean(1).std()), "dark": float((f.mean(1) < 8).mean()), "sat": float((f.min(1) > 250).mean())}

    out = {}
    for r in range(2):
        env.reset()
        hold = torch.cat([robot.data.joint_pos.torch[:, :7], torch.zeros(a.num_envs, 1, device=env.device)], 1)
        o = None
        for _ in range(a.steps):
            o = env.step(hold)[0]
        for short, key in cams.items():
            imgs = o["image_obs"][key].cpu().numpy().astype(np.uint8)
            tiles = list(imgs)
            if short in real:
                rr = cv2.resize(real[short], (imgs.shape[2], imgs.shape[1]))
                rr = cv2.rectangle(rr.copy(), (0, 0), (rr.shape[1] - 1, rr.shape[0] - 1), (255, 0, 0), 6)
                tiles = [rr] + tiles
            cols = int(np.ceil(np.sqrt(len(tiles))))
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
            cv2.imwrite(os.path.join(a.out, f"grid_{short}_r{r}.jpg"), grid[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.setdefault(short, {"sim": []})["sim"] += [stats(i) for i in imgs]
    summary = {}
    for short, d in out.items():
        if short in real:
            rs = stats(cv2.resize(real[short], a.res))
            d["real"] = rs
            summary[short] = {k: round(100 * float(np.mean([s[k] <= rs[k] for s in d["sim"]])), 1) for k in ("mean", "r/g", "b/g", "std")}
        bad = [i for i, s in enumerate(d["sim"]) if s["dark"] > 0.5 or s["sat"] > 0.3 or s["std"] < 5]
        d["absurd"] = bad
        print(f"[check_dr] {short}: real percentile {summary.get(short)}; absurd samples {bad}", flush=True)
    json.dump({"cams": out, "real_percentile": summary}, open(os.path.join(a.out, "dr_stats.json"), "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main()
