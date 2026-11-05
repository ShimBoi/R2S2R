# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator check for articulated fixtures and joint randomization, with 4 cloned envs:
- a cabinet (root "fixed": body, revolute door, prismatic drawer) and a waffle maker (root "floating" on a
  30 kg base, revolute lid with a grasp lip), both built by articulated_usd;
- reset_fixture moves each env's fixture to its own jittered pose and the simulated pose follows (GPU PhysX
  ignores pose writes to kinematic bodies; articulation roots must not have that problem);
- joint_pos at reset is applied per env and read back by joint_position;
- the plan's "joints" section gives each env its own damping, and the drawer's motion follows it.

Run inside the container on a GPU node:
  python RoboLab/tests/sim2real/smoke_sim.py [--out DIR]
Prints one line per check and SMOKE PASS / SMOKE FAIL.
"""
import argparse
import math
import os
import tempfile


def specs(out_dir):
    cabinet = {
        "name": "cabinet", "root": "fixed", "out": os.path.join(out_dir, "cabinet", "cabinet.usda"),
        "links": [
            {"name": "body", "mass": 10.0, "shapes": [
                {"type": "box", "name": "back", "pos": [0, 0.19, 0.3], "size": [0.4, 0.02, 0.6]},
                {"type": "box", "name": "side_l", "pos": [-0.19, 0, 0.3], "size": [0.02, 0.4, 0.6]},
                {"type": "box", "name": "side_r", "pos": [0.19, 0, 0.3], "size": [0.02, 0.4, 0.6]},
                {"type": "box", "name": "floor", "pos": [0, 0, 0.01], "size": [0.4, 0.4, 0.02]},
                {"type": "box", "name": "shelf", "pos": [0, 0, 0.3], "size": [0.36, 0.38, 0.02]},
                {"type": "box", "name": "roof", "pos": [0, 0, 0.59], "size": [0.4, 0.4, 0.02]}]},
            {"name": "door", "mass": 1.0, "shapes": [
                {"type": "box", "name": "panel", "pos": [0, -0.21, 0.45], "size": [0.38, 0.02, 0.26]},
                {"type": "box", "name": "door_handle", "pos": [0.14, -0.24, 0.45], "size": [0.02, 0.03, 0.1]}]},
            {"name": "drawer", "mass": 1.0, "shapes": [
                {"type": "box", "name": "front", "pos": [0, -0.2, 0.15], "size": [0.36, 0.02, 0.24]},
                {"type": "box", "name": "tray", "pos": [0, -0.02, 0.05], "size": [0.34, 0.34, 0.01]},
                {"type": "box", "name": "drawer_handle", "pos": [0, -0.23, 0.15], "size": [0.1, 0.03, 0.02]}]}],
        "joints": [
            {"name": "door", "type": "revolute", "parent": "body", "child": "door", "anchor": [-0.19, -0.21, 0.45],
             "axis": [0, 0, -1], "limits": [0, 110]},
            {"name": "drawer", "type": "prismatic", "parent": "body", "child": "drawer", "anchor": [0, 0, 0.15],
             "axis": [0, -1, 0], "limits": [0, 0.3]}],
    }
    waffle = {
        "name": "waffle", "root": "floating", "base_mass": 30, "out": os.path.join(out_dir, "waffle", "waffle.usda"),
        "links": [
            {"name": "base", "mass": 1.5, "shapes": [
                {"type": "cylinder", "name": "body", "pos": [0, 0, 0.021], "radius": 0.09, "height": 0.042},
                {"type": "box", "name": "pedestal", "pos": [0, 0.092, 0.021], "size": [0.088, 0.075, 0.042]}]},
            {"name": "lid", "mass": 0.55, "shapes": [
                {"type": "cylinder", "name": "shell", "pos": [0, 0, 0.0735], "radius": 0.093, "height": 0.063},
                {"type": "box", "name": "housing", "pos": [0, 0.092, 0.08], "size": [0.088, 0.075, 0.076]},
                {"type": "box", "name": "lip", "pos": [0, -0.105, 0.06], "size": [0.05, 0.025, 0.012]}]}],
        "joints": [{"name": "hinge", "type": "revolute", "parent": "base", "child": "lid", "anchor": [0, 0.1295, 0.042],
                    "axis": [-1, 0, 0], "limits": [0, 100]}],
    }
    return cabinet, waffle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from isaaclab.app import AppLauncher
    AppLauncher(headless=True)

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from robolab.core.sim2real import articulated_usd, fixtures
    from robolab.core.sim2real.builders import build_joints

    out = a.out or tempfile.mkdtemp(prefix="sim2real_smoke_")
    cab_spec, waf_spec = specs(out)
    cab_usd, waf_usd = articulated_usd.build(cab_spec), articulated_usd.build(waf_spec)

    CAB, WAF = (0.0, 0.5, 0.0, 0.0), (0.0, -0.4, 0.0, -14.0)

    def with_spawn(cfg, usd):
        cfg.spawn = sim_utils.UsdFileCfg(usd_path=usd)
        return cfg

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
        cabinet = with_spawn(fixtures.articulated_fixture_cfg(
            "cabinet", CAB[:3], CAB[3], prim_path="{ENV_REGEX_NS}/cabinet",
            joints={"door": {"damping": 0.05}, "drawer": {"type": "prismatic", "damping": 2.0}}), cab_usd)
        waffle = with_spawn(fixtures.articulated_fixture_cfg(
            "waffle", WAF[:3], WAF[3], prim_path="{ENV_REGEX_NS}/waffle", joints={"hinge": {"damping": 0.01}}), waf_usd)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    scene = InteractiveScene(SceneCfg(num_envs=4, env_spacing=2.0))
    sim.reset()

    class Env:
        pass

    env = Env()
    env.scene, env.num_envs, env.device, env.sim = scene, 4, "cuda:0", sim
    ids = torch.arange(4, device="cuda:0")
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"[smoke] {'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)

    def step(n):
        for _ in range(n):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

    torch.manual_seed(0)
    fixtures.reset_fixture(env, ids, "cabinet", CAB, pos_m=0.05, yaw_deg=10, joint_pos={"door": [0.2, 1.0]})
    fixtures.reset_fixture(env, ids, "waffle", WAF, pos_m=0.05, yaw_deg=10)
    want = {n: fixtures.fixture_poses(env)[n].clone() for n in ("cabinet", "waffle")}
    door0 = fixtures.joint_position(env, "cabinet", "door").clone()
    step(60)

    for n, link in (("cabinet", "drawer"), ("waffle", "lid")):
        got = fixtures.fixture_centre(env, n)
        err = (got[:, :2] - want[n][:, :2]).norm(dim=1)
        spread = (want[n][:, :2] - want[n][0, :2]).norm(dim=1).max()
        check(f"{n} pose follows reset_fixture", bool(err.max() < 0.005) and bool(spread > 0.005),
              f"max xy error {err.max():.4f} m over 4 envs, sampled spread {spread:.3f} m")
        art = scene[n]
        moving = art.data.body_pos_w.torch[:, art.body_names.index(link)]
        off = (moving - art.data.root_pos_w.torch).norm(dim=1)
        check(f"{n} {link} stays with its own env's {n}", bool(off.max() < 0.005),
              f"closed {link} frame distance from its root per env {[round(float(v), 4) for v in off]} m")

    door = fixtures.joint_position(env, "cabinet", "door")
    check("door joint_pos reset per env", bool((door - door0).abs().max() < 0.05) and bool(door0.std() > 0.05),
          f"set {[round(float(v), 2) for v in door0]} read {[round(float(v), 2) for v in door]}")
    lid = fixtures.joint_position(env, "waffle", "hinge")
    check("waffle lid closed after reset", bool(lid.abs().max() < 0.02), f"hinge {[round(float(v), 3) for v in lid]}")

    terms = build_joints({"cabinet": {"drawer": {"damping": [0.5, 40.0]}}})
    for t in terms.values():
        t.params["asset_cfg"].resolve(scene)
        t.params["asset_cfg"].finalize(env.device)
        t.func = t.func(t, env) if isinstance(t.func, type) else t.func
        t.func(env, ids, **t.params)
    art = scene["cabinet"]
    j = art.joint_names.index("drawer")
    damp = art.data.joint_damping.torch[:, j].clone()
    check("joint damping randomized per env", bool(damp.std() > 1.0), f"drawer damping {[round(float(v), 2) for v in damp]}")

    fixtures.reset_fixture(env, ids, "cabinet", CAB)
    vel = art.data.joint_vel.torch.clone()
    vel[:, j] = 0.5
    art.write_joint_velocity_to_sim_index(velocity=vel, env_ids=ids)
    step(30)
    travel = fixtures.joint_position(env, "cabinet", "drawer")
    order = torch.argsort(damp)
    mono = bool(torch.all(travel[order][1:] <= travel[order][:-1] + 1e-4))
    check("drawer motion follows its damping", mono and bool(travel.max() > travel.min() + 0.005),
          f"damping {[round(float(v), 1) for v in damp]} -> travel {[round(float(v), 3) for v in travel]} m")

    print("SMOKE PASS" if all(results) else "SMOKE FAIL", flush=True)
    os._exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
