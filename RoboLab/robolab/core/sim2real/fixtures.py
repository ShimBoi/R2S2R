# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Articulated scene fixtures (lids, doors, drawers, appliances) built by articulated_usd with root "fixed"
or "floating", and included in a scene USD under {ENV_REGEX_NS}/scene/<name>.

    scene.cabinet = articulated_fixture_cfg("cabinet", pos=(0.6, 0.3, 0.0), yaw_deg=90,
                                            joints={"door": {"damping": 0.05}, "drawer": {"damping": 5.0}})
    "reset_cabinet": EventTerm(func=reset_fixture, mode="reset",
                               params={"name": "cabinet", "pose": (0.6, 0.3, 0.0, 90)})
    door_open = joint_position(env, "cabinet", "door") > math.radians(60)

Joint gains are SI: revolute damping N*m*s/rad, stiffness N*m/rad, friction N*m; prismatic per metre / N.
A plan's "joints" section randomizes them per env, its "joint_reset" section the opening at reset, and its
"fixtures" section the fixture's pose (plan.build_events merges it into the matching reset_fixture term).
"""
import math

import torch

from robolab.core.sim2real.randomizers import _ids, _u, fixture_poses

JOINT_DEFAULTS = {
    "revolute": {"damping": 0.01, "stiffness": 0.0, "friction": 0.0},
    "prismatic": {"damping": 2.0, "stiffness": 0.0, "friction": 0.0},
}


def _yaw_quat_xyzw(yaw_deg):
    h = math.radians(yaw_deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))


def articulated_fixture_cfg(name, pos, yaw_deg=0.0, joints=None, joint_pos=None, prim_path=None):
    """ArticulationCfg for a fixture already in the scene USD. joints maps a joint-name regex to
    {"type": "revolute" | "prismatic", "damping", "stiffness", "friction", "armature"} (unset values take
    JOINT_DEFAULTS for the type); joints no regex matches keep the USD drive's values. The default (None)
    gives every joint the revolute defaults. joint_pos maps a regex to the default (closed) position.
    All joints share one passive actuator group, which is what mdp.randomize_actuator_gains expects when the
    plan selects a subset of joints."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    joints = joints or {".*": {}}
    vals = {k: {} for k in ("stiffness", "damping", "friction", "armature")}
    for expr, g in joints.items():
        d = {**JOINT_DEFAULTS[g.get("type", "revolute")], **g}
        for k in vals:
            if d.get(k) is not None:
                vals[k][expr] = float(d[k])
    return ArticulationCfg(
        prim_path=prim_path or f"{{ENV_REGEX_NS}}/scene/{name}",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pos), rot=_yaw_quat_xyzw(yaw_deg),
                                                   joint_pos=joint_pos or {".*": 0.0}),
        actuators={"passive": ImplicitActuatorCfg(
            joint_names_expr=list(joints), stiffness=vals["stiffness"], damping=vals["damping"],
            friction=vals["friction"], armature=vals["armature"] or None,
            effort_limit_sim=1e6, velocity_limit_sim=100.0)},
    )


def joint_position(env, name, joint):
    """Joint position (rad or m) of one joint of fixture name, per env."""
    art = env.scene[name]
    return art.data.joint_pos.torch[:, art.joint_names.index(joint)]


def fixture_centre(env, name):
    """Fixture root position in the env frame, per env (it moves when the pose is randomized)."""
    return env.scene[name].data.root_pos_w.torch - env.scene.env_origins


def reset_fixture(env, env_ids, name, pose, pos_m=0.0, yaw_deg=0.0, joint_pos=None):
    """Reset fixture name to pose = (x, y, z, yaw_deg) in the env frame, jittered by +-pos_m in x / y and
    +-yaw_deg, with its joints at their default positions, or uniform in joint_pos {regex: [lo, hi]}.
    The sampled pose is kept in randomizers.fixture_poses(env)[name] as (x, y, z, yaw_rad)."""
    import re

    ids = _ids(env, env_ids)
    n, dev = len(ids), env.device
    art = env.scene[name]
    p = torch.tensor(pose[:3], device=dev, dtype=torch.float32).repeat(n, 1)
    p[:, :2] += _u(-pos_m, pos_m, (n, 2), dev)
    yaw = math.radians(pose[3]) + _u(-math.radians(yaw_deg), math.radians(yaw_deg), (n,), dev)
    q = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], 1)
    art.write_root_pose_to_sim_index(root_pose=torch.cat([p + env.scene.env_origins[ids], q], 1), env_ids=ids)
    art.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    jp = art.data.default_joint_pos.torch[ids].clone()
    for expr, (lo, hi) in (joint_pos or {}).items():
        for j, jn in enumerate(art.joint_names):
            if re.fullmatch(expr, jn):
                jp[:, j] = _u(lo, hi, (n,), dev)
    art.write_joint_position_to_sim_index(position=jp, env_ids=ids)
    art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=ids)
    store = fixture_poses(env).setdefault(
        name, torch.tensor([[*pose[:3], math.radians(pose[3])]], device=dev).repeat(env.num_envs, 1))
    store[ids] = torch.cat([p, yaw[:, None]], 1)
