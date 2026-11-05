"""Built-in plan sections. Ranges are [lo, hi].

 "cameras": {"<sensor>": {"pos_m": 0.02, "rot_deg": 2.0}}                     per env, every reset
 "image": {"<sensor>": {"gain": [0.92, 1.08], "wb": 0.06, "contrast": [0.9, 1.1], "gamma": [0.9, 1.1],
                        "noise": [0, 4], "zoom": [0.97, 1.03], "blur_p": 0.2}}     per env, every reset
 "lights": {"<prim name>": {"intensity": [0.8, 1.2], "color": 0.05, "pos_m": 0.3}}  scales the task's values
 "background": {"dir": "__ASSET_DIR__/<scene>/dome_variants", "intensity": [500, 900]}  replaces set_background
 "mass": {"<entity>": {"range": [0.03, 0.06], "operation": "abs"}}                per env, at startup
 "material": {"<entity>": {"static": [0.8, 1.4], "dynamic": [0.6, 1.2], "restitution": [0, 0.1]}}
 "com": {"<entity>": 0.01}                                                        metres, each axis
 "actuator_gains": {"<entity>": {"joints": "panda_joint.*", "stiffness": [0.85, 1.15], "damping": [0.85, 1.15]}}
                                                                                   scale, robot actuators
 "joints": {"<fixture>": {"<joint regex>": {"damping": [0.005, 0.02], "stiffness": [0, 0], "friction": [0, 0.05],
                                            "armature": [0, 0.01], "operation": "abs"}}}  SI units, startup
 "custom": {"<term>": {"func": "package.module:function", "mode": "reset", "params": {...}}}
"""
import importlib
import os

from robolab.core.sim2real import randomizers as R
from robolab.core.sim2real.registry import register


def _event(func, mode, **params):
    from isaaclab.managers import EventTermCfg

    return EventTermCfg(func=func, mode=mode, params=params)


def _entity(name, joints=None):
    from isaaclab.managers import SceneEntityCfg

    return SceneEntityCfg(name, joint_names=joints) if joints is not None else SceneEntityCfg(name)


@register("cameras")
def build_cameras(cfg):
    return {"dr_camera_pose": _event(R.perturb_camera_poses, "reset", cameras=cfg)}


@register("image")
def build_image(cfg):
    return {"dr_image_install": _event(R.install_image_aug, "startup", image=cfg),
            "dr_image_sample": _event(R.resample_image_aug, "reset", image=cfg)}


@register("lights")
def build_lights(cfg):
    return {"dr_lights": _event(R.randomize_lights, "reset", lights=cfg)}


@register("background")
def build_background(cfg):
    from robolab.constants import ASSET_DIR

    d = cfg.get("dir", "").replace("__ASSET_DIR__", ASSET_DIR)
    files = cfg.get("files") or [os.path.join(d, f) for f in sorted(os.listdir(d))
                                 if f.startswith("dome_") and f.endswith(".png")]
    return {"set_background": _event(R.randomize_background, "reset", files=files,
                                     intensity=tuple(cfg.get("intensity", [700, 700])),
                                     prim_path=cfg.get("prim_path", "/World/background"))}


@register("mass")
def build_mass(cfg):
    import isaaclab.envs.mdp as mdp

    return {f"dr_mass_{e}": _event(mdp.randomize_rigid_body_mass, "startup", asset_cfg=_entity(e),
                                   mass_distribution_params=tuple(m["range"]), operation=m.get("operation", "abs"))
            for e, m in cfg.items()}


@register("material")
def build_material(cfg):
    import isaaclab.envs.mdp as mdp

    return {f"dr_material_{e}": _event(mdp.randomize_rigid_body_material, "startup", asset_cfg=_entity(e),
                                       static_friction_range=tuple(m["static"]),
                                       dynamic_friction_range=tuple(m["dynamic"]),
                                       restitution_range=tuple(m.get("restitution", [0, 0])), num_buckets=64)
            for e, m in cfg.items()}


@register("com")
def build_com(cfg):
    import isaaclab.envs.mdp as mdp

    return {f"dr_com_{e}": _event(mdp.randomize_rigid_body_com, "startup", asset_cfg=_entity(e),
                                  com_range={k: (-r, r) for k in ("x", "y", "z")})
            for e, r in cfg.items()}


@register("actuator_gains")
def build_actuator_gains(cfg):
    import isaaclab.envs.mdp as mdp

    return {f"dr_gains_{e}": _event(mdp.randomize_actuator_gains, "startup",
                                    asset_cfg=_entity(e, g.get("joints", ".*")),
                                    stiffness_distribution_params=_opt(g.get("stiffness")),
                                    damping_distribution_params=_opt(g.get("damping")), operation="scale")
            for e, g in cfg.items()}


@register("joints")
def build_joints(cfg):
    import isaaclab.envs.mdp as mdp

    t = {}
    for ent, by_joint in cfg.items():
        for i, (expr, g) in enumerate(by_joint.items()):
            op = g.get("operation", "abs")
            if "damping" in g or "stiffness" in g:
                t[f"dr_joint_gains_{ent}_{i}"] = _event(
                    mdp.randomize_actuator_gains, "startup", asset_cfg=_entity(ent, expr),
                    stiffness_distribution_params=_opt(g.get("stiffness")),
                    damping_distribution_params=_opt(g.get("damping")), operation=op)
            if "friction" in g or "armature" in g:
                t[f"dr_joint_params_{ent}_{i}"] = _event(
                    mdp.randomize_joint_parameters, "startup", asset_cfg=_entity(ent, expr),
                    friction_distribution_params=_opt(g.get("friction")),
                    armature_distribution_params=_opt(g.get("armature")), operation=op)
    return t


@register("custom")
def build_custom(cfg):
    t = {}
    for name, c in cfg.items():
        mod, fn = c["func"].split(":")
        t[name] = _event(getattr(importlib.import_module(mod), fn), c.get("mode", "reset"), **c.get("params", {}))
    return t


def _opt(r):
    return tuple(r) if r is not None else None
