"""Event functions for sim-to-real domain randomization. Each takes (env, env_ids, **params) and is wrapped
into an EventTerm by a plan section (builders.py); a scene can also call them from its own event terms.

What each mechanism can and cannot do (Isaac Lab 3, tiled cameras):
- Camera pose: per env, by writing each camera prim's local pose (env-relative for fixed cameras,
  link-relative for mounted ones), so wrist and external cameras both work.
- Camera intrinsics: the tiled renderer uses one projection for all envs, so per-env FOV is done in image
  space ("zoom": crop-and-resize about the centre, < 1 zooms out with edge padding).
- Lights: prims under an env (cloned per env) are sampled per env; stage-wide ones for everyone at each reset.
  "color" shifts warm / cool; "pos_m" moves the light, which changes shadow directions.
- Background: the dome is one stage-wide prim, so the image variant and intensity are drawn once per reset
  call; with partial resets the dome changes under running episodes.
"""
import math

import torch
from isaaclab.utils.modifiers import ModifierCfg


def _u(lo, hi, shape, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def _ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _torch(a):
    return a.torch if hasattr(a, "torch") else __import__("warp").to_torch(a)


def _quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], -1)


def _small_rot(n, max_rad, device):
    v = _u(-max_rad, max_rad, (n, 3), device)
    ang = v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return torch.cat([v / ang * torch.sin(ang / 2), torch.cos(ang / 2)], 1)


def perturb_camera_poses(env, env_ids, cameras):
    import warp as wp

    ids = _ids(env, env_ids)
    nominal = env.__dict__.setdefault("_dr_camera_nominal", {})
    for name, rng in cameras.items():
        cam = env.scene.sensors[name]
        view = cam._view
        if name not in nominal:
            t, q = view.get_local_poses()
            nominal[name] = (_torch(t).clone(), _torch(q).clone())
        t0, q0 = nominal[name]
        n = len(ids)
        t = t0[ids] + _u(-rng.get("pos_m", 0.0), rng.get("pos_m", 0.0), (n, 3), env.device)
        q = _quat_mul(q0[ids], _small_rot(n, math.radians(rng.get("rot_deg", 0.0)), env.device))
        view.set_local_poses(
            wp.from_torch(t.contiguous().float(), dtype=wp.vec3f),
            wp.from_torch(q.contiguous().float(), dtype=wp.vec4f),
            wp.from_torch(ids.to(torch.int32).contiguous(), dtype=wp.int32),
        )


_IMAGE_STATE = {}


def _image_state(env):
    return _IMAGE_STATE.setdefault(id(env), {})


def _sample_image(env, ids, name, rng):
    st = _image_state(env)[name]
    n, dev = len(ids), env.device
    st["gain"][ids] = _u(*rng.get("gain", [1, 1]), (n,), dev)
    wb = rng.get("wb", 0.0)
    st["wb"][ids] = _u(1 - wb, 1 + wb, (n, 3), dev)
    st["contrast"][ids] = _u(*rng.get("contrast", [1, 1]), (n,), dev)
    st["gamma"][ids] = _u(*rng.get("gamma", [1, 1]), (n,), dev)
    st["noise"][ids] = _u(*rng.get("noise", [0, 0]), (n,), dev)
    st["zoom"][ids] = _u(*rng.get("zoom", [1, 1]), (n,), dev)
    st["blur"][ids] = (torch.rand(n, device=dev) < rng.get("blur_p", 0.0)).float()


def _augment(obs, name, env_key):
    st = _IMAGE_STATE[env_key][name]
    x = obs.float()
    n, h, w, _ = x.shape
    z = st["zoom"]
    if not torch.all(z == 1):
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0] = 1 / z
        theta[:, 1, 1] = 1 / z
        grid = torch.nn.functional.affine_grid(theta, (n, 3, h, w), align_corners=False)
        x = torch.nn.functional.grid_sample(x.permute(0, 3, 1, 2), grid, padding_mode="border",
                                            align_corners=False).permute(0, 2, 3, 1)
    if st["blur"].any():
        k = torch.tensor([1.0, 2.0, 1.0], device=x.device)
        k = (k[:, None] * k[None, :] / 16).expand(3, 1, 3, 3)
        b = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), k, padding=1, groups=3).permute(0, 2, 3, 1)
        m = st["blur"].view(n, 1, 1, 1)
        x = b * m + x * (1 - m)
    x = 255.0 * (x / 255.0).clamp(0, 1) ** st["gamma"].view(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * st["contrast"].view(n, 1, 1, 1) + mean
    x = x * st["gain"].view(n, 1, 1, 1) * st["wb"].view(n, 1, 1, 3)
    x = x + torch.randn_like(x) * st["noise"].view(n, 1, 1, 1)
    x = x.clamp(0, 255)
    return x.round().to(obs.dtype) if not obs.is_floating_point() else x


def install_image_aug(env, env_ids, image, group="image_obs"):
    om = env.observation_manager
    names, cfgs = om._group_obs_term_names[group], om._group_obs_term_cfgs[group]
    st = _image_state(env)
    n, dev = env.num_envs, env.device
    for name, cfg in zip(names, cfgs):
        if name not in image:
            continue
        st[name] = {k: torch.ones(n, device=dev) for k in ("gain", "contrast", "gamma", "zoom")}
        st[name].update(wb=torch.ones(n, 3, device=dev), noise=torch.zeros(n, device=dev), blur=torch.zeros(n, device=dev))
        cfg.modifiers = list(cfg.modifiers or []) + [ModifierCfg(func=_augment, params={"name": name, "env_key": id(env)})]
        _sample_image(env, torch.arange(n, device=dev), name, image[name])
    missing = set(image) - set(names)
    if missing:
        print(f"[sim2real] image augmentation: no observation term for {sorted(missing)}", flush=True)


def resample_image_aug(env, env_ids, image):
    ids = _ids(env, env_ids)
    for name, rng in image.items():
        if name in _image_state(env):
            _sample_image(env, ids, name, rng)


_LIGHTS = {}


def randomize_lights(env, env_ids, lights):
    import re

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    key = id(env)
    if key not in _LIGHTS:
        found = []
        for prim in stage.Traverse():
            if prim.GetName() in lights and prim.HasAttribute("inputs:intensity"):
                m = re.search(r"/env_(\d+)/", str(prim.GetPath()))
                ops = [o for o in UsdGeom.Xformable(prim).GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                col = prim.GetAttribute("inputs:color").Get() if prim.HasAttribute("inputs:color") else None
                found.append(dict(prim=prim, name=prim.GetName(), env=int(m.group(1)) if m else None,
                                  intensity=float(prim.GetAttribute("inputs:intensity").Get()),
                                  color=tuple(col) if col is not None else None,
                                  op=ops[0] if ops else None, pos=Gf.Vec3d(ops[0].Get()) if ops else None))
        _LIGHTS[key] = found
        print(f"[sim2real] lights: {len(found)} prims ({sorted({f['name'] for f in found})})", flush=True)
    ids = set(_ids(env, env_ids).tolist())
    rand = lambda lo, hi: lo + (hi - lo) * torch.rand(1).item()
    for f in _LIGHTS[key]:
        if f["env"] is not None and f["env"] not in ids:
            continue
        rng = lights[f["name"]]
        f["prim"].GetAttribute("inputs:intensity").Set(f["intensity"] * rand(*rng.get("intensity", [1, 1])))
        c = rng.get("color", 0.0)
        if f["color"] is not None and c:
            warm = rand(-c, c)
            mult = (1 + warm, 1 + rand(-c, c) / 3, 1 - warm)
            f["prim"].GetAttribute("inputs:color").Set(Gf.Vec3f(*(min(1.0, v * m) for v, m in zip(f["color"], mult))))
        d = rng.get("pos_m", 0.0)
        if f["op"] is not None and d:
            f["op"].Set(f["pos"] + Gf.Vec3d(rand(-d, d), rand(-d, d), rand(-d, d)))


def randomize_background(env, env_ids, files, intensity, prim_path="/World/background"):
    import random

    import omni.usd
    from pxr import Sdf

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for attr_name in ("inputs:texture:file", "texture:file"):
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(Sdf.AssetPath(random.choice(files)))
            break
    prim.GetAttribute("inputs:intensity").Set(float(random.uniform(*intensity)))


def place_rigid_fixture(env, env_ids, entities, pose, pos_m=0.0, yaw_deg=0.0):
    """Reset a "rigid_links" articulated object (base first, then jointed links, all sharing the object frame) to
    its closed pose at pose = (x, y, z, yaw_deg) in the env frame, jittered by +-pos_m in x / y and +-yaw_deg.
    Every link gets the same pose, so joints stay satisfied and moving links reset closed. The base must be
    built with "base_mass" (a heavy dynamic body): on GPU PhysX a pose write to a kinematic body updates the
    reported pose but not the simulated one, and there is no kinematic-target setter. plan.build_events runs
    it after the task's other reset terms (reset_pose_to_presets(reset_to_default_otherwise=True) would put
    the links back to their default poses). The per-env pose is kept in fixture_poses(env)[entities[0]] as
    (x, y, z, yaw_rad). New fixtures should use articulated_usd root "fixed" / "floating" and fixtures.py."""
    ids = _ids(env, env_ids)
    n, dev = len(ids), env.device
    p = torch.tensor(pose[:3], device=dev).repeat(n, 1)
    p[:, :2] += _u(-pos_m, pos_m, (n, 2), dev)
    yaw = math.radians(pose[3]) + _u(-math.radians(yaw_deg), math.radians(yaw_deg), (n,), dev)
    q = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], 1)
    root = torch.cat([p + env.scene.env_origins[ids], q], 1)
    for name in entities:
        ent = env.scene[name]
        ent.write_root_pose_to_sim_index(root_pose=root, env_ids=ids)
        ent.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    store = fixture_poses(env).setdefault(entities[0], torch.tensor([[*pose[:3], math.radians(pose[3])]], device=dev).repeat(env.num_envs, 1))
    store[ids] = torch.cat([p, yaw[:, None]], 1)


def fixture_poses(env):
    return env.__dict__.setdefault("_dr_fixture_poses", {})
