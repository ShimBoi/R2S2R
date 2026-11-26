import glob
import json
import logging
import os

import numpy as np
import torch

def scene_names(scene):
    names = {f"articulation/{k}": list(a.joint_names) for k, a in scene.articulations.items()}
    names.update({f"rigid_object/{k}": [] for k in scene.rigid_objects})
    return names


def capture(scene, ids):
    out = {}
    for kind, assets in scene.get_state(is_relative=True).items():
        for name, value in assets.items():
            if isinstance(value, dict):
                for field, t in value.items():
                    out[f"{kind}/{name}/{field}"] = t[ids].float().cpu().numpy()
            else:
                out[f"{kind}/{name}"] = value[ids].float().cpu().numpy()
    robot = scene.articulations.get("robot")
    if robot is not None and "base_link" in robot.body_names:
        b = robot.body_names.index("base_link")
        pos = robot.data.body_pos_w.torch[ids, b] - scene.env_origins[ids]
        quat = robot.data.body_quat_w.torch[ids, b]
        out["extra/ee_pose"] = torch.cat([pos, quat], -1).float().cpu().numpy()
    return out


def concat(chunks):
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def write(directory, filename, arrays, names):
    os.makedirs(directory, exist_ok=True)
    names_path = os.path.join(directory, "names.json")
    if not os.path.exists(names_path):
        with open(names_path, "w") as f:
            json.dump(names, f, indent=1)
    path = os.path.join(directory, filename)
    np.savez_compressed(path, **arrays)
    return path


def next_file_number(directory, writer_id):
    files = glob.glob(os.path.join(directory, f"*_{writer_id}.npz"))
    return 1 + max((int(os.path.basename(f).split("_")[0]) for f in files), default=-1)


def read_success_end_states(dirs):
    chunks, names = [], None
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not files:
            logging.getLogger(__name__).warning(f"no success end states in {d}")
            continue
        with open(os.path.join(d, "names.json")) as f:
            dir_names = json.load(f)
        if names is not None and dir_names != names:
            raise ValueError(f"{d}/names.json does not match {dirs[0]}/names.json")
        names = dir_names
        for path in files:
            with np.load(path) as z:
                chunks.append({k: z[k] for k in z.files})
    if not chunks:
        return None, None
    return concat(chunks), names


def load_captured_init_states(env, env_ids):
    captured = env._captured_init_states
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = next(iter(captured.values())).shape[0]
    pick = torch.randint(n, (len(env_ids),), device=env.device)
    state = {kind: {} for kind in ("articulation", "cable_object", "deformable_object", "rigid_object", "gripper")}
    for key, v in captured.items():
        kind, name, *field = key.split("/")
        if kind not in state:
            continue
        t = v[pick]
        if field and field[0].endswith("velocity"):
            t = torch.zeros_like(t)
        if field:
            state[kind].setdefault(name, {})[field[0]] = t
        else:
            state[kind][name] = t
    env.scene.reset_to(state, env_ids, is_relative=True)
