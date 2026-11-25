import glob
import json
import os

import numpy as np
import torch

EE_BODY = "base_link"


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
    if robot is not None and EE_BODY in robot.body_names:
        b = robot.body_names.index(EE_BODY)
        pos = robot.data.body_pos_w.torch[ids, b] - scene.env_origins[ids]
        quat = robot.data.body_quat_w.torch[ids, b]
        out["extra/ee_pose"] = torch.cat([pos, quat], -1).float().cpu().numpy()
    return out


def concat(chunks):
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def _atomic(path, write):
    tmp = f"{path}.tmp{os.getpid()}"
    write(tmp)
    os.replace(tmp, path)


def write(directory, filename, arrays, names):
    os.makedirs(directory, exist_ok=True)
    names_path = os.path.join(directory, "names.json")
    if not os.path.exists(names_path):
        _atomic(names_path, lambda p: open(p, "w").write(json.dumps(names, indent=1)))
    path = os.path.join(directory, filename)
    _atomic(path, lambda p: np.savez_compressed(open(p, "wb"), **arrays))
    return path


def next_index(directory, tag):
    files = glob.glob(os.path.join(directory, f"*_{tag}.npz"))
    return 1 + max((int(os.path.basename(f).split("_")[0]) for f in files), default=-1)
