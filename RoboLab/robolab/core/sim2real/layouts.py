# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start layouts for a reconstructed scene: preset 0 is the photographed layout, the rest are sampled, with a
train / test split and an env switch for the photographed layout. Feed them to reset_pose_to_presets.

    PRESETS = sample_layouts(101, objects={
        "bread": {"photo": (0.378, -0.14, -34), "x": (0.30, 0.72), "y": (-0.30, 0.30), "r": 0.085},
        "plate": {"photo": (0.69, 0.045, 0), "x": (0.40, 0.75), "y": (-0.25, 0.30), "r": 0.12},
        "mug":   {"photo": (0.536, -0.083, -30), "jitter": 0.02, "r": 0.04}},
        z=0.05, keepouts=[(0.419, 0.148, 0.19)], min_dist={("bread", "plate"): 0.24})
    indices = preset_indices(train=range(1, 81), test=range(81, 101))

An object with "jitter" stays near its photographed pose (any yaw when jittered); one with "x" / "y" ranges is
placed anywhere in them at a random yaw. "r" is its footprint radius for the clearance checks.
"""
import math
import os
import random


def _quat_wxyz(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def sample_layouts(n, objects, z, keepouts=(), min_dist=None, jitter_on=None, seed=0, max_tries=100000):
    """n presets, each a list of (x, y, z, qw, qx, qy, qz) in objects' order. keepouts are (x, y, r) discs
    (fixtures) no object may overlap; min_dist {(a, b): d} adds centre distances between pairs. jitter_on
    (default: ROBOLAB_DR != 0) turns the "jitter" objects' jitter on."""
    if jitter_on is None:
        jitter_on = os.environ.get("ROBOLAB_DR", "1") != "0"
    min_dist = min_dist or {}
    names = list(objects)
    rng = random.Random(seed)
    photo = [(*o["photo"][:2], z, *_quat_wxyz(math.radians(o["photo"][2]))) for o in objects.values()]
    presets = [photo]
    for _ in range(max_tries):
        if len(presets) >= n:
            break
        placed = []
        for name in names:
            o = objects[name]
            if "x" in o:
                x, y, yaw = rng.uniform(*o["x"]), rng.uniform(*o["y"]), rng.uniform(-math.pi, math.pi)
            else:
                j = o.get("jitter", 0.0) if jitter_on else 0.0
                x, y = o["photo"][0] + rng.uniform(-j, j), o["photo"][1] + rng.uniform(-j, j)
                yaw = rng.uniform(-math.pi, math.pi) if j else math.radians(o["photo"][2])
            placed.append((name, x, y, yaw))
        if _clear(placed, objects, keepouts, min_dist):
            presets.append([(x, y, z, *_quat_wxyz(yaw)) for _, x, y, yaw in placed])
    if len(presets) < n:
        raise RuntimeError(f"sample_layouts: only {len(presets)} of {n} layouts fit; widen the ranges")
    return presets


def _clear(placed, objects, keepouts, min_dist):
    for i, (a, xa, ya, _) in enumerate(placed):
        ra = objects[a].get("r", 0.0)
        if any(math.hypot(xa - kx, ya - ky) < kr + ra for kx, ky, kr in keepouts):
            return False
        for b, xb, yb, _ in placed[i + 1:]:
            need = max(ra + objects[b].get("r", 0.0), min_dist.get((a, b), min_dist.get((b, a), 0.0)))
            if math.hypot(xa - xb, ya - yb) < need:
                return False
    return True


def preset_indices(train, test):
    """Preset pool for this run: [0] with ROBOLAB_PHOTO_LAYOUT=1, test with ROBOLAB_EVAL_ONLY=1, else train."""
    if os.environ.get("ROBOLAB_PHOTO_LAYOUT", "0") == "1":
        return [0]
    if os.environ.get("ROBOLAB_EVAL_ONLY", "0") == "1":
        return list(test)
    return list(train)
