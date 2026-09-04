# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real starting-state grounding for Phase A, replacing the screenshot mechanism.

Two data sources:

  - **Root node**: ``RoboLab/assets/objects/polaris/initial_conditions.json`` -- the 100-preset
    pool. Schema: ``{"instruction": str, "poses": [ {<name>_eval: [x,y,z,qw,qx,qy,qz]}, ... ]}``.
    Object keys use task-text aliases (``latteartcup_eval``, ``cuttingboard_eval``, ``coke_eval``),
    not real scene names -- ``ROOT_POSE_KEY_ALIASES`` below maps between them, matching
    ``RoboLab/robolab/tasks/benchmark/starting_states.py``'s ``load_preset_poses``.
  - **Non-root nodes**: the real end-states JSONL a producing edge's training run collects.
    Schema: one JSON object per line, ``{"objects": {<real_object_name>: [x,y,z,qw,qx,qy,qz,
    vx,vy,vz,wx,wy,wz]}, "robot_joint_pos": [...], "robot_joint_vel": [...]}`` -- real scene
    names already, no alias table needed.

Only available once a node's producing edge has been trained and end-states collected
(``manifest.get_reset_states_path_for_children(node)`` is not ``None``). Returns ``None``
otherwise -- never fabricates data for a node nothing has run for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, FrozenSet, Iterable, Optional, Union

from .manifest import Manifest
from .predicates import find_repo_root

# Task-text alias -> real scene object name, from RoboLab's load_preset_poses. "latteartcup" is
# the task-text alias for ceramic_mug.
ROOT_POSE_KEY_ALIASES = {
    "cutting_board_a": "cuttingboard_eval",
    "ceramic_mug": "latteartcup_eval",
    "coke": "coke_eval",
}


def default_initial_conditions_path() -> Optional[Path]:
    root = find_repo_root(Path(__file__).resolve())
    if root is None:
        return None
    path = root / "RoboLab" / "assets" / "objects" / "polaris" / "initial_conditions.json"
    return path if path.is_file() else None


def _resolve_root_pose_key(object_name: str, available_keys: Iterable[str]) -> Optional[str]:
    """Best-effort match from a real scene object name to an ``initial_conditions.json`` pose
    key. Tries the alias table first, then a loose suffix-stripped substring match for objects
    the table doesn't cover.
    """
    if object_name in ROOT_POSE_KEY_ALIASES and ROOT_POSE_KEY_ALIASES[object_name] in available_keys:
        return ROOT_POSE_KEY_ALIASES[object_name]
    normalized_target = object_name.replace("_", "").lower()
    for key in available_keys:
        stripped = key[:-5] if key.endswith("_eval") else key
        normalized_key = stripped.replace("_", "").lower()
        if normalized_key == normalized_target or normalized_key in normalized_target or normalized_target in normalized_key:
            return key
    return None


def _format_pose_row(name: str, values: list[float], *, extra_label: str = "") -> str:
    pos = values[:3]
    quat = values[3:7]
    pos_str = ", ".join(f"{v:.3f}" for v in pos)
    quat_str = ", ".join(f"{v:.3f}" for v in quat)
    line = f"    {name}: pos=({pos_str}) quat_wxyz=({quat_str})"
    if len(values) > 7 and extra_label:
        vel_str = ", ".join(f"{v:.3f}" for v in values[7:])
        line += f" {extra_label}=({vel_str})"
    return line


def summarize_root_starting_state(
    scene_objects: Iterable[str],
    *,
    num_samples: int = 5,
    path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Real, numeric summary of a handful of the 100 root preset poses, for scene_objects.

    Returns ``None`` (never fabricates) if the file can't be found/parsed, or if none of
    ``scene_objects`` have a resolvable pose key -- callers fall back to a text-only note.
    """
    resolved_path = Path(path) if path else default_initial_conditions_path()
    if resolved_path is None or not resolved_path.is_file():
        return None
    try:
        data = json.loads(resolved_path.read_text())
        all_poses = data["poses"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    if not all_poses:
        return None

    scene_objects = list(scene_objects)
    available_keys = all_poses[0].keys()
    key_map = {
        obj: _resolve_root_pose_key(obj, available_keys) for obj in scene_objects
    }
    if not any(key_map.values()):
        return None  # none of this scene's objects have any resolvable pose data

    # Evenly-spaced (deterministic, not random) sample so prompts stay reproducible.
    n = max(1, min(num_samples, len(all_poses)))
    stride = max(1, len(all_poses) // n)
    sample_indices = list(range(0, len(all_poses), stride))[:n]

    lines = [
        f"{len(all_poses)} preset starting layouts exist in total; showing {len(sample_indices)} "
        f"representative samples (poses are pos=(x,y,z) meters, quat_wxyz=(w,x,y,z)):"
    ]
    for idx in sample_indices:
        row = all_poses[idx]
        lines.append(f"  sample #{idx}:")
        for obj in scene_objects:
            key = key_map.get(obj)
            if key is None or key not in row:
                lines.append(f"    {obj}: (no pose data found for this object name)")
                continue
            lines.append(_format_pose_row(obj, row[key]))
    return "\n".join(lines)


def summarize_node_starting_state(
    node: FrozenSet[str],
    manifest: Manifest,
    scene_objects: Iterable[str],
    *,
    num_samples: int = 5,
) -> Optional[str]:
    """Real, numeric summary of a handful of a non-root node's actually-collected end-states.

    Returns ``None`` if ``manifest.get_reset_states_path_for_children(node)`` doesn't point at a
    real, readable JSONL file -- never fabricates data for a node nothing has run for.
    """
    reset_states_path = manifest.get_reset_states_path_for_children(node)
    if not reset_states_path:
        return None
    path = Path(reset_states_path)
    if not path.is_file():
        return None

    scene_objects = list(scene_objects)
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= num_samples * 20:  # cap how much of a huge file we ever parse
                break
    if not rows:
        return None

    n = max(1, min(num_samples, len(rows)))
    stride = max(1, len(rows) // n)
    sample_indices = list(range(0, len(rows), stride))[:n]

    lines = [
        f"{len(rows)}+ real captured end-states are available from actually training the edge(s) "
        f"that produced this node; showing {len(sample_indices)} representative samples (this is "
        f"ACTUAL data from a trained policy reaching this node, not a random preset):"
    ]
    for idx in sample_indices:
        row = rows[idx]
        objects = row.get("objects", {})
        lines.append(f"  sample #{idx}:")
        for obj in scene_objects:
            if obj not in objects:
                lines.append(f"    {obj}: (not captured in this row)")
                continue
            lines.append(_format_pose_row(obj, objects[obj], extra_label="vel"))
    return "\n".join(lines)


def summarize_starting_state(
    node: FrozenSet[str],
    manifest: Manifest,
    scene_objects: Iterable[str],
    *,
    num_samples: int = 5,
    initial_conditions_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Dispatch to the root or non-root real-data summarizer, whichever applies to ``node``."""
    node = frozenset(node)
    if not node:
        return summarize_root_starting_state(
            scene_objects, num_samples=num_samples, path=initial_conditions_path
        )
    return summarize_node_starting_state(node, manifest, scene_objects, num_samples=num_samples)
