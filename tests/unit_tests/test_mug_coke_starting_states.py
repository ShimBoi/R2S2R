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

"""Pure-Python checks for RoboLab/robolab/tasks/benchmark/starting_states.py.

No IsaacLab dependency is involved (this module only touches json + a path
constant), so it is imported for real rather than mocked.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOLAB_ROOT = REPO_ROOT / "RoboLab"


def _import_load_preset_poses():
    sys.path.insert(0, str(ROBOLAB_ROOT))
    try:
        from robolab.tasks.benchmark.starting_states import load_preset_poses
    finally:
        sys.path.remove(str(ROBOLAB_ROOT))
    return load_preset_poses


def test_load_preset_poses_shape_and_order():
    load_preset_poses = _import_load_preset_poses()
    poses = load_preset_poses()

    assert len(poses) == 100, "initial_conditions.json should contribute 100 rows"
    for row in poses:
        # [cutting_board_a_pose, ceramic_mug_pose, coke_pose] -- order must match
        # the asset_cfg order used by SharedRandomization in _mug_coke_common.py.
        assert len(row) == 3
        for pose in row:
            assert len(pose) == 7, "each pose is (x, y, z, qw, qx, qy, qz)"
            assert all(isinstance(v, float) for v in pose)


def test_load_preset_poses_matches_raw_json_keys():
    import json

    load_preset_poses = _import_load_preset_poses()
    poses = load_preset_poses()

    with open(
        ROBOLAB_ROOT / "assets" / "objects" / "polaris" / "initial_conditions.json"
    ) as f:
        raw = json.load(f)["poses"]

    assert len(poses) == len(raw)
    for parsed_row, raw_row in zip(poses, raw):
        assert parsed_row[0] == tuple(raw_row["cuttingboard_eval"])
        assert parsed_row[1] == tuple(raw_row["latteartcup_eval"])
        assert parsed_row[2] == tuple(raw_row["coke_eval"])
        # cleaner_eval must be dropped, not silently included anywhere.
        assert "cleaner_eval" not in raw_row or len(parsed_row) == 3
