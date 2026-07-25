# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared starting-state pool for the mug+coke long-horizon task files (see _mug_coke_common.py)."""

import json

from robolab.constants import ASSET_DIR

_INITIAL_CONDITIONS_PATH = f"{ASSET_DIR}/objects/polaris/initial_conditions.json"


def load_preset_poses() -> list[list[tuple]]:
    """Load the 100-row PolaRiS preset pose pool.

    Returns:
        [[cutting_board_a_pose, ceramic_mug_pose, coke_pose], ...] -- 100 rows,
        cleaner_eval dropped. Pose order matches the asset_cfg order
        ["cutting_board_a", "ceramic_mug", "coke"] used in _mug_coke_common.py.
    """
    with open(_INITIAL_CONDITIONS_PATH) as f:
        data = json.load(f)
    return [
        [
            tuple(row["cuttingboard_eval"]),
            tuple(row["latteartcup_eval"]),
            tuple(row["coke_eval"]),
        ]
        for row in data["poses"]
    ]
