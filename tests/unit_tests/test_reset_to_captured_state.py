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

"""Mocked unit test for RoboLab/robolab/core/events/reset_pose.py::reset_to_captured_state.

IsaacLab is not installed here, so isaaclab.* is stubbed with the minimal
surface reset_pose.py touches at import time, and env.scene[...] returns fake
RigidObject/Articulation-like assets that just record the tensors written to
them so the position/velocity/joint-state math can be checked directly.
"""

import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESET_POSE_PATH = (
    REPO_ROOT / "RoboLab" / "robolab" / "core" / "events" / "reset_pose.py"
)


@pytest.fixture
def reset_pose_module(monkeypatch):
    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    isaaclab = _mod("isaaclab")
    isaaclab.__path__ = []
    isaaclab_sim = _mod("isaaclab.sim")
    isaaclab_sim.__path__ = []
    isaaclab_sim_utils = _mod(
        "isaaclab.sim.utils", find_matching_prims=lambda *a, **k: []
    )
    isaaclab_utils = _mod("isaaclab.utils", configclass=lambda cls: cls)
    isaaclab_utils_math = _mod(
        "isaaclab.utils.math",
        sample_uniform=lambda *a, **k: torch.zeros(1),
        quat_from_euler_xyz=lambda *a, **k: torch.zeros(4),
        quat_mul=lambda *a, **k: torch.zeros(4),
    )

    class _DummyRigidObject:
        pass

    class _DummyArticulation:
        pass

    isaaclab_assets = _mod(
        "isaaclab.assets",
        Articulation=_DummyArticulation,
        RigidObject=_DummyRigidObject,
    )

    class _DummyManagerBasedEnv:
        pass

    isaaclab_envs = _mod("isaaclab.envs", ManagerBasedEnv=_DummyManagerBasedEnv)

    class _DummyEventTermCfg:
        def __init__(self, *a, **k):
            pass

    class _DummySceneEntityCfg:
        def __init__(self, name=None, *a, **k):
            self.name = name

    isaaclab_managers = _mod(
        "isaaclab.managers",
        EventTermCfg=_DummyEventTermCfg,
        SceneEntityCfg=_DummySceneEntityCfg,
    )

    robolab = _mod("robolab")
    robolab.__path__ = []
    robolab_constants = _mod("robolab.constants", VERBOSE=False, DEBUG=False)
    robolab_core = _mod("robolab.core")
    robolab_core.__path__ = []
    robolab_core_utils = _mod("robolab.core.utils")
    robolab_core_utils.__path__ = []
    robolab_core_utils_usd_utils = _mod(
        "robolab.core.utils.usd_utils", get_dimensions=lambda prim: [0.1, 0.1, 0.1]
    )

    for name, mod in {
        "isaaclab": isaaclab,
        "isaaclab.sim": isaaclab_sim,
        "isaaclab.sim.utils": isaaclab_sim_utils,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.math": isaaclab_utils_math,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.managers": isaaclab_managers,
        "robolab": robolab,
        "robolab.constants": robolab_constants,
        "robolab.core": robolab_core,
        "robolab.core.utils": robolab_core_utils,
        "robolab.core.utils.usd_utils": robolab_core_utils_usd_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    spec = importlib.util.spec_from_file_location(
        "robolab.core.events.reset_pose", RESET_POSE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeAsset:
    """Records the tensors written to it, like RigidObject/Articulation would."""

    def __init__(self, device="cpu"):
        self.device = device
        self.last_pose = None
        self.last_velocity = None
        self.last_pose_env_ids = None

    def write_root_pose_to_sim(self, pose, env_ids):
        self.last_pose = pose.clone()
        self.last_pose_env_ids = env_ids.clone()

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.last_velocity = velocity.clone()


class _FakeRobot:
    def __init__(self, device="cpu"):
        self.device = device
        self.last_joint_pos = None
        self.last_joint_vel = None

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.last_joint_pos = joint_pos.clone()
        self.last_joint_vel = joint_vel.clone()


class _FakeScene(dict):
    def __init__(self, env_origins):
        super().__init__()
        self.env_origins = env_origins


class _FakeEnv:
    def __init__(self, num_envs):
        self.env_origins = torch.tensor([[10.0, 20.0, 0.0]] * num_envs)
        self.scene = _FakeScene(self.env_origins)


def test_reset_to_captured_state_writes_offset_pose_and_velocity(
    reset_pose_module, monkeypatch
):
    monkeypatch.setattr(
        random, "seed", random.seed
    )  # no-op, keeps randint deterministic path visible
    torch.manual_seed(0)

    env = _FakeEnv(num_envs=2)
    board = _FakeAsset()
    mug = _FakeAsset()
    coke = _FakeAsset()
    robot = _FakeRobot()
    env.scene["cutting_board_a"] = board
    env.scene["ceramic_mug"] = mug
    env.scene["coke"] = coke
    env.scene["robot"] = robot

    captured_states = [
        {
            "objects": {
                "cutting_board_a": [1, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ceramic_mug": [4, 5, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "coke": [7, 8, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            "robot_joint_pos": [0.1] * 7,
            "robot_joint_vel": [0.2] * 7,
        }
    ]

    env_ids = torch.tensor([0, 1])
    reset_pose_module.reset_to_captured_state(
        env, env_ids, captured_states, robot_name="robot"
    )

    # Only one captured state exists, so every env must land on it regardless
    # of the random sample index.
    expected_board_pos = torch.tensor([
        [1.0 + 10.0, 2.0 + 20.0, 3.0],
        [1.0 + 10.0, 2.0 + 20.0, 3.0],
    ])
    assert torch.allclose(board.last_pose[:, 0:3], expected_board_pos)
    assert torch.allclose(
        board.last_pose[:, 3:7], torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    )
    assert torch.allclose(board.last_velocity, torch.zeros(2, 6))

    expected_mug_pos = torch.tensor([
        [4.0 + 10.0, 5.0 + 20.0, 6.0],
        [4.0 + 10.0, 5.0 + 20.0, 6.0],
    ])
    assert torch.allclose(mug.last_pose[:, 0:3], expected_mug_pos)

    expected_coke_pos = torch.tensor([
        [7.0 + 10.0, 8.0 + 20.0, 9.0],
        [7.0 + 10.0, 8.0 + 20.0, 9.0],
    ])
    assert torch.allclose(coke.last_pose[:, 0:3], expected_coke_pos)

    assert torch.allclose(robot.last_joint_pos, torch.full((2, 7), 0.1))
    assert torch.allclose(robot.last_joint_vel, torch.full((2, 7), 0.2))


def test_reset_to_captured_state_samples_across_multiple_rows(reset_pose_module):
    """With many captured rows and many envs, both distinct positions should
    show up across a large-enough env_ids batch (sanity check against a
    constant-row bug, not a rigorous distribution test)."""
    env = _FakeEnv(num_envs=50)
    board = _FakeAsset()
    robot = _FakeRobot()
    env.scene["cutting_board_a"] = board
    env.scene["robot"] = robot

    captured_states = [
        {
            "objects": {
                "cutting_board_a": [float(i), 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            "robot_joint_pos": [0.0] * 7,
            "robot_joint_vel": [0.0] * 7,
        }
        for i in range(5)
    ]

    env_ids = torch.arange(50)
    reset_pose_module.reset_to_captured_state(
        env,
        env_ids,
        captured_states,
        object_names=["cutting_board_a"],
        robot_name="robot",
    )

    distinct_x = set((board.last_pose[:, 0] - 10.0).round().tolist())
    assert len(distinct_x) > 1, (
        "expected sampling across multiple captured rows, got a constant row"
    )
    assert distinct_x.issubset({0.0, 1.0, 2.0, 3.0, 4.0})
