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

"""Mocked unit tests for the long-horizon subtask-handoff plumbing added to
rlinf/envs/isaaclab/tasks/robolab_task.py.

IsaacLab is not installed in this environment, so rlinf/envs/isaaclab/__init__.py
(which eagerly imports isaaclab-dependent tasks) cannot be imported for real.
Instead robolab_task.py is loaded directly via importlib against a minimal fake
`rlinf.envs.isaaclab` package chain, with a faithful stand-in IsaacLabBaseEnv
(method bodies copied from the real isaaclab_env.py) so RoboLabDroidEnv's
super() calls behave identically to production.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOLAB_TASK_PATH = (
    REPO_ROOT / "rlinf" / "envs" / "isaaclab" / "tasks" / "robolab_task.py"
)


class _FakeIsaaclabBaseEnv:
    """Stand-in for rlinf/envs/isaaclab/isaaclab_env.py::IsaaclabBaseEnv.

    Only the methods RoboLabDroidEnv calls via super() are reproduced, with
    bodies copied verbatim from the real file so behavior matches production.
    """

    def _init_metrics(self):
        self.success_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=bool).to(self.device)
        self.returns = torch.zeros(self.num_envs).to(self.device)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | (step_reward > 0)
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    @property
    def elapsed_steps(self):
        return self._elapsed_steps.to(self.device)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward
        if self.use_rel_reward:
            return reward_diff
        return reward

    def _handle_auto_reset(self, dones, _final_obs, infos):  # pragma: no cover
        raise NotImplementedError("auto-reset is not exercised in these tests")

    def _wrap_obs(self, obs):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def robolab_task_module(monkeypatch):
    fake_pkg_isaaclab = types.ModuleType("rlinf.envs.isaaclab")
    fake_pkg_isaaclab.__path__ = []
    fake_pkg_tasks = types.ModuleType("rlinf.envs.isaaclab.tasks")
    fake_pkg_tasks.__path__ = []

    isaaclab_env_module = types.ModuleType("rlinf.envs.isaaclab.isaaclab_env")
    isaaclab_env_module.IsaaclabBaseEnv = _FakeIsaaclabBaseEnv

    monkeypatch.setitem(sys.modules, "rlinf.envs.isaaclab", fake_pkg_isaaclab)
    monkeypatch.setitem(sys.modules, "rlinf.envs.isaaclab.tasks", fake_pkg_tasks)
    monkeypatch.setitem(
        sys.modules, "rlinf.envs.isaaclab.isaaclab_env", isaaclab_env_module
    )

    spec = importlib.util.spec_from_file_location(
        "rlinf.envs.isaaclab.tasks.robolab_task", ROBOLAB_TASK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_object_on_top(monkeypatch):
    """Stubs robolab.core.task.conditionals.object_on_top, which
    NoAutoResetManagerBasedRLEnv.step() imports locally on every call.

    Tests configure per-object-name return values via `.set(object_name, tensor)`
    before invoking step(); calls are recorded for assertions.
    """

    state = {"returns": {}, "calls": []}

    def _object_on_top(
        env, object, reference_object, require_gripper_detached=True, env_id=None
    ):
        state["calls"].append((object, reference_object))
        return state["returns"][object]

    fake_conditionals = types.ModuleType("robolab.core.task.conditionals")
    fake_conditionals.object_on_top = _object_on_top

    fake_robolab = types.ModuleType("robolab")
    fake_robolab.__path__ = []
    fake_core = types.ModuleType("robolab.core")
    fake_core.__path__ = []
    fake_task = types.ModuleType("robolab.core.task")
    fake_task.__path__ = []

    monkeypatch.setitem(sys.modules, "robolab", fake_robolab)
    monkeypatch.setitem(sys.modules, "robolab.core", fake_core)
    monkeypatch.setitem(sys.modules, "robolab.core.task", fake_task)
    monkeypatch.setitem(
        sys.modules, "robolab.core.task.conditionals", fake_conditionals
    )

    def set_return(object_name, tensor):
        state["returns"][object_name] = tensor

    return SimpleNamespace(set=set_return, calls=state["calls"])


def _make_fake_manager_env_cls(module):
    class _FakeManagerBasedRLEnv:
        def __init__(self, num_envs, device="cpu"):
            self.num_envs = num_envs
            self.device = device
            self._reset_idx = lambda env_ids: None
            self._next_step_result = None
            self._next_reset_result = None

        def step(self, action):
            return self._next_step_result

        def reset(self, *args, **kwargs):
            return self._next_reset_result

    class Env(module.NoAutoResetManagerBasedRLEnv, _FakeManagerBasedRLEnv):
        pass

    return Env


SUBTASKS_CFG = {
    "object_1": "ceramic_mug",
    "object_2": "coke",
    "surface": "cutting_board_a",
    "instruction_1": "pick up the mug",
    "instruction_2": "pick up the coke",
}


# ---------------------------------------------------------------------------
# NoAutoResetManagerBasedRLEnv: backward compatibility (no subtasks configured)
# ---------------------------------------------------------------------------


def test_no_subtasks_cfg_step_is_unchanged(robolab_task_module, fake_object_on_top):
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    assert env._subtasks_cfg is None  # class default

    raw_extras = {"some_existing_key": 1}
    terminated = torch.tensor([False, True])
    time_out = torch.tensor([False, False])
    env._next_step_result = ("obs", "reward", terminated, time_out, raw_extras)

    original_reset_idx = env._reset_idx
    obs, reward, out_terminated, out_time_out, extras = env.step(action=None)

    assert obs == "obs" and reward == "reward"
    assert torch.equal(out_terminated, terminated)
    assert torch.equal(out_time_out, time_out)
    assert extras is raw_extras and "task_descriptions" not in extras
    assert not fake_object_on_top.calls  # object_on_top never invoked
    # _reset_idx suppressed during super().step() and restored after.
    assert env._reset_idx is original_reset_idx


def test_no_subtasks_cfg_reset_is_unchanged(robolab_task_module):
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    env._next_reset_result = ("obs", {"info": True})

    obs, info = env.reset(env_ids=torch.tensor([0]))
    assert obs == "obs" and info == {"info": True}
    assert not hasattr(env, "_subtask_idx")  # ratchet never initialized


# ---------------------------------------------------------------------------
# subtask_1 / subtask_2 modes: single-instruction termination
# ---------------------------------------------------------------------------


def test_subtask_1_mode_terminates_on_object_1_only(
    robolab_task_module, fake_object_on_top
):
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    env._subtasks_cfg = SUBTASKS_CFG
    env._mode = "subtask_1"

    fake_object_on_top.set("ceramic_mug", torch.tensor([True, False]))
    fake_object_on_top.set("coke", torch.tensor([False, False]))

    terminated = torch.tensor([False, False])
    time_out = torch.tensor([False, False])
    env._next_step_result = ("obs", "reward", terminated, time_out, {})

    _, _, out_terminated, _, extras = env.step(action=None)

    assert torch.equal(out_terminated, torch.tensor([True, False]))
    assert extras["task_descriptions"] == ["pick up the mug", "pick up the mug"]
    assert torch.equal(extras["subtask_1_success"], torch.tensor([True, False]))
    assert torch.equal(extras["subtask_2_success"], torch.tensor([False, False]))
    assert "current_subtask_idx" not in extras  # only set in "full" mode


def test_subtask_2_mode_terminates_on_object_2_only(
    robolab_task_module, fake_object_on_top
):
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    env._subtasks_cfg = SUBTASKS_CFG
    env._mode = "subtask_2"

    fake_object_on_top.set(
        "ceramic_mug", torch.tensor([True, True])
    )  # irrelevant in this mode
    fake_object_on_top.set("coke", torch.tensor([False, True]))

    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        {},
    )

    _, _, out_terminated, _, extras = env.step(action=None)

    assert torch.equal(out_terminated, torch.tensor([False, True]))
    assert extras["task_descriptions"] == ["pick up the coke", "pick up the coke"]


# ---------------------------------------------------------------------------
# full mode: the VLM-simulated ratchet handoff
# ---------------------------------------------------------------------------


def test_full_mode_ratchet_advances_and_terminates(
    robolab_task_module, fake_object_on_top
):
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    env._subtasks_cfg = SUBTASKS_CFG
    env._mode = "full"
    env._next_reset_result = ("obs", {})
    env.reset()  # initializes the ratchet: idx=[0,0], descriptions=[I1,I1]

    assert torch.equal(env._subtask_idx, torch.tensor([0, 0]))
    assert env._task_descriptions == ["pick up the mug", "pick up the mug"]

    # Step 1: env0 completes subtask 1, env1 does not.
    fake_object_on_top.set("ceramic_mug", torch.tensor([True, False]))
    fake_object_on_top.set("coke", torch.tensor([False, False]))
    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        {},
    )
    _, _, terminated, _, extras = env.step(action=None)

    assert torch.equal(env._subtask_idx, torch.tensor([1, 0]))
    assert extras["task_descriptions"] == ["pick up the coke", "pick up the mug"]
    assert torch.equal(terminated, torch.tensor([False, False]))
    assert torch.equal(extras["current_subtask_idx"], torch.tensor([1, 0]))

    # Step 2: env0 completes subtask 2 (episode ends); env1's cup now appears
    # (bumped-object noise) but must NOT re-trigger since env0 already advanced
    # and env1 never advanced past idx 0 for object_1 in this step either.
    fake_object_on_top.set("ceramic_mug", torch.tensor([False, True]))
    fake_object_on_top.set("coke", torch.tensor([True, False]))
    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        {},
    )
    _, _, terminated, _, extras = env.step(action=None)

    assert torch.equal(env._subtask_idx, torch.tensor([2, 1]))
    assert torch.equal(terminated, torch.tensor([True, False]))
    assert extras["task_descriptions"] == ["pick up the coke", "pick up the coke"]


def test_full_mode_ratchet_is_irreversible(robolab_task_module, fake_object_on_top):
    """Once advanced past subtask 1, a later False reading for object_1 must not
    revert the ratchet (mirrors: a VLM that already moved on won't re-issue the
    cup instruction just because the cup got bumped)."""
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=1)
    env._subtasks_cfg = SUBTASKS_CFG
    env._mode = "full"
    env._next_reset_result = ("obs", {})
    env.reset()

    fake_object_on_top.set("ceramic_mug", torch.tensor([True]))
    fake_object_on_top.set("coke", torch.tensor([False]))
    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False]),
        torch.tensor([False]),
        {},
    )
    env.step(action=None)
    assert torch.equal(env._subtask_idx, torch.tensor([1]))

    # object_1 now reads False again; ratchet must stay at idx 1, not drop to 0.
    fake_object_on_top.set("ceramic_mug", torch.tensor([False]))
    fake_object_on_top.set("coke", torch.tensor([False]))
    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False]),
        torch.tensor([False]),
        {},
    )
    env.step(action=None)
    assert torch.equal(env._subtask_idx, torch.tensor([1]))
    assert env._task_descriptions == ["pick up the coke"]


def test_full_mode_reset_partial_env_ids(robolab_task_module, fake_object_on_top):
    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    env._subtasks_cfg = SUBTASKS_CFG
    env._mode = "full"
    env._next_reset_result = ("obs", {})
    env.reset()

    # Advance env0 to idx 1.
    fake_object_on_top.set("ceramic_mug", torch.tensor([True, True]))
    fake_object_on_top.set("coke", torch.tensor([False, False]))
    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        {},
    )
    env.step(action=None)
    assert torch.equal(env._subtask_idx, torch.tensor([1, 1]))

    # Partial reset of env0 only (kwarg, matching venv.py's call convention).
    env._next_reset_result = ("obs", {})
    env.reset(env_ids=torch.tensor([0]))

    assert torch.equal(env._subtask_idx, torch.tensor([0, 1]))
    assert env._task_descriptions == ["pick up the mug", "pick up the coke"]


# ---------------------------------------------------------------------------
# End-state collection (save_end_state_path)
# ---------------------------------------------------------------------------


def test_end_state_collection_writes_jsonl_on_subtask_1_success(
    robolab_task_module, fake_object_on_top, monkeypatch, tmp_path
):
    class _FakeWorld:
        def get_pose(self, name, is_relative=True, env_id=None):
            return torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 0.0, 0.0, 0.0])

        def get_velocity(self, name, env_id=None):
            return torch.zeros(6)

        def get_joint_positions(self, name, env_id=None):
            return torch.zeros(7)

        def get_joint_velocity(self, name, env_id=None):
            return torch.zeros(7)

    fake_world_state = types.ModuleType("robolab.core.world.world_state")
    fake_world_state.get_world = lambda env: _FakeWorld()
    fake_world_pkg = types.ModuleType("robolab.core.world")
    fake_world_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "robolab.core.world", fake_world_pkg)
    monkeypatch.setitem(sys.modules, "robolab.core.world.world_state", fake_world_state)

    Env = _make_fake_manager_env_cls(robolab_task_module)
    env = Env(num_envs=2)
    env._subtasks_cfg = SUBTASKS_CFG
    env._mode = "subtask_1"
    out_path = tmp_path / "end_states.jsonl"
    env._save_end_state_path = str(out_path)

    fake_object_on_top.set("ceramic_mug", torch.tensor([True, False]))
    fake_object_on_top.set("coke", torch.tensor([False, False]))
    env._next_step_result = (
        "obs",
        "reward",
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        {},
    )
    env.step(action=None)

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1  # only env0 succeeded

    import json

    row = json.loads(lines[0])
    assert set(row["objects"].keys()) == {"ceramic_mug", "coke", "cutting_board_a"}
    assert row["robot_joint_pos"] == [0.0] * 7
