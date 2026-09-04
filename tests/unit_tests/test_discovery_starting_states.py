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

"""Unit tests for rlinf.envs.isaaclab.tasks.discovery.starting_states.

Root grounding is tested against the REAL initial_conditions.json in this checkout (a real
regression check, not just "doesn't crash" -- confirms the alias table actually resolves real
scene object names to real pose data). Non-root grounding is tested against synthetic JSONL
files matching the real end-state schema (confirmed by reading robolab_task.py's
end-state-writing code), via tmp_path -- no real training/collection ever runs.
"""

import json

from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest
from rlinf.envs.isaaclab.tasks.discovery.starting_states import (
    default_initial_conditions_path,
    summarize_node_starting_state,
    summarize_root_starting_state,
    summarize_starting_state,
)

SCENE_OBJECTS = ["ceramic_mug", "coke", "cutting_board_a"]


def test_default_initial_conditions_path_resolves_in_this_checkout():
    path = default_initial_conditions_path()
    assert path is not None
    assert path.is_file()
    assert path.name == "initial_conditions.json"


def test_summarize_root_starting_state_real_file_resolves_real_object_names():
    summary = summarize_root_starting_state(SCENE_OBJECTS, num_samples=3)
    assert summary is not None
    # The alias table (cutting_board_a<-cuttingboard_eval, ceramic_mug<-latteartcup_eval,
    # coke<-coke_eval) must have actually resolved -- real scene object names appear with
    # real pos=(...) data, not a "no pose data found" placeholder.
    for obj in SCENE_OBJECTS:
        assert f"{obj}: pos=(" in summary
        assert f"{obj}: (no pose data found" not in summary
    assert "pos=(" in summary and "quat_wxyz=(" in summary


def test_summarize_root_starting_state_sample_count_respected():
    summary = summarize_root_starting_state(SCENE_OBJECTS, num_samples=4)
    assert summary.count("sample #") == 4


def test_summarize_root_starting_state_unknown_object_gets_placeholder_not_crash():
    summary = summarize_root_starting_state(["not_a_real_object"], num_samples=2)
    # No resolvable object at all -> None (nothing useful to show), not a crash.
    assert summary is None


def test_summarize_root_starting_state_missing_file_returns_none(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    summary = summarize_root_starting_state(SCENE_OBJECTS, path=missing)
    assert summary is None


def _write_fake_end_states(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_summarize_node_starting_state_real_schema(tmp_path):
    end_states_path = tmp_path / "end_states.jsonl"
    rows = [
        {
            "objects": {
                "cutting_board_a": [0.5, -0.2, 0.14, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ceramic_mug": [0.6, 0.1, 0.12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "coke": [0.34, -0.39, 0.13, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
            "robot_joint_pos": [0.0] * 7,
            "robot_joint_vel": [0.0] * 7,
        }
        for _ in range(10)
    ]
    _write_fake_end_states(end_states_path, rows)

    manifest = Manifest(tmp_path / "manifest.json")
    node = frozenset({"latteartcup_on_cuttingboard"})
    manifest.set_reset_states_path_for_children(node, str(end_states_path))

    summary = summarize_node_starting_state(node, manifest, SCENE_OBJECTS, num_samples=3)
    assert summary is not None
    assert "ACTUAL data from a trained policy" in summary
    assert "coke: pos=(0.340, -0.390, 0.130)" in summary


def test_summarize_node_starting_state_none_when_not_yet_trained(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    node = frozenset({"latteartcup_on_cuttingboard"})
    # Nothing was ever recorded for this node's children -- must return None, not fabricate.
    summary = summarize_node_starting_state(node, manifest, SCENE_OBJECTS)
    assert summary is None


def test_summarize_node_starting_state_none_when_path_recorded_but_file_missing(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    node = frozenset({"latteartcup_on_cuttingboard"})
    manifest.set_reset_states_path_for_children(node, str(tmp_path / "never_written.jsonl"))
    summary = summarize_node_starting_state(node, manifest, SCENE_OBJECTS)
    assert summary is None


def test_summarize_starting_state_dispatches_root_vs_node(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")

    root_summary = summarize_starting_state(frozenset(), manifest, SCENE_OBJECTS, num_samples=2)
    assert root_summary is not None
    assert "preset starting layouts exist in total" in root_summary

    # Non-root, nothing trained yet -> honestly None, not root data reused.
    node_summary = summarize_starting_state(
        frozenset({"latteartcup_on_cuttingboard"}), manifest, SCENE_OBJECTS
    )
    assert node_summary is None

    # Once real end-states are recorded for that node, dispatch picks them up.
    end_states_path = tmp_path / "end_states.jsonl"
    _write_fake_end_states(
        end_states_path,
        [
            {
                "objects": {obj: [0.1, 0.2, 0.3, 1, 0, 0, 0] for obj in SCENE_OBJECTS},
                "robot_joint_pos": [],
                "robot_joint_vel": [],
            }
        ],
    )
    manifest.set_reset_states_path_for_children(
        frozenset({"latteartcup_on_cuttingboard"}), str(end_states_path)
    )
    node_summary_after = summarize_starting_state(
        frozenset({"latteartcup_on_cuttingboard"}), manifest, SCENE_OBJECTS
    )
    assert node_summary_after is not None
    assert "ACTUAL data from a trained policy" in node_summary_after
