# Copyright 2025 The RLinf Authors.
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

import os

import torch
from omegaconf import OmegaConf

from ..isaaclab_env import IsaaclabBaseEnv


class NoAutoResetManagerBasedRLEnv:
    """Mixin that suppresses ManagerBasedRLEnv's internal auto-reset during step().

    Isaac Lab resets terminated/truncated envs inside step() before returning,
    which corrupts observations when the RLinf env worker controls episode resets
    externally (auto_reset=False). This mixin makes _reset_idx a no-op for the
    duration of each step() call so resets only happen via explicit env.reset().

    Also optionally drives a VLM-simulated long-horizon subtask handoff, controlled
    entirely by the class attributes below (set per-instance by
    RoboLabDroidEnv._make_env_function from `init_params.subtasks`/`edge_spec`/`plan`/
    `mode`/`save_end_state_path`). When none of `_subtasks_cfg`/`_edge_spec`/
    `_active_plan` is set (the default), the handoff logic is skipped entirely and the
    mixin only suppresses auto-reset.

    Two families of modes:
      - The original, hardcoded two-subtask pipeline (mug then coke -- see
        long_horizon_task_composition_plan.md): "subtask_1" | "subtask_2" | "full",
        driven by `_subtasks_cfg`. Unchanged by the generalization below.
      - The generalized, tree-discovered pipeline (PLAN.md): "single_edge" (one
        training-mode edge, driven by `_edge_spec` -- see generic_single_edge_task.py)
        and "plan" (the eval-time ratchet over however many steps a resolved plan has,
        driven by `_active_plan` -- see generic_eval_task.py). Both dispatch their
        predicate(s) by name via robolab.core.task.conditionals instead of hardcoding
        object_on_top, per PLAN.md section 5.1.
    """

    _subtasks_cfg = (
        None  # {"object_1", "object_2", "surface", "instruction_1", "instruction_2"}
    )
    _edge_spec = None  # {"predicate": str, "predicate_args": dict, "instruction": str}
    _active_plan = (
        None  # [{"predicate": str, "predicate_args": dict, "instruction": str}, ...]
    )
    _mode = "full"  # "subtask_1" | "subtask_2" | "full" | "single_edge" | "plan"
    _save_end_state_path = (
        None  # set only on the dedicated subtask_1-checkpoint collection run
    )

    def _init_ratchet(self):
        self._subtask_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._task_descriptions = [self._subtasks_cfg["instruction_1"]] * self.num_envs
        # Guards against crediting subtask_2 completion when object_2 was already
        # resting on the surface at the moment of handoff (e.g. placed out of
        # order, before instruction_2 was even issued) -- see step() below.
        self._obj2_needs_fresh_placement = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _init_plan_ratchet(self):
        """Generalized counterpart to _init_ratchet() above, for mode == "plan": an
        N-step ratchet over `_active_plan` instead of a hardcoded 2-step one. See the
        "plan" branch of step() below for how `_needs_fresh_placement` generalizes
        `_obj2_needs_fresh_placement`'s out-of-order-completion gate to N steps.
        """
        self._plan_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._task_descriptions = [
            self._active_plan[0]["instruction"]
        ] * self.num_envs
        self._needs_fresh_placement = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        if self._subtasks_cfg and self._mode == "full":
            if not hasattr(self, "_subtask_idx"):
                self._init_ratchet()
            # venv.py's _torch_worker always passes env_ids (when present) as a
            # kwarg, never positionally; args[0] is kept only as defensive fallback.
            env_ids = kwargs.get("env_ids", args[0] if args else None)
            if env_ids is None:
                self._subtask_idx[:] = 0
                self._obj2_needs_fresh_placement[:] = False
                self._task_descriptions = [
                    self._subtasks_cfg["instruction_1"]
                ] * self.num_envs
            else:
                ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
                self._subtask_idx[env_ids] = 0
                self._obj2_needs_fresh_placement[env_ids] = False
                for eid in ids:
                    self._task_descriptions[eid] = self._subtasks_cfg["instruction_1"]
        elif self._active_plan and self._mode == "plan":
            if not hasattr(self, "_plan_idx"):
                self._init_plan_ratchet()
            env_ids = kwargs.get("env_ids", args[0] if args else None)
            first_instruction = self._active_plan[0]["instruction"]
            if env_ids is None:
                self._plan_idx[:] = 0
                self._needs_fresh_placement[:] = False
                self._task_descriptions = [first_instruction] * self.num_envs
            else:
                ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
                self._plan_idx[env_ids] = 0
                self._needs_fresh_placement[env_ids] = False
                for eid in ids:
                    self._task_descriptions[eid] = first_instruction
        return obs, info

    def step(self, action):
        _reset_idx = self._reset_idx
        self._reset_idx = lambda env_ids: None
        try:
            result = super().step(action)
        finally:
            self._reset_idx = _reset_idx

        obs, reward, terminated, time_out, extras = result

        obj1_now = obj2_now = None
        if self._subtasks_cfg:
            from robolab.core.task.conditionals import object_on_top

            surface = self._subtasks_cfg["surface"]
            object_1 = self._subtasks_cfg.get("object_1")
            object_2 = self._subtasks_cfg.get("object_2")
            if object_1:
                obj1_now = object_on_top(
                    self,
                    object=object_1,
                    reference_object=surface,
                    require_gripper_detached=True,
                    env_id=None,
                )
                extras["subtask_1_success"] = obj1_now
            if object_2:
                obj2_now = object_on_top(
                    self,
                    object=object_2,
                    reference_object=surface,
                    require_gripper_detached=True,
                    env_id=None,
                )
                extras["subtask_2_success"] = obj2_now

        # Generic single-edge dispatch (mode == "single_edge"): the predicate/args
        # aren't known ahead of time -- looked up by name from conditionals.py instead
        # of the hardcoded object_on_top call above, per PLAN.md section 5.1.
        edge_success_now = None
        if self._edge_spec:
            from robolab.core.task import conditionals

            predicate_fn = getattr(conditionals, self._edge_spec["predicate"])
            edge_success_now = predicate_fn(
                self, env_id=None, **self._edge_spec["predicate_args"]
            )
            extras["subtask_1_success"] = edge_success_now

        # End-state collection -- fires whenever the "first" subtask's condition
        # succeeds (either _subtasks_cfg's hardcoded subtask_1, or the single edge
        # being trained under mode == "single_edge"). Used on the dedicated
        # checkpoint-collection eval run, not during normal training. capture_names
        # comes from the subtasks_cfg template (object_1/object_2/surface) when
        # present; for a generic single edge it falls back to this scene's fixed
        # object set (this design is one tree per scene, not per edge -- see
        # PLAN.md section 0.1 -- so a fixed capture list is correct here, not a
        # simplification specific to this edge).
        save_trigger_now = obj1_now if obj1_now is not None else edge_success_now
        if self._save_end_state_path and save_trigger_now is not None and save_trigger_now.any():
            import json

            from robolab.core.world.world_state import get_world

            world = get_world(self)
            capture_names = (
                [
                    self._subtasks_cfg["object_1"],
                    self._subtasks_cfg["object_2"],
                    self._subtasks_cfg["surface"],
                ]
                if self._subtasks_cfg
                else ["cutting_board_a", "ceramic_mug", "coke"]
            )
            for eid in save_trigger_now.nonzero(as_tuple=True)[0].tolist():
                objects = {}
                for name in capture_names:
                    pos, quat = world.get_pose(name, is_relative=True, env_id=eid)
                    vel = world.get_velocity(name, env_id=eid)
                    objects[name] = pos.tolist() + quat.tolist() + vel.tolist()
                row = {
                    "objects": objects,
                    "robot_joint_pos": world.get_joint_positions(
                        "robot", env_id=eid
                    ).tolist(),
                    "robot_joint_vel": world.get_joint_velocity(
                        "robot", env_id=eid
                    ).tolist(),
                }
                with open(self._save_end_state_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

        if self._subtasks_cfg:
            if self._mode == "subtask_1":
                terminated = terminated | obj1_now
                extras["task_descriptions"] = [
                    self._subtasks_cfg["instruction_1"]
                ] * self.num_envs
            elif self._mode == "subtask_2":
                terminated = terminated | obj2_now
                extras["task_descriptions"] = [
                    self._subtasks_cfg["instruction_2"]
                ] * self.num_envs
            elif self._mode == "full":
                # Irreversible ratchet: a real VLM that already moved on to the coke-can
                # step has no mechanism to notice the cup got bumped later and wouldn't
                # re-issue the cup instruction, so we don't re-check obj1 once advanced.
                if not hasattr(self, "_subtask_idx"):
                    self._init_ratchet()

                on_1 = self._subtask_idx == 0
                advance = on_1 & obj1_now
                if advance.any():
                    # object_2 may already be sitting on the surface at the exact
                    # moment of handoff (e.g. it was placed out of order, before
                    # instruction_2 was ever issued, and just never moved). That
                    # shouldn't retroactively satisfy instruction_2 -- require a
                    # fresh off-then-on placement observed after the handoff.
                    self._obj2_needs_fresh_placement = torch.where(
                        advance, obj2_now, self._obj2_needs_fresh_placement
                    )
                    self._subtask_idx = torch.where(
                        advance, torch.ones_like(self._subtask_idx), self._subtask_idx
                    )
                    for eid in advance.nonzero(as_tuple=True)[0].tolist():
                        self._task_descriptions[eid] = self._subtasks_cfg[
                            "instruction_2"
                        ]

                on_2 = self._subtask_idx == 1
                # Once object_2 is observed off the surface while on subtask_2,
                # any later "on" reading is a genuine new placement -- clear the gate.
                self._obj2_needs_fresh_placement = torch.where(
                    on_2 & ~obj2_now,
                    torch.zeros_like(self._obj2_needs_fresh_placement),
                    self._obj2_needs_fresh_placement,
                )
                complete = on_2 & obj2_now & ~self._obj2_needs_fresh_placement
                if complete.any():
                    self._subtask_idx = torch.where(
                        complete,
                        torch.full_like(self._subtask_idx, 2),
                        self._subtask_idx,
                    )

                terminated = terminated | (self._subtask_idx == 2)
                extras["current_subtask_idx"] = self._subtask_idx.clone()
                extras["task_descriptions"] = list(self._task_descriptions)
        elif self._edge_spec and self._mode == "single_edge":
            # Generic counterpart to the "subtask_1"/"subtask_2" branches above: one
            # edge, one predicate, no ratchet needed (training on a single edge is
            # never a multi-step sequence -- see generic_single_edge_task.py).
            terminated = terminated | edge_success_now
            extras["task_descriptions"] = [
                self._edge_spec["instruction"]
            ] * self.num_envs
        elif self._active_plan and self._mode == "plan":
            # Generic counterpart to the "full" branch above: an N-step ratchet over
            # `_active_plan` (see generic_eval_task.py) instead of a hardcoded 2-step
            # one, dispatching each step's predicate by name (PLAN.md section 5.1).
            #
            # Generalizes the irreversible, out-of-order-completion-proof ratchet
            # fix above (the `_obj2_needs_fresh_placement` gate) from exactly one
            # handoff (subtask_1 -> subtask_2) to N-1 handoffs. The same reasoning
            # applies at every transition, not just the first: a step's completion,
            # once its predicate goes true, is only credited if it was NOT already
            # true at the moment its instruction became the active one (i.e. it was
            # satisfied out of order, before this step was even reached) -- it must
            # be freshly (re-)satisfied while actually active. Since only one step
            # is ever active per env at a time, a single per-env gate
            # (`_needs_fresh_placement`) suffices, snapshotted at each handoff and
            # cleared the first time the new active step's predicate is observed
            # false while active -- exactly generalizing the original two-step logic
            # (on_1/advance/on_2/complete above) to a loop over every step index.
            if not hasattr(self, "_plan_idx"):
                self._init_plan_ratchet()

            from robolab.core.task import conditionals

            plan = self._active_plan
            n = len(plan)
            # Evaluate every step's predicate for the whole batch, unconditionally,
            # exactly as obj1_now/obj2_now are computed unconditionally above --
            # cheap (n is small, bounded by the tree's max depth) and keeps the
            # gate-snapshot value below consistent with what "now" means for the
            # rest of this same step() call.
            preds_now = [
                getattr(conditionals, spec["predicate"])(
                    self, env_id=None, **spec["predicate_args"]
                )
                for spec in plan
            ]
            # Backward-compatible metric keys for the common (and currently only
            # exercised) 2-step case -- lets RoboLabDroidEnv._record_metrics's
            # existing subtask_1_success_once / subtask_2_success_once aggregation
            # keep working unchanged for a 2-step plan. N > 2 plans don't yet get
            # per-step metrics beyond current_subtask_idx (final_subtask_idx) below;
            # that would need _record_metrics itself extended, out of scope here.
            if n >= 1:
                extras["subtask_1_success"] = preds_now[0]
            if n >= 2:
                extras["subtask_2_success"] = preds_now[1]

            for i in range(n):
                on_i = self._plan_idx == i
                if not on_i.any():
                    continue
                pred_i_now = preds_now[i]
                if i == 0:
                    # The very first step was never handed off into -- there's no
                    # prior instruction it could have been satisfied out of order
                    # with respect to, so it's never gated.
                    gated = torch.zeros_like(pred_i_now)
                else:
                    self._needs_fresh_placement = torch.where(
                        on_i & ~pred_i_now,
                        torch.zeros_like(self._needs_fresh_placement),
                        self._needs_fresh_placement,
                    )
                    gated = self._needs_fresh_placement
                advance = on_i & pred_i_now & ~gated
                if not advance.any():
                    continue
                next_idx = i + 1
                if next_idx < n:
                    next_pred_now = preds_now[next_idx]
                    self._needs_fresh_placement = torch.where(
                        advance, next_pred_now, self._needs_fresh_placement
                    )
                    next_instruction = plan[next_idx]["instruction"]
                else:
                    next_instruction = "done"
                for eid in advance.nonzero(as_tuple=True)[0].tolist():
                    self._task_descriptions[eid] = next_instruction
                self._plan_idx = torch.where(
                    advance, torch.full_like(self._plan_idx, next_idx), self._plan_idx
                )

            terminated = terminated | (self._plan_idx >= n)
            extras["current_subtask_idx"] = self._plan_idx.clone()
            extras["task_descriptions"] = list(self._task_descriptions)

        return obs, reward, terminated, time_out, extras


class RoboLabDroidEnv(IsaaclabBaseEnv):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        def make_env_isaaclab():
            import sys

            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            from isaaclab.envs import ManagerBasedRLEnv
            from robolab.core.environments.config import parse_env_cfg
            from robolab.registrations.droid.auto_env_registrations_jointpos import (
                auto_register_droid_envs,
            )
            from robolab.registrations.droid.camera_presets import WRIST_POLARIS

            os.environ["ROBOLAB_EVAL_ONLY"] = (
                "1" if bool(getattr(self.cfg, "eval_only", False)) else "0"
            )

            task_file = getattr(self.cfg.init_params, "task_file", None)

            # Long-horizon subtask-handoff plumbing, opt-in via init_params.
            reset_states_path = getattr(self.cfg.init_params, "reset_states_path", None)
            if reset_states_path:
                from robolab.constants import TASK_DIR
                from robolab.core.task.task_utils import (
                    load_task_from_file,
                    resolve_task_path,
                )

                # Must resolve via the same (task_file, TASK_DIR) pair that
                # auto_register_droid_envs's internal EnvFactory uses (it defaults to
                # TASK_DIR too), so load_task_from_file's abspath-keyed module cache
                # hits the same module object auto_register_droid_envs will use below.
                resolved_path, _ = resolve_task_path(task_file, TASK_DIR)
                task_class = load_task_from_file(resolved_path)
                sys.modules[task_class.__module__].RESET_STATES_PATH = reset_states_path

            subtasks_raw = getattr(self.cfg.init_params, "subtasks", None)
            subtasks_cfg = None
            if subtasks_raw:
                subtasks_raw = dict(subtasks_raw)
                # fmt doubles as the str.format() substitution dict for instruction_N templates.
                fmt = {
                    "object_1": subtasks_raw.get("object_1"),
                    "object_2": subtasks_raw.get("object_2"),
                    "surface": subtasks_raw.get("surface"),
                }
                subtasks_cfg = {
                    **fmt,
                    "instruction_1": subtasks_raw.get("instruction_1", "").format(
                        **fmt
                    ),
                    "instruction_2": subtasks_raw.get("instruction_2", "").format(
                        **fmt
                    ),
                }

            # Generic single-edge (training) / plan (eval) plumbing, opt-in via
            # init_params, parallel to subtasks_cfg above. Unlike reset_states_path,
            # predicate/predicate_args/instruction never need to reach the task-file
            # module itself (see generic_single_edge_task.py's RESET_STATES_PATH
            # docstring) -- they're only consumed by this mixin, which already has
            # direct access to self.cfg.init_params, so no sys.modules injection is
            # needed on this side. Two equivalent ways in, for whichever is more
            # convenient upstream (the orchestrator's generate_training_config /
            # resolve_plan): an inline (possibly nested) Hydra field
            # (init_params.edge_spec / init_params.plan -- OmegaConf.to_container
            # fully resolves nested predicate_args dicts into plain dicts so
            # **spec["predicate_args"] unpacking in step() works regardless of a
            # predicate's own arg shapes), or a path to a small JSON file
            # (init_params.edge_spec_path / init_params.plan_path -- same shape,
            # mirroring reset_states_path's path-to-data-file convention for
            # whichever side finds it easier to produce, e.g. writing a VLM-derived
            # predicate_args dict via json.dump instead of assembling Hydra
            # overrides for an arbitrarily-shaped nested dict). If both are given
            # for the same field, the *_path file wins.
            edge_spec_path = getattr(self.cfg.init_params, "edge_spec_path", None)
            if edge_spec_path:
                import json as _json

                with open(edge_spec_path) as _f:
                    edge_spec = _json.load(_f)
            else:
                edge_spec_raw = getattr(self.cfg.init_params, "edge_spec", None)
                edge_spec = (
                    OmegaConf.to_container(edge_spec_raw, resolve=True)
                    if edge_spec_raw is not None
                    else None
                )

            plan_path = getattr(self.cfg.init_params, "plan_path", None)
            if plan_path:
                import json as _json

                with open(plan_path) as _f:
                    active_plan = _json.load(_f)
            else:
                plan_raw = getattr(self.cfg.init_params, "plan", None)
                active_plan = (
                    OmegaConf.to_container(plan_raw, resolve=True)
                    if plan_raw is not None
                    else None
                )

            mode = str(getattr(self.cfg.init_params, "mode", "full"))
            save_end_state_path = getattr(
                self.cfg.init_params, "save_end_state_path", None
            )

            auto_register_droid_envs(task=task_file, cameras=WRIST_POLARIS)

            isaac_env_cfg = parse_env_cfg(
                self.isaaclab_env_id,
                device="cuda:0",
                seed=self.seed,
                num_envs=self.cfg.init_params.num_envs,
            )
            isaac_env_cfg.recorders = None
            isaac_env_cfg.scene.over_shoulder_left_camera.height = (
                self.cfg.init_params.over_shoulder_cam.height
            )
            isaac_env_cfg.scene.over_shoulder_left_camera.width = (
                self.cfg.init_params.over_shoulder_cam.width
            )
            isaac_env_cfg.scene.over_shoulder_left_camera.spawn.clipping_range = (
                0.01,
                5.0,
            )
            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width = self.cfg.init_params.wrist_cam.width
            isaac_env_cfg.scene.wrist_cam.spawn.clipping_range = (0.01, 5.0)

            # Build a per-call subclass so NoAutoResetManagerBasedRLEnv can use super()
            # on the concrete ManagerBasedRLEnv class loaded at runtime.
            NoAutoResetEnvCls = type(
                "NoAutoResetEnvCls",
                (NoAutoResetManagerBasedRLEnv, ManagerBasedRLEnv),
                {
                    "_subtasks_cfg": subtasks_cfg,
                    "_edge_spec": edge_spec,
                    "_active_plan": active_plan,
                    "_mode": mode,
                    "_save_end_state_path": save_end_state_path,
                },
            )
            env = NoAutoResetEnvCls(cfg=isaac_env_cfg)
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs, dynamic_task_descriptions=None):
        arm_joint_pos = obs["proprio_obs"]["arm_joint_pos"]  # [N, 7]
        gripper_pos = obs["proprio_obs"]["gripper_pos"]  # [N, 1]
        states = torch.cat([arm_joint_pos, gripper_pos], dim=1)  # [N, 8]
        return {
            "main_images": obs["image_obs"]["over_shoulder_left_camera"],
            "wrist_images": obs["image_obs"]["wrist_cam"],
            "states": states,
            "task_descriptions": (
                dynamic_task_descriptions
                if dynamic_task_descriptions is not None
                else [self.task_description] * self.num_envs
            ),
        }

    def _init_metrics(self):
        super()._init_metrics()
        self.subtask_1_success_once = torch.zeros(self.num_envs, dtype=bool).to(
            self.device
        )
        self.subtask_2_success_once = torch.zeros(self.num_envs, dtype=bool).to(
            self.device
        )

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx)
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool).to(self.device)
            mask[env_idx] = True
            self.subtask_1_success_once[mask] = False
            self.subtask_2_success_once[mask] = False
        else:
            self.subtask_1_success_once[:] = False
            self.subtask_2_success_once[:] = False

    def _record_metrics(
        self,
        step_reward,
        terminations,
        infos,
        subtask_1=None,
        subtask_2=None,
        subtask_idx=None,
    ):
        infos = super()._record_metrics(step_reward, terminations, infos)
        if subtask_1 is not None:
            self.subtask_1_success_once = (
                self.subtask_1_success_once | subtask_1.to(self.device).bool()
            )
            infos["episode"]["subtask_1_success_once"] = (
                self.subtask_1_success_once.clone()
            )
        if subtask_2 is not None:
            self.subtask_2_success_once = (
                self.subtask_2_success_once | subtask_2.to(self.device).bool()
            )
            infos["episode"]["subtask_2_success_once"] = (
                self.subtask_2_success_once.clone()
            )
        if subtask_idx is not None:
            infos["episode"]["final_subtask_idx"] = subtask_idx.to(self.device).float()
        return infos

    def step(self, actions=None, auto_reset=True):
        obs, _, terminations, truncations, infos = self.env.step(actions)

        subtask_1 = infos.get("subtask_1_success") if isinstance(infos, dict) else None
        subtask_2 = infos.get("subtask_2_success") if isinstance(infos, dict) else None
        subtask_idx = (
            infos.get("current_subtask_idx") if isinstance(infos, dict) else None
        )
        subtask_desc = (
            infos.get("task_descriptions") if isinstance(infos, dict) else None
        )

        terminations = terminations.clone()
        truncations = truncations.clone()
        obs = self._wrap_obs(obs, dynamic_task_descriptions=subtask_desc)

        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations

        step_reward = self._calc_step_reward(terminations)
        infos = self._record_metrics(
            step_reward,
            terminations,
            {},
            subtask_1=subtask_1,
            subtask_2=subtask_2,
            subtask_idx=subtask_idx,
        )

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.clone()
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, step_reward, terminations, truncations, infos


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def main():

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image, ImageDraw

    cfg = OmegaConf.create({
        "env_type": "robolab_droid",
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": False,
        "reward_coef": 1.0,
        "seed": 0,
        "max_episode_steps": 210,
        "max_steps_per_rollout_epoch": 210,
        "video_cfg": {
            "save_video": False,
            "info_on_video": False,
            "fps": 10,
            "video_base_dir": "/tmp/rlinf_test",
        },
        "init_params": {
            "id": "MugOnCuttingBoardTask",
            "task_file": "mug_on_cutting_board_task.py",
            "num_envs": None,
            "task_description": "Pick up the coffee mug and place it on the cutting board",
            "over_shoulder_cam": {"height": 224, "width": 224},
            "wrist_cam": {"height": 224, "width": 224},
        },
    })

    env = RoboLabDroidEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    frames = []

    for preset_idx in range(100):
        os.environ["ROBOLAB_FORCE_PRESET_IDX"] = str(preset_idx)

        print(f"\n===== Forcing preset {preset_idx} =====")

        obs, _ = env.reset()

        img = obs["main_images"][0].detach().cpu().numpy().astype(np.uint8)

        # Draw the preset index on the image.

        pil = Image.fromarray(img)

        draw = ImageDraw.Draw(pil)

        draw.rectangle((0, 0, 55, 20), fill="black")

        draw.text((4, 2), str(preset_idx), fill="white")

        frames.append(np.array(pil))

    os.environ.pop("ROBOLAB_FORCE_PRESET_IDX", None)

    rows = []

    for r in range(10):
        row = np.concatenate(frames[r * 10 : (r + 1) * 10], axis=1)

        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    Image.fromarray(grid).save("robolab_preset_grid_10x10.png")

    print("Saved -> robolab_preset_grid_10x10.png")

    env.close()

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--preset-idx", type=int, default=0)
    # parser.add_argument("--settle-steps", type=int, default=60)
    # args = parser.parse_args()

    # # Force the specific preset via the ROBOLAB_FORCE_PRESET_IDX hook already
    # # added to RoboLab/robolab/core/events/reset_pose.py, so this is directly
    # # comparable to a PolaRiS run reset to the same initial_conditions index.
    # os.environ["ROBOLAB_FORCE_PRESET_IDX"] = str(args.preset_idx)

    # os.environ.pop("DISPLAY", None)
    # from isaaclab.app import AppLauncher
    # sim_app = AppLauncher(headless=True, enable_cameras=True).app

    # import numpy as np
    # import torch as _torch
    # import isaaclab.utils.math as math_utils
    # from isaaclab.envs import ManagerBasedRLEnv
    # from isaacsim.core.utils.stage import get_current_stage
    # from pxr import Usd, UsdGeom, Gf

    # from robolab.core.environments.config import parse_env_cfg
    # from robolab.registrations.droid.auto_env_registrations_jointpos import (
    #     auto_register_droid_envs,
    # )
    # from robolab.registrations.droid.camera_presets import WRIST_POLARIS

    # auto_register_droid_envs(task="mug_on_cutting_board_task.py", cameras=WRIST_POLARIS)

    # isaac_env_cfg = parse_env_cfg(
    #     "MugOnCuttingBoardTask", device="cuda:0", seed=0, num_envs=1
    # )
    # isaac_env_cfg.recorders = None
    # isaac_env_cfg.scene.over_shoulder_left_camera.height = 224
    # isaac_env_cfg.scene.over_shoulder_left_camera.width = 224
    # isaac_env_cfg.scene.wrist_cam.height = 224
    # isaac_env_cfg.scene.wrist_cam.width = 224

    # NoAutoResetEnvCls = type(
    #     "NoAutoResetEnvCls", (NoAutoResetManagerBasedRLEnv, ManagerBasedRLEnv), {}
    # )
    # env = NoAutoResetEnvCls(cfg=isaac_env_cfg)
    # env.reset()  # ROBOLAB_FORCE_PRESET_IDX makes reset_pose_to_presets use --preset-idx

    # stage = get_current_stage()
    # scene = env.scene

    # # -----------------------------------------------------------------
    # # 0) Ground truth: what raw preset tuple was actually asked for.
    # # -----------------------------------------------------------------
    # from robolab.tasks.benchmark.mug_on_cutting_board_task import _PRESET_POSES
    # raw_preset = _PRESET_POSES[args.preset_idx]
    # print(f"\n=== Preset idx {args.preset_idx} — raw input from _PRESET_POSES ===")
    # for obj_name, pose in zip(["cutting_board_a", "ceramic_mug"], raw_preset):
    #     print(f"  {obj_name}: (x,y,z,qw,qx,qy,qz) = {pose}")

    # env_origin = scene.env_origins[0].detach().cpu().numpy()

    # def print_pose(tag):
    #     print(f"\n--- {tag} (env_origin = {tuple(env_origin)}) ---")
    #     for name in ["cutting_board_a", "ceramic_mug"]:
    #         asset = scene[name]
    #         root_pos_w = asset.data.root_pos_w[0].detach().cpu().numpy()
    #         root_quat_w = asset.data.root_quat_w[0].detach().cpu().numpy()  # (w, x, y, z)
    #         root_pos_local = root_pos_w - env_origin
    #         print(f"  {name}: pos_rel={tuple(root_pos_local)}  quat_wxyz={tuple(root_quat_w)}")

    # # -----------------------------------------------------------------
    # # 1) Pose immediately after reset (t=0), before any physics stepping.
    # # -----------------------------------------------------------------
    # print_pose("t=0, immediately after reset")

    # # -----------------------------------------------------------------
    # # 2) Let physics settle with a zero/no-op action, checking at
    # # intervals whether pose drifts away from the written preset.
    # # If it's stable the whole way, the settle-vs-penetration theory
    # # is dead and we look elsewhere. If it visibly rotates/moves,
    # # that confirms an initial-penetration/contact-resolution issue.
    # # -----------------------------------------------------------------
    # action_space = env.action_space
    # action_shape = action_space.shape if hasattr(action_space, "shape") else action_space[0].shape
    # zero_action = _torch.zeros((1,) + tuple(action_shape[-1:]), device="cuda:0")

    # print(f"\n=== Stepping {args.settle_steps} times with zero action to check for drift ===")
    # check_points = {0, 1, 2, 4, 9, 19, 29, 39, 59, args.settle_steps - 1}
    # for step_i in range(args.settle_steps):
    #     env.step(zero_action)
    #     if step_i in check_points:
    #         print_pose(f"t={step_i + 1} steps after reset")

    # # -----------------------------------------------------------------
    # # 3) Final settled pose + static local geometry-child transform,
    # # as before, for reference.
    # # -----------------------------------------------------------------
    # print_pose(f"FINAL, after {args.settle_steps} steps")

    # PRIM_PREFIX = "/World/envs/env_0/scene"
    # print(f"\n=== Static local geometry-child transform (authoring-time, sanity check) ===")
    # for name in ["cutting_board_a", "ceramic_mug"]:
    #     root_prim = stage.GetPrimAtPath(f"{PRIM_PREFIX}/{name}")
    #     if not root_prim.IsValid():
    #         print(f"{name}: prim not found at {PRIM_PREFIX}/{name}")
    #         continue

    #     mesh_prim = stage.GetPrimAtPath(f"{PRIM_PREFIX}/{name}/geometry/mesh")
    #     if not mesh_prim.IsValid():
    #         mesh_prim = None
    #         for desc in Usd.PrimRange(root_prim):
    #             if desc.IsA(UsdGeom.Mesh):
    #                 mesh_prim = desc
    #                 break
    #         if mesh_prim is None:
    #             print(f"  no Mesh-typed prim found under {name} at all.")
    #             continue

    #     xformable = UsdGeom.Xformable(mesh_prim)
    #     local_matrix: Gf.Matrix4d = xformable.GetLocalTransformation()
    #     translation = local_matrix.ExtractTranslation()
    #     rotation_quat = local_matrix.ExtractRotationQuat()

    #     print(f"{name} -> {mesh_prim.GetPath()}:")
    #     print(f"  local translation: {tuple(translation)}")
    #     print(f"  local orientation (quat, real+imag): "
    #           f"{rotation_quat.GetReal()}, {tuple(rotation_quat.GetImaginary())}")

    # env.close()
    # sim_app.close()


if __name__ == "__main__":
    main()
