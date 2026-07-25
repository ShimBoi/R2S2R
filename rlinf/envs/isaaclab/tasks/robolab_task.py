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

from ..isaaclab_env import IsaaclabBaseEnv


class NoAutoResetManagerBasedRLEnv:
    """Mixin that suppresses ManagerBasedRLEnv's internal auto-reset during step().

    Isaac Lab resets terminated/truncated envs inside step() before returning,
    which corrupts observations when the RLinf env worker controls episode resets
    externally (auto_reset=False). This mixin makes _reset_idx a no-op for the
    duration of each step() call so resets only happen via explicit env.reset().

    Also optionally drives a VLM-simulated long-horizon subtask handoff, controlled
    entirely by the three class attributes below (set per-instance by
    RoboLabDroidEnv._make_env_function from `init_params.subtasks`/`mode`/
    `save_end_state_path`). When `_subtasks_cfg` is unset (the default), the handoff
    logic is skipped entirely and the mixin only suppresses auto-reset.
    """

    _subtasks_cfg = (
        None  # {"object_1", "object_2", "surface", "instruction_1", "instruction_2"}
    )
    _mode = "full"  # "subtask_1" | "subtask_2" | "full"
    _save_end_state_path = (
        None  # set only on the dedicated subtask_1-checkpoint collection run
    )

    def _init_ratchet(self):
        self._subtask_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._task_descriptions = [self._subtasks_cfg["instruction_1"]] * self.num_envs

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
                self._task_descriptions = [
                    self._subtasks_cfg["instruction_1"]
                ] * self.num_envs
            else:
                ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
                self._subtask_idx[env_ids] = 0
                for eid in ids:
                    self._task_descriptions[eid] = self._subtasks_cfg["instruction_1"]
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

        # End-state collection -- fires whenever subtask_1's condition succeeds. Used
        # on the dedicated subtask_1-checkpoint collection eval run, not during normal
        # training. Object names come from the same subtasks_cfg template (object_1/
        # object_2/surface) rather than being hardcoded here, so this stays task-agnostic;
        # the consumer (RoboLab's reset_to_captured_state) must be given matching names.
        if self._save_end_state_path and obj1_now is not None and obj1_now.any():
            import json

            from robolab.core.world.world_state import get_world

            world = get_world(self)
            capture_names = [
                self._subtasks_cfg["object_1"],
                self._subtasks_cfg["object_2"],
                self._subtasks_cfg["surface"],
            ]
            for eid in obj1_now.nonzero(as_tuple=True)[0].tolist():
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
                    self._subtask_idx = torch.where(
                        advance, torch.ones_like(self._subtask_idx), self._subtask_idx
                    )
                    for eid in advance.nonzero(as_tuple=True)[0].tolist():
                        self._task_descriptions[eid] = self._subtasks_cfg[
                            "instruction_2"
                        ]

                on_2 = self._subtask_idx == 1
                complete = on_2 & obj2_now
                if complete.any():
                    self._subtask_idx = torch.where(
                        complete,
                        torch.full_like(self._subtask_idx, 2),
                        self._subtask_idx,
                    )

                terminated = terminated | (self._subtask_idx == 2)
                extras["current_subtask_idx"] = self._subtask_idx.clone()
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
