import os

import gymnasium as gym
import torch

from ..isaaclab_env import IsaaclabBaseEnv


class NoAutoResetManagerBasedRLEnv:
    """Mixin that suppresses ManagerBasedRLEnv's internal auto-reset during step().

    Isaac Lab resets terminated/truncated envs inside step() before returning,
    which corrupts observations when the RLinf env worker controls episode resets
    externally (auto_reset=False). This mixin makes _reset_idx a no-op for the
    duration of each step() call so resets only happen via explicit env.reset().
    """

    def step(self, action):
        _reset_idx = self._reset_idx
        self._reset_idx = lambda env_ids: None
        try:
            result = super().step(action)
        finally:
            self._reset_idx = _reset_idx
        return result


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
            import os
            os.environ.pop("DISPLAY", None)
            
            from isaaclab.app import AppLauncher
            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            from isaaclab.envs import ManagerBasedRLEnv
            from robolab.core.environments.config import parse_env_cfg
            from robolab.registrations.droid.auto_env_registrations_jointpos import (
                auto_register_droid_envs,
            )
            from robolab.registrations.droid.camera_presets import WRIST_POLARIS

            task_file = getattr(self.cfg.init_params, "task_file", None)
            auto_register_droid_envs(task=task_file, cameras=WRIST_POLARIS)

            isaac_env_cfg = parse_env_cfg(
                self.isaaclab_env_id,
                device="cuda:0",
                seed=self.seed,
                num_envs=self.cfg.init_params.num_envs,
            )
            isaac_env_cfg.recorders = None
            isaac_env_cfg.scene.over_shoulder_left_camera.height = self.cfg.init_params.over_shoulder_cam.height
            isaac_env_cfg.scene.over_shoulder_left_camera.width  = self.cfg.init_params.over_shoulder_cam.width
            isaac_env_cfg.scene.over_shoulder_left_camera.spawn.clipping_range = (0.01, 5.0)
            isaac_env_cfg.scene.wrist_cam.height = self.cfg.init_params.wrist_cam.height
            isaac_env_cfg.scene.wrist_cam.width  = self.cfg.init_params.wrist_cam.width
            isaac_env_cfg.scene.wrist_cam.spawn.clipping_range = (0.01, 5.0)

            # Build a per-call subclass so NoAutoResetManagerBasedRLEnv can use super()
            # on the concrete ManagerBasedRLEnv class loaded at runtime.
            NoAutoResetEnvCls = type(
                "NoAutoResetEnvCls",
                (NoAutoResetManagerBasedRLEnv, ManagerBasedRLEnv),
                {},
            )
            env = NoAutoResetEnvCls(cfg=isaac_env_cfg)
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        arm_joint_pos = obs["proprio_obs"]["arm_joint_pos"]  # [N, 7]
        gripper_pos   = obs["proprio_obs"]["gripper_pos"]    # [N, 1]
        states = torch.cat([arm_joint_pos, gripper_pos], dim=1)  # [N, 8]
        return {
            "main_images":      obs["image_obs"]["over_shoulder_left_camera"],
            "wrist_images":     obs["image_obs"]["wrist_cam"],
            "states":           states,
            "task_descriptions": [self.task_description] * self.num_envs,
        }

    def step(self, actions=None, auto_reset=True):
        obs, _, terminations, truncations, infos = self.env.step(actions)

        terminations = terminations.clone()
        truncations  = truncations.clone()
        obs = self._wrap_obs(obs)

        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations

        step_reward = self._calc_step_reward(terminations)
        infos = self._record_metrics(step_reward, terminations, {})

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
    import os

    import numpy as np

    from PIL import Image, ImageDraw

    from omegaconf import OmegaConf

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

        img = (

            obs["main_images"][0]

            .detach()

            .cpu()

            .numpy()

            .astype(np.uint8)

        )

        # Draw the preset index on the image.

        pil = Image.fromarray(img)

        draw = ImageDraw.Draw(pil)

        draw.rectangle((0, 0, 55, 20), fill="black")

        draw.text((4, 2), str(preset_idx), fill="white")

        frames.append(np.array(pil))

    os.environ.pop("ROBOLAB_FORCE_PRESET_IDX", None)

    rows = []

    for r in range(10):

        row = np.concatenate(frames[r * 10:(r + 1) * 10], axis=1)

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
