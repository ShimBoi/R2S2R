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
      import numpy as np
      from PIL import Image
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
          "video_cfg": {"save_video": False, "info_on_video": False, "fps": 10, "video_base_dir": "/tmp/rlinf_test"},
          "init_params": {
              "id": "MugOnCuttingBoardTask",
              "task_file": "mug_on_cutting_board_task.py",
              "num_envs": None,
              "task_description": "Pick up the coffee mug and place it on the cutting board",
              "over_shoulder_cam": {"height": 224, "width": 224},
              "wrist_cam": {"height": 224, "width": 224},
          },
      })

      env = RoboLabDroidEnv(cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
      
      frames = []
      for i in range(16):
          obs, _ = env.reset()
          img = obs["main_images"][0].detach().cpu().numpy().astype("uint8")
          frames.append(img)
          print(f"Reset {i+1}/16")

      # tile into a 4x4 grid
      rows = [np.concatenate(frames[i*4:(i+1)*4], axis=1) for i in range(4)]
      grid = np.concatenate(rows, axis=0)
      Image.fromarray(grid).save("robolab_randomization_grid.png")
      print("Saved → robolab_randomization_grid.png")

      env.close()


if __name__ == "__main__":
    main()
