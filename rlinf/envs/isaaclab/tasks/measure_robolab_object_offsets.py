"""Measure the root->bbox-centroid offset for cutting_board_a / ceramic_mug in
RoboLab, the same way as the PolaRiS-side snippet you already have. Run this and
put the printed offsets side by side with the PolaRiS numbers -- if they match,
the root-vs-centroid gap is a property of the shared mesh.usd payload and
cancels out between the two sims (copying root poses 1:1 is fine). If RoboLab's
offset comes out ~0 while PolaRiS's doesn't, that confirms RoboLab's import
pipeline re-centers the rigid body and Fable's Finding 1 is real -- you'd need
to correct _PRESET_POSES by the *difference* between the two offsets, not just
adopt PolaRiS's raw offset wholesale.

Run this inside your RoboLab/IsaacLab python environment:

    python measure_robolab_object_offsets.py --preset-idx 0
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset-idx", type=int, default=0)
    args = parser.parse_args()

    # Force the same preset used on the PolaRiS side, via the ROBOLAB_FORCE_PRESET_IDX
    # hook already added to RoboLab/robolab/core/events/reset_pose.py.
    os.environ["ROBOLAB_FORCE_PRESET_IDX"] = str(args.preset_idx)

    from isaaclab.app import AppLauncher
    sim_app = AppLauncher(headless=True, enable_cameras=True).app

    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom
    import isaaclab.utils.math as math_utils

    from robolab.core.environments.config import parse_env_cfg
    from robolab.registrations.droid.auto_env_registrations_jointpos import (
        auto_register_droid_envs,
    )
    from robolab.registrations.droid.camera_presets import WRIST_POLARIS

    auto_register_droid_envs(task="mug_on_cutting_board_task.py", cameras=WRIST_POLARIS)

    isaac_env_cfg = parse_env_cfg("MugOnCuttingBoardTask", device="cuda:0", seed=0, num_envs=1)
    env = ManagerBasedRLEnv(cfg=isaac_env_cfg)
    env.reset()  # ROBOLAB_FORCE_PRESET_IDX makes reset_pose_to_presets use --preset-idx

    stage = get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    # Adjust this prim path prefix if it doesn't match your scene -- print
    # `stage.GetPrimAtPath("/World/envs/env_0").GetChildren()` to check names
    # if this errors with "prim does not exist".
    PRIM_PREFIX = "/World/envs/env_0/scene"

    for name in ["cutting_board_a", "ceramic_mug"]:
        prim = stage.GetPrimAtPath(f"{PRIM_PREFIX}/{name}")
        if not prim.IsValid():
            print(f"{name}: prim not found at {PRIM_PREFIX}/{name} -- "
                  f"check the actual prim path in your scene and edit PRIM_PREFIX.")
            continue

        centroid_w = np.array(cache.ComputeWorldBound(prim).ComputeCentroid())
        asset = env.scene[name]
        root_p = asset.data.root_pos_w[0].detach().cpu().numpy()
        root_q = asset.data.root_quat_w[0]  # (w, x, y, z), stays on device

        offset_local = math_utils.quat_apply_inverse(
            root_q, torch.tensor(centroid_w - root_p, device=root_q.device).float()
        )

        print(f"{name}:")
        print(f"  root world pos:      {root_p.tolist()}")
        print(f"  bbox centroid world: {centroid_w.tolist()}")
        print(f"  root->centroid offset (prim/local frame): {offset_local.detach().cpu().numpy().tolist()}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
