# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import torch
from isaaclab.envs import ManagerBasedEnv


def randomize_dome_light_texture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    background_files: list[str],
    prim_path: str = "/World/background",
):
    """Swap the dome light texture to a randomly chosen HDR/EXR file on every reset.

    This is an Isaac Lab EventTerm function (mode="reset"). Because the dome light is
    shared across all parallel envs, the texture is swapped once per reset call
    regardless of which env_ids triggered it.

    Args:
        env: The manager-based environment instance.
        env_ids: Indices of environments being reset (unused; light is global).
        background_files: List of absolute paths to HDR or EXR background files to
            sample from uniformly at random.
        prim_path: USD prim path of the dome light. Defaults to "/World/background".
    """
    try:
        import omni.usd
        from pxr import Sdf

        chosen = random.choice(background_files)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        # Try the two attribute names used across Isaac Sim versions
        for attr_name in ("inputs:texture:file", "texture:file"):
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(Sdf.AssetPath(chosen))
                break

    except Exception as e:
        print(f"[randomize_background] skipped: {e}")
