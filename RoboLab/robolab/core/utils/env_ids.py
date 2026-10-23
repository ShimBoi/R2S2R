# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch


def resolve_env_ids(env, env_ids: torch.Tensor | Sequence[int] | slice | None) -> torch.Tensor:
    """Return ``env_ids`` as a 1-D long tensor of indices on ``env.device``.

    Isaac Lab 3 passes ``slice(None)`` (rather than an index tensor) to reset events and
    ``_reset_idx`` when every env is being reset; ``None`` means the same thing.
    """
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
