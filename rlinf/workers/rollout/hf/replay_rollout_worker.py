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

"""ReplayRolloutWorker: same Worker/Channel/Ray plumbing as MultiStepRolloutWorker
(rlinf/workers/rollout/hf/huggingface_worker.py), with only `predict()` overridden
to replay a pre-recorded action stream instead of running the model.

The pre-recorded stream is exactly what `rlinf.envs.wrappers.collect_episode.CollectEpisode`
already writes when `env.eval.data_collection.enabled=True` -- a pickle per episode
with a flat, per-step `"actions"` list (see `_flush_episode` in that file). This
worker just loads one such pickle and feeds its actions back out in
`num_action_chunks`-sized chunks, in order, ignoring the incoming obs entirely.

Everything else (init_worker, weight sync, channel comms, offload) is inherited
unchanged from MultiStepRolloutWorker -- this intentionally does NOT reimplement
any of that, so it stays byte-for-byte compatible with how eval_embodied_agent.py
already launches rollout workers.

Note: `init_worker()` is inherited as-is, so this still loads the real model
weights via `get_model(...)` even though `predict()` never calls the model. That's
wasted GPU memory/time but keeps every other code path (device placement, weight
sync handshake, etc.) identical to the real rollout worker -- the safer tradeoff
given this is only used for a short verification rollout.
"""

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class ReplayRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg):
        super().__init__(cfg)
        replay_pkl = cfg.rollout.replay_pkl
        self._replay_pkl_path = replay_pkl
        with open(replay_pkl, "rb") as f:
            episode = pickle.load(f)

        # CollectEpisode's buffer stores one entry per env.step() call (see
        # `_record_step` in collect_episode.py), already in order -- this is the
        # exact per-step action stream, so we just chunk it back up.
        actions = episode["actions"]
        self._replay_actions = np.stack(
            [np.asarray(a) for a in actions], axis=0
        )  # [T, action_dim]
        self._replay_step = 0
        self._chunk_size = cfg.actor.model.num_action_chunks

        n_eval_chunk_steps = cfg.env.eval.max_steps_per_rollout_epoch // self._chunk_size
        requested_steps = n_eval_chunk_steps * self._chunk_size
        recorded_steps = self._replay_actions.shape[0]
        self.log_info(
            f"ReplayRolloutWorker loaded {recorded_steps} recorded actions from "
            f"{replay_pkl}; env.eval config will request {requested_steps} steps "
            f"({n_eval_chunk_steps} chunks x {self._chunk_size}). "
            + (
                f"Recorded episode is SHORTER than the requested rollout -- the "
                f"last {requested_steps - recorded_steps} steps will hold the "
                f"final recorded action rather than replaying real data. If you "
                f"want an exact, padding-free comparison, set "
                f"env.eval.max_steps_per_rollout_epoch/max_episode_steps to a "
                f"multiple of {self._chunk_size} that is <= {recorded_steps}."
                if recorded_steps < requested_steps
                else "Lengths match, no padding will occur."
            )
        )

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        start = self._replay_step
        end = start + self._chunk_size
        chunk = self._replay_actions[start:end]

        if chunk.shape[0] < self._chunk_size:
            pad_needed = self._chunk_size - chunk.shape[0]
            # `chunk[-1:]` is only safe to pad from when chunk is non-empty --
            # once `start` is already past the end of the recorded episode,
            # chunk itself has 0 rows and chunk[-1:] is *also* empty (there's no
            # "last row" of nothing), which previously produced a 0-length pad
            # and crashed `torch.stack` downstream. Fall back to the last row of
            # the full recorded array in that case.
            if chunk.shape[0] > 0:
                pad_source = chunk[-1:]
            elif self._replay_actions.shape[0] > 0:
                pad_source = self._replay_actions[-1:]
            else:
                raise RuntimeError(
                    f"Replay pickle at {self._replay_pkl_path} contains 0 recorded "
                    "actions -- nothing to replay."
                )
            pad = np.repeat(pad_source, pad_needed, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        self._replay_step = end

        actions = torch.from_numpy(chunk).float().unsqueeze(0)  # [1, chunk_size, action_dim]
        return actions, {"expert_label_flag": False}
