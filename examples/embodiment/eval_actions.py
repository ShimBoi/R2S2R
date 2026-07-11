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

"""Sim-to-sim verification driver.

This is `eval_embodied_agent.py`, unchanged, with one addition: if
`cfg.rollout.replay_pkl` is set, it launches `ReplayRolloutWorker` (which replays
a pre-recorded action stream) instead of `MultiStepRolloutWorker` (which runs the
model). Everything else -- Cluster, HybridComponentPlacement, EnvWorker,
EmbodiedEvalRunner -- is identical to eval_embodied_agent.py.

Usage (two passes, two configs, same script):

  1. Collect actions in RoboLab (uses the real MultiStepRolloutWorker + your
     policy; requires `env.eval.data_collection.enabled=True` in the config,
     which writes a pickle per episode via the existing CollectEpisode wrapper):

        python eval_actions.py --config-name robolab_collect_actions

  2. Replay those exact actions in PolaRiS (uses ReplayRolloutWorker; point
     `rollout.replay_pkl` at the pickle written in step 1):

        python eval_actions.py --config-name polaris_replay_actions \\
            rollout.replay_pkl=/path/to/rank_0_env_0_episode_0_step_150_success.pkl
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.rollout.hf.replay_rollout_worker import ReplayRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="sim2sim_verify"
)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Only this line differs from eval_embodied_agent.py: pick the rollout worker
    # class based on whether we're replaying a saved action stream.
    rollout_cls = (
        ReplayRolloutWorker if cfg.rollout.get("replay_pkl", None) else MultiStepRolloutWorker
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = rollout_cls.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
