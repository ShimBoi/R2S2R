"""Screening, train and eval jobs built from task specs, and their results read back from the run's tensorboard log.

A job is the command for one RLinf run plus its output dir. Spec file names (stems) name the candidates, so the
specs of one screening run must have distinct stems.
"""
import glob
import os
from pathlib import Path

CONFIG = "robolab_mug_board_grpo_openpi_pi05"
EVAL_SCRIPT = "examples/embodiment/eval_embodiment.sh"
TRAIN_SCRIPT = "examples/embodiment/run_embodiment.sh"


def _common(out, ckpt):
    return [
        f"runner.logger.log_path={out}",
        "runner.logger.logger_backends=[tensorboard]",
        f"++actor.model.lora_path={ckpt if ckpt else 'null'}",
    ]


def _eval_envs(episodes, env_workers):
    if episodes % env_workers:
        raise ValueError(f"{episodes} episodes cannot be split evenly over {env_workers} env workers")
    return episodes // env_workers


def screen(specs, out, ckpt=None, episodes=16, env_workers=4, config=CONFIG):
    """One eval run where every spec gets `episodes` episodes: episodes / env_workers envs on each env worker."""
    per_worker = _eval_envs(episodes, env_workers)
    overrides = _common(out, ckpt) + [
        f"env.eval.init_params.task_spec=[{','.join(str(s) for s in specs)}]",
        f"env.eval.total_num_envs={per_worker * env_workers * len(specs)}",
        "algorithm.eval_rollout_epoch=1",
    ]
    return {"script": EVAL_SCRIPT, "config": config, "overrides": overrides, "out": str(out)}


def evaluate(spec, out, ckpt=None, episodes=320, epochs=4, env_workers=4, save_states=None, config=CONFIG):
    """A single-spec eval of `episodes` episodes over `epochs` eval epochs; saves successful end states if asked."""
    if episodes % epochs:
        raise ValueError(f"{episodes} episodes cannot be split evenly over {epochs} epochs")
    per_worker = _eval_envs(episodes // epochs, env_workers)
    overrides = _common(out, ckpt) + [
        f"env.eval.init_params.task_spec=[{spec}]",
        f"env.eval.total_num_envs={per_worker * env_workers}",
        f"algorithm.eval_rollout_epoch={epochs}",
        f"env.eval.open_ended_task_proposal.success_end_states_save_path={save_states if save_states else 'null'}",
    ]
    return {"script": EVAL_SCRIPT, "config": config, "overrides": overrides, "out": str(out)}


TRAIN_RECIPE = [
    "env.train.total_num_envs=512",
    "algorithm.rollout_epoch=1",
    "actor.global_batch_size=2048",
    "actor.micro_batch_size=16",
    "algorithm.update_epoch=4",
    "algorithm.filter_rewards=True",
    "env.train.use_rel_reward=True",
]


def train(spec, out, steps=25, ckpt=None, save_every=5, config=CONFIG):
    """`steps` GRPO steps on one spec from LoRA `ckpt` (None: a fresh LoRA) with TRAIN_RECIPE, the recipe of
    the bread-on-plate finetune; saves a checkpoint every `save_every` steps and at the last one, and the end
    states of successful rollouts, under `out`."""
    overrides = _common(out, ckpt) + TRAIN_RECIPE + [
        f"env.train.init_params.task_spec=[{spec}]",
        f"runner.max_epochs={steps}",
        f"runner.save_interval={min(save_every, steps)}",
        f"env.train.open_ended_task_proposal.success_end_states_save_path={Path(out) / 'end_states'}",
    ]
    return {"script": TRAIN_SCRIPT, "config": config, "overrides": overrides, "out": str(out)}


def scalars(out):
    """{tag: [(step, value), ...]} from the run's tensorboard log."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(os.path.join(out, "tensorboard", "events.out.tfevents.*")))
    if not files:
        raise FileNotFoundError(f"no tensorboard log in {out}/tensorboard")
    result = {}
    for f in files:
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            result.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
    return result


def screen_results(out, specs, episodes=16):
    """{spec stem: successes out of `episodes`} for a screening run."""
    s = scalars(out)
    stems = [Path(p).stem for p in specs]
    if s["eval/num_trajectories"][-1][1] != episodes * len(specs):
        raise ValueError(f"{out}: {s['eval/num_trajectories'][-1][1]:.0f} episodes, expected {episodes * len(specs)}")
    tags = [f"eval/success_once/{n}" for n in stems] if len(stems) > 1 else ["eval/success_once"]
    return {n: round(s[t][-1][1] * episodes) for n, t in zip(stems, tags)}


def eval_results(out):
    """(success rate, episodes) for a single-spec eval run."""
    s = scalars(out)
    return s["eval/success_once"][-1][1], int(s["eval/num_trajectories"][-1][1])


def train_results(out):
    """The training success rate per step, the last checkpoint's LoRA dir and the saved end-state dir."""
    s = scalars(out)
    ckpts = glob.glob(os.path.join(out, "*", "checkpoints", "global_step_*"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {out}/<experiment>/checkpoints")
    last = max(ckpts, key=lambda p: int(p.rsplit("_", 1)[-1]))
    states = os.path.join(out, "end_states")
    return {
        "success": [v for _, v in sorted(s["env/success_once"])],
        "ckpt": os.path.join(last, "actor"),
        "end_states": states if glob.glob(os.path.join(states, "*.npz")) else None,
    }
