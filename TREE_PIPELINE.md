# VLM-discovered subtask tree

A VLM (GPT-4o) decides which manipulation subtasks a VLA policy (π0.5, LoRA + GRPO) should
learn, the subtasks are trained one after another into a single checkpoint, and at eval time
the VLM turns a free-text instruction into an ordered sequence of those subtasks.

See also: [SETUP.md](SETUP.md) (install), [DEBUG.md](DEBUG.md) (known errors), [PLAN.md](PLAN.md)
(original design).

---

## 1. Concepts (read this first)

**Success checks.** RoboLab ships a fixed set of hand-written success-criteria checks in
`RoboLab/robolab/core/task/conditionals.py`, e.g. `object_on_top(env, object, reference_object)`,
`stacked`, `object_in_container`, about 33 in all. Each returns True/False per env from the
simulator state. Every subtask in this pipeline is one of these checks plus its arguments
(stored in code as `predicate` + `predicate_args`).

**Node** = the set of subtasks already completed. Root is `{}`.
**Edge** = one trainable subtask: from an exact node, make one success check true, arriving at a
new node. Example:

```
{}                              --object_on_top(coke, board)-->  {place_coke}
{place_coke}                    --object_on_top(mug,  board)-->  {place_coke, place_mug}
```

The tree for the mug/coke scene has two roots (mug first / coke first), each with one child.

Each edge's training episodes start from the end states its parent edge produced (root edges
start from the scene's preset initial conditions). Edges are finetuned in breadth-first order,
all into one LoRA checkpoint, each warm-started from the previous one.

### Where the VLM is and isn't involved

| Step | VLM? |
|---|---|
| Propose subtasks for a node (Phase A) | Yes |
| Validate proposals | No: coded checks |
| Train / evaluate / decide success | No: always one of RoboLab's success checks |
| Turn an instruction into a plan (Phase B) | Yes, only at branch points |

The VLM never writes code. It is shown the list of available success checks (function names +
signatures, read from `conditionals.py`) and only picks one plus its arguments, so every
subtask's success is judged by real, deterministic code.
(Claude Code was used to build and monitor this, but nothing at runtime depends on it.)

---

## 2. Phase A: discovering subtasks

`orchestrator._discover_node_edges(node)` does, for one node:

1. **Ground the prompt in real state** (`starting_states.py`). Root: summary of the scene's
   preset initial conditions (see "Initial conditions" below). Other nodes: summary of the
   parent edge's captured end-states. The VLM reasons about where objects really are, not a
   screenshot or a guess.
2. **Call the VLM** (`phase_a.call_vlm_phase_a`) with the scene objects, already-completed
   subtasks, the state summary, and the list of available success checks. It returns a JSON list of candidates
   (`id`, `instruction`, `predicate`, `predicate_args`, ...). The prompt is scene-agnostic.
3. **Validate** (`predicates.validate_phase_a`). Rejects:
   - a success check that doesn't exist, or arguments naming objects not in the scene
   - goals already true in the starting state (trivial)
   - passive/ambient checks (e.g. "object is upright") that aren't something to do
   - if `stable_base_objects` is given: stacking on anything not in that set (e.g. a board on a
     mug). Opt-in because what counts as a stable base is scene-specific.
4. **De-duplicate ids** (`_dedupe_id`). The VLM can propose the same id (e.g.
   `place_coke_on_cutting_board`) at different nodes. Collisions are renamed by appending the
   precondition (`..._given_place_mug_on_cutting_board`) so ids stay unique across the run.

A node where Phase A returns nothing is marked terminal (a leaf).

### Initial conditions (root starting states)

The current scene is a sim reconstruction of PolaRiS's DROID-MoveLatteCup environment, so its
root starting states are PolaRiS's 100 measured presets:
`RoboLab/assets/objects/polaris/initial_conditions.json`. Format:

```json
{"instruction": "put the latte art cup on top of the cutting board",
 "poses": [
   {"cuttingboard_eval": [x, y, z, qw, qx, qy, qz],
    "latteartcup_eval":  [x, y, z, qw, qx, qy, qz],
    "coke_eval":         [x, y, z, qw, qx, qy, qz]},
   ...   // one entry per preset
 ]}
```

Two places read it, both currently hardcoded to this file and its PolaRiS key names:
- Sim resets: `RoboLab/robolab/tasks/benchmark/starting_states.py` (`load_preset_poses`, maps
  keys to the scene's asset order).
- VLM prompt: `discovery/starting_states.py` (`default_initial_conditions_path`, plus
  `ROOT_POSE_KEY_ALIASES` mapping scene names like `ceramic_mug` → `latteartcup_eval`).

For a new scene (e.g. one generated from an image), write a file in the same format next to the
scene, keyed by the scene's own object names (then no alias table is needed), and point both
readers at it. See the TODO at the bottom.

## 3. Training loop

`orchestrator.discover_and_train_by_level()` is the production entrypoint. Per depth:
discover every node's edges → for each edge: train → collect end-states → add to manifest →
children become the next depth's frontier.

How one edge becomes a training run:

1. `generate_training_config()` writes `edge_specs/<id>.json` and returns Hydra overrides for
   `generic_single_edge_grpo_openpi_pi05`:
   - `+env.{train,eval}.init_params.edge_spec_path=<spec>`: predicate, args, instruction
   - `env.{train,eval}.init_params.reset_states_path=<parent end_states.jsonl or null>`
   - `+actor.model.lora_path=<previous checkpoint>`: warm start
   (`env.eval` mirrors `env.train` because end-state collection runs through `eval_embodiment.sh`,
   which reads `env.eval`.)
2. `RoboLabDroidEnv` (`rlinf/envs/isaaclab/tasks/robolab_task.py`), in `mode: single_edge`, calls
   `getattr(conditionals, spec["predicate"])(env, **predicate_args)` every step. True means
   success and the episode ends. The VLA sees `spec["instruction"]`.
3. Reset states: `generic_single_edge_task.py` can't receive config directly (RoboLab constructs
   it), so `robolab_task.py` writes `reset_states_path` onto the task module's
   `RESET_STATES_PATH` attribute before the env is built. `null` means random preset pool.
4. The final checkpoint lands at `logs/<ts>-<config>-<edge_id>/<config>/checkpoints/global_step_<N>/actor` (`N` = `runner.max_epochs`).
5. End-state collection: an eval run with the new checkpoint and
   `env.eval.init_params.save_end_state_path=...`. Successful envs append their state to the
   JSONL.
6. `manifest.add_edge()` records the edge with its checkpoint and `reset_states_path`.

**Before/after success rates** (shown on the dashboard): there is no separate pre-training eval.
"Before" = success rate in the edge's first training rollout epoch, i.e. the warm-started
checkpoint before any gradient updates on that edge. "After" = success rate from the
end-state collection run (step 5), which doubles as the post-training eval.

**Resuming.** Relaunching skips edges already in `manifest.json`. A re-run of Phase A might
propose an already-trained edge under a different (deduped) id, so "already trained" is matched
by content (precondition + predicate + args), not id. The running checkpoint is restored from
the last edge in the manifest.

**Failure handling.** Each stage is launched fully detached (`launch_and_wait`: `setsid`,
redirected output, exit-code file). Success requires exit code 0 **and** no `Traceback`/
`RuntimeError` in the log, because the `tee` in RLinf's launch scripts can hide a crash behind
exit 0. Any failure stops the run; nothing auto-retries.

## 4. Run scripts vs. env modes

Two separate settings, easy to confuse:
- **Script**: `run_embodiment.sh` trains (rollouts + gradient updates). `eval_embodiment.sh`
  only rolls out, no weight updates.
- **Env mode** (`init_params.mode`): what the env checks and which instruction it gives.

| Purpose | Script | Mode |
|---|---|---|
| Train one edge | `run_embodiment.sh` | `single_edge` |
| Collect end-states after training | `eval_embodiment.sh` + `save_end_state_path` | `single_edge` |
| Test time: language instruction → plan → rollout | `eval_embodiment.sh` | `plan` |
| Original hardcoded mug→coke pipeline | either | `subtask_1` / `subtask_2` / `full` |

## 5. Phase B: instruction → plan → eval

`phase_b.resolve_plan(instruction, manifest)` builds the plan one step at a time, starting
from the root (nothing done yet). At each step it looks at the trained subtasks that start from
exactly the current state:
- none → stop
- exactly one → take it, no VLM call
- several → ask the VLM which of these subtasks to do next for this instruction (it may also
  answer `done` or `unreachable`)

Result: ordered list of `{predicate, predicate_args, instruction}`, saved as JSON. This happens
once, before the rollout. No VLM calls during the episode.

**Current limitation:** the planner can only choose subtasks that already exist in the trained
tree. An instruction like "put the coke can in the mug" has no trained subtask, so the plan
comes back empty (`unreachable`) or wrong. Also, "just place the mug" still resolves to
mug → coke, because single-option steps are taken without asking. The eval side does **not**
have this limitation: `plan` mode runs any list of success checks + instructions. See the TODO
at the bottom.

**Eval rollout** (`mode: plan`, `generic_eval_task.py`): episodes start from the random preset
pool. The VLA is given step 0's instruction. When step *i*'s predicate becomes true, the
instruction switches to step *i+1*.

The subtle part is the **fresh-completion gate**. Step *i+1* only counts if its predicate goes
false→true *after* it becomes active. Without this, if the policy happened to place the coke
while it was supposed to be placing the mug, then placed the mug, the coke step would be
credited instantly at handoff even though it was done out of order. At each handoff the env
records whether the next predicate is already true; if so, it must become false and then true
again before it counts.

Metrics: `eval/subtask_1_success_once`, `eval/subtask_2_success_once`, `eval/success_once`
(whole chain), `eval/final_subtask_idx`.

---

## 6. Where things are

| Path | What |
|---|---|
| `rlinf/envs/isaaclab/tasks/discovery/` | `manifest.py` (tree storage), `predicates.py` (reads the success-check list from `conditionals.py` + validates proposals), `phase_a.py`, `phase_b.py`, `starting_states.py`, `vlm_client.py`, `orchestrator.py`. Importable without torch/IsaacLab. |
| `rlinf/envs/isaaclab/tasks/robolab_task.py` | Env + modes `single_edge`, `plan` (and the original `subtask_1/2`/`full`). |
| `RoboLab/robolab/tasks/benchmark/generic_{single_edge,eval}_task.py` | Generic task files. |
| `examples/embodiment/config/generic_{single_edge,eval}_grpo_openpi_pi05.yaml` | Train-one-edge / eval-a-plan configs (+ `env/`). |
| `logs/tree_manifest/driver.py`, `launch_driver.sh` | Autonomous run for this scene. |
| `logs/tree_manifest/launch_scripts/run_stage_inner.sh` | In-container launcher used per stage. |
| `logs/tree_manifest/eval_plans/{resolve,run_eval}.py` | Phase B harness. |
| `logs/tree_manifest/dashboard/` | Progress dashboard (`build.py`). |
| `tests/unit_tests/test_discovery_*.py` | Unit tests (no GPU). |

Run state (gitignored) under `logs/tree_manifest/`:

- **`manifest.json`**: the record of everything trained. `edges` holds one entry per trained
  subtask: `id`, `instruction`, `predicate`, `predicate_args`, `objects_involved`,
  `precondition` (what was done when it was trained), `produces_node` (what's done after),
  `checkpoint` (the checkpoint right after training it), `reset_states_path` (saved states it
  trained from). Also `terminal_nodes` (states with nothing more to propose) and
  `reset_paths_for_children` (state → end-states file its children start from). Training reads
  it to resume, continue the checkpoint chain, and find child start states; Phase B reads it to
  know what's trained.
- `edge_specs/<id>.json`: the spec handed to each training run. `end_states/<id>_end_states.jsonl`:
  captured success states. `root_candidates.json`: saved root proposals, reused to avoid
  re-calling the VLM. `eval_plans/*.json`: resolved plans.
- Display only: `driver_status.json` + `driver.log` (progress for humans; never read back by the
  pipeline), `level_<d>_queue.json` (written only so the dashboard can show queued subtasks), and
  everything in `dashboard/`.

Scripts under `logs/tree_manifest/` hardcode `/scratch/cluster/jshim12`; edit the constants at
the top when moving machines.

Batch-size rule for the configs (otherwise GRPO crashes): `total_num_envs` divisible by
`group_size × num_GPUs`, and `total_num_envs × rollout_epoch == global_batch_size`.

---

## 7. Commands

Inside the container (`bash /scratch/cluster/jshim12/enter_robolab.sh`) unless noted. Load the
key without printing it: `set -a; source /scratch/cluster/jshim12/.env; set +a`.
GPU jobs use all 8 GPUs: one at a time, and check logs for `Traceback`, not just exit code.

**Sanity check**
```bash
python -m pytest tests/unit_tests/test_discovery_*.py -q
python -m rlinf.envs.isaaclab.tasks.robolab_task        # env smoke test
```

**Preview discovery (VLM only, ~$0.40, no GPU)**
```python
from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest
from rlinf.envs.isaaclab.tasks.discovery.orchestrator import discover_tree
from rlinf.envs.isaaclab.tasks.discovery.predicates import build_predicate_menu, list_predicate_names
edges = discover_tree(Manifest("/tmp/preview.json"),
                      scene_objects=["ceramic_mug", "coke", "cutting_board_a"],
                      predicate_module=list_predicate_names(), predicate_menu=build_predicate_menu(),
                      stable_base_objects={"cutting_board_a"}, max_depth=2)
```
Below the root, preview has no captured end-states to ground on, which is why real runs use
the interleaved driver. Returns a list of candidate edges:
```json
[{"id": "place_coke_on_cutting_board",
  "instruction": "Place the coke can on top of the cutting board.",
  "predicate": "object_on_top",
  "predicate_args": {"object": "coke", "reference_object": "cutting_board_a", "require_gripper_detached": true},
  "objects_involved": ["coke", "cutting_board_a"],
  "precondition": [],
  "produces_node": ["place_coke_on_cutting_board"]},
 ...]
```

**Train one edge.** The spec is one candidate from above plus where to reset from:
```json
{"id": "place_ceramic_mug_on_cutting_board",
 "instruction": "Place the ceramic mug on top of the cutting board.",
 "predicate": "object_on_top",
 "predicate_args": {"object": "ceramic_mug", "reference_object": "cutting_board_a", "require_gripper_detached": true},
 "objects_involved": ["ceramic_mug", "cutting_board_a"],
 "precondition": ["place_coke_on_cutting_board"],
 "produces_node": ["place_ceramic_mug_on_cutting_board", "place_coke_on_cutting_board"],
 "reset_states_path": "logs/tree_manifest/end_states/place_coke_on_cutting_board_end_states.jsonl"}
```
```bash
LOG_DIR=$(pwd)/logs/$(date +%Y%m%d-%H:%M:%S)-my_edge \
bash examples/embodiment/run_embodiment.sh generic_single_edge_grpo_openpi_pi05 \
  +env.train.init_params.edge_spec_path=SPEC.json +env.eval.init_params.edge_spec_path=SPEC.json \
  env.train.init_params.reset_states_path=PARENT_END_STATES_OR_null \
  env.eval.init_params.reset_states_path=PARENT_END_STATES_OR_null \
  +actor.model.lora_path=PREV_CKPT/actor          # omit to start from base model
```
~12 h on 8×A40. Wrap in `setsid bash -c '...' </dev/null >run.log 2>&1 & disown` to survive
logout.

**Collect end-states:** same overrides with `eval_embodiment.sh`, `lora_path` = the new
checkpoint, plus `env.eval.init_params.save_end_state_path=logs/tree_manifest/end_states/<id>_end_states.jsonl`.
Use `+actor.model.lora_path`, never `runner.ckpt_path`, for LoRA checkpoints. One line per
successful episode; per object: position (3), quaternion (4), velocity (6):
```json
{"objects": {"cutting_board_a": [0.60, -0.20, -0.02, 1.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             "ceramic_mug":     [0.54,  0.20,  0.02, 0.60, 0.0, 0.0, 0.80, ...],
             "coke":            [...]},
 "robot_joint_pos": [...], "robot_joint_vel": [...]}
```
After training + collection, the edge is appended to `manifest.json` as the spec above plus
`"checkpoint": "logs/<run>/.../global_step_<N>/actor"`.

**Whole tree, unattended** (host, key exported)
```bash
bash logs/tree_manifest/launch_driver.sh
cat logs/tree_manifest/driver_status.json     # phase, current edge, errors
```
```json
{"phase": "training",              // starting | training | training_done | collecting_end_states | DONE | FAILED
 "current_edge": "place_coke_on_cutting_board__given_place_mug_on_cutting_board",
 "warm_start_from": "logs/.../place_ceramic_mug_on_cutting_board/.../global_step_60/actor",
 "log_dir": "logs/20260806-11:00:25-generic_single_edge_grpo_openpi_pi05-place_coke_...",
 "updated_at_utc": "2026-08-06T11:00:30Z"}
```
On failure: `"phase": "FAILED"` plus `error` and `traceback`.
Scene knobs at the top of `driver.py`: `SCENE_OBJECTS`, `STABLE_BASE_OBJECTS`, `CONFIG_NAME`,
`MAX_EPOCHS`, base checkpoint (`edge1_checkpoint.txt`). Defaults: `max_depth=3`,
`max_total_edges=30`. Fresh tree: point `MANIFEST_DIR` at an empty dir and remove seed files.

**Instruction → plan → eval**
```python
from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest
from rlinf.envs.isaaclab.tasks.discovery.phase_b import resolve_plan
import json
plan = resolve_plan("Put the coke on the board, then the mug.", Manifest("logs/tree_manifest/manifest.json"))
json.dump([{k: e[k] for k in ("predicate", "predicate_args", "instruction")} for e in plan],
          open("logs/tree_manifest/eval_plans/my_plan.json", "w"))
```
`my_plan.json`:
```json
[{"predicate": "object_on_top",
  "predicate_args": {"object": "coke", "reference_object": "cutting_board_a", "require_gripper_detached": true},
  "instruction": "Place the coke can on top of the cutting board."},
 {"predicate": "object_on_top",
  "predicate_args": {"object": "ceramic_mug", "reference_object": "cutting_board_a", "require_gripper_detached": true},
  "instruction": "Place the ceramic mug on top of the cutting board."}]
```
```bash
LOG_DIR=$(pwd)/logs/$(date +%Y%m%d-%H:%M:%S)-generic_eval_grpo_openpi_pi05-my_plan \
bash examples/embodiment/eval_embodiment.sh generic_eval_grpo_openpi_pi05 \
  env.eval.init_params.plan_path=$(pwd)/logs/tree_manifest/eval_plans/my_plan.json \
  +actor.model.lora_path=FINAL_CKPT \
  runner.logger.logger_backends=[]
```
Omit `lora_path` for a base-model baseline (name the dir `...-my_plan-baseline` for the
dashboard). `logger_backends=[]` disables wandb, which can time out on slow nodes; metrics still
print to the log. ~40 min. Result line in `eval_embodiment.log`:
```
[INFO 03:22:17 RLinf] {'eval/subtask_1_success_once': array(0.95875), 'eval/subtask_2_success_once': array(0.4225),
                       'eval/success_once': array(0.41875), 'eval/final_subtask_idx': array(1.3775),
                       'eval/num_trajectories': 800, ...}
```

**Dashboard:** `python logs/tree_manifest/dashboard/build.py` → `dashboard/dashboard_built.html`.

---

## 8. New scene

1. Build the scene + preset initial conditions (`RoboLab/.claude/skills/robolab-scenegen`).
2. Every goal must be expressible with `conditionals.py`; add predicates there if not.
3. Write its `initial_conditions.json` and point both readers at it (see "Initial conditions" in section 2), plus the generic task files at the new scene/objects.
4. In `driver.py`: new `SCENE_OBJECTS`, `STABLE_BASE_OBJECTS`, empty `MANIFEST_DIR`, base checkpoint.
5. Run the cheap discovery preview first, then the driver.

---

## TODO: free decomposition at test time

**Goal.** Real-to-sim scene → VLM proposes as many subtasks as it can in sim → the policy is
continually finetuned on all of them → sim-to-real transfer. At test time, any long-horizon
language instruction is broken down by the VLM into subtasks, and **every** subtask it produces
is sent to the VLA, whether or not it was trained on it.

**Gap.** `resolve_plan()` only picks from subtasks already in the trained tree. Instructions
needing anything else ("put the coke can in the mug", "stack the coke on the mug") fail.

**Plan.**
1. New function `decompose_instruction(instruction, scene_objects, starting_state_summary)` in
   `discovery/phase_b.py`, next to `resolve_plan` (keep the old one for comparison):
   - One VLM call returning an ordered list of steps, each `{instruction, predicate,
     predicate_args}`, built from the same success-check list and real-state grounding Phase A
     uses.
   - The prompt includes the trained subtasks as reference (preferred when they fit), but the
     VLM is free to output anything expressible with the success checks.
2. Validate each step with the same checks as Phase A (`predicates.validate_phase_a`: check
   exists, objects exist, not already satisfied at that point in the plan, not passive).
   - Open question: what to do when no success check can express a step (e.g. "pour",
     "open"). Options: reject the instruction, or add new checks to `conditionals.py`.
3. Label each step:
   - `trained`: matches a manifest edge by content (same check + arguments) and its trained
     precondition equals the steps completed before it.
   - `trained_out_of_context`: the check matches a trained edge, but it's being run from a
     different preceding state than it was trained from.
   - `new`: nothing in the manifest matches.
   Save labels in the plan JSON.
4. Eval: no change needed to the rollout. `plan` mode already accepts any list.
   - Extend metrics to report success per label (`trained` / `trained_out_of_context` / `new`),
     plus whole-chain success.
   - Remove the 2-step metric hardcoding in `robolab_task.py` (`subtask_1/2_success_once`) so
     plans longer than 2 steps get per-step metrics.
5. Test set: the two existing instructions (all `trained`), plus instructions requiring new
   subtasks in the same scene (coke in mug, coke on mug, mug next to coke), base vs. final
   checkpoint for each.
6. Update this doc's Phase B section and dashboard composed-eval cards to show per-step labels.

## TODO: scene-agnostic initial conditions

Both readers of the root initial conditions (`RoboLab/robolab/tasks/benchmark/starting_states.py`
and `discovery/starting_states.py`) are hardcoded to `RoboLab/assets/objects/polaris/initial_conditions.json`
and PolaRiS's key names (`latteartcup_eval`, ...). For image-to-sim scenes:
1. Convention: `RoboLab/assets/scenes/<scene>/initial_conditions.json`, same format, keyed by the
   scene's own object names.
2. Pass the path via config (`init_params.initial_conditions_path`) to the sim-reset reader and
   via `driver.py` to the prompt reader; drop `ROOT_POSE_KEY_ALIASES` when keys already match.
3. Have the image-to-sim step write this file alongside the generated `.usda`.
