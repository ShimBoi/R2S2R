---
name: open-ended-task-proposal
description: "Proposes the next tasks for a robot task curriculum from one state of a RoboLab scene: reads a request file (the node's history of completed tasks, the scene's objects and task file, the success predicates, every task proposed so far with its measured success rates, and optional images) and writes task specs a DROID-trained policy (pi0.5) can be fine-tuned on, each with a difficulty guess. Use when the curriculum orchestrator or a user hands over a proposal request file, or asks what tasks to train next from a node."
---

# Open-ended task proposal

Given one state of a scene (a node of the curriculum tree), propose tasks that the policy cannot do yet but
plausibly can learn next, each written as a task spec. The orchestrator screens every candidate on the GPU
(16 episodes each, with the current policy) and trains the ones closest to 50%. Your job is to give screening
good candidates, not to decide what gets trained.

You read one request file and write one response file. Never launch jobs, edit the tree or run the simulator.
Paths are relative to the repository root (the one containing `RoboLab/`).

## The request

- `node`: `id`, `level`, and `paths`: one list of instructions per path from `s0` that reaches this node, in
  order. These tasks were done to get here. `s0` has one empty path: nothing has been done yet.
- `scene`: `base_task` (every spec uses it), `task_file` (read it: objects, layouts, what starts where) and
  `objects` (the only names a spec may use, plus `gripper` where an argument allows it).
- `predicates`: every success predicate with its arguments and a one-line meaning. Read
  `RoboLab/robolab/core/task/curriculum_conditionals.py` for the details of the ones you use.
- `tree`: every task proposed so far. `screened` is successes out of `n` with the `level` and `ckpt` it was
  measured at; `eval` is the held-out success rate after training, if trained.
- `images`: renders or photos of the node's start states, if any. Look at them.
- `goal`: an optional user goal. Prefer tasks that move toward it.
- `min_candidates`: propose at least this many. There is no upper limit.
- `response`: the path to write.

## Procedure

1. Read the task file and the images. List what each object affords in this state: graspable, a surface, a
   container, something that opens.
2. Read the node's paths and the tree. Work out what the current policy can already do: tasks trained to a
   high eval and tasks screened near `n`/`n` are easy for it now; tasks screened at 0 are out of reach.
   Screens from earlier levels are older than the current policy, which has trained since; trained tasks only
   get easier.
3. Generate candidates, then cut every one that breaks a rule below.
4. Write each spec, its `difficulty_guess` and `rationale`, and the response file.

## Rules

- **One verb per task.** A step that needs two verbs is two tasks on consecutive levels.
- **Never trivial.** The success condition must be false in every start state of this node, and reaching it
  must move an object. No "keep the mug upright" when it starts upright, no "put the mug next to the board"
  when it starts there. Check the task file's layouts and the images, not just the instruction.
- **Not too easy.** Do not propose a task you expect the current policy to succeed at in more than about 90%
  of episodes (difficulty 1 below): nothing that matches or barely varies a task screened at or near `n`/`n`,
  or one trained to a high eval.
- **Not out of reach.** Avoid tasks far beyond anything the policy has shown, such as a long sequence or a
  precise insertion when simple placements still screen low. Prefer the next step up from what it can do.
- **Do not repeat the path.** Nothing that is already a task on any of the node's paths, in any wording: after
  "put the mug on the cutting board" then "pick up the mug", not "put the mug back on the cutting board".
  Undoing a step is fine when the result is a task the path has not done: after "put the mug on the cutting
  board", "put the mug on the table" is a new task.
- **Nothing already proposed from this node**, or from a node merged into it, including blocked tasks.
  Rephrasing a task is not a new task. Tasks proposed from other nodes are not duplicates: they start from
  different states, and their rates are evidence of how hard a similar task is here.
- **Success uses a predicate from the menu**, with arguments that mean what the instruction says. Placement
  tasks set `require_gripper_detached: true`, so holding the object over the target does not count.
- **Success is what a person would judge from the instruction**, neither looser nor stricter. Read how the
  predicate is computed before using it. A check that also passes for something else (touching for
  "grab", pressing for "push") is too loose. A check that adds conditions the words do not say (a narrow cone
  for "in front of", lifting measured from the board the object rests on, so carrying it off the board fails
  "pick up") is too strict; measure a lift from the table. If no choice of arguments makes the task honest
  from this node's start states, drop the task rather than bend the check. If no
  predicate fits, you may add one to `curriculum_conditionals.py`: add only, never edit or delete an existing
  one, and say so in `notes`.
- **Prefer tasks whose end states open useful next steps**, so the tree grows into chains.
- **Spread candidates** over different objects, targets and predicates.

## Instructions: DROID style

The policy was trained on DROID, whose instructions are short imperative phrases a person would say to the
robot. Write them that way:

- one verb, the object and the target: "put the marker in the cup", "place the mug on the cutting board"
- everyday names, as a person sees them: "the coffee mug", "the cutting board", "the red cup"; never the
  scene's asset names (`ceramic_mug`, `cutting_board_a`)
- lower case is fine, no trailing explanation, usually 3 to 10 words

Good, DROID-like: "pick up the coffee mug", "put the mug on the cutting board", "move the mug to the left of
the cutting board", "put the spoon in the bowl", "open the drawer", "close the lid of the box", "push the cup
to the back of the table", "turn the cup upside down", "stack the red cup on the blue cup".

Not like this: "pick up the mug and then put it on the board and push it" (three verbs), "tidy up the table"
(no single success condition), "translate ceramic_mug 10 cm along +x" (robot-speak and asset names), "carefully
and slowly lift the mug while keeping it level so nothing spills" (padding the policy never saw).

## Difficulty guess

For each candidate, guess how often the current policy succeeds, before any training on it:

| difficulty_guess | Expected success now |
|---|---|
| 1 | above 90% (too easy: do not propose) |
| 2 | 60 to 90% |
| 3 | 30 to 60% |
| 4 | 10 to 30% |
| 5 | below 10% |

Base it on the tree: what screened how, at which level, and what has trained since. The guess is logged next to
the measured rate to calibrate later proposals; nothing filters or trains on it. Aim most candidates at 3 or 4.

## The response

Write exactly this JSON to the request's `response` path:

```json
{
  "node": "<the request's node id>",
  "candidates": [
    {
      "spec": {
        "base_task": "<the request's base_task>",
        "instruction": "put the mug on the cutting board",
        "success": {"predicate": "object_on_top",
                    "args": {"object": "ceramic_mug", "reference_object": "cutting_board_a",
                             "gripper_name": "gripper", "require_gripper_detached": true}}
      },
      "difficulty_guess": 3,
      "rationale": "the mug starts on the table; placing it is one step up from grasping, which screened 14/16"
    }
  ],
  "notes": "anything the orchestrator or a person should know, such as an added predicate"
}
```

Leave out `init_states`: the orchestrator sets each spec's start states from the node. Every candidate is
checked with `validate_spec` (`RoboLab/robolab/core/task/task_spec.py`); one that fails is cut, so check
object names and argument types against the menu and that file's checks before writing.
