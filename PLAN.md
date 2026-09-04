# Open-Ended Task Discovery and Decomposition: a VLM High-Level Policy

Extends `long_horizon_task_composition_plan.md` (subtask_1 → subtask_2 → eval) from "exactly two
hand-specified subtasks in a fixed order" to a **decision tree of subtasks a VLM discovers**, trained
incrementally and later sequenced live at eval time. Revises the first draft of this document: that
draft assumed a flat, order-fixed list of subtasks; this one reflects that it's actually a tree over
"which subtasks are already satisfied," per the discussion with Jiaheng. Starts with the same
mug/coke/cuttingboard scene already built, rather than a new/larger object set — deliberately, so
this is a generalization of what exists, not a new testbed.

---

## 0. The decision tree, and how the existing plan is already one edge of it

A **node** is a specific set of already-satisfied subtasks (root = `{}`, nothing done). An **edge**
out of a node is one RL finetuning run: reset from that node's captured end-states, train until some
subtask succeeds, and that success defines a **child node** = parent's set plus the new subtask. For
the 2-object scene:

```
                    {}                      <- root: standard 100-preset scattered start
                 /      \
   {latteartcup_on_board}   {coke_on_board}         <- two possible first moves
           |                      |
   {both, order=cup-first}   {both, order=can-first}  <- kept SEPARATE, not merged (see below)
```

`subtask_1` (existing plan) is exactly the left edge out of root. `subtask_2` is exactly the edge
from `{latteartcup_on_board}` to `{both, cup-first}`. Nothing about the existing task files,
`reset_to_captured_state`, or the `mode` mechanism needs to change — what's new is: (a) more edges,
discovered by a VLM instead of hand-specified, (b) the tree is built by a recursive/automated loop
rather than two manual runs, (c) at eval time the "next subtask" choice has to respect which node
you're actually at.

**Why `{both, cup-first}` and `{both, can-first}` stay separate nodes, not merged into one "both
done" node:** even though the final object positions look similar either way, the robot's arm
trajectory getting there differs, and the whole point of `reset_to_captured_state` capturing full
state (not just object poses) was to make resets faithful to what a checkpoint actually produced.
Merging would mean training subtask_2-style edges against a captured pool blending two genuinely
different precondition states. Costs some redundant training (two separate finetunes ending up at a
similar-looking final state) but avoids DAG-merging logic for a savings that likely isn't worth it —
storing more captured states is cheap, per Jiaheng's point.

**Exact-match preconditions, not subset-match — this is the one place worth being careful about
generalizing too loosely.** A subtask should only ever be considered valid to invoke when the current
completed-set exactly equals its recorded precondition, not merely contains it. If a larger scene had
a third object and you'd only ever trained "object_1, from nothing," invoking that subtask in a state
where object_3 also happened to already be done would be exactly the reset-distribution mismatch the
original plan's `reset_states_path` design exists to avoid — training for that subtask never saw a
reset including object_3 already placed. Every validation and lookup below uses exact match.

**One checkpoint, not one per edge or one per branch.** The tree above describes *what gets trained
and in what order*, not separate model lineages — there's a single policy, continually finetuned
through the whole tree using the CRL technique, the same one already shown to scale across many
sequential tasks. This has a real consequence for the tree structure: if root branches into two
children, you can't train each branch as its own separate continuation from a common ancestor and
expect one final checkpoint that knows both — that gives you two divergent checkpoints, one per leaf
path. What actually has to happen is a single sequential order over the *whole* tree (respecting that
a node's edge can't be trained before its parent exists), with the one checkpoint carried through all
of it. The tree still determines what to train next and which precondition/instruction goes with each
stage — it just doesn't mean multiple lineages. §1 works through the two ways to sequence that.

### 0.1 What "dynamic" actually covers — one tree per scene, not per overarching task

Worth being explicit about the scope, since it's easy to conflate two different things this design
makes dynamic:

**Within one scene's already-trained tree — genuinely dynamic, no new env or task file per
overarching task.** Any overarching task expressible as some path through the subtasks already
discovered and trained for that scene's object set runs through `resolve_plan` (§5.1) and the same
generic eval task file, whatever the specific wording. That's the "just pass in a task and eval it"
ease this was built for, and it applies to any number of distinct overarching tasks against the same
scene, not just one.

**Across scenes with different objects — not dynamic, and can't be, by construction of what RL
training means.** If a second overarching task needs an object or a skill outside anything ever
discovered (Phase A) and trained (§4's CRL sequence) for some scene, there's no success criterion and
no trained behavior for it — a policy has no way to execute something it has no experience with, and
no amount of live decomposition logic substitutes for that. This isn't a gap in the design to close
with more engineering; it's the actual cost of the approach, same as it would be for any RL-trained
policy. Getting a new scene evaluable requires the same discovery-and-train loop from §4 to run for
it, once — a coding agent (`robolab-scenegen` for the scene, then the orchestrator for discovery and
training) is the right tool for that step specifically, not for skipping it. Given the CRL method is
about one checkpoint scaling across many tasks, the natural extension across scenes is the same one
already established within a scene: the same checkpoint lineage keeps extending into each new scene's
stages, rather than a fresh model per scene.

---

## 1. One checkpoint, built sequentially — edge-per-stage vs. level-per-stage

Two ways to flatten the tree into a sequence of CRL stages for the single checkpoint, and the
tradeoff is real, not just a style choice:

**Edge-per-stage:** every single (precondition → subtask) pair is its own sequential stage. Simple to
set up — one precondition, one captured-states pool to reset from, one instruction, no per-episode
branching inside the stage. Number of stages = number of edges in the tree, which can grow fast once
a level has many nodes (a wider scene with more objects means more possible orderings at each depth).

**Level-per-stage:** one stage per tree depth, covering every node at that depth simultaneously.
Fewer, larger stages — but the stage becomes multi-task internally: different envs in the same
rollout batch need different preconditions (whichever nodes exist at that depth), and each has to get
the *correct* instruction for *its own* precondition, not one instruction for the whole batch. That's
the mapping you're describing, and it's real, but worth being precise about what kind of complexity it
actually is: **the mapping itself is static and known before the stage starts** — Phase A's discovery
calls for everything through depth L have already run by the time you're training depth L, so "which
precondition maps to which instruction" is a lookup into already-collected data, not a live decision
during rollout. The new engineering is in the reset+mixin mechanism, not in needing more intelligence:

```python
def reset_to_level_mix(env, env_ids, level_specs: list[dict]):
    """level_specs: one entry per node at this depth, each with its own
    {"reset_states_path", "predicate", "predicate_args", "instruction"}.
    Each env independently samples one entry -- not one shared choice for the whole batch."""
    chosen = [random.choice(level_specs) for _ in env_ids]
    for spec in level_specs:
        matching = [eid for eid, c in zip(env_ids, chosen) if c is spec]
        if matching:
            reset_to_captured_state(env, torch.tensor(matching),
                                     captured_states=load(spec["reset_states_path"]), ...)
    for eid, spec in zip(env_ids, chosen):
        env._active_spec_per_env[eid] = spec  # mixin reads this per-env, not one shared value
```

The mixin's success-check and instruction-assignment (currently one shared predicate/instruction for
the whole batch, per `long_horizon_task_composition_plan.md` §5.2) would need to become per-env,
looking up `_active_spec_per_env[eid]` instead of a single shared `_subtasks_cfg`.

**Before committing to either, this is a question only your paper can answer, not something I should
guess at:** does your CRL method itself assume strictly one new task introduced per stage, or does it
support (or need) multi-task mixing within a stage? If it's characterized specifically for
"sequentially introduce one task at a time," level-batching isn't just more implementation work, it
might fall outside what your method's results actually cover. And separately — has it been validated
on a *branching* task structure, or a linear sequence? If the latter, edge-per-stage with a
topological flattening is the safer match to what you've already shown works, and level-batching would
be a new claim about the method, not just an engineering choice.

For right now, at 2 objects (at most 4 edges total either way), this barely matters — I'd validate the
whole pipeline with whichever matches your method's actual assumptions, and revisit the choice
specifically when the object count grows enough that edge count becomes the bottleneck. A middle
ground exists too, worth keeping in mind rather than treating this as strictly binary: batching a few
same-level nodes together without going all the way to "every node at that level, however many."

---

## 2. The subtask spec, revised with a precondition

```json
{
  "id": "coke_on_cuttingboard__given_latteartcup_on_cuttingboard",
  "precondition": ["latteartcup_on_cuttingboard"],
  "predicate": "object_on_top",
  "predicate_args": {"object": "coke", "reference_object": "cuttingboard", "require_gripper_detached": true},
  "instruction": "Pick up the coke can and place it on the cutting board",
  "objects_involved": ["coke", "cuttingboard"],
  "checkpoint": "/path/to/checkpoint",
  "reset_states_path": "/path/to/parent/end_states.jsonl",
  "produces_node": ["coke_on_cuttingboard", "latteartcup_on_cuttingboard"]
}
```

`precondition` is the exact parent node (sorted list of subtask ids, `[]` for root). `produces_node`
is the resulting child node. `checkpoint` is a snapshot of the *single* continually-trained policy
right after this stage finished (not a separate model per edge — see §1) — recorded per edge mainly
so the next stage in the sequence knows exactly where to resume from, and for debugging/resuming if
a later stage needs to be re-run. The manifest for a scene is the list of these — an edge list for
the tree, not a flat list of independent options. `predicate`/`predicate_args` still must reference
the existing function library in `conditionals.py` (~35 atomic predicates, confirmed by grep) —
nothing about that part of the design changes.

---

## 3. Phase A prompt — propose the next edge(s) out of a given node

Stateful now: takes the current node (which subtasks are already satisfied) and proposes what's
plausible *from there* — which may be several options at the root, narrowing as the tree gets deeper,
or nothing at all once there's nothing useful left to do.

```
You are proposing the next training subtask(s) for a robot manipulation curriculum, built
incrementally as a decision tree over "which subtasks are already satisfied."

SCENE OBJECTS (exact names, use verbatim):
{object_list}

ALREADY-SATISFIED SUBTASKS AT THIS POINT IN THE TREE:
{satisfied_so_far}
  # e.g. [] for the root, or ["latteartcup_on_cuttingboard"]

SCENE IMAGE (current reachable state -- i.e. what the scene looks like once everything in
ALREADY-SATISFIED has actually happened):
{scene_screenshot}

AVAILABLE SUCCESS-CHECK FUNCTIONS (use only these, never invent new ones):
{predicate_menu}

TASK
Propose ALL the plausible next atomic subtasks from this point in the tree. There may be more than
one reasonable option (e.g. at the root, either object could go first), exactly one (e.g. once one
object is placed, there's usually only one sensible thing left), or none (return an empty list if
nothing further is useful given what's already satisfied).

Do not propose a subtask that's already satisfied. Do not propose anything requiring an object not
in the scene or a predicate not in the list above.

OUTPUT: a JSON list (possibly empty). Each entry:
{
  "id": "<snake_case_identifier, unique>",
  "predicate": "<exact function name from the list above>",
  "predicate_args": {"<param_name>": "<exact object name or value>", ...},
  "instruction": "<natural-language instruction>",
  "objects_involved": ["<object names used>"]
}
Do not include "precondition" or "produces_node" -- those are filled in by the calling script from
ALREADY-SATISFIED, not by you, since they need to be exact copies, not paraphrased.
```

---

## 4. The orchestrator — automating the discovery-and-train loop

Recommendation, not just options: **a plain script drives the mechanical loop; the VLM call is one
step inside it.** RL finetuning is long-running and checkpoint-driven — a job-scheduling problem, not
a reasoning problem — and a script is far more robust to being left running across the many stages a
tree like this will have, even a small one. Reserve Claude Code / a human for: initial scene setup
(one-time), and stepping in when something the script can't resolve alone happens (training doesn't
converge, a captured state looks wrong on spot-check, the VLM's proposal keeps failing validation).

This sketch assumes **edge-per-stage** (§1) and a single evolving checkpoint carried through a
topologically-ordered flattening of the tree — not a separate lineage per branch, which was wrong in
the first draft of this section. Discovery (deciding what to train next) still walks the tree
recursively; the actual training sequence is one flat, ordered list built from that walk, all
continuing the same `current_checkpoint`:

```python
def discover_tree(node, max_depth=4, visited=None):
    """Recursively discovers edges via Phase A, without training anything yet -- returns a
    topologically-ordered list of edges to train (parents always before children)."""
    visited = visited if visited is not None else set()
    ordered_edges = []
    if node in visited or len(node) >= max_depth:
        return ordered_edges
    visited.add(node)

    parent_entry = manifest.get_producing_edge(node)  # None for root
    reset_states_path = parent_entry.reset_states_path_for_children if parent_entry else None

    candidates = call_vlm_phase_a(
        object_list=SCENE_OBJECTS, satisfied_so_far=sorted(node),
        scene_screenshot=render_current_node(node), predicate_menu=PREDICATE_MENU,
    )
    candidates = validate_phase_a(candidates, known_objects=SCENE_OBJECTS, predicate_module=conditionals)
    if not candidates:
        manifest.mark_terminal(node)
        return ordered_edges

    for candidate in candidates:
        candidate["precondition"] = sorted(node)
        candidate["produces_node"] = sorted(node | {candidate["id"]})
        candidate["reset_states_path"] = reset_states_path  # None -> default 100 presets, root only
        ordered_edges.append(candidate)
        ordered_edges += discover_tree(node | {candidate["id"]}, max_depth, visited)
    return ordered_edges


def train_sequence(ordered_edges, base_checkpoint):
    """The actual CRL sequence: ONE checkpoint, carried through every edge in order."""
    current_checkpoint = base_checkpoint
    for edge in ordered_edges:
        config_path = generate_training_config(edge, reset_states_path=edge["reset_states_path"])

        current_checkpoint = launch_and_wait(
            f"bash examples/embodiment/run_embodiment.sh {config_path}",
            warm_start_from=current_checkpoint,  # same lineage every stage, not per-branch
            poll_for_convergence=...,
        )

        # Collect end-states from this stage for whichever child edges need them (reuses
        # eval_embodiment.sh + save_end_state_path exactly as in the existing plan)
        child_states_path = launch_and_wait(
            f"bash examples/embodiment/eval_embodiment.sh {config_path} "
            f"runner.ckpt_path={current_checkpoint} "
            f"env.eval.init_params.save_end_state_path={paths.new_states_file(edge['id'])}"
        )

        edge["checkpoint"] = current_checkpoint
        manifest.add_edge(edge)
        manifest.set_reset_states_path_for_children(edge["produces_node"], child_states_path)

    return current_checkpoint  # the final, single, continually-trained policy
```

(If you go with level-per-stage instead, `discover_tree` still walks the same way to find what
exists at each depth, but `train_sequence` groups edges by depth and calls the level-mix reset from
§1 once per depth rather than once per edge — the discovery half doesn't change, only how the
training loop batches what it found.)

Things this sketch leaves for you to fill in rather than guessing at: `poll_for_convergence` (fixed
step count is simplest to start with; a real convergence check needs a threshold on the
`subtask_1_success_once`-style eval metric from the existing plan, checked periodically); the
ordering among siblings when a node has multiple children (topological order only requires parents
before children — beyond that, whether to do all of one branch before starting the other, or
interleave, is exactly the kind of thing your CRL paper's own findings about task ordering and
forgetting should decide, not something I'd guess at); `render_current_node`; and retry/failure
handling (I'd have the script give up on one edge, log clearly, and continue rather than blocking the
whole sequence on one bad stage).

---

## 5. Phase B prompt — decomposition at eval time, now precondition-aware

Same reactive shape as before (re-invoked once per subtask transition, not a single upfront plan),
but the validation now checks the *exact* node, not just repertoire membership:

```
You are the high-level controller for a robot arm. A low-level policy executes exactly one
instruction at a time; you decide which instruction to give it next.

OVERARCHING TASK (from the human operator):
{overarching_task}

SUBTASKS ALREADY COMPLETED THIS EPISODE, IN ORDER:
{completed_so_far}
  # this defines which node of the trained tree you're currently at

SUBTASKS AVAILABLE FROM THIS EXACT POINT (precondition == completed_so_far exactly -- these are
the ONLY valid next choices; nothing else was ever trained from this specific state):
{available_edges}
  # e.g.:
  # - id: coke_on_cuttingboard__given_latteartcup_on_cuttingboard
  #   instruction: "Pick up the coke can and place it on the cutting board"

CURRENT SCENE STATE (ground truth, computed programmatically -- trust this over anything you might
infer visually):
{state_description}

TASK
Decide the single next subtask to execute, chosen from SUBTASKS AVAILABLE FROM THIS EXACT POINT
only, to make progress on the overarching task.

Return ONLY JSON, exactly one of:
{"status": "next", "subtask_id": "<id from the available list above>", "reasoning": "<one sentence>"}
{"status": "done", "reasoning": "<why the overarching task is now satisfied>"}
{"status": "unreachable", "reasoning": "<what's missing from the trained tree>"}

Never return a subtask_id that isn't in the AVAILABLE list above -- not the broader set of
everything ever trained, only what's valid from exactly this point.
```

The validation change from the first draft: check `subtask_id` against `manifest.edges_from(node)`
(exact match on `node == completed_so_far`), not against the full flat manifest. This is the concrete
enforcement of the exact-match precondition rule from §0 — regardless of whether training happened
edge-per-stage or level-per-stage (§1), a single trained subtask should only be offered as valid when
you're actually at the precondition it was trained under.

### 5.1 Resolving the plan once, before the rollout — not live during it

Important simplification, and the concrete answer to "how does the instruction swap happen
autonomously during the trajectory": **it doesn't happen live during the trajectory at all.** Look at
where genuine decisions exist in a tree like this one — the root may have several valid options, but
every non-root node reached so far in this scene has exactly one. There's nothing to decide at a
node with one valid edge; call the VLM only where real ambiguity exists, and take the only option
mechanically everywhere else. So: resolve the whole decomposition once, before the episode starts,
by walking the manifest:

```python
def resolve_plan(overarching_task, manifest, node=frozenset()):
    """Walk the manifest from root, calling the VLM only at genuine branch points (>1 valid edge).
    Returns a plain ordered list of edge specs. No VLM calls happen after this returns."""
    plan = []
    while True:
        edges = manifest.edges_from(node)  # exact-match precondition lookup
        if not edges:
            break  # leaf -- nothing more to do
        elif len(edges) == 1:
            chosen = edges[0]  # no ambiguity, no VLM call
        else:
            response = call_vlm_phase_b(overarching_task, completed_so_far=sorted(node), available_edges=edges)
            if response["status"] != "next":
                break  # "done" or "unreachable"
            chosen = next(e for e in edges if e["id"] == response["subtask_id"])
        plan.append(chosen)
        node = node | {chosen["id"]}
    return plan
```

For the mug/coke tree (§8), this means exactly **one** VLM call total (at the root, deciding cup-
first vs. can-first) resolves the entire plan — the second step is deterministic traversal, no call
needed. Nothing here is specific to a 2-node tree: `while True` keeps walking regardless of depth,
and the `len(edges) > 1` check fires at *whichever* nodes happen to be genuine branch points, however
deep. A tree with, say, 5 objects and a branch point 3 levels in behaves identically — deterministic
traversal until that node, one VLM call there, deterministic again after. The resulting plain list (`[{"instruction": ..., "predicate": ..., "predicate_args": ...},
...]`) is assigned to the mixin once, at reset, and the ratchet just walks it using the coded
predicate checks from §0/§2 — no API calls anywhere inside `step()`:

```python
def reset(self, *args, **kwargs):
    obs, info = super().reset(*args, **kwargs)
    # self._active_plan set externally, once, before rollout starts, from resolve_plan() above
    self._plan_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._task_descriptions = [self._active_plan[0]["instruction"]] * self.num_envs
    return obs, info

def step(self, action):
    ...
    for eid in range(self.num_envs):
        idx = self._plan_idx[eid].item()
        if idx >= len(self._active_plan):
            continue
        spec = self._active_plan[idx]
        predicate_fn = getattr(conditionals, spec["predicate"])
        if predicate_fn(self, env_id=eid, **spec["predicate_args"]):
            self._plan_idx[eid] += 1
            self._task_descriptions[eid] = (
                self._active_plan[idx + 1]["instruction"] if idx + 1 < len(self._active_plan) else "done"
            )
    terminated = terminated | (self._plan_idx >= len(self._active_plan))
    ...
```

**This is also why a Claude Code session isn't the right tool for this specific call.** Claude Code is
well-suited to orchestrating §4's offline discovery-and-train loop — long-running, resumable,
benefits from an agent or human checking progress. The live "what's next" decision, when one is
actually needed, is a single synchronous API call (Anthropic or OpenAI, plain text-in/JSON-out) made
directly from `resolve_plan`, called once before the episode starts — not an interactive coding
session, and (per this section) not something that needs to run *during* the simulated trajectory at
all for a tree this shallow. Live re-planning mid-episode — re-calling the VLM if execution diverges
from what was expected — is a real option for later, once there's an actual reason to want it (a
deeper or more branch-heavy tree where deterministic traversal alone isn't enough); it isn't needed
to get this scene working.

---

## 6. Everything else from the first draft, unchanged

- **Predicate dispatch by name** (generalizing the mixin off hardcoded `object_on_top`) — same code
  change as before, unaffected by the tree structure.
- **Validation before anything runs** — same principle (check predicate names and object names
  against ground truth before trusting any VLM output), now also checking precondition exactness (§4).
- **Claude Code skill for offline authoring** — still the right complement for when a genuinely new
  scene/object set is needed, separate from the automated discovery-and-train loop in §4.
- **Cost/latency/caching for Phase B** — same reasoning: cache the decomposition per overarching-task
  string where possible, since re-deriving it from scratch on every eval episode is wasted API calls
  for the common case where the same task text recurs.

---

## 7. Open questions, updated

1. **Does your CRL method assume one-task-per-stage, or does it support multi-task mixing within a
   stage?** (§1) This is the one that should actually decide edge-per-stage vs. level-per-stage, not
   my own aesthetic preference — if your published results specifically characterize sequential
   single-task introduction, level-batching would be a new claim about the method, not just an
   implementation choice.
2. **Has it been validated on branching task structures, or only a linear sequence?** (§1) If the
   latter, edge-per-stage with a topological flattening (§4) is the safer match to what's already
   published; the ordering among sibling branches would then be worth deciding based on whatever your
   paper found about task-order effects on forgetting, rather than an arbitrary choice.
3. **Max tree depth / breadth budget** (§4) — worth an explicit cap before running this unattended.
   For the 2-object scene, is depth 2 (both orders, both leaves — 4 edges total) enough to validate
   the pipeline before scaling up?
4. **Convergence criterion for "done training this stage"** (§4) — fixed step count vs. a real
   plateau check on the eval success rate. Fixed-count is simpler to script first.
5. **What `render_current_node` actually does** (§4) — does Phase A get a fresh screenshot per node,
   or is the scene simple enough that the satisfied-subtasks list alone is sufficient context?

---

## 8. Worked example — the full flow for the mug/coke/cuttingboard scene, edge-per-stage

Concrete, step by step, with real object names and example VLM outputs. Ends with one final
checkpoint that has sequentially learned all four edges of the tree.

### 8.0 One-time setup

- Scene: `mug_and_coke_on_cutting_board.usda` (already built per the existing plan — the copy of the
  original, extended with `coke`).
- Root starting-state pool: `initial_conditions.json`'s 100 rows (already built).
- Predicate menu: extracted once from `conditionals.md` for the Phase A prompt.
- **Two generic task files, not four hardcoded ones** — this is the one thing not fully spelled out
  before now. Confirming edge-per-stage collapses what would've been per-edge files into two:
  - `generic_single_edge_task.py` — used for every training stage. Takes an injected edge spec
    (predicate, predicate_args, instruction, and a `reset_states_path` that's `null` for root edges
    and a real path otherwise) via the same module-attribute injection mechanism already built for
    `subtask_2`'s `RESET_STATES_PATH` — extended to inject the whole spec, not just the path.
    `events()`: if `reset_states_path` is set, `reset_to_captured_state`; else `SharedRandomization`
    (the null-check that already existed for subtask_2 turns out to *be* the root-vs-non-root
    distinction, no separate file needed for root edges after all). `terminations`: time_out only,
    the mixin decides success by dispatching `predicate`/`predicate_args` by name (§ predicate
    dispatch, unchanged from the first draft).
  - `generic_eval_task.py` — the ratchet, generalized from exactly 2 fixed instructions to however
    many steps Phase B's decomposition returns. Same `SharedRandomization` (root pool) always, since
    eval always starts from scratch. Termination: mixin-decided, ratchet reaches the end of whatever
    sequence Phase B produced.
- Manifest starts empty.

### 8.1 Discovery + training, edge by edge

**Phase A call #1 — node `{}` (root).** Inputs: `object_list=["latteartcup","coke","cuttingboard"]`,
`satisfied_so_far=[]`. Example output:
```json
[
  {"id": "latteartcup_on_cuttingboard", "predicate": "object_on_top",
   "predicate_args": {"object": "latteartcup", "reference_object": "cuttingboard", "require_gripper_detached": true},
   "instruction": "Pick up the latte art cup and place it on the cutting board",
   "objects_involved": ["latteartcup", "cuttingboard"]},
  {"id": "coke_on_cuttingboard", "predicate": "object_on_top",
   "predicate_args": {"object": "coke", "reference_object": "cuttingboard", "require_gripper_detached": true},
   "instruction": "Pick up the coke can and place it on the cutting board",
   "objects_involved": ["coke", "cuttingboard"]}
]
```
Two candidates — both objects are equally valid first moves from an empty scene. Validated: both
predicates exist in `conditionals.py`, both object names are in the scene.

**Edge 1 — `{} → {latteartcup_on_cuttingboard}`.** Config: `reset_states_path: null` (root pool),
`predicate: object_on_top`, `predicate_args` per above. `bash run_embodiment.sh <config>` — this is
the **first** CRL stage, so no warm-start, trains from the base VLA checkpoint. Converges →
`checkpoint_v1`. Collect: `eval_embodiment.sh <config> runner.ckpt_path=checkpoint_v1
env.eval.init_params.save_end_state_path=cup_done_states.jsonl`.

**Edge 2 — `{} → {coke_on_cuttingboard}`.** Same `reset_states_path: null` — **root pool again, not
`cup_done_states.jsonl`** — this is a sibling of edge 1, not a continuation of it; both start from
scratch. What *does* continue is the checkpoint: warm-started from `checkpoint_v1`, so the CRL method
is what's responsible for it learning "place the can" without forgetting "place the cup" it just
learned in edge 1. Converges → `checkpoint_v2`. Collect →
`eval_embodiment.sh ... runner.ckpt_path=checkpoint_v2 ... save_end_state_path=can_done_states.jsonl`.

**Phase A call #2 — node `{latteartcup_on_cuttingboard}`.** `satisfied_so_far=["latteartcup_on_cuttingboard"]`.
Example output — exactly one candidate now, not two:
```json
[{"id": "coke_on_cuttingboard__given_cup", "predicate": "object_on_top",
  "predicate_args": {"object": "coke", "reference_object": "cuttingboard", "require_gripper_detached": true},
  "instruction": "Pick up the coke can and place it on the cutting board",
  "objects_involved": ["coke", "cuttingboard"]}]
```
This is Jiaheng's "once one succeeds, there's only one next task possible" — the VLM correctly
narrows once the cup is already accounted for.

**Edge 3 — `{latteartcup_on_cuttingboard} → {both, cup-first}`.** `reset_states_path:
cup_done_states.jsonl` (edge 1's captured states — full state, arm included, not synthetic).
Warm-started from `checkpoint_v2` (which by now knows both first-moves). Converges →
`checkpoint_v3`.

**Phase A call #3 — node `{coke_on_cuttingboard}`.** Mirror of call #2 — exactly one candidate,
`latteartcup_on_cuttingboard__given_can`.

**Edge 4 — `{coke_on_cuttingboard} → {both, can-first}`.** `reset_states_path:
can_done_states.jsonl` (edge 2's captured states). Warm-started from `checkpoint_v3`. Converges →
**`checkpoint_v4` — the final checkpoint.**

**Phase A calls at the two `{both, ...}` leaf nodes** each return `[]` — nothing useful left once
both objects are placed — so the tree stops growing here. Total: 4 edges, 4 sequential CRL stages,
one checkpoint that has now seen, in order: cup-from-scratch, can-from-scratch, can-given-cup-done,
cup-given-can-done.

The finished manifest is 4 entries like the one in §2, each with its `checkpoint` field recording
which snapshot resulted (all pointing into the same lineage — `checkpoint_v4` is the one that
actually matters for eval, but the intermediate ones are kept for resuming/debugging per §2).

### 8.2 Eval, with `checkpoint_v4`

Human gives an overarching task, e.g. *"put the latte cup and the coke can on the cutting board."*
Before the episode starts, `resolve_plan` (§5.1) walks the manifest:

**Root — genuine branch point.** Two valid edges (`latteartcup_on_cuttingboard`,
`coke_on_cuttingboard`), so this is the one real decision. Example VLM output:
```json
{"status": "next", "subtask_id": "latteartcup_on_cuttingboard", "reasoning": "Starting with the cup since the task mentions it first."}
```

**`{latteartcup_on_cuttingboard}` — no ambiguity.** Exactly one valid edge
(`coke_on_cuttingboard__given_cup`) — taken directly, no VLM call.

**`{both, cup-first}` — leaf.** No further edges; `resolve_plan` stops. Total VLM calls to resolve
this entire episode's plan: **one.**

The rollout then runs with the plain resolved plan
`[{"instruction": "Pick up the latte art cup...", "predicate": "object_on_top", "predicate_args": {"object": "latteartcup", ...}}, {"instruction": "Pick up the coke can...", "predicate": "object_on_top", "predicate_args": {"object": "coke", ...}}]`
— `checkpoint_v4` executes step 0's instruction; the mixin checks
`object_on_top(latteartcup, cuttingboard, ...)` every step using the plain code from §5.1, no API
calls involved. The instant it fires, `task_descriptions` switches to step 1's instruction — same
checkpoint, no model switching, purely a coded-condition-triggered instruction swap. When step 1's
condition fires too, `_plan_idx` reaches the end of the plan, `terminated=True`,
`success_once = True`, `final_subtask_idx = 2`.

If the human's overarching task had needed something the trained tree can't do (e.g. a fourth object
never part of this scene), `resolve_plan` would have returned `status: "unreachable"` before the
rollout ever started — the episode would never begin executing rather than failing partway through.

That's the whole loop, start to finish, for this scene: 2 Phase A discovery calls that branch, 2 that
narrow to one option each, 4 sequential CRL training stages producing one final checkpoint, and
exactly 1 VLM call to resolve the entire eval-time plan before the rollout even starts — everything
after that is coded predicate checks and a plain list, no API calls inside the simulation loop.
