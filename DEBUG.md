# Debugging reference

Real bugs hit while building this pipeline, kept as a lookup table: match your symptom, apply
the fix. Not a history of how each was found.

## Training / env

**`RuntimeError: shape '[-1, 8]' is invalid for input of size 1`**
Cause: `total_num_envs` isn't a multiple of `group_size × num_GPU_ranks`.
Fix: keep `total_num_envs / (group_size × num_ranks)` an integer (e.g. 64 / (8 × 8) = 1). Also
keep `total_num_envs × rollout_epoch == global_batch_size`.

**`IsADirectoryError` when loading a checkpoint**
Cause: passing a LoRA adapter directory as `runner.ckpt_path` — `torch.load()` expects a single
file.
Fix: use `+actor.model.lora_path=<checkpoint dir>` instead, never `runner.ckpt_path` for LoRA.

**Hydra `ConfigCompositionException: Could not append to config... 'actor.model.lora_path'`**
Cause: `+actor.model.lora_path=...` passed twice (e.g. a training-time warm-start override reused
verbatim for a later eval/collection stage that needs its own value).
Fix: strip any existing `+actor.model.lora_path=` override before adding a new one.

**`TypeError: Configuration for the term '__module__' is not of type EventTermCfg`**
Cause: a task file's `_events()` returned a bare `@configclass` class instead of an instance
(missing `()`).
Fix: `return SharedRandomization()`, not `return SharedRandomization`.

**Full-mode eval overall `success_once` inflated / out-of-order subtask still counts as
success**
Cause: a ratchet advancing subtask N+1 in the same env-step it also re-checks N+1's completion,
so an already-satisfied-out-of-order condition gets credited immediately on handoff.
Fix: gate completion on a fresh (re-)satisfaction observed *after* handoff, not just "currently
true" — snapshot whether the next condition was already true at handoff, require it go false
then true again before crediting.

**Training-mode reward doesn't penalize touching the wrong object**
Not a bug — single-subtask training reward is a flat pass/fail on the target object only, by
design. If you want a "clean" single-object skill, add a penalty/gate on the other object's pose
moving during training; not implemented here.

## Orchestration (discovery / tree training)

**Only one lineage of a branching tree gets trained**
Cause: depth-first traversal (`discover_and_train()`) fully recurses into the first child before
trying siblings — from a root with 2 children it only ever walks one.
Fix: use `discover_and_train_by_level()` (breadth-first) instead — trains every edge at a depth
before advancing, still threading one checkpoint through the whole flattened order.

**Resumed run warm-starts a later depth from the wrong (too-old) checkpoint**
Cause: the "already trained, skip" branch didn't update the running `checkpoint` variable.
Fix: `checkpoint = manifest.all_edges()[-1]["checkpoint"] if len(manifest) else base_checkpoint`.

**Manifest id collision on a relaunched/resumed run**
Cause: seeded initial edges (`initial_edges_by_node`) bypassed the normal dedup path.
Fix: dedupe seeded nodes explicitly before use, same as freshly-discovered ones.

**`gpt-4o` JSON response parse failure**
Cause: model appends trailing prose after a ` ```json ` fence, or returns a bare object instead
of a one-element list.
Fix: extract with `re.search(r"```(?:json)?\s*(.*?)```", ...)` instead of naive `strip("`")`;
wrap a bare dict response in `[dict]`.

**A real crash reports exit code 0**
Cause: `run_embodiment.sh`/`eval_embodiment.sh` pipe through `tee`, which does not propagate the
piped process's real exit status.
Fix: never trust exit code alone — grep the log for `Traceback`/`Error` fatal signatures too.

**A backgrounded training job dies the instant an agent session gets resumed/rebuilt**
Cause: `setsid`-less/non-detached background process gets SIGHUP'd when its parent shell session
tears down.
Fix: launch with `setsid ... </dev/null >logfile 2>&1 & disown` (full session detachment), and
poll for completion rather than nudging a live background job.

## Dashboard extraction

**Stale/wrong precondition shown for an edge**
Cause: candidate-preview files scanned in the wrong order — an older, non-deduped file
(`*_children.json`) read before the current, correctly-deduped one (`level_*_queue.json` /
`root_candidates.json`).
Fix: scan order must be `level_*_queue.json` → `root_candidates.json` → `*_children.json`.

**Same edge shown twice under different ids**
Cause: a re-discovery pass can legitimately re-propose the same real edge, renamed by id-dedup
since the raw id collides with an already-trained edge.
Fix: dedupe by content signature (`precondition` + `predicate` + `predicate_args`), not by id.

## Infra / node

**Vulkan "No device could be created" / `vk_icdGetInstanceProcAddr` failures**
Cause: a bad cluster node — the host driver's Vulkan ICD bootstrap fails unconditionally for
every caller, confirmed by even a minimal raw ctypes Vulkan probe hitting the identical failure.
Fix: allocate a different node. Not a config problem — don't re-debug ICD binding/env vars here.

**`undefined symbol: __malloc_hook` (or `__realloc_hook`/`__free_hook`/`ErrorF`) at `dlopen`
time, breaking `vkCreateInstance`**
Cause: one specific old NVIDIA driver build (535.216.03) references glibc malloc hooks removed
in glibc >=2.34. Host-driver-version-specific, unrelated to the bad-node issue above (though it
can present with a similar symptom).
Fix: `LD_PRELOAD` the shim: `gcc -shared -fPIC -o setup/libmalloc_hook_shim.so
setup/glibc_malloc_hook_shim.c`, then `export LD_PRELOAD=$(pwd)/setup/libmalloc_hook_shim.so`.
Only needed if you hit this exact error.

**A freshly-allocated node is extremely slow on the first heavy Python import (minutes for
`import torch`)**
Cause: cold local page cache — even if the same files were read on a *different* node recently,
each node's cache is independent. Compounded when many subprocesses (e.g. torch-inductor's
compile-worker pool) all cold-read the same large libraries concurrently.
Fix: not a bug, just a one-time cost per node — let it finish; subsequent imports on that node
are fast.

**`wandb.sdk.lib.service.service_port_file.ServicePollForTokenError: Failed to read port info
after 30.0 seconds`**
Cause: wandb's local sync service has its own hardcoded 30s startup-readiness timeout, which can
fire under the same elevated subprocess-spawn latency as the cold-cache issue above.
Fix: disable wandb for the run if sync isn't required: `runner.logger.logger_backends=[]`.

**`Exception: The current node timed out during startup... raylet failed to startup or the GCS
has become overloaded`**
Cause: same elevated subprocess-spawn latency as above — the GCS server itself starts fine (its
own log shows healthy heartbeats), Ray's client-side wait just times out first.
Fix: transient — retry. Not a real config problem.
