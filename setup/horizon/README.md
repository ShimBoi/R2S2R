# Running on TACC Horizon (GB200, aarch64)

This branch runs RLinf + RoboLab on Isaac Sim 6.2 / Isaac Lab 3.0. Isaac Sim 5.1 (`main`) has no
aarch64 build and crashes on Horizon's drivers.

## One-time setup

```bash
W=$SCRATCH/r2s2r            # workspace: container sandbox + caches (home quota is ~46 GB)
# 1. Container sandbox, from setup/rlinf_u22.def (Ubuntu 22.04)
module load tacc-apptainer
apptainer build --sandbox $W/rlinf_sandbox setup/rlinf_u22.def
# 2. Python venv (.venv-il3 in this repo), the RoboLab assets not vendored here, and the
#    pi0.5 checkpoint (model/pi05_droid_jointpos, ~7 GB)
bash setup/horizon/enter_container.sh bash setup/horizon/install.sh
bash setup/horizon/enter_container.sh bash setup/horizon/fetch_robolab_assets.sh
bash setup/horizon/enter_container.sh bash setup/horizon/get_model.sh
```

## Running

Every command goes through the container, on a compute node:

```bash
srun -p debug -A CDA26011 -N 1 -n 1 -t 2-00:00:00 --mem=800G --pty bash
bash setup/horizon/enter_container.sh bash examples/embodiment/eval_embodiment.sh robolab_mug_grpo_openpi_pi05 \
    runner.only_eval=True 'runner.logger.logger_backends=[tensorboard]'
```

`+rollout.enable_torch_compile=True` makes pi0.5 rollouts ~3x faster. Verified for evals; not yet
for training.

## What `enter_container.sh` works around

- TACC's XALT tracker sets `LD_PRELOAD` and bind paths that don't exist in the container.
- Host settings (`CC=nvc`, module dirs in `PATH`, CUDA variables) would leak into the container.
- srun/ssh shells can be pinned to one CPU core; `taskset` gives the container the allocation.
- Apptainer's `--nv` misses driver libs: `libnvidia-rmapi-tegra` (Vulkan on GB200), NGX and
  OptiX (RTX denoising). They're bound at the node's own driver version.
- Isaac Sim on aarch64 won't start unless `libgomp` is preloaded.

`install.sh` also patches Isaac Sim's aarch64 startup check, which otherwise crashes inside
RLinf's daemonic env worker.

## Driver 610 vs 615

Horizon nodes run driver 615 or 610, and Slurm can't tell them apart. Check with
`nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1`.

- **615:** Isaac Sim's default renderer (real-time path tracing + DLSS Ray Reconstruction).
- **610:** NGX can't create DLSS Ray Reconstruction (`NGX CreateFeature failed: 0xbad00002`), so
  default renders are noise and the policy gets ~0%. `enter_container.sh` detects 610 and sets
  `R2S2R_CARB_SETTINGS` to path tracing + the OptiX denoiser. Images match 615 visually and pi0.5's
  success rate matched, but rendering is ~2.3x slower.

Only compare evals run with the same renderer. Set `R2S2R_CARB_SETTINGS='{}'` to opt out.

## Port fixes on top of the Isaac Lab 3 migration

- Physics uses Isaac Lab 3's default `enable_external_forces_every_iteration=True`. The Isaac Lab 2
  setup on `main` ran with False, so results from before the port aren't directly comparable.
- `WorldState.visualize_object` converts xyzw to wxyz for the axis display.
- RLinf registers RoboLab envs without the viewport camera, which RLinf never reads but RTX
  rendered every step (`init_params.viewport_camera`, default False).
