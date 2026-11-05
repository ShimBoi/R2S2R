# real2sim-robolab reference

Paths are relative to the repo root (your R2S2R checkout or worktree) unless absolute.
`SK=.claude/skills/real2sim-robolab`. `S2R=RoboLab/robolab/core/sim2real` (the shared sim-to-real package).
Everything here is generic. The reconstructions in §1 are cited only as labelled worked examples. Copy their
patterns, never their numbers. Their files may be on other branches or outside the repo; skip any you cannot
read.

## Contents

1. Worked examples: bread_on_plate_lab (phone photos), bread_on_plate_lab2 (policy frames)
2. Conventions
3. Running (container, nodes, commands)
4. Intake questions, report template
5. Supplementary images
6. Metric scale
7. Cameras
8. Objects and physical properties
9. Scene geometry
10. Background
11. Lighting and image match
12. Task definition
13. Validation
14. Domain randomization
15. Helper scripts
16. Known gaps and open decisions

---

## 1. Worked example: bread_on_plate_lab

This is an example only. The scene, objects and numbers below belong to one lab bench.

| What | Where |
|---|---|
| Working scripts (bread-specific originals) | `/scratch/11369/jshim12/r2s2r/bread_scene/*.py`, `eval_*.sh`, `views_*.json` |
| Task | `RoboLab/robolab/tasks/benchmark/bread_on_plate_lab_task.py` |
| Scene | `RoboLab/assets/scenes/bread_on_plate_lab.usda`, written by `bread_scene/gen_scene.py` |
| Textures, background | `RoboLab/assets/lab_scene/{gingham.png, lab_background.png}` |
| Objects | `RoboLab/assets/objects/lab/{plate,baguette,twist_bread}/` (`prep_assets.py`, `make_twist_bread.py`) |
| External camera | `LabZedLeftCameraCfg` in `RoboLab/robolab/variations/camera.py`; preset `WRIST_LAB` in `RoboLab/robolab/registrations/droid/camera_presets.py` |
| RLinf configs | `examples/embodiment/config/env/robolab_bread_on_plate_lab.yaml`, `examples/embodiment/config/robolab_bread_grpo_openpi_pi05.yaml` |
| Intensity override in background event | uncommitted `intensity=` parameter in `RoboLab/robolab/core/events/randomize_background.py` |
| Review bundle | `/scratch/11369/jshim12/r2s2r/bread_scene/for_review/` |

All of it was uncommitted work on the first workstation; treat the paths as historical.

What that reconstruction did, and what to learn from it:

- **Its inputs were only 3 phone photos.** There was no frame from the policy camera. The policy camera's
  pose was inferred from the camera body visible in the photos, using tablecloth vanishing points and the
  camera body's known width (`vp.py`, `rectify.py`).
  - The result was only good to about ±10° in heading and ±15% in scale.
  - The inferred camera was very close to the table: 0.16 m above the top, yaw -45°, pitch -20°.
  - A real policy-camera frame would have pinned it down. This is why this skill requires labelled
    policy-camera images.
- **The background was first filled with smooth (Telea) inpainting.** The user rejected it as blurry.
  It was redone with SDXL inpainting plus outpainting (`inpaint_photos.py`, then `make_background.py`).
  This is why the skill forbids blur.
- **The wrist camera was rotated 180° on the user's recollection.** That took pi0.5 zero-shot success from
  79% to 35%. Checking the real wrist image showed the default DROID mount was right.
  Always verify the wrist mount against a real wrist frame. Never change it from memory.
- **The procedural object differed from the library asset.** A procedural twisted roll (`make_twist_bread.py`:
  14.5 cm long, 40 g, friction 1.2/1.0, convex hull) replaced a library baguette (120 g). Zero-shot success went
  79% → 88%: lighter, grippier and shorter is easier. Object physics moves success by about 10 points.
- **Open mismatch: the image pipeline.** RLinf renders the external camera straight to 224x224 with the full
  horizontal FOV. The vertical FOV is then as wide as the horizontal one. The real pi0.5 client letterboxes 16:9
  to 224x224. See §7.4.
- **The FOV values disagree.** `views_*.json` used `hfov_deg` 101.9, but the camera class's aperture gives
  2·atan(1.2726/1.0476) = 101.08°. Always derive the FOV from the class, not from memory.
- **Zero-shot results (800 held-out episodes):**

  | Run | Success |
  |---|---|
  | v0, blurry background | 80.9% |
  | Wrist rolled 180° | 35.1% |
  | Default wrist | 79.0% |
  | Twisted roll | 88.0% |

### 1.2 Worked example 2: bread_on_plate_lab2 (policy-camera frames, no measurements)

This is an example only. The same bench was rebuilt from one ZED-left frame, one wrist frame (both at the
DROID home pose) and two phone photos. Nothing was measured; the user accepted the uncertainty and planned
domain randomization for it. Files are on branch `real2sim-skill-test` in the worktree `R2S2R-real2sim/`:

| What | Where |
|---|---|
| Task | `RoboLab/robolab/tasks/benchmark/bread_on_plate_lab2_task.py` (preset 0 = photo layout via `ROBOLAB_PHOTO_LAYOUT=1`) |
| Camera | `LabZedLeftV2CameraCfg`, preset `WRIST_LAB2` |
| Scene spec per version | `real2sim_test/scene_v*.json` → `RoboLab/assets/scenes/bread_on_plate_lab2.usda` |
| Labels, version log, sheets | `real2sim_test/skilltest/{labels.json, versions.md, v*/sheet.jpg}` |
| Sim-to-real plan (example) | `real2sim_test/sim2real_plan.md` |

How it went:

- **v1: pattern-plane fit.**
  - `fit_lattice.py` on the gingham: fx 614 (92.4° HFOV), k1 ≈ -0.05.
  - `plane_from_camera.py` on the wrist frame: plane yaw 88°, z = 0.0975 at an assumed 6 cm period.
  - `compose_ext.py` with the bread as anchor put the robot base within 5 px of where it is in the photo.
- **The old phone-only camera was badly wrong.** It was 16 cm above the table looking flat across it. The real
  camera is about 36 cm above the table, pitched down 25°.
- **Scale from the robot.** The robot-silhouette scan (scales 0.92–1.25) peaked at 1.08–1.13 (IoU 0.77).
  - So the period is about 6.5–6.75 cm, and the table top is 0.05 m above the base mount, not at z = 0.
  - Objects grew with the scale: plate 22 cm, mugs 7 × 8.5 cm, bread 16 cm.
- **Version loop:**

  | Version | Failures | Fix for the next version |
  |---|---|---|
  | v6 | plate 15 px off; table far edge 10 px (table too long and not rotated); far cloth 16% too dark; wrist 25% too bright | v7: table x1 0.96 → 0.89, yaw -2.5°, plate moved 3 cm, RoboLab sphere light 5000 → 2700 warm |
  | v7 | far cloth 12% too dark | v8: back rect light 4800 at (1.3, 0.45, 1.6) |
  | v8 | none: PASS 13/13 | — |

  Earlier, v2–v5 fixed the floor plane hiding the photo dome and a waffle maker that was too tall.
- **v9–v14: the waffle maker articulated**, for the user's open-the-waffle-maker task.
  - The lid hinges at the white tape, per the user.
  - The dial block stands inside the lid's footprint, so the lid is notched around it.
  - Lid position was fitted from rim points in both views. It was 13% too large at first, then moved 2 cm
    and raised 8 mm.
  - The white mug was moved 5 cm: its base is hidden by the lid, so the first guess was wrong.
  - The sphere light was dimmed after the waffle maker's shadow changed.
  - v14 passes every check, with polygon labels for the lid and the white mug.
- **Lessons (all folded into SKILL.md):**
  - Whole-image overlay scores are meaningless with a photo dome.
  - The cloth folds region had to leave the static polygons, because the sim cannot represent cloth folds.
  - Visibility-toggle masks pick up render noise and settling objects; use instance segmentation.
  - A single-bin FFT period is about 9% coarse; zero-pad and match the direction.
  - `refine_camera.py`'s Nelder-Mead did not move on noisy renders.

### 1.3 Run 2: the whole skill end to end on the same four images

Work dir `real2sim_test/run2/`, task files `bread_on_plate_lab4_task.py` and `open_waffle_bread_lab4_task.py`,
scene `lab4.usda`, camera `LabZedLeftV4CameraCfg` / `WRIST_LAB4`, DR plan `RoboLab/assets/lab4_scene/dr.json`.

| Step | Result |
|---|---|
| `fit_lattice.py` ext / wrist | fx 613 (HFOV 92.4°); wrist 34/37 inliers |
| `plane_from_camera.py`, `compose_ext.py`, scale scan | IoU peak 0.771 at scale 1.125 (plateau 1.08–1.15); table z 0.0465 m; camera within 5 mm of run 1 |
| `backproject.py` | plate, bread, mugs, waffle maker (centre from the dial, yaw −14°) |
| v1 | lid 66 / 79 px off; pink mug 18 px; white mug IoU 0.56 |
| v2 | mugs moved by their back-projected offsets; lid mask restricted to the shell parts the label outlines (`lid=shell_main+...`): the housing now moves with the lid but is not in the label |
| v3 | lid moved (+2.0, −2.5) cm, agreed by both views: PASS |
| `check_articulation.py` | caught all envs' lids at env 0 (static base) → kinematic base → `env_spread` 0, closes from ≤ 65°, opens from ≥ 75° |
| v4 (kinematic base) | PASS |
| `check_dr.py` | 15 envs × 2 resets; real image at the 10th–60th percentile of every statistic; no absurd frames |
| Base eval, waffle task | The policy went to the lid and lifted it slightly, but the lip was a visual-only tape strip with a 3 mm overhang: nothing to grip. Lesson: grasp parts need collision and about a finger's depth of overhang (SKILL.md Phase 3.5). |

### 1.4 Run 3: a fresh agent with only the skill

The same four images, no other context, branch `real2sim-skill-test2`. v8 passed 24/24. It inferred the
hinge and moving set correctly (lid = shell, top, tape lip, dial housing; hinge at the housing's outer bottom
edge) and gave the lip collision (5 × 2.4 × 1.2 cm, 2 cm out). Its gaps are now folded into SKILL.md: unique
names, gripper in the robot mask, near-field wrist background, non-linear light solves, fixing bad labels,
rolling round objects, damping units, cameras that disagree, the camera-preset change, parallel renders,
sparse asset fetch.

## 2. Conventions

| Thing | Convention |
|---|---|
| World / env frame | Robot base frame: +x forward, +y left, +z up. Origin at the robot base (robot `init_state.pos = (0, 0, 0)` in `DroidCfg`). |
| Camera `OffsetCfg.rot` | Quaternion **(x, y, z, w)** in Isaac Lab 3 (`isaaclab/sensors/camera/camera_cfg.py`). Use with `convention="opengl"` (looks along -Z, +Y up). |
| PnP / OpenCV | cv camera frame (+Z forward, +Y down). `cam_math.gl_quat_to_cv_R` / `cv_R_to_gl_quat` convert. |
| Preset poses (`reset_pose_to_presets`) | `(x, y, z, qw, qx, qy, qz)`, **scalar-first**. They are converted to xyzw internally. |
| Body / root poses read from sim | `body_quat_w`, `root_quat_w` are xyzw. Positions are world: subtract `env.scene.env_origins`. |
| Pinhole intrinsics | `fx = fy = W * focal_length / horizontal_aperture`, `cx = W/2`, `cy = H/2` (`Camera._initialize_intrinsics`). `vertical_aperture` defaults to `horizontal_aperture * H / W`. Off-centre principal points need `horizontal/vertical_aperture_offset` or `OpenCvPinholeDistortionCfg` (fields `fx, fy, cx, cy, image_size, k1..`) on `PinholeCameraCfg.distortion`. |
| Control | `dt = 1/120`, `decimation = 8` → **15 Hz** (`auto_register_droid_envs`). pi0.5 chunk = 15 actions (1 s). Actions are 7 absolute joint targets plus a gripper value; > 0.5 closes (`BinaryJointPositionZeroToOneAction`: `finger_joint` 0 or π/4). |
| Arm gains | `ImplicitActuatorCfg` stiffness 400, damping 80; effort 87/12 N·m; velocity 2.175/2.61 rad/s (`RoboLab/robolab/robots/droid.py`). The gripper uses its USD drive with effort limit 200. |
| Dome texture | Equirect: `u = -azimuth/2π`, `v = 0.5 - elevation/π`; +x forward is at u = 0. Verified by axis renders (`bread_scene/views_dome.json`). |

## 3. Running

- **Container.** Everything that imports Isaac runs through the container:
  ```bash
  cd <repo root>
  export R2S2R_VENV=$SCRATCH/r2s2r/R2S2R/.venv-il3 CUDA_VISIBLE_DEVICES=${GPU:-0}
  timeout 900 bash setup/horizon/enter_container.sh python $PWD/$SK/render_views.py \
      --task-file my_task_task.py --task MyTaskTask --preset WRIST_MYTASK --ext-fit $WD/fit_ext.json \
      --object-poses $WD/photo_layout.json --out $WD/render_v1 > $WD/render_v1.log 2>&1
  ```
  `enter_container.sh` handles the driver-specific library binds, including DLSS-RR on driver-610 nodes.
  If renders look like noise or the log says `Failed to create NGX context`, the libraries were not bound.
- **Login node.** CPU-only helpers (`fit_camera.py`, `rectify.py`, `overlay.py`, `color_match.py`,
  `make_dome.py`, `make_asset.py`, `gen_scene.py`) run there with the venv python directly:
  `$R2S2R_VENV/bin/python`. They need numpy, cv2, pxr, scipy and trimesh.
  `python -m robolab.core.sim2real.articulated_usd spec.json` runs there too, with `PYTHONPATH=RoboLab`.
- **Nodes.** Run GPU work only on a node the user names. Check `squeue -u $USER`, and never run on a node that
  hosts a live training job. A new node may need `ssh -o StrictHostKeyChecking=accept-new` the first time.
  Kill processes with bracketed `pkill -f` patterns.
- **Worktree for experiments.** Never edit the tree a training run uses. Setup:
  ```bash
  cd <repo root> && git -c core.hooksPath=/dev/null -c filter.lfs.process= -c filter.lfs.smudge=cat \
      -c filter.lfs.clean=cat -c filter.lfs.required=false worktree add -b <branch> ../R2S2R-<name> HEAD
  git diff | (cd ../R2S2R-<name> && git apply)        # carry uncommitted work
  ```
  - Copy untracked files you need.
  - Copy the ignored assets: `RoboLab/assets/{robots,fixtures,materials,backgrounds}`. `git status --ignored`
    lists them. Without them, renders fail with "Unable to resolve ... franka_robotiq_2f_85_flattened.usd".
    Copy them from another checkout if one exists. Otherwise fetch only those four folders: a full
    `setup/horizon/fetch_robolab_assets.sh` checkout takes hours of small-file writes on scratch, and a
    partial clone hangs on per-file LFS fetches. What works (inside the container, which has git-lfs):
    ```bash
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --no-checkout https://github.com/NVlabs/RoboLab $UP && cd $UP
    git sparse-checkout init --cone && git sparse-checkout set assets/robots assets/backgrounds assets/materials assets/fixtures
    git checkout -q b6cfcf6          # the REV in fetch_robolab_assets.sh
    for d in robots backgrounds materials fixtures; do git lfs pull --include="assets/$d/**"; cp -r assets/$d <worktree>/RoboLab/assets/; done
    ```
  - Copy real files over any LFS pointer files the scene uses.
  - `enter_container.sh` puts the worktree root first on `PYTHONPATH`, so its code is the code that runs.
- **Model downloads** (e.g. the SDXL inpainting model). `enter_container.sh` sets `HF_HOME` to the shared
  workspace cache. If you cannot write there, pass your own: `bash setup/horizon/enter_container.sh env
  HF_HOME=<dir> python ...`.
- **Parallel renders.** Version renders of different scene files can share a node if the task reads its scene
  and layout paths from env vars (SKILL.md Phase 7).
- **Isaac Lab 3 quaternions are xyzw** everywhere (`body_quat_w`, `quat_w_opengl`, `OffsetCfg.rot`).
  Reading them as wxyz flips a camera's look direction.
- **Base-policy eval** (pi0.5, held-out presets):
  ```bash
  timeout 7200 bash setup/horizon/enter_container.sh bash examples/embodiment/eval_embodiment.sh \
      robolab_<task>_grpo_openpi_pi05 runner.only_eval=True +rollout.enable_torch_compile=True \
      runner.logger.log_path=$WD/eval_base 'runner.logger.logger_backends=[tensorboard]' > $WD/eval_base.log 2>&1
  grep -oE "eval/success_once.: array\([0-9.]+" $WD/eval_base.log | tail -1
  ```
  The defaults are 80 envs × 10 eval epochs = 800 episodes, about 25-30 minutes. The background is drawn once
  per GPU per epoch, so the effective background sample size is GPUs × epochs. Compare only evals with the
  same epoch structure.
- **Exact policy inputs.** Copy the `bread_scene/policy_frames.py` pattern with your env YAML. It builds
  `RoboLabDroidEnv`, steps a hold chunk, and saves `main_images` / `wrist_images` as the policy gets them.

## 4. Intake questions, report template

**Checklist** (ask only what the images do not answer, in one message):

1. **Policy.** Which checkpoint, and which cameras it observes. Get the exact preprocessing: resolution,
   letterbox / crop / squash, JPEG, and any other augmentation.
2. **Cameras.** Model and serial for each policy camera. Get factory or DROID calibration (intrinsics,
   distortion, extrinsics) if they exist, and whether the policy sees rectified images.
3. **Joint state.** The joint angles (7 + gripper) at the moment each policy-camera image was taken.
   Otherwise, the policy's home pose, and whether the photo was taken there.
4. **Geometry.** Robot base to work-surface offset (x, y to a corner; z = surface height above the base mount).
   Surface size. Positions and sizes of fixtures.
5. **Objects.** For each: dimensions, mass (kitchen scale), material, softness (does it squish under the
   gripper; how much), whether it has moving parts, and whether it is transparent or reflective.
6. **Task.** The exact instruction string. The success condition (on top, inside, released, upright, ...).
   The real episode length. The real reachable region for start poses.
7. **Real performance.** Real success rate and number of trials. Any recorded real trajectories
   (joint stream, gripper, video).
8. **Lighting.** Windows, overhead lights, time of day, and whether the lighting changes between runs.

**Report** (`$WD/report.md` plus `$WD/for_review/`). Fill in two tables:

| Quantity | Value | Source (measured / user / photo-estimated / assumed) | ± uncertainty | Used where |
|---|---|---|---|---|
| Surface top z rel. base | 0.000 m | user tape | 0.005 | scene spec |
| obj_a mass | 0.040 kg | assumed (no scale) | 50% | make_asset, DR range |

| Assumption | Why | Effect if wrong | How to check |
|---|---|---|---|

Then add:

- the validation numbers: fit RMS, `overlay.py` lines for each camera, base-eval success with CI vs real
- the side-by-side images
- the open mismatches
- the exact render and eval commands

## 5. Supplementary images

Supplementary images are optional, unlabelled phone or other photos from any viewpoint.

- **Uses.**
  - Geometry the policy cameras cannot see: backs of objects, the full surface extent, fixtures behind the robot.
  - Clean, unoccluded texture for surfaces and background. More views means less generative fill.
  - Object identity, size and material.
  - Views behind and around the robot for the dome.
- **Best view per item.**
  - For each surface, take the photo that sees it most frontally and sharply. Check `rectify.py`'s
    "source pixels per output pixel" ≥ 1.
  - For each object, use the views that show its full silhouette plus a close-up for texture.
  - For the background, use views from near the policy camera's position looking outward.
    Photos taken far from that position put the wrong perspective onto the dome.
- **Posing them** (needed for `make_dome.py`). Run `fit_camera.py pnp` with ≥ 4 points of known position:
  the robot base, surface corners, fixtures, or robot bodies at a known joint state. Phone intrinsics come from
  EXIF (`focal_mm` with `sensor_width_mm`, or 35 mm equivalent: `fx = f35/36·W`) or from `--fit-focal`.
- **Multi-view reconstruction** is optional. If COLMAP or a learned multi-view method (DUSt3R/MASt3R-style) is
  available, run it on all photos together, policy-camera images included. It gives relative poses and a point
  cloud, to measure object sizes and the layout, and to cross-check the camera fits.
  - Fix its scale with one known distance.
  - Align it to the robot frame with ≥ 3 known points (Umeyama).
  - Never require it. PnP per photo is enough.
- **Appearance ground truth is the policy cameras.** Supplementary photos have different white balance,
  exposure and tone curve. Record that. Colour-match every texture cut from them to the same surface as seen
  by a policy camera:
  `color_match.py tex.png policy_cam.png out.png --src-region ... --ref-region ...`.
  Never take lighting or brightness from them.

## 6. Metric scale

Use any reference whose true size is known. Cross-check at least two. Record which ones you used.

- **User measurements** (best): tape-measured surface edges, distances between fixtures, object sizes.
- **The robot itself** (always present):
  - Franka link geometry: base `panda_link0` to shoulder axis 0.333 m; upper arm 0.316 m; forearm 0.384 m;
    flange offset 0.107 m. These are standard Panda DH lengths.
  - Robotiq 2F-85 stroke: 85 mm max opening.
  - Body origins at a known joint state: `refine_camera.py --dump-keypoints`.
- **Standard objects:**

  | Object | Size |
  |---|---|
  | A4 / Letter paper | 210×297 / 216×279 mm |
  | Credit card | 85.6×54.0 mm |
  | 12 oz can | ⌀ 66 mm × 122 mm |
  | Dinner plate | typically ⌀ 250–280 mm (measure) |
  | Standard door | 2.03–2.13 m |
  | Floor and ceiling tiles | 300/600 mm |
  | Pegboard holes | 1 in pitch |

- **Camera bodies of known size:** e.g. ZED 2 width 175 mm, ZED Mini 124.5 mm, RealSense D435 90 mm.
  Check the vendor spec for the exact model.
- **Repeating patterns.** Count the periods across a measured span. Never assume the period.

## 7. Cameras

### 7.1 Intrinsics

Priority:

1. **Factory calibration from the user.** For ZED, run `get_camera_information().camera_configuration.calibration_parameters.left_cam`
   (fx, fy, cx, cy, disto) for the resolution actually streamed. For RealSense, use `rs2_intrinsics`.
2. **DROID / robot-stack calibration files.**
3. **Datasheet FOV** at the streamed resolution. Treat it as approximate; vendors quote different FOVs per mode.
4. **Estimate from the image** with `fit_camera.py pnp --fit-focal`, using ≥ 8 spread points. Refinement can
   free fx too (`refine_camera.py --fit-focal`).

Write `PinholeCameraCfg` with `cam_math.py cfg` (any `focal_length` works; only the ratio matters).
Any resolution or FOV is supported: set the camera class's `width`/`height` to the real stream and the YAML
render size to what the policy gets (§7.4). Square pixels and a centred principal point are assumed. If
|cx − W/2| or |cy − H/2| > 1% of W, use `PinholeCameraCfg.from_intrinsic_matrix(...)` or `OpenCvPinholeDistortionCfg`.
If the policy sees **unrectified** images, model the distortion with `OpenCvPinholeDistortionCfg`. Otherwise use an ideal pinhole.

### 7.2 External / observation camera extrinsics (no pattern needed)

1. **Initial pose by PnP** (`fit_camera.py pnp`). Use ≥ 6 points spread over the image and in depth.
   Coplanar points alone are weak for focal estimation.
   - Points with known positions: robot base corners, surface corners, fixture corners, measured tape marks.
   - Robot bodies: dump body origins at the recorded joint state with
     `refine_camera.py --dump-keypoints kp.json --joint-pos ...`. Click the matching pixels: base, shoulder,
     elbow, wrist, flange, fingertips. Reference them as `"body": "panda_link4"`.
   - `vp` mode only when a straight-line grid lies on a plane. It picks the axis mapping with the lowest RMS.
2. **Refine by render-and-compare** (`refine_camera.py --camera ext --init fit_ext.json --real ext.png --joint-pos ...
   --real-mask robot_mask.png --score static.json`). It optimizes 6-DoF (plus fx with `--fit-focal`) so the
   sim robot silhouette and edges, plus static edges, match the real image. The robot is the calibration
   target, since its kinematics are known.
   - Make the real robot mask by filling a polygon around the robot in the real image, or with a segmenter if
     one is available.
   - Run it with 1 env and `--render-width 640`. Each evaluation is 2 renders × 4 steps.
3. **Phone-only fallback.** If no policy-camera frame exists, locate the camera body in posed supplementary
   photos and apply the lens offset from the body centre (stereo cameras: the left lens is half the baseline to
   the left). Expect about ±10° error. Ask for one real frame.
4. **Commit the result** as a new camera class and preset. These are existing files, so propose the diff:
   ```python
   # RoboLab/robolab/variations/camera.py  (cam_math.py / refine_camera.py print this block)
   @configclass
   class MyTaskExternalCameraCfg:
       over_shoulder_left_camera = TiledCameraCfg(
           prim_path="{ENV_REGEX_NS}/over_shoulder_left_camera", height=H, width=W, data_types=["rgb"],
           spawn=sim_utils.PinholeCameraCfg(focal_length=..., focus_distance=28.0, horizontal_aperture=..., vertical_aperture=...),
           offset=TiledCameraCfg.OffsetCfg(pos=(...), rot=(qx, qy, qz, qw), convention="opengl"))
   # RoboLab/robolab/registrations/droid/camera_presets.py
   WRIST_MYTASK = [MyTaskExternalCameraCfg, WristCameraCfg]
   ```
   Keep the attribute name `over_shoulder_left_camera`. `RoboLabDroidEnv._wrap_obs`
   (`rlinf/envs/isaaclab/tasks/robolab_task.py`) reads exactly `over_shoulder_left_camera` and `wrist_cam`.
   A policy with other camera sets needs a wrapper change (§16).

### 7.3 Wrist camera

- **Start from the mount extrinsic.** The sim default is `_WRIST_CAM` in `RoboLab/robolab/robots/droid.py`:
  parent `Robotiq_2F_85/base_link`, offset pos (0.011, -0.031, -0.074), rot xyzw (0.570, 0.576, -0.409, -0.420),
  opengl. Intrinsics `focal_length=2.8`, `horizontal_aperture=5.376` give 87.6° HFOV at any resolution.
  - If the user has a CAD or measured mount (bracket drawing, hand-eye calibration), convert it to an
    OffsetCfg relative to `base_link`.
  - Take intrinsics from the camera's spec or calibration (§7.1). Never assume a model.
- **Refine.** Use a real wrist image taken at recorded joints:
  `refine_camera.py --camera wrist --real wrist.png --joint-pos ... --gripper G --real-mask fingers_mask.png
  --score surface_edges.json`. The fingers fix the mount (they are rigid in the camera frame). Surface or
  object edges fix the residual. It prints `offset_pos` / `offset_rot` on `base_link`.
  - Check the roll: the fingers appear at the same image location as in the real frame.
- **Quick checks without refinement:** `render_views.py --wrist-offset px py pz qx qy qz qw`.
- **Applying a new offset** needs a change: either `_WRIST_CAM` (affects every task) or a per-task override of
  `cfg.scene.wrist_cam.offset` in `robolab_task.py`. Both are existing files. Propose before editing.

### 7.4 Policy image pipeline

The sim camera must produce what the real camera produces. The same preprocessing then runs on both.

- **pi0.5 (openpi).** `ModelTransformFactory` adds `ResizeImages(224, 224)`, which uses `image_tools.resize_with_pad`
  (letterbox). It runs inside RLinf too (`rlinf/models/embodiment/openpi/__init__.py`, `setup_wrappers`).
  The real client feeds full 16:9 frames.
  - Sim today: the env YAML `over_shoulder_cam`/`wrist_cam` = 224x224 renders a **square** with the full HFOV.
    That exposes a taller VFOV than the real letterboxed image, with no black bars.
  - Fix: set the YAML render size to the real aspect (e.g. `height: 126, width: 224`, or `252×448` for more
    detail). Openpi then letterboxes exactly like the real stack.
  - Verify with the policy-frames dump: black bars top and bottom, content identical to the real letterboxed frame.
- **OpenVLA-OFT.** `PrismaticImageProcessor` (`rlinf/models/embodiment/prismatic/processing_prismatic.py`)
  resizes with no letterbox, then center-crops if `center_crop` is set in the model config.
  - Read the checkpoint's `preprocessor_config.json` (`image_resize_strategy`, input size) and the real client's
    resize.
  - Render at the real aspect and let the same processor run.
- **Compression and noise.** If the policy's training data was JPEG-compressed or noisy, keep the sim clean and
  add it as augmentation (§14).

### 7.5 Pattern plane + robot scale scan (method A, no measurements)

Use this when the work surface has a periodic planar pattern (checked or gingham cloth, floor or wall tiles,
pegboard, a grid mat) and the policy-camera frames were taken at a known joint state.

```bash
# 1. lattice in both frames (check *_overlay.png: red dots on every pattern cell)
python $SK/fit_lattice.py ext.png $WD/lat_ext.json --auto
python $SK/fit_lattice.py wrist.png $WD/lat_wrist.json --auto --fix-fx <wrist fx> --roi 0,0,W,<above fingers>
# 2. wrist camera pose at the recorded joints: dump it from sim (camera pos / quat_w_opengl / K) into cam.json
python $SK/plane_from_camera.py $WD/wrist_cam.json $WD/lat_wrist.json --period <guess> --max-v <above fingers> \
    --point anchor:<u,v,dz> --out $WD/plane.json
# 3. one candidate per scale; --check-uv = pixel of the robot base in the ext frame
python $SK/compose_ext.py $WD/lat_ext.json $WD/plane.json --anchor anchor:<u,v in ext> --check-uv <u,v> \
    --scales 0.9 1.0 1.08 1.12 1.17 1.25 --out $WD/cands.json
# 4. robot-silhouette IoU per candidate (GPU), take the peak
python $SK/render_check.py ... --cands $WD/cands.json --real-robot-mask robot_mask.png --out $WD/scan
```

- **Anchor.** Pick an object visible in both frames and back-project its centre at roughly half its height
  (`--point name:u,v,dz`).
- **Check point.** It resolves the pattern's 4-fold yaw symmetry; a wrong yaw puts the robot base off by
  hundreds of pixels.
- **Scale.** Scale, plane height, anchor position and camera distance move together along the wrist camera's
  rays. Only the robot (known size) or a measurement breaks the tie. The robot mask can be a rough polygon,
  because IoU still peaks at the right scale.
- **fx.** It is observable from the strongly tilted external view, not from the near-fronto-parallel wrist
  view. Fix the wrist fx to its spec.
- **The basis trap.** A sheared lattice basis fits the homography too but gives an absurd fx.
  `fit_lattice.py` searches bases within `--hfov-range` (default 50–130°).

### 7.6 Pass / fail

For landmark fits (method B). Scene versions are judged by `check_version.py` (§13.1).

Measure at 1280 px width; the scripts normalize.

| Check | Limit |
|---|---|
| `fit_camera.py` landmark RMS | ≤ 2 px |
| `fit_camera.py` landmark max | ≤ 4 px |
| `overlay.py` shift | ≤ 6.4 px (0.5% of width, ≈1 px at 224) |
| `overlay.py` scale | ≤ 1% |
| `overlay.py` rotation | ≤ 0.3° |
| `overlay.py` chamfer | ≤ 4 px |

- Score regions must hold static structure only: surface edges, robot base, fixtures.
- Chamfer is weak on dense repetitive texture; ECC shift/scale/rotation is the primary signal.
- Apply the same limits to the wrist camera, within a fingers-plus-surface region at the recorded joints.
- Iterate the fit, the intrinsics and the robot or surface placement (§9) until all pass.
- If they cannot pass, report the residual and its likely cause: distortion, wrong intrinsics, wrong joint state,
  or wrong base offset.

## 8. Objects and physical properties

### 8.1 Mesh source by object kind

| Kind | Mesh source | Physics |
|---|---|---|
| Rigid, common category | Asset library first, if available: `assets/objects/` in an NVlabs/RoboLab upstream checkout (e.g. `$SCRATCH/r2s2r/.robolab-upstream`; sparse-fetch `assets/objects` as in §3 otherwise). `README.md` and `object_catalog.json` list dims, mass, friction; datasets ycb, hope, handal, hot3d, vomp, fruits_veggies, basic, objaverse. Rescale to measured dims. | RigidBodyAPI + MassAPI on the root (`make_asset.py`). |
| Rigid, simple shape (box, cylinder, roll, block) | Procedural mesh in numpy, then `make_asset.py` (pattern: example `make_twist_bread.py`). | Same. |
| Rigid, unusual | Image-to-3D from the supplementary close-ups (TRELLIS / Hunyuan3D / similar, if installed or the user can run it), or a phone scan (photogrammetry / LiDAR app) as .glb/.obj, then `make_asset.py`. Check scale and watertightness; fix texture colour with `color_match.py`. | Same. |
| Articulated (drawer, lid, door, hinged box, appliance) | `$S2R/articulated_usd.py` from primitive shapes and convex prisms, or a PartNet-Mobility-style USD with an articulation root. | See §8.4. An articulation, root `fixed` or `floating` (heavy base); the task registers it with `fixtures.articulated_fixture_cfg`. Joint limits from the real behaviour (ask the user); SI joint gains in the task. |
| Soft / compliant (foam toy, sponge, bag, fruit) | As rigid. | Start with a rigid body + **compliant contact**: `make_asset.py --compliant-stiffness K --compliant-damping D` writes `physxMaterial:compliantContactStiffness/Damping`. Pick K so the gripper squeezes the real amount: `K ≈ grip force / indentation`, e.g. 20 N / 5 mm = 4000 N/m. Full FEM soft bodies (Isaac Lab `DeformableObjectCfg`) are not supported by RoboLab's scene import or contact conditionals, cost much more, and are an open decision. |
| Transparent / reflective (glass, clear plastic, metal) | Library or generated mesh with a proper material: MDL glass/translucent (e.g. `SimPBR_Translucent.mdl` ships with a plate asset under `RoboLab/assets/objects/lab/plate/materials/`), or UsdPreviewSurface with `opacity`/`ior`, or high `metallic`, low `roughness`. | Normal rigid physics. **Risks:** RTX renders and the real camera both differ a lot by background and lighting; the policy may fail on them in both worlds. Compare renders against the policy camera specifically, keep the background accurate (reflections show it), and flag such objects in the report. |
| Thin / flat (paper, cloth, cards) | Thin box mesh. | Thin boxes need convex hull, not SDF. Grasping them often fails in sim; report it. |

### 8.2 Physical properties: how to set them

- **Mass.** Use a kitchen scale (ask). Otherwise estimate as `density × volume`. `make_asset.py` prints
  `mass / hull volume`; compare it with the material, remembering that hollow objects are far below solid
  density. Rough solid densities in kg/m³:

  | Material | Density |
  |---|---|
  | Foam | 20–100 |
  | Bread | 200–300 |
  | Wood | 500–800 |
  | Fruit | 900–1000 |
  | Plastics | 900–1400 |
  | Ceramic | 2000–2400 |
  | Glass | 2500 |
  | Aluminium | 2700 |

  Mass mostly changes grasp slip and lift dynamics. Heavy objects slip out of a binary-closing gripper.
- **Friction.**
  - Typical (static/dynamic): glazed ceramic 0.3–0.5; plastic 0.3–0.5; wood 0.4–0.6; rubber and foam 0.8–1.2;
    library defaults 0.8/1.0 or 2.0/2.0.
  - Too high hides real slip failures (sim too easy). Too low makes grasps fail that work on the real robot.
  - Dynamic friction must be ≤ static friction. Effective friction combines both bodies' materials
    (PhysX default combine is average).
- **Restitution.** Near 0 for almost everything on a table (0–0.1). Higher values make objects bounce off the
  gripper and fixtures.
- **Centre of mass.** Set `--com` only for off-balance objects (mugs with handles, tools, partly filled
  containers). It changes how the object rotates after a pinch grasp.
- **Collision geometry.**

  | Approximation | Use for | Effect on grasping |
  |---|---|---|
  | `convexHull` | Convex objects | Cheapest. Fills concavities, so fingers rest on the hull: grasps look easier, and nothing can sit inside a bowl or plate. |
  | `convexDecomposition` | Plates, bowls, mugs, handles, anything another object or a finger must enter | Contact closer to the real shape. Check that no thin piece is missing. |
  | `sdf` | Detailed concave dynamic meshes | Most accurate, most expensive. |
  | `boundingCube` / `boundingSphere` | Only for crude distractors | Crude. |

  Watch that collision hulls don't make placed objects float (on-top checks use contact plus the footprint).
- **Stability check.** Render after settling (20 steps): drift < 2 mm, no jitter, no sinking. Run a scripted
  pick at a nominal pose if possible.
- **Record** every value and its source in the report table. Every DR range (§14) is centred on these values.

### 8.3 Appearance

- Texture from the best photo: the policy-camera view for colour, supplementary close-ups for detail.
  Colour-match to the policy camera.
- Set roughness and metallic to match highlights in the policy image.
- Procedural meshes can use per-vertex colour (`make_asset.py` keeps glb vertex colours).

### 8.4 Articulated objects

**Build.** Write an `articulated_usd` spec (format in the module docstring) in the object frame: origin at the
footprint centre on the support surface, z up. Build it with
`PYTHONPATH=RoboLab python -m robolab.core.sim2real.articulated_usd spec.json`.

- **`"root"`:**
  - `"fixed"`: the first link is fixed to the world. Cabinets, wall units, anything bolted down.
  - `"floating"` with `"base_mass": 30`: the first link is a heavy dynamic body on its support. Use it for
    appliances and anything whose pose the plan randomizes. 30 kg does not move under the robot's pushes.
  - `"rigid_links"`: legacy (separate rigid bodies, plain joints; no runtime joint parameters). Only for old
    assets; reset them with `randomizers.place_rigid_fixture` and check them with `check_articulation.py --link`.
- **Links** are lists of shapes:
  - `cylinder`: a smooth visual mesh plus an implicit collision cylinder
  - `box`
  - `prism`: a convex polygon extruded in z, with convex-hull collision
  - `disc`: visual top only, colour or texture
  - `"visible": false` makes a collision-only proxy
- **Non-convex parts** (a lid notched around a control block) are built from several convex prisms.
- **Joints:** `revolute` or `prismatic`, with anchor and axis in the object frame and limits (degrees or
  metres). Positive motion is right-handed about the axis; check that it opens. Joint prims live under
  `<object>/joints/`, so a joint may share a link's name (`door`, `drawer`).
- **Grasp parts** (handle, lip, tab, knob): collision on, standing out from the body by about a finger's depth
  (1.5–2.5 cm), thick enough to pinch (≥ 1 cm). Ask the user for the real size.
- **Names:** every link and shape name must differ from every scene object's name (SKILL.md ground rules).
- **Place it in the scene** with `gen_scene.py` as `{"shape": "usd", "usd": ".../<name>.usda", "pos": [...], "yaw_deg": ...}`,
  so it lands at `{ENV_REGEX_NS}/scene/<name>`.

**Why an articulation.** The object is spawned inside the scene USD, which is cloned per env.
- An articulation root gives each env its own joint, its own pose writes (both fixed and floating roots move on
  GPU PhysX) and per-env joint damping, stiffness, friction and armature through Isaac Lab's
  `mdp.randomize_actuator_gains` / `mdp.randomize_joint_parameters` (plan section `joints`).
- The older designs fail in specific ways. A base with no rigid body (plain static colliders) makes PhysX treat
  the joint's body0 as the world, so every env's lid is pulled to env 0 (the first lab3 evals: 0.5% on the
  waffle task). A kinematic base cannot be moved on GPU PhysX (the reported pose changes, the lid stays). Plain
  joints between rigid bodies have no runtime joint API, so their damping cannot be randomized.

**Task integration** (`fixtures.py`):

```python
from isaaclab.managers import EventTermCfg as EventTerm
from robolab.core.sim2real import build_events
from robolab.core.sim2real.fixtures import articulated_fixture_cfg, fixture_centre, joint_position, reset_fixture

_WAFFLE = (0.42, 0.15, 0.047, -14.0)                     # x, y, z, yaw_deg in the env frame, from the fit
scene = import_scene("<scene>.usda", contact_object_list)
scene.waffle_maker = articulated_fixture_cfg("waffle_maker", _WAFFLE[:3], _WAFFLE[3],
                                             joints={"hinge": {"type": "revolute", "damping": 0.01}})
task_terms = {
    "init_pose": EventTerm(func=reset_pose_to_presets, mode="reset", params={...}),
    "reset_waffle_maker": EventTerm(func=reset_fixture, mode="reset",
                                    params={"name": "waffle_maker", "pose": _WAFFLE}),
}
events = lambda: build_events(task_terms, "<task>_scene/dr.json")

def lid_open(env):
    return joint_position(env, "waffle_maker", "hinge") > math.radians(45)
```

- `articulated_fixture_cfg(..., joints={regex: {...}})` puts all joints in one passive actuator group with
  per-joint SI gains; unset values take `JOINT_DEFAULTS` for the type (revolute damping 0.01 N·m·s/rad,
  prismatic 2.0 N·s/m). `joint_pos` sets the closed (default) position.
- `reset_fixture` writes the root pose (plus the plan's `pos_m` / `yaw_deg` jitter) and the joint positions
  (default, or uniform in the plan's `joint_pos`), with zero velocity, and stores the sampled pose in
  `randomizers.fixture_poses(env)[name]`.
- `build_events` moves fixture resets after the task's other reset terms: `reset_pose_to_presets(...,
  reset_to_default_otherwise=True)` otherwise puts the fixture back at its default pose.
- `fixture_centre(env, name)` is the fixture's root position per env. Use it for success regions and layout
  keep-outs that follow the fixture; the joint value is already relative to the base.
- **Success** for "open X and put Y in" combines:
  - joint value past a threshold
  - Y's centre inside the opened region, at the right height
  - Y nearly still
  - gripper away from Y
- The first use of `articulated_fixture_cfg` inside a scene USD (rather than spawned on its own, as in
  `RoboLab/tests/sim2real/smoke_sim.py`) should be checked with `check_articulation.py`: `import_scene` lists
  only top-level prims, so the fixture must not also appear as a `RigidObjectCfg`.

**Validate** with `check_articulation.py --fixture <name> --joint <joint>` (§15).

- **Lab2 waffle maker** (final: lid link = shell + control housing + dial; base = body, plate, hinge
  pedestal; hinge at the housing's rear bottom edge (0, 0.1295, 0.042), axis -x, 0–100°, lid 0.55 kg):
  - closed reads 0.0°
  - released at 60° or less → 0°
  - released at 75° or more → falls open to the 100° stop and stays
  - balance at about 71°, from the lid's centre of mass above the hinge
- **Lab2 hinge history** (what each wrong guess looked like in the check):
  - tape side (user's first description, misread): opened away from the robot; physically plausible, but the
    wrong side
  - top edge by the knob: the lid hung at 48° (centre of mass below the hinge)
  - bottom edge by the knob, block fixed: the lid's back swept into the block; collision proxies let it open
    but released angles stuck mid-way
  - rear of the knob housing, housing on the lid: matches the real one
- **Two failures it caught:**
  - With the dial block inside the lid's swept circle, the lid came to rest on the block at 24°. The fix
    was a notched lid of three prisms.
  - A hinge-side guess must come from the user.
- `"filtered_pairs"` (link / collider pairs that ignore each other) exists, but PhysX did not honour it against
  a static base in lab2; prefer reshaping the moving set or proxy colliders.

**Geometry from two views.** Pick rim points of the round part in both policy images and back-project them
onto a common unknown height. Least-squares fit the centre, radius and height (lab2: rim radius 9.0 cm,
residuals up to 1.5 cm). Then check the rendered outline against polygon labels; there the lid was first 13%
too large, and then 2 cm off until moved.

## 9. Scene geometry

- **`gen_scene.py` spec.** Any number of `surfaces`:
  - box or cylinder with metric UV textures (`period_m`) or one rectified photo (`stretch`), plus optional
    physics material and optional `drape` skin
  - `usd` fixtures from `RoboLab/assets/fixtures/` (e.g. `franka_table.usd` robot stand, tables, kitchens, bins)
    or your own
  - optional floor, lights, any number of objects
  There may be no table at all (shelf, floor, kitchen counter).
- **Placement.** Surface top height relative to the robot base mount comes from the user or from the camera fit.
  Example only: in bread_on_plate_lab the top is at z = 0 and the robot stand payload sits at x = -0.087 with
  yaw 180°.
- **Near field.** Anything within about 1.5 m that the cameras see with parallax must be geometry: walls next to
  the robot, shelves, the robot stand, monitors. The dome is only for far content.
- **Entity names.** Movable objects are scene entities named by their prim; use these names in presets,
  conditionals and DR. List supporting surfaces in the task's `contact_object_list` if conditionals or contact
  sensors use them.

## 10. Background

1. **Remove what won't exist in sim.** Mask per image: the robot, people, cables, other cameras, the near-field
   geometry you model separately, and any objects that will be randomized.
   - Prefer real pixels: if a supplementary photo sees behind an occluder, that photo contributes the pixels
     in `make_dome.py` (its weight), not the generator.
   - Generate only where no photo saw (`inpaint.py`).
   - Write a prompt that describes the actual environment (indoor lab, kitchen, office, outdoor yard: walls,
     windows, furniture, sky). The default negative prompt removes robots and people.
   - Outpaint (`--pad T R B L`) where a sim camera sees past the photo edge. The wrist camera sweeps widely.
2. **Dome.** `make_dome.py` with the fitted pose of every photo. The near-field rectangles are excluded.
   `check` must show about 0 holes for each policy camera at its calibrated pose. Fill holes by outpainting more,
   never by blurring.
3. **Hook it into the task** with the `randomize_dome_light_texture` event: `background_files=[...]`, and
   `intensity=` (uncommitted parameter, see §1). Tune the intensity until the background brightness gain from
   `overlay.py` is about 1.0. The example used 700.
4. **Limits.**
   - Infinite distance: no parallax, so the wrist and any moving camera see it slide wrongly.
   - No occlusion of near objects and no correct shadows from it.
   - One texture per process, shared by all envs.
   - Outdoor scenes: an HDR sky gives the right lighting, but a photo-LDR dome gives flat, weak light. Add a
     DistantLight for the sun (§11).

## 11. Lighting and image match

- **RoboLab defaults.** `auto_register_droid_envs` always adds `SphereLightCfg` (intensity 5000 at (0, -0.6, 0.7))
  plus the dome. Add the real dominant lights as scene lights (`gen_scene.py` `lights`):
  - RectLight for ceiling panels
  - DistantLight for sun through windows
  - SphereLight for bulbs
  Position them from shadow directions in the photos.
- **Exposure and white balance.**
  - Iterate light intensity and colour (and dome intensity) until `overlay.py`'s RGB gain over static
    regions is within about 5% of 1.0 per channel. Check both cameras.
  - Do not post-process sim images to fix this: the policy sees the raw render.
- **Shadows.** Compare shadow direction and softness under objects with the policy image. Large soft area lights
  vs hard point lights changes how object boundaries look to the policy.

### 11.1 Overriding RoboLab's built-in light

Every DROID registration adds `SphereLightCfg`: white, intensity 5000, at (0, -0.6, 0.7) env-local. It
usually dominates the work surface. Override it per task with a startup event (no change to shared files):

```python
def _set_sphere_light(env, env_ids, intensity, color):
    import omni.usd
    for prim in omni.usd.get_context().get_stage().Traverse():
        if prim.GetName() == "sphere" and prim.HasAttribute("inputs:intensity"):
            prim.GetAttribute("inputs:intensity").Set(float(intensity))
            prim.GetAttribute("inputs:color").Set(tuple(float(c) for c in color))

set_sphere_light = EventTerm(func=_set_sphere_light, mode="startup",
                             params={"intensity": 2700.0, "color": (1.0, 0.92, 0.82)})
```

Then shape the gradient with `gen_scene.py` lights (e.g. a RectLight behind the table if the far side is
brighter in the photo).

## 12. Task definition

### 12.1 Task file

Template (generalized from the example task file; names are placeholders):

```python
import math, os, random
from dataclasses import dataclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.constants import ASSET_DIR
from robolab.core.events.randomize_background import randomize_dome_light_texture
from robolab.core.events.reset_pose import reset_pose_to_presets
from robolab.core.scenes.utils import import_scene
from robolab.core.sim2real import build_events
from robolab.core.sim2real.layouts import preset_indices, sample_layouts
from robolab.core.task.conditionals import object_on_top, pick_and_place_on_surface
from robolab.core.task.task import Task

_SCENE = os.environ.get("MY_TASK_SCENE", "my_task.usda")
_BACKGROUND = os.path.join(ASSET_DIR, "my_task_scene", "background.png")
_OBJECTS = {                                       # photo pose (x, y, yaw_deg) from the version loop
    "obj_a": {"photo": (0.38, -0.14, -34), "x": (0.35, 0.70), "y": (-0.25, 0.10), "r": 0.08},
    "obj_b": {"photo": (0.69, 0.045, 0), "x": (0.40, 0.72), "y": (-0.25, 0.10), "r": 0.12},
    "mug": {"photo": (0.54, -0.08, -30), "jitter": 0.015, "r": 0.04},   # fixed distractor
}
_KEEPOUTS = [(0.45, 0.30, 0.15)]                   # (x, y, r) of each fixture's footprint
_PRESETS = sample_layouts(101, _OBJECTS, z=0.005, keepouts=_KEEPOUTS, min_dist={("obj_a", "obj_b"): 0.25})
_TRAIN, _TEST = range(1, 81), range(81, 101)       # preset 0 is the photo layout

def _events():
    terms = {
        "set_background": EventTerm(func=randomize_dome_light_texture, mode="reset",
                                    params={"background_files": [_BACKGROUND], "intensity": 700.0}),
        "init_pose": EventTerm(func=reset_pose_to_presets, mode="reset",
                               params={"presets": _PRESETS, "asset_cfg": list(_OBJECTS),
                                       "reset_to_default_otherwise": True,
                                       "preset_indices_pool": preset_indices(_TRAIN, _TEST)}),
    }
    return build_events(terms, "my_task_scene/dr.json")

@configclass
class MyTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=object_on_top, params={"object": "obj_a", "reference_object": "obj_b",
                       "gripper_name": "gripper", "require_gripper_detached": True})

@dataclass
class MyTaskTask(Task):
    contact_object_list = ["obj_a", "obj_b", "table"]
    scene = import_scene(_SCENE, contact_object_list)
    terminations = MyTerminations
    events = _events
    instruction = {"default": "<exact real instruction>", "vague": "...", "specific": "..."}
    episode_length_s: int = 50
    attributes = ["semantics"]
    subtasks = [pick_and_place_on_surface(object=["obj_a"], surface="obj_b", logical="all", score=1.0)]
```

- **Success conditions.** Other conditionals live in `RoboLab/robolab/core/task/conditionals.py`:
  `object_in_container`, `object_upright`, `object_inside`, `object_left_of`, `stacked`, `object_grabbed`, and more.
  Always require the gripper to be detached for placement tasks, so that holding the object over the target does
  not count.
- **Layouts.** `sample_layouts` objects take `x` / `y` ranges (sampled) or `jitter` (around the photo pose, only
  when DR is on), plus a footprint radius `r`; `keepouts` and `min_dist` keep them apart. `preset_indices` gives
  `[0]` with `ROBOLAB_PHOTO_LAYOUT=1`, the test pool with `ROBOLAB_EVAL_ONLY=1`, else the train pool.
- **Workspace bounds.** Use the real start-pose region: where the user actually places objects, intersected with
  where the real robot reaches. Keep train and test presets from the same distribution with disjoint indices.
- **Episode length.** `episode_length_s` and RLinf `max_episode_steps` (15 Hz) must be at least the real
  episode length.
- **Instruction strings.** Use the exact string the real policy will receive. Variants go in `instruction`.

### 12.2 RLinf configs and reward

- Copy the example env YAML and set `init_params.id`, `task_file`, `camera_preset`, `task_description`
  (= the real instruction), and `over_shoulder_cam` / `wrist_cam` sizes (§7.4).
- Copy the training YAML and point its `defaults` at the new env YAML.
- **Reward.** `IsaaclabBaseEnv._calc_step_reward` returns `reward_coef * terminations`, or the change from the
  previous step when `use_rel_reward`. RoboLab's success termination is true on every step after success, so
  absolute rewards pay once per step. Use `use_rel_reward: True` (+1 once) for GRPO.
  `filter_rewards` needs per-episode binarized success (uncommitted fix in `rlinf/utils/utils.py` and
  `fsdp_actor_worker.py` on the working tree).

## 13. Validation

- **Renders.** Use `render_views.py` with `--ext-fit` / the committed preset, `--object-poses` (the layout in the
  photo, env-local, scalar-first) and `--joint-pos` (the recorded joints). Run `overlay.py` per camera.
  Put real | sim | edges side by side in `for_review/`.
- **Interpreting the base eval.** Compare sim success p_sim (n = 800) with real p_real (n_real trials) using
  Wilson intervals; with 20 real trials the CI is about ±20 points.
  - **Sim far above real:** physics too forgiving (friction, collision hulls, mass), layouts easier than real,
    success condition looser than the real judge, visual too clean.
  - **Sim far below real:** a camera or pipeline mismatch (wrist orientation, FOV, letterbox), wrong
    instruction string, wrong object scale, background or lighting far off. The example's 79% → 35% from a wrist
    roll is the template.
- **Open-loop replay** (`replay_traj.py`). Take a real episode with `joint_position [T, 7]`,
  `gripper_position [T]` and, if recorded, commanded `action_joint_position`, at 15 Hz (DROID logs these; other
  stacks: export them). Set the initial object poses from its first frame.
  - Compare: joint tracking error (controller match; should be ≪ 0.05 rad), first grasp-contact step, whether
    the object lifts or slips, and the final object position (`--final-poses`).
  - Tune object friction, mass and collision first. Tune gains (`randomize_actuator_gains` range) only if
    tracking is off.

### 13.1 Version loop and check_version.py

`labels.json` (written once, on the real images):

```json
{"ext":   {"image": "ext.png", "robot_mask": "robot_mask.png",
           "objects": {"bread": [722, 390], "plate": [422, 378]},
           "static_polys": [[[0, 370], [470, 235], [520, 255], [520, 300], [0, 490]]],
           "color_regions": [[650, 450, 950, 560], [100, 420, 300, 500], [600, 600, 900, 700]]},
 "wrist": {"image": "wrist.png", "objects": {"bread": [480, 300]},
           "color_regions": [[50, 20, 350, 180]], "pattern": [0, 0, 1280, 420]}}
```

- **Object centroids:** the centre of the visible body, by eye.
- **Static polygons:** rigid edges only (table edges, fixture outlines). Exclude cloth folds and drapes,
  cables, anything movable, and the dome-only background.
- **Colour regions:** plain work-surface patches, near and far, away from objects and shadows.
- **Pattern box:** a patch of the patterned surface above the gripper fingers.

Per version:

```bash
ROBOLAB_PHOTO_LAYOUT=1 python $SK/render_check.py --task-file <task>.py --task <Task> --preset <PRESET> \
    --joint-pos <recorded> --entities robot <objects...> --out $WD/vN
python $SK/check_version.py $WD/labels.json $WD/vN --name vN --log $WD/versions.md
```

- **Masks.** `render_check.py` takes them from instance-id segmentation, so each object's mask is exact.
  It lets physics settle first (`--settle 90` steps), so objects spawned slightly above the surface stop
  moving.
- **Pattern check.** `check_version.py` compares the pattern period along the real image's dominant spectral
  direction, zero-padded for sub-bin accuracy. The wrist check catches table-distance and wrist-FOV errors
  the external camera cannot see.
- **Fix order** follows the dependency chain:
  1. scale / camera (`robot_iou`, `pattern_scale`)
  2. surface geometry (`static_chamfer`)
  3. object placement (`obj_px`; move an object by back-projecting the sim and real centroids with
     `backproject.py point` at its mid height and adding the difference)
  4. lighting (`color_gain`)

## 14. Sim-to-real support plan (domain randomization)

### 14.0 Plan template (`$WD/sim2real_plan.md`)

```markdown
# Sim-to-real support plan: <task>
Deployment condition: <objects present, fixed or moving camera, lighting, who resets the scene>

| # | Category | Factor | Reconstructed value | Source | σ | Range | Mechanism | Code | Tier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Viewpoint | ext camera position | (x, y, z) | fitted (lattice + robot IoU) | 1 cm | ±2 cm | per-env camera reset event | new | 1 |
| 2 | Viewpoint | ext camera FOV | 92.4° | fitted (lattice) | 1.5% | ±3% | same event (intrinsics) | new | 1 |
| 3 | Image | wrist brightness | 1.0 | colour_gain residual | 0.12 | 0.75–1.0 | obs augmentation | new | 1 |
| 4 | Object physics | <obj> friction | 1.2 | assumed | — | 0.8–1.4 | randomize_rigid_body_material | exists | 1 |
| 5 | Scene geometry | surface height | 0.049 m | fitted (scale scan) | 1 cm | ±2 cm | per-env offset | new, per-env physics | 3 |
| … | Distractors | — | none | deployment condition | — | not randomized: test is objects-only | — | — | — |
```

Columns:
- **Source:** measured / user / fitted / assumed.
- **σ:** from the fit (§14.1) or "—" when assumed.
- **Range:** derived per Phase 9.3.
- **Code:** exists / new / needs per-env physics.

Every category in SKILL.md Phase 9.2 must appear, even if only as "not randomized" with a reason.

### 14.0.1 Mechanisms

| Factor | Supported now | Params / notes |
|---|---|---|
| Object layout, train/test split | `reset_pose_to_presets` (presets + `preset_indices_pool`), `reset_pose_uniform` (`pose_range`, collision check) — `robolab/core/events/reset_pose.py` | Bounds from the real reachable workspace. |
| Fixture / articulated-object pose and opening | Plan `fixtures` → `fixtures.reset_fixture` (per env, every reset): `pos_m`, `yaw_deg`, `joint_pos` | Root `fixed` or `floating`. `build_events` orders it after the preset event. Spawn regions must clear the jittered footprint. |
| Fixture joint damping, stiffness, friction, armature | Plan `joints` → `mdp.randomize_actuator_gains` / `mdp.randomize_joint_parameters` (per env, startup) | SI units, `operation` `abs` by default. Needs an articulation (not `rigid_links`). |
| Background texture and brightness | Plan `background` + `make_dome_variants.py` | Photometric variants of the reconstructed dome (variant 0 = original) rather than unrelated HDRIs, unless broad robustness is the goal; intensity range too. The dome is stage-wide: one draw per reset call (per GPU per rollout epoch in RLinf). Replaces the task's `set_background` term. |
| Camera pose jitter | Plan `cameras` (per env, every reset) | Samples around the fitted pose by writing each camera prim's local pose, so it works for fixed and mounted (wrist) cameras. RoboLab's `reset_camera_pose_uniform` adds a delta to the *current* pose and random-walks; do not use it. |
| Camera intrinsics | Plan `image` `zoom` | The tiled renderer uses one projection for all envs (`Camera.set_intrinsic_matrices` warns), so per-env FOV is a centre crop-and-resize. |
| Broader viewpoint prior | **None** | Sample a share of envs (10–20%) from the policy's training-camera distribution, e.g. DROID camera-to-base extrinsics (community re-calibration on Hugging Face `KarlP/droid`; check the file and its counts first). Same per-env camera event. Plot the real camera inside that distribution first. |
| Per-camera image formation (gain, white balance, contrast, gamma, noise, blur) | Plan `image` (per env, every reset) | Installed at startup as a modifier on the observation term, so it applies in RoboLab evals and in RLinf alike. Per camera, because cameras auto-expose differently. |
| Action latency | **None** | Delay the action buffer by 0–1 control steps in the env wrapper if the real stack has a known lag. |
| Friction, restitution | Plan `material` → `mdp.randomize_rigid_body_material` (mode `startup`; `static_friction_range`, `dynamic_friction_range`, `restitution_range`, `num_buckets`, `asset_cfg`, `make_consistent=True`) | PhysX samples buckets once at startup. |
| Mass | Plan `mass` → `mdp.randomize_rigid_body_mass` (`range`, `operation` `abs` by default) | |
| Centre of mass | Plan `com` → `mdp.randomize_rigid_body_com` (± metres on each axis) | |
| Controller gains | Plan `actuator_gains` → `mdp.randomize_actuator_gains` (scale) | Only if replay shows tracking differences. |
| Object scale | `mdp.randomize_rigid_body_scale` (mode `usd`) | Needs `scene.replicate_physics=False`, which RoboLab does not set (Isaac default True). Needs a change. |
| Object / surface texture or colour | `mdp.randomize_visual_texture_material`, `mdp.randomize_visual_color` (Replicator) | Also need `replicate_physics=False`. |
| Light intensity, colour, position | Plan `lights` (every reset, around the task's tuned values) | Scene lights and RoboLab's sphere light are cloned per env, so they are sampled per env. `pos_m` moves the light (shadow direction). |
| Distractor objects | Partly: put extra objects in the scene USD and place them via presets or `reset_pose_uniform`; park unused ones off-table | A pool/visibility mechanism is new code. Keep distractors similar to real clutter. |
| Per-env backgrounds | **None**: the dome is global | Needs a design change. |
| JPEG compression | **None** | Register an `extras` section if the real stack compresses. |
| Anything else | Plan `custom` (`"module:function"`) or a section registered from an `extras` module | Keep scene-specific code under `$S2R/extras/`. |

Example `RoboLab/assets/<task>_scene/dr.json` (ranges centred on the reconstructed values):

```json
{"cameras": {"over_shoulder_left_camera": {"pos_m": 0.02, "rot_deg": 1.5}, "wrist_cam": {"pos_m": 0.005, "rot_deg": 1.5}},
 "image": {"over_shoulder_left_camera": {"gain": [0.85, 1.12], "wb": 0.05, "contrast": [0.9, 1.1], "zoom": [0.97, 1.03]},
           "wrist_cam": {"gain": [0.68, 0.98], "wb": 0.05, "noise": [0, 4], "blur_p": 0.2}},
 "lights": {"sphere": {"intensity": [0.75, 1.25], "color": 0.05, "pos_m": 0.3}},
 "background": {"dir": "__ASSET_DIR__/<task>_scene/dome_variants", "intensity": [600, 900]},
 "mass": {"obj_a": {"range": [0.03, 0.06]}},
 "material": {"obj_a": {"static": [0.8, 1.4], "dynamic": [0.6, 1.2], "restitution": [0, 0.05]}},
 "com": {"obj_a": 0.01},
 "actuator_gains": {"robot": {"joints": "panda_joint.*", "stiffness": [0.9, 1.1], "damping": [0.9, 1.1]}},
 "joints": {"waffle_maker": {"hinge": {"damping": [0.005, 0.03], "friction": [0, 0.02]}}},
 "fixtures": {"waffle_maker": {"pos_m": 0.015, "yaw_deg": 5}}}
```

The task's own terms run first, then fixture resets, then the plan's sections. For an ablation, point
`ROBOLAB_DR_OVERRIDE` at a file such as `{"lights": {"enabled": false}}`.

### 14.1 Ranges from the reconstruction's own uncertainty

When measurements are missing, the fit tells you how wide to randomize:

| Quantity | σ source | Example (lab2) |
|---|---|---|
| Global scale (period) | Width of the robot-IoU peak in the scale scan | ±4% |
| Surface height | The scale σ propagated along the wrist rays | ±2 cm |
| External camera pose | Residual of the version loop (centroid error) plus the scale σ | ±1.5 cm, ±1° |
| Object size | The scale σ, plus how sharp each object's edges were | ±5% |
| Brightness and white balance, per camera | `color_gain` residual | wrist 0.75–1.0, ext ±5% |
| Surface yaw / table extents | Edge residual | ±2°, ±2 cm |

- Camera pose jitter stays around the calibrated pose (`randomizers.perturb_camera_poses` stores the nominal
  pose at the first reset).
- Table height and object scale need `replicate_physics=False` or per-env USD variants (§16).

## 15. Helper scripts

| Script | Runs where | Purpose |
|---|---|---|
| `cam_math.py` | login | Quaternion and frame conversions, lookat, intrinsics ↔ PinholeCameraCfg, projection, camera class snippet |
| `fit_camera.py` | login | PnP (any known points, robot keypoints, calibration K + distortion), optional VP mode; RMS + overlay |
| `refine_camera.py` | GPU, container | Render-and-compare refinement on the robot for ext or wrist; dumps robot keypoints; wrist offset on `base_link` |
| `rectify.py` | login | Metric top-down texture of any planar surface, plus a tile |
| `color_match.py` | login | Colour-match supplementary textures to the policy camera |
| `inpaint.py` | GPU | SDXL inpaint/outpaint with full-resolution composite (no blur) |
| `make_dome.py` | login | Equirect dome from posed photos; near-field exclusion; hole check per camera |
| `make_asset.py` | login | Mesh (USD/glb/obj/...) → RoboLab rigid object: scale, orientation, mass, CoM, friction, restitution, collision approximation, compliant contact |
| `gen_scene.py` | login | Scene USD from a spec: arbitrary surfaces/fixtures, floor, lights, objects |
| `render_views.py` | GPU, container | Renders both cameras at a joint state and object layout; runtime ext pose and wrist offset overrides |
| `overlay.py` | login | Sim vs real alignment score (ECC shift/scale/rotation, chamfer) + RGB gain; PASS/FAIL exit code |
| `replay_traj.py` | GPU, container | Open-loop replay of real joint trajectories; logs tracking, contacts, object poses |
| `fit_lattice.py` | login | Periodic-pattern detection, lattice homography, single-view calibration (fx, k1, plane pose in pattern units) |
| `plane_from_camera.py` | login | Plane height, pattern yaw and phase in the robot frame from a known-pose camera; back-projects anchors |
| `compose_ext.py` | login | External-camera candidates (one per scale) from its lattice fit + plane + anchor; picks the yaw symmetry |
| `backproject.py` | login | Pixel → plane point; object base / centre / height / width from a fitted camera |
| `render_check.py` | GPU, container | Renders a version (both cameras, photographed layout) with per-entity segmentation masks; scans camera candidates by robot IoU |
| `check_articulation.py` | GPU, container | Releases a fixture's joint (`--fixture --joint`) or a legacy moving link (`--link`) at several values, records settling, renders closed/open from both cameras |
| `make_dome_variants.py` | login | Photometric variants of a dome background (exposure, white balance, contrast, saturation, directional field, blobs, yaw roll), contact sheet |
| `check_dr.py` | GPU, container | Grid of randomized envs per camera (real image first), two resets; real image's percentile in sim brightness / colour stats; flags dark, saturated or flat frames |
| `check_version.py` | login | Scores a version against labels: robot IoU, object centroids, static chamfer, colour gain, pattern period; sheet + versions log; PASS/FAIL |

In the repo, not the skill (`$S2R`, tests in `RoboLab/tests/sim2real/`):

| Module | Purpose |
|---|---|
| `articulated_usd.py` | Articulated object USD from a spec; root `fixed` / `floating` / `rigid_links`; CLI `python -m robolab.core.sim2real.articulated_usd spec.json` |
| `fixtures.py` | `articulated_fixture_cfg`, `reset_fixture`, `joint_position`, `fixture_centre` |
| `layouts.py` | `sample_layouts`, `preset_indices` |
| `plan.py` | `load_plan`, `build_events`, `dr_enabled`; `ROBOLAB_DR`, `ROBOLAB_DR_OVERRIDE` |
| `builders.py`, `registry.py` | Plan sections; `@register` |
| `randomizers.py` | Camera, image, light, background events; `place_rigid_fixture` for `rigid_links` objects |

**Testing status.**

- Tested on the login node with synthetic data: `fit_camera.py` (PnP recovers a known pose to < 1 cm;
  VP auto-axes), `overlay.py` (detects an 8 px shift), `rectify.py`, `color_match.py`, `make_dome.py`
  (round-trip), `make_asset.py` (glb with vertex colours, library USD, procedural USD), `gen_scene.py`.
- Run on real data (lab2 example, run2): the randomizers (then in `sim2real_dr.py`), `check_dr.py`,
  `check_articulation.py --link`, `articulated_usd` (then `make_articulated.py`, `rigid_links`), `fit_lattice.py`, `plane_from_camera.py`, `compose_ext.py`, `backproject.py`,
  `render_check.py` (segmentation masks and scan), `check_version.py`, `render_views.py`, `make_dome.py`,
  `gen_scene.py`, `make_asset.py`, `refine_camera.py`. `refine_camera.py` ran, but it barely moved the pose (see §16).
- `$S2R`: unit tests (`pytest RoboLab/tests/sim2real/test_plan.py`) and a 4-env simulator test
  (`RoboLab/tests/sim2real/smoke_sim.py`: a fixed cabinet with door and drawer, a floating waffle maker; per-env
  pose, opening and damping) pass. Not yet run inside a real task: `articulated_fixture_cfg` in a scene USD,
  `check_articulation.py --fixture`, `build_events` end to end.
- Not yet run: `replay_traj.py`. `inpaint.py` ran in run 3.
  Check their first runs closely.
  - In `refine_camera.py`, check that toggling robot mesh visibility changes the render. If the sim robot mask
    comes out empty, the USD visibility toggle is not reaching the renderer. Switch to an
    `instance_id_segmentation_fast` camera data type instead.

## 16. Known gaps and open decisions

- **Camera sets.** `RoboLabDroidEnv._wrap_obs` hard-wires one external camera (`over_shoulder_left_camera`) plus
  `wrist_cam`. A policy with other cameras needs a wrapper change.
- **Wrist offset.** A per-task wrist offset needs a change in `droid.py` or `robolab_task.py`.
- **Letterboxing.** The 224x224 square render vs real letterbox is unresolved in the example. Fix it via the YAML
  render size (§7.4) and verify.
- **Gripper.** The sim gripper is binary (open/close to π/4 on `finger_joint`), and its force comes from the USD
  drive. A real Robotiq driven continuously, or with different force or speed settings, behaves differently on
  soft or heavy objects.
- **Object types.** An articulated object the robot carries (a light floating root) is untested. True
  deformables, texture / scale and table-height randomization, per-env backgrounds and the DROID viewpoint prior
  still need new code (§8.1, §14).
- **Dome.** It cannot give parallax or occlusion for near content. Model near content as geometry.
- **`refine_camera.py` stalls.** Its Nelder-Mead makes 1 cm / 1° steps on a cost that the renderer's temporal
  noise changes by ±20% between identical poses. In lab2 it moved the camera 1 cm in 200 evaluations. Use
  `render_check.py --cands` scans instead. Making it work would take averaging several frames per evaluation
  plus a coarse-to-fine grid.
- **Wrist exposure.** Real wrist cameras auto-expose. Sim renders them with the scene's fixed exposure, about
  25% brighter in lab2. Randomize brightness rather than darkening the scene for one camera.
- **Deformables in the photo.** Draped cloth folds, cables and the like cannot match; leave them out of the
  static polygons and say so.
- **Render-and-compare.** It depends on accurate real joint states. Joint-encoder error or a mis-recorded joint
  state biases the camera pose. Ask for joint states logged at the image timestamp.
