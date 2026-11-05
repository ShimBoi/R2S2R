---
name: real2sim-robolab
description: "Reconstructs any real robot scene as a RoboLab (Isaac Lab) sim task from a few real photos, for real-to-sim-to-real transfer of VLA policies (pi0.5 / OpenVLA-OFT) on a DROID-style Franka with a Robotiq gripper. Use when the user supplies labelled images from the cameras the policy observes (e.g. an external/'left exterior' camera and a wrist camera), optionally plus extra unlabelled photos of the scene, and wants a sim scene/task recreated from them. The sim must have pixel-matched cameras, objects with measured physical properties, a background built from the photos, lighting, the real task's success condition, and domain randomization, all validated against the real images and real robot before RL. Also use to validate or extend an existing reconstruction."
---

# Real-to-sim scene reconstruction in RoboLab

Turn real photos of a robot setup into a RoboLab task that a VLA policy can be evaluated and fine-tuned in,
then transferred back to the real robot. The sim must match three things:

- what the policy sees: every policy camera, pixel for pixel
- what the gripper touches: geometry, mass, friction, compliance
- what the real task counts as success

Nothing in the procedure depends on a particular scene, object category, surface, pattern or camera model.
Formulas, snippets, API references, object-kind guidance and labelled worked examples are in
[reference.md](reference.md). The helper scripts here are generic. Files cited from the worked examples may live
on other branches or outside the repo; the numbers and lessons quoted are enough, so skip any path you cannot read.

The reusable sim code is the package `RoboLab/robolab/core/sim2real/` (already in the repo; do not copy it):

| Module | Use it for |
|---|---|
| `articulated_usd` | Lids, doors, drawers, appliances: USD from a JSON spec, as an articulation (Phase 3.5) |
| `fixtures` | `articulated_fixture_cfg` (scene entry, SI joint gains), `reset_fixture` (per-env pose and opening), `joint_position`, `fixture_centre` |
| `layouts` | Start presets: preset 0 = photographed layout, sampled train / test presets with keep-outs (Phase 7) |
| `plan` | `build_events(task_terms, "<task>_scene/dr.json")`: the task's events plus everything in the plan (Phase 10) |
| `builders`, `registry`, `randomizers` | The plan's sections, and `@register` for new ones |

Paths are relative to the repo root (the R2S2R checkout or worktree you work in). `SK=.claude/skills/real2sim-robolab`.
`WD=$SCRATCH/r2s2r/<task>_scene/` is the work dir for inputs, fits, renders, logs and the report.

## Input contract

1. **Required: labelled policy-camera images.** One image per camera the policy observes, each labelled with
   its role (e.g. `exterior_left`, `wrist`). Support whatever cameras the policy config uses.
   - These are the **ground truth**: the sim cameras must reproduce them exactly (pose, intrinsics,
     resolution, crop/resize), and every alignment check is against them.
   - Ideally they were taken at the robot's start pose with the **joint state recorded** (7 joints + gripper),
     so the robot itself can serve as the calibration target.
   - If no joint state is given, ask for it, or for the policy's home pose and whether the photo was taken there.
   - If a label is missing or doubtful, infer it and ask to confirm. A wrist view is a close-up with the gripper
     fingers at a fixed image location; an external view shows the robot base and the workspace.
2. **Optional: N unlabelled supplementary images** (phone photos from any viewpoint, close-ups). Use them for:
   - geometry the policy cameras cannot see
   - clean texture for surfaces and background (less generative fill)
   - object identity, size and material
   - views behind and around the robot

   They never override the policy-camera images for appearance as seen by the policy (colour, exposure,
   lighting). Note their white-balance and exposure differences, and colour-match anything taken from them
   to the policy cameras. See reference.md §5.
3. **Then ask for what is still missing**, after looking at all images, in one short checklist:
   - object sizes and weights
   - softness / compliance and moving parts
   - camera model / intrinsics / calibration
   - robot base-to-work-surface offset
   - joint states
   - exact instruction string
   - success condition
   - real success rate and any recorded trajectories

   Full list: reference.md §4.

## Ground rules

- **New files only, unless the user agrees.** One existing-file change is always needed: the camera class in
  `RoboLab/robolab/variations/camera.py` and its preset in `RoboLab/robolab/registrations/droid/camera_presets.py`,
  because the render, check and training scripts select cameras by preset name. Keep it to appended additions,
  save it as a patch in `$WD`, and say so. Anything else that touches existing files (wrist offset, the
  `sim2real` package, shared events) needs the user's yes first. Never commit or push unless asked.
- **Names are unique across the whole scene,** including shapes and links inside fixtures. Contact sensors match
  prims by name, so a waffle-maker part called `plate` next to a `plate` object breaks env startup.
- **GPU work only on a node the user names.** Never run on a node hosting a live job (`squeue -u $USER`).
  Isaac code runs through `setup/horizon/enter_container.sh` (reference.md §3). CPU helpers run on the login node.
- **Record every value.** Each number goes in the report tables with its source (measured / user /
  photo-estimated / assumed) and its uncertainty. Assumed values are flagged.
- **Never blur what the policy sees.** Use real pixels first, then generative fill. No smoothing.
- **Work in a git worktree** on its own branch, so scene experiments never touch a tree that training runs
  from. Setup has two traps (reference.md §3):
  - git-lfs may be missing, so bypass its filters.
  - Ignored assets (robots, fixtures, materials, backgrounds) must be copied in.
- **Iterate in numbered versions until `check_version.py` passes** (Phase 8). Every version is one change
  set, a `render_check.py` render, a `check_version.py` score, a line in `versions.md`, and a sheet sent to
  the user.
- **Uncertainty is expected.** Do not block on measurements the user does not have. Fit what the images
  support, record the uncertainty of each fitted value, and turn it into domain-randomization ranges (Phase 9).

## Phase 0: Intake and survey

1. Record each image's role, native resolution and source (raw sensor frame, phone, screenshot), plus EXIF focal
   length for phone photos.
2. Survey all images. List:
   - surfaces and fixtures
   - objects: kind (rigid / articulated / soft / transparent-reflective / thin), task-relevant or clutter
   - lights and windows
   - everything that will not exist in sim (people, cables, other robots or cameras)
3. Send the Input-contract checklist for what is missing. Proceed meanwhile with assumed values, marked as such.

## Phase 1: Metric frame, scale, robot placement

1. Use the robot base frame (+x forward, +y left, z up, origin at the base) as the env origin. Express
   everything in it.
2. Scale comes from any known reference: user measurements, robot link lengths and gripper stroke, standard
   objects, camera bodies of known size, counted pattern periods (reference.md §6). Cross-check two.
3. Place the work surface(s) and fixtures relative to the base, including the surface **height** relative to the
   base mount. A few centimetres off shifts every grasp even when images look right.

**Check:** Phase 2's reprojection of surface corners and the robot base lands within the Phase 2 limits.

## Phase 2: Policy cameras (exact match)

1. **Intrinsics.** In order of preference:
   - factory / robot-stack calibration (any model, any resolution, distortion if the policy sees unrectified frames)
   - datasheet FOV
   - estimation (`--fit-focal`)

   Write them with `cam_math.py cfg`. Support whatever resolution and FOV the real stream has.
2. **External camera pose.** Pick the first method the inputs support (reference.md §7.2, §7.5):
   - **A. Pattern plane + robot (no measurements needed; the default when the work surface has a periodic
     pattern such as checked cloth, tiles or a grid mat).**
     1. `fit_lattice.py` on the external image gives fx, tilt and height in pattern units.
     2. `fit_lattice.py --fix-fx` on the wrist image.
     3. `plane_from_camera.py` with the wrist camera pose at the recorded joints (dumped from sim) gives the
        plane height and pattern yaw in the robot frame, plus anchor points (an object both cameras see).
     4. `compose_ext.py --anchor ... --check-uv <robot base pixel> --scales ...` gives one candidate camera per
        pattern-period scale.
     5. `render_check.py --cands --real-robot-mask` scores each candidate by robot-silhouette IoU. Take the peak.
        The width of the IoU curve is the scale uncertainty.
   - **B. PnP on measured points:** ≥ 6 points with known robot-frame positions (measured corners, tape marks,
     robot body origins from `refine_camera.py --dump-keypoints`):
     `fit_camera.py pnp points.json $WD/fit_ext --keypoints kp.json`. Use `vp` mode only for a line grid.
   - **C. Render-and-compare refinement** (`refine_camera.py`) to polish A or B. Its Nelder-Mead stalls on
     render noise, so prefer discrete scans (`render_check.py --cands`) for anything larger than about 1 cm.
3. **Fix the scale before anything else is placed.** Scale, table height, camera distance and every object
   size are coupled. A 10% scale error moves the table about 3 cm. Resolve it with the robot silhouette (A.5)
   or a ruler measurement, then derive every other position from the chosen scale.
4. **Wrist camera.**
   - Start from the gripper-mount extrinsic (sim default `_WRIST_CAM` in `RoboLab/robolab/robots/droid.py`, or
     the user's CAD / hand-eye numbers).
   - Refine with `refine_camera.py --camera wrist` on the real wrist image at its recorded joints. The fingers
     fix the mount. Never change the mount from memory: a wrong 180° roll once cut zero-shot success from 79% to 35%.
5. **Image pipeline.** Reproduce the real resolution and aspect. Let the policy's own preprocessing run on the
   sim frames:
   - pi0.5 letterboxes (`resize_with_pad`).
   - OpenVLA-OFT resizes and may center-crop.
   - RLinf's 224x224 square render is a known mismatch.

   See reference.md §7.4.
6. **Validate** each camera in the Phase 8 version loop (`render_check.py` + `check_version.py`).
   - A whole-image `overlay.py` score is meaningless once the dome is built from the same photo, because the
     background then aligns trivially. Score only rigid scene structure, the robot and the objects.
   - Landmark fits (method B) must also reach RMS ≤ 2 px, max ≤ 4 px.
   - If a check cannot pass, report the residual and its likely cause.

**Measuring object positions from a fitted camera:** `backproject.py` maps any pixel to the plane
(`point`), or gives an object's base position, height and width from base / top / left / right pixels
(`object`). Use the base front edge plus radius along the view ray for round objects.

**When the two cameras disagree** on an object's position, the error is almost always along one camera's line of
sight (depth is the weak direction of a single view). Take each coordinate from the camera that sees it
across its rays: typically the wrist camera (looking down) for position on the table, the external camera for
height. Record the disagreement as that object's position σ.

## Phase 3: Objects and physical properties

Assume no object category. For each object, follow reference.md §8:

1. **Identify** its kind: rigid, articulated (drawer, lid), soft / compliant, transparent / reflective, or thin.
   Decide whether it is task-relevant or clutter.
2. **Measure** dimensions from the user, or from photos against the Phase 1 scale. Use supplementary views for
   the sides you cannot see.
3. **Mesh source**, in order of preference:
   - asset library lookup (the RoboLab upstream object catalog, if it is available; reference.md §8.1)
   - procedural mesh for simple shapes
   - image-to-3D from the close-ups, or a scan

   Scale, orient and texture it. Colour-match to the policy cameras.
   Build it with `make_asset.py`.
4. **Physics**, set deliberately:
   - mass / density, static and dynamic friction, restitution, centre of mass
   - collision approximation (hull vs decomposition vs SDF, and its effect on grasps)
   - compliance (compliant contact for soft objects)
   - rendering material for transparent and reflective objects

   Note how each choice moves grasp success.
5. **Articulated objects** (lids, doors, drawers, appliances; reference.md §8.4).
   - **Ask the user how it moves:** where the hinge or slide is, which way it opens, how far, whether it
     stays open, **which parts move with it** (a housing, knob or handle on top often belongs to the lid), and
     **where and how the robot grabs it** (a handle, a lip, a tab: its size and how far it stands out).
     Photos rarely show this; a wrong hinge side or moving set makes the task physically different. Lab2
     needed two corrections from the user: the tape lip is where you lift (not the hinge), and the whole top,
     knob housing included, swings on a rear hinge.
   - **Build it with `robolab.core.sim2real.articulated_usd`** from primitive shapes:
     `python -m robolab.core.sim2real.articulated_usd spec.json` (spec format in the module docstring).
     - `"root": "fixed"` for objects that never move (cabinets, wall units).
     - `"root": "floating"` with `"base_mass": 30` for objects whose pose is randomized (appliances on a
       counter). A heavy base sits still under the robot's pushes but can be moved per env at reset.
     - Both are articulations, so each env has its own joint, its own pose and its own joint damping,
       stiffness and friction, all changeable at runtime.
   - **The grasp part must be graspable.** A handle, lip or tab the robot pulls needs collision on and must stand
     out from the body by at least a finger's depth (about 1.5–2.5 cm; ask for the real size). A visual-only
     tape strip or a 3 mm overhang leaves the policy nothing to hold, and it will push at the lid without lifting it.
   - **Put revolute hinges at the seam, not on top.** A hinge above the moving part's centre of mass makes it
     hang open like a pendulum (the lab2 test hung at 48°).
   - **Notch moving parts around fixed ones.** Anything that overlaps the moving part's swept volume (a control
     block, a handle) gives it a notch; otherwise the part rests on it instead of closing.
   - **Joint gains are SI** and are set in the task, not in the spec: revolute damping in N·m·s/rad, prismatic
     in N·s/m. Defaults are 0.01 (revolute) and 2.0 (prismatic); a free-swinging lid wants about 0.005–0.05. (USD
     drive damping, used only by legacy `rigid_links` objects, is per degree: 0.3 there is about 17 N·m·s/rad,
     and a lid released at 30° crept to 16° in 4 s.)
   - **Integrate it in the task** (reference.md §8.4):
     - `scene.<name> = fixtures.articulated_fixture_cfg(name, pos, yaw_deg, joints={...})`
     - a `fixtures.reset_fixture` reset term that returns it to its pose, closed
     - `fixtures.joint_position(env, name, joint)` in success conditions
     - `fixtures.fixture_centre(env, name)` for anything placed relative to it
   - **Validate with `check_articulation.py --fixture <name> --joint <joint>`:** `env_spread` about 0 over several
     envs, about 0 when closed, closes fully from below the balance angle, and reaches and holds its limit (or
     stays put, for damped joints) like the real one.
6. **Fit the size and position of round or symmetric objects from two views.** Back-project rim or edge
   points from both calibrated cameras onto a common height, and least-squares fit the centre, radius and
   height. Then confirm against polygon labels in the version loop.
7. **Check:**
   - size within 5% of the measurement
   - `mass / hull volume` plausible
   - settles with < 2 mm drift and no jitter. Round objects (rolls, fruit, cans on their side) roll while
     settling; give them a slightly flat bottom or the resting orientation they have in the photo, and check
     their position after the settle steps, not at spawn.
   - looks right in both policy cameras

## Phase 4: Scene geometry and textures

1. For each planar surface, pick the most frontal sharp view (policy camera or supplementary). Rectify a metric
   texture or tile with `rectify.py`, then `color_match.py` it to the policy camera. Real pixels only.
2. Write a `gen_scene.py` spec:
   - any number of surfaces (box / cylinder / USD fixture; textured or coloured; physics material; optional
     drape skin)
   - optional floor, lights, all objects

   A table is optional. Generate `RoboLab/assets/scenes/<task>.usda`.
   - Set `"floor": null` when the background dome comes from a policy-camera photo. A floor plane hides
     every dome pixel below the horizon (counters, cabinets and floor seen past the table).
   - Rotate surfaces by their fitted yaw (`yaw_deg`). Tables are rarely square to the robot, and the
     pattern yaw from `plane_from_camera.py` tells you the true angle.
3. Model as geometry everything within about 1.5 m that a policy camera sees with parallax: walls, shelves,
   stands, monitors.

## Phase 5: Background

1. Mask everything that will not exist in sim in each photo: robot, people, cables, other devices, modelled
   near-field geometry, randomized objects.
2. Fill from other photos first (posed supplementary views in `make_dome.py`). Use `inpaint.py` with a prompt
   describing the actual environment only where no photo saw. Outpaint with `--pad` where sim cameras see past
   the photos.
3. `make_dome.py` with every posed photo. `check` must show about 0 holes for each policy camera. Outpaint more
   instead of blurring.
   - A wrist camera at the start pose often sees only near field (table, floor, a cabinet front), which the dome
     cannot show correctly anyway. Model that as geometry (a floor or cabinet box, a strip past the table edge,
     textured from the photo) and run the hole check only on the pixels that see past it.
   - If the supplementary photos cannot be posed (fewer than 4 known points each), use the policy-camera photos
     alone and say how much of the sphere the dome covers.
4. Tune the dome intensity by the `overlay.py` RGB gain. State the limits: no parallax, no occlusion or shadows
   from the dome, one texture per process.

## Phase 6: Lighting and image match

1. Add the dominant real lights (ceiling panels, windows / sun, lamps) as scene lights, placed from shadow
   directions.
   - RoboLab always adds its own white SphereLight (intensity 5000 at (0, -0.6, 0.7)), and it usually
     dominates.
   - Override it per task with a `mode="startup"` event that sets the `sphere` prim's intensity and colour
     (reference.md §11).
2. Match exposure and white balance to the policy cameras by tuning light and dome intensity and colour.
   Do not post-process.
   - Iterate with renders. The renderer's tone mapping is not linear, so solving light intensities from two
     test renders lands about 25% off; use a solve only as the first guess.
   - Tune the dome intensity against background regions and the scene lights against the work surface; when
     they conflict, the work surface wins (the policy looks at it).
   - Keep RoboLab's sphere light unless the photo clearly has no light from that side. Turning it off is a
     change to record in `versions.md`.
   - Measure with `check_version.py` color regions on the work surface, spread near / far.
   - A brightness gradient (far brighter than near, or the reverse) means a light is misplaced, not too
     strong.
   - Wrist cameras often auto-expose darker than sim. Accept up to about 25% there and randomize brightness
     instead.
3. If the policy's training images were JPEG-compressed or noisy, plan that as augmentation (Phase 9).

## Phase 7: Task definition

1. Write `RoboLab/robolab/tasks/benchmark/<task>_task.py` from reference.md §12:
   - presets from the real start-pose region, with a train/test split
   - the success conditional matching the real judge, with the gripper released
   - the background event
   - the exact instruction string(s)
   - an episode length ≥ the real one
   - **preset 0 = the photographed layout**, selectable by an env var (e.g. `ROBOLAB_PHOTO_LAYOUT=1`), so
     renders and checks reproduce the photo. Training and test presets exclude it.
   - fixed distractors listed in the preset with their photographed poses, and spawn regions for task objects
     that keep clear of them and of every fixture's footprint
   - build the presets with `robolab.core.sim2real.layouts`: `sample_layouts(n, objects, z, keepouts=...,
     min_dist=...)` keeps preset 0 as the photo layout, and `preset_indices(train, test)` reads the switches
   - events through `plan.build_events(task_terms, "<task>_scene/dr.json")` (Phase 10). It runs fixture
     resets after the layout reset, which would otherwise put fixtures back at their default pose.
   - read the scene USD and any layout file paths through env-var overrides with defaults (e.g.
     `<TASK>_SCENE`, `<TASK>_LAYOUT`), so version renders can run in parallel on different files
2. Write the env and training YAMLs: `camera_preset`, `task_description`, `max_episode_steps` and
   `max_steps_per_rollout_epoch` at 15 Hz, and camera render sizes from Phase 2.5. Copy keys from a working
   config; one existing env YAML misspells `umax_steps_per_rollout_epoch`, which RLinf ignores.
3. Use `use_rel_reward: True` for GRPO (reference.md §12.2).

## Phase 8: Version loop, then validate before any RL

1. **Labels, once per scene** (`$WD/labels.json`, reference.md §13.1). Write them on the real images:
   - the real robot mask (polygon fill), including the gripper wherever it is visible. The robot-IoU scan
     uses segmentation masks from `render_check.py`, which include the gripper; a mask without it lowers
     every candidate's score.
   - each visible object's centroid. Use a **polygon outline** for large or partly occluded objects, because a
     point picked by eye and a mask centroid only agree when both cover the same visible region. Polygons
     also add a per-object IoU check.
   - static polygons over rigid edges only (table edges, fixtures; never cloth folds or anything deformable)
   - colour regions on the work surface (near and far)
   - a pattern box if the surface is patterned
2. **Loop over versions v1, v2, ...** For each version:
   1. Make one change set. Record what changed and why.
   2. Regenerate the scene and assets.
   3. Render: `render_check.py --task-file ... --joint-pos <recorded> --entities robot <objects...> --out $WD/vN`,
      with the photographed layout on and DR off (`ROBOLAB_DR=0`). Mask exactly what the label outlines:
      `name=part1+part2` unions prim names, e.g. a lid's shell without the housing that moves with it.
   4. Score: `check_version.py $WD/labels.json $WD/vN --name vN --log $WD/versions.md`.
   5. Send `vN/sheet.jpg` (real | sim | edges with object markers, both cameras) to the user.
   6. Fix the worst failure first: scale and camera, then table geometry, then object placement, then
      lighting. Then start the next version.
      - Fix occluders before the objects they hide. Moving or resizing a fixture changes the visible part,
        and so the centroid, of everything behind it.
      - Removing or moving a large fixture also moves its shadow, so re-check colour regions after
        geometry changes.

   Stop at PASS. Default limits, at 1280 px:

   | Check | Limit |
   |---|---|
   | Robot IoU | ≥ 0.65 |
   | Object centroid, ext | ≤ 15 px |
   | Object centroid, wrist | ≤ 25 px |
   | Static chamfer | ≤ 6 px |
   | Colour gain, ext | ± 10% |
   | Colour gain, wrist | ± 25% |
   | Pattern period | ± 5% |

   - If a check cannot pass after a reasoned attempt, record the residual and its cause, and move it to DR.
   - Never loosen a limit or drop a label region to get a PASS unless it measures something the sim cannot
     represent (e.g. cloth folds). Say so in the report.
   - A label that was wrong (a colour region placed on a fold or shadow, a centroid on the wrong object) may be
     fixed. Record the old and new label and why in `versions.md`, and re-score earlier versions with it.
3. **Policy inputs.** Show the exact policy inputs once (resolution, letterbox / crop).
4. **Base policy.** Zero-shot eval on held-out presets (reference.md §3). Compare with real success using
   confidence intervals. Chase any large gap in either direction (reference.md §13).
5. **Replay.** If real trajectories exist, run `replay_traj.py`. Compare joint tracking, first grasp contact,
   slip and final object pose. Tune object physics first.

Do not start RL until these pass, or until the user accepts the listed gaps.

## Phase 9: Sim-to-real support plan

Every reconstruction ships with an explicit plan for what the sim varies to cover the gap to the real robot.
Write it as `$WD/sim2real_plan.md` (template in reference.md §14) and show it to the user. Phase 10 implements
it. Without a plan, the default is a sim that is exact for one photo and brittle everywhere else.

1. **Inventory the reconstruction.** For every quantity the sim fixed, record:
   - **Source:** measured, user-given, fitted from images, or assumed.
   - **Uncertainty (σ),** from the fit itself where possible:

     | Quantity | Where its σ comes from |
     |---|---|
     | Global scale | Width of the robot-IoU peak in the scale scan |
     | Camera pose | Version-loop residuals, plus the scale σ |
     | Exposure and white balance | `color_gain` residual, per camera |
     | Positions and sizes | Edge and centroid residuals |
     | Physical parameters never measured | Assumed: wide |

2. **Fill the plan, one row per factor** in each of these categories. Mark "not randomized" with a reason
   when a factor is pinned down.
   - **Viewpoint:** each policy camera's pose and intrinsics. Optionally a share of envs from a broader
     prior (e.g. the policy's training-camera distribution, such as DROID extrinsics) to avoid overfitting one
     viewpoint.
   - **Image formation:** brightness, white balance, contrast, noise, blur, compression, per camera.
   - **Lighting:** intensity, colour, direction of each light.
   - **Scene geometry:** support-surface height and extent, fixture poses, global scale.
   - **Object geometry and appearance:** scale, texture and colour of task objects.
   - **Object physics:** mass, friction, restitution, centre of mass, compliance.
   - **Robot:** controller gains, action latency, gripper force and speed.
   - **Background:** dome variants (crops of the real panorama, other rooms).
   - **Layout:** start poses of task objects (train / test split).
   - **Distractors:** none, the real ones at fixed poses, or random clutter. Choose by the deployment
     condition, which you must ask the user about.
3. **Ranges.** Centre each range on the reconstructed value. Width = max(2σ, the real variation the user
   expects across deployments).
   - A pinned-down camera on a fixed mount gets a narrow range.
   - A guessed friction gets a wide one.
   - Never centre a range on a default you did not fit.
4. **Prioritize, and match the deployment condition.**
   - **Tier 1 (cheap, high value):** viewpoint, image formation, object physics, robot gains, joint damping and
     friction of articulated fixtures, fixture pose. All are plan sections already.
   - **Tier 2:** the broader viewpoint prior, lighting, background variants.
   - **Tier 3:** per-env geometry and appearance (surface height, object scale, textures). These need
     `replicate_physics=False` or per-env USD variants, which cost startup time and memory at hundreds of envs.
     No plan section exists for them yet; list them as "not randomized: needs per-env geometry".

   The train-time scene (distractors, objects present) must match how the policy will be tested. Say so in
   the plan.
Mechanisms that already exist and what needs new code are listed in reference.md §14.0.1.

## Phase 10: Implement and check the randomization

The randomization is data: a `dr.json` plan read by `robolab.core.sim2real.plan`. Do not write event code for
anything a plan section covers.

1. **Write `RoboLab/assets/<task>_scene/dr.json`** from the plan's Tier 1–2 rows. Sections (full schema in the
   `builders` module docstring; ranges are `[lo, hi]`):
   - `cameras`: per-camera pose jitter (`pos_m`, `rot_deg`), per env every reset
   - `image`: per-camera gain, white balance, contrast, gamma, noise, zoom (stands in for FOV), blur
   - `lights`: intensity, warm / cool colour and position (shadow direction) of named light prims, every reset,
     as factors on the task's tuned values
   - `background`: photometric variants of the reconstructed dome (`make_dome_variants.py`: exposure, white
     balance, contrast, saturation, a directional illumination field, soft local blobs, a small yaw roll; content
     unchanged, never blurred) and dome intensity, every reset. It replaces the task's `set_background` term.
   - `mass`, `material`, `com`: per task object and moving link (startup)
   - `actuator_gains`: the robot's arm joints (startup, scale)
   - `joints`: per articulated fixture and joint regex, damping / stiffness / friction / armature in SI units
     (startup, per env)
   - `fixtures`: per fixture, pose jitter (`pos_m`, `yaw_deg`) and opening at reset (`joint_pos`
     `{regex: [lo, hi]}`), merged into the task's `reset_fixture` term

   Every range is centred on the reconstructed value. Layout presets stay in the task file.
2. **Wire it in the task:** `events = lambda: build_events(task_terms, "<task>_scene/dr.json")`, with the task's
   own terms as a dict. Nothing to copy or install.
   - `ROBOLAB_DR=0` turns the plan off (fixtures at their nominal pose): use it for version renders, physics
     checks and the clean base eval.
   - `ROBOLAB_DR_OVERRIDE=<file>` merges another plan last, key by key, for ablations; `{"enabled": false}`
     drops a section or an entry.
   - **Something no section covers** (a scene-specific effect): write the event function in a module under
     `RoboLab/robolab/core/sim2real/extras/` and either name it in a `custom` entry
     (`{"func": "module:function", "mode": "reset", "params": {...}}`), or register a new section with
     `@register("<section>")` and list the module in the plan's `"extras"`. Replacing a built-in section needs
     `replace=True`; say why in the plan.
3. **Check** with `check_dr.py` (GPU): a grid per camera with the real image first, two resets so resampling
   is visible, and the real image's percentile in the sim brightness / colour statistics.
   - The real image must fall within the 5th–95th percentile of each statistic; widen the range otherwise.
   - No sample may be absurd: black or saturated frames, camera inside geometry, objects floating, an object
     missing in some envs.
   - Send the grids to the user.
4. **Eval twice**: base policy with DR off (Phase 8.4) and with DR on. A large drop under DR means a range is too
   wide or a factor the policy never saw (check the grid for that factor).

## Phase 11: Deliverables

- `RoboLab/assets/scenes/<task>.usda`, `RoboLab/assets/objects/<task>/...`, `RoboLab/assets/<task>_scene/`
  (background, textures)
- the task file, and the env and training YAMLs
- the camera class, preset and any wrist-offset change (after approval)
- in `$WD`: fit and refine JSONs, renders, overlays, eval logs, `versions.md`, `sim2real_plan.md`,
  `for_review/` (side-by-sides, policy inputs, eval video)
- `report.md`: the measurement and assumption tables, validation numbers, open mismatches, and the exact render
  and base-eval commands. If you cannot write it (some agent setups block report files), put the same content
  in your final message.
- any existing-file change as a patch in `$WD` (at least the camera class and preset)

## Checklist

- [ ] Policy-camera images labelled (roles confirmed), joint states known; supplementary photos surveyed
- [ ] Missing facts requested in one checklist; assumptions recorded with uncertainty
- [ ] Scale cross-checked by 2 references; surface height and robot placement set
- [ ] Every policy camera: intrinsics sourced, pose fitted (pattern plane + robot scan, or PnP), wrist mount verified
- [ ] Version loop: labels written, `versions.md` shows each change, final version PASSES `check_version.py`
- [ ] Sim image pipeline (resolution, aspect, letterbox / crop) matches the real stack
- [ ] Articulated objects: built with `articulated_usd` (root `fixed` / `floating`), grasp part collidable and
      protruding, joint gains in SI, `check_articulation.py` env_spread ~0, closing / opening matches the real one
- [ ] Every object: kind, mesh source, size, mass, friction, restitution, collision, compliance / joints / material recorded
- [ ] Textures from real pixels, colour-matched to the policy cameras; near field is geometry
- [ ] Background without blur; about 0 holes for every policy camera
- [ ] Lighting direction and exposure matched (gain within about 5%)
- [ ] Task: exact instruction, real success condition, episode length, real start-pose region, train/test split
- [ ] Base policy vs real success compared; replay done if trajectories exist
- [ ] `sim2real_plan.md` written (every factor: source, σ, range, mechanism, tier)
- [ ] `dr.json` wired into the task through `build_events`; `check_dr.py` grid sent, real images inside the set
- [ ] Scene names unique (fixture parts included)
- [ ] Deliverables in place; existing-file changes approved or saved as a patch; nothing committed
