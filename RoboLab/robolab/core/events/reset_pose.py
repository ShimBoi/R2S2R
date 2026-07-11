# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import isaaclab.sim.utils as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.constants
import robolab.core.utils.usd_utils as usd_utils


########################################################
#  Config class for randomizing initial pose
########################################################
@configclass
class RandomizeInitPoseUniform:
    """Configuration for randomizing initial pose uniformly."""

    @classmethod
    def from_params(cls,
        objects: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
        pose_range: dict,
        velocity_range: dict = None,
        collision_margin: float = 0.0,
        max_retries: int = 100):
        """Create a RandomizeInitPoseUniform instance with custom parameters.

        Args:
            objects: The asset(s) to randomize.
            pose_range: Dictionary of pose ranges for each axis (x, y, z, roll, pitch, yaw).
            velocity_range: Dictionary of velocity ranges for each axis.
            collision_margin: Additional margin between objects beyond their bounding boxes.
                Set to > 0 to enable collision-aware sampling. The actual collision distance
                will be: radius1 + radius2 + collision_margin, where radii are computed from
                the objects' bounding boxes.
            max_retries: Maximum number of resampling attempts when collision checking is enabled.
        """
        if velocity_range is None:
            velocity_range = {}

        # Create a new class with the custom parameters
        class CustomRandomizeInitPoseUniform(cls):
            randomize_init_pose = EventTerm(
                func=reset_pose_uniform,
                mode="reset",
                params={
                    "pose_range": pose_range,
                    "velocity_range": velocity_range,
                    "asset_cfg": objects,
                    "collision_margin": collision_margin,
                    "max_retries": max_retries,
                }
            )

        return CustomRandomizeInitPoseUniform()

########################################################
#  Object pose initialization
########################################################

def _parse_asset_cfg(asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str) -> list[str]:
    """Parse asset_cfg into a list of asset names.

    Args:
        asset_cfg: Asset configuration - can be a SceneEntityCfg, string, or list of either.

    Returns:
        List of asset name strings.
    """
    if isinstance(asset_cfg, SceneEntityCfg):
        return [asset_cfg.name]
    elif isinstance(asset_cfg, str):
        return [asset_cfg]
    elif isinstance(asset_cfg, list):
        return [each.name if isinstance(each, SceneEntityCfg) else each for each in asset_cfg if isinstance(each, (SceneEntityCfg, str))]
    else:
        return list(asset_cfg)


def _get_object_radius(asset: RigidObject | Articulation, env_id: int = 0) -> float:
    """Get the bounding radius of an object from its USD prim dimensions.

    Args:
        asset: The asset to get radius for.
        env_id: The environment ID to get the correct prim.

    Returns:
        The bounding radius (half of max XY dimension).
    """
    try:
        prim_path = asset.cfg.prim_path
        prims = sim_utils.find_matching_prims(prim_path)
        env_id_str = f"env_{env_id}"
        for prim in prims:
            if env_id_str in str(prim.GetPath()):
                dims = usd_utils.get_dimensions(prim)
                # Use max of X and Y dimensions for the bounding circle radius
                return float(max(dims[0], dims[1]) / 2)
        # Fallback: use first prim found
        if prims:
            dims = usd_utils.get_dimensions(prims[0])
            return float(max(dims[0], dims[1]) / 2)
    except Exception as e:
        if robolab.constants.VERBOSE:
            print(f"Warning: Could not get object radius: {e}")
    return 0.0


def _check_collision_with_others(
    position: torch.Tensor,
    other_positions: list[tuple[torch.Tensor, float]],
    obj_radius: float,
    collision_margin: float = 0.0,
) -> bool:
    """Check if a position collides with any of the other positions using bounding circles.

    Args:
        position: The sampled position (3,) tensor.
        other_positions: List of (position, radius) tuples for other objects.
        obj_radius: Bounding radius of the object being placed.
        collision_margin: Additional margin to add between objects.

    Returns:
        True if collision detected, False otherwise.
    """
    if not other_positions:
        return False

    for other_pos, other_radius in other_positions:
        # Minimum distance is sum of radii plus margin
        min_dist = obj_radius + other_radius + collision_margin
        # Use XY distance for collision check (objects on the same surface)
        dist_xy = torch.sqrt((position[0] - other_pos[0]) ** 2 + (position[1] - other_pos[1]) ** 2)
        if dist_xy < min_dist:
            return True
    return False


def sample_pose_uniform(
    env: ManagerBasedEnv,
    asset: RigidObject | Articulation,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    other_positions: list[list[tuple[torch.Tensor, float]]] | None = None,
    obj_radius: float = 0.0,
    collision_margin: float = 0.0,
    max_retries: int = 100,
):
    """Sample a random pose uniformly within the given ranges, optionally avoiding collisions.

    Args:
        env: The environment instance.
        asset: The asset to sample pose for.
        env_ids: The environment indices to sample for.
        pose_range: Dictionary of pose ranges for each axis.
        velocity_range: Dictionary of velocity ranges for each axis.
        other_positions: List of placed positions per env_id. Each element is a list of
            (position, radius) tuples for that environment.
        obj_radius: Bounding radius of this object (from bounding box).
        collision_margin: Additional margin between objects.
        max_retries: Maximum number of resampling attempts per environment.

    Returns:
        Tuple of (positions, orientations, velocities) tensors.
    """
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)

    positions_before = root_states[:, 0:3] + env.scene.env_origins[env_ids]

    # Sample positions with collision checking if enabled
    if other_positions is not None and (obj_radius > 0 or collision_margin > 0):
        # Sample with collision avoidance - process each environment separately
        all_positions = []
        for idx, env_id in enumerate(env_ids):
            env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
            env_other_positions = other_positions[env_id_int] if env_id_int < len(other_positions) else []

            # Try sampling until collision-free or max retries
            for attempt in range(max_retries):
                rand_sample = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (1, 6), device=asset.device)
                position = positions_before[idx] + rand_sample[0, 0:3]

                if not _check_collision_with_others(position, env_other_positions, obj_radius, collision_margin):
                    break
                if attempt == max_retries - 1 and robolab.constants.VERBOSE:
                    print(f"Warning: Max retries ({max_retries}) reached for collision-free sampling")

            all_positions.append(position)

        positions = torch.stack(all_positions, dim=0)
        # Sample orientations separately
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    else:
        # Original behavior - sample all at once
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        positions = positions_before + rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    if robolab.constants.VERBOSE:
        print(f"positions: {positions} positions_before: {positions_before}")

    return positions, orientations, velocities

def reset_pose_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
    reset_to_default_otherwise: bool = True,
    use_collision_check: bool = True,
    collision_margin: float = 0.01,
    max_retries: int = 100,
    pose_ranges: list[dict[str, tuple[float, float]]] | None = None,
):
    """Reset asset root state to a random position and velocity uniformly within the given ranges.
    Adapted from isaaclab.envs.mdp.events.reset_root_state_uniform

    This function randomizes the root position and velocity of the assets in the asset_cfg list.
    If reset_to_default_otherwise is True, assets NOT in the asset_cfg list are reset to their default pose as specified in the scene configuration.
    Otherwise, they are left unchanged at reset time.
    If use_collision_check is True, the function will check for collisions with other objects during sampling. This prevents objects from being placed inside each other.
    The actual collision distance will be: radius1 + radius2 + collision_margin, where radii are computed from
    the objects' bounding boxes.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        pose_range: Dictionary of pose ranges for each axis, applied to all assets unless
            overridden by ``pose_ranges``.
        velocity_range: Dictionary of velocity ranges for each axis.
        asset_cfg: The asset(s) to randomize.
        reset_to_default_otherwise: If True, reset all other assets to default.
        use_collision_check: If True, check for collisions with other objects.
        collision_margin: Additional margin between objects beyond their bounding boxes.
            The actual collision distance will be: radius1 + radius2 + collision_margin, where radii are computed from
            the objects' bounding boxes.
        max_retries: Maximum number of resampling attempts when collision checking is enabled.
        pose_ranges: Optional list of per-asset pose range dicts, one entry per asset in
            ``asset_cfg`` (in the same order). When provided, each asset uses its own range
            instead of the shared ``pose_range``. Assets placed earlier in the list are
            collision-checked against when placing later ones.
    """
    asset_names = _parse_asset_cfg(asset_cfg)
    sampled_pose_assets = set(asset_names)

    # If reset_to_default_otherwise is True, reset all other assets to default first
    if reset_to_default_otherwise:
        all_asset_names = _get_all_asset_names(env)
        default_pose_assets = all_asset_names - sampled_pose_assets
        print(f"Resetting '{sampled_pose_assets}' via random uniform pose sampling and all other assets {default_pose_assets} to default")
        if default_pose_assets:
            _reset_assets_to_default(env, env_ids, default_pose_assets)

    # Track placed positions for collision avoidance (per environment)
    # Each entry is a list of (position, radius) tuples
    num_envs = env.num_envs
    placed_positions: list[list[tuple[torch.Tensor, float]]] = [[] for _ in range(num_envs)]

    # Randomize the specified assets
    for i, object_name in enumerate(asset_names):
        asset: RigidObject | Articulation = env.scene[object_name]

        # Per-asset range takes priority over the shared pose_range
        this_pose_range = pose_ranges[i] if (pose_ranges is not None and i < len(pose_ranges)) else pose_range

        if robolab.constants.VERBOSE:
            print(f"Randomizing initial pose for {object_name} according to: {this_pose_range} and {velocity_range}")

        # Get bounding radius for this object if collision checking is enabled
        obj_radius = 0.0
        if use_collision_check:
            # Use env_id 0 for getting radius (geometry is same across envs)
            obj_radius = _get_object_radius(asset, env_id=0)
            if robolab.constants.VERBOSE:
                print(f"  Object {object_name} bounding radius: {obj_radius:.4f}")

        positions, orientations, velocities = sample_pose_uniform(
            env, asset, env_ids, this_pose_range, velocity_range,
            other_positions=placed_positions if use_collision_check else None,
            obj_radius=obj_radius,
            collision_margin=collision_margin,
            max_retries=max_retries,
        )

        # Track placed positions for subsequent objects
        if use_collision_check:
            for idx, env_id in enumerate(env_ids):
                env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else env_id
                placed_positions[env_id_int].append((positions[idx].clone(), obj_radius))

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def _reset_assets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_names_set: set[str],
):
    """Internal helper to reset specified assets to their default pose.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        asset_names_set: Set of asset names to reset.
    """
    # Reset rigid bodies that are in the asset list
    for name, rigid_object in env.scene.rigid_objects.items():
        if name not in asset_names_set:
            continue
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    # Reset articulations that are in the asset list
    for name, articulation_asset in env.scene.articulations.items():
        if name not in asset_names_set:
            continue
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    # Reset deformable objects that are in the asset list
    for name, deformable_object in env.scene.deformable_objects.items():
        if name not in asset_names_set:
            continue
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def _get_all_asset_names(env: ManagerBasedEnv) -> set[str]:
    """Get all asset names in the scene."""
    all_names = set()
    all_names.update(env.scene.rigid_objects.keys())
    all_names.update(env.scene.articulations.keys())
    all_names.update(env.scene.deformable_objects.keys())
    return all_names


def reset_pose_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
):
    """Reset the specified assets' pose to the default state specified in the scene configuration.

    Only resets assets that are in the asset_cfg list.
    Every other asset is left unchanged.
    Adapted from isaaclab.envs.mdp.events.reset_scene_to_default.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        asset_cfg: The asset(s) to reset to default pose. Can be a single asset or list of assets.
    """
    asset_names_set = set(_parse_asset_cfg(asset_cfg))
    _reset_assets_to_default(env, env_ids, asset_names_set)


def reset_pose_to_presets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    presets: list[list[tuple]],
    asset_cfg: list[SceneEntityCfg] | SceneEntityCfg | list[str] | str,
    reset_to_default_otherwise: bool = True,
):
    """Reset assets to a randomly sampled preset pose.

    Each preset entry is a list of poses, one per asset in asset_cfg.  Each
    pose is either a 3-tuple ``(x, y, z)`` (orientation taken from the USD
    default) or a 7-tuple ``(x, y, z, qw, qx, qy, qz)`` (explicit
    orientation in Isaac Lab wxyz convention).

    A preset is chosen independently at random for each env being reset,
    unless the ``ROBOLAB_FORCE_PRESET_IDX`` environment variable is set, in
    which case every env being reset uses that exact preset index instead of
    a random draw. This exists for sim-to-sim verification, where RoboLab
    and PolaRiS need to be reset to the identical initial condition (indices
    are aligned 1:1 between this file's ``_PRESET_POSES`` and PolaRiS's
    ``initial_conditions.json`` -- see the docstring above ``_PRESET_POSES``
    in mug_on_cutting_board_task.py). Velocities are always zeroed.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        presets: List of presets.  Each preset is a list of 3- or 7-tuples,
            one entry per asset in ``asset_cfg`` (same order).
        asset_cfg: The asset(s) to place.  Order must match poses in each preset.
        reset_to_default_otherwise: If True, reset all other assets to their
            default pose first.
    """
    asset_names = _parse_asset_cfg(asset_cfg)
    sampled_pose_assets = set(asset_names)

    if reset_to_default_otherwise:
        all_asset_names = _get_all_asset_names(env)
        default_pose_assets = all_asset_names - sampled_pose_assets
        if default_pose_assets:
            _reset_assets_to_default(env, env_ids, default_pose_assets)

    num_presets = len(presets)
    forced_idx = os.environ.get("ROBOLAB_FORCE_PRESET_IDX")
    if forced_idx is not None:
        forced_idx = int(forced_idx)
        assert 0 <= forced_idx < num_presets, (
            f"ROBOLAB_FORCE_PRESET_IDX={forced_idx} out of range for "
            f"{num_presets} presets."
        )
        preset_indices = torch.full((len(env_ids),), forced_idx, dtype=torch.int64)
    else:
        # Sample one random preset index per env being reset
        preset_indices = torch.randint(0, num_presets, (len(env_ids),))

    for i, object_name in enumerate(asset_names):
        asset: RigidObject | Articulation = env.scene[object_name]
        root_states = asset.data.default_root_state[env_ids].clone()

        positions = []
        orientations = []
        for idx, env_id in enumerate(env_ids):
            env_id_int = env_id.item() if isinstance(env_id, torch.Tensor) else int(env_id)
            preset_idx_used = preset_indices[idx].item()
            pose = presets[preset_idx_used][i]
            print(
                f"[reset_pose_to_presets] env_id={env_id_int} asset={object_name} "
                f"preset_idx={preset_idx_used} raw_pose(x,y,z,qw,qx,qy,qz)={tuple(pose)}"
            )
            pos = torch.tensor(pose[:3], device=asset.device, dtype=torch.float32)
            pos = pos + env.scene.env_origins[env_id_int]
            positions.append(pos)
            if len(pose) == 7:
                ori = torch.tensor(pose[3:7], device=asset.device, dtype=torch.float32)
            else:
                ori = root_states[idx, 3:7]
            orientations.append(ori)

        positions = torch.stack(positions, dim=0)
        orientations = torch.stack(orientations, dim=0)
        velocities = torch.zeros_like(root_states[:, 7:13])

        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


########################################################
#  Logging functions
########################################################

def log_init_poses(init_object_poses, output_dir, title=""):
    """Randomize the pose of a single object."""
    import os

    from robolab.core.utils.plot_utils import plot_objects

    title = title.lower().replace(' ', '_')
    plot_objects(init_object_poses, title=title, image_path=os.path.join(output_dir, f"{title}.png"))

    import json
    with open(os.path.join(output_dir, f"{title}.json"), "w") as f:
        json.dump(init_object_poses, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
