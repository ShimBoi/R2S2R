# rlinf/models/embodiment/openpi/dataconfig/droid_jointpos_dataconfig.py
import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.ascontiguousarray(image.transpose(1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(_transforms.DataTransformFn):
    """Handles both raw DROID dataset keys (training) and RLinf env keys (rollout/eval)."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "observation/state" in data:
            # Eval/rollout path: RLinf env already gives us a flat state vector.
            state = np.asarray(data["observation/state"])
        else:
            # Training path: raw DROID dataset stores joint pos + gripper pos separately.
            gripper_pos = np.asarray(data["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([data["observation/joint_position"], gripper_pos])

        base_key = (
            "observation/image"
            if "observation/image" in data
            else "observation/exterior_image_1_left"
        )
        wrist_key = (
            "observation/wrist_image"
            if "observation/wrist_image" in data
            else "observation/wrist_image_left"
        )

        base_image = _parse_image(data[base_key])
        wrist_image = (
            _parse_image(data[wrist_key])
            if wrist_key in data
            else np.zeros_like(base_image)
        )

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :8])}


@dataclasses.dataclass(frozen=True)
class RoboLabDroidJointPosDataConfig(DataConfigFactory):
    """OpenPI data config for Droid joint position control."""

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[DroidInputs(model_type=model_config.model_type)],
            outputs=[
                _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                DroidOutputs(),
            ],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
