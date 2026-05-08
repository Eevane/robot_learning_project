# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def camera_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """RGB image from a camera sensor.

    Isaac Lab camera outputs can contain an alpha channel. For policy inputs we keep
    only RGB and preserve the original HWC layout expected by downstream dataset code.
    """

    camera: Camera = env.scene[sensor_cfg.name]

    # During environment construction, observation terms are queried once to infer output shapes
    # before camera sensors receive their PLAY-time initialization callback. In that phase we can
    # only rely on the static camera config.
    if not getattr(camera, "_is_initialized", False) or not hasattr(camera, "_is_outdated"):
        return torch.zeros(
            (env.num_envs, camera.cfg.height, camera.cfg.width, 3),
            device=env.device,
            dtype=torch.uint8,
        )

    rgb = camera.data.output["rgb"][..., :3]
    return rgb.contiguous()


def body_ang_vel_in_asset_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Angular velocity of selected articulation bodies in the asset root frame.

    The returned tensor is flattened over bodies, so selecting a single body yields a
    `(num_envs, 3)` tensor and selecting multiple bodies yields `(num_envs, 3 * num_bodies)`.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]

    if body_ang_vel_w.ndim == 2:
        return math_utils.quat_rotate_inverse(asset.data.root_quat_w, body_ang_vel_w)

    num_envs, num_bodies, _ = body_ang_vel_w.shape
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)
    body_ang_vel_b = math_utils.quat_rotate_inverse(
        root_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    )
    return body_ang_vel_b.view(num_envs, num_bodies * 3)
