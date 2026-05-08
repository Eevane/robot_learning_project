# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz

from orbit.surgical.tasks.surgical.lift import mdp
from orbit.surgical.tasks.surgical.lift.lift_env_cfg import ObservationsCfg as BaseObservationsCfg

from . import ik_rel_env_cfg


def _quat_tuple(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert XYZ Euler angles to a quaternion tuple for config use."""

    quat = quat_from_euler_xyz(
        torch.tensor([roll], dtype=torch.float32),
        torch.tensor([pitch], dtype=torch.float32),
        torch.tensor([yaw], dtype=torch.float32),
    )[0]
    return tuple(float(value) for value in quat.tolist())


OVERVIEW_CAMERA_ROT = _quat_tuple(0.0, -0.65, math.pi)
OVERVIEW_CAMERA_ROT_NEW = tuple([0, 0, 0.2672, 0.9636])
SIDE_CAMERA_ROT = _quat_tuple(0.0, -0.35, -math.pi / 2.0)
SIDE_CAMERA_ROT_NEW = tuple([0.6938, 0.5788, -0.2859, -0.3189])


@configclass
class VisionObservationsCfg(BaseObservationsCfg):
    """Observation config for visual imitation learning."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        overview_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("overview_camera")})
        side_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("side_camera")})

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class NeedleLiftVisionEnvCfg(ik_rel_env_cfg.NeedleLiftEnvCfg):
    """IK-relative needle lift environment with two RGB cameras for ACT-style data collection."""

    observations: VisionObservationsCfg = VisionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.overview_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/OverviewCamera",
            update_period=0.0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.3, 0.6),
                rot=OVERVIEW_CAMERA_ROT_NEW,
                convention="opengl",
            ),
        )
        self.scene.side_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/SideCamera",
            update_period=0.0,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.3, -0.2, 0.10),
                rot=SIDE_CAMERA_ROT_NEW,
                convention="opengl",
            ),
        )


@configclass
class NeedleLiftVisionEnvCfg_PLAY(NeedleLiftVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
