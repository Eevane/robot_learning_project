# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.utils import configclass

from . import ik_abs_env_cfg
from .ik_rel_vision_env_cfg import SIDE_CAMERA_ROT_NEW, OVERVIEW_CAMERA_ROT_NEW, VisionObservationsCfg


@configclass
class NeedleLiftVisionEnvCfg(ik_abs_env_cfg.NeedleLiftEnvCfg):
    """IK-absolute needle lift environment with two RGB cameras for scripted data collection."""

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
