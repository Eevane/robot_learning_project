# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from orbit.surgical.tasks.surgical.lift import mdp

from . import ik_abs_vision_env_cfg


@configclass
class BiActVisionObservationsCfg:
    """Observation config for vision policies using joint velocity and end-effector angular velocity."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_ang_vel = ObsTerm(
            func=mdp.body_ang_vel_in_asset_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="psm_tool_tip_link")},
        )
        overview_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("overview_camera")})
        side_rgb = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("side_camera")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class NeedleLiftBiVisionEnvCfg(ik_abs_vision_env_cfg.NeedleLiftVisionEnvCfg):
    """IK-absolute needle lift environment for the Bi-ACT vision baseline."""

    observations: BiActVisionObservationsCfg = BiActVisionObservationsCfg()


@configclass
class NeedleLiftBiVisionEnvCfg_PLAY(NeedleLiftBiVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
