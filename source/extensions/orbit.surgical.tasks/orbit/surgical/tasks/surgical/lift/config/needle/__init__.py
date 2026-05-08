# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import gymnasium as gym

from . import (
    agents,
    ik_abs_bi_vision_env_cfg,
    ik_abs_env_cfg,
    ik_abs_vision_env_cfg,
    ik_rel_env_cfg,
    ik_rel_vision_env_cfg,
    joint_pos_env_cfg,
)

ROBOMIMIC_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "agents", "robomimic")
BC_CFG = os.path.join(ROBOMIMIC_CONFIG_DIR, "bc.json")
BCQ_CFG = os.path.join(ROBOMIMIC_CONFIG_DIR, "bcq.json")
BC_VISION_CFG = os.path.join(ROBOMIMIC_CONFIG_DIR, "bc_vision.json")

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-Needle-PSM-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.NeedleLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.NeedleLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Abs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.NeedleLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": BC_CFG,
        "robomimic_bcq_cfg_entry_point": BCQ_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Abs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.NeedleLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_vision_env_cfg.NeedleLiftVisionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": BC_VISION_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Abs-Vision-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_vision_env_cfg.NeedleLiftVisionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Abs-Bi-Vision-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_bi_vision_env_cfg.NeedleLiftBiVisionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": BC_VISION_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Abs-Bi-Vision-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_bi_vision_env_cfg.NeedleLiftBiVisionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Rel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": BC_CFG,
        "robomimic_bcq_cfg_entry_point": BCQ_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Rel-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.NeedleLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Rel-Vision-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_vision_env_cfg.NeedleLiftVisionEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
        "robomimic_bc_cfg_entry_point": BC_VISION_CFG,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Needle-PSM-IK-Rel-Vision-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_vision_env_cfg.NeedleLiftVisionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftNeedlePPORunnerCfg,
    },
    disable_env_checker=True,
)
