# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperate a visual needle-lift task and inspect camera viewpoints."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from collections.abc import Mapping

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation runner for vision-enabled Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Needle-PSM-IK-Rel-Vision-v0",
    help="Name of the task.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--save_images_dir",
    type=str,
    default=None,
    help="Directory to save PNG snapshots from camera observations. Disabled when omitted.",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=0,
    help="Automatically save camera snapshots every N simulation steps. Disabled when 0.",
)
parser.add_argument(
    "--print_obs_info",
    action="store_true",
    default=False,
    help="Print camera observation keys and shapes after reset and every save event.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

import carb

from omni.isaac.lab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from orbit.surgical.ext.devices import Se3KeyboardDualArm
import orbit.surgical.tasks  # noqa: F401


CAMERA_KEYS = ("overview_rgb", "side_rgb")


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    if "Reach" in args_cli.task:
        return delta_pose

    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    return torch.concat([delta_pose, gripper_vel], dim=1)


def process_actions(teleop_interface, env, is_dual: bool) -> torch.Tensor:
    """Process actions for the environment."""
    if is_dual:
        delta_pose_0, gripper_command_0, delta_pose_1, gripper_command_1 = teleop_interface.advance()
        delta_pose_0 = torch.tensor(delta_pose_0.astype("float32"), device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        delta_pose_1 = torch.tensor(delta_pose_1.astype("float32"), device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        actions_0 = pre_process_actions(delta_pose_0, gripper_command_0)
        actions_1 = pre_process_actions(delta_pose_1, gripper_command_1)
        return torch.concat([actions_0, actions_1], dim=1)

    delta_pose, gripper_command = teleop_interface.advance()
    delta_pose = torch.tensor(delta_pose.astype("float32"), device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
    return pre_process_actions(delta_pose, gripper_command)


def ensure_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Convert a camera tensor to uint8 HWC for saving."""
    image = image.detach().cpu()
    if image.dtype == torch.uint8:
        return image
    if image.is_floating_point():
        if image.max() <= 1.0:
            image = image * 255.0
        return image.clamp(0, 255).to(torch.uint8)
    return image.clamp(0, 255).to(torch.uint8)


def log_camera_observations(obs_dict: Mapping[str, torch.Tensor]) -> None:
    """Print camera observation keys, shapes and dtypes for the first environment."""
    print("[INFO] Policy observation keys:", list(obs_dict.keys()))
    for key in CAMERA_KEYS:
        if key in obs_dict:
            value = obs_dict[key]
            print(f"[INFO] {key}: shape={tuple(value.shape)} dtype={value.dtype}")


def save_camera_snapshot(obs_dict: Mapping[str, torch.Tensor], output_dir: str, step: int) -> None:
    """Save the current first-environment camera frames as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    for key in CAMERA_KEYS:
        if key not in obs_dict:
            continue
        image = ensure_uint8_image(obs_dict[key][0])
        output_path = os.path.join(output_dir, f"{key}_step_{step:06d}.png")
        plt.imsave(output_path, image.numpy())
        print(f"[INFO] Saved {key} snapshot to: {output_path}")


def main():
    """Run teleoperation with visual observations enabled."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    if args_cli.teleop_device.lower() == "keyboard" and "Dual" not in args_cli.task:
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "keyboard" and "Dual" in args_cli.task:
        teleop_interface = Se3KeyboardDualArm(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")

    latest_obs = {"policy": None}

    def reset_env():
        obs, _ = env.reset()
        latest_obs["policy"] = obs["policy"]
        if args_cli.print_obs_info:
            log_camera_observations(latest_obs["policy"])

    def dump_snapshot():
        if latest_obs["policy"] is None or args_cli.save_images_dir is None:
            return
        save_camera_snapshot(latest_obs["policy"], args_cli.save_images_dir, step_count[0])
        if args_cli.print_obs_info:
            log_camera_observations(latest_obs["policy"])

    teleop_interface.add_callback("L", reset_env)
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface.add_callback("O", dump_snapshot)

    print(teleop_interface)
    print("[INFO] Keyboard shortcuts:")
    print("  L: reset environment")
    if args_cli.teleop_device.lower() == "keyboard":
        print("  O: save current overview/side camera PNGs")

    step_count = [0]
    reset_env()
    teleop_interface.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            is_dual = "Dual" in args_cli.task
            actions = process_actions(teleop_interface, env, is_dual)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            latest_obs["policy"] = obs_dict["policy"]
            step_count[0] += 1

            if args_cli.save_images_dir and args_cli.save_interval > 0 and step_count[0] % args_cli.save_interval == 0:
                dump_snapshot()

            dones = terminated | truncated
            if dones.any():
                reset_env()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
