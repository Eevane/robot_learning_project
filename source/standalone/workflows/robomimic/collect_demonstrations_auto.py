# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect scripted ACT-style demonstrations as per-episode .pt files."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import re
from collections.abc import Sequence

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect scripted demonstrations for ACT-style imitation learning.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0",
    help="Name of the task.",
)
parser.add_argument("--num_demos", type=int, default=1, help="Number of successful episodes to store.")
parser.add_argument("--filename", type=str, default="act_dataset", help="Name of the output dataset directory.")
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=300,
    help="Maximum number of control steps per episode before forcing a reset.",
)
parser.add_argument(
    "--wait_scale",
    type=float,
    default=1.0,
    help="Global multiplier for scripted state-machine wait times. Values < 1.0 make the expert faster.",
)
parser.add_argument(
    "--object_x_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Override object reset x-range in meters, e.g. --object_x_range -0.05 0.05.",
)
parser.add_argument(
    "--object_y_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Override object reset y-range in meters, e.g. --object_y_range -0.05 0.05.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import warp as wp

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab.utils.math import subtract_frame_transforms

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401
from orbit.surgical.tasks.surgical.lift import mdp

wp.init()


SUPPORTED_TASKS = {
    "Isaac-Lift-Needle-PSM-IK-Abs-v0",
    "Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0",
    "Isaac-Lift-Needle-PSM-IK-Abs-Bi-Vision-v0",
}


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(0.7)
    GRASP_OBJECT = wp.constant(0.5)
    LIFT_OBJECT = wp.constant(2.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    rest_wait: float,
    approach_above_wait: float,
    approach_object_wait: float,
    grasp_wait: float,
    lift_wait: float,
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= rest_wait:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= approach_above_wait:
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= approach_object_wait:
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= grasp_wait:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= lift_wait:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """Scripted pick-and-lift expert in task space."""

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", wait_scale: float = 1.0):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.wait_scale = float(wait_scale)
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.05
        self.offset[:, -1] = 1.0
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.wait_times = {
            "rest": max(float(PickSmWaitTime.REST) * self.wait_scale, self.dt),
            "approach_above": max(float(PickSmWaitTime.APPROACH_ABOVE_OBJECT) * self.wait_scale, self.dt),
            "approach_object": max(float(PickSmWaitTime.APPROACH_OBJECT) * self.wait_scale, self.dt),
            "grasp": max(float(PickSmWaitTime.GRASP_OBJECT) * self.wait_scale, self.dt),
            "lift": max(float(PickSmWaitTime.LIFT_OBJECT) * self.wait_scale, self.dt),
        }

    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.wait_times["rest"],
                self.wait_times["approach_above"],
                self.wait_times["approach_object"],
                self.wait_times["grasp"],
                self.wait_times["lift"],
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(object_pose.contiguous(), wp.transform),
                wp.from_torch(des_object_pose.contiguous(), wp.transform),
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def compute_expert_actions(env, pick_sm: PickAndLiftSm) -> torch.Tensor:
    robot: RigidObject = env.scene["robot"]
    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
    tcp_rest_position_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position
    )
    tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

    object_data: RigidObjectData = env.unwrapped.scene["object"].data
    object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
    object_position_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_position
    )
    object_orientation = object_data.root_quat_w
    desired_pose = env.unwrapped.command_manager.get_command("object_pose")

    return pick_sm.compute(
        torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1),
        torch.cat([object_position_b, object_orientation], dim=-1),
        desired_pose,
    )


def clone_policy_obs(obs_dict: dict[str, torch.Tensor], env_id: int) -> dict[str, torch.Tensor]:
    return {key: value[env_id].detach().cpu().clone() for key, value in obs_dict.items()}


def clone_action(actions: torch.Tensor, env_id: int) -> torch.Tensor:
    return actions[env_id].detach().cpu().clone()


def init_episode_buffer(num_envs: int) -> list[dict[str, list]]:
    return [{"obs": [], "actions": [], "dones": []} for _ in range(num_envs)]


def finalize_episode(buffer: dict[str, list], success: bool, metadata: dict) -> dict:
    obs_by_key: dict[str, torch.Tensor] = {}
    first_obs = buffer["obs"][0]
    for key in first_obs:
        obs_by_key[key] = torch.stack([step_obs[key] for step_obs in buffer["obs"]], dim=0)

    return {
        "obs": obs_by_key,
        "actions": torch.stack(buffer["actions"], dim=0),
        "dones": torch.tensor(buffer["dones"], dtype=torch.bool),
        "success": bool(success),
        "metadata": metadata,
    }


def load_existing_manifest(dataset_dir: str) -> dict | None:
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_next_episode_index(episodes_dir: str, existing_manifest: dict | None) -> int:
    max_index = -1

    if existing_manifest is not None:
        for item in existing_manifest.get("episodes", []):
            max_index = max(max_index, int(item.get("episode_index", -1)))

    if os.path.exists(episodes_dir):
        pattern = re.compile(r"episode_(\d+)\.pt$")
        for filename in os.listdir(episodes_dir):
            match = pattern.match(filename)
            if match is not None:
                max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def main():
    if args_cli.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{args_cli.task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = args_cli.max_episode_steps * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    pose_range = env_cfg.events.reset_object_position.params["pose_range"]
    if args_cli.object_x_range is not None:
        pose_range["x"] = tuple(args_cli.object_x_range)
    if args_cli.object_y_range is not None:
        pose_range["y"] = tuple(args_cli.object_y_range)

    env = gym.make(args_cli.task, cfg=env_cfg)

    dataset_dir = os.path.abspath(os.path.join("logs", "act", args_cli.task, args_cli.filename))
    episodes_dir = os.path.join(dataset_dir, "episodes")
    params_dir = os.path.join(dataset_dir, "params")
    os.makedirs(episodes_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    existing_manifest = load_existing_manifest(dataset_dir)
    next_episode_index = infer_next_episode_index(episodes_dir, existing_manifest)
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)

    obs_dict, _ = env.reset()
    env.sim.step()
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        wait_scale=args_cli.wait_scale,
    )
    actions = compute_expert_actions(env, pick_sm)

    episode_buffers = init_episode_buffer(env.unwrapped.num_envs)
    saved_episodes: list[dict] = [] if existing_manifest is None else list(existing_manifest.get("episodes", []))
    success_count = 0
    discarded_count = 0 if existing_manifest is None else int(existing_manifest.get("num_discarded_episodes", 0))

    with torch.inference_mode():
        while success_count < args_cli.num_demos:
            for env_id in range(env.unwrapped.num_envs):
                episode_buffers[env_id]["obs"].append(clone_policy_obs(obs_dict["policy"], env_id))
                episode_buffers[env_id]["actions"].append(clone_action(actions, env_id))

            obs_dict, _, terminated, truncated, _ = env.step(actions)
            if env.unwrapped.sim.is_stopped():
                break

            dones = (terminated | truncated).detach().cpu()
            successes = env.termination_manager.get_term("object_reached_goal").detach().cpu()

            for env_id in range(env.unwrapped.num_envs):
                episode_buffers[env_id]["dones"].append(bool(dones[env_id]))

                if not dones[env_id]:
                    continue

                if successes[env_id]:
                    episode_index = next_episode_index
                    episode = finalize_episode(
                        episode_buffers[env_id],
                        success=True,
                        metadata={
                            "task_name": args_cli.task,
                            "action_space": "ik_abs",
                            "episode_index": episode_index,
                            "num_cameras": sum("rgb" in key for key in episode_buffers[env_id]["obs"][0]),
                            "obs_keys": list(episode_buffers[env_id]["obs"][0].keys()),
                            "control_dt": env_cfg.sim.dt * env_cfg.decimation,
                        },
                    )
                    episode_path = os.path.join(episodes_dir, f"episode_{episode_index:05d}.pt")
                    torch.save(episode, episode_path)
                    saved_episodes.append(
                        {
                            "episode_index": episode_index,
                            "file": os.path.basename(episode_path),
                            "num_steps": len(episode_buffers[env_id]["actions"]),
                            "success": True,
                        }
                    )
                    success_count += 1
                    next_episode_index += 1
                    print(f"[INFO] Saved successful episode {episode_index} to {episode_path}")
                else:
                    discarded_count += 1
                    print(
                        "[INFO] Discarded unsuccessful episode "
                        f"(env={env_id}, steps={len(episode_buffers[env_id]['actions'])})."
                    )

                episode_buffers[env_id] = {"obs": [], "actions": [], "dones": []}

            reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                pick_sm.reset_idx(reset_env_ids)
            actions = compute_expert_actions(env, pick_sm)

    manifest = {
        "task_name": args_cli.task,
        "num_episodes": len(saved_episodes),
        "num_discarded_episodes": discarded_count,
        "num_envs": env.unwrapped.num_envs,
        "action_space": "ik_abs",
        "max_episode_steps": args_cli.max_episode_steps,
        "episodes": saved_episodes,
    }
    with open(os.path.join(dataset_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
