# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained ACT-style policy in Isaac Lab."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained ACT-style policy.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--query_frequency", type=int, default=8, help="Replan every N control steps.")
parser.add_argument("--print_every", type=int, default=25, help="Print one action summary every N control steps.")
parser.add_argument(
    "--ensemble_decay",
    type=float,
    default=0.05,
    help="Exponential decay for temporal ensembling. Larger values favor newer predictions.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--max_episode_steps", type=float, default=0, help="Maximum episode length in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401
from orbit.surgical.tasks.surgical.lift import mdp

try:
    from .act_policy import flatten_state
    from .checkpoint_utils import load_checkpoint
except ImportError:
    from act_policy import flatten_state
    from checkpoint_utils import load_checkpoint


@dataclass
class PlannedChunk:
    start_step: int
    actions: torch.Tensor


def preprocess_policy_obs(
    obs: dict[str, torch.Tensor],
    image_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    processed = {}
    for key, value in obs.items():
        if key in image_keys:
            processed[key] = value.permute(0, 3, 1, 2).float() / 255.0
        else:
            processed[key] = value.float()
    return processed


def temporally_ensemble_actions(
    plans: deque[PlannedChunk],
    current_step: int,
    decay: float,
) -> tuple[torch.Tensor, int]:
    candidates: list[torch.Tensor] = []
    weights: list[float] = []

    for plan in plans:
        offset = current_step - plan.start_step
        if 0 <= offset < plan.actions.shape[0]:
            candidates.append(plan.actions[offset])
            weights.append(float(torch.exp(torch.tensor(-decay * offset)).item()))

    if not candidates:
        raise RuntimeError("No valid action candidates available for temporal ensembling.")

    stacked = torch.stack(candidates, dim=0)
    weight_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1.0e-8)
    ensembled = (stacked * weight_tensor.unsqueeze(-1)).sum(dim=0)
    return ensembled, len(candidates)


def update_state_history(
    obs: dict[str, torch.Tensor],
    non_image_keys: tuple[str, ...],
    state_history: dict[str, deque[torch.Tensor]],
    history_length: int,
) -> dict[str, torch.Tensor]:
    history_obs = dict(obs)
    for key in non_image_keys:
        state_history[key].append(obs[key][0].detach().clone())
        history_list = list(state_history[key])
        if len(history_list) < history_length:
            history_list = [history_list[0].clone()] * (history_length - len(history_list)) + history_list
        history_obs[key] = torch.stack(history_list, dim=0).unsqueeze(0).to(obs[key].device).float()
    return history_obs


def main() -> None:
    device = torch.device(args_cli.device)
    model, checkpoint = load_checkpoint(args_cli.checkpoint, device=device)
    metadata = checkpoint["metadata"]
    config = checkpoint["config"]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # increase episode length
    if args_cli.max_episode_steps > 0:
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.max_episode_steps)
    env = gym.make(args_cli.task, cfg=env_cfg)

    state_mean = metadata["normalization_stats"]["state"]["mean"].to(device)
    state_std = metadata["normalization_stats"]["state"]["std"].to(device)
    action_mean = metadata["normalization_stats"]["action"]["mean"].to(device)
    action_std = metadata["normalization_stats"]["action"]["std"].to(device)
    non_image_keys = tuple(metadata["non_image_keys"])
    action_chunk_size = int(config["action_chunk_size"])
    history_length = int(config.get("history_length", metadata.get("history_length", 1)))

    obs_dict, _ = env.reset()
    obs = preprocess_policy_obs(obs_dict["policy"], tuple(metadata["image_keys"]))
    state_history = {key: deque(maxlen=history_length) for key in non_image_keys}
    obs = update_state_history(obs, non_image_keys, state_history, history_length)
    plans: deque[PlannedChunk] = deque(maxlen=max(action_chunk_size * 2, 32))
    global_step = 0
    episode_count = 0
    success_count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            if not plans or global_step % max(args_cli.query_frequency, 1) == 0:
                state = flatten_state(obs, non_image_keys)
                norm_state = (state - state_mean) / state_std
                prediction = model(obs=obs, state=norm_state)["pred_actions"]
                prediction = prediction * action_std.view(1, 1, -1) + action_mean.view(1, 1, -1)
                plans.append(PlannedChunk(start_step=global_step, actions=prediction[0]))

            action, num_candidates = temporally_ensemble_actions(
                plans=plans,
                current_step=global_step,
                decay=args_cli.ensemble_decay,
            )
            action = action.view(1, -1)
            action = torch.clamp(action, min=-1.0, max=1.0)
            if args_cli.print_every > 0 and global_step % args_cli.print_every == 0:
                print(
                    f"[ACT] step={global_step} candidates={num_candidates} "
                    f"action={action[0].detach().cpu().tolist()}"
                )
            obs_dict, _, terminated, truncated, _ = env.step(action)
            obs = preprocess_policy_obs(obs_dict["policy"], tuple(metadata["image_keys"]))
            obs = update_state_history(obs, non_image_keys, state_history, history_length)
            global_step += 1

            if bool((terminated | truncated).item()):
                episode_count += 1
                is_success = bool(env.termination_manager.get_term("object_reached_goal").item())
                success_count += int(is_success)
                success_rate = success_count / max(episode_count, 1)
                print(
                    f"[ACT] episode={episode_count} success={is_success} "
                    f"success_rate={success_rate:.3f} ({success_count}/{episode_count})"
                )
                plans.clear()
                obs_dict, _ = env.reset()
                obs = preprocess_policy_obs(obs_dict["policy"], tuple(metadata["image_keys"]))
                state_history = {key: deque(maxlen=history_length) for key in non_image_keys}
                obs = update_state_history(obs, non_image_keys, state_history, history_length)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
