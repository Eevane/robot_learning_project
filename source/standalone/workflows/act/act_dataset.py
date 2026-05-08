# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch datasets for ACT-style episode .pt demonstrations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EpisodeIndex:
    episode_id: int
    step_id: int


class ActEpisodeDataset(Dataset):
    """Dataset that returns full episodes saved by collect_demonstrations_auto.py."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.episodes_dir = os.path.join(self.dataset_dir, "episodes")
        manifest_path = os.path.join(self.dataset_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Could not find dataset manifest: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.episode_files = [os.path.join(self.episodes_dir, item["file"]) for item in self.manifest["episodes"]]

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, index: int) -> dict:
        return torch.load(self.episode_files[index], map_location="cpu")


class ActSequenceDataset(Dataset):
    """Dataset that returns (observation, future action chunk) pairs for ACT-style training."""

    def __init__(
        self,
        dataset_dir: str,
        action_chunk_size: int,
        image_keys: tuple[str, ...] = ("overview_rgb", "side_rgb"),
        episode_indices: list[int] | None = None,
        history_length: int = 1,
        exclude_obs_keys: tuple[str, ...] = (),
    ):
        if action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive.")
        if history_length <= 0:
            raise ValueError("history_length must be positive.")

        self.dataset = ActEpisodeDataset(dataset_dir)
        self.action_chunk_size = action_chunk_size
        self.image_keys = image_keys
        self.history_length = history_length
        self.exclude_obs_keys = set(exclude_obs_keys)
        if episode_indices is None:
            self.episode_indices = list(range(len(self.dataset)))
        else:
            self.episode_indices = sorted(episode_indices)
        self.episodes = [self.dataset[idx] for idx in self.episode_indices]
        self.flat_index: list[EpisodeIndex] = []
        for episode_id, episode in enumerate(self.episodes):
            num_steps = episode["actions"].shape[0]
            self.flat_index.extend(EpisodeIndex(episode_id, step_id) for step_id in range(num_steps))

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        epi_index = self.flat_index[index]
        episode = self.episodes[epi_index.episode_id]
        step_id = epi_index.step_id

        obs = {}
        for key, value in episode["obs"].items():
            if key in self.image_keys:
                item = value[step_id]
                obs[key] = item.permute(2, 0, 1).float() / 255.0
            else:
                history = []
                history_mask = torch.zeros(self.history_length, dtype=torch.bool)
                start_id = max(0, step_id - self.history_length + 1)
                history_values = value[start_id : step_id + 1]
                pad_count = self.history_length - history_values.shape[0]
                if pad_count > 0:
                    history.extend([history_values[0].clone()] * pad_count)
                history.extend([item.clone() for item in history_values])
                history_mask[pad_count:] = True
                obs[key] = torch.stack(history, dim=0).float()
                obs[f"{key}_history_mask"] = history_mask

        actions = episode["actions"]
        chunk = actions[step_id : step_id + self.action_chunk_size]
        valid_len = chunk.shape[0]
        if valid_len < self.action_chunk_size:
            pad = chunk[-1:].repeat(self.action_chunk_size - valid_len, 1)
            chunk = torch.cat([chunk, pad], dim=0)

        padding_mask = torch.zeros(self.action_chunk_size, dtype=torch.bool)
        padding_mask[:valid_len] = True

        return {
            "obs": obs,
            "actions": chunk.float(),
            "action_is_valid": padding_mask,
            "episode_id": torch.tensor(epi_index.episode_id, dtype=torch.long),
            "step_id": torch.tensor(step_id, dtype=torch.long),
        }

    @property
    def non_image_keys(self) -> tuple[str, ...]:
        first_obs = self.episodes[0]["obs"]
        return tuple(
            key for key in first_obs.keys() if key not in self.image_keys and key not in self.exclude_obs_keys
        )

    @property
    def obs_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self.episodes[0]["obs"].keys() if key not in self.exclude_obs_keys)

    @property
    def action_dim(self) -> int:
        return int(self.episodes[0]["actions"].shape[-1])

    def infer_shapes(self) -> dict[str, object]:
        first_obs = self.episodes[0]["obs"]
        image_shapes = {key: tuple(first_obs[key].shape[1:]) for key in self.image_keys if key in first_obs}
        single_state_dim = sum(int(first_obs[key].shape[-1]) for key in self.non_image_keys)
        return {
            "action_dim": self.action_dim,
            "state_dim": single_state_dim * self.history_length,
            "single_state_dim": single_state_dim,
            "history_length": self.history_length,
            "obs_keys": self.obs_keys,
            "image_keys": tuple(key for key in self.image_keys if key in first_obs),
            "image_shapes": image_shapes,
        }

    def compute_normalization_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        state_tensors = []
        action_tensors = []
        for episode in self.episodes:
            per_key_histories = []
            for key in self.non_image_keys:
                values = episode["obs"][key].float()
                history_slices = []
                for step_id in range(values.shape[0]):
                    start_id = max(0, step_id - self.history_length + 1)
                    history_values = values[start_id : step_id + 1]
                    pad_count = self.history_length - history_values.shape[0]
                    if pad_count > 0:
                        pad = history_values[:1].repeat(pad_count, 1)
                        history_values = torch.cat([pad, history_values], dim=0)
                    history_slices.append(history_values.reshape(-1))
                per_key_histories.append(torch.stack(history_slices, dim=0))
            state_tensors.append(torch.cat(per_key_histories, dim=-1))
            action_tensors.append(episode["actions"].float())

        states = torch.cat(state_tensors, dim=0)
        actions = torch.cat(action_tensors, dim=0)
        eps = 1.0e-6
        return {
            "state": {
                "mean": states.mean(dim=0),
                "std": states.std(dim=0).clamp_min(eps),
            },
            "action": {
                "mean": actions.mean(dim=0),
                "std": actions.std(dim=0).clamp_min(eps),
            },
        }


def split_episode_indices(
    num_episodes: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_episodes, generator=generator).tolist()
    val_count = min(max(int(round(num_episodes * val_ratio)), 1 if num_episodes > 1 and val_ratio > 0.0 else 0), num_episodes - 1)
    val_indices = sorted(perm[:val_count])
    train_indices = sorted(perm[val_count:])
    return train_indices, val_indices


def build_dataloader(
    dataset_dir: str,
    action_chunk_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    episode_indices: list[int] | None = None,
    history_length: int = 1,
    exclude_obs_keys: tuple[str, ...] = (),
) -> DataLoader:
    dataset = ActSequenceDataset(
        dataset_dir=dataset_dir,
        action_chunk_size=action_chunk_size,
        episode_indices=episode_indices,
        history_length=history_length,
        exclude_obs_keys=exclude_obs_keys,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
