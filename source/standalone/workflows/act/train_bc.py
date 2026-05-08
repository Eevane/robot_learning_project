# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a simple visual behavior cloning baseline on ACT-format demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .act_dataset import ActEpisodeDataset, ActSequenceDataset, split_episode_indices
    from .bc_checkpoint_utils import save_checkpoint
    from .bc_policy import BehaviorCloningPolicy, flatten_state
except ImportError:
    from act_dataset import ActEpisodeDataset, ActSequenceDataset, split_episode_indices
    from bc_checkpoint_utils import save_checkpoint
    from bc_policy import BehaviorCloningPolicy, flatten_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a visual BC baseline from .pt episode demonstrations.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join("logs", "act", "Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0", "act_dataset"),
        help="Path to the dataset directory containing manifest.json and episodes/.",
    )
    parser.add_argument("--task", type=str, default="Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0", help="Task name.")
    parser.add_argument("--name", type=str, default="bc_default", help="Experiment name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-4, help="AdamW weight decay.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation episode split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--save_every", type=int, default=25, help="Checkpoint save interval in epochs.")
    parser.add_argument("--history_length", type=int, default=4, help="Number of low-dimensional observation frames to stack.")
    parser.add_argument("--gripper_loss_weight", type=float, default=5.0, help="Additional loss weight on the gripper action dimension.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {sub_key: sub_value.to(device) for sub_key, sub_value in value.items()}
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model: BehaviorCloningPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    non_image_keys: tuple[str, ...],
    stats: dict[str, dict[str, torch.Tensor]],
    gripper_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_steps = 0

    state_mean = stats["state"]["mean"].to(device)
    state_std = stats["state"]["std"].to(device)
    action_mean = stats["action"]["mean"].to(device)
    action_std = stats["action"]["std"].to(device)

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        obs = batch["obs"]
        state = flatten_state(obs, non_image_keys)
        norm_state = (state - state_mean) / state_std
        target_action = (batch["actions"][:, 0] - action_mean) / action_std
        pred_action = model(obs=obs, state=norm_state)

        action_weights = torch.ones((1, target_action.shape[-1]), device=device, dtype=target_action.dtype)
        action_weights[..., -1] = gripper_loss_weight
        l1_loss = (torch.abs(pred_action - target_action) * action_weights).sum() / action_weights.sum().clamp_min(1.0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            l1_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = target_action.shape[0]
        total_loss += float(l1_loss.item()) * batch_size
        total_steps += batch_size

    return {"loss": total_loss / max(total_steps, 1)}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    episode_dataset = ActEpisodeDataset(args.dataset)
    train_indices, val_indices = split_episode_indices(len(episode_dataset), args.val_ratio, args.seed)
    train_dataset = ActSequenceDataset(args.dataset, action_chunk_size=1, episode_indices=train_indices, history_length=args.history_length)
    val_dataset = (
        ActSequenceDataset(args.dataset, action_chunk_size=1, episode_indices=val_indices, history_length=args.history_length)
        if val_indices
        else None
    )

    metadata = train_dataset.infer_shapes()
    stats = train_dataset.compute_normalization_stats()
    non_image_keys = train_dataset.non_image_keys

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        if val_dataset is not None
        else None
    )

    model = BehaviorCloningPolicy(
        state_dim=metadata["state_dim"],
        action_dim=metadata["action_dim"],
        image_keys=metadata["image_keys"],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    exp_dir = os.path.abspath(os.path.join("logs", "bc", args.task, "runs", args.name))
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    config = {
        "dataset": os.path.abspath(args.dataset),
        "task": args.task,
        "name": args.name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "history_length": args.history_length,
        "gripper_loss_weight": args.gripper_loss_weight,
    }
    full_metadata = {
        **metadata,
        "non_image_keys": non_image_keys,
        "normalization_stats": {key: {sub_key: tensor.cpu() for sub_key, tensor in value.items()} for key, value in stats.items()},
        "train_episode_indices": train_indices,
        "val_episode_indices": val_indices,
    }

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "metadata": {**metadata, "non_image_keys": list(non_image_keys)}}, f, indent=2)

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pt")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, non_image_keys, stats, args.gripper_loss_weight)
        metric_line = f"Epoch {epoch:04d} | train_loss={train_metrics['loss']:.5f}"

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, device, non_image_keys, stats, args.gripper_loss_weight)
            metric_line += f" | val_loss={val_metrics['loss']:.5f}"
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(best_path, model, optimizer, epoch, config, full_metadata)
        else:
            save_checkpoint(best_path, model, optimizer, epoch, config, full_metadata)

        print(metric_line)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, config, full_metadata)


if __name__ == "__main__":
    main()
