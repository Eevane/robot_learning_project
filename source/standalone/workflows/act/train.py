# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an ACT-style policy on per-episode .pt demonstrations."""

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
    from .act_policy import ActPolicy, flatten_state
    from .checkpoint_utils import save_checkpoint
except ImportError:
    from act_dataset import ActEpisodeDataset, ActSequenceDataset, split_episode_indices
    from act_policy import ActPolicy, flatten_state
    from checkpoint_utils import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ACT-style policy from .pt episode demonstrations.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join("logs", "act", "Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0", "act_dataset"),
        help="Path to the dataset directory containing manifest.json and episodes/.",
    )
    parser.add_argument("--task", type=str, default="Isaac-Lift-Needle-PSM-IK-Abs-Vision-v0", help="Task name.")
    parser.add_argument("--name", type=str, default="default", help="Experiment name.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1.0e-4, help="AdamW weight decay.")
    parser.add_argument("--action_chunk_size", type=int, default=16, help="Number of future actions to predict.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Transformer hidden size.")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent CVAE dimension.")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Transformer encoder layers.")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Transformer decoder layers.")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout.")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL loss weight.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation episode split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--save_every", type=int, default=25, help="Checkpoint save interval in epochs.")
    parser.add_argument("--history_length", type=int, default=4, help="Number of low-dimensional observation frames to stack.")
    parser.add_argument(
        "--exclude_low_dim_keys",
        nargs="*",
        default=(),
        help="Low-dimensional observation keys to exclude from the policy input.",
    )
    parser.add_argument(
        "--gripper_loss_weight",
        type=float,
        default=1.0,
        help="Additional loss weight applied to the last action dimension (gripper).",
    )
    parser.add_argument(
        "--aux_object_key",
        type=str,
        default=None,
        help="Observation key used as auxiliary object-position supervision. Disabled when omitted.",
    )
    parser.add_argument(
        "--aux_object_loss_weight",
        type=float,
        default=0.0,
        help="Weight for the auxiliary object-position prediction loss.",
    )
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
    model: ActPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    non_image_keys: tuple[str, ...],
    stats: dict[str, dict[str, torch.Tensor]],
    kl_weight: float,
    gripper_loss_weight: float,
    aux_object_key: str | None,
    aux_object_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    total_aux = 0.0
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
        target_actions = (batch["actions"] - action_mean.view(1, 1, -1)) / action_std.view(1, 1, -1)

        output = model(
            obs=obs,
            state=norm_state,
            action_chunk=target_actions,
            action_is_valid=batch["action_is_valid"],
        )
        pred_actions = output["pred_actions"]

        valid_mask = batch["action_is_valid"].unsqueeze(-1).float()
        action_weights = torch.ones((1, 1, target_actions.shape[-1]), device=device, dtype=target_actions.dtype)
        action_weights[..., -1] = gripper_loss_weight
        weighted_mask = valid_mask * action_weights
        denom = weighted_mask.sum().clamp_min(1.0)
        l1_loss = (torch.abs(pred_actions - target_actions) * weighted_mask).sum() / denom
        kl_loss = model.kl_divergence(output["latent_mu"], output["latent_logvar"])
        aux_loss = torch.zeros((), device=device, dtype=target_actions.dtype)
        if aux_object_key is not None and aux_object_loss_weight > 0.0:
            if aux_object_key not in obs:
                raise KeyError(f"Auxiliary object key '{aux_object_key}' not found in batch observations.")
            aux_target = obs[aux_object_key].float()
            if aux_target.ndim == 3:
                aux_target = aux_target[:, -1, :]
            pred_object_position = output["pred_object_position"]
            if pred_object_position is None:
                raise RuntimeError("Model did not return auxiliary object-position predictions.")
            aux_loss = torch.mean(torch.abs(pred_object_position - aux_target))
        loss = l1_loss + kl_weight * kl_loss + aux_object_loss_weight * aux_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = batch["actions"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_l1 += float(l1_loss.item()) * batch_size
        total_kl += float(kl_loss.item()) * batch_size
        total_aux += float(aux_loss.item()) * batch_size
        total_steps += batch_size

    return {
        "loss": total_loss / max(total_steps, 1),
        "l1_loss": total_l1 / max(total_steps, 1),
        "kl_loss": total_kl / max(total_steps, 1),
        "aux_loss": total_aux / max(total_steps, 1),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    episode_dataset = ActEpisodeDataset(args.dataset)
    train_indices, val_indices = split_episode_indices(
        num_episodes=len(episode_dataset),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_dataset = ActSequenceDataset(
        args.dataset,
        args.action_chunk_size,
        episode_indices=train_indices,
        history_length=args.history_length,
        exclude_obs_keys=tuple(args.exclude_low_dim_keys),
    )
    val_dataset = (
        ActSequenceDataset(
            args.dataset,
            args.action_chunk_size,
            episode_indices=val_indices,
            history_length=args.history_length,
            exclude_obs_keys=tuple(args.exclude_low_dim_keys),
        )
        if val_indices
        else None
    )

    metadata = train_dataset.infer_shapes()
    stats = train_dataset.compute_normalization_stats()
    non_image_keys = train_dataset.non_image_keys
    auxiliary_object_dim = 0
    if args.aux_object_key is not None and args.aux_object_loss_weight > 0.0:
        first_obs = train_dataset.episodes[0]["obs"]
        if args.aux_object_key not in first_obs:
            raise KeyError(
                f"Auxiliary object key '{args.aux_object_key}' is not present in dataset observations."
            )
        auxiliary_object_dim = int(first_obs[args.aux_object_key].shape[-1])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        if val_dataset is not None
        else None
    )

    model = ActPolicy(
        state_dim=metadata["state_dim"],
        action_dim=metadata["action_dim"],
        action_chunk_size=args.action_chunk_size,
        image_keys=metadata["image_keys"],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        auxiliary_object_dim=auxiliary_object_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    exp_dir = os.path.abspath(os.path.join("logs", "act", args.task, "runs", args.name))
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
        "action_chunk_size": args.action_chunk_size,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "kl_weight": args.kl_weight,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "history_length": args.history_length,
        "exclude_low_dim_keys": list(args.exclude_low_dim_keys),
        "gripper_loss_weight": args.gripper_loss_weight,
        "aux_object_key": args.aux_object_key,
        "aux_object_loss_weight": args.aux_object_loss_weight,
        "auxiliary_object_dim": auxiliary_object_dim,
    }
    full_metadata = {
        **metadata,
        "non_image_keys": non_image_keys,
        "normalization_stats": {
            key: {sub_key: tensor.cpu() for sub_key, tensor in value.items()} for key, value in stats.items()
        },
        "train_episode_indices": train_indices,
        "val_episode_indices": val_indices,
    }

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "metadata": {**metadata, "non_image_keys": list(non_image_keys)}}, f, indent=2)

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            non_image_keys=non_image_keys,
            stats=stats,
            kl_weight=args.kl_weight,
            gripper_loss_weight=args.gripper_loss_weight,
            aux_object_key=args.aux_object_key,
            aux_object_loss_weight=args.aux_object_loss_weight,
        )
        metric_line = (
            f"Epoch {epoch:04d} | train_loss={train_metrics['loss']:.5f} "
            f"| train_l1={train_metrics['l1_loss']:.5f} | train_kl={train_metrics['kl_loss']:.5f}"
            f" | train_aux={train_metrics['aux_loss']:.5f}"
        )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    non_image_keys=non_image_keys,
                    stats=stats,
                    kl_weight=args.kl_weight,
                    gripper_loss_weight=args.gripper_loss_weight,
                    aux_object_key=args.aux_object_key,
                    aux_object_loss_weight=args.aux_object_loss_weight,
                )
            metric_line += (
                f" | val_loss={val_metrics['loss']:.5f} "
                f"| val_l1={val_metrics['l1_loss']:.5f} | val_kl={val_metrics['kl_loss']:.5f}"
                f" | val_aux={val_metrics['aux_loss']:.5f}"
            )
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
