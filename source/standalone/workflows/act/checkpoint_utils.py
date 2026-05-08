# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint helpers for ACT workflows."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

try:
    from .act_policy import ActPolicy
except ImportError:
    from act_policy import ActPolicy


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def save_checkpoint(
    checkpoint_path: str,
    model: ActPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metadata": metadata,
        },
        checkpoint_path,
    )

    with open(os.path.splitext(checkpoint_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "config": _to_jsonable(config), "metadata": _to_jsonable(metadata)}, f, indent=2)


def load_checkpoint(checkpoint_path: str, device: torch.device | str) -> tuple[ActPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint["metadata"]
    config = checkpoint["config"]

    model = ActPolicy(
        state_dim=metadata["state_dim"],
        action_dim=metadata["action_dim"],
        action_chunk_size=config["action_chunk_size"],
        image_keys=tuple(metadata["image_keys"]),
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        auxiliary_object_dim=int(config.get("auxiliary_object_dim", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
