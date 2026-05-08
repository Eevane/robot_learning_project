# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple visual behavior cloning policy on top of ACT dataset format."""

from __future__ import annotations

import torch
from torch import nn

try:
    from .act_policy import ImageEncoder, flatten_state
except ImportError:
    from act_policy import ImageEncoder, flatten_state


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_keys: tuple[str, ...],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_keys = image_keys
        self.hidden_dim = hidden_dim

        self.state_encoder = _build_mlp(state_dim, hidden_dim, hidden_dim)
        self.image_encoders = nn.ModuleDict({key: ImageEncoder(hidden_dim) for key in image_keys})
        fusion_dim = hidden_dim * (1 + len(image_keys))
        self.policy_head = _build_mlp(fusion_dim, hidden_dim * 2, action_dim)

    def forward(self, obs: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        features = [self.state_encoder(state)]
        for key in self.image_keys:
            features.append(self.image_encoders[key](obs[key]))
        fused = torch.cat(features, dim=-1)
        return self.policy_head(fused)


__all__ = ["BehaviorCloningPolicy", "flatten_state"]
