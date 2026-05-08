# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ACT-style policy with image encoders, transformer decoder, and CVAE latent."""

from __future__ import annotations

import math

import torch
from torch import nn


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


class ActPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_size: int,
        image_keys: tuple[str, ...],
        hidden_dim: int = 256,
        latent_dim: int = 32,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        auxiliary_object_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.image_keys = image_keys
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.auxiliary_object_dim = auxiliary_object_dim

        self.state_encoder = _build_mlp(state_dim, hidden_dim, hidden_dim)
        self.image_encoders = nn.ModuleDict({key: ImageEncoder(hidden_dim) for key in image_keys})
        self.obs_type_embeddings = nn.Parameter(torch.randn(1 + len(image_keys), hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.obs_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.latent_cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.latent_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, num_encoder_layers // 2))
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(action_chunk_size, hidden_dim) * 0.02)
        self.output_head = nn.Linear(hidden_dim, action_dim)
        self.object_position_head = (
            _build_mlp(hidden_dim, hidden_dim, auxiliary_object_dim) if auxiliary_object_dim > 0 else None
        )

    def _build_obs_tokens(self, obs: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        tokens = [self.state_encoder(state) + self.obs_type_embeddings[0]]
        for idx, key in enumerate(self.image_keys, start=1):
            tokens.append(self.image_encoders[key](obs[key]) + self.obs_type_embeddings[idx])
        return torch.stack(tokens, dim=1)

    def _encode_latent(
        self,
        obs_tokens: torch.Tensor,
        action_chunk: torch.Tensor | None,
        action_is_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = obs_tokens.shape[0]
        if action_chunk is None:
            zeros = torch.zeros(batch_size, self.latent_dim, device=obs_tokens.device, dtype=obs_tokens.dtype)
            return zeros, zeros, zeros

        cls_token = self.latent_cls.expand(batch_size, -1, -1)
        action_tokens = self.action_encoder(action_chunk)
        tokens = torch.cat([cls_token, obs_tokens, action_tokens], dim=1)

        num_obs_tokens = obs_tokens.shape[1]
        padding = torch.zeros((batch_size, 1 + num_obs_tokens), dtype=torch.bool, device=obs_tokens.device)
        if action_is_valid is None:
            action_padding = torch.zeros(
                (batch_size, self.action_chunk_size), dtype=torch.bool, device=obs_tokens.device
            )
        else:
            action_padding = ~action_is_valid.bool()
        src_key_padding_mask = torch.cat([padding, action_padding], dim=1)

        encoded = self.latent_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        mu, logvar = self.latent_proj(encoded[:, 0]).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar

    def _decode(self, obs_memory: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch_size = obs_memory.shape[0]
        latent_token = self.latent_to_hidden(latent).unsqueeze(1)
        memory = torch.cat([latent_token, obs_memory], dim=1)
        query = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(tgt=query, memory=memory)
        return self.output_head(decoded)

    def _predict_object_position(self, obs_memory: torch.Tensor) -> torch.Tensor | None:
        if self.object_position_head is None:
            return None
        pooled_memory = obs_memory.mean(dim=1)
        return self.object_position_head(pooled_memory)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        state: torch.Tensor,
        action_chunk: torch.Tensor | None = None,
        action_is_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs_tokens = self._build_obs_tokens(obs=obs, state=state)
        obs_memory = self.obs_encoder(obs_tokens)
        latent, mu, logvar = self._encode_latent(
            obs_tokens=obs_memory,
            action_chunk=action_chunk,
            action_is_valid=action_is_valid,
        )
        pred_actions = self._decode(obs_memory=obs_memory, latent=latent)
        pred_object_position = self._predict_object_position(obs_memory=obs_memory)
        return {
            "pred_actions": pred_actions,
            "pred_object_position": pred_object_position,
            "latent_mu": mu,
            "latent_logvar": logvar,
        }

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu.square() - 1.0 - logvar, dim=-1))


def flatten_state(obs: dict[str, torch.Tensor], non_image_keys: tuple[str, ...]) -> torch.Tensor:
    flattened = []
    for key in non_image_keys:
        value = obs[key].float()
        if value.ndim == 3:
            flattened.append(value.reshape(value.shape[0], -1))
        else:
            flattened.append(value)
    return torch.cat(flattened, dim=-1)
