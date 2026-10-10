from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from wheelleg_dreamwaq.schemas.observation import (
    HISTORY_OBS_DIM,
    PROPRIO_RECONSTRUCTION_TARGET_DIM,
    VELOCITY_TARGET_DIM,
)


CONTEXT_DIM = 16
CENET_OUTPUT_DIM = VELOCITY_TARGET_DIM + 2 * CONTEXT_DIM
DECODER_INPUT_DIM = CONTEXT_DIM + VELOCITY_TARGET_DIM + 6


class CENetEncoding(NamedTuple):
    estimated_velocity: torch.Tensor
    context_mu: torch.Tensor
    context_logvar: torch.Tensor


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class CENet(nn.Module):
    """Feed-forward context-aided estimator frozen by DreamWaQAlgorithmContractV1."""

    def __init__(
        self,
        *,
        history_dim: int = HISTORY_OBS_DIM,
        velocity_dim: int = VELOCITY_TARGET_DIM,
        context_dim: int = CONTEXT_DIM,
        reconstruction_target_dim: int = PROPRIO_RECONSTRUCTION_TARGET_DIM,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
    ) -> None:
        super().__init__()
        if history_dim != HISTORY_OBS_DIM:
            raise ValueError(f"history_dim must be {HISTORY_OBS_DIM}, got {history_dim}")
        if velocity_dim != VELOCITY_TARGET_DIM:
            raise ValueError(f"velocity_dim must be {VELOCITY_TARGET_DIM}, got {velocity_dim}")
        if context_dim != CONTEXT_DIM:
            raise ValueError(f"context_dim must be {CONTEXT_DIM}, got {context_dim}")
        if reconstruction_target_dim != PROPRIO_RECONSTRUCTION_TARGET_DIM:
            raise ValueError(
                f"reconstruction_target_dim must be {PROPRIO_RECONSTRUCTION_TARGET_DIM}, "
                f"got {reconstruction_target_dim}"
            )
        if tuple(encoder_hidden_dims) != (128, 64):
            raise ValueError(f"encoder hidden dimensions must be [128, 64], got {encoder_hidden_dims}")
        if tuple(decoder_hidden_dims) != (64, 128):
            raise ValueError(f"decoder hidden dimensions must be [64, 128], got {decoder_hidden_dims}")
        self.history_dim = history_dim
        self.velocity_dim = velocity_dim
        self.context_dim = context_dim
        self.reconstruction_target_dim = reconstruction_target_dim
        self.encoder = _mlp(history_dim, tuple(encoder_hidden_dims), CENET_OUTPUT_DIM)
        self.decoder = _mlp(DECODER_INPUT_DIM, tuple(decoder_hidden_dims), reconstruction_target_dim)

    def encode(self, history: torch.Tensor) -> CENetEncoding:
        if history.ndim != 2 or history.shape[-1] != self.history_dim:
            raise ValueError(f"history must have shape [B,{self.history_dim}], got {tuple(history.shape)}")
        if history.dtype != torch.float32:
            raise TypeError(f"history must be float32, got {history.dtype}")
        encoded = self.encoder(history)
        return CENetEncoding(
            encoded[:, : self.velocity_dim],
            encoded[:, self.velocity_dim : self.velocity_dim + self.context_dim],
            encoded[:, self.velocity_dim + self.context_dim :],
        )

    def infer(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoding = self.encode(history)
        return encoding.estimated_velocity, encoding.context_mu

    def sample_context(
        self,
        context_mu: torch.Tensor,
        context_logvar: torch.Tensor,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_mu.shape != context_logvar.shape or context_mu.ndim != 2:
            raise ValueError("context_mu and context_logvar must have identical rank-2 shapes")
        if context_mu.shape[-1] != self.context_dim:
            raise ValueError(f"context tensors must have last dimension {self.context_dim}")
        if epsilon is None:
            epsilon = torch.randn_like(context_mu)
        elif epsilon.shape != context_mu.shape or epsilon.dtype != context_mu.dtype:
            raise ValueError("epsilon must match context_mu shape and dtype")
        return context_mu + torch.exp(0.5 * context_logvar) * epsilon

    def decode(
        self,
        context_z: torch.Tensor,
        estimated_velocity: torch.Tensor,
        clipped_action: torch.Tensor,
    ) -> torch.Tensor:
        if context_z.ndim != 2 or context_z.shape[-1] != self.context_dim:
            raise ValueError(f"context_z must have shape [B,{self.context_dim}]")
        if estimated_velocity.ndim != 2 or estimated_velocity.shape[-1] != self.velocity_dim:
            raise ValueError(f"estimated_velocity must have shape [B,{self.velocity_dim}]")
        if clipped_action.ndim != 2 or clipped_action.shape[-1] != 6:
            raise ValueError("clipped_action must have shape [B,6]")
        decoder_input = torch.cat(
            (context_z, estimated_velocity.detach(), clipped_action.detach()),
            dim=-1,
        )
        return self.decoder(decoder_input)

