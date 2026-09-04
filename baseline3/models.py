from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn


class GeneratorBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.net(values))


class ConditionalGenerator(nn.Module):
    """Generate one joint [Electricity, Heat, Cooling, PV] daily trajectory."""

    def __init__(
        self,
        condition_channels: int,
        target_channels: int = 4,
        seq_len: int = 24,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        total_conditions = condition_channels + 1  # PV year trajectory.
        self.condition_projection = nn.Conv1d(total_conditions, hidden_dim, 1)
        self.noise_projection = nn.Linear(latent_dim, hidden_dim * seq_len)
        self.position = nn.Parameter(torch.zeros(1, hidden_dim, seq_len))
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *(GeneratorBlock(hidden_dim) for _ in range(num_blocks))
        )
        self.output = nn.Conv1d(hidden_dim, target_channels, 1)

    def forward(
        self,
        noise: torch.Tensor,
        conditions: torch.Tensor,
        pv_year: torch.Tensor,
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(f"noise must be [B,{self.latent_dim}]")
        condition = torch.cat([conditions, pv_year], dim=1)
        condition_hidden = self.condition_projection(condition) + self.position
        noise_hidden = self.noise_projection(noise).view(
            noise.shape[0], -1, self.seq_len
        )
        hidden = self.fusion(torch.cat([condition_hidden, noise_hidden], dim=1))
        return self.output(self.blocks(hidden))


class ConditionalCritic(nn.Module):
    """Joint conditional critic without sigmoid or batch normalization."""

    def __init__(
        self,
        condition_channels: int,
        target_channels: int = 4,
        seq_len: int = 24,
        hidden_dim: int = 128,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        input_channels = target_channels + condition_channels + 1
        layers: list[nn.Module] = [
            nn.Conv1d(input_channels, hidden_dim, 1),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_blocks):
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.position = nn.Parameter(torch.zeros(1, hidden_dim, seq_len))
        self.score = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        trajectories: torch.Tensor,
        conditions: torch.Tensor,
        pv_year: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.cat([trajectories, conditions, pv_year], dim=1)
        return self.score(self.features(values) + self.position).reshape(-1)


def gradient_penalty(
    critic: ConditionalCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    conditions: torch.Tensor,
    pv_year: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device, dtype=real.dtype)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated, conditions, pv_year)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norms = gradients.flatten(1).norm(2, dim=1)
    return ((norms - 1.0) ** 2).mean()


def build_models(
    config: dict[str, Any], condition_channels: int
) -> tuple[ConditionalGenerator, ConditionalCritic]:
    model = config["model"]
    common = {
        "condition_channels": condition_channels,
        "target_channels": 4,
        "seq_len": int(config["data"]["seq_len"]),
        "hidden_dim": int(model["hidden_dim"]),
        "num_blocks": int(model["num_blocks"]),
    }
    generator = ConditionalGenerator(
        **common,
        latent_dim=int(model["latent_dim"]),
    )
    critic = ConditionalCritic(**common)
    return generator, critic


def checkpoint_payload(
    config: dict[str, Any],
    generator: nn.Module,
    critic: nn.Module,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "conditional-wgan-gp-joint-baseline3",
        "epoch": epoch,
        "generator": deepcopy(generator.state_dict()),
        "critic": deepcopy(critic.state_dict()),
        "config": deepcopy(config),
        "metrics": deepcopy(metrics),
    }
