from __future__ import annotations

from dataclasses import dataclass

import torch


COMMAND_SAMPLING_VERSION = "CommandSamplingV2"
STAND_MODE = 0
STRAIGHT_MODE = 1
ROTATE_MODE = 2
COMBINED_MODE = 3


@dataclass(frozen=True)
class CommandRanges:
    vx: tuple[float, float] = (-1.5, 1.5)
    yaw_rate: tuple[float, float] = (-1.0, 1.0)
    base_height: tuple[float, float] = (0.16, 0.24)
    mode_probabilities: tuple[float, float, float, float] = (0.20, 0.30, 0.20, 0.30)
    hold_for_episode: bool = True


def sample_command_batch(
    count: int,
    *,
    device: str | torch.device,
    ranges: CommandRanges,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count < 0:
        raise ValueError("count must be non-negative")
    probabilities = torch.tensor(ranges.mode_probabilities, dtype=torch.float64)
    if probabilities.shape != (4,) or torch.any(probabilities < 0.0):
        raise ValueError("mode_probabilities must contain four non-negative values")
    if not torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=probabilities.dtype)):
        raise ValueError("mode_probabilities must sum to one")

    output_device = torch.device(device)
    sampling_device = torch.device(generator.device) if generator is not None else output_device
    unit = torch.rand((count, 3), device=sampling_device, generator=generator)
    low = unit.new_tensor((ranges.vx[0], ranges.yaw_rate[0], ranges.base_height[0]))
    high = unit.new_tensor((ranges.vx[1], ranges.yaw_rate[1], ranges.base_height[1]))
    commands = low + unit * (high - low)

    mode_unit = torch.rand(count, device=sampling_device, generator=generator)
    thresholds = torch.cumsum(mode_unit.new_tensor(ranges.mode_probabilities), dim=0)
    modes = torch.bucketize(mode_unit, thresholds[:-1])
    commands[modes == STAND_MODE, :2] = 0.0
    commands[modes == STRAIGHT_MODE, 1] = 0.0
    commands[modes == ROTATE_MODE, 0] = 0.0
    return commands.to(output_device), modes.to(output_device)


def sample_commands(
    count: int,
    *,
    device: str | torch.device,
    ranges: CommandRanges,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return sample_command_batch(count, device=device, ranges=ranges, generator=generator)[0]
