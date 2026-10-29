from __future__ import annotations

from dataclasses import asdict, dataclass

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
    practice_schedule: str = "disabled"


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

STOP_REVERSE_COMMAND_PRACTICE_VERSION = "StopReverseCommandPracticeV1"


def command_contract_payload(ranges: CommandRanges) -> dict:
    payload = asdict(ranges)
    if ranges.practice_schedule == "disabled":
        if ranges.hold_for_episode is not True:
            raise ValueError("Disabled command practice requires hold_for_episode=True")
        payload["practice_contract"] = None
    elif ranges.practice_schedule == STOP_REVERSE_COMMAND_PRACTICE_VERSION:
        if ranges.hold_for_episode is not False:
            raise ValueError("Stop/reverse command practice requires hold_for_episode=False")
        payload["practice_contract"] = {
            "version": STOP_REVERSE_COMMAND_PRACTICE_VERSION,
            "stage_end_control_steps": [100, 150, 250, 300, 400, 500],
            "factors": [1, 0, -1, 0, 1, 0],
            "control_dt_s": 0.02,
            "clock": "real_completed_control_steps_per_environment",
            "update_timing": "after_reward_before_next_observation",
            "height_behavior": "hold_reset_sample",
            "final_phase_behavior": "hold_last_factor",
        }
    else:
        raise ValueError(f"Unknown command practice schedule: {ranges.practice_schedule}")
    return payload


def command_practice_factor(completed_steps: torch.Tensor) -> torch.Tensor:
    if completed_steps.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Command practice ticks must be integers")
    if bool((completed_steps < 0).any().item()):
        raise ValueError("Command practice ticks must be non-negative")
    boundaries = completed_steps.new_tensor([100, 150, 250, 300, 400], dtype=torch.long)
    indices = torch.bucketize(completed_steps.to(dtype=torch.long), boundaries, right=True)
    factors = torch.tensor([1, 0, -1, 0, 1, 0], dtype=torch.float32, device=completed_steps.device)
    return factors[indices]


class CommandPracticeState:
    """Only command state; no simulator, RNG, episode-length or history dependency."""

    def __init__(self, num_envs: int, device: str | torch.device):
        self.initial_commands = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor, commands: torch.Tensor) -> None:
        if env_ids.ndim != 1 or commands.shape != (env_ids.numel(), 3):
            raise ValueError("Command practice reset shape mismatch")
        if not bool(torch.isfinite(commands).all().item()):
            raise ValueError("Command practice reset commands must be finite")
        self.initial_commands[env_ids] = commands
        self.steps[env_ids] = 0

    def current_commands(self) -> torch.Tensor:
        commands = self.initial_commands.clone()
        commands[:, :2] *= command_practice_factor(self.steps).unsqueeze(-1)
        return commands

    def advance(self) -> torch.Tensor:
        self.steps += 1
        return self.current_commands()
