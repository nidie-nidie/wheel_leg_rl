from __future__ import annotations

import pytest
import torch

from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import (
    COMBINED_MODE,
    ROTATE_MODE,
    STAND_MODE,
    STRAIGHT_MODE,
    CommandRanges,
    sample_command_batch,
)


def test_command_sampling_v2_mode_masks_and_ranges() -> None:
    ranges = CommandRanges()
    generator = torch.Generator().manual_seed(20261004)
    commands, modes = sample_command_batch(
        20_000,
        device="cpu",
        ranges=ranges,
        generator=generator,
    )

    assert set(modes.tolist()) == {STAND_MODE, STRAIGHT_MODE, ROTATE_MODE, COMBINED_MODE}
    assert torch.count_nonzero(commands[modes == STAND_MODE, :2]) == 0
    assert torch.count_nonzero(commands[modes == STRAIGHT_MODE, 1]) == 0
    assert torch.count_nonzero(commands[modes == ROTATE_MODE, 0]) == 0
    assert torch.all((commands[:, 0] >= ranges.vx[0]) & (commands[:, 0] <= ranges.vx[1]))
    assert torch.all(
        (commands[:, 1] >= ranges.yaw_rate[0]) & (commands[:, 1] <= ranges.yaw_rate[1])
    )
    assert torch.all(
        (commands[:, 2] >= ranges.base_height[0])
        & (commands[:, 2] <= ranges.base_height[1])
    )


def test_command_sampling_v2_mode_probabilities_are_close_to_contract() -> None:
    ranges = CommandRanges()
    generator = torch.Generator().manual_seed(7)
    _, modes = sample_command_batch(100_000, device="cpu", ranges=ranges, generator=generator)
    observed = torch.bincount(modes, minlength=4).float() / modes.numel()
    expected = torch.tensor(ranges.mode_probabilities)
    assert torch.all(torch.abs(observed - expected) < 0.01)


def test_command_sampling_v2_contract_values() -> None:
    ranges = CommandRanges()
    assert ranges.vx == (-1.5, 1.5)
    assert ranges.yaw_rate == (-1.0, 1.0)
    assert ranges.base_height == (0.16, 0.24)
    assert ranges.mode_probabilities == (0.20, 0.30, 0.20, 0.30)
    assert ranges.hold_for_episode is True


def test_command_sampling_with_cpu_generator_is_reproducible() -> None:
    first_generator = torch.Generator(device="cpu").manual_seed(99)
    second_generator = torch.Generator(device="cpu").manual_seed(99)
    first = sample_command_batch(32, device="cpu", ranges=CommandRanges(), generator=first_generator)
    second = sample_command_batch(32, device="cpu", ranges=CommandRanges(), generator=second_generator)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_command_sampling_can_use_cpu_generator_for_cuda_output() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    commands, modes = sample_command_batch(
        16,
        device="cuda:0",
        ranges=CommandRanges(),
        generator=generator,
    )
    assert commands.device.type == "cuda"
    assert modes.device.type == "cuda"
