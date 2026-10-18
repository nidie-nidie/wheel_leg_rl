from __future__ import annotations

import pytest
import torch

from debug.sim2sim.constraint_finetune_math import (
    diagonal_gaussian_kl,
    excess_over_budget,
    validate_constraint_tensor,
    window_rms,
)


def test_window_rms_and_excess() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    assert torch.allclose(window_rms(values), torch.tensor(2.5))
    assert torch.allclose(
        excess_over_budget(torch.tensor([0.2, 0.1]), 0.15),
        torch.tensor([0.05, 0.0]),
    )


def test_gaussian_kl_zero_for_identical_distribution() -> None:
    mean = torch.zeros(2, 6)
    std = torch.ones(2, 6)
    assert torch.allclose(diagonal_gaussian_kl(mean, std, mean, std), torch.zeros(2))


def test_nonfinite_is_rejected() -> None:
    with pytest.raises(FloatingPointError):
        validate_constraint_tensor(torch.tensor([float("nan")]), "metric")




