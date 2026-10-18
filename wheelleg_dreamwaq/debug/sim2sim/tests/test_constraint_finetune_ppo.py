from __future__ import annotations

import torch

from debug.sim2sim.constraint_finetune_math import diagonal_gaussian_kl
from debug.sim2sim.constraint_finetune_ppo import ConstrainedDreamWaQPPO


def test_reference_kl_is_zero_for_identical_distributions() -> None:
    mean = torch.zeros(2, 6)
    std = torch.ones(2, 6)
    value = ConstrainedDreamWaQPPO.reference_kl_from_distributions(mean, std, mean, std)
    assert torch.allclose(value, torch.zeros(2), atol=1.0e-7)
    assert torch.allclose(value, diagonal_gaussian_kl(mean, std, mean, std))


def test_reference_kl_detects_mean_drift() -> None:
    mean = torch.zeros(1, 6)
    reference_mean = torch.ones(1, 6)
    value = ConstrainedDreamWaQPPO.reference_kl_from_distributions(
        mean, torch.ones_like(mean), reference_mean, torch.ones_like(mean)
    )
    assert torch.all(value > 0.0)

