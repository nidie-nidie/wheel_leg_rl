from __future__ import annotations

import torch


def validate_constraint_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def window_rms(values: torch.Tensor) -> torch.Tensor:
    validate_constraint_tensor(values, "window values")
    if values.numel() == 0:
        raise ValueError("window values cannot be empty")
    return values.square().mean().sqrt()


def excess_over_budget(rms: torch.Tensor, budget: float) -> torch.Tensor:
    validate_constraint_tensor(rms, "window RMS")
    if not torch.isfinite(torch.as_tensor(budget)):
        raise ValueError("budget must be finite")
    return (rms - float(budget)).clamp_min(0.0)


def diagonal_gaussian_kl(
    mean: torch.Tensor,
    std: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_std: torch.Tensor,
) -> torch.Tensor:
    for name, value in (
        ("mean", mean),
        ("std", std),
        ("reference_mean", reference_mean),
        ("reference_std", reference_std),
    ):
        validate_constraint_tensor(value, name)
    if mean.shape != std.shape or mean.shape != reference_mean.shape or mean.shape != reference_std.shape:
        raise ValueError("Gaussian tensors must have identical shapes")
    safe_std = std.clamp_min(1.0e-6)
    safe_reference_std = reference_std.clamp_min(1.0e-6)
    ratio = safe_std.square() + (mean - reference_mean).square()
    result = 0.5 * (
        ratio / safe_reference_std.square()
        - 1.0
        + 2.0 * (safe_reference_std.log() - safe_std.log())
    ).sum(dim=-1)
    validate_constraint_tensor(result, "Gaussian KL")
    return result

