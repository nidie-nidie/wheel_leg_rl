"""Physical parameter contract for a PACE v0.1.2 A1 optimizer mean."""

from pathlib import Path
from typing import NamedTuple

import torch


A1_PACE_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)


class A1PaceMean(NamedTuple):
    """Named physical parameters in the PACE v0.1.2 A1 layout."""

    armature: dict[str, float]
    viscous_friction: dict[str, float]
    coulomb_friction: dict[str, float]
    encoder_bias: dict[str, float]
    delay_steps: int


def _validate_bounds(
    name: str, values: torch.Tensor, lower: float, upper: float
) -> None:
    if torch.any(values < lower) or torch.any(values > upper):
        raise ValueError(
            f"{name} is outside frozen physical bounds [{lower}, {upper}]"
        )


def _named(values: torch.Tensor) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            A1_PACE_JOINT_ORDER, values.tolist(), strict=True
        )
    }


def parse_a1_mean(mean: torch.Tensor) -> A1PaceMean:
    """Parse exactly one physical 49-value PACE v0.1.2 A1 mean."""

    if not isinstance(mean, torch.Tensor):
        raise TypeError("A1 PACE mean must be a torch.Tensor")
    if mean.ndim != 1 or mean.shape != (49,):
        raise ValueError(
            f"A1 PACE mean must have shape (49,), got {tuple(mean.shape)}"
        )
    if not torch.is_floating_point(mean):
        raise TypeError("A1 PACE mean must use a floating dtype")
    if not torch.isfinite(mean).all():
        raise ValueError("A1 PACE mean must contain only finite values")

    physical = mean.detach().to(device="cpu")
    _validate_bounds("armature", physical[:12], 1.0e-5, 1.0)
    _validate_bounds("viscous friction", physical[12:24], 0.0, 7.0)
    _validate_bounds("Coulomb friction", physical[24:36], 0.0, 0.5)
    _validate_bounds("encoder bias", physical[36:48], -0.1, 0.1)
    _validate_bounds("global delay", physical[48:49], 0.0, 10.0)

    return A1PaceMean(
        armature=_named(physical[:12]),
        viscous_friction=_named(physical[12:24]),
        coulomb_friction=_named(physical[24:36]),
        encoder_bias=_named(physical[36:48]),
        delay_steps=int(physical[48].to(torch.int).item()),
    )


def load_a1_mean(path: Path | str) -> A1PaceMean:
    """Load and validate a physical A1 mean on the CPU."""

    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError("A1 PACE mean file must contain one torch.Tensor")
    return parse_a1_mean(value)
