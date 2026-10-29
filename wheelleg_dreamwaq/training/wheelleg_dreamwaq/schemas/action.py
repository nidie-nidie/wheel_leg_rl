from __future__ import annotations

import torch


ACTION_SCHEMA_VERSION = "ActionV1"
ACTION_DIM = 6
CANONICAL_JOINT_ORDER = (
    "jIJ",
    "jIO",
    "jAB",
    "jAG",
    "jwheel_left",
    "jwheel_right",
)
LEG_JOINT_ORDER = CANONICAL_JOINT_ORDER[:4]
WHEEL_JOINT_ORDER = CANONICAL_JOINT_ORDER[4:]

PACKET_LEG_ORDER = ("jIO", "jAG", "jIJ", "jAB")
POLICY_TO_PACKET_LEG = (1, 3, 0, 2)
PACKET_TO_POLICY_LEG = (2, 0, 3, 1)
WHEEL_JOINT_SIGN_USD = (1.0, -1.0)
CONTROLLED_JOINT_SIGN_USD = (1.0, 1.0, 1.0, 1.0, *WHEEL_JOINT_SIGN_USD)


def _select_last_dim(values: torch.Tensor, order: tuple[int, ...]) -> torch.Tensor:
    if values.shape[-1] != len(order):
        raise ValueError(f"Expected last dimension {len(order)}, got {values.shape[-1]}")
    indices = torch.tensor(order, dtype=torch.long, device=values.device)
    return values.index_select(-1, indices)


def policy_to_packet_legs(values: torch.Tensor) -> torch.Tensor:
    return _select_last_dim(values, POLICY_TO_PACKET_LEG)


def packet_to_policy_legs(values: torch.Tensor) -> torch.Tensor:
    return _select_last_dim(values, PACKET_TO_POLICY_LEG)


def wheel_targets_control_to_usd(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 2:
        raise ValueError(f"Expected two wheel targets, got {values.shape[-1]}")
    signs = values.new_tensor(WHEEL_JOINT_SIGN_USD)
    return values * signs


def wheel_feedback_usd_to_control(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 2:
        raise ValueError(f"Expected two wheel feedback values, got {values.shape[-1]}")
    signs = values.new_tensor(WHEEL_JOINT_SIGN_USD)
    return values * signs


def controlled_joint_feedback_usd_to_control(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} controlled joint values, got {values.shape[-1]}")
    signs = values.new_tensor(CONTROLLED_JOINT_SIGN_USD)
    return values * signs


def clip_action(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected ActionV1 dimension {ACTION_DIM}, got {values.shape[-1]}")
    return torch.clamp(values, -1.0, 1.0)
