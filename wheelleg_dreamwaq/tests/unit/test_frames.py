from __future__ import annotations

import math

import torch

from wheelleg_dreamwaq.schemas.frames import (
    R_CONTROL_FROM_USD,
    projected_gravity_from_quaternion,
    transform_usd_vector_to_control,
    quat_rotate_wxyz,
    upright_cosine_from_projected_gravity,
    validate_rotation_matrix,
)


def test_control_rotation_is_proper_orthogonal() -> None:
    validate_rotation_matrix(R_CONTROL_FROM_USD)
    identity = R_CONTROL_FROM_USD @ R_CONTROL_FROM_USD.T
    assert torch.allclose(identity, torch.eye(3), atol=1.0e-7)
    assert torch.isclose(torch.linalg.det(R_CONTROL_FROM_USD), torch.tensor(1.0))


def test_usd_to_control_axis_mapping() -> None:
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = transform_usd_vector_to_control(vectors)
    expected = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert torch.allclose(result, expected)


def test_identity_quaternion_has_downward_projected_gravity() -> None:
    quat_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    projected = projected_gravity_from_quaternion(quat_wxyz)
    assert torch.allclose(projected, torch.tensor([[0.0, 0.0, -1.0]]), atol=1.0e-6)


def test_positive_roll_rotates_gravity_into_negative_y() -> None:
    half = 0.5 * math.pi / 2.0
    quat_wxyz = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    projected = projected_gravity_from_quaternion(quat_wxyz)
    assert torch.allclose(projected, torch.tensor([[0.0, -1.0, 0.0]]), atol=1.0e-6)


def test_forward_quaternion_rotation_matches_positive_roll() -> None:
    half = 0.5 * math.pi / 2.0
    quat_wxyz = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    rotated = quat_rotate_wxyz(quat_wxyz, torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(rotated, torch.tensor([[0.0, 0.0, 1.0]]), atol=1.0e-6)


def test_upright_cosine_distinguishes_upright_sideways_and_inverted() -> None:
    projected = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.equal(upright_cosine_from_projected_gravity(projected), torch.tensor([1.0, 0.0, -1.0]))
