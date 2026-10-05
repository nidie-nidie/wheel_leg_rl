from __future__ import annotations

import torch


CONTROL_FRAME_VERSION = "ControlFrameV1"
R_CONTROL_FROM_USD = torch.tensor(
    (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=torch.float32,
)
R_CONTROL_FROM_IMU = torch.eye(3, dtype=torch.float32)


def validate_rotation_matrix(matrix: torch.Tensor, atol: float = 1.0e-6) -> None:
    if matrix.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {tuple(matrix.shape)}")
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    if not torch.allclose(matrix @ matrix.T, identity, atol=atol):
        raise ValueError("Rotation matrix is not orthogonal")
    if not torch.allclose(torch.linalg.det(matrix), matrix.new_tensor(1.0), atol=atol):
        raise ValueError("Rotation matrix determinant must be +1")


def transform_usd_vector_to_control(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 3:
        raise ValueError(f"Expected 3D vectors, got last dimension {values.shape[-1]}")
    rotation = R_CONTROL_FROM_USD.to(dtype=values.dtype, device=values.device)
    return values @ rotation.T


def quat_rotate_inverse_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("Expected quaternion [...,4] and vector [...,3]")
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    first_cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector - 2.0 * scalar * first_cross + 2.0 * torch.linalg.cross(xyz, first_cross, dim=-1)


def quat_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4 or vector.shape[-1] != 3:
        raise ValueError("Expected quaternion [...,4] and vector [...,3]")
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    first_cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * scalar * first_cross + 2.0 * torch.linalg.cross(xyz, first_cross, dim=-1)


def projected_gravity_from_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    world_down = torch.zeros((*quaternion_wxyz.shape[:-1], 3), dtype=quaternion_wxyz.dtype, device=quaternion_wxyz.device)
    world_down[..., 2] = -1.0
    return quat_rotate_inverse_wxyz(quaternion_wxyz, world_down)


def upright_cosine_from_projected_gravity(projected_gravity: torch.Tensor) -> torch.Tensor:
    if projected_gravity.shape[-1] != 3:
        raise ValueError(f"Expected projected gravity [...,3], got {projected_gravity.shape[-1]}")
    return torch.clamp(-projected_gravity[..., 2], -1.0, 1.0)
