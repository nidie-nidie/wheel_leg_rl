from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


EXPECTED_BODY_ORDER = (
    "base",
    "jIO",
    "jOP",
    "jwheel_left",
    "jIO_dummy_child_link1",
    "jIO_dummy_child_link2",
    "jAG",
    "jGH",
    "jwheel_right",
    "jAG_dummy_child_link1",
    "jAG_dummy_child_link2",
    "jIJ",
    "jJM",
    "jMK",
    "jKN",
    "jKN_dummy_child_link1",
    "jKN_dummy_child_link2",
    "jMK_dummy_child1",
    "jMK_dummy_child2",
    "jAB",
    "jBE",
    "jEC",
    "jCF",
    "jCF_dummy_child_link1",
    "jCF_dummy_child_link2",
    "jEC_dummy_child_link1",
    "jEC_dummy_child_link2",
)


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return array


@dataclass(frozen=True)
class BodyProperty:
    name: str
    mass: float
    com_position_world: np.ndarray
    inertia_com_body: np.ndarray
    rotation_world_from_body: np.ndarray
    com_linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Body name is required")
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("Body mass must be positive and finite")
        object.__setattr__(self, "com_position_world", _vector(self.com_position_world, 3, "COM"))
        object.__setattr__(self, "inertia_com_body", _matrix(self.inertia_com_body, (3, 3), "inertia"))
        rotation = _matrix(self.rotation_world_from_body, (3, 3), "rotation")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-10, rtol=0.0):
            raise ValueError("rotation_world_from_body is not orthonormal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("rotation_world_from_body must be right-handed")
        object.__setattr__(self, "rotation_world_from_body", rotation)
        object.__setattr__(
            self,
            "com_linear_velocity_world",
            _vector(self.com_linear_velocity_world, 3, "COM velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity_world",
            _vector(self.angular_velocity_world, 3, "angular velocity"),
        )


@dataclass(frozen=True)
class CompositeProperty:
    mass: float
    com_position_world: np.ndarray
    inertia_com_world: np.ndarray
    linear_momentum_world: np.ndarray
    angular_momentum_com_world: np.ndarray
    angular_momentum_origin_world: np.ndarray
    kinetic_energy: float


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = _vector(vector, 3, "skew vector")
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    direction = _vector(axis, 3, "axis")
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("Rotation axis cannot be zero")
    direction = direction / norm
    cross = skew(direction)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def inertia_world(body: BodyProperty) -> np.ndarray:
    rotation = body.rotation_world_from_body
    return rotation @ body.inertia_com_body @ rotation.T


def momentum_about_origin(
    body: BodyProperty, origin_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = _vector(origin_world, 3, "origin")
    linear = body.mass * body.com_linear_velocity_world
    angular_com = inertia_world(body) @ body.angular_velocity_world
    angular_origin = angular_com + np.cross(body.com_position_world - origin, linear)
    return linear, angular_com, angular_origin


def composite_properties(
    bodies: Iterable[BodyProperty], origin_world: np.ndarray | None = None
) -> CompositeProperty:
    items = tuple(bodies)
    if not items:
        raise ValueError("At least one body is required")
    origin = np.zeros(3) if origin_world is None else _vector(origin_world, 3, "origin")
    mass = float(sum(body.mass for body in items))
    com = sum((body.mass * body.com_position_world for body in items), np.zeros(3)) / mass
    inertia_composite = np.zeros((3, 3), dtype=np.float64)
    linear = np.zeros(3, dtype=np.float64)
    angular_com = np.zeros(3, dtype=np.float64)
    angular_origin = np.zeros(3, dtype=np.float64)
    kinetic = 0.0
    for body in items:
        offset = body.com_position_world - com
        body_inertia = inertia_world(body)
        inertia_composite += body_inertia + body.mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
        body_linear, body_angular_com, body_angular_origin = momentum_about_origin(body, origin)
        linear += body_linear
        angular_origin += body_angular_origin
        angular_com += body_angular_com + np.cross(offset, body_linear)
        kinetic += 0.5 * body.mass * float(np.dot(body.com_linear_velocity_world, body.com_linear_velocity_world))
        kinetic += 0.5 * float(body.angular_velocity_world @ body_inertia @ body.angular_velocity_world)
    return CompositeProperty(
        mass=mass,
        com_position_world=com,
        inertia_com_world=inertia_composite,
        linear_momentum_world=linear,
        angular_momentum_com_world=angular_com,
        angular_momentum_origin_world=angular_origin,
        kinetic_energy=kinetic,
    )


def golden_body_property() -> BodyProperty:
    rotation = axis_angle_rotation(np.asarray([1.0, 2.0, 3.0]), 0.61)
    body_origin = np.asarray([0.3, -0.2, 0.5])
    com_offset_body = np.asarray([0.11, -0.07, 0.05])
    return BodyProperty(
        name="golden_body",
        mass=2.3,
        com_position_world=body_origin + com_offset_body,
        inertia_com_body=np.diag([0.031, 0.047, 0.073]),
        rotation_world_from_body=rotation,
        com_linear_velocity_world=np.asarray([0.7, -0.4, 0.9]),
        angular_velocity_world=np.asarray([1.1, -0.8, 0.6]),
    )


def validate_body_mapping(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"schema_version", "bodies"}:
        raise ValueError("Body mapping must contain only schema_version and bodies")
    if payload["schema_version"] != "WheelLegBodyMappingV1":
        raise ValueError("Body mapping schema mismatch")
    bodies = payload["bodies"]
    if not isinstance(bodies, list) or len(bodies) != len(EXPECTED_BODY_ORDER):
        raise ValueError("Body mapping must contain exactly 27 ordered entries")
    for index, canonical in enumerate(EXPECTED_BODY_ORDER):
        expected_name = "base_link" if canonical == "base" else canonical
        expected = {
            "canonical": canonical,
            "isaac": expected_name,
            "mujoco": "base" if canonical == "base" else canonical,
            "source_audit": expected_name,
        }
        if bodies[index] != expected:
            raise ValueError(f"Body mapping entry {index} does not match the frozen mapping")
