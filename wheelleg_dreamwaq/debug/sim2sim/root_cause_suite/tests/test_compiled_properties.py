from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from debug.sim2sim.root_cause_suite.compiled_properties import (
    BodyProperty,
    composite_properties,
    golden_body_property,
    momentum_about_origin,
    validate_body_mapping,
)


MAPPING = Path(__file__).resolve().parents[1] / "body_mapping_v1.json"


def test_body_mapping_is_exact_and_complete() -> None:
    payload = json.loads(MAPPING.read_text(encoding="utf-8"))
    validate_body_mapping(payload)
    assert len(payload["bodies"]) == 27
    assert payload["bodies"][0] == {
        "canonical": "base",
        "isaac": "base_link",
        "mujoco": "base",
        "source_audit": "base_link",
    }


def test_single_body_golden_momentum() -> None:
    body = golden_body_property()
    linear, angular_com, angular_origin = momentum_about_origin(body, np.zeros(3))
    assert np.allclose(linear, body.mass * body.com_linear_velocity_world)
    assert np.allclose(angular_origin, angular_com + np.cross(body.com_position_world, linear))


def test_composite_mass_and_center() -> None:
    identity = np.eye(3)
    zero = np.zeros(3)
    bodies = [
        BodyProperty("a", 1.0, np.array([1.0, 0.0, 0.0]), identity, identity, zero, zero),
        BodyProperty("b", 3.0, np.array([-1.0, 0.0, 0.0]), identity, identity, zero, zero),
    ]
    result = composite_properties(bodies)
    assert result.mass == 4.0
    assert np.allclose(result.com_position_world, [-0.5, 0.0, 0.0])

