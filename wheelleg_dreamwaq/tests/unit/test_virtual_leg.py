from __future__ import annotations

import math

import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import (
    load_usd_anchor_audit,
    offset_geometry_from_audit,
    offset_leg_fk,
    phi0_symmetry_error,
    virtual_leg_from_body_vector,
    wrap_to_pi,
)


def test_virtual_leg_uses_base_y_and_downward_base_z() -> None:
    vector = torch.tensor([[0.4, 0.12, -0.16]])
    length, phi0 = virtual_leg_from_body_vector(vector)
    assert torch.allclose(length, torch.tensor([0.20]))
    assert torch.allclose(phi0, torch.tensor([math.atan2(0.16, 0.12)]))


def test_lateral_offset_does_not_change_virtual_leg_coordinates() -> None:
    vectors = torch.tensor([[0.0, -0.01, -0.20], [0.08, -0.01, -0.20]])
    length, phi0 = virtual_leg_from_body_vector(vectors)
    assert torch.allclose(length[0], length[1])
    assert torch.allclose(phi0[0], phi0[1])


def test_wrapped_phi0_symmetry_is_continuous_at_pi() -> None:
    epsilon = 1.0e-4
    phi0 = torch.tensor([[math.pi - epsilon, -math.pi + epsilon]])
    delta = wrap_to_pi(phi0[:, 0] - phi0[:, 1])
    assert torch.allclose(torch.abs(delta), torch.tensor([2.0 * epsilon]), atol=1.0e-6)
    assert torch.allclose(phi0_symmetry_error(phi0), delta.square())


def test_known_nominal_virtual_leg_reference() -> None:
    length_reference = 0.199507440221
    phi_reference = math.radians(90.539807010673)
    vector = torch.tensor(
        [[0.0, length_reference * math.cos(phi_reference), -length_reference * math.sin(phi_reference)]],
        dtype=torch.float64,
    )
    length, phi0 = virtual_leg_from_body_vector(vector)
    assert torch.allclose(length, torch.tensor([length_reference], dtype=torch.float64), atol=1.0e-12)
    assert torch.allclose(phi0, torch.tensor([phi_reference], dtype=torch.float64), atol=1.0e-12)


def test_offset_fk_nominal_is_close_to_audited_reference() -> None:
    q_nominal = torch.tensor([[-0.33367134, 0.33367134]], dtype=torch.float64)
    result = offset_leg_fk(q_nominal)
    assert result.valid.item()
    assert abs(result.length.item() - 0.199507440221) < 5.0e-4
    assert abs(wrap_to_pi(result.phi0 - math.radians(90.539807010673)).item()) < math.radians(0.1)


def test_usd_audited_left_and_right_geometry_match_nominal_golden_vectors() -> None:
    audit = load_usd_anchor_audit()
    cases = (
        ("left", (-0.33367151, 0.33367157), 0.199506867746, 90.539731893820),
        ("right", (-0.33367088, 0.33367112), 0.199506677725, 90.539725723463),
    )
    for side, q, expected_length, expected_phi_deg in cases:
        geometry = offset_geometry_from_audit(audit["legs"][side])
        result = offset_leg_fk(torch.tensor([q], dtype=torch.float64), geometry)
        assert result.valid.item()
        assert abs(result.length.item() - expected_length) < 1.0e-9
        assert abs(math.degrees(result.phi0.item()) - expected_phi_deg) < 1.0e-7
