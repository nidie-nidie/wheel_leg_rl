from __future__ import annotations

import math

import pytest
import torch

from wheelleg_dreamwaq.schemas.heading_control import (
    HeadingHold, HeadingHoldConfig, heading_contract, heading_profile_name,
    RAW_BASELINE_PROFILE, RAW_BASELINE_NO_HEADING_PROFILE, yaw_from_quaternion, yaw_from_rotation,
)
from debug.no_action_clip.heading_control import HeadingController


def test_wrap_sign_saturation_and_manual_release():
    controller = HeadingHold(HeadingHoldConfig(), 2, dtype=torch.float64)
    controller.reset(torch.tensor([math.pi - 0.01, -math.pi + 0.01], dtype=torch.float64))
    yaw = torch.tensor([-math.pi + 0.01, math.pi - 0.01], dtype=torch.float64)
    source = torch.tensor([[1., 0., .2], [-1., 0., .2]], dtype=torch.float64)
    result = controller.update(yaw, torch.zeros(2), source)
    torch.testing.assert_close(result[:, 1], torch.tensor([-.18, .18], dtype=torch.float64))
    controller.reset(torch.zeros(2, dtype=torch.float64))
    result = controller.update(torch.tensor([1., -1.]), torch.zeros(2), source)
    assert result[:, 1].tolist() == [-1., 1.]
    assert controller.saturated.all()
    source[:, 1] = torch.tensor([.6, -.6])
    assert torch.equal(controller.update(yaw, torch.zeros(2), source), source)
    source[:, 1] = 0
    assert torch.equal(controller.update(yaw + .1, torch.zeros(2), source), source)
    reference = controller.reference.clone()
    source[:, 0] = -source[:, 0]
    controller.update(yaw + .1, torch.zeros(2), source)
    assert torch.equal(controller.reference, reference)


def test_partial_reset_toggle_repeated_read_and_scalar_agreement():
    controller = HeadingHold(HeadingHoldConfig(), 2, dtype=torch.float64)
    controller.reset(torch.tensor([.2, -.3], dtype=torch.float64))
    scalar = HeadingController(kp=9, kd=0, enabled=True)
    scalar.reset(.2)
    for yaw, source_yaw in ((.22, 0.), (.4, .5), (.6, 0.), (.63, 0.)):
        source = torch.tensor([[1., source_yaw, .2], [0., 0., .2]], dtype=torch.float64)
        angles = torch.tensor([yaw, -.3], dtype=torch.float64)
        result = controller.update(angles, torch.zeros(2), source)
        assert result[0, 1].item() == pytest.approx(scalar.update(yaw, 0., source_yaw).command_yaw_rate)
        reference = controller.reference.clone()
        assert torch.equal(result, controller.update(angles, torch.zeros(2), source))
        assert torch.equal(reference, controller.reference)
    controller.reset(torch.tensor([1.], dtype=torch.float64), torch.tensor([1]))
    assert controller.reference[0] == reference[0]
    assert controller.reference[1] == 1
    controller.toggle(False, torch.zeros(2))
    assert torch.equal(controller.update(torch.ones(2), torch.zeros(2), source), source)


def test_rotation_and_quaternion_use_identical_control_frame():
    angle = .7
    rotation = torch.tensor([[[math.cos(angle), -math.sin(angle), 0.],
                              [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]]], dtype=torch.float64)
    quaternion = torch.tensor([[math.cos(angle / 2), 0., 0., math.sin(angle / 2)]], dtype=torch.float64)
    torch.testing.assert_close(yaw_from_rotation(rotation), torch.tensor([angle], dtype=torch.float64))
    torch.testing.assert_close(yaw_from_quaternion(quaternion), yaw_from_rotation(rotation))
    assert yaw_from_quaternion(torch.tensor([[1., 0., 0., 0.]])).item() == 0


@pytest.mark.parametrize("kwargs", [{"kp": float("nan")}, {"kd": -1.}, {"max_yaw_rate": 0.},
                                   {"control_dt_s": float("inf")}, {"enabled": 1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        HeadingHoldConfig(**kwargs)


def test_invalid_input_fails_before_state_changes():
    controller = HeadingHold(HeadingHoldConfig(), 1)
    with pytest.raises(ValueError):
        controller.update(torch.zeros(1), torch.zeros(1), torch.zeros(1, 3))
    controller.reset(torch.zeros(1))
    with pytest.raises(ValueError):
        controller.update(torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        yaw_from_quaternion(torch.zeros(1, 4))


def test_disabled_training_heading_passes_source_through_pose_and_turn_changes():
    controller = HeadingHold(HeadingHoldConfig(enabled=False), 2)
    controller.reset(torch.tensor([.2, -.3]))
    for source_yaw in (0., .6, 0., -.6, 0.):
        source = torch.tensor([[1., source_yaw, .2], [-1., -source_yaw, .2]])
        actual = controller.update(torch.tensor([.9, -.8]), torch.tensor([.3, -.4]), source)
        assert torch.equal(actual, source)
        assert not controller.saturated.any()
        assert not controller.was_turning.any()
    controller.reset(torch.tensor([1.]), torch.tensor([1]))
    assert controller.enabled is False


@pytest.mark.parametrize("enabled,name", [(True, RAW_BASELINE_PROFILE), (False, RAW_BASELINE_NO_HEADING_PROFILE)])
def test_named_heading_contract_roundtrip(enabled, name):
    saved = heading_contract(HeadingHoldConfig(enabled=enabled))
    assert heading_profile_name(saved) == name


@pytest.mark.parametrize("field,value", [("enabled", 0), ("enabled", "false"), ("kp", 3.),
                                        ("kd", .1), ("max_yaw_rate", 2.), ("control_dt_s", .01),
                                        ("wheel_differential_gate", "source_yaw_eq_0")])
def test_named_heading_rejects_unknown_or_mistyped_saved_contract(field, value):
    saved = heading_contract(HeadingHoldConfig(enabled=False))
    saved[field] = value
    with pytest.raises(ValueError):
        heading_profile_name(saved)
