from __future__ import annotations

import torch

from wheelleg_dreamwaq.schemas.normalization import NormalizationV1, NormalizationV2


def test_zero_velocity_and_nominal_height_map_to_zero() -> None:
    cfg = NormalizationV1()
    commands = torch.tensor([[0.0, 0.0, cfg.nominal_base_height]])
    normalized = cfg.normalize_command(commands)
    assert torch.allclose(normalized, torch.zeros_like(normalized))


def test_command_extrema_map_to_unit_interval() -> None:
    cfg = NormalizationV1()
    commands = torch.tensor(
        [
            [cfg.vx_max_abs, cfg.yaw_rate_max_abs, cfg.nominal_base_height + cfg.height_command_span],
            [-cfg.vx_max_abs, -cfg.yaw_rate_max_abs, cfg.nominal_base_height - cfg.height_command_span],
        ]
    )
    normalized = cfg.normalize_command(commands)
    assert torch.allclose(normalized[0], torch.ones(3))
    assert torch.allclose(normalized[1], -torch.ones(3))


def test_actor_fields_use_fixed_scales_and_clips() -> None:
    cfg = NormalizationV1()
    angular_velocity = cfg.normalize_angular_velocity(torch.tensor([[4.0, -4.0, 100.0]]))
    joint_velocity = cfg.normalize_joint_velocity(torch.tensor([[20.0, -20.0, 200.0, 0.0, 1.0, -1.0]]))
    previous_action = cfg.normalize_previous_action(torch.tensor([[2.0, -2.0, 0.0, 0.5, -0.5, 1.0]]))
    assert torch.allclose(angular_velocity, torch.tensor([[1.0, -1.0, cfg.angular_velocity_clip]]))
    assert torch.allclose(joint_velocity[:, :2], torch.tensor([[1.0, -1.0]]))
    assert joint_velocity[0, 2] == cfg.joint_velocity_clip
    assert torch.equal(previous_action, torch.tensor([[1.0, -1.0, 0.0, 0.5, -0.5, 1.0]]))


def test_normalization_v2_uses_contract_v4_command_ranges() -> None:
    cfg = NormalizationV2()
    commands = torch.tensor(
        [
            [1.5, 1.0, 0.24],
            [-1.5, -1.0, 0.16],
            [0.0, 0.0, 0.20],
        ]
    )
    normalized = cfg.normalize_command(commands)
    assert torch.allclose(normalized[0], torch.ones(3))
    assert torch.allclose(normalized[1], -torch.ones(3))
    assert torch.allclose(normalized[2], torch.zeros(3))
    assert cfg.schema_version == "NormalizationV2"


def test_normalization_v1_is_preserved_for_historical_contracts() -> None:
    cfg = NormalizationV1()
    assert cfg.vx_max_abs == 1.0
    assert cfg.yaw_rate_max_abs == 1.5
    assert cfg.schema_version == "NormalizationV1"
