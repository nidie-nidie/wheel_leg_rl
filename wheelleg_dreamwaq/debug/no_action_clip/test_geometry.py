from __future__ import annotations

import pytest
import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import (
    MAX_LOOP_CLOSURE_ERROR_M, MIN_VIRTUAL_LEG_LENGTH_M,
    load_usd_anchor_audit, offset_geometry_from_audit, offset_leg_fk,
)

from debug.no_action_clip.contract import GEOMETRY_GUARD_CONTRACT
from debug.no_action_clip.geometry import fk_diagnostic_metrics, verify_actual_geometry


def recorded_failure_geometry():
    # 2026-10-15 env201 左腿异常输出中的关节角、实际腿长、phi0 和闭环间隙。
    position = torch.tensor([[-0.400209903717041, -0.2522619366645813]], dtype=torch.float64)
    geometry = offset_geometry_from_audit(load_usd_anchor_audit()["legs"]["left"])
    fk = offset_leg_fk(position, geometry)
    length = torch.tensor([0.139107287], dtype=torch.float64)
    phi0 = torch.tensor([1.92066717], dtype=torch.float64)
    vector = torch.stack((torch.zeros_like(length), length * phi0.cos(), -length * phi0.sin()), dim=-1)
    return {
        "length_true": length, "phi0_true": phi0,
        "length_fk": fk.length, "phi0_fk": fk.phi0, "fk_valid": fk.valid,
        "wheel_error": torch.linalg.vector_norm(fk.wheel_vector_body - vector, dim=-1),
        "loop_error": torch.tensor([[0.0003140150220133364, 0.001674048718996346,
                                     1.7835638573160395e-5, 2.2373347746906802e-5]], dtype=torch.float64),
    }, vector


def test_recorded_fk_disagreement_is_diagnostic_and_does_not_change_values():
    state, vector = recorded_failure_geometry()
    original = {name: values.clone() for name, values in state.items()}
    assert state["wheel_error"].item() == pytest.approx(0.0052499, abs=1e-7)
    verify_actual_geometry(state, vector)
    metrics = fk_diagnostic_metrics(state)
    assert metrics["Debug/fk_disagreement_fraction"].item() == 1.0
    assert metrics["Debug/fk_invalid_fraction"].item() == 0.0
    for name in state:
        torch.testing.assert_close(state[name], original[name], rtol=0, atol=0)


def test_fk_invalid_flag_does_not_replace_actual_geometry_checks():
    state, vector = recorded_failure_geometry()
    state["fk_valid"][:] = False
    verify_actual_geometry(state, vector)
    assert fk_diagnostic_metrics(state)["Debug/fk_invalid_fraction"].item() == 1.0


@pytest.mark.parametrize("field", ["vector", "length_true", "phi0_true", "loop_error"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_actual_geometry_nonfinite_values_rejected(field, bad):
    state, vector = recorded_failure_geometry()
    values = vector if field == "vector" else state[field]
    values.reshape(-1)[0] = bad
    with pytest.raises(RuntimeError, match="非有限"):
        verify_actual_geometry(state, vector)


@pytest.mark.parametrize("length", [0.05, 0.04, 0.0])
def test_actual_minimum_leg_length_is_preserved(length):
    state, vector = recorded_failure_geometry()
    state["length_true"][:] = length
    with pytest.raises(RuntimeError, match="最小值"):
        verify_actual_geometry(state, vector)


def test_actual_loop_error_limit_is_preserved():
    state, vector = recorded_failure_geometry()
    state["loop_error"][:] = MAX_LOOP_CLOSURE_ERROR_M
    verify_actual_geometry(state, vector)
    state["loop_error"][0, 0] += 1e-6
    with pytest.raises(RuntimeError, match="闭环锚点"):
        verify_actual_geometry(state, vector)


def test_declared_geometry_limits_match_formal_values():
    runtime = GEOMETRY_GUARD_CONTRACT["isaac_runtime"]
    assert runtime["min_actual_leg_length_m"] == MIN_VIRTUAL_LEG_LENGTH_M
    assert runtime["max_actual_loop_error_m"] == MAX_LOOP_CLOSURE_ERROR_M
