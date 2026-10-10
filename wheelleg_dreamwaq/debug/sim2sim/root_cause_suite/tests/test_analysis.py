from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from debug.sim2sim.root_cause_suite.analysis import (
    compare_p50_impact_trace_directories,
    compare_p50_slide_trace_directories,
    _classify_c70_groups,
    compare_p40_trace_directories,
    compare_p30_trace_directories,
    effective_tolerance,
    explanation_ratio,
    first_persistent_divergence,
    odd_even_response,
)
from debug.sim2sim.root_cause_suite.scenario_catalog import (
    robot_probe_profiles,
    sphere_probe_profiles,
)
from debug.sim2sim.root_cause_suite.trace_contract import FieldSpec, write_trace


def test_effective_tolerance_uses_repeatability_envelope() -> None:
    assert effective_tolerance(1.0e-3, 1.0e-4) == 1.0e-3
    assert effective_tolerance(1.0e-3, 4.0e-4) == 2.0e-3


def test_odd_even_response() -> None:
    plus = np.asarray([3.0, 5.0])
    zero = np.asarray([1.0, 1.0])
    minus = np.asarray([-1.0, -3.0])
    odd, even = odd_even_response(plus, zero, minus)
    assert np.allclose(odd, [2.0, 4.0])
    assert np.allclose(even, [0.0, 0.0])


def test_explanation_ratio_and_zero_denominator() -> None:
    assert explanation_ratio(10.0, 2.0, tolerance=1.0) == 0.8
    assert explanation_ratio(0.5, 0.1, tolerance=1.0) is None


def test_first_persistent_divergence_requires_three_samples() -> None:
    mask = np.asarray([False, True, True, False, True, True, True])
    assert first_persistent_divergence(mask, width=3) == 4


def _write_p30_trace(path: Path, *, scale: float, repeat_offset: float) -> None:
    profiles = robot_probe_profiles("p30_open_direct", repetitions=3)
    times = np.asarray([0.0, 0.005, 0.010, 0.015, 0.020, 0.040, 0.100])
    channel = np.asarray([item["channel"] for item in profiles], dtype=np.int16)
    sign = np.asarray([item["sign"] for item in profiles], dtype=np.int8)
    repetition = np.asarray([item["repetition"] for item in profiles], dtype=np.int16)
    velocity = np.zeros((len(times), len(profiles), 6), dtype=np.float64)
    for profile_index, item in enumerate(profiles):
        active = int(item["channel"])
        velocity[:, profile_index, active] = (
            float(item["sign"]) * scale * times
            + repeat_offset * int(item["repetition"])
        )
    arrays = {
        "time_s": np.broadcast_to(times[:, None], (len(times), len(profiles))),
        "profile_channel": np.broadcast_to(channel[None, :], (len(times), len(profiles))),
        "profile_sign": np.broadcast_to(sign[None, :], (len(times), len(profiles))),
        "profile_repetition": np.broadcast_to(
            repetition[None, :], (len(times), len(profiles))
        ),
        "controlled_velocity_canonical": velocity,
    }
    fields = {
        name: FieldSpec("1", "canonical", "post_step", name) for name in arrays
    }
    write_trace(path, arrays, fields)


def test_p30_comparison_groups_profiles_and_reports_repeat_envelope(tmp_path: Path) -> None:
    _write_p30_trace(tmp_path / "isaac", scale=2.0, repeat_offset=1.0e-5)
    _write_p30_trace(tmp_path / "mujoco", scale=1.0, repeat_offset=0.0)
    result = compare_p30_trace_directories(tmp_path / "isaac", tmp_path / "mujoco")
    assert result["normalized_rmse"] > 0.0
    assert result["normalization_scale"] == 1.0
    assert result["material"] is True
    assert result["isaac_repeat_envelope"] == pytest.approx(2.0e-5)
    assert result["mujoco_repeat_envelope"] == 0.0


def _write_p40_trace(path: Path, *, scale: float, repeat_offset: float) -> None:
    profiles = robot_probe_profiles("p40_on", repetitions=3)
    times = np.asarray([0.0, 0.005, 0.010, 0.015, 0.020, 0.040, 0.100])
    channel = np.asarray([item["channel"] for item in profiles], dtype=np.int16)
    sign = np.asarray([item["sign"] for item in profiles], dtype=np.int8)
    repetition = np.asarray([item["repetition"] for item in profiles], dtype=np.int16)
    velocity = np.zeros((len(times), len(profiles), 4), dtype=np.float64)
    for profile_index, item in enumerate(profiles):
        velocity[:, profile_index] = (
            float(item["sign"]) * scale * times[:, None]
            + repeat_offset * int(item["repetition"])
        )
    arrays = {
        "time_s": np.broadcast_to(times[:, None], (len(times), len(profiles))),
        "profile_channel": np.broadcast_to(channel[None, :], (len(times), len(profiles))),
        "profile_sign": np.broadcast_to(sign[None, :], (len(times), len(profiles))),
        "profile_repetition": np.broadcast_to(
            repetition[None, :], (len(times), len(profiles))
        ),
        "all_hinge_velocity": velocity,
    }
    fields = {
        name: FieldSpec("1", "canonical", "post_step", name) for name in arrays
    }
    write_trace(path, arrays, fields)


def test_p40_comparison_reports_closure_ablation_and_effect(tmp_path: Path) -> None:
    _write_p40_trace(tmp_path / "isaac_on", scale=4.0, repeat_offset=1.0e-5)
    _write_p40_trace(tmp_path / "isaac_off", scale=2.0, repeat_offset=0.0)
    _write_p40_trace(tmp_path / "mujoco_on", scale=3.0, repeat_offset=0.0)
    _write_p40_trace(tmp_path / "mujoco_off", scale=2.0, repeat_offset=0.0)
    result = compare_p40_trace_directories(
        tmp_path / "isaac_on",
        tmp_path / "isaac_off",
        tmp_path / "mujoco_on",
        tmp_path / "mujoco_off",
    )
    assert result["closure_on"]["normalized_rmse"] > 0.0
    assert result["closure_off"]["normalized_rmse"] == pytest.approx(0.0)
    assert result["explanation_ratio"] == pytest.approx(1.0)
    assert result["repeat_envelope"] == pytest.approx(2.0e-5)
    assert result["closure_effect_normalized_rmse"] > 0.0


def _write_sphere_trace(
    path: Path,
    *,
    mode: str,
    velocity_scale: float,
    contact_shift_s: float = 0.0,
) -> None:
    profiles = sphere_probe_profiles(mode, repetitions=3)
    times = np.arange(0.0, 0.505, 0.005)
    count = len(profiles)
    velocity = np.zeros((len(times), count, 3), dtype=np.float64)
    contact = np.zeros((len(times), count), dtype=np.int8)
    for index, profile in enumerate(profiles):
        if mode == "impact":
            contact_time = 0.10 + contact_shift_s
            contact[:, index] = (times >= contact_time).astype(np.int8)
            velocity[:, index, 2] = np.where(
                times < contact_time,
                float(profile["vertical_velocity_mps"]) - 9.81 * times,
                velocity_scale * (times - contact_time),
            )
        else:
            velocity[:, index, 0] = (
                float(profile["horizontal_velocity_mps"])
                * np.maximum(0.0, 1.0 - velocity_scale * times)
            )
            contact[:, index] = 1
    arrays = {
        "time_s": np.broadcast_to(times[:, None], (len(times), count)),
        "profile_height_m": np.broadcast_to(
            np.asarray([row["height_m"] for row in profiles])[None, :],
            (len(times), count),
        ),
        "profile_vertical_velocity_mps": np.broadcast_to(
            np.asarray([row["vertical_velocity_mps"] for row in profiles])[None, :],
            (len(times), count),
        ),
        "profile_horizontal_velocity_mps": np.broadcast_to(
            np.asarray([row["horizontal_velocity_mps"] for row in profiles])[None, :],
            (len(times), count),
        ),
        "profile_sign": np.broadcast_to(
            np.asarray([row["sign"] for row in profiles])[None, :],
            (len(times), count),
        ),
        "profile_repetition": np.broadcast_to(
            np.asarray([row["repetition"] for row in profiles])[None, :],
            (len(times), count),
        ),
        "com_velocity_world": velocity,
        "contact_count": contact,
    }
    write_trace(
        path,
        arrays,
        {name: FieldSpec("1", "world", "post_step", name) for name in arrays},
    )


def test_p50_impact_requires_two_predeclared_material_conditions(tmp_path: Path) -> None:
    _write_sphere_trace(tmp_path / "isaac-impact", mode="impact", velocity_scale=1.0)
    _write_sphere_trace(tmp_path / "mujoco-impact", mode="impact", velocity_scale=0.0)
    result = compare_p50_impact_trace_directories(
        tmp_path / "isaac-impact", tmp_path / "mujoco-impact"
    )
    assert result["valid_condition_count"] == 6
    assert result["material_condition_count"] == 6
    assert result["normal_primary_supported"] is True


def test_p50_slide_ratio_uses_zero_friction_ablation(tmp_path: Path) -> None:
    _write_sphere_trace(tmp_path / "isaac-nominal", mode="slide", velocity_scale=4.0)
    _write_sphere_trace(tmp_path / "mujoco-nominal", mode="slide", velocity_scale=1.0)
    _write_sphere_trace(tmp_path / "isaac-zero", mode="slide", velocity_scale=0.0)
    _write_sphere_trace(tmp_path / "mujoco-zero", mode="slide", velocity_scale=0.0)
    result = compare_p50_slide_trace_directories(
        tmp_path / "isaac-nominal",
        tmp_path / "isaac-zero",
        tmp_path / "mujoco-nominal",
        tmp_path / "mujoco-zero",
    )
    assert result["nominal"]["material"] is True
    assert result["zero_friction"]["material"] is False
    assert result["explanation_ratio"] == pytest.approx(1.0)
    assert result["tangential_primary_supported_before_normal_gate"] is True


def _c70_group(*, ratio: float, difference: float, dream: float, ppo: float) -> dict:
    return {
        "eligible": True,
        "gain_ratio": ratio,
        "gain_difference": difference,
        "signs": {
            "plus": {"dreamwaq_gain": dream, "phase1r_gain": ppo},
            "minus": {"dreamwaq_gain": dream, "phase1r_gain": ppo},
        },
    }


def test_c70_amplifier_requires_dual_threshold_and_two_groups() -> None:
    groups = {
        "a": _c70_group(ratio=2.0, difference=0.5, dream=1.0, ppo=0.5),
        "b": _c70_group(ratio=1.8, difference=0.4, dream=0.9, ppo=0.5),
    }
    result = _classify_c70_groups(groups, saturation_fraction_difference=0.0)
    assert result["classification"] == "amplifier"
    assert result["eligible_group_count"] == 2


def test_c70_tie_zone_and_middle_zone() -> None:
    tied = {
        "a": _c70_group(ratio=1.0, difference=0.02, dream=0.52, ppo=0.5),
        "b": _c70_group(ratio=1.1, difference=0.04, dream=0.54, ppo=0.5),
    }
    assert (
        _classify_c70_groups(tied, saturation_fraction_difference=0.01)[
            "classification"
        ]
        == "not_distinguished"
    )
    middle = {
        "a": _c70_group(ratio=1.4, difference=0.2, dream=0.7, ppo=0.5),
        "b": _c70_group(ratio=1.4, difference=0.2, dream=0.7, ppo=0.5),
    }
    result = _classify_c70_groups(middle, saturation_fraction_difference=0.0)
    assert result["classification"] == "inconclusive"
    assert result["threshold_proximity_1_25_to_1_50"] is True
