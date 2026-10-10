from __future__ import annotations
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from wheelleg_mujoco.command_practice import (
    STAGE_PAYLOAD, command_at_tick, summarize_practice, validate_command_practice,
)
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from test_mujoco_adapters import MODEL_PATH, _contract


def _manifest():
    return {"command_sampling": {"practice_schedule": STAGE_PAYLOAD["version"],
            "hold_for_episode": False, "practice_contract": deepcopy(STAGE_PAYLOAD)},
            "training_reward_weights": {"tracking_vx": 2., "tracking_vx_enhance": 2.}}


def test_exported_timeline_validation_and_boundaries():
    payload = validate_command_practice(_manifest())
    initial = np.array((.5, -.6, .2))
    for tick, factor in [(0, 1), (99, 1), (100, 0), (149, 0), (150, -1),
                         (249, -1), (250, 0), (299, 0), (300, 1), (399, 1),
                         (400, 0), (499, 0), (500, 0), (501, 0)]:
        np.testing.assert_array_equal(command_at_tick(initial, tick, payload), initial * (factor, factor, 1))
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            command_at_tick(initial, bad, payload)
    for field, value in [("stage_end_control_steps", [99, 150, 250, 300, 400, 500]),
                          ("final_phase_behavior", "restart")]:
        bad = _manifest()
        bad["command_sampling"]["practice_contract"][field] = value
        with pytest.raises(ValueError):
            validate_command_practice(bad)
    bad = _manifest()
    del bad["training_reward_weights"]
    with pytest.raises(ValueError):
        validate_command_practice(bad)
    assert validate_command_practice({"command_sampling": {"hold_for_episode": True}}) is None


def test_next_command_only_enters_new_history_frame():
    contract = replace(_contract(), policy_input_dimension=125, history_length=5, history_layout="frame_major")
    runtime = WheelLegMujocoRuntime(MODEL_PATH, contract)
    initial = np.array((.5, 0., .2))
    observation = runtime.reset(command_at_tick(initial, 99, STAGE_PAYLOAD))
    old_frames = observation.reshape(5, 25).copy()
    result = runtime.step(np.zeros(6), command_at_tick(initial, 100, STAGE_PAYLOAD))
    frames = result.observation.reshape(5, 25)
    np.testing.assert_array_equal(frames[:4], old_frames[1:])
    np.testing.assert_array_equal(frames[-1, 6:9], (0., 0., 0.))
    np.testing.assert_array_equal(observation.reshape(5, 25), old_frames)
    assert runtime.data.time == pytest.approx(.02)


def _perfect_rows(initial, count=500):
    rows = []
    for tick in range(count):
        target = command_at_tick(initial, tick, STAGE_PAYLOAD)
        rows.append({"tick": tick, "vx_mps": target[0], "yaw_rate_rad_s": target[1],
                     "com_xy_before": [tick * .01, 0.], "com_xy_after": [(tick + 1) * .01, 0.]})
    return rows


def test_response_confirmation_and_stop_path_include_boundary():
    initial = np.array((.5, 0., .2))
    summary = summarize_practice("perfect", initial, _perfect_rows(initial), None)
    assert summary["performance_accepted"]
    stop = summary["stages"][1]
    assert stop["stop_distance_m"] == pytest.approx(.50)
    assert stop["stop_net_displacement_m"] == pytest.approx(.50)
    assert stop["steady_sample_ticks"] == 25
    reverse = summary["stages"][2]
    assert reverse["held_error_band_confirmation_time_s"] == pytest.approx(.2)
    assert reverse["first_correct_sign_time_s"] == pytest.approx(.02)
    assert summarize_practice("prefix", initial, _perfect_rows(initial, 180), "tilt")["performance_accepted"] is False
    partial = summarize_practice("prefix", initial, _perfect_rows(initial, 180), "tilt")["stages"][2]
    assert partial["completed"] is False
    assert partial["performance_accepted"] is False
    assert summarize_practice("empty", initial, [], "non_finite_action")["stages"][0]["steady_vx_mae_mps"] is None


def test_band_requires_ten_consecutive_samples_and_ends_window():
    initial = np.array((0., .6, .2))
    rows = _perfect_rows(initial)
    # Break the first possible reverse window at the ninth sample.
    rows[158]["yaw_rate_rad_s"] = .6
    summary = summarize_practice("yaw", initial, rows, None)
    assert summary["stages"][2]["held_error_band_confirmation_time_s"] == pytest.approx(.38)


def test_second_direction_reversal_must_also_respond_within_one_second():
    initial = np.array((.5, 0., .2))
    rows = _perfect_rows(initial)
    for row in rows[300:400]:
        row["vx_mps"] = 0.
    summary = summarize_practice("ignored_second_reversal", initial, rows, None)
    assert summary["completed"]
    assert summary["stages"][2]["performance_accepted"]
    assert summary["stages"][4]["held_error_band_confirmation_time_s"] is None
    assert summary["stages"][4]["performance_accepted"] is False
    assert summary["performance_accepted"] is False
