import math

import pytest

from debug.no_action_clip.heading_control import HeadingController


def test_heading_error_uses_shortest_turn_and_correct_sign():
    controller = HeadingController(kp=0.5)
    controller.reset(math.pi - 0.1)
    result = controller.update(-math.pi + 0.1, 0.0, 0.0)
    assert result.error_rad == pytest.approx(-0.2)
    assert result.command_yaw_rate == pytest.approx(-0.1)
    assert result.mode == "hold"


def test_manual_turn_passes_through_then_captures_new_heading():
    controller = HeadingController()
    controller.reset(0.0)
    result = controller.update(0.8, 0.5, 0.5)
    assert result.command_yaw_rate == 0.5
    assert result.mode == "turn"
    released = controller.update(0.9, 0.0, 0.0)
    assert released.target_rad == pytest.approx(0.9)
    assert released.command_yaw_rate == 0.0
    corrected = controller.update(1.0, 0.0, 0.0)
    assert corrected.command_yaw_rate < 0.0


def test_toggle_and_reset_capture_current_heading():
    controller = HeadingController()
    controller.reset(0.0)
    controller.set_enabled(False, 0.4)
    assert controller.update(0.8, 0.0, 0.0).command_yaw_rate == 0.0
    controller.set_enabled(True, 0.8)
    assert controller.update(0.8, 0.0, 0.0).command_yaw_rate == 0.0
    assert controller.update(0.9, 0.0, 0.0).command_yaw_rate < 0.0
    controller.reset(1.2)
    assert controller.update(1.2, 0.0, 0.0).command_yaw_rate == 0.0


def test_damping_and_command_bound():
    controller = HeadingController(kp=2.0, kd=0.1)
    controller.reset(1.0)
    assert controller.update(0.0, 0.0, 0.0).command_yaw_rate == 1.0
    controller.reset(0.0)
    assert controller.update(0.0, 0.5, 0.0).command_yaw_rate == pytest.approx(-0.05)


@pytest.mark.parametrize("parameter", ["kp", "kd", "max_yaw_rate"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, True])
def test_invalid_parameters_fail(parameter, invalid):
    with pytest.raises(ValueError):
        HeadingController(**{parameter: invalid})


@pytest.mark.parametrize("values", [(float("nan"), 0.0, 0.0),
                                   (0.0, float("inf"), 0.0),
                                   (0.0, 0.0, float("nan")), (0.0, 0.0, 1.1)])
def test_invalid_feedback_or_user_command_fails(values):
    controller = HeadingController()
    controller.reset(0.0)
    with pytest.raises(ValueError):
        controller.update(*values)


def test_uninitialized_controller_and_invalid_reset_fail():
    controller = HeadingController()
    with pytest.raises(RuntimeError):
        controller.update(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        controller.reset(float("nan"))


def test_current_command_update_keeps_physics_and_other_history_unchanged():
    import numpy as np
    from pathlib import Path
    from debug.no_action_clip.mujoco_adapter import load_runtime
    from debug.no_action_clip.play_mujoco_viewer import policy_observation_for_command
    from wheelleg_mujoco.observation import _normalize_command

    export = Path(__file__).resolve().parent / "runs/fresh1000-split-straight-lambda01-env4096-20261016/export"
    runtime = load_runtime(export)
    runtime.reset(np.array([0.0, 0.0, 0.20]))
    before = runtime._history.copy()
    qpos, qvel = runtime.data.qpos.copy(), runtime.data.qvel.copy()
    command = np.array([1.0, -0.3, 0.23])
    observation = policy_observation_for_command(runtime, command).reshape(5, 25)
    expected = before.copy()
    expected[-1, 6:9] = _normalize_command(command, runtime.contract)
    np.testing.assert_array_equal(observation, expected)
    np.testing.assert_array_equal(runtime.data.qpos, qpos)
    np.testing.assert_array_equal(runtime.data.qvel, qvel)
    assert runtime.data.time == 0.0
    runtime.step(np.zeros(6), command)
    np.testing.assert_array_equal(runtime._history[:-1], expected[1:])
