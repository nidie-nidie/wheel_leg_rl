from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from wheelleg_dreamwaq.schemas.heading_control import HeadingHoldConfig, heading_contract
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[3]
SIM = PROJECT / "sim2sim/mujoco"
EXPORT = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03"


def runtime(*, enabled=True):
    original, _, model = load_policy_contract(EXPORT / "policy_manifest.json", EXPORT / "actor.ts",
                                              SIM / "model_manifest.json")
    contract = replace(original, action_clip=np.inf,
                       heading_hold=heading_contract(HeadingHoldConfig(enabled=enabled)))
    return WheelLegMujocoRuntime(SIM / "models" / model["model_xml"]["path"], contract, model)


def test_real_reset_heading_prepare_history_and_raw_action():
    env = runtime()
    source = np.array([1., 0., .2])
    initial = env.reset(source)
    assert initial.shape == (125,)
    assert env.heading.reference.item() == pytest.approx(env.last_reset_metrics["yaw_rad"])
    first = env.prepare_policy_observation(source)
    np.testing.assert_array_equal(first.reshape(5, 25), np.repeat(first[-25:][None, :], 5, axis=0))
    before = env._history.copy()
    np.testing.assert_array_equal(env.prepare_policy_observation(source), first)
    np.testing.assert_array_equal(env._history, before)
    result = env.step(np.array([2., -2., 1.5, -1.5, 2., 2.]), source)
    np.testing.assert_array_equal(result.observation[-6:], result.clipped_action)
    np.testing.assert_array_equal(env._history[:-1], before[1:])
    assert result.metrics["action_saturation_fraction"] == 0
    assert result.metrics["action_outside_former_unit_interval_fraction"] == 1
    after = env._history.copy()
    prepared = env.prepare_policy_observation(source)
    np.testing.assert_array_equal(env._history[:-1], after[:-1])
    mask = np.ones(25, dtype=bool)
    mask[6:9] = False
    np.testing.assert_array_equal(prepared[-25:][mask], after[-1][mask])


def test_real_pose_sign_turn_release_toggle_and_finite_rejection():
    env = runtime()
    source = np.array([0., 0., .2])
    env.reset(source)
    # 只改变初态姿态，随后由 MuJoCo 正式前向计算产生真实反馈。
    env.data.qpos[3:7] = [np.cos(.05), 0, 0, np.sin(.05)]
    mujoco.mj_forward(env.model, env.data)
    env.prepare_policy_observation(source)
    assert env.effective_command[1] == pytest.approx(-.9)
    source[1] = .6
    env.prepare_policy_observation(source)
    assert env.effective_command[1] == .6
    source[1] = 0
    env.prepare_policy_observation(source)
    assert env.effective_command[1] == pytest.approx(0)
    env.set_heading_enabled(False)
    env.prepare_policy_observation(source)
    np.testing.assert_array_equal(env.effective_command, source)
    time_before = env.data.time
    with pytest.raises(ValueError):
        env.step(np.full(6, np.nan), source)
    assert env.data.time == time_before
    env.reset(source)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(6), source)


def test_disabled_saved_heading_passes_source_through_real_steps_and_reset():
    env = runtime(enabled=False)
    source = np.array([1., 0., .2])
    env.reset(source)
    assert env.heading.enabled is False
    env.data.qpos[3:7] = [np.cos(.05), 0, 0, np.sin(.05)]
    mujoco.mj_forward(env.model, env.data)
    for yaw in (0., .6, 0., -.6, 0.):
        source[1] = yaw
        env.prepare_policy_observation(source)
        np.testing.assert_array_equal(env.effective_command, source)
        env.step(np.zeros(6), source)
    env.reset(source)
    env.prepare_policy_observation(source)
    assert env.heading.enabled is False
    np.testing.assert_array_equal(env.effective_command, source)
    env.set_heading_enabled(True)
    assert env.heading.enabled is True
