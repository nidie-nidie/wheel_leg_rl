from __future__ import annotations

import pathlib
import json
import importlib.util
import sys
import tempfile
import types
import unittest

import numpy as np
import jsonschema


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fit.common import DynamicsSeries, require_dm_baseline
from fit.fit_leg_dm import fit_leg_series
from fit.fit_leg_dm import fit_leg_dm
from fit.fit_wheel_lk import (
    _fit_torque,
    _fit_velocity,
    _fit_with_delay,
    fit_wheel_lk,
)
from fit.model_manifest import (
    create_model_manifest,
    replace_motor_result,
    validate_model_manifest,
    write_model_manifest,
)
from isaac.leg_dm_adapter import load_leg_dm_contract
from isaac.wheel_lk_adapter import load_wheel_lk_contract
from pace_raw.manifest import load_firmware_manifest


MUJOCO_ADAPTER_PATH = ROOT / "mujoco" / "actuator_replay.py"
MUJOCO_SPEC = importlib.util.spec_from_file_location(
    "wheel_leg_motor_pace_mujoco_adapter", MUJOCO_ADAPTER_PATH
)
MUJOCO_MODULE = importlib.util.module_from_spec(MUJOCO_SPEC)
MUJOCO_SPEC.loader.exec_module(MUJOCO_MODULE)
ActuatorReplay = MUJOCO_MODULE.ActuatorReplay


def future_shift(effective: np.ndarray, delay_steps: int) -> np.ndarray:
    command = np.empty_like(effective)
    if delay_steps == 0:
        command[:] = effective
        return command
    command[:-delay_steps] = effective[delay_steps:]
    command[-delay_steps:] = effective[-1]
    return command


def motion(sample_count: int, phase: float, scale: float = 1.0):
    time = np.arange(sample_count, dtype=np.float64) * 0.002
    q = scale * (
        0.20 * np.sin(2.0 * np.pi * 0.7 * time + phase)
        + 0.08 * np.sin(2.0 * np.pi * 2.1 * time + 0.3 * phase)
        + 0.03 * np.sin(2.0 * np.pi * 4.0 * time + 0.7)
    )
    velocity = np.gradient(q, time)
    acceleration = np.gradient(velocity, time)
    valid = np.ones(sample_count, dtype=np.bool_)
    valid[:3] = False
    valid[-3:] = False
    return time, q, velocity, acceleration, valid


def dm_fixture(parameters: dict, delay_steps: int, phase: float, scale: float = 1.0):
    time, q, velocity, acceleration, valid = motion(3500, phase, scale)
    kp = np.full(time.shape, 20.0)
    kd = np.full(time.shape, 0.6)
    friction = parameters["friction"] * np.tanh(velocity / 0.05)
    required = (
        parameters["armature"] * acceleration
        + parameters["viscous"] * velocity
        + friction
    )
    effective_command = q + (required + kd * velocity) / kp - parameters["bias"]
    command = future_shift(effective_command, delay_steps)
    return DynamicsSeries(
        time_s=time,
        command=command,
        position=q,
        velocity=velocity,
        acceleration=acceleration,
        kp=kp,
        kd=kd,
        saturated=np.zeros(time.shape, dtype=np.bool_),
        valid=valid,
        sample_period_s=0.002,
    )


def wheel_torque_fixture(parameters: dict, delay_steps: int, phase: float):
    time, q, velocity, acceleration, valid = motion(3500, phase, 2.0)
    effective = (
        parameters["armature"] * acceleration
        + parameters["viscous"] * velocity
        + parameters["friction"] * np.tanh(velocity / 0.10)
    )
    return DynamicsSeries(
        time_s=time,
        command=future_shift(effective, delay_steps),
        position=q,
        velocity=velocity,
        acceleration=acceleration,
        kp=np.zeros(time.shape),
        kd=np.zeros(time.shape),
        saturated=np.zeros(time.shape, dtype=np.bool_),
        valid=valid,
        sample_period_s=0.002,
    )


def wheel_velocity_fixture(parameters: dict, delay_steps: int, phase: float):
    time, q, velocity, acceleration, valid = motion(3500, phase, 4.0)
    effective = (
        velocity + parameters["time_constant"] * acceleration
    ) / parameters["gain"]
    return DynamicsSeries(
        time_s=time,
        command=future_shift(effective, delay_steps),
        position=q,
        velocity=velocity,
        acceleration=acceleration,
        kp=np.zeros(time.shape),
        kd=np.zeros(time.shape),
        saturated=np.zeros(time.shape, dtype=np.bool_),
        valid=valid,
        sample_period_s=0.002,
    )


def metric_stub():
    return {
        "sample_count": 100,
        "rmse": 0.01,
        "p95_abs_error": 0.02,
        "max_abs_error": 0.03,
        "estimated_delay_steps": 3,
        "delay_error_steps": 0,
        "saturation_rate": 0.0,
    }


def complete_manifest():
    firmware = load_firmware_manifest()
    decoded = types.SimpleNamespace(
        header=types.SimpleNamespace(
            session_type=1,
            sample_rate_hz=500,
            motor_order_version=1,
            scale_manifest_hash=firmware.manifest_hash,
            session_id=7,
            experiment_config_hash=0x50414331,
            firmware_build_id=0x20260928,
            robot_variant=0x574C5031,
        ),
        footer=types.SimpleNamespace(
            overflow=False, statistics_complete=True, dropped_frames=0
        ),
        summary=lambda: {"issues": {}},
    )
    dataset = types.SimpleNamespace(decoded=decoded, manifest=firmware)
    model = create_model_manifest(dataset)
    for index in range(4):
        replace_motor_result(
            model,
            index,
            {
                "type": "pace_dm_pd_v1",
                "armature": 0.08 + 0.01 * index,
                "viscous_friction": 0.12,
                "static_friction": 0.04,
                "dynamic_friction": 0.04,
                "encoder_bias_rad": 0.01,
                "friction_transition_velocity_rad_s": 0.05,
                "delay_steps": 3,
                "controller": {
                    "kp": 20.0,
                    "kd": 0.6,
                    "dq_des_rad_s": 0.0,
                    "torque_feedforward_nm": 0.0,
                },
                "effort_limit_nm": 54.0,
                "parameter_semantics": "synthetic_test",
            },
            {"metrics": metric_stub()},
            metric_stub(),
        )
    for index in (4, 5):
        replace_motor_result(
            model,
            index,
            {
                "type": "lk9025_dual_mode_v1",
                "torque_mode": {
                    "armature": 0.15,
                    "viscous_friction": 0.08,
                    "static_friction": 0.03,
                    "dynamic_friction": 0.03,
                    "friction_transition_velocity_rad_s": 0.10,
                    "delay_steps": 2,
                },
                "velocity_mode": {
                    "command_gain": 0.94,
                    "time_constant_s": 0.08,
                    "delay_steps": 2,
                },
                "effort_limit_nm": 2.41,
                "parameter_semantics": "synthetic_test",
            },
            {"metrics": metric_stub()},
            metric_stub(),
        )
    return model


def pipeline_dataset():
    firmware = load_firmware_manifest()
    decoded = types.SimpleNamespace(
        header=types.SimpleNamespace(
            session_type=1,
            sample_rate_hz=500,
            motor_order_version=1,
            scale_manifest_hash=firmware.manifest_hash,
            session_id=9,
            experiment_config_hash=0x50414331,
            firmware_build_id=0x20260928,
            robot_variant=0x574C5031,
        ),
        footer=types.SimpleNamespace(
            overflow=False, statistics_complete=True, dropped_frames=0
        ),
        summary=lambda: {"issues": {}},
    )
    dm_parameters = [
        {"armature": 0.06, "viscous": 0.11, "friction": 0.035, "bias": -0.020},
        {"armature": 0.08, "viscous": 0.14, "friction": 0.045, "bias": 0.015},
        {"armature": 0.10, "viscous": 0.17, "friction": 0.055, "bias": -0.010},
        {"armature": 0.12, "viscous": 0.20, "friction": 0.065, "bias": 0.025},
    ]
    dm_fit = [dm_fixture(value, 3, index * 0.4) for index, value in enumerate(dm_parameters)]
    dm_validation = [
        dm_fixture(value, 3, index * 0.4 + 0.2, 0.8)
        for index, value in enumerate(dm_parameters)
    ]
    wheel_torque = [
        wheel_torque_fixture(
            {"armature": 0.16 + 0.02 * local, "viscous": 0.08, "friction": 0.04},
            2,
            0.3 + local * 0.5,
        )
        for local in range(2)
    ]
    wheel_velocity = [
        wheel_velocity_fixture(
            {"gain": 0.90 + 0.02 * local, "time_constant": 0.07 + 0.01 * local},
            4,
            0.7 + local * 0.4,
        )
        for local in range(2)
    ]
    rows = []

    def add_dm_stage(stage_id, role, series_set, time_offset):
        for sample_index in range(series_set[0].time_s.size):
            row = {
                "time_s": time_offset + series_set[0].time_s[sample_index],
                "stage_id": stage_id,
                "role": role,
            }
            for index, series in enumerate(series_set):
                prefix = f"m{index}_"
                row.update(
                    {
                        prefix + "fit_eligible": bool(series.valid[sample_index]),
                        prefix + "fb_q_rad": series.position[sample_index],
                        prefix + "fb_dq_rad_s": series.velocity[sample_index],
                        prefix + "cmd_q_rad": series.command[sample_index],
                        prefix + "kp": 20.0,
                        prefix + "kd": 0.6,
                        prefix + "saturated": False,
                        prefix + "command_mode": 1,
                        prefix + "cmd_dq_rad_s": 0.0,
                        prefix + "cmd_tau_nm": 0.0,
                    }
                )
            for index in (4, 5):
                prefix = f"m{index}_"
                row.update(
                    {
                        prefix + "command_mode": 5,
                        prefix + "fit_eligible": False,
                        prefix + "fb_position_rad": 0.0,
                        prefix + "fb_velocity_rad_s": 0.0,
                        prefix + "saturated": False,
                        prefix + "cmd_torque_nm": 0.0,
                        prefix + "cmd_velocity_rad_s": np.nan,
                    }
                )
            rows.append(row)

    def add_wheel_stage(stage_id, mode, series_set, time_offset):
        command_key = "cmd_torque_nm" if mode == 5 else "cmd_velocity_rad_s"
        other_key = "cmd_velocity_rad_s" if mode == 5 else "cmd_torque_nm"
        for sample_index in range(series_set[0].time_s.size):
            row = {
                "time_s": time_offset + series_set[0].time_s[sample_index],
                "stage_id": stage_id,
                "role": "PROVISIONAL_FIT",
            }
            for index in range(4):
                prefix = f"m{index}_"
                row.update(
                    {
                        prefix + "fit_eligible": False,
                        prefix + "fb_q_rad": 0.0,
                        prefix + "fb_dq_rad_s": 0.0,
                        prefix + "cmd_q_rad": 0.0,
                        prefix + "kp": 20.0,
                        prefix + "kd": 0.6,
                        prefix + "saturated": False,
                        prefix + "command_mode": 1,
                        prefix + "cmd_dq_rad_s": 0.0,
                        prefix + "cmd_tau_nm": 0.0,
                    }
                )
            for local, series in enumerate(series_set):
                prefix = f"m{local + 4}_"
                row.update(
                    {
                        prefix + "command_mode": mode,
                        prefix + "fit_eligible": bool(series.valid[sample_index]),
                        prefix + "fb_position_rad": series.position[sample_index],
                        prefix + "fb_velocity_rad_s": series.velocity[sample_index],
                        prefix + "saturated": False,
                        prefix + command_key: series.command[sample_index],
                        prefix + other_key: np.nan,
                    }
                )
            rows.append(row)

    add_dm_stage(2, "PROVISIONAL_FIT", dm_fit, 0.0)
    add_dm_stage(3, "VALIDATION", dm_validation, 8.0)
    add_wheel_stage(4, 5, wheel_torque, 16.0)
    add_wheel_stage(5, 6, wheel_velocity, 24.0)
    return types.SimpleNamespace(decoded=decoded, manifest=firmware, rows=rows)


class FittingTest(unittest.TestCase):
    def test_leg_pace_family_recovers_known_parameters(self):
        expected = [
            {"armature": 0.06, "viscous": 0.11, "friction": 0.035, "bias": -0.020},
            {"armature": 0.08, "viscous": 0.14, "friction": 0.045, "bias": 0.015},
            {"armature": 0.10, "viscous": 0.17, "friction": 0.055, "bias": -0.010},
            {"armature": 0.12, "viscous": 0.20, "friction": 0.065, "bias": 0.025},
        ]
        fit = [dm_fixture(value, 3, index * 0.4) for index, value in enumerate(expected)]
        validation = [
            dm_fixture(value, 3, index * 0.4 + 0.2, 0.8)
            for index, value in enumerate(expected)
        ]
        delay, fitted, fit_metrics, validation_metrics = fit_leg_series(fit, validation)
        self.assertEqual(3, delay)
        for reference, result in zip(expected, fitted):
            self.assertAlmostEqual(reference["armature"], result["armature"], delta=2.0e-4)
            self.assertAlmostEqual(reference["viscous"], result["viscous_friction"], delta=2.0e-4)
            self.assertAlmostEqual(reference["friction"], result["dynamic_friction"], delta=2.0e-4)
            self.assertAlmostEqual(reference["bias"], result["encoder_bias_rad"], delta=2.0e-4)
        self.assertTrue(all(item["rmse"] < 1.0e-5 for item in fit_metrics))
        self.assertTrue(all(item["delay_error_steps"] == 0 for item in validation_metrics))

    def test_wheel_modes_recover_known_parameters(self):
        torque_expected = {"armature": 0.18, "viscous": 0.09, "friction": 0.04}
        velocity_expected = {"gain": 0.91, "time_constant": 0.075}
        torque_series = wheel_torque_fixture(torque_expected, 2, 0.3)
        velocity_series = wheel_velocity_fixture(velocity_expected, 4, 0.7)
        torque_delay, (torque_fit, torque_error, _) = _fit_with_delay(
            torque_series, _fit_torque, 10
        )
        velocity_delay, (velocity_fit, velocity_error, _) = _fit_with_delay(
            velocity_series, _fit_velocity, 10
        )
        self.assertEqual(2, torque_delay)
        self.assertEqual(4, velocity_delay)
        self.assertAlmostEqual(torque_expected["armature"], torque_fit["armature"], delta=2.0e-4)
        self.assertAlmostEqual(torque_expected["viscous"], torque_fit["viscous_friction"], delta=2.0e-4)
        self.assertAlmostEqual(torque_expected["friction"], torque_fit["dynamic_friction"], delta=2.0e-4)
        self.assertAlmostEqual(velocity_expected["gain"], velocity_fit["command_gain"], delta=2.0e-4)
        self.assertAlmostEqual(velocity_expected["time_constant"], velocity_fit["time_constant_s"], delta=2.0e-4)
        self.assertLess(float(np.sqrt(np.mean(torque_error * torque_error))), 1.0e-5)
        self.assertLess(float(np.sqrt(np.mean(velocity_error * velocity_error))), 1.0e-5)

    def test_dm_baseline_metadata_is_enforced(self):
        rows = []
        for _ in range(40):
            rows.append(
                {
                    "m0_command_mode": 1,
                    "m0_kp": 20.0,
                    "m0_kd": 0.6,
                    "m0_cmd_dq_rad_s": 0.0,
                    "m0_cmd_tau_nm": 0.0,
                }
            )
        require_dm_baseline(rows, 0)
        rows[0]["m0_kp"] = 25.0
        with self.assertRaisesRegex(ValueError, "Kp"):
            require_dm_baseline(rows, 0)

    def test_both_simulator_adapters_preserve_order(self):
        model = complete_manifest()
        validate_model_manifest(model)
        schema_path = ROOT / "models" / "motor_model_schema.json"
        jsonschema.validate(model, json.loads(schema_path.read_text(encoding="utf-8")))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.json"
            write_model_manifest(model, path)
            leg = load_leg_dm_contract(path)
            wheel = load_wheel_lk_contract(path)
            replay = ActuatorReplay(path)
        self.assertEqual([0, 1, 2, 3], leg["canonical_indices"])
        self.assertEqual([4, 5], wheel["canonical_indices"])
        self.assertEqual([2.41, 2.41], wheel["effort_limit_nm"])
        self.assertEqual(
            ("L_front", "L_rear", "R_rear", "R_front", "L_wheel", "R_wheel"),
            replay.canonical_order,
        )
        first = replay.compute_dm_effort([0.1] * 4, [0.0] * 4, [0.0] * 4)
        second = replay.compute_dm_effort([0.1] * 4, [0.0] * 4, [0.0] * 4)
        third = replay.compute_dm_effort([0.1] * 4, [0.0] * 4, [0.0] * 4)
        fourth = replay.compute_dm_effort([0.1] * 4, [0.0] * 4, [0.0] * 4)
        self.assertTrue(np.allclose(first, 0.0))
        self.assertTrue(np.allclose(second, 0.0))
        self.assertTrue(np.allclose(third, 0.0))
        self.assertTrue(np.all(fourth > 0.0))

    def test_full_provisional_fit_pipeline_builds_complete_manifest(self):
        dataset = pipeline_dataset()
        model = fit_leg_dm(dataset)
        model = fit_wheel_lk(dataset, model=model)
        validate_model_manifest(model)
        self.assertEqual("provisional", model["model_maturity"])
        self.assertEqual(3, model["motors"][0]["model"]["delay_steps"])
        self.assertEqual(2, model["motors"][4]["model"]["torque_mode"]["delay_steps"])
        self.assertEqual(4, model["motors"][4]["model"]["velocity_mode"]["delay_steps"])
        self.assertEqual(2.41, model["motors"][4]["model"]["effort_limit_nm"])
        self.assertEqual(2.41, model["motors"][5]["model"]["effort_limit_nm"])
        self.assertFalse(model["motors"][4]["validation"]["available"])


if __name__ == "__main__":
    unittest.main()
