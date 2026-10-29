from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import time_ns

import mujoco
import numpy as np
import pytest

from recovery.calibrate import (CalibrationConfig, accepted_samples, action_differences, compatibility,
    load_calibration, recovery_terminal_pair, run_calibration, validate_recovery_terminal_records,
    _validate_terminal_action)
from recovery.collect_pose_bank import minimum_mesh_height, orient_for_drop
from recovery.collect_bootstrap_pose_bank import BootstrapClassification, BootstrapCollectionConfig, collect_bootstrap
from recovery.contact_model import SOURCE_XML, assert_contact_filter, build_contact_model, file_sha256, load_contact_model
from recovery.contracts import write_json
from recovery.mujoco_runtime import MujocoRecoveryRuntime
from recovery.pose_bank import PoseBank, pose_record_physical_sha256
from recovery.evaluate import EvaluationProtocol, create_case_supervisor, select_source_command, verify_test_isolation


ROOT = Path(__file__).resolve().parents[3]
WALKER = ROOT / "logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/export"


@pytest.fixture(scope="module")
def directory():
    result = ROOT / "recovery/artifacts/mujoco_tests" / str(time_ns())
    result.mkdir(parents=True)
    return result


@pytest.fixture(scope="module")
def contact(directory):
    identity = file_sha256(SOURCE_XML)
    path = build_contact_model(directory / "model", friction=0.7)
    assert file_sha256(SOURCE_XML) == identity
    return path.with_name("contact_manifest.json")


@pytest.fixture
def runtime(contact):
    result = MujocoRecoveryRuntime(contact, WALKER)
    result.reset_nominal()
    return result


def test_compiled_complete_contact_keeps_dynamics_and_explicit_material(contact):
    model, record = load_contact_model(contact)
    source = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
    for field in ("body_mass", "body_inertia", "dof_armature", "dof_damping", "jnt_axis", "key_qpos"):
        np.testing.assert_array_equal(getattr(model, field), getattr(source, field))
    assert model.npair == 17
    np.testing.assert_array_equal(model.pair_friction[:, :2], np.full((17, 2), 0.7))
    assert record["geometry_representation"] == "MuJoCo_convex_hull_per_source_mesh"
    assert_contact_filter(model)


def test_candidate_reads_preserve_entire_state_and_history(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    before = runtime.snapshot()
    expected = runtime.walker_action()
    for _ in range(11):
        np.testing.assert_array_equal(runtime.walker_action(), expected)
    assert before == runtime.snapshot()


def test_recovery_targets_exceed_one_radian_with_real_torque_caps_and_walker_limits(runtime):
    raw = np.array([5.0, -5.0, 5.0, -5.0, 1.0, -1.0])
    before = runtime.snapshot()
    expected_legs = runtime.q_reference + 0.35 * raw[:4]
    assert np.all(np.abs(expected_legs) > 1.0)
    recovery = runtime.step(raw)
    np.testing.assert_allclose(recovery.leg_target, expected_legs, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(recovery.wheel_target, 25.0 * raw[4:])
    np.testing.assert_array_equal(recovery.action, raw)
    for substep in recovery.substeps:
        assert np.all(np.abs(substep["torque"]) <= runtime.effort_limits)
    np.testing.assert_array_equal(np.abs(recovery.substeps[0]["torque"]), runtime.effort_limits)
    np.testing.assert_array_equal(runtime.data.actuator_force[runtime.model_map.actuator_ids],
                                  recovery.torque * np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]))
    runtime.restore_snapshot(before)
    walker = runtime.step(raw, locomotion=True)
    np.testing.assert_array_equal(walker.leg_target, np.clip(expected_legs, -1.0, 1.0))
    np.testing.assert_array_equal(walker.wheel_target, recovery.wheel_target)
    for substep in walker.substeps:
        assert np.all(np.abs(substep["torque"]) <= runtime.effort_limits)


def test_compatibility_decodes_recovery_and_walker_modes_without_changing_real_state(runtime):
    raw = np.array([5.0, -5.0, 5.0, -5.0, 1.0, -1.0])
    before = runtime.snapshot()
    limits = {"target_difference_limits": [0.1] * 6, "torque_difference_limits": [1.0] * 6}
    recovery_leg, recovery_wheel = runtime.decoded(raw)
    walker_leg, walker_wheel = runtime.decoded(raw, locomotion=True)
    recovery_target, recovery_torque = action_differences(runtime, raw, raw)
    np.testing.assert_array_equal(recovery_target,
                                  np.abs(np.concatenate((recovery_leg - walker_leg, recovery_wheel - walker_wheel))))
    np.testing.assert_array_equal(recovery_torque,
                                  np.abs(runtime.estimated_torque(raw) - runtime.estimated_torque(raw, locomotion=True)))
    accepted, blockers, differences = compatibility(runtime, raw, raw, limits)
    assert not accepted and blockers
    np.testing.assert_array_equal(differences["target_difference"], recovery_target)
    walker_target, walker_torque = action_differences(runtime, raw, raw, actual_is_walker=True)
    np.testing.assert_array_equal(walker_target, np.zeros(6))
    np.testing.assert_array_equal(walker_torque, np.zeros(6))
    accepted, blockers, differences = compatibility(runtime, raw, raw, limits, actual_is_walker=True)
    assert accepted and not blockers
    np.testing.assert_array_equal(differences["target_difference"], np.zeros(6))
    np.testing.assert_array_equal(differences["torque_difference_nm"], np.zeros(6))
    assert before == runtime.snapshot()


def test_dynamic_pose_restore_keeps_native_joints_root_velocity_and_history(runtime, directory):
    for _ in range(7):
        runtime.step(runtime.walker_action(), locomotion=True)
    snapshot = runtime.snapshot()
    record = runtime.pose_record(record_id="restore-real", trajectory_id="real-trajectory", split="train",
                                 category="near_upright", motion_kind="dynamic", metadata={"origin": "interface_validation"})
    path = directory / f"bank-{time_ns()}.json"
    PoseBank([record], {"purpose": "interface_validation"}).save(path)
    runtime.step(np.array([0.1, -0.1, 0.1, -0.1, 0.05, -0.05]))
    runtime.restore_pose(PoseBank.load(path).records[0])
    after = runtime.snapshot()
    for field in ("joint_positions", "joint_velocities", "root_position", "root_quaternion_wxyz",
                  "root_linear_velocity_world", "root_angular_velocity_world", "actor_history", "walker_history", "prior_actions"):
        np.testing.assert_allclose(after[field], snapshot[field], atol=1e-12, rtol=0)


def test_rigid_rotation_preserves_loop_and_real_nonwheel_floor_contact(runtime):
    orient_for_drop(runtime, "supine", np.random.default_rng(42), 0.01)
    assert abs(minimum_mesh_height(runtime.model, runtime.data) - 0.01) < 1e-7
    contacted = set()
    for _ in range(30):
        result = runtime.step(np.zeros(6))
        for substep in result.substeps:
            assert substep["loop_error_m"] < 0.005
            contacted.update(substep["contact"]["body_normal_force_n"])
    assert "base" in contacted


def test_compatibility_uses_shared_strict_difference_boundary(runtime):
    actual = np.zeros(6)
    candidate = actual.copy()
    candidate[4] = 0.04
    thresholds = {"target_difference_limits": [1.0] * 6, "torque_difference_limits": [18.0] * 6}
    accepted, blockers, differences = compatibility(runtime, actual, candidate, thresholds)
    assert differences["target_difference"][4] == 1.0
    assert not accepted
    assert "target_difference/joint4" in blockers


def test_real_calibration_rejects_samples_without_stable_takeover_and_geometry(runtime, directory):
    config = CalibrationConfig(seed=42, training_samples=2, validation_samples=2, warmup_steps=1,
        takeover_steps=2, maximum_raw_action_perturbation=(0.01,) * 6, quantile=0.95,
        minimum_successful_training_samples=1, minimum_accepted_validation_samples=1,
        allowed_validation_failure_rate=0, support_force_n=1.0, maximum_contact_penetration_m=0.001,
        minimum_clearance_m=0.0, allowed_body_pairs=[])
    result = run_calibration(runtime, config)
    assert result["status"] == "rejected"
    assert result["schema_version"] == "RecoveryCompatibilityCalibrationV2"
    assert result["identity"]["recovery_contract_sha256"] == runtime.contract.sha256
    assert len(result["samples"]) == 4
    assert all(sample["takeover"]["success"] is False for sample in result["samples"])
    assert all(sample["initial_physical_validation"] == "nonallowed_body_clearance" for sample in result["samples"])
    path = directory / "calibration-real-rejected.json"
    write_json(path, result)
    with pytest.raises(ValueError, match="尚未通过真实独立样本验证"):
        load_calibration(path, runtime)
    for contract_hash in (None, "0" * 64):
        changed = deepcopy(result)
        if contract_hash is None:
            changed["identity"].pop("recovery_contract_sha256")
        else:
            changed["identity"]["recovery_contract_sha256"] = contract_hash
        write_json(path, changed)
        with pytest.raises(ValueError, match="当前起身动作契约不同"):
            load_calibration(path, runtime)
    changed = deepcopy(result)
    changed["schema_version"] = "RecoveryCompatibilityCalibrationV1"
    write_json(path, changed)
    with pytest.raises(ValueError, match="动作契约版本无效"):
        load_calibration(path, runtime)


def test_terminal_target_validation_recomputes_raw_action_when_two_records_share_clipped_targets(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    step = runtime.step(np.array([5.0, -5.0, 5.0, -5.0, 1.0, -1.0]))
    record = runtime.pose_record(record_id="target-recomputation", trajectory_id="real-target-trajectory", split="train",
        category="near_upright", motion_kind="dynamic", metadata={"origin": "interface_validation"})
    _validate_terminal_action(runtime, record, step)
    clipped_step = replace(step, leg_target=np.clip(step.leg_target, -1.0, 1.0))
    clipped_record = replace(record, motor_targets=np.concatenate((clipped_step.leg_target, clipped_step.wheel_target)).tolist())
    with pytest.raises(ValueError, match="当前起身动作契约重算"):
        _validate_terminal_action(runtime, clipped_record, clipped_step)


def test_terminal_pair_uses_last_actual_action_at_restored_end_state_without_stepping(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    last_actual = np.array([0.09, -0.07, 0.06, -0.08, 0.04, -0.03])
    runtime.step(last_actual)
    record = runtime.pose_record(record_id="pair-endstate", trajectory_id="pair-real-trajectory", split="train",
        category="near_upright", motion_kind="dynamic", metadata={"origin": "interface_validation"})
    runtime.restore_pose(record)
    expected_state = runtime.snapshot()
    expected_candidate = runtime.walker_action()
    expected_target, expected_torque = action_differences(runtime, last_actual, expected_candidate)
    runtime.step(runtime.walker_action(), locomotion=True)
    pair = recovery_terminal_pair(runtime, record)
    np.testing.assert_array_equal(pair["actual_action"], last_actual)
    np.testing.assert_array_equal(pair["candidate_action"], expected_candidate)
    np.testing.assert_array_equal(pair["target_difference"], expected_target)
    np.testing.assert_array_equal(pair["torque_difference_nm"], expected_torque)
    assert pair["state"] == expected_state == runtime.snapshot()
    assert runtime.frame_id == 0
    assert runtime.data.time == 0


def test_terminal_origin_label_requires_verifiable_actor_and_action_trace(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    record = runtime.pose_record(record_id="unverified-terminal", trajectory_id="unverified-trajectory", split="train",
        category="near_upright", motion_kind="dynamic", metadata={"origin": "recovery_terminal"})
    bank = PoseBank([record], {"purpose": "negative_provenance_regression"})
    with pytest.raises(ValueError, match="缺少可核验策略"):
        validate_recovery_terminal_records(runtime, bank)


def test_terminal_split_rejects_renamed_identical_physical_state(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    record = runtime.pose_record(record_id="terminal-training", trajectory_id="terminal-training-trajectory", split="train",
        category="near_upright", motion_kind="dynamic", metadata={"origin": "recovery_terminal"})
    renamed = replace(record, record_id="terminal-testing", trajectory_id="terminal-testing-trajectory", split="test")
    bank = PoseBank([record, renamed], {"purpose": "negative_split_regression"})
    with pytest.raises(ValueError, match="轨迹或物理内容重叠"):
        validate_recovery_terminal_records(runtime, bank)


def test_terminal_validation_requires_acceptance_by_common_thresholds(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    actual = np.array([0.09, -0.07, 0.06, -0.08, 0.04, -0.03])
    runtime.step(actual)
    target_difference, torque_difference = action_differences(runtime, actual, runtime.walker_action())
    # 此处仅验证校准接受集合的逻辑；成功标签不作为真实起身能力证据。
    samples = [{"partition": partition, "source": "recovery_terminal", "takeover": {"success": True},
                "initial_physical_validation": None, "target_difference": target_difference.tolist(),
                "torque_difference_nm": torque_difference.tolist()} for partition in ("train", "test")]
    target_limits = np.maximum(target_difference * 2, 1e-3)
    torque_limits = np.maximum(torque_difference * 2, 1e-3)
    accepted, validated = accepted_samples(samples, target_limits, torque_limits)
    assert len(accepted) == 2 and validated
    samples[1]["target_difference"][4] = target_limits[4]
    accepted, validated = accepted_samples(samples, target_limits, torque_limits)
    assert len(accepted) == 1 and not validated
    samples[1]["target_difference"][4] = target_difference[4]
    samples[1]["takeover"]["success"] = False
    assert not accepted_samples(samples, target_limits, torque_limits)[1]


def test_evaluation_uses_valid_independent_context_and_current_zero_command(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    record = runtime.pose_record(record_id="evaluation-context", trajectory_id="evaluation-trajectory", split="test",
                                 category="near_upright", motion_kind="dynamic", metadata={"origin": "interface_validation"})
    protocol = EvaluationProtocol(name="interface-regression", randomization_profile="nominal", maximum_case_steps=500,
                                   minimum_cases_per_fallen_category=1,
                                   command_segments=[{"command": [0.4, 0.3, 0.20], "steps": 20}], diagnostic=True)
    independent = create_case_supervisor(record, protocol, "standalone")
    assert independent.context.stage == "independent"
    assert independent.snapshot().recovery_steps == 0
    combined = create_case_supervisor(record, protocol, "combined")
    current = combined.snapshot()
    assert current.zero_command
    command, index, step = select_source_command(current, protocol, 0, 13)
    np.testing.assert_array_equal(command, [0.0, 0.0, 0.20])
    assert (index, step) == (0, 13)


def test_test_bank_identity_changes_cannot_reclassify_training_physical_state(runtime):
    for _ in range(5):
        runtime.step(runtime.walker_action(), locomotion=True)
    record = runtime.pose_record(record_id="physical-train-record", trajectory_id="physical-train-trajectory", split="train",
                                 category="near_upright", motion_kind="dynamic", metadata={"origin": "interface_validation"})
    source = [{"record_id": record.record_id, "trajectory_id": record.trajectory_id, "split": "train",
               "physical_state_sha256": pose_record_physical_sha256(record)}]
    renamed = replace(record, record_id="new-test-name", trajectory_id="new-test-trajectory", split="test")
    with pytest.raises(ValueError, match="完整物理内容重叠"):
        verify_test_isolation([renamed], source)


def test_bootstrap_collection_keeps_real_categories_histories_and_diagnostic_labels(runtime, directory):
    limits = {}
    for joint in range(runtime.model.njnt):
        if runtime.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        limits[name] = runtime.model.jnt_range[joint].tolist() if runtime.model.jnt_limited[joint] else [None,None]
    config = BootstrapCollectionConfig(seed=440501,training_trajectories_per_category=1,
        testing_trajectories_per_category=1,minimum_record_control_steps=5,maximum_record_control_steps=5)
    classification = BootstrapClassification()
    bank = collect_bootstrap(runtime,config,classification,{'source':'actual_compiled_model_native_joint_ranges',
        'joint_position_limits_native_rad':limits,'mechanical_physical_range_calibrated':False},
        output_dir=directory/f'bootstrap-{time_ns()}')
    assert len(bank.records)==12
    assert len({record.trajectory_id for record in bank.records})==12
    for record in bank.records:
        runtime.restore_pose(record)
        assert record.initial_category==record.metadata['actual_category']
        assert record.metadata['physical_control_steps']==5
        assert record.metadata['physical_substeps_checked']==100
        assert record.metadata['stage']=='bootstrap_diagnostic'
        assert record.metadata['real_nonconvex_validation_pending'] is True
        assert 'physical_validation_passed' not in record.metadata
        gravity=runtime.state().projected_gravity_control
        if record.initial_category=='left':
            assert gravity[1] > .20
        elif record.initial_category=='right':
            assert gravity[1] < -.20
