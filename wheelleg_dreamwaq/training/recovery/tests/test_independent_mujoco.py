import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recovery.contracts import file_sha256
from recovery.independent_evaluate import IndependentMujocoRuntime, final_suite_cases, physical_initial_variant, wilson
from recovery.independent_poses import IndependentPoseBank
from recovery.independent_state import StandingTask
from recovery.standing_reference import StandingReference


PROJECT = Path(__file__).resolve().parents[3]
ARTIFACT = PROJECT / "recovery/artifacts/independent-stand-implementation-20261029"
GENERATED = ARTIFACT / "pose-generation-check/generated-v3"


@pytest.fixture(scope="module")
def physical_data():
    bank = IndependentPoseBank.load(GENERATED / "pose_bank.json")
    geometry = json.loads((ARTIFACT / "pose-generation-check/geometry_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROJECT / "recovery/artifacts/standing-reference401-20261029/manifest.json").read_text(encoding="utf-8"))
    runtime = IndependentMujocoRuntime(geometry["contact_manifest"], geometry_config=geometry,
                                      q_nominal=manifest["target_contract"]["q_nominal"])
    return bank, runtime


def test_named_full_state_restore_and_com_matches_real_401_and_fallen_states(physical_data):
    bank, runtime = physical_data
    for source in ("front", "back", "left", "right", "supine", "near_upright", "standing", "perturbed"):
        record = next(r for r in bank.records if r["source"] == source and r["split"] == "test")
        runtime.restore_pose(record)
        state = runtime.state()
        np.testing.assert_allclose(state.joint_position_control[:4], record["metadata"]["q_control_rad"], atol=1e-7)
        np.testing.assert_allclose(state.projected_gravity_control, record["metadata"]["projected_gravity"], atol=1e-7)
        assert abs(state.base_height_m - record["metadata"]["com_height_m"]) < 1e-7
        assert runtime.inspect_physics()["loop_error_m"] < .005
        assert runtime.history.shape == (5, 22)
        np.testing.assert_array_equal(runtime.history[0], runtime.history[-1])
        np.testing.assert_allclose(runtime.history[-1, -6:], record["bounded_action"], atol=1e-7)


def test_actual_new_action_history_and_pd_torque_limits_in_six_fallen_categories(physical_data):
    bank, runtime = physical_data
    for source in ("front", "back", "left", "right", "supine", "near_upright"):
        record = next(r for r in bank.records if r["source"] == source and r["split"] == "test")
        for sign in (-1., 1.):
            runtime.restore_pose(record)
            action = np.array([sign] * 4 + [sign, -sign])
            old = runtime.history[-1].copy()
            result = runtime.step(action)
            assert np.isfinite(runtime.data.qpos).all()
            assert np.isfinite(runtime.data.qvel).all()
            assert np.all(np.abs(runtime.last_torque) <= runtime.effort + 1e-8)
            np.testing.assert_array_equal(runtime.previous_action, action)
            np.testing.assert_array_equal(runtime.history[-2], old)
            np.testing.assert_allclose(runtime.history[-1, -6:], action, atol=0.)
            assert result["loop_error_m"] >= 0.


def test_generated_split_sources_and_actual_standing_support_evidence(physical_data):
    bank, runtime = physical_data
    initialization = json.loads((GENERATED / "initialization.json").read_text(encoding="utf-8"))
    report = bank.validate_split_ids(initialization)
    assert report["passed"]
    assert set(initialization["train_ids"]) == {"front", "back", "left", "right", "supine", "standing", "perturbed"}
    for record in bank.records:
        if record["source"] not in ("standing", "perturbed"):
            continue
        runtime.restore_pose(record)
        assert runtime.inspect_physics()["loop_error_m"] < .005
        evidence = record["metadata"]["standing_gate_evidence"]
        assert evidence["archive_sha256"] == file_sha256(evidence["archive"])
        assert evidence["consecutive_actual_hold_steps_at_least"] >= 100
        if record["source"] == "perturbed":
            perturbation = evidence["perturbation"]
            assert abs(perturbation["roll_rad"]) <= .08
            assert abs(perturbation["pitch_rad"]) <= .08
            assert np.max(np.abs(perturbation["leg_delta_rad"])) <= .05
            assert np.max(np.abs(perturbation["linear_velocity_control"][:2])) <= .05


def test_actual_single_wheel_body_inverted_low_and_sliding_never_complete_standing(physical_data):
    _, runtime = physical_data
    root = ARTIFACT / "physical-negative-check-v1"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contact_manifest_sha256"] == file_sha256(runtime.contact_manifest_path)
    source = PROJECT / "recovery/artifacts/standing-reference401-20261029"
    reference_path = source / "recovery_contact_paired_reference.npz"
    reference = StandingReference.load(source / "manifest.json", reference_path, file_sha256(reference_path))
    for episode in manifest["episodes"]:
        path = root / episode["file"]
        assert file_sha256(path) == episode["sha256"]
        task = StandingTask(1, "cpu", reference, runtime.weight_n, mode="eval")
        with np.load(path, allow_pickle=False) as data:
            for index in range(len(data["height_m"])):
                t = lambda key: torch.tensor(data[key][index:index + 1], dtype=torch.float32)
                result = task.step(q=t("q_control_rad")[:, :4], dq=t("dq_control_rad_s"),
                    omega=t("angular_velocity_control_rad_s"), gravity=t("projected_gravity"),
                    linear=t("linear_velocity_control_m_s"), height=t("height_m"),
                    wheel_forces=t("wheel_normal_force_n"), nonwheel_force=t("nonwheel_normal_force_n"),
                    world_xy=t("base_com_world_m")[:, :2], physical_failure=t("physical_failure").bool())
                assert not result.success[0]
        assert episode["required_condition_observed"]


def test_final_suite_has_600_plus_100_and_six_actual_valid_variants(physical_data):
    bank, runtime = physical_data
    cases = final_suite_cases(bank)
    assert len(cases) == 700
    assert len({case["case_id"] for case in cases}) == 700
    assert sum(case["collection"] == "six_category" for case in cases) == 600
    assert sum(case["collection"] == "standing_perturbation" for case in cases) == 100
    for source in ("front", "back", "left", "right", "supine", "near_upright"):
        selected = [case for case in cases if bank.by_id[case["base_record_id"]]["source"] == source]
        assert len(selected) == 100
        variant = physical_initial_variant(runtime, bank.by_id[selected[0]["base_record_id"]], selected[0]["seed"])
        assert variant["physical_validation_passed"]
        assert variant["closed_chain_velocity_residual"] < 1e-5
        assert not runtime.inspect_physics()["physical_failure"]
    assert wilson(0, 0) is None
    assert wilson(0, 100)[0] == 0.
    assert wilson(100, 100)[1] == 1.
