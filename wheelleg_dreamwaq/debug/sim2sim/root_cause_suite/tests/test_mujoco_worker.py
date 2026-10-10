from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from debug.sim2sim.root_cause_suite.causal_contract import validate_ratio_pair
from debug.sim2sim.root_cause_suite.contracts import (
    EvidenceIntegrityError,
    canonical_json_bytes,
    sha256_file,
    stable_hash,
)
from debug.sim2sim.root_cause_suite.mujoco_worker import (
    FORMAL_MODEL,
    OPTION_FIELDS,
    collect_adapter_probe,
    collect_properties,
    collect_instrumentation_probe,
    collect_replay,
    collect_robot_probe,
    collect_robot_probe_batch,
    collect_sphere_probe,
    collect_sphere_probe_batch,
    compiled_option_snapshot,
    direct_effort_roundtrip,
    run_golden,
    validate_direct_effort_bypass,
)
from debug.sim2sim.root_cause_suite.repeatability import build_threshold_snapshot
from debug.sim2sim.root_cause_suite import mujoco_worker
from debug.sim2sim.root_cause_suite.variant_builders import build_robot_variant


def _rebind_configuration_identity(payload: dict) -> dict:
    rebound = deepcopy(payload)
    rebound["configuration_semantics_hash"] = stable_hash(
        rebound["configuration_semantics"]
    )
    rebound["configuration_hash"] = stable_hash(
        {
            "engine": rebound["engine"],
            "source_model_sha256": rebound["source_model_sha256"],
            "model_artifact_sha256": rebound["model_artifact_sha256"],
            "transform_manifest_sha256": rebound["transform_manifest_sha256"],
            "transform_semantics_hash": rebound["transform_semantics_hash"],
            "configuration_semantics_hash": rebound[
                "configuration_semantics_hash"
            ],
            "worker_source_sha256": rebound["worker_source_sha256"],
        }
    )
    rebound["repeatability_key"] = stable_hash(
        {
            "configuration_hash": rebound["configuration_hash"],
            "pre_forward_initial_condition_hash": rebound[
                "pre_forward_initial_condition_hash"
            ],
            "post_forward_state_hash": rebound["post_forward_state_hash"],
            "reset_returned_policy_hash": rebound.get(
                "reset_returned_policy_hash"
            ),
            "excitation_hash": rebound["excitation_hash"],
        }
    )
    return rebound


def test_compiled_option_snapshot_has_exact_frozen_field_set() -> None:
    model = mujoco.MjModel.from_xml_path(str(FORMAL_MODEL))
    snapshot = compiled_option_snapshot(model)
    assert tuple(snapshot) == OPTION_FIELDS
    assert snapshot["solver"]["symbol"] == "mjSOL_NEWTON"


def test_properties_match_source_and_total_mass(tmp_path: Path) -> None:
    result = collect_properties(tmp_path / "properties")
    assert result["passed"] is True
    assert result["body_count"] == 27
    assert abs(result["total_mass_kg"] - 4.396253988146782) <= 1.0e-12


def test_single_body_golden_passes(tmp_path: Path) -> None:
    result = run_golden(tmp_path / "golden")
    assert result["passed"] is True
    assert max(result["errors"].values()) <= 1.0e-10


def test_instrumentation_observer_is_neutral(tmp_path: Path) -> None:
    result = collect_instrumentation_probe(
        tmp_path / "instrumentation", repetitions=5, ticks=10
    )
    assert result["passed"] is True
    assert result["identity_equal"] is True
    assert all(result["exact_fields"].values())
    assert max(result["maximum_errors"].values()) <= 1.0e-12


def test_adapter_probe_uses_real_runtime_and_six_channel_pulses(
    tmp_path: Path,
) -> None:
    result = collect_adapter_probe(tmp_path / "adapter")
    current = np.asarray(
        result["reset_phases"]["returned_policy"]["actor_obs_current"],
        dtype=np.float32,
    )
    history = np.asarray(
        result["reset_phases"]["returned_policy"]["policy_input"],
        dtype=np.float32,
    )
    assert np.array_equal(history.reshape(5, 25), np.repeat(current[None, :], 5, axis=0))
    assert result["clock"]["physics_dt_s"] == pytest.approx(0.001)
    assert result["clock"]["physics_steps_per_action"] == 20
    assert result["clock"]["time_after_s"] - result["clock"]["time_before_s"] == pytest.approx(
        0.02
    )
    assert result["clock"]["target_refresh_first_substep"] is True
    assert result["clock"]["targets_constant_within_action"] is True
    assert np.allclose(
        result["actor_chain"]["previous_action_after"],
        result["actor_chain"]["clipped_action"],
        atol=0.0,
        rtol=0.0,
    )
    assert set(result["pulse_records"]) == {str(index) for index in range(6)}
    assert all(
        record["driven_channel_target_sign"] == 1
        and record["driven_channel_velocity_sign"] == 1
        for record in result["pulse_records"].values()
    )
    right = result["pulse_records"]["5"]
    assert right["plus"]["target_engine_native"][5] < 0.0


def test_drive_off_direct_effort_round_trip(tmp_path: Path) -> None:
    variant = build_robot_variant(
        FORMAL_MODEL,
        tmp_path / "model" / "drive-off.xml",
        operations=("no_ground", "gravity_off", "closure_off", "drive_off", "fixed_base"),
    )
    model = mujoco.MjModel.from_xml_path(str(variant.model_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "contract_v4_reset")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    assert validate_direct_effort_bypass(model)["hinge_count"] == 26
    result = direct_effort_roundtrip(
        model, data, np.asarray([1, -2, 3, -4, 5, -6], dtype=np.float64)
    )
    assert max(result["errors"].values()) <= 1.0e-12


def test_zero_gravity_robot_probe_writes_verified_trace(tmp_path: Path) -> None:
    before = sha256_file(FORMAL_MODEL)
    result = collect_robot_probe(tmp_path / "probe", scenario="p10_a", steps=5)
    assert result["steps"] == 5
    assert result["direct_effort_bypass"]["hinge_count"] == 26
    batch_result = collect_robot_probe_batch(
        tmp_path / "batch",
        scenario="p10_a",
        control_ticks=1,
        repetitions=3,
    )
    contact_exclusion = batch_result["configuration_semantics"]["contact"][
        "contact_exclusion"
    ]
    assert contact_exclusion["contact_free"] is True
    assert contact_exclusion["collision_enabled_geoms"] == []
    assert contact_exclusion["explicit_contact_pairs"] == []
    assert sha256_file(FORMAL_MODEL) == before
    metadata = json.loads(
        (tmp_path / "probe" / "trace" / "trace.json").read_text(encoding="utf-8")
    )
    assert metadata["row_count"] == 6


def test_sphere_impact_reaches_contact(tmp_path: Path) -> None:
    result = collect_sphere_probe(
        tmp_path / "sphere",
        mode="impact",
        steps=500,
        height_m=0.05,
        vertical_velocity_mps=0.0,
        friction=0.0,
    )
    assert result["first_contact_time_s"] is not None
    assert 0.05 < result["first_contact_time_s"] < 0.2


def test_sphere_batch_uses_frozen_profile_matrix(tmp_path: Path) -> None:
    result = collect_sphere_probe_batch(
        tmp_path / "sphere-batch",
        mode="slide",
        friction=0.0,
        repetitions=3,
        duration_s=0.01,
    )
    assert result["profile_count"] == 9
    assert result["physics_dt_s"] == pytest.approx(0.001)
    assert result["valid_contact_count"] == 9
    trace = np.load(tmp_path / "sphere-batch" / "trace" / "trace.npz")
    assert trace["com_velocity_world"].shape == (11, 9, 3)


def test_real_compiled_ratio_pairs_and_hidden_covariates(tmp_path: Path) -> None:
    p30_target = collect_robot_probe_batch(
        tmp_path / "p30-target",
        scenario="p30_open_target",
        control_ticks=1,
        repetitions=3,
    )
    p30_direct = collect_robot_probe_batch(
        tmp_path / "p30-direct",
        scenario="p30_open_direct",
        control_ticks=1,
        repetitions=3,
    )
    p30 = validate_ratio_pair(
        p30_target,
        p30_direct,
        pair_id="P30_OPEN_TARGET_TO_DIRECT",
    )
    assert len(p30["semantic_diff_paths"]) == 23
    p30_repeatability = build_threshold_snapshot(
        [
            {
                "result": p30_direct,
                "result_sha256": sha256_file(
                    tmp_path / "p30-direct" / "result.json"
                ),
                "trace_path": tmp_path / "p30-direct" / "trace",
            }
        ]
    )
    assert len(p30_repeatability["keys"]) == 18
    assert all(
        record["sample_count"] == 3
        for record in p30_repeatability["keys"].values()
    )

    option_mutation = deepcopy(p30_direct)
    option_mutation["configuration_semantics"]["engine_options"]["integrator"][
        "value"
    ] = 99
    with pytest.raises(EvidenceIntegrityError, match="semantic diff"):
        validate_ratio_pair(
            p30_target,
            _rebind_configuration_identity(option_mutation),
            pair_id="P30_OPEN_TARGET_TO_DIRECT",
        )

    p40_on = collect_robot_probe_batch(
        tmp_path / "p40-on",
        scenario="p40_on",
        control_ticks=1,
        repetitions=3,
    )
    p40_off = collect_robot_probe_batch(
        tmp_path / "p40-off",
        scenario="p40_off",
        control_ticks=1,
        repetitions=3,
    )
    validate_ratio_pair(
        p40_on,
        p40_off,
        pair_id="P40_CLOSURE_ON_TO_OFF",
    )
    constraint_mutation = deepcopy(p40_off)
    first_constraint = next(
        iter(constraint_mutation["configuration_semantics"]["closure"]["constraints"])
    )
    constraint_mutation["configuration_semantics"]["closure"]["constraints"][
        first_constraint
    ]["solref"][0] = 123.0
    with pytest.raises(EvidenceIntegrityError, match="semantic diff"):
        validate_ratio_pair(
            p40_on,
            _rebind_configuration_identity(constraint_mutation),
            pair_id="P40_CLOSURE_ON_TO_OFF",
        )

    p50_nominal = collect_sphere_probe_batch(
        tmp_path / "p50-nominal",
        mode="slide",
        friction=1.0,
        repetitions=3,
        duration_s=0.01,
    )
    p50_zero = collect_sphere_probe_batch(
        tmp_path / "p50-zero",
        mode="slide",
        friction=0.0,
        repetitions=3,
        duration_s=0.01,
    )
    p50 = validate_ratio_pair(
        p50_nominal,
        p50_zero,
        pair_id="P50_NOMINAL_TO_ZERO_FRICTION",
    )
    assert p50["semantic_diff_paths"] == [
        "/contact/pairs/floor_coupon/friction/0",
        "/contact/pairs/floor_coupon/friction/1",
    ]
    contact_mutation = deepcopy(p50_zero)
    contact_mutation["configuration_semantics"]["contact"]["pairs"][
        "floor_coupon"
    ]["solref"][0] = 123.0
    with pytest.raises(EvidenceIntegrityError, match="semantic diff"):
        validate_ratio_pair(
            p50_nominal,
            _rebind_configuration_identity(contact_mutation),
            pair_id="P50_NOMINAL_TO_ZERO_FRICTION",
        )


def test_replay_loads_bound_npz_action_sequence(tmp_path: Path) -> None:
    import hashlib

    source = tmp_path / "source"
    source.mkdir()
    actions = np.zeros((3, 6), dtype=np.float32)
    np.savez(
        source / "actions.npz",
        action_sequence=actions,
        environment_action_sequence=np.zeros((3, 8, 6), dtype=np.float32),
    )
    replay_identity = {"horizon": 3, "command": [0.0, 0.0, 0.2]}
    source_result = {
        "schema_version": "RootCauseIsaacReplaySourceV1",
        "replay_source_identity": replay_identity,
        "replay_source_identity_hash": stable_hash(replay_identity),
        "source_trace_sha256": "A" * 64,
        "action_sequence_file": "actions.npz",
        "clipped_action_file_sha256": sha256_file(source / "actions.npz"),
        "clipped_action_sequence_sha256": hashlib.sha256(
            np.ascontiguousarray(actions).tobytes()
        ).hexdigest().upper(),
    }
    (source / "result.json").write_bytes(canonical_json_bytes(source_result) + b"\n")

    result = collect_replay(
        tmp_path / "mujoco-replay",
        source / "actions.npz",
        np.asarray((0.0, 0.0, 0.2), dtype=np.float64),
        source / "result.json",
    )

    assert result["replay_source_identity_hash"] == stable_hash(replay_identity)
    assert result["action_count"] == 3
    trace = np.load(tmp_path / "mujoco-replay" / "trace" / "trace.npz")
    assert trace["actor_obs_policy_pre_step"].shape == (3, 125)
    assert trace["actor_obs_current_pre_step"].shape == (3, 25)


def test_replay_cli_keeps_subcommand_separate_from_command_vector() -> None:
    args = mujoco_worker._parser().parse_args(
        [
            "replay",
            "--output",
            "out",
            "--actions",
            "actions.npz",
            "--source-result",
            "result.json",
            "--command",
            "0",
            "0",
            "0.2",
        ]
    )
    assert args.command == "replay"
    assert args.command_vector == [0.0, 0.0, 0.2]
