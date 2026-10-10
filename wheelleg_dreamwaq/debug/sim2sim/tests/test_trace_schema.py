from __future__ import annotations

import json

import numpy as np
import pytest

from debug.sim2sim.trace_schema import (
    SCHEMA_VERSION,
    canonical_to_engine_native,
    create_run_directory,
    engine_native_to_canonical,
    load_npz,
    load_action_sequence,
    save_npz,
    sha256_array,
    stable_payload_hash,
    validate_control_trace,
    validate_substep_trace,
)


def _valid_substep_trace(rows: int = 2) -> dict[str, np.ndarray]:
    zeros6 = np.zeros((rows, 6), dtype=np.float64)
    zeros26 = np.zeros((rows, 26), dtype=np.float64)
    trace = {
        "control_tick": np.zeros(rows, dtype=np.int64),
        "substep_index": np.arange(rows, dtype=np.int16),
        "active_joint_position_canonical_pre_step": zeros6.copy(),
        "active_joint_velocity_canonical_pre_step": zeros6.copy(),
        "active_joint_position_canonical_post_step": zeros6.copy(),
        "active_joint_velocity_canonical_post_step": zeros6.copy(),
        "all_hinge_position_named_pre_step": zeros26.copy(),
        "all_hinge_velocity_named_pre_step": zeros26.copy(),
        "all_hinge_position_named_post_step": zeros26.copy(),
        "all_hinge_velocity_named_post_step": zeros26.copy(),
    }
    return trace


def test_controlled_native_round_trip() -> None:
    native = np.array([1, 2, 3, 4, 5, -6], dtype=np.float64)
    canonical = engine_native_to_canonical(native)
    np.testing.assert_array_equal(canonical, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(canonical_to_engine_native(canonical), native)


def test_stable_payload_hash_ignores_json_key_order() -> None:
    assert stable_payload_hash({"b": 2, "a": 1}) == stable_payload_hash({"a": 1, "b": 2})


def test_array_hash_includes_dtype_shape_and_content() -> None:
    values = np.arange(6, dtype=np.float32).reshape(1, 6)
    assert sha256_array(values) == sha256_array(values.copy())
    assert sha256_array(values) != sha256_array(values.astype(np.float64))
    assert sha256_array(values) != sha256_array(values.reshape(6))


def test_replay_action_loader_validates_sidecar(tmp_path) -> None:
    replay = tmp_path / "replay.npz"
    actions = np.arange(12, dtype=np.float32).reshape(2, 6)
    save_npz(replay, {"isaac_policy_replay": actions})
    sidecar = {
        "schema_version": "IsaacPolicyReplayActionsV1",
        "key": "isaac_policy_replay",
        "control_ticks": 2,
        "source_control_trace_sha256": "A" * 64,
        "output_sha256": __import__("debug.sim2sim.trace_schema", fromlist=["sha256_file"]).sha256_file(replay),
    }
    replay.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    loaded, identity = load_action_sequence(
        replay,
        "isaac_policy_replay",
        requested_ticks=2,
        require_replay_sidecar=True,
    )
    np.testing.assert_array_equal(loaded, actions)
    assert identity["source"] == "isaac_policy_replay_npz"
    assert identity["source_control_trace_sha256"] == "A" * 64


def test_replay_action_loader_rejects_wrong_key_metadata(tmp_path) -> None:
    replay = tmp_path / "replay.npz"
    actions = np.zeros((2, 6), dtype=np.float32)
    save_npz(replay, {"isaac_policy_replay": actions})
    replay.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": "IsaacPolicyReplayActionsV1",
                "key": "wrong",
                "control_ticks": 2,
                "source_control_trace_sha256": "A" * 64,
                "output_sha256": __import__("debug.sim2sim.trace_schema", fromlist=["sha256_file"]).sha256_file(replay),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar mismatch"):
        load_action_sequence(
            replay,
            "isaac_policy_replay",
            requested_ticks=2,
            require_replay_sidecar=True,
        )


def test_action_loader_rejects_non_float32_sequence(tmp_path) -> None:
    actions = tmp_path / "actions.npz"
    save_npz(actions, {"sequence": np.zeros((2, 6), dtype=np.float64)})
    with pytest.raises(TypeError, match="float32"):
        load_action_sequence(actions, "sequence", requested_ticks=2)


def test_npz_rejects_object_arrays_and_loads_without_pickle(tmp_path) -> None:
    output = tmp_path / "trace.npz"
    save_npz(output, {"value": np.arange(3, dtype=np.float32)})
    np.testing.assert_array_equal(load_npz(output)["value"], np.arange(3, dtype=np.float32))
    with pytest.raises(TypeError, match="numeric"):
        save_npz(output, {"bad": np.array([{"x": 1}], dtype=object)})


def test_create_run_directory_never_overwrites(tmp_path) -> None:
    first = create_run_directory(tmp_path, "fixed")
    assert first.name == "fixed"
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "fixed")


def test_substep_continuity_rejects_gap() -> None:
    trace = _valid_substep_trace()
    trace["active_joint_position_canonical_pre_step"][1, 0] = 0.1
    with pytest.raises(ValueError, match="substep continuity"):
        validate_substep_trace(trace, physics_steps_per_action=2, continuity_atol=1.0e-12)


def test_substep_shape_and_schema_validation_passes() -> None:
    trace = _valid_substep_trace()
    validate_substep_trace(trace, physics_steps_per_action=2, continuity_atol=1.0e-12)
    assert SCHEMA_VERSION == "Sim2SimDebugTraceV1"


def test_control_trace_accepts_dreamwaq_history_dimension() -> None:
    rows = 2
    trace = {
        "control_tick": np.arange(rows, dtype=np.int64),
        "actor_obs_policy_pre_step": np.zeros((rows, 125), dtype=np.float32),
        "actor_output_raw": np.zeros((rows, 6), dtype=np.float64),
        "action_clipped": np.zeros((rows, 6), dtype=np.float64),
        "target_command_canonical": np.zeros((rows, 6), dtype=np.float64),
        "target_command_engine_native": np.zeros((rows, 6), dtype=np.float64),
        "active_joint_position_canonical_post_step_pre_reset": np.zeros((rows, 6)),
        "active_joint_velocity_canonical_post_step_pre_reset": np.zeros((rows, 6)),
        "common_diagnostic_flags_int8": np.zeros((rows, 8), dtype=np.int8),
        "next_actor_obs_policy_returned": np.zeros((rows, 125), dtype=np.float32),
        "next_obs_is_reset_int8": np.zeros(rows, dtype=np.int8),
    }
    validate_control_trace(trace, policy_observation_dimension=125)


def test_saved_metadata_is_plain_json(tmp_path) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "value": 1}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == payload
