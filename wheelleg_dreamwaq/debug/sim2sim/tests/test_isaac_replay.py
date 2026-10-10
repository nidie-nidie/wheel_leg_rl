from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from debug.sim2sim.compare_isaac_equivalence import (
    EXACT_FIELDS,
    FLOAT_FIELDS,
    ORIENTATION_FIELD,
    SHARED_IDENTITY_FIELDS,
)
from debug.sim2sim.compare_isaac_replay import REPLAY_FLOAT_FIELDS, compare_isaac_replay_runs
from debug.sim2sim.compare_traces import current_engine_source_paths, load_verified_engine_run
from debug.sim2sim.trace_schema import (
    SCHEMA_VERSION,
    build_file_hashes,
    save_npz,
    sha256_file,
    sha256_array,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _control_trace(rows: int = 2) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {
        name: np.zeros((rows, 1), dtype=np.float32) for name in set(REPLAY_FLOAT_FIELDS) | set(FLOAT_FIELDS)
    }
    widths = {
        "actor_obs_policy_pre_step": 25,
        "actor_output_raw": 6,
        "action_clipped": 6,
        "target_command_canonical": 6,
        "target_command_engine_native": 6,
        "active_joint_position_canonical_post_step_pre_reset": 6,
        "active_joint_velocity_canonical_post_step_pre_reset": 6,
        "all_hinge_position_named_post_step_pre_reset": 26,
        "all_hinge_velocity_named_post_step_pre_reset": 26,
        "base_com_position_engine_world_post_step_pre_reset": 3,
        "base_com_position_diag_post_step_pre_reset": 3,
        "base_linear_velocity_control_post_step_pre_reset": 3,
        "base_angular_velocity_control_post_step_pre_reset": 3,
        "projected_gravity_post_step_pre_reset": 3,
        "virtual_leg_length_post_step_pre_reset": 2,
        "virtual_leg_phi0_post_step_pre_reset": 2,
        "loop_closure_error_post_step_pre_reset": 2,
        "next_actor_obs_policy_returned": 25,
        "previous_action_before_inference": 6,
    }
    for name, width in widths.items():
        trace[name] = np.zeros((rows, width), dtype=np.float32)
    trace[ORIENTATION_FIELD] = np.tile(np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (rows, 1))
    trace["control_tick"] = np.arange(rows, dtype=np.int64)
    trace["native_terminated_int8"] = np.zeros(rows, dtype=np.int8)
    trace["native_truncated_int8"] = np.zeros(rows, dtype=np.int8)
    trace["common_diagnostic_flags_int8"] = np.zeros((rows, 8), dtype=np.int8)
    trace["next_obs_is_reset_int8"] = np.zeros(rows, dtype=np.int8)
    return trace


def _substeps(rows: int = 2) -> dict[str, np.ndarray]:
    zeros6 = np.zeros((rows, 6), dtype=np.float64)
    zeros26 = np.zeros((rows, 26), dtype=np.float64)
    return {
        "control_tick": np.arange(rows, dtype=np.int64),
        "substep_index": np.zeros(rows, dtype=np.int16),
        "active_joint_position_canonical_pre_step": zeros6.copy(),
        "active_joint_velocity_canonical_pre_step": zeros6.copy(),
        "active_joint_position_canonical_post_step": zeros6.copy(),
        "active_joint_velocity_canonical_post_step": zeros6.copy(),
        "all_hinge_position_named_pre_step": zeros26.copy(),
        "all_hinge_velocity_named_pre_step": zeros26.copy(),
        "all_hinge_position_named_post_step": zeros26.copy(),
        "all_hinge_velocity_named_post_step": zeros26.copy(),
    }


def _metadata(variant: str, action_identity: dict, collection_id: str) -> dict:
    source_hashes = build_file_hashes(current_engine_source_paths(PROJECT_ROOT, "isaac_sim"))
    metadata = {name: f"fixed-{name}" for name in SHARED_IDENTITY_FIELDS}
    metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "collection_id": collection_id,
            "engine": "isaac_sim",
            "frozen_files": {"actor": "A" * 64},
            "scenario_variant": variant,
            "action_sequence_identity": action_identity,
            "command": [0.0, 0.0, 0.2],
            "random_seed": 0,
            "control_dt_s": 0.02,
            "physics_dt_s": 0.02,
            "physics_steps_per_action": 1,
            "substep_continuity_atol": 0.0,
            "canonical_joint_order": ["j"] * 6,
            "canonical_from_engine_native": [1, 1, 1, 1, 1, -1],
            "all_hinge_order": [f"j{index}" for index in range(26)],
            "r_diag_from_engine_world": np.eye(3).tolist(),
            "solver_position_iterations": 96,
            "solver_velocity_iterations": 4,
            "clone_in_fabric": False,
            "use_fabric": True,
            "render_mode": "headless",
            "contact_observation_mode": "disabled",
            "observer_mode": "debug_step_subclass",
            "formal_source_sha256": {
                name: digest for name, digest in source_hashes.items() if name.startswith("formal_")
            },
            "completed_control_ticks": 2,
            "configured_episode_length_s": 10.02,
        }
    )
    return metadata


def _write_run(path: Path, variant: str, action_identity: dict) -> None:
    path.mkdir()
    trace = _control_trace()
    trace["collection_nonce"] = np.asarray([sum(map(ord, path.name))], dtype=np.int64)
    save_npz(path / "control_trace.npz", trace)
    save_npz(path / "substep_trace.npz", _substeps())
    write_json(path / "metadata.json", _metadata(variant, action_identity, path.name))
    write_json(path / "reset_snapshot.json", {"schema_version": SCHEMA_VERSION})
    hashes = build_file_hashes(
        {
            "metadata": path / "metadata.json",
            "reset_snapshot": path / "reset_snapshot.json",
            "control_trace": path / "control_trace.npz",
            "substep_trace": path / "substep_trace.npz",
        }
    )
    if action_identity.get("content_sha256") is not None:
        action_path = save_npz(
            path / "input_action_sequence.npz",
            {"action_sequence": _control_trace()["action_clipped"]},
        )
        hashes["input_action_sequence"] = sha256_file(action_path)
    hashes.update(build_file_hashes(current_engine_source_paths(PROJECT_ROOT, "isaac_sim")))
    write_json(path / "file_hashes.json", hashes)


def test_replay_gate_accepts_identical_natural_envelopes(tmp_path) -> None:
    original_paths = [tmp_path / "original-1", tmp_path / "original-2"]
    closed_loop_identity = {
        "source": "policy_closed_loop",
        "file_sha256": None,
        "sidecar_sha256": None,
        "key": None,
        "content_sha256": None,
        "rows": None,
        "source_control_trace_sha256": None,
    }
    for path in original_paths:
        _write_run(path, "closed_loop", closed_loop_identity)
    source_hash = sha256_file(original_paths[0] / "control_trace.npz")
    replay_identity = {
        "source": "isaac_policy_replay_npz",
        "file_sha256": "3" * 64,
        "sidecar_sha256": "4" * 64,
        "key": "isaac_policy_replay",
        "content_sha256": sha256_array(_control_trace()["action_clipped"]),
        "rows": 2,
        "source_control_trace_sha256": source_hash,
    }
    replay_paths = [tmp_path / "replay-1", tmp_path / "replay-2"]
    for path in replay_paths:
        _write_run(path, "isaac_policy_replay", replay_identity)
    report = compare_isaac_replay_runs(original_paths, replay_paths)
    assert report["passed"]
    assert not report["replay_identity_failures"]


def test_verified_run_rejects_trace_modified_after_hashing(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run, "closed_loop", {"source": "policy_closed_loop"})
    trace = _control_trace()
    trace["action_clipped"][0, 0] = 1.0
    save_npz(run / "control_trace.npz", trace)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_verified_engine_run(run)


def test_verified_run_rejects_missing_control_schema_field(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run, "closed_loop", {"source": "policy_closed_loop"})
    trace = _control_trace()
    del trace["next_actor_obs_policy_returned"]
    save_npz(run / "control_trace.npz", trace)
    hashes = json.loads((run / "file_hashes.json").read_text(encoding="utf-8"))
    hashes["control_trace"] = sha256_file(run / "control_trace.npz")
    write_json(run / "file_hashes.json", hashes)
    with pytest.raises(ValueError, match="missing required fields"):
        load_verified_engine_run(run)


def test_verified_run_rejects_source_hash_not_bound_to_workspace(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run, "closed_loop", {"source": "policy_closed_loop"})
    hashes = json.loads((run / "file_hashes.json").read_text(encoding="utf-8"))
    hashes["collector"] = "A" * 64
    write_json(run / "file_hashes.json", hashes)
    with pytest.raises(ValueError, match="current workspace"):
        load_verified_engine_run(run)
