from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from debug.sim2sim.root_cause_suite.contracts import (
    EvidenceIntegrityError,
    sha256_file,
)
from debug.sim2sim.root_cause_suite.repeatability import (
    build_repeatability_record,
    build_threshold_snapshot,
    frozen_repeat_envelope,
    validate_threshold_snapshot,
)
from debug.sim2sim.root_cause_suite.trace_contract import FieldSpec, write_trace


def _records(*, reset_suffix: str = "same", excitation: float = 0.3):
    return [
        build_repeatability_record(
            engine="mujoco",
            scenario_id="P30_A_OPEN_DIRECT_EFFORT",
            repeatability_family="p30_open_direct",
            configuration_hash="configuration",
            profile_index=repetition,
            repetition=repetition,
            profile_semantics={"channel": 0, "sign": 1},
            pre_forward_payload={"reset": reset_suffix},
            post_forward_payload={"reset": reset_suffix},
            reset_returned_policy_payload=None,
            excitation_semantics={"input": excitation, "channel": 0, "sign": 1},
        )
        for repetition in range(3)
    ]


def _execution(tmp_path: Path, values: np.ndarray, records: list[dict]):
    trace_path = tmp_path / "trace"
    arrays = {
        "time_s": np.asarray(
            [[0.0, 0.0, 0.0], [0.02, 0.02, 0.02]], dtype=np.float64
        ),
        "profile_channel": np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.int16),
        "profile_sign": np.asarray([[1, 1, 1], [1, 1, 1]], dtype=np.int8),
        "profile_repetition": np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int16),
        "controlled_position_canonical": values,
        "controlled_velocity_canonical": values,
    }
    fields = {
        name: FieldSpec(
            "rad/s" if "velocity" in name else "1",
            "canonical",
            "post_step",
            name,
        )
        for name in arrays
    }
    write_trace(trace_path, arrays, fields)
    result = {
        "engine": "mujoco",
        "scenario_id": "P30_A_OPEN_DIRECT_EFFORT",
        "configuration_hash": "configuration",
        "repeatability_records": records,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")
    return {
        "result": result,
        "result_sha256": sha256_file(result_path),
        "trace_path": trace_path,
    }


def test_snapshot_freezes_exact_three_repeat_envelope(tmp_path: Path) -> None:
    values = np.asarray(
        [
            [[0.0], [0.0], [0.0]],
            [[0.1], [0.1004], [0.0996]],
        ],
        dtype=np.float64,
    )
    records = _records()
    snapshot = build_threshold_snapshot([_execution(tmp_path, values, records)])
    validate_threshold_snapshot(snapshot)
    key = records[0]["repeatability_key"]
    frozen = snapshot["keys"][key]
    assert frozen["sample_count"] == 3
    assert frozen["usable"] is True
    assert frozen_repeat_envelope(
        snapshot, records, "controlled_velocity_canonical"
    ) == pytest.approx(8.0e-4)


def test_declared_but_unexecuted_key_is_rejected(tmp_path: Path) -> None:
    values = np.zeros((2, 3, 1), dtype=np.float64)
    records = _records()
    snapshot = build_threshold_snapshot([_execution(tmp_path, values, records)])
    missing = _records(excitation=0.4)
    with pytest.raises(EvidenceIntegrityError, match="unavailable"):
        frozen_repeat_envelope(
            snapshot, missing, "controlled_velocity_canonical"
        )


def test_reset_identity_mutation_cannot_share_one_envelope(tmp_path: Path) -> None:
    values = np.zeros((2, 3, 1), dtype=np.float64)
    records = _records()
    records[2] = build_repeatability_record(
        engine="mujoco",
        scenario_id="P30_A_OPEN_DIRECT_EFFORT",
        repeatability_family="p30_open_direct",
        configuration_hash="configuration",
        profile_index=2,
        repetition=2,
        profile_semantics={"channel": 0, "sign": 1},
        pre_forward_payload={"reset": "changed"},
        post_forward_payload={"reset": "same"},
        reset_returned_policy_payload=None,
        excitation_semantics={"input": 0.3, "channel": 0, "sign": 1},
    )
    with pytest.raises(EvidenceIntegrityError, match="exact repetitions"):
        build_threshold_snapshot([_execution(tmp_path, values, records)])


def test_high_noise_key_is_unusable_not_posthoc_widened(tmp_path: Path) -> None:
    values = np.asarray(
        [
            [[0.0], [0.0], [0.0]],
            [[0.0], [0.02], [-0.02]],
        ],
        dtype=np.float64,
    )
    records = _records()
    snapshot = build_threshold_snapshot([_execution(tmp_path, values, records)])
    key = records[0]["repeatability_key"]
    assert snapshot["keys"][key]["usable"] is False
    assert "nondeterministic:controlled_velocity_canonical" in snapshot["keys"][
        key
    ]["unusable_reasons"]
    with pytest.raises(EvidenceIntegrityError, match="unusable"):
        frozen_repeat_envelope(
            snapshot, records, "controlled_velocity_canonical"
        )


def test_snapshot_identity_tamper_is_rejected(tmp_path: Path) -> None:
    values = np.zeros((2, 3, 1), dtype=np.float64)
    snapshot = build_threshold_snapshot(
        [_execution(tmp_path, values, _records())]
    )
    snapshot["minimum_repetitions"] = 4
    with pytest.raises(EvidenceIntegrityError, match="identity"):
        validate_threshold_snapshot(snapshot)
