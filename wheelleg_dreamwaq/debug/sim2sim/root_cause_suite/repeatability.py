from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import EvidenceIntegrityError, stable_hash
from .trace_contract import VerifiedTrace, load_verified_trace


RECORD_SCHEMA_VERSION = "RootCauseRepeatabilityRecordV1"
SNAPSHOT_SCHEMA_VERSION = "RootCauseThresholdSnapshotV2"
MINIMUM_REPETITIONS = 3

PROFILE_FIELDS = {
    "profile_channel",
    "profile_sign",
    "profile_repetition",
    "profile_height_m",
    "profile_vertical_velocity_mps",
    "profile_horizontal_velocity_mps",
}

MATERIAL_FLOORS: dict[str, float] = {
    "actor_obs_current_pre_step": 1.0e-6,
    "actor_obs_policy_pre_step": 1.0e-6,
    "raw_action": 1.0e-6,
    "clipped_action": 1.0e-6,
    "commanded_input_canonical": 1.0e-6,
    "canonical_torque_equivalent": 1.0e-6,
    "host_applied_torque_canonical": 1.0e-6,
    "controlled_position_canonical": 1.0e-3,
    "controlled_position_canonical_post_step": 1.0e-3,
    "all_hinge_position": 1.0e-3,
    "controlled_velocity_canonical": 1.0e-2,
    "controlled_velocity_canonical_post_step": 1.0e-2,
    "all_hinge_velocity": 1.0e-2,
    "base_com_position_diag": 1.0e-3,
    "system_com_position_control": 1.0e-3,
    "com_position_world": 1.0e-3,
    "base_height_post_step": 1.0e-3,
    "base_linear_velocity_control": 2.0e-2,
    "base_linear_velocity_control_post_step": 2.0e-2,
    "com_velocity_world": 2.0e-2,
    "base_angular_velocity_control": 2.0e-2,
    "base_angular_velocity_control_post_step": 2.0e-2,
    "angular_velocity_world": 2.0e-2,
    "projected_gravity": 1.0e-3,
    "virtual_leg_phi0": 1.0e-3,
    "virtual_leg_length": 1.0e-4,
    "closure_residual_m": 1.0e-4,
    "loop_closure_error_post_step": 1.0e-4,
    "first_contact_time_s": 5.0e-3,
    "normal_impulse_ns": 2.0e-2,
}

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "P10_A_REST_CLOSURE_ON": (
        "all_hinge_velocity",
        "system_com_position_control",
        "closure_residual_m",
    ),
    "P10_B_FREEFALL_CLOSURE_ON": (
        "all_hinge_velocity",
        "system_com_position_control",
        "closure_residual_m",
    ),
    "P10_C_REST_CLOSURE_OFF": (
        "all_hinge_velocity",
        "system_com_position_control",
    ),
    "P30_A_OPEN_DIRECT_EFFORT": (
        "controlled_position_canonical",
        "controlled_velocity_canonical",
    ),
    "P30_B_OPEN_FORMAL_TARGET": (
        "controlled_position_canonical",
        "controlled_velocity_canonical",
    ),
    "P30_C_CLOSED_DIRECT_EFFORT": (
        "controlled_position_canonical",
        "controlled_velocity_canonical",
        "closure_residual_m",
    ),
    "P30_C_CLOSED_FORMAL_TARGET": (
        "controlled_position_canonical",
        "controlled_velocity_canonical",
        "closure_residual_m",
    ),
    "P40_B_CLOSURE_ON": ("all_hinge_velocity", "closure_residual_m"),
    "P40_B_CLOSURE_OFF": ("all_hinge_velocity",),
    "P50_A_SPHERE_IMPACT": (
        "com_position_world",
        "com_velocity_world",
        "first_contact_time_s",
    ),
    "P50_C_NOMINAL_FRICTION": (
        "com_velocity_world",
        "normal_impulse_ns",
    ),
    "P50_C_ZERO_FRICTION": (
        "com_velocity_world",
        "normal_impulse_ns",
    ),
    "P60_B_ZERO_ACTION_DRIVE_OFF": ("base_angular_velocity_control",),
    "P60_C_SHARED_FIRST_ACTION": ("controlled_velocity_canonical",),
    "P60_D_OPEN_LOOP_REPLAY": (
        "actor_obs_current_pre_step",
        "actor_obs_policy_pre_step",
        "clipped_action",
        "controlled_velocity_canonical_post_step",
        "base_angular_velocity_control_post_step",
    ),
}


def _repeatability_key(
    *,
    configuration_hash: str,
    pre_forward_initial_condition_hash: str,
    post_forward_state_hash: str,
    reset_returned_policy_hash: str | None,
    excitation_hash: str,
) -> str:
    return stable_hash(
        {
            "configuration_hash": configuration_hash,
            "pre_forward_initial_condition_hash": pre_forward_initial_condition_hash,
            "post_forward_state_hash": post_forward_state_hash,
            "reset_returned_policy_hash": reset_returned_policy_hash,
            "excitation_hash": excitation_hash,
        }
    )


def build_repeatability_record(
    *,
    engine: str,
    scenario_id: str,
    repeatability_family: str,
    configuration_hash: str,
    profile_index: int,
    repetition: int,
    profile_semantics: Mapping[str, Any],
    pre_forward_payload: Mapping[str, Any],
    post_forward_payload: Mapping[str, Any],
    reset_returned_policy_payload: Any | None,
    excitation_semantics: Mapping[str, Any],
) -> dict[str, Any]:
    if engine not in {"isaac", "mujoco"}:
        raise ValueError(f"Unknown repeatability engine: {engine}")
    if not repeatability_family:
        raise ValueError("Repeatability family must be non-empty")
    if profile_index < 0 or repetition < 0:
        raise ValueError("Profile index and repetition must be non-negative")
    pre_hash = stable_hash(pre_forward_payload)
    post_hash = stable_hash(post_forward_payload)
    returned_hash = (
        None
        if reset_returned_policy_payload is None
        else stable_hash(reset_returned_policy_payload)
    )
    excitation_hash = stable_hash(excitation_semantics)
    key = _repeatability_key(
        configuration_hash=configuration_hash,
        pre_forward_initial_condition_hash=pre_hash,
        post_forward_state_hash=post_hash,
        reset_returned_policy_hash=returned_hash,
        excitation_hash=excitation_hash,
    )
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "engine": engine,
        "scenario_id": scenario_id,
        "repeatability_family": repeatability_family,
        "configuration_hash": configuration_hash,
        "profile_index": int(profile_index),
        "repetition": int(repetition),
        "profile_semantics": dict(profile_semantics),
        "profile_identity_hash": stable_hash(profile_semantics),
        "pre_forward_initial_condition_hash": pre_hash,
        "post_forward_state_hash": post_hash,
        "reset_returned_policy_hash": returned_hash,
        "excitation_semantics": dict(excitation_semantics),
        "excitation_hash": excitation_hash,
        "repeatability_key": key,
    }
    record["record_identity_hash"] = stable_hash(record)
    return record


def validate_repeatability_record(
    record: Mapping[str, Any],
    *,
    expected_engine: str | None = None,
    expected_scenario_id: str | None = None,
    expected_configuration_hash: str | None = None,
) -> None:
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise EvidenceIntegrityError("Repeatability record schema mismatch")
    unsigned = dict(record)
    identity_hash = unsigned.pop("record_identity_hash", None)
    if identity_hash != stable_hash(unsigned):
        raise EvidenceIntegrityError("Repeatability record identity mismatch")
    if stable_hash(record.get("profile_semantics")) != record.get(
        "profile_identity_hash"
    ):
        raise EvidenceIntegrityError("Repeatability profile identity mismatch")
    if stable_hash(record.get("excitation_semantics")) != record.get(
        "excitation_hash"
    ):
        raise EvidenceIntegrityError("Repeatability excitation identity mismatch")
    expected_key = _repeatability_key(
        configuration_hash=str(record.get("configuration_hash")),
        pre_forward_initial_condition_hash=str(
            record.get("pre_forward_initial_condition_hash")
        ),
        post_forward_state_hash=str(record.get("post_forward_state_hash")),
        reset_returned_policy_hash=record.get("reset_returned_policy_hash"),
        excitation_hash=str(record.get("excitation_hash")),
    )
    if expected_key != record.get("repeatability_key"):
        raise EvidenceIntegrityError("Repeatability key mismatch")
    if expected_engine is not None and record.get("engine") != expected_engine:
        raise EvidenceIntegrityError("Repeatability record engine mismatch")
    if expected_scenario_id is not None and record.get(
        "scenario_id"
    ) != expected_scenario_id:
        raise EvidenceIntegrityError("Repeatability record scenario mismatch")
    if expected_configuration_hash is not None and record.get(
        "configuration_hash"
    ) != expected_configuration_hash:
        raise EvidenceIntegrityError("Repeatability record configuration mismatch")


def _trace_time_vector(trace: VerifiedTrace) -> np.ndarray:
    for name in ("time_s", "control_time_s"):
        if name not in trace.arrays:
            continue
        values = np.asarray(trace.arrays[name], dtype=np.float64)
        if values.ndim == 1:
            return values
        if values.ndim == 2 and np.allclose(
            values, values[:, :1], rtol=0.0, atol=1.0e-12
        ):
            return values[:, 0]
        raise EvidenceIntegrityError(f"Trace {name} is not one shared time vector")
    raise EvidenceIntegrityError("Trace has no supported time vector")


def _first_contact_time(trace: VerifiedTrace, profile_index: int) -> float | None:
    if "contact_count" not in trace.arrays:
        return None
    contact = np.asarray(trace.arrays["contact_count"])
    if contact.ndim != 2 or profile_index >= contact.shape[1]:
        raise EvidenceIntegrityError("Contact trace profile axis is invalid")
    indices = np.flatnonzero(contact[:, profile_index] > 0)
    if not indices.size:
        return None
    return float(_trace_time_vector(trace)[int(indices[0])])


def _pairwise_envelope(values: Sequence[np.ndarray]) -> tuple[list[float], float]:
    if len(values) < MINIMUM_REPETITIONS:
        raise EvidenceIntegrityError("Repeatability key has fewer than three samples")
    shapes = {tuple(np.asarray(value).shape) for value in values}
    if len(shapes) != 1:
        raise EvidenceIntegrityError("Repeatability samples have different shapes")
    differences = [
        float(np.max(np.abs(np.asarray(values[left], dtype=np.float64) - np.asarray(values[right], dtype=np.float64))))
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    return differences, max(differences, default=0.0)


def _profile_axis_size(trace: VerifiedTrace) -> int:
    if "profile_repetition" not in trace.arrays:
        return 1
    values = np.asarray(trace.arrays["profile_repetition"])
    if values.ndim == 1:
        return int(values.shape[0])
    if values.ndim == 2 and np.array_equal(
        values, np.broadcast_to(values[:1], values.shape)
    ):
        return int(values.shape[1])
    raise EvidenceIntegrityError("Trace profile repetition field is invalid")


def _profile_value(trace: VerifiedTrace, field: str, profile_index: int) -> np.ndarray:
    values = np.asarray(trace.arrays[field])
    profile_count = _profile_axis_size(trace)
    if profile_count == 1 and field not in PROFILE_FIELDS:
        return np.asarray(values)
    if values.ndim < 2 or values.shape[1] != profile_count:
        raise EvidenceIntegrityError(
            f"Trace field {field} has no exact profile axis"
        )
    return np.asarray(values[:, profile_index])


def build_threshold_snapshot(
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    samples_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    execution_rows: list[dict[str, Any]] = []
    for execution in executions:
        result = execution.get("result")
        if not isinstance(result, Mapping):
            raise EvidenceIntegrityError("Repeatability execution result is missing")
        trace_path = Path(str(execution.get("trace_path"))).resolve(strict=True)
        trace = load_verified_trace(trace_path)
        records = result.get("repeatability_records")
        if not isinstance(records, list) or not records:
            raise EvidenceIntegrityError("Executed probe has no repeatability records")
        if len(records) != _profile_axis_size(trace):
            raise EvidenceIntegrityError(
                "Repeatability record count differs from the trace profile axis"
            )
        engine = str(result.get("engine"))
        scenario_id = str(result.get("scenario_id"))
        configuration_hash = str(result.get("configuration_hash"))
        for expected_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise EvidenceIntegrityError("Repeatability record is not an object")
            validate_repeatability_record(
                record,
                expected_engine=engine,
                expected_scenario_id=scenario_id,
                expected_configuration_hash=configuration_hash,
            )
            if int(record.get("profile_index", -1)) != expected_index:
                raise EvidenceIntegrityError("Repeatability profile index is not exact")
            samples_by_key[str(record["repeatability_key"])].append(
                {"record": dict(record), "trace": trace}
            )
        execution_rows.append(
            {
                "engine": engine,
                "scenario_id": scenario_id,
                "result_sha256": execution.get("result_sha256"),
                "trace_sha256": trace.identity.trace_sha256,
                "trace_metadata_sha256": trace.identity.metadata_sha256,
                "record_count": len(records),
            }
        )

    frozen_keys: dict[str, Any] = {}
    index: dict[str, str] = {}
    for key in sorted(samples_by_key):
        samples = samples_by_key[key]
        records = [sample["record"] for sample in samples]
        repetitions = sorted(int(record["repetition"]) for record in records)
        if repetitions != list(range(MINIMUM_REPETITIONS)):
            raise EvidenceIntegrityError(
                f"Repeatability key {key} does not have exact repetitions 0..2"
            )
        first = records[0]
        invariant_fields = (
            "engine",
            "scenario_id",
            "repeatability_family",
            "configuration_hash",
            "profile_identity_hash",
            "pre_forward_initial_condition_hash",
            "post_forward_state_hash",
            "reset_returned_policy_hash",
            "excitation_hash",
            "repeatability_key",
        )
        for record in records[1:]:
            if any(record[name] != first[name] for name in invariant_fields):
                raise EvidenceIntegrityError(
                    f"Repeatability key {key} mixes incompatible identities"
                )
        available_fields = set(samples[0]["trace"].arrays)
        for sample in samples[1:]:
            available_fields &= set(sample["trace"].arrays)
        fields: dict[str, Any] = {}
        for field in sorted(available_fields - PROFILE_FIELDS - {"time_s", "control_time_s"}):
            values = [
                _profile_value(
                    sample["trace"],
                    field,
                    int(sample["record"]["profile_index"]),
                )
                for sample in samples
            ]
            pairwise, envelope = _pairwise_envelope(values)
            floor = MATERIAL_FLOORS.get(field)
            fields[field] = {
                "pairwise_max_abs": pairwise,
                "envelope": envelope,
                "material_floor": floor,
                "usable": None if floor is None else bool(envelope <= floor),
            }
        contact_times = [
            _first_contact_time(
                sample["trace"], int(sample["record"]["profile_index"])
            )
            for sample in samples
        ]
        if any(value is not None for value in contact_times):
            if any(value is None for value in contact_times):
                pairwise = []
                envelope = float("inf")
            else:
                pairwise, envelope = _pairwise_envelope(
                    [np.asarray(value) for value in contact_times]
                )
            floor = MATERIAL_FLOORS["first_contact_time_s"]
            fields["first_contact_time_s"] = {
                "pairwise_max_abs": pairwise,
                "envelope": envelope,
                "material_floor": floor,
                "usable": bool(envelope <= floor),
            }
        required = REQUIRED_FIELDS.get(str(first["scenario_id"]))
        if required is None:
            raise EvidenceIntegrityError(
                f"Scenario has no frozen repeatability signal set: {first['scenario_id']}"
            )
        missing = [field for field in required if field not in fields]
        unusable = [
            field
            for field in required
            if field in fields and fields[field]["usable"] is not True
        ]
        usable = not missing and not unusable
        frozen_keys[key] = {
            "engine": first["engine"],
            "scenario_id": first["scenario_id"],
            "repeatability_family": first["repeatability_family"],
            "configuration_hash": first["configuration_hash"],
            "profile_identity_hash": first["profile_identity_hash"],
            "profile_semantics": first["profile_semantics"],
            "pre_forward_initial_condition_hash": first[
                "pre_forward_initial_condition_hash"
            ],
            "post_forward_state_hash": first["post_forward_state_hash"],
            "reset_returned_policy_hash": first["reset_returned_policy_hash"],
            "excitation_hash": first["excitation_hash"],
            "sample_count": len(samples),
            "repetitions": repetitions,
            "record_identity_hashes": [
                record["record_identity_hash"] for record in records
            ],
            "required_fields": list(required),
            "field_envelopes": fields,
            "usable": usable,
            "unusable_reasons": [
                *(f"missing:{field}" for field in missing),
                *(f"nondeterministic:{field}" for field in unusable),
            ],
        }
        index_key = stable_hash(
            {
                "engine": first["engine"],
                "scenario_id": first["scenario_id"],
                "configuration_hash": first["configuration_hash"],
                "profile_identity_hash": first["profile_identity_hash"],
                "excitation_hash": first["excitation_hash"],
            }
        )
        if index_key in index:
            raise EvidenceIntegrityError("Repeatability snapshot index is ambiguous")
        index[index_key] = key

    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "frozen_before_cross_engine_physics": True,
        "minimum_repetitions": MINIMUM_REPETITIONS,
        "effective_tolerance_rule": "max(material_floor,5*repeat_envelope)",
        "high_noise_rule": "mark_key_unusable_when_repeat_envelope_exceeds_material_floor",
        "material_floors": MATERIAL_FLOORS,
        "executions": sorted(
            execution_rows,
            key=lambda row: (str(row["engine"]), str(row["scenario_id"])),
        ),
        "keys": frozen_keys,
        "index": index,
    }
    snapshot["identity_hash"] = stable_hash(snapshot)
    return snapshot


def validate_threshold_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise EvidenceIntegrityError("Threshold snapshot schema mismatch")
    unsigned = dict(snapshot)
    identity_hash = unsigned.pop("identity_hash", None)
    if identity_hash != stable_hash(unsigned):
        raise EvidenceIntegrityError("Threshold snapshot identity mismatch")
    keys = snapshot.get("keys")
    index = snapshot.get("index")
    if not isinstance(keys, Mapping) or not isinstance(index, Mapping):
        raise EvidenceIntegrityError("Threshold snapshot key maps are missing")
    for index_key, repeatability_key in index.items():
        if not isinstance(index_key, str) or repeatability_key not in keys:
            raise EvidenceIntegrityError("Threshold snapshot index is invalid")


def frozen_repeat_envelope(
    snapshot: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    validate_threshold_snapshot(snapshot)
    keys = snapshot["keys"]
    envelopes: list[float] = []
    for record in records:
        validate_repeatability_record(record)
        key = str(record["repeatability_key"])
        frozen = keys.get(key)
        if not isinstance(frozen, Mapping):
            raise EvidenceIntegrityError(
                f"Repeatability key is unavailable in the frozen snapshot: {key}"
            )
        if frozen.get("usable") is not True:
            raise EvidenceIntegrityError(
                f"Repeatability key is nondeterministic/unusable: {key}"
            )
        field_record = frozen.get("field_envelopes", {}).get(field)
        if not isinstance(field_record, Mapping):
            raise EvidenceIntegrityError(
                f"Repeatability field is unavailable for key {key}: {field}"
            )
        if field_record.get("usable") is not True:
            raise EvidenceIntegrityError(
                f"Repeatability field is nondeterministic/unusable: {key}/{field}"
            )
        envelopes.append(float(field_record["envelope"]))
    if not envelopes:
        raise EvidenceIntegrityError("No exact repeatability records were selected")
    return max(envelopes)
